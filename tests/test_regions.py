# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Region discovery and filtering across partitions."""

import pytest

from bedrock_usage_analyzer.sync import regions as r
from bedrock_usage_analyzer.utils.yaml_handler import save_yaml


class FakePaginator:
    def __init__(self, names):
        self.names = names

    def paginate(self, **_):
        yield {'Regions': [{'RegionName': n} for n in self.names]}


class FakeAccount:
    def __init__(self, names=None, error=None):
        self.names, self.error = names or [], error

    def get_paginator(self, _):
        if self.error:
            raise self.error
        return FakePaginator(self.names)


class FakeEc2:
    def __init__(self, names=None, error=None):
        self.names, self.error = names or [], error

    def describe_regions(self):
        if self.error:
            raise self.error
        return {'Regions': [{'RegionName': n} for n in self.names]}


def install_clients(monkeypatch, account, ec2, calls):
    def fake_create(service, region=None, **_):
        calls.append((service, region))
        return {'account': account, 'ec2': ec2}[service]
    monkeypatch.setattr(r, 'create_client', fake_create)


def test_normalize_accepts_strings_and_dicts():
    entries = ['us-east-1', {'name': 'us-gov-west-1', 'type': 'govcloud'}, 'us-east-1', None, 7]
    assert r.normalize_region_names(entries) == ['us-east-1', 'us-gov-west-1']
    assert r.normalize_region_names(None) == []


def test_bundled_regions_include_both_partitions():
    names = r.load_region_names()
    assert 'us-west-2' in names and 'us-gov-west-1' in names and 'us-gov-east-1' in names
    assert not set(names) & r.SKIP_REGIONS


def test_user_regions_file_overrides_bundle(tmp_path):
    (tmp_path / 'data').mkdir()
    save_yaml(str(tmp_path / 'data' / 'regions.yml'), {'regions': [{'name': 'eu-west-1'}]})
    assert r.load_region_names() == ['eu-west-1']


def test_regions_for_credentials_filters_by_partition(monkeypatch):
    monkeypatch.setattr(r, 'detect_credentials_partition', lambda _=None: 'aws-us-gov')
    assert r.regions_for_credentials(['us-east-1', 'us-gov-west-1']) == (['us-gov-west-1'], 'aws-us-gov')
    monkeypatch.setattr(r, 'detect_credentials_partition', lambda _=None: None)
    assert r.regions_for_credentials(['us-east-1', 'us-gov-west-1']) == (['us-east-1', 'us-gov-west-1'], None)


def test_fetch_uses_account_api_in_commercial(monkeypatch):
    calls = []
    install_clients(monkeypatch, FakeAccount(['us-east-1', 'eu-west-1']), FakeEc2(), calls)
    assert r.fetch_enabled_regions('aws', 'us-west-2') == ['eu-west-1', 'us-east-1']
    assert calls == [('account', 'us-west-2')]


def test_fetch_falls_back_to_ec2_in_govcloud(monkeypatch):
    calls = []
    install_clients(monkeypatch, FakeAccount(error=RuntimeError('UnknownEndpoint')),
                    FakeEc2(['us-gov-west-1', 'us-gov-east-1']), calls)
    assert r.fetch_enabled_regions('aws-us-gov', None) == ['us-gov-east-1', 'us-gov-west-1']
    # Calls are pinned to a GovCloud region, never a commercial endpoint
    assert calls == [('account', 'us-gov-west-1'), ('ec2', 'us-gov-west-1')]


def test_fetch_drops_regions_of_other_partitions(monkeypatch):
    install_clients(monkeypatch, FakeAccount(['us-gov-west-1', 'us-east-1']), FakeEc2(), [])
    assert r.fetch_enabled_regions('aws-us-gov', 'us-gov-west-1') == ['us-gov-west-1']


def test_fetch_uses_static_list_when_apis_fail(monkeypatch):
    install_clients(monkeypatch, FakeAccount(error=RuntimeError('denied')),
                    FakeEc2(error=RuntimeError('denied')), [])
    assert r.fetch_enabled_regions('aws-us-gov', None) == ['us-gov-east-1', 'us-gov-west-1']


def test_merge_keeps_other_partitions():
    existing = ['us-east-1', 'us-west-2', 'us-gov-west-1']
    assert r.merge_regions(existing, ['eu-west-1'], 'aws') == ['eu-west-1', 'us-gov-west-1']
    assert r.merge_regions(existing, ['us-gov-east-1'], 'aws-us-gov') == ['us-east-1', 'us-gov-east-1', 'us-west-2']


def test_refresh_regions_skips_disrupted_regions_and_merges(monkeypatch):
    monkeypatch.setattr(r, 'detect_credentials_partition', lambda _=None: 'aws')
    monkeypatch.setattr(r, 'fetch_enabled_regions',
                        lambda partition, hint: ['me-central-1', 'me-south-1', 'us-east-1'])
    data = r.refresh_regions(existing=['us-gov-west-1', 'ap-south-1'])
    assert data == {'regions': ['us-east-1', 'us-gov-west-1']}


def test_refresh_regions_exits_when_nothing_found(monkeypatch):
    monkeypatch.setattr(r, 'detect_credentials_partition', lambda _=None: 'aws')
    monkeypatch.setattr(r, 'fetch_enabled_regions', lambda partition, hint: ['me-south-1'])
    with pytest.raises(SystemExit):
        r.refresh_regions()


def test_fetch_exits_when_credentials_do_not_work(monkeypatch):
    """No silent fallback to every Bedrock region when the caller identity fails."""
    monkeypatch.setattr(r, 'detect_credentials_partition', lambda _=None: None)
    with pytest.raises(SystemExit):
        r.fetch_enabled_regions(None, None)
    with pytest.raises(SystemExit):
        r.discover_regions()


def test_update_bundle_replaces_only_credentials_partition(monkeypatch, tmp_path):
    """A stale GovCloud list in the user file must not overwrite the bundle's GovCloud regions."""
    import sys
    import bedrock_usage_analyzer.__main__ as cli
    from bedrock_usage_analyzer.utils.yaml_handler import load_yaml
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    save_yaml(str(bundle / 'regions.yml'), {'regions': ['us-east-1', 'us-gov-east-1', 'us-gov-west-1']})
    (tmp_path / 'data').mkdir()
    save_yaml(str(tmp_path / 'data' / 'regions.yml'), {'regions': ['us-east-1', 'us-gov-west-1']})
    monkeypatch.setattr(cli, 'get_bundle_path', lambda: bundle)
    monkeypatch.setattr(r, 'detect_credentials_partition', lambda _=None: 'aws')
    monkeypatch.setattr(r, 'fetch_enabled_regions', lambda partition, hint: ['eu-west-1', 'us-east-1'])
    monkeypatch.setattr(sys, 'argv', ['bua', 'refresh', 'regions', '--update-bundle'])
    cli.main()
    assert load_yaml(str(bundle / 'regions.yml'))['regions'] == ['eu-west-1', 'us-east-1', 'us-gov-east-1', 'us-gov-west-1']
    assert load_yaml(str(tmp_path / 'data' / 'regions.yml'))['regions'] == ['eu-west-1', 'us-east-1', 'us-gov-west-1']
