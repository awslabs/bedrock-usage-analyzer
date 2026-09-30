# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Partition helpers: offline resolution, ARNs, console links, caller identity."""

import pytest

from bedrock_usage_analyzer.aws.client_factory import create_client
from bedrock_usage_analyzer.utils import partition as p


@pytest.mark.parametrize('region,expected', [
    ('us-west-2', 'aws'),
    ('ap-southeast-7', 'aws'),
    ('us-gov-west-1', 'aws-us-gov'),
    ('us-gov-east-1', 'aws-us-gov'),
    ('cn-north-1', 'aws-cn'),
    ('cn-northwest-1', 'aws-cn'),
    ('us-iso-east-1', 'aws-iso'),
    ('us-isob-east-1', 'aws-iso-b'),
    ('us-gov-future-9', 'aws-us-gov'),   # unknown to botocore: prefix fallback
    ('cn-future-9', 'aws-cn'),
    ('xx-future-9', 'aws'),
    (None, 'aws'),
    ('', 'aws'),
])
def test_partition_for_region(region, expected):
    assert p.get_partition_for_region(region) == expected


def test_region_predicates():
    assert p.is_govcloud_region('us-gov-west-1')
    assert not p.is_govcloud_region('us-west-2')
    assert not p.is_govcloud_region('cn-north-1')


def test_display_names_come_from_botocore():
    assert p.get_region_display_name('us-gov-west-1') == 'AWS GovCloud (US-West)'
    assert p.get_region_display_name('ap-southeast-2') == 'Asia Pacific (Sydney)'
    assert p.get_region_display_name('xx-nowhere-1') == 'xx-nowhere-1'
    assert p.get_region_display_name(None) == 'Unknown Region'


def test_region_info():
    info = p.get_region_info('us-gov-east-1')
    assert info == {
        'name': 'us-gov-east-1',
        'display_name': 'AWS GovCloud (US-East)',
        'partition': 'aws-us-gov',
        'partition_name': 'AWS GovCloud (US)',
        'is_govcloud': True,
    }
    assert p.get_region_info('us-east-1')['is_govcloud'] is False


def test_build_arn_uses_region_partition_without_api_calls():
    assert p.build_arn('bedrock', 'us-west-2', '', 'foundation-model/m') == \
        'arn:aws:bedrock:us-west-2::foundation-model/m'
    assert p.build_arn('bedrock', 'us-gov-west-1', '', 'foundation-model/m') == \
        'arn:aws-us-gov:bedrock:us-gov-west-1::foundation-model/m'
    assert p.build_arn('bedrock', 'cn-north-1', '1', 'x') == 'arn:aws-cn:bedrock:cn-north-1:1:x'


def test_console_urls_per_partition():
    assert p.get_service_quota_url('us-west-2', 'bedrock', 'L-1') == \
        'https://console.aws.amazon.com/servicequotas/home/services/bedrock/quotas/L-1?region=us-west-2'
    assert p.get_service_quota_url('us-gov-west-1', 'bedrock', 'L-1') == \
        'https://console.amazonaws-us-gov.com/servicequotas/home/services/bedrock/quotas/L-1?region=us-gov-west-1'
    assert p.get_service_quota_url('cn-north-1', 'bedrock', 'L-1').startswith('https://console.amazonaws.cn/')
    # No public console for ISO partitions: no link rather than a wrong one
    assert p.get_service_quota_url('us-iso-east-1', 'bedrock', 'L-1') is None
    assert p.get_service_quotas_console_url('us-gov-east-1') == \
        'https://console.amazonaws-us-gov.com/servicequotas/home?region=us-gov-east-1'
    assert p.get_service_quotas_console_url() == 'https://console.aws.amazon.com/servicequotas/home'


def test_filter_regions_by_partition():
    regions = ['us-east-1', 'us-gov-west-1', 'cn-north-1', 'us-gov-east-1']
    assert p.filter_regions_by_partition(regions, 'aws-us-gov') == ['us-gov-west-1', 'us-gov-east-1']
    assert p.filter_regions_by_partition(regions, 'aws') == ['us-east-1']
    assert p.filter_regions_by_partition(regions, None) == regions


def test_partition_regions_static_list():
    assert p.partition_regions('aws-us-gov') == ['us-gov-east-1', 'us-gov-west-1']
    assert p.partition_regions('nope') == []


class FakeSts:
    def __init__(self, arn_value, account='111122223333'):
        self.arn = arn_value
        self.account = account
        self.calls = 0

    def get_caller_identity(self):
        self.calls += 1
        return {'Account': self.account, 'Arn': self.arn, 'UserId': 'X'}


def test_caller_identity_is_cached_per_region(monkeypatch):
    sts = FakeSts('arn:aws-us-gov:iam::111122223333:user/alice')
    created = []

    def fake_create(service, region=None, **_):
        created.append((service, region))
        return sts

    monkeypatch.setattr('bedrock_usage_analyzer.aws.client_factory.create_client', fake_create)
    first = p.get_caller_identity('us-gov-west-1')
    second = p.get_caller_identity('us-gov-west-1')
    assert first == second == {'Account': '111122223333',
                               'Arn': 'arn:aws-us-gov:iam::111122223333:user/alice',
                               'Partition': 'aws-us-gov'}
    assert sts.calls == 1
    assert created == [('sts', 'us-gov-west-1')]
    assert p.detect_credentials_partition('us-gov-west-1') == 'aws-us-gov'


def test_detect_partition_returns_none_on_error(monkeypatch):
    def boom(*_, **__):
        raise RuntimeError('no credentials')

    monkeypatch.setattr('bedrock_usage_analyzer.aws.client_factory.create_client', boom)
    assert p.detect_credentials_partition('us-east-1') is None


def test_region_hint_prefers_env(monkeypatch):
    monkeypatch.setenv('AWS_REGION', 'us-gov-east-1')
    assert p.region_hint() == 'us-gov-east-1'


@pytest.mark.parametrize('service,region,url', [
    ('bedrock', 'us-gov-west-1', 'https://bedrock.us-gov-west-1.amazonaws.com'),
    ('cloudwatch', 'us-gov-west-1', 'https://monitoring.us-gov-west-1.amazonaws.com'),
    ('service-quotas', 'us-gov-east-1', 'https://servicequotas.us-gov-east-1.amazonaws.com'),
    ('sts', 'us-gov-west-1', 'https://sts.us-gov-west-1.amazonaws.com'),
    ('bedrock', 'cn-north-1', 'https://bedrock.cn-north-1.amazonaws.com.cn'),
    ('bedrock-runtime', 'us-west-2', 'https://bedrock-runtime.us-west-2.amazonaws.com'),
])
def test_client_factory_resolves_partition_endpoints(service, region, url):
    client = create_client(service, region)
    assert client.meta.endpoint_url == url
    assert client.meta.region_name == region
    expected_mode = 'adaptive' if service in ('cloudwatch', 'service-quotas', 'bedrock-runtime') else 'standard'
    assert client.meta.config.retries['mode'] == expected_mode
    assert client.meta.config.max_pool_connections >= 16


def test_region_hint_caches_config_lookup(monkeypatch):
    monkeypatch.delenv('AWS_REGION', raising=False)
    monkeypatch.delenv('AWS_DEFAULT_REGION', raising=False)
    created = []

    class FakeSession:
        region_name = 'us-gov-east-1'

        def __init__(self):
            created.append(1)

    monkeypatch.setattr(p.boto3.session, 'Session', FakeSession)
    assert p.region_hint() == 'us-gov-east-1'
    assert p.region_hint() == 'us-gov-east-1'
    assert len(created) == 1


@pytest.mark.parametrize('name,valid', [('us-gov-west-1', True), ('eusc-de-east-1', True), ('../x', False),
                                        (None, False), ('', False)])
def test_is_valid_region_name(name, valid):
    assert p.is_valid_region_name(name) is valid


def test_resolve_caller_identity_retries_home_region_only_on_token_rejection():
    calls = []

    def lookup(region):
        calls.append(region)
        if region == 'us-gov-east-1':
            raise RuntimeError('UnrecognizedClientException')
        if region == 'ap-east-2':
            raise RuntimeError('Could not connect to the endpoint URL')
        return {'Partition': p.get_partition_for_region(region)}

    assert p.resolve_caller_identity('us-gov-east-1', lookup)['Partition'] == 'aws-us-gov'
    assert calls == ['us-gov-east-1', 'us-gov-west-1']
    with pytest.raises(RuntimeError, match='Could not connect'):
        p.resolve_caller_identity('ap-east-2', lookup)
    assert calls[-1] == 'ap-east-2'


def test_identity_fallback_is_remembered():
    calls = []

    def lookup(region):
        calls.append(region)
        if region == 'ap-east-1':
            raise RuntimeError('InvalidClientTokenId')
        return {'Partition': 'aws'}

    p.resolve_caller_identity('ap-east-1', lookup)
    p.resolve_caller_identity('ap-east-1', lookup)
    assert calls == ['ap-east-1', 'us-east-1', 'us-east-1']
