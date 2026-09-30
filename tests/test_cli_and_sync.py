# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI argument handling, error hints, and metadata sync behaviour."""

import sys

import pytest

import bedrock_usage_analyzer.__main__ as cli
from bedrock_usage_analyzer.core.errors import troubleshooting_hint
from bedrock_usage_analyzer.sync import quota_mapper as qm
from bedrock_usage_analyzer.utils.yaml_handler import load_yaml, save_yaml


def test_granularity_single_and_json():
    assert cli._parse_granularity('5min') == {p: 300 for p in ['1hour', '1day', '7days', '14days', '30days']}
    parsed = cli._parse_granularity('{"1hour":"1min","1day":"5min","7days":"1hour","14days":"1hour","30days":"1hour"}')
    assert parsed == {'1hour': 60, '1day': 300, '7days': 3600, '14days': 3600, '30days': 3600}


@pytest.mark.parametrize('bad', ['2min', '{"1hour":"1min"}', '{bad json', '{"1hour":"x","1day":"5min","7days":"1hour","14days":"1hour","30days":"1hour"}'])
def test_granularity_rejects(bad):
    with pytest.raises(ValueError):
        cli._parse_granularity(bad)


def run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, 'argv', ['bua'] + argv)
    captured = {}
    monkeypatch.setattr(cli, 'cmd_analyze', lambda args: captured.setdefault('args', args))
    cli.main()
    return captured['args']


def test_model_id_is_repeatable(monkeypatch):
    args = run_main(monkeypatch, ['analyze', '-r', 'us-west-2', '-m', 'a.b', '-m', 'abc123', '-y'])
    assert args.model_id == ['a.b', 'abc123'] and args.yes and args.region == 'us-west-2'


def test_no_command_prints_help(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['bua'])
    with pytest.raises(SystemExit):
        cli.main()


def test_errors_get_a_partition_hint(monkeypatch, caplog):
    monkeypatch.setattr(sys, 'argv', ['bua', 'analyze', '-r', 'us-gov-west-1'])

    def fail(args):
        raise RuntimeError('UnrecognizedClientException: The security token included in the request is invalid')

    monkeypatch.setattr(cli, 'cmd_analyze', fail)
    with pytest.raises(SystemExit):
        cli.main()
    assert 'Hint:' in caplog.text and 'GovCloud' in caplog.text


@pytest.mark.parametrize('message,region,fragment', [
    ('Unable to locate credentials', None, "run 'aws sts get-caller-identity'."),
    ('ExpiredToken: expired', 'us-west-2', '--region us-west-2.'),
    ('AccessDeniedException: not authorized to perform bedrock:ListInferenceProfiles', None, 'IAM permissions'),
    ('Could not connect to the endpoint URL', 'cn-north-1', 'for cn-north-1'),
])
def test_troubleshooting_hint(message, region, fragment):
    assert fragment in troubleshooting_hint(RuntimeError(message), region)


def test_troubleshooting_hint_none_for_other_errors():
    assert troubleshooting_hint(ValueError('bad value'), 'us-west-2') is None


def test_refresh_fm_list_rejects_bad_region(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['bua', 'refresh', 'fm-list', '../x'])
    with pytest.raises(SystemExit):
        cli.main()


def test_refresh_regions_keeps_other_partitions(monkeypatch, tmp_path):
    from bedrock_usage_analyzer.sync import regions as r
    monkeypatch.setattr(r, 'detect_credentials_partition', lambda _=None: 'aws')
    monkeypatch.setattr(r, 'fetch_enabled_regions', lambda partition, hint: ['eu-west-1', 'us-east-1'])
    monkeypatch.setattr(sys, 'argv', ['bua', 'refresh', 'regions'])
    cli.main()
    saved = load_yaml(str(tmp_path / 'data' / 'regions.yml'))['regions']
    # Commercial list replaced; bundled GovCloud regions kept
    assert saved == ['eu-west-1', 'us-east-1', 'us-gov-east-1', 'us-gov-west-1']


def test_quota_mapper_keeps_unmapped_endpoints(monkeypatch, tmp_path):
    (tmp_path / 'data').mkdir()
    save_yaml(str(tmp_path / 'data' / 'regions.yml'), {'regions': ['us-east-1']})
    save_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'), {'models': [
        {'model_id': 'amazon.nova-lite-v1:0', 'provider': 'Amazon',
         'endpoints': {'base': {'quotas': {}}, 'us': {'quotas': {}}}}]})
    monkeypatch.setattr(qm, 'fetch_service_quotas', lambda region: [{'QuotaName': 'x'}])
    monkeypatch.setattr(qm, 'extract_common_name', lambda *a: 'nova')
    monkeypatch.setattr('bedrock_usage_analyzer.sync.regions.detect_credentials_partition', lambda _=None: 'aws')
    mapper = qm.QuotaMapper('us-east-1', 'model', 'us-east-1')
    monkeypatch.setattr(mapper, '_get_quota_mapping',
                        lambda region, model_id, common, endpoint, quotas:
                        {'tpm': {'code': 'L-1', 'name': 'n'}} if endpoint == 'base' else None)
    mapper.run()
    endpoints = load_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'))['models'][0]['endpoints']
    assert set(endpoints) == {'base', 'us'}
    assert endpoints['base']['quotas']['tpm']['code'] == 'L-1'


def test_quota_mapper_rejects_region_outside_partition(monkeypatch):
    monkeypatch.setattr('bedrock_usage_analyzer.sync.regions.detect_credentials_partition', lambda _=None: 'aws')
    with pytest.raises(SystemExit):
        qm.QuotaMapper('us-east-1', 'model', 'us-gov-west-1')._get_regions_to_process()


def test_get_quota_details_handles_missing_quota(monkeypatch):
    from botocore.exceptions import ClientError
    from bedrock_usage_analyzer.aws import servicequotas

    class Client:
        def get_service_quota(self, **_):
            raise ClientError({'Error': {'Code': 'NoSuchResourceException', 'Message': 'x'}}, 'GetServiceQuota')

    monkeypatch.setattr(servicequotas, 'create_client', lambda *a, **k: Client())
    assert servicequotas.get_quota_details('L-1', 'us-east-1') is None


def test_sts_get_account_id(monkeypatch):
    from bedrock_usage_analyzer.aws import sts
    monkeypatch.setattr(sts, 'get_caller_identity', lambda region=None: {'Account': '42'})
    assert sts.get_account_id('us-gov-west-1') == '42'


def test_fm_list_refresh_keeps_bundled_quota_mappings_and_prefixes(monkeypatch, tmp_path):
    """Refreshing on a clean machine must not wipe bundled quota codes or the us-gov prefix."""
    from bedrock_usage_analyzer.sync import fm_list
    from bedrock_usage_analyzer.utils.paths import get_data_path

    bundled = load_yaml(get_data_path('fm-list-us-east-1.yml'))['models']
    mapped = next(m for m in bundled
                  if any(q for e in m.get('endpoints', {}).values() for q in (e.get('quotas') or {}).values()))
    monkeypatch.setattr(fm_list, 'discover_prefix_mapping', lambda region: [])
    monkeypatch.setattr(fm_list, 'fetch_foundation_models', lambda region: [
        {'model_id': mapped['model_id'], 'provider': mapped['provider'],
         'inference_types': mapped.get('inference_types', [])}])
    monkeypatch.setattr(fm_list, 'fetch_all_inference_profiles', lambda region: [])
    fm_list.refresh_region('us-east-1')

    saved = load_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'))['models'][0]
    assert saved['endpoints'] == mapped['endpoints']
    prefixes = {p['prefix'] for p in load_yaml(str(tmp_path / 'data' / 'prefix-mapping.yml'))['prefixes']}
    assert {'us-gov', 'au', 'jp', 'base', 'global'} <= prefixes
