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
    ('ExpiredToken: expired', 'us-west-2', "run 'aws sts get-caller-identity --region us-west-2'."),
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
    # The user file gets only the refreshed partition; bundled GovCloud regions are added
    # at read time (so later releases can update them)
    assert saved == ['eu-west-1', 'us-east-1']
    from bedrock_usage_analyzer.sync.regions import load_region_names
    assert load_region_names() == ['eu-west-1', 'us-east-1', 'us-gov-east-1', 'us-gov-west-1']


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


def test_sts_get_account_id(monkeypatch):
    from bedrock_usage_analyzer.aws import sts
    monkeypatch.setattr(sts, 'resolve_caller_identity', lambda region=None: {'Account': '42'})
    assert sts.get_account_id('us-gov-west-1') == '42'


def test_fm_list_refresh_keeps_bundled_quota_mappings_and_prefixes(monkeypatch, tmp_path):
    """Refreshing on a clean machine must not wipe bundled quota codes or the us-gov prefix."""
    from bedrock_usage_analyzer.sync import fm_list
    from bedrock_usage_analyzer.utils.paths import get_data_path

    bundled = load_yaml(get_data_path('fm-list-us-east-1.yml'))['models']
    mapped = next(m for m in bundled
                  if any(q for e in m.get('endpoints', {}).values() for q in (e.get('quotas') or {}).values()))
    monkeypatch.setattr(fm_list, 'discover_prefix_mapping', lambda region, profiles=None: [])
    monkeypatch.setattr(fm_list, 'fetch_foundation_models', lambda region: [
        {'model_id': mapped['model_id'], 'provider': mapped['provider'],
         'inference_types': mapped.get('inference_types', [])}])
    monkeypatch.setattr(fm_list, 'fetch_all_inference_profiles', lambda region: [])
    fm_list.refresh_region('us-east-1')

    saved = load_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'))['models'][0]
    assert saved['endpoints'] == mapped['endpoints']
    # Nothing new discovered: no user copy of the bundled prefixes (it would hide later
    # bundled changes), and the merged mapping still has every prefix
    assert not (tmp_path / 'data' / 'prefix-mapping.yml').exists()
    from bedrock_usage_analyzer.aws.bedrock import load_prefix_mapping
    assert {'us-gov', 'au', 'jp', 'base', 'global'} <= {p['prefix'] for p in load_prefix_mapping(refresh=True)}


def test_quota_mapping_picker_exits_without_regions(monkeypatch):
    from bedrock_usage_analyzer.utils import ui
    monkeypatch.setattr('bedrock_usage_analyzer.sync.regions.regions_for_credentials',
                        lambda regions, region=None: ([], 'aws-us-gov'))
    with pytest.raises(SystemExit):
        ui.select_quota_mapping_params()


def _mapper_fixture(monkeypatch, tmp_path, endpoints):
    (tmp_path / 'data').mkdir()
    save_yaml(str(tmp_path / 'data' / 'regions.yml'), {'regions': ['us-east-1']})
    save_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'), {'models': [
        {'model_id': 'openai.gpt-oss-safeguard-20b', 'provider': 'OpenAI', 'endpoints': endpoints}]})
    monkeypatch.setattr(qm, 'extract_common_name', lambda *a: 'gpt-oss')
    monkeypatch.setattr('bedrock_usage_analyzer.sync.regions.detect_credentials_partition', lambda _=None: 'aws')
    return qm.QuotaMapper('us-east-1', 'model', 'us-east-1')


def _saved_endpoints(tmp_path):
    return load_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'))['models'][0]['endpoints']


def test_quota_mapper_keeps_saved_codes_when_no_new_match(monkeypatch, tmp_path):
    """A keyword miss or failed LLM call must not wipe correct codes (seen with GPT OSS Safeguard)."""
    saved = {'base': {'quotas': {'tpm': {'code': 'L-5D8F2F54', 'name': 'TPM'}}}}
    mapper = _mapper_fixture(monkeypatch, tmp_path, saved)
    monkeypatch.setattr(qm, 'fetch_service_quotas', lambda region: [])
    monkeypatch.setattr(mapper, '_get_quota_mapping', lambda *a: None)
    mapper.run()
    assert _saved_endpoints(tmp_path) == saved


def test_quota_mapper_skips_region_when_listing_fails(monkeypatch, tmp_path):
    saved = {'base': {'quotas': {'tpm': {'code': 'L-5D8F2F54', 'name': 'TPM'}}}}
    mapper = _mapper_fixture(monkeypatch, tmp_path, saved)
    monkeypatch.setattr(qm, 'fetch_service_quotas', lambda region: None)
    calls = []
    monkeypatch.setattr(mapper, '_get_quota_mapping', lambda *a: calls.append(a))
    mapper.run()
    assert calls == [] and _saved_endpoints(tmp_path) == saved


def test_keyword_match_treats_hyphens_as_spaces(monkeypatch, tmp_path):
    mapper = _mapper_fixture(monkeypatch, tmp_path, {'base': {'quotas': {}}})
    quotas = [{'QuotaName': 'On-demand model inference tokens per minute for GPT OSS Safeguard 20B', 'QuotaCode': 'L-5D8F2F54'},
              {'QuotaName': 'On-demand model inference tokens per minute for Nova Lite', 'QuotaCode': 'L-X'}]
    assert [q['code'] for q in mapper._find_matching_quotas(quotas, 'gpt-oss', 'base')] == ['L-5D8F2F54']


def test_fetch_service_quotas_returns_none_on_error(monkeypatch):
    from bedrock_usage_analyzer.aws import servicequotas

    class Client:
        def get_paginator(self, _):
            raise RuntimeError('InvalidPaginationTokenException')

    monkeypatch.setattr(servicequotas, 'create_client', lambda *a, **k: Client())
    assert servicequotas.fetch_service_quotas('sa-east-1') is None


def _index_fixture(tmp_path, codes):
    (tmp_path / 'data').mkdir(exist_ok=True)
    for region in ('us-east-1', 'us-west-2'):
        save_yaml(str(tmp_path / 'data' / f'fm-list-{region}.yml'), {'models': [
            {'model_id': 'm1', 'provider': 'X', 'endpoints': {'base': {'quotas': {
                'tpm': {'code': codes['tpm'], 'name': 'old name'},
                'rpm': {'code': codes['rpm'], 'name': 'rpm'}}}}}]})


def test_quota_index_removes_only_codes_reported_missing(monkeypatch, tmp_path, no_bundle, commercial_creds):
    from bedrock_usage_analyzer.sync import quota_index
    from bedrock_usage_analyzer.aws.servicequotas import QUOTA_MISSING, QUOTA_ERROR
    _index_fixture(tmp_path, {'tpm': 'L-GONE', 'rpm': 'L-FLAKY'})
    answers = {'L-GONE': (QUOTA_MISSING, None), 'L-FLAKY': (QUOTA_ERROR, None)}
    monkeypatch.setattr(quota_index, 'check_quota', lambda code, region: answers[code])
    quota_index.QuotaIndexGenerator().run()
    for region in ('us-east-1', 'us-west-2'):
        quotas = load_yaml(str(tmp_path / 'data' / f'fm-list-{region}.yml'))['models'][0]['endpoints']['base']['quotas']
        assert quotas['tpm'] is None                       # missing: removed in every region
        assert quotas['rpm']['code'] == 'L-FLAKY'          # API error: kept


def test_quota_index_refreshes_names(monkeypatch, tmp_path, no_bundle, commercial_creds):
    from bedrock_usage_analyzer.sync import quota_index
    from bedrock_usage_analyzer.aws.servicequotas import QUOTA_OK
    _index_fixture(tmp_path, {'tpm': 'L-1', 'rpm': 'L-2'})
    names = {'L-1': 'On-demand tokens per minute for Model One', 'L-2': 'On-demand requests per minute for Model One'}
    monkeypatch.setattr(quota_index, 'check_quota', lambda code, region: (QUOTA_OK, {'QuotaName': names[code]}))
    gen = quota_index.QuotaIndexGenerator()
    gen.run()
    assert {e['quota_name'] for e in gen.entries} == set(names.values())


def test_check_quota_statuses(monkeypatch):
    from botocore.exceptions import ClientError
    from bedrock_usage_analyzer.aws import servicequotas as sq

    def client_for(error_code):
        class Client:
            def get_service_quota(self, **_):
                if error_code:
                    raise ClientError({'Error': {'Code': error_code, 'Message': 'x'}}, 'GetServiceQuota')
                return {'Quota': {'QuotaName': 'n'}}
        return Client()

    for code, expected in ((None, sq.QUOTA_OK), ('NoSuchResourceException', sq.QUOTA_MISSING),
                           ('ThrottlingException', sq.QUOTA_ERROR)):
        monkeypatch.setattr(sq, 'create_client', lambda *a, _c=code, **k: client_for(_c))
        sq._clients.clear()
        assert sq.check_quota('L-1', 'us-east-1')[0] == expected


def test_refresh_fm_list_all_regions_stops_without_partition(monkeypatch):
    monkeypatch.setattr('bedrock_usage_analyzer.sync.regions.detect_credentials_partition', lambda _=None: None)
    monkeypatch.setattr(sys, 'argv', ['bua', 'refresh', 'fm-list'])
    with pytest.raises(SystemExit):
        cli.main()


def test_load_bundled_yaml_reads_package_resources():
    from bedrock_usage_analyzer.utils.paths import load_bundled_yaml, get_bundled_file
    assert 'us-gov-west-1' in load_bundled_yaml('regions.yml')['regions']
    assert load_bundled_yaml('missing.yml') is None
    assert get_bundled_file('missing.yml') is None


def test_quota_index_keeps_code_where_it_exists(monkeypatch, tmp_path, no_bundle, commercial_creds):
    """Missing in the source region but present in another: removed only where missing."""
    from bedrock_usage_analyzer.sync import quota_index
    from bedrock_usage_analyzer.aws.servicequotas import QUOTA_OK, QUOTA_MISSING
    _index_fixture(tmp_path, {'tpm': 'L-PEGASUS', 'rpm': 'L-OK'})
    def check(code, region):
        if code == 'L-PEGASUS' and region == 'us-east-1':
            return QUOTA_MISSING, None
        return QUOTA_OK, {'QuotaName': code}
    monkeypatch.setattr(quota_index, 'check_quota', check)
    gen = quota_index.QuotaIndexGenerator()
    gen.run()
    tpm = {r: load_yaml(str(tmp_path / 'data' / f'fm-list-{r}.yml'))['models'][0]['endpoints']['base']['quotas']['tpm']
           for r in ('us-east-1', 'us-west-2')}
    assert tpm['us-east-1'] is None and tpm['us-west-2']['code'] == 'L-PEGASUS'
    assert gen.error_entries == []
    assert 'L-PEGASUS' in (tmp_path / 'data' / 'quota-index.csv').read_text()


def test_list_data_files_merges_user_and_bundled(tmp_path):
    from bedrock_usage_analyzer.utils.paths import list_data_files
    (tmp_path / 'data').mkdir()
    save_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'), {'models': []})
    files = list_data_files('fm-list-*.yml')
    names = [f.name for f in files]
    assert len(names) == len(set(names)) and len(names) > 20      # bundled regions still listed
    assert next(f for f in files if f.name == 'fm-list-us-east-1.yml').parent == tmp_path / 'data'


def test_probe_clients_fail_fast():
    from bedrock_usage_analyzer.aws.client_factory import create_client
    c = create_client('sts', 'cn-north-1', probe=True)
    assert c.meta.config.retries['total_max_attempts'] == 1 and c.meta.config.connect_timeout == 3


def test_error_classifiers():
    from bedrock_usage_analyzer.core.errors import is_access_denied
    from bedrock_usage_analyzer.utils.partition import is_token_rejection
    assert is_token_rejection(RuntimeError('An error occurred (InvalidClientTokenId) when calling'))
    assert not is_token_rejection(RuntimeError('Could not connect to the endpoint URL'))
    assert is_access_denied(RuntimeError('AccessDeniedException: User is not authorized'))
    assert not is_access_denied(RuntimeError('ThrottlingException'))


def test_fm_list_refresh_writes_only_new_prefixes_to_user_file(monkeypatch, tmp_path):
    from bedrock_usage_analyzer.sync import fm_list
    new = {'prefix': 'mx', 'quota_keyword': 'cross-region', 'description': 'cross-region inference profile',
           'is_regional': True, 'source': 'discovered'}
    monkeypatch.setattr(fm_list, 'discover_prefix_mapping', lambda region, profiles=None: [new])
    monkeypatch.setattr(fm_list, 'fetch_foundation_models', lambda region: None)
    monkeypatch.setattr(fm_list, 'fetch_all_inference_profiles', lambda region: [])
    fm_list.refresh_region('mx-central-1')
    saved = load_yaml(str(tmp_path / 'data' / 'prefix-mapping.yml'))['prefixes']
    assert [p['prefix'] for p in saved] == ['mx']


def test_hint_classification_by_botocore_type_and_code():
    from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError
    assert 'network' in troubleshooting_hint(EndpointConnectionError(endpoint_url='https://sts.x'), 'us-east-1')
    assert 'credentials' in troubleshooting_hint(NoCredentialsError(), None)
    expired = ClientError({'Error': {'Code': 'ExpiredTokenException', 'Message': 'x'}}, 'GetCallerIdentity')
    assert 'Check your AWS credentials' in troubleshooting_hint(expired, 'us-east-1')
    denied = ClientError({'Error': {'Code': 'AccessDeniedException', 'Message': 'x'}}, 'ListInferenceProfiles')
    assert 'permission' in troubleshooting_hint(denied, 'us-east-1')
    # A proxy failure whose text mentions credentials is a network problem
    assert 'network' in troubleshooting_hint(
        RuntimeError('Could not connect to the endpoint URL while refreshing credentials'), None)
    assert troubleshooting_hint(RuntimeError('refreshing credentials failed: something odd'), None) is None


def test_gov_quota_mapping_options_follow_region_metadata():
    from bedrock_usage_analyzer.utils.ui import _claude_endpoints_in
    east = _claude_endpoints_in('us-gov-east-1')
    assert east and not any(o.startswith('anthropic.') for o in east)   # no on-demand there
    assert 'anthropic.claude-3-5-sonnet-20240620-v1:0' in _claude_endpoints_in('us-gov-west-1')
    assert not any(o.count(':') > 1 for o in east)


def test_regions_module_main_saves_like_the_cli(monkeypatch, tmp_path):
    from bedrock_usage_analyzer.sync import regions as r
    monkeypatch.setattr(r, 'detect_credentials_partition', lambda _=None: 'aws')
    monkeypatch.setattr(r, 'fetch_enabled_regions', lambda partition, hint: ['eu-west-1'])
    monkeypatch.setattr(sys, 'argv', ['regions.py'])
    r.main()
    assert load_yaml(str(tmp_path / 'data' / 'regions.yml'))['regions'] == ['eu-west-1']
    assert 'us-gov-west-1' in r.load_region_names()      # bundled GovCloud still offered


def test_quota_mapping_picker_uses_target_region_for_sts(monkeypatch):
    from bedrock_usage_analyzer.utils import ui
    seen = []
    monkeypatch.setattr('bedrock_usage_analyzer.sync.regions.regions_for_credentials',
                        lambda regions, region=None: seen.append(region) or (['us-gov-west-1'], 'aws-us-gov'))
    ui.select_quota_mapping_params(target_region='us-gov-west-1', bedrock_region='us-gov-west-1',
                                   model_id='us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0')
    assert seen == ['us-gov-west-1']


def test_gov_quota_mapping_options_newest_first():
    from bedrock_usage_analyzer.utils.ui import _claude_endpoints_in
    west = _claude_endpoints_in('us-gov-west-1')
    assert west[0] == 'anthropic.claude-3-haiku-20240307-v1:0'          # Haiku first (cheap bulk calls)
    assert west.index('us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0') < \
        west.index('anthropic.claude-3-7-sonnet-20250219-v1:0') < west.index('anthropic.claude-3-5-sonnet-20240620-v1:0')
    commercial = _claude_endpoints_in('us-east-1')
    assert commercial and 'haiku' in commercial[0] and len(commercial) <= 12


def test_quota_mapping_rejects_bedrock_region_of_other_partition(monkeypatch):
    from bedrock_usage_analyzer.utils import ui
    monkeypatch.setattr('bedrock_usage_analyzer.sync.regions.regions_for_credentials',
                        lambda regions, region=None: (['us-gov-east-1', 'us-gov-west-1'], 'aws-us-gov'))
    with pytest.raises(SystemExit):
        ui.select_quota_mapping_params(target_region='us-gov-west-1', bedrock_region='us-east-1', model_id='m')


def test_fm_quotas_with_all_args_still_checks_bedrock_partition(monkeypatch):
    import argparse
    from bedrock_usage_analyzer import __main__ as cli
    monkeypatch.setattr('bedrock_usage_analyzer.sync.regions.regions_for_credentials',
                        lambda regions, region=None: (['us-gov-east-1', 'us-gov-west-1'], 'aws-us-gov'))
    monkeypatch.setattr('bedrock_usage_analyzer.sync.quota_mapper.QuotaMapper',
                        lambda *a, **k: pytest.fail('mapper must not start'))
    args = argparse.Namespace(target_region='us-gov-west-1', bedrock_region='us-east-1',
                              model_id='m', update_bundle=False)
    with pytest.raises(SystemExit):
        cli.cmd_refresh_fm_quotas(args)


def test_fm_list_refresh_adds_base_when_a_listed_model_gains_on_demand(monkeypatch, tmp_path):
    from bedrock_usage_analyzer.sync import fm_list
    (tmp_path / 'data').mkdir(exist_ok=True)
    save_yaml(str(tmp_path / 'data' / 'fm-list-ap-south-1.yml'), {'models': [
        {'model_id': 'x.prov-only-v1:0', 'provider': 'X', 'inference_types': ['PROVISIONED'],
         'endpoints': {'apac': {'quotas': {'tpm': {'code': 'L-A', 'name': 'n'}}}}}]})
    monkeypatch.setattr(fm_list, 'discover_prefix_mapping', lambda region, profiles=None: [])
    monkeypatch.setattr(fm_list, 'fetch_all_inference_profiles', lambda region: [])
    monkeypatch.setattr(fm_list, 'fetch_foundation_models', lambda region: [
        {'model_id': 'x.prov-only-v1:0', 'provider': 'X', 'inference_types': ['ON_DEMAND', 'PROVISIONED']}])
    fm_list.refresh_region('ap-south-1')
    saved = load_yaml(str(tmp_path / 'data' / 'fm-list-ap-south-1.yml'))['models'][0]
    assert 'base' in saved['endpoints']
    assert saved['endpoints']['apac']['quotas']['tpm']['code'] == 'L-A'        # kept
