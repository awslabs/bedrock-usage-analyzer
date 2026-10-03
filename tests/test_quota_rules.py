# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic vetoes for wrong quota mappings, and their use in mapping and validation."""

import pytest

from bedrock_usage_analyzer.sync import quota_index, quota_mapper as qm
from bedrock_usage_analyzer.sync.quota_rules import mapping_conflict, model_version, quota_versions
from bedrock_usage_analyzer.utils.yaml_handler import load_yaml, save_yaml

REGIONAL = {'us', 'eu', 'apac', 'jp', 'au', 'ca', 'in', 'us-gov'}


@pytest.mark.parametrize('model_id,version', [
    ('anthropic.claude-sonnet-4-20250514-v1:0', '4'),
    ('anthropic.claude-sonnet-4-6', '4.6'),
    ('anthropic.claude-haiku-4-5-20251001-v1:0', '4.5'),
    ('anthropic.claude-3-5-sonnet-20241022-v2:0', '3.5'),
    ('anthropic.claude-opus-4-1-20250805-v1:0', '4.1'),
    ('meta.llama3-2-1b-instruct-v1:0', '3.2'),
    ('meta.llama4-scout-17b-instruct-v1:0', '4'),
    ('openai.gpt-6.1-sol', '6.1'),
    ('cohere.rerank-v3-5:0', '3.5'),
    ('amazon.nova-2-lite-v1:0', '2'),
    ('zai.glm-4.7-flash', '4.7'),
    ('amazon.nova-lite-v1:0', None),
    ('amazon.titan-embed-text-v2:0', None),
    ('cohere.embed-v4:0', None),
    ('mistral.mistral-large-2407-v1:0', None),
    ('deepseek.v3.2', None),
    ('openai.gpt-oss-20b-1:0', None),
])
def test_model_version(model_id, version):
    assert model_version(model_id) == version


def test_quota_versions():
    assert quota_versions('Cross-region model inference tokens per minute for Anthropic Claude Sonnet 4.6') == {'4.6'}
    assert quota_versions('Global cross-region model inference tokens per minute for GPT-6 Sol') == {'6'}
    assert quota_versions('On-demand model inference tokens per minute for Meta Llama 3.2 1B Instruct') == {'3.2'}
    assert quota_versions('On-demand model inference requests per minute for Ministral 14B 3.0') == {'3'}
    assert quota_versions('On-demand ... for Amazon Titan Text Embeddings V2') == set()
    assert quota_versions('... for Moonshot AI Kimi K2.5') == set()


@pytest.mark.parametrize('model_id,endpoint,name,conflict', [
    # Found in the bundled metadata by the iteration 3 review
    ('anthropic.claude-sonnet-4-20250514-v1:0', 'us',
     'Global cross-region model inference tokens per minute for Anthropic Claude Sonnet 4', True),
    ('anthropic.claude-sonnet-4-20250514-v1:0', 'global',
     'Global cross-region model inference tokens per minute for Anthropic Claude Sonnet 4.6', True),
    ('anthropic.claude-sonnet-5', 'base', 'On-demand model inference tokens per minute for Anthropic Claude 3.5 Sonnet', True),
    ('openai.gpt-6.1-sol', 'global', 'Global cross-region model inference tokens per day for GPT-6 Sol', True),
    ('amazon.nova-lite-v1:0', 'base', 'Cross-region model inference tokens per minute for Amazon Nova Lite', True),
    ('amazon.nova-lite-v1:0', 'global', 'Cross-region model inference tokens per minute for Amazon Nova Lite', True),
    # Correct mappings pass
    ('anthropic.claude-sonnet-4-20250514-v1:0', 'us',
     'Cross-region model inference tokens per minute for Anthropic Claude Sonnet 4 V1', False),
    ('anthropic.claude-sonnet-4-6', 'global',
     'Global cross-region model inference tokens per minute for Anthropic Claude Sonnet 4.6', False),
    ('mistral.ministral-3-14b-instruct', 'base', 'On-demand model inference requests per minute for Ministral 14B 3.0', False),
    ('cohere.rerank-v3-5:0', 'base', 'On-demand model inference requests per minute for Cohere Rerank 3.5', False),
    ('openai.gpt-oss-safeguard-20b', 'base', 'On-demand model inference tokens per minute for GPT OSS Safeguard 20B', False),
    ('amazon.nova-lite-v1:0', 'us-gov', 'Cross-region model inference tokens per minute for Amazon Nova Lite', False),
    ('x.y', 'us', None, False),
])
def test_mapping_conflict(model_id, endpoint, name, conflict):
    assert bool(mapping_conflict(model_id, endpoint, name, REGIONAL)) is conflict


SONNET4 = 'anthropic.claude-sonnet-4-20250514-v1:0'
QUOTAS = [
    {'QuotaName': 'Cross-region model inference tokens per minute for Anthropic Claude Sonnet 4 V1', 'QuotaCode': 'L-US'},
    {'QuotaName': 'Global cross-region model inference tokens per minute for Anthropic Claude Sonnet 4 V1', 'QuotaCode': 'L-GL4'},
    {'QuotaName': 'Global cross-region model inference tokens per minute for Anthropic Claude Sonnet 4.6', 'QuotaCode': 'L-GL46'},
]


def test_prefilter_drops_global_and_other_versions():
    mapper = qm.QuotaMapper('us-east-1', 'm')
    assert [c['code'] for c in mapper._find_matching_quotas(QUOTAS, 'claude', 'us', SONNET4)] == ['L-US']
    assert [c['code'] for c in mapper._find_matching_quotas(QUOTAS, 'claude', 'global', SONNET4)] == ['L-GL4']


def test_llm_picks_outside_candidates_or_conflicting_are_rejected(monkeypatch):
    mapper = qm.QuotaMapper('us-east-1', 'm')
    monkeypatch.setattr(qm, 'extract_quota_codes', lambda *a: {
        'tpm': {'code': 'L-US', 'name': 'x'}, 'rpm': {'code': 'L-INVENTED', 'name': 'Unknown'},
        'tpd': None, 'concurrent': None})
    mapping = mapper._get_quota_mapping('us-east-1', SONNET4, 'claude', 'us', QUOTAS)
    assert mapping['tpm']['code'] == 'L-US' and mapping['rpm'] is None


def test_cached_codes_are_reused_only_where_they_exist(monkeypatch):
    """The 'in' profile codes cached in us-east-1 did not exist in ap-south-1."""
    mapper = qm.QuotaMapper('us-east-1', 'm')
    calls = []

    def llm(region, model, fm_model_id, endpoint, candidates):
        calls.append([c['code'] for c in candidates])
        return {'tpm': {'code': candidates[0]['code'], 'name': ''}, 'rpm': None, 'tpd': None, 'concurrent': None}

    monkeypatch.setattr(qm, 'extract_quota_codes', llm)
    first = mapper._get_quota_mapping('us-east-1', SONNET4, 'claude', 'us', QUOTAS)
    again = mapper._get_quota_mapping('us-west-2', SONNET4, 'claude', 'us', QUOTAS)
    other = [{'QuotaName': QUOTAS[0]['QuotaName'], 'QuotaCode': 'L-US-OTHER'}]
    elsewhere = mapper._get_quota_mapping('ap-south-1', SONNET4, 'claude', 'in', other)
    assert first == again and len(calls) == 2
    assert elsewhere['tpm']['code'] == 'L-US-OTHER'


def test_quota_index_removes_saved_mismatches_in_every_region(monkeypatch, tmp_path, no_bundle, commercial_creds):
    (tmp_path / 'data').mkdir()
    wrong = {'code': 'L-GL46', 'name': 'Global cross-region model inference tokens per minute for Anthropic Claude Sonnet 4.6'}
    right = {'code': 'L-US', 'name': 'Cross-region model inference tokens per minute for Anthropic Claude Sonnet 4 V1'}
    for region, us_tpm in (('us-east-1', right), ('us-west-2', wrong)):
        save_yaml(str(tmp_path / 'data' / f'fm-list-{region}.yml'), {'models': [
            {'model_id': SONNET4, 'provider': 'Anthropic', 'endpoints': {
                'us': {'quotas': {'tpm': us_tpm}}, 'global': {'quotas': {'tpm': wrong}}}}]})
    names = {'L-GL46': wrong['name'], 'L-US': right['name']}
    monkeypatch.setattr(quota_index, 'check_quota', lambda code, region: ('ok', {'QuotaName': names[code]}))
    gen = quota_index.QuotaIndexGenerator()
    gen.run()
    east = load_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'))['models'][0]['endpoints']
    west = load_yaml(str(tmp_path / 'data' / 'fm-list-us-west-2.yml'))['models'][0]['endpoints']
    assert east['us']['quotas']['tpm'] == right
    assert east['global']['quotas']['tpm'] is None
    assert west['us']['quotas']['tpm'] is None        # never sampled by the index, still cleaned
    csv_text = (tmp_path / 'data' / 'quota-index.csv').read_text()
    assert 'L-US' in csv_text and 'L-GL46' not in csv_text


def test_partial_new_mapping_keeps_other_saved_metrics(monkeypatch, tmp_path):
    (tmp_path / 'data').mkdir()
    save_yaml(str(tmp_path / 'data' / 'regions.yml'), {'regions': ['us-east-1']})
    saved = {'tpm': {'code': 'L-A', 'name': 'a'}, 'rpm': {'code': 'L-B', 'name': 'b'}, 'tpd': None}
    save_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'), {'models': [
        {'model_id': 'amazon.nova-lite-v1:0', 'provider': 'Amazon', 'endpoints': {'base': {'quotas': saved}}}]})
    monkeypatch.setattr(qm, 'fetch_service_quotas', lambda region: [])
    monkeypatch.setattr(qm, 'extract_common_name', lambda *a: 'nova')
    monkeypatch.setattr('bedrock_usage_analyzer.sync.regions.detect_credentials_partition', lambda _=None: 'aws')
    mapper = qm.QuotaMapper('us-east-1', 'model', 'us-east-1')
    monkeypatch.setattr(mapper, '_get_quota_mapping', lambda *a: {
        'tpm': {'code': 'L-NEW', 'name': 'n'}, 'rpm': None, 'tpd': {'code': 'L-TPD', 'name': 't'}, 'concurrent': None})
    mapper.run()
    quotas = load_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'))['models'][0]['endpoints']['base']['quotas']
    assert quotas == {'tpm': {'code': 'L-NEW', 'name': 'n'}, 'rpm': {'code': 'L-B', 'name': 'b'},
                      'tpd': {'code': 'L-TPD', 'name': 't'}, 'concurrent': None}


def test_unverified_conflicting_entry_is_left_out_of_csv(monkeypatch, tmp_path, no_bundle, commercial_creds):
    (tmp_path / 'data').mkdir()
    wrong = {'code': 'L-GL46', 'name': 'Global cross-region model inference tokens per minute for Anthropic Claude Sonnet 4.6'}
    save_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'), {'models': [
        {'model_id': SONNET4, 'provider': 'Anthropic', 'endpoints': {'us': {'quotas': {'tpm': wrong}}}}]})
    monkeypatch.setattr(quota_index, 'check_quota', lambda code, region: ('error', None))   # e.g. throttled
    quota_index.QuotaIndexGenerator().run()
    assert 'L-GL46' not in (tmp_path / 'data' / 'quota-index.csv').read_text()
    assert load_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'))['models'][0]['endpoints']['us']['quotas']['tpm'] is None


def test_prefix_file_can_mark_a_known_prefix_not_regional(tmp_path):
    from bedrock_usage_analyzer.aws import bedrock
    (tmp_path / 'data').mkdir()
    save_yaml(str(tmp_path / 'data' / 'prefix-mapping.yml'), {'prefixes': [
        {'prefix': 'ca', 'quota_keyword': 'cross-region', 'description': 'x', 'is_regional': False}]})
    bedrock._prefix_mapping_cache = None
    prefixes = bedrock.get_regional_profile_prefixes()
    assert 'ca' not in prefixes and {'us', 'us-gov', 'in'} <= set(prefixes)


def test_saved_conflicting_codes_are_dropped_on_refresh(monkeypatch, tmp_path):
    """A wrong code written by an older version goes even if nothing replaces it."""
    (tmp_path / 'data').mkdir()
    save_yaml(str(tmp_path / 'data' / 'regions.yml'), {'regions': ['us-east-1']})
    wrong = {'code': 'L-GL46', 'name': 'Global cross-region model inference tokens per minute for Anthropic Claude Sonnet 4.6'}
    save_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'), {'models': [
        {'model_id': SONNET4, 'provider': 'Anthropic', 'endpoints': {'us': {'quotas': {'tpd': wrong}}}}]})
    monkeypatch.setattr(qm, 'fetch_service_quotas', lambda region: [])
    monkeypatch.setattr(qm, 'extract_common_name', lambda *a: 'claude')
    monkeypatch.setattr('bedrock_usage_analyzer.sync.regions.detect_credentials_partition', lambda _=None: 'aws')
    mapper = qm.QuotaMapper('us-east-1', 'model', 'us-east-1')
    monkeypatch.setattr(mapper, '_get_quota_mapping', lambda *a: None)
    mapper.run()
    quotas = load_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'))['models'][0]['endpoints']['us']['quotas']
    assert quotas['tpd'] is None


def test_quota_index_ignores_other_partitions(monkeypatch, tmp_path, no_bundle, commercial_creds):
    (tmp_path / 'data').mkdir()
    for region in ('us-east-1', 'us-gov-west-1'):
        save_yaml(str(tmp_path / 'data' / f'fm-list-{region}.yml'), {'models': [
            {'model_id': 'amazon.nova-lite-v1:0', 'provider': 'Amazon', 'endpoints': {'base': {'quotas': {
                'tpm': {'code': f'L-{region}', 'name': 'On-demand tokens per minute for Amazon Nova Lite'}}}}}]})
    checked = []
    monkeypatch.setattr(quota_index, 'check_quota', lambda code, region: checked.append(region) or ('ok', {'QuotaName': 'n'}))
    gen = quota_index.QuotaIndexGenerator()
    gen.run()
    assert set(checked) == {'us-east-1'} and set(gen._fm_data) == {'us-east-1'}


def test_quota_index_cleanup_never_creates_user_copies(monkeypatch, tmp_path, commercial_creds):
    """Bundled lists are left alone (a user copy would hide future bundled updates)."""
    before = set((tmp_path / 'data').glob('*')) if (tmp_path / 'data').exists() else set()
    monkeypatch.setattr(quota_index, 'check_quota', lambda code, region: ('missing', None))
    quota_index.QuotaIndexGenerator().run()
    created = {p.name for p in (tmp_path / 'data').glob('fm-list-*.yml')} - {p.name for p in before}
    assert created == set()


def test_analyzer_skips_saved_conflicting_codes(tmp_path, monkeypatch):
    from bedrock_usage_analyzer.core.analyzer import BedrockAnalyzer
    (tmp_path / 'data').mkdir()
    save_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'), {'models': [
        {'model_id': SONNET4, 'provider': 'Anthropic', 'endpoints': {'us': {'quotas': {
            'tpm': {'code': 'L-GL46', 'name': 'Global cross-region model inference tokens per minute for Anthropic Claude Sonnet 4.6'},
            'rpm': {'code': 'L-US', 'name': 'Cross-region model inference requests per minute for Anthropic Claude Sonnet 4 V1'}}}}}]})
    analyzer = BedrockAnalyzer.__new__(BedrockAnalyzer)
    analyzer.region, analyzer._fm_models = 'us-east-1', None
    codes = analyzer._load_quota_codes(SONNET4, 'us')
    assert codes['tpm'] is None and codes['rpm']['code'] == 'L-US'
    assert analyzer._load_quota_codes(SONNET4, 'unknown') == {}
    assert analyzer._load_quota_codes('missing.model', 'us') == {}


def test_mapper_drops_conflicts_even_without_common_name(monkeypatch, tmp_path):
    (tmp_path / 'data').mkdir()
    save_yaml(str(tmp_path / 'data' / 'regions.yml'), {'regions': ['us-east-1']})
    wrong = {'code': 'L-GL46', 'name': 'Global cross-region model inference tokens per minute for Anthropic Claude Sonnet 4.6'}
    save_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'), {'models': [
        {'model_id': SONNET4, 'provider': 'Anthropic', 'endpoints': {'us': {'quotas': {'tpm': wrong}}}}]})
    monkeypatch.setattr(qm, 'fetch_service_quotas', lambda region: [])
    monkeypatch.setattr(qm, 'extract_common_name', lambda *a: None)      # LLM failed
    monkeypatch.setattr('bedrock_usage_analyzer.sync.regions.detect_credentials_partition', lambda _=None: 'aws')
    qm.QuotaMapper('us-east-1', 'model', 'us-east-1').run()
    assert load_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'))['models'][0]['endpoints']['us']['quotas']['tpm'] is None


def test_long_context_variant_quota_is_rejected():
    name = 'Model invocation max tokens per day for Anthropic Claude Sonnet 4.5 V1 1M Context Length (doubled for cross-region calls)'
    assert mapping_conflict('anthropic.claude-sonnet-4-5-20250929-v1:0', 'us', name, REGIONAL) == 'long-context variant quota'
    std = 'Model invocation max tokens per day for Anthropic Claude Sonnet 4.5 V1 (doubled for cross-region calls)'
    assert mapping_conflict('anthropic.claude-sonnet-4-5-20250929-v1:0', 'us', std, REGIONAL) is None


def test_quota_index_keeps_other_partitions_rows(monkeypatch, tmp_path, no_bundle):
    """A GovCloud run must not drop the commercial rows from quota-index.csv."""
    (tmp_path / 'data').mkdir()
    for region, code in (('us-east-1', 'L-COMM'), ('us-gov-west-1', 'L-GOV')):
        save_yaml(str(tmp_path / 'data' / f'fm-list-{region}.yml'), {'models': [
            {'model_id': 'amazon.nova-lite-v1:0', 'provider': 'Amazon', 'endpoints': {'base': {'quotas': {
                'tpm': {'code': code, 'name': 'On-demand tokens per minute for Amazon Nova Lite'}}}}}]})
    monkeypatch.setattr('bedrock_usage_analyzer.sync.regions.detect_credentials_partition', lambda _=None: 'aws-us-gov')
    checked = []
    monkeypatch.setattr(quota_index, 'check_quota', lambda code, region: checked.append(code) or ('ok', {'QuotaName': 'n'}))
    quota_index.QuotaIndexGenerator().run()
    csv_text = (tmp_path / 'data' / 'quota-index.csv').read_text()
    assert 'L-COMM' in csv_text and 'L-GOV' in csv_text
    assert set(checked) == {'L-GOV'}                  # never the other partition's codes


def test_quota_index_prefers_home_and_enabled_regions_and_writes_partition(monkeypatch, tmp_path, no_bundle, commercial_creds):
    (tmp_path / 'data').mkdir()
    save_yaml(str(tmp_path / 'data' / 'regions.yml'), {'regions': ['eu-west-1', 'us-east-1']})
    for region in ('af-south-1', 'eu-west-1', 'us-east-1'):
        save_yaml(str(tmp_path / 'data' / f'fm-list-{region}.yml'), {'models': [
            {'model_id': 'amazon.nova-lite-v1:0', 'provider': 'Amazon', 'endpoints': {'base': {'quotas': {
                'tpm': {'code': 'L-1', 'name': 'On-demand tokens per minute for Amazon Nova Lite'}}}}}]})
    checked = []
    monkeypatch.setattr(quota_index, 'check_quota', lambda code, region: checked.append(region) or ('ok', {'QuotaName': 'n'}))
    quota_index.QuotaIndexGenerator().run()
    assert checked[0] == 'us-east-1'                    # validated in the home region, not af-south-1
    header = (tmp_path / 'data' / 'quota-index.csv').read_text().splitlines()[0]
    assert header.endswith(',partition')


def test_endpoint_listed_tells_missing_endpoint_from_unmapped_one():
    from bedrock_usage_analyzer.core.analyzer import BedrockAnalyzer
    analyzer = BedrockAnalyzer.__new__(BedrockAnalyzer)
    analyzer.region = 'us-east-1'
    analyzer._fm_models = [{'model_id': 'm', 'endpoints': {'base': {'quotas': {}}, 'us': None}}]
    assert analyzer._endpoint_listed('m', None) and analyzer._endpoint_listed('m', 'us')
    assert not analyzer._endpoint_listed('m', 'eu')
    assert not analyzer._endpoint_listed('new-model', None)


def test_shared_tpd_quota_is_allowed_on_base():
    name = 'Model invocation max tokens per day for Amazon Nova Lite (doubled for cross-region calls)'
    assert mapping_conflict('amazon.nova-lite-v1:0', 'base', name, {'us', 'eu', 'apac'}) is None
    assert mapping_conflict('amazon.nova-lite-v1:0', 'base',
                            'Cross-region model inference tokens per minute for Amazon Nova Lite',
                            {'us', 'eu', 'apac'})


def test_quota_index_tolerates_null_endpoints():
    from bedrock_usage_analyzer.sync.quota_index import QuotaIndexGenerator
    gen = QuotaIndexGenerator.__new__(QuotaIndexGenerator)
    gen.models, gen.entries = {}, []
    key = ('aws', 'm')
    gen.models[key] = {'model_id': 'm', 'partition': 'aws', 'endpoints': {}}
    gen._merge_endpoints(key, {'endpoints': {'us': None, 'base': {'quotas': None}}}, 'us-east-1')
    gen._extract_quota_entries()
    assert gen.entries == []


def test_quota_index_removes_code_missing_only_outside_source_region(monkeypatch, tmp_path, no_bundle, commercial_creds):
    (tmp_path / 'data').mkdir()
    save_yaml(str(tmp_path / 'data' / 'regions.yml'), {'regions': ['ap-southeast-2', 'us-east-1']})
    for region in ('ap-southeast-2', 'us-east-1'):
        save_yaml(str(tmp_path / 'data' / f'fm-list-{region}.yml'), {'models': [
            {'model_id': 'amazon.nova-lite-v1:0', 'provider': 'Amazon', 'endpoints': {'base': {'quotas': {
                'tpm': {'code': 'L-ABC', 'name': 'On-demand tokens per minute for Amazon Nova Lite'}}}}}]})
    monkeypatch.setattr(quota_index, 'check_quota', lambda code, region:
                        ('missing', None) if region == 'ap-southeast-2' else ('ok', {'QuotaName': 'n'}))
    quota_index.QuotaIndexGenerator().run()
    sydney = load_yaml(str(tmp_path / 'data' / 'fm-list-ap-southeast-2.yml'))
    virginia = load_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'))
    assert sydney['models'][0]['endpoints']['base']['quotas']['tpm'] is None
    assert virginia['models'][0]['endpoints']['base']['quotas']['tpm']['code'] == 'L-ABC'
    assert 'L-ABC' in (tmp_path / 'data' / 'quota-index.csv').read_text()


def test_quota_index_uses_region_listings_and_confirms_absent_codes(monkeypatch, tmp_path, no_bundle, commercial_creds):
    (tmp_path / 'data').mkdir()
    save_yaml(str(tmp_path / 'data' / 'regions.yml'), {'regions': ['ap-southeast-2', 'us-east-1']})
    for region in ('ap-southeast-2', 'us-east-1'):
        save_yaml(str(tmp_path / 'data' / f'fm-list-{region}.yml'), {'models': [
            {'model_id': 'amazon.nova-lite-v1:0', 'provider': 'Amazon', 'endpoints': {'base': {'quotas': {
                'tpm': {'code': 'L-ABC', 'name': 'On-demand tokens per minute for Amazon Nova Lite'}}}}}]})
    listed = {'us-east-1': {'L-ABC': {'QuotaCode': 'L-ABC', 'QuotaName': 'On-demand tokens per minute for Amazon Nova Lite'}},
              'ap-southeast-2': {}}
    monkeypatch.setattr(quota_index, 'list_quota_codes', lambda region: listed[region])
    confirmed = []
    monkeypatch.setattr(quota_index, 'check_quota', lambda code, region: confirmed.append((code, region)) or ('missing', None))
    quota_index.QuotaIndexGenerator().run()
    assert confirmed == [('L-ABC', 'ap-southeast-2')]          # only the code absent from a listing
    sydney = load_yaml(str(tmp_path / 'data' / 'fm-list-ap-southeast-2.yml'))
    assert sydney['models'][0]['endpoints']['base']['quotas']['tpm'] is None


def test_quota_index_skips_regions_the_account_has_not_enabled(monkeypatch, tmp_path, no_bundle, commercial_creds):
    (tmp_path / 'data').mkdir()
    save_yaml(str(tmp_path / 'data' / 'regions.yml'), {'regions': ['us-east-1']})
    for region in ('af-south-1', 'us-east-1'):
        save_yaml(str(tmp_path / 'data' / f'fm-list-{region}.yml'), {'models': [
            {'model_id': 'amazon.nova-lite-v1:0', 'provider': 'Amazon', 'endpoints': {'base': {'quotas': {
                'tpm': {'code': 'L-1', 'name': 'On-demand tokens per minute for Amazon Nova Lite'}}}}}]})
    listed, asked = [], []
    monkeypatch.setattr(quota_index, 'list_quota_codes', lambda region: listed.append(region) or None)
    monkeypatch.setattr(quota_index, 'check_quota', lambda code, region: asked.append(region) or ('ok', {'QuotaName': 'n'}))
    quota_index.QuotaIndexGenerator().run()
    assert 'af-south-1' not in listed + asked                 # opt-in region not enabled: no calls
    cape = load_yaml(str(tmp_path / 'data' / 'fm-list-af-south-1.yml'))
    assert cape['models'][0]['endpoints']['base']['quotas']['tpm']['code'] == 'L-1'   # kept
