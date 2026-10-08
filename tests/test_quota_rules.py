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


def test_a_cached_mapping_with_an_empty_slot_is_not_reused_where_the_region_has_more(monkeypatch):
    """us-east-1 has no TPD quota for the model; us-west-2 lists one: it is mapped there."""
    mapper = qm.QuotaMapper('us-east-1', 'm')
    calls = []

    def llm(region, model, fm_model_id, endpoint, candidates):
        codes = {c['code'] for c in candidates}
        calls.append(codes)
        return {'tpm': {'code': 'L-US', 'name': ''}, 'rpm': None, 'concurrent': None,
                'tpd': {'code': 'L-USTPD', 'name': ''} if 'L-USTPD' in codes else None}

    monkeypatch.setattr(qm, 'extract_quota_codes', llm)
    tpd = {'QuotaName': 'Model invocation max tokens per day for Anthropic Claude Sonnet 4 V1 (doubled for '
                        'cross-region calls)', 'QuotaCode': 'L-USTPD'}
    assert mapper._get_quota_mapping('us-east-1', SONNET4, 'claude', 'us', QUOTAS)['tpd'] is None
    assert mapper._get_quota_mapping('us-east-2', SONNET4, 'claude', 'us', QUOTAS)['tpd'] is None  # cached
    assert mapper._get_quota_mapping('us-west-2', SONNET4, 'claude', 'us', QUOTAS + [tpd])['tpd']['code'] == 'L-USTPD'
    assert len(calls) == 2 and 'L-USTPD' in calls[1]  # asked again only where the TPD quota is listed
    # Regions with and without it alternate: each reuses its own mapping
    assert mapper._get_quota_mapping('eu-west-1', SONNET4, 'claude', 'us', QUOTAS)['tpd'] is None
    assert mapper._get_quota_mapping('eu-west-2', SONNET4, 'claude', 'us', QUOTAS + [tpd])['tpd']['code'] == 'L-USTPD'
    assert len(calls) == 2
    # A candidate of a slot the mapping already fills does not make it ask again
    other_tpm = {'QuotaName': QUOTAS[0]['QuotaName'] + ' (legacy)', 'QuotaCode': 'L-US-OLD'}
    mapper._get_quota_mapping('eu-west-3', SONNET4, 'claude', 'us', QUOTAS + [tpd, other_tpm])
    assert len(calls) == 2


def test_a_candidate_the_llm_turned_down_is_not_asked_about_again(monkeypatch):
    mapper = qm.QuotaMapper('us-east-1', 'm')
    calls = []

    def llm(region, model, fm_model_id, endpoint, candidates):
        calls.append(1)  # always ignores the sibling's TPD quota
        return {'tpm': {'code': 'L-US', 'name': ''}, 'rpm': None, 'tpd': None, 'concurrent': None}

    monkeypatch.setattr(qm, 'extract_quota_codes', llm)
    sibling_tpd = {'QuotaName': 'Model invocation max tokens per day for Anthropic Claude Sonnet 4 V1 Large '
                                '(doubled for cross-region calls)', 'QuotaCode': 'L-SIB'}
    for region in ('us-east-1', 'us-east-2', 'us-west-2'):
        assert mapper._get_quota_mapping(region, SONNET4, 'claude', 'us', QUOTAS + [sibling_tpd])['tpd'] is None
    entries = mapper.lcode_cache[(SONNET4, mapper._rules()[0]['us'])]
    assert len(calls) == 1 and len(entries) == 1 and 'L-SIB' in entries[0][1]
    # Another region offers a second sibling: asked once, the same answer is one entry that
    # has now seen both, so a region offering either is not asked again
    sibling2 = {'QuotaName': sibling_tpd['QuotaName'].replace('Large', 'Small'), 'QuotaCode': 'L-SIB2'}
    mapper._get_quota_mapping('eu-west-1', SONNET4, 'claude', 'us', QUOTAS + [sibling2])
    mapper._get_quota_mapping('eu-west-2', SONNET4, 'claude', 'us', QUOTAS + [sibling_tpd, sibling2])
    assert len(calls) == 2 and len(entries) == 1 and {'L-SIB', 'L-SIB2'} <= entries[0][1]


def test_quota_index_removes_saved_mismatches_in_every_region(monkeypatch, tmp_path, no_bundle, commercial_creds):
    (tmp_path / 'data').mkdir()
    wrong = {'code': 'L-GL46', 'name': 'Global cross-region model inference tokens per minute for Anthropic Claude Sonnet 4.6'}
    right = {'code': 'L-US', 'name': 'Cross-region model inference tokens per minute for Anthropic Claude Sonnet 4 V1'}
    for region, us_tpm in (('us-east-1', right), ('us-west-2', wrong)):
        save_yaml(str(tmp_path / 'data' / f'fm-list-{region}.yml'), {'models': [
            {'model_id': SONNET4, 'provider': 'Anthropic', 'endpoints': {
                'us': {'quotas': {'tpm': us_tpm}}, 'global': {'quotas': {'tpm': wrong}}}}]})
    names = {'L-GL46': wrong['name'], 'L-US': right['name']}
    monkeypatch.setattr('bedrock_usage_analyzer.aws.servicequotas.check_quota', lambda code, region: ('ok', {'QuotaName': names[code]}))
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
    monkeypatch.setattr(qm, 'list_quota_codes', lambda region: {})
    monkeypatch.setattr(qm, 'extract_common_name', lambda *a: 'nova')
    monkeypatch.setattr('bedrock_usage_analyzer.sync.regions.detect_partition', lambda _=None: ('aws', None))
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
    monkeypatch.setattr('bedrock_usage_analyzer.aws.servicequotas.check_quota', lambda code, region: ('error', None))   # e.g. throttled
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
    monkeypatch.setattr(qm, 'list_quota_codes', lambda region: {})
    monkeypatch.setattr(qm, 'extract_common_name', lambda *a: 'claude')
    monkeypatch.setattr('bedrock_usage_analyzer.sync.regions.detect_partition', lambda _=None: ('aws', None))
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
    monkeypatch.setattr('bedrock_usage_analyzer.aws.servicequotas.check_quota', lambda code, region: checked.append(region) or ('ok', {'QuotaName': 'n'}))
    gen = quota_index.QuotaIndexGenerator()
    gen.run()
    assert set(checked) == {'us-east-1'} and set(gen._fm_data) == {'us-east-1'}


def test_quota_index_cleanup_never_creates_user_copies(monkeypatch, tmp_path, commercial_creds):
    """Bundled lists are left alone (a user copy would hide future bundled updates)."""
    before = set((tmp_path / 'data').glob('*')) if (tmp_path / 'data').exists() else set()
    monkeypatch.setattr('bedrock_usage_analyzer.aws.servicequotas.check_quota', lambda code, region: ('missing', None))
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


def test_model_level_quotas_are_not_base_quotas_of_an_entry_with_endpoints(tmp_path):
    from bedrock_usage_analyzer.core.analyzer import BedrockAnalyzer
    (tmp_path / 'data').mkdir()
    save_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'), {'models': [
        {'model_id': SONNET4, 'provider': 'Anthropic', 'endpoints': {'us': {'quotas': {}}},
         'quotas': {'tpm': {'code': 'L-OLD', 'name': 'On-demand tokens per minute for Sonnet 4'}}}]})
    analyzer = BedrockAnalyzer.__new__(BedrockAnalyzer)
    analyzer.region, analyzer._fm_models = 'us-east-1', None
    assert analyzer._load_quota_codes(SONNET4, None) == {}


def test_mapper_drops_conflicts_even_without_common_name(monkeypatch, tmp_path):
    (tmp_path / 'data').mkdir()
    save_yaml(str(tmp_path / 'data' / 'regions.yml'), {'regions': ['us-east-1']})
    wrong = {'code': 'L-GL46', 'name': 'Global cross-region model inference tokens per minute for Anthropic Claude Sonnet 4.6'}
    save_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'), {'models': [
        {'model_id': SONNET4, 'provider': 'Anthropic', 'endpoints': {'us': {'quotas': {'tpm': wrong}}}}]})
    monkeypatch.setattr(qm, 'list_quota_codes', lambda region: {})
    monkeypatch.setattr(qm, 'extract_common_name', lambda *a: None)      # LLM failed
    monkeypatch.setattr('bedrock_usage_analyzer.sync.regions.detect_partition', lambda _=None: ('aws', None))
    qm.QuotaMapper('us-east-1', 'model', 'us-east-1').run()
    assert load_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'))['models'][0]['endpoints']['us']['quotas']['tpm'] is None


def test_mapper_skips_custom_endpoints_where_the_region_has_no_custom_quotas(monkeypatch, tmp_path):
    (tmp_path / 'data').mkdir()
    save_yaml(str(tmp_path / 'data' / 'regions.yml'), {'regions': ['us-east-1']})
    save_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'), {'models': [
        {'model_id': 'amazon.nova-lite-v1:0:300k', 'provider': 'Amazon', 'endpoints': {'custom': {'quotas': {}}}}]})
    monkeypatch.setattr(qm, 'list_quota_codes', lambda region: {'L-1': {
        'QuotaCode': 'L-1', 'QuotaName': 'On-demand model inference tokens per minute for Amazon Nova Lite'}})
    asked = []
    monkeypatch.setattr(qm, 'extract_common_name', lambda *a: asked.append(a))
    monkeypatch.setattr('bedrock_usage_analyzer.sync.regions.detect_partition', lambda _=None: ('aws', None))
    qm.QuotaMapper('us-east-1', 'model', 'us-east-1').run()
    assert asked == []


def test_long_context_variant_quota_is_rejected():
    name = 'Model invocation max tokens per day for Anthropic Claude Sonnet 4.5 V1 1M Context Length (doubled for cross-region calls)'
    assert mapping_conflict('anthropic.claude-sonnet-4-5-20250929-v1:0', 'us', name, REGIONAL) == 'long-context variant quota'
    std = 'Model invocation max tokens per day for Anthropic Claude Sonnet 4.5 V1 (doubled for cross-region calls)'
    assert mapping_conflict('anthropic.claude-sonnet-4-5-20250929-v1:0', 'us', std, REGIONAL) is None


def test_a_version_inside_the_name_needs_it_in_the_quota():
    v1 = 'On-demand InvokeModel concurrent requests for Amazon Nova Sonic'
    assert mapping_conflict('amazon.nova-2-5-sonic', 'base', v1, REGIONAL) == 'quota names no version, model is 2.5'
    assert mapping_conflict('amazon.nova-sonic-v1:0', 'base', v1, REGIONAL) is None
    # A trailing or glued version is often left out of quota names
    assert mapping_conflict('twelvelabs.pegasus-1-2-v1:0', 'base',
                            'On-demand model inference requests per minute for Twelve Labs Pegasus', REGIONAL) is None
    assert mapping_conflict('qwen.qwen3-32b-v1:0', 'base',
                            'On-demand model inference tokens per minute for Qwen3 32B V1', REGIONAL) is None


def test_a_quota_of_another_api_version_is_rejected():
    v2 = 'Cross-Region model inference requests per minute for Anthropic Claude 3.5 Sonnet V2'
    assert mapping_conflict('anthropic.claude-3-5-sonnet-20240620-v1:0', 'apac', v2, REGIONAL) == \
        'quota is for V2, model is v1'
    assert mapping_conflict('anthropic.claude-3-5-sonnet-20241022-v2:0', 'apac', v2, REGIONAL) is None
    # A name without an API version, or with a generation such as 'V2.7', says nothing
    assert mapping_conflict('anthropic.claude-3-5-sonnet-20240620-v1:0', 'apac',
                            'Cross-region model inference requests per minute for Anthropic Claude 3.5 Sonnet',
                            REGIONAL) is None
    # Sibling models the version rules cannot tell apart are pinned to their names
    v1 = 'On-demand model inference tokens per minute for Anthropic Claude 3.5 Sonnet'
    assert 'sibling' in mapping_conflict('anthropic.claude-3-5-sonnet-20241022-v2:0', 'base', v1, REGIONAL)
    nano = 'On-demand model inference tokens per minute for NVIDIA Nemotron Nano 2'
    assert 'sibling' in mapping_conflict('nvidia.nemotron-nano-12b-v2', 'base', nano, REGIONAL)
    assert mapping_conflict('nvidia.nemotron-nano-12b-v2', 'base', nano + ' VL', REGIONAL) is None
    assert mapping_conflict('nvidia.nemotron-nano-9b-v2', 'base', nano, REGIONAL) is None
    assert 'sibling' in mapping_conflict('nvidia.nemotron-nano-9b-v2', 'base', nano + ' VL', REGIONAL)
    # A context-window variant of the model ID (where the fm-list keeps the 'custom' endpoint)
    custom_v2 = '(Model customization) Sum of on demand custom model deployment tokens per minute for ' \
                'Anthropic Claude 3.5 Sonnet V2'
    assert mapping_conflict('anthropic.claude-3-5-sonnet-20240620-v1:0:200k', 'custom', custom_v2,
                            REGIONAL) == 'quota is for V2, model is v1'
    assert mapping_conflict('anthropic.claude-3-5-sonnet-20241022-v2:0:200k', 'custom', custom_v2, REGIONAL) is None
    assert 'sibling' in mapping_conflict('anthropic.claude-3-5-sonnet-20241022-v2:0:200k', 'custom',
                                         custom_v2.replace(' V2', ''), REGIONAL)
    # A context variant without a unit (':512' tokens)
    embed_v4 = 'On-demand model inference requests per minute for Cohere Embed English V4'
    assert mapping_conflict('cohere.embed-english-v3:0:512', 'base', embed_v4, REGIONAL) == \
        mapping_conflict('cohere.embed-english-v3', 'base', embed_v4, REGIONAL) == 'quota is for V4, model is v3'
    # The API version written in lowercase
    assert mapping_conflict('anthropic.claude-3-5-sonnet-20240620-v1:0', 'apac',
                            'Cross-region model inference requests per minute for Anthropic Claude 3.5 Sonnet v2',
                            REGIONAL) == 'quota is for V2, model is v1'
    # A 'V3' that is the model's generation, as its ID says
    assert mapping_conflict('deepseek.v3-v1:0', 'base',
                            'On-demand model inference tokens per minute for DeepSeek V3', REGIONAL) is None
    assert mapping_conflict('twelvelabs.marengo-embed-2-7-v1:0', 'base',
                            'On-demand model inference requests per minute for TwelveLabs Marengo Embed V2.7',
                            REGIONAL) is None


def test_latency_optimized_quota_is_rejected():
    name = 'On-Demand, latency-optimized model inference tokens per minute for Amazon Nova Pro V1'
    assert mapping_conflict('amazon.nova-pro-v1:0', 'base', name, REGIONAL) == 'latency-optimized inference quota'
    std = 'On-demand model inference tokens per minute for Amazon Nova Pro'
    assert mapping_conflict('amazon.nova-pro-v1:0', 'base', std, REGIONAL) is None


def test_quota_index_keeps_other_partitions_rows(monkeypatch, tmp_path, no_bundle):
    """A GovCloud run must not drop the commercial rows from quota-index.csv."""
    (tmp_path / 'data').mkdir()
    for region, code in (('us-east-1', 'L-COMM'), ('us-gov-west-1', 'L-GOV')):
        save_yaml(str(tmp_path / 'data' / f'fm-list-{region}.yml'), {'models': [
            {'model_id': 'amazon.nova-lite-v1:0', 'provider': 'Amazon', 'endpoints': {'base': {'quotas': {
                'tpm': {'code': code, 'name': 'On-demand tokens per minute for Amazon Nova Lite'}}}}}]})
    monkeypatch.setattr('bedrock_usage_analyzer.sync.regions.detect_partition', lambda _=None: ('aws-us-gov', None))
    checked = []
    monkeypatch.setattr('bedrock_usage_analyzer.aws.servicequotas.check_quota', lambda code, region: checked.append(code) or ('ok', {'QuotaName': 'n'}))
    quota_index.QuotaIndexGenerator().run()
    csv_text = (tmp_path / 'data' / 'quota-index.csv').read_text()
    assert 'L-COMM' in csv_text and 'L-GOV' in csv_text
    assert set(checked) == {'L-GOV'}                  # never the other partition's codes


def test_quota_index_keeps_a_slot_only_a_later_region_maps(monkeypatch, tmp_path, no_bundle, commercial_creds):
    """ap-south-1 maps a TPD the first region leaves empty: it is indexed, checked there."""
    (tmp_path / 'data').mkdir()
    save_yaml(str(tmp_path / 'data' / 'regions.yml'), {'regions': ['ap-northeast-2', 'ap-south-1']})
    name = 'Cross-Region model inference {} for Anthropic Claude 3.5 Sonnet V2'
    for region, tpd in (('ap-northeast-2', None), ('ap-south-1', {'code': 'L-TPD', 'name': name.format('tokens per day')})):
        save_yaml(str(tmp_path / 'data' / f'fm-list-{region}.yml'), {'models': [
            {'model_id': 'anthropic.claude-3-5-sonnet-20241022-v2:0', 'provider': 'Anthropic', 'endpoints': {
                'apac': {'quotas': {'rpm': {'code': 'L-RPM', 'name': name.format('requests per minute')}, 'tpd': tpd}}}}]})
    checked = []
    names = {'L-RPM': name.format('requests per minute'), 'L-TPD': name.format('tokens per day')}
    monkeypatch.setattr('bedrock_usage_analyzer.aws.servicequotas.check_quota',
                        lambda code, region: checked.append((code, region)) or ('ok', {'QuotaName': names[code]}))
    quota_index.QuotaIndexGenerator().run()
    csv_text = (tmp_path / 'data' / 'quota-index.csv').read_text()
    assert 'L-RPM' in csv_text and 'L-TPD' in csv_text
    assert ('L-RPM', 'ap-northeast-2') in checked and ('L-TPD', 'ap-south-1') in checked


def test_quota_index_prefers_home_and_enabled_regions_and_writes_partition(monkeypatch, tmp_path, no_bundle, commercial_creds):
    (tmp_path / 'data').mkdir()
    save_yaml(str(tmp_path / 'data' / 'regions.yml'), {'regions': ['eu-west-1', 'us-east-1']})
    for region in ('af-south-1', 'eu-west-1', 'us-east-1'):
        save_yaml(str(tmp_path / 'data' / f'fm-list-{region}.yml'), {'models': [
            {'model_id': 'amazon.nova-lite-v1:0', 'provider': 'Amazon', 'endpoints': {'base': {'quotas': {
                'tpm': {'code': 'L-1', 'name': 'On-demand tokens per minute for Amazon Nova Lite'}}}}}]})
    checked = []
    monkeypatch.setattr('bedrock_usage_analyzer.aws.servicequotas.check_quota', lambda code, region: checked.append(region) or ('ok', {'QuotaName': 'n'}))
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
    monkeypatch.setattr('bedrock_usage_analyzer.aws.servicequotas.check_quota', lambda code, region:
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
    monkeypatch.setattr(quota_index, 'list_quota_codes', lambda region, **_: listed[region])
    confirmed = []
    monkeypatch.setattr('bedrock_usage_analyzer.aws.servicequotas.check_quota', lambda code, region: confirmed.append((code, region)) or ('missing', None))
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
    monkeypatch.setattr(quota_index, 'list_quota_codes', lambda region, **_: listed.append(region) or None)
    monkeypatch.setattr('bedrock_usage_analyzer.aws.servicequotas.check_quota', lambda code, region: asked.append(region) or ('ok', {'QuotaName': 'n'}))
    quota_index.QuotaIndexGenerator().run()
    assert 'af-south-1' not in listed + asked                 # opt-in region not enabled: no calls
    cape = load_yaml(str(tmp_path / 'data' / 'fm-list-af-south-1.yml'))
    assert cape['models'][0]['endpoints']['base']['quotas']['tpm']['code'] == 'L-1'   # kept


def test_quota_index_ignores_regions_file_of_another_partition(monkeypatch, tmp_path, no_bundle):
    (tmp_path / 'data').mkdir()
    save_yaml(str(tmp_path / 'data' / 'regions.yml'), {'regions': ['us-east-1', 'us-west-2']})
    for region in ('us-gov-east-1', 'us-gov-west-1'):
        save_yaml(str(tmp_path / 'data' / f'fm-list-{region}.yml'), {'models': [
            {'model_id': 'amazon.nova-lite-v1:0', 'provider': 'Amazon', 'endpoints': {'base': {'quotas': {
                'tpm': {'code': 'L-G', 'name': 'On-demand tokens per minute for Amazon Nova Lite'}}}}}]})
    monkeypatch.setattr('bedrock_usage_analyzer.sync.regions.detect_partition', lambda _=None: ('aws-us-gov', None))
    asked = []
    monkeypatch.setattr('bedrock_usage_analyzer.aws.servicequotas.check_quota', lambda code, region: asked.append(region) or ('ok', {'QuotaName': 'n'}))
    quota_index.QuotaIndexGenerator().run()
    assert 'us-gov-east-1' in asked                          # not dropped by a commercial regions.yml


def test_fm_quotas_drops_saved_codes_missing_from_the_listing(monkeypatch):
    fm = {'model_id': 'm', 'endpoints': {'base': {'quotas': {
        'tpm': {'code': 'L-GONE', 'name': 'x'}, 'rpm': {'code': 'L-OK', 'name': 'y'},
        'tpd': {'code': 'L-UNLISTED', 'name': 'z'}}}}}
    # Absent from the listing: dropped only when GetServiceQuota confirms it is missing
    monkeypatch.setattr('bedrock_usage_analyzer.aws.servicequotas.check_quota', lambda code, region:
                        ('missing', None) if code == 'L-GONE' else ('ok', {}))
    qm.QuotaMapper._drop_unlisted_saved_codes(fm, {'L-OK'}, 'us-east-1')
    assert fm['endpoints']['base']['quotas'] == {
        'tpm': None, 'rpm': {'code': 'L-OK', 'name': 'y'}, 'tpd': {'code': 'L-UNLISTED', 'name': 'z'}}
    qm.QuotaMapper._drop_unlisted_saved_codes(fm, set(), 'us-east-1')          # failed listing: no change
    assert fm['endpoints']['base']['quotas']['rpm']['code'] == 'L-OK'


def test_fm_quotas_confirms_each_unlisted_code_once(monkeypatch):
    shared = {'code': 'L-SHARED', 'name': 'x'}
    fm = {'model_id': 'm', 'endpoints': {'jp': {'quotas': {'tpm': dict(shared)}},
                                         'apac': {'quotas': {'tpm': dict(shared)}}}}
    asked = []
    monkeypatch.setattr('bedrock_usage_analyzer.aws.servicequotas.check_quota', lambda code, region: asked.append(code) or ('ok', {}))
    qm.QuotaMapper._drop_unlisted_saved_codes(fm, {'L-OTHER'}, 'ap-northeast-1', {})
    assert asked == ['L-SHARED']


def test_scrub_conflicting_reports_and_nulls():
    from bedrock_usage_analyzer.sync.quota_rules import scrub_conflicting
    quotas = {'tpm': {'code': 'L-G', 'name': 'Global cross-region model inference tokens per minute for X'},
              'rpm': {'code': 'L-R', 'name': 'Cross-region model inference requests per minute for X'}}
    removed = scrub_conflicting('x.model-v1:0', 'us', quotas, {'us'})
    assert [m for m, _, _ in removed] == ['tpm'] and quotas['tpm'] is None and quotas['rpm']['code'] == 'L-R'


def test_saved_quota_of_another_metric_is_scrubbed():
    """An RPM quota saved in the TPM slot (by an older version) is not shown as the TPM limit."""
    from bedrock_usage_analyzer.sync.quota_rules import scrub_conflicting, slot_conflict
    quotas = {'tpm': {'code': 'L-R', 'name': 'On-demand model inference requests per minute for Amazon Nova Pro'},
              'rpm': {'code': 'L-R', 'name': 'On-demand model inference requests per minute for Amazon Nova Pro'},
              'tpd': {'code': 'L-X', 'name': 'shortened name'}}
    removed = scrub_conflicting('amazon.nova-pro-v1:0', 'base', quotas, {'us'})
    assert [m for m, _, _ in removed] == ['tpm'] and quotas['rpm'] and quotas['tpd']
    assert 'requests per minute' in slot_conflict('amazon.nova-pro-v1:0', 'base', 'tpm',
                                                  'On-demand model inference requests per minute for X', {'us'})


def test_quota_index_without_regions_file_checks_only_account_regions(monkeypatch, tmp_path, no_bundle, commercial_creds):
    (tmp_path / 'data').mkdir()
    for region in ('ap-east-2', 'us-east-1'):
        save_yaml(str(tmp_path / 'data' / f'fm-list-{region}.yml'), {'models': [
            {'model_id': 'amazon.nova-lite-v1:0', 'provider': 'Amazon', 'endpoints': {'base': {'quotas': {
                'tpm': {'code': 'L-1', 'name': 'On-demand tokens per minute for Amazon Nova Lite'}}}}}]})
    monkeypatch.setattr(quota_index, '_account_regions', lambda partition: ['us-east-1'])
    asked = []
    monkeypatch.setattr('bedrock_usage_analyzer.aws.servicequotas.check_quota',
                        lambda code, region: asked.append(region) or ('ok', {'QuotaName': 'n'}))
    quota_index.QuotaIndexGenerator().run()
    assert 'ap-east-2' not in asked                           # opt-in region the account has not enabled


def test_quota_index_tolerates_malformed_fm_lists(monkeypatch, tmp_path, no_bundle, commercial_creds):
    (tmp_path / 'data').mkdir()
    save_yaml(str(tmp_path / 'data' / 'regions.yml'), {'regions': ['us-east-1', 'us-west-2']})
    (tmp_path / 'data' / 'fm-list-us-west-2.yml').write_text('models:\n')
    save_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'), {'models': [{'provider': 'X'}, {
        'model_id': 'amazon.nova-lite-v1:0', 'endpoints': {'base': {'quotas': {
            'tpm': {'code': 'L-1', 'name': 'On-demand tokens per minute for Amazon Nova Lite'}}}}}]})
    monkeypatch.setattr('bedrock_usage_analyzer.aws.servicequotas.check_quota', lambda code, region: ('ok', {'QuotaName': 'n'}))
    quota_index.QuotaIndexGenerator().run()                  # no crash
    assert 'L-1' in (tmp_path / 'data' / 'quota-index.csv').read_text()


def test_quota_index_cleanup_keeps_malformed_entries_in_the_user_file(monkeypatch, tmp_path, no_bundle, commercial_creds):
    (tmp_path / 'data').mkdir()
    save_yaml(str(tmp_path / 'data' / 'regions.yml'), {'regions': ['us-east-1']})
    user_file = tmp_path / 'data' / 'fm-list-us-east-1.yml'
    save_yaml(str(user_file), {'models': [{'model-id': 'typo', 'endpoints': {}}, {
        'model_id': 'amazon.nova-lite-v1:0', 'endpoints': {'base': {'quotas': {
            'tpm': {'code': 'L-X', 'name': 'Cross-region model inference tokens per minute for Amazon Nova Lite'}}}}}]})
    monkeypatch.setattr('bedrock_usage_analyzer.aws.servicequotas.check_quota', lambda code, region: ('ok', {'QuotaName': 'n'}))
    quota_index.QuotaIndexGenerator().run()
    models = load_yaml(str(user_file))['models']
    assert models[0] == {'model-id': 'typo', 'endpoints': {}}                  # kept as written
    assert models[1]['endpoints']['base']['quotas']['tpm'] is None             # mismatch removed


def test_fm_quotas_tolerates_null_endpoints(monkeypatch):
    mapper = qm.QuotaMapper('us-east-1', 'm')
    assert mapper._get_endpoints_to_process({'model_id': 'x', 'endpoints': None}) == []


def test_quota_index_lists_bundled_files_of_a_zipped_package(monkeypatch):
    from bedrock_usage_analyzer.utils import paths
    monkeypatch.setattr(paths, 'list_data_files', lambda pattern='*.yml': [])   # no file paths (zip)
    assert 'fm-list-us-east-1.yml' in paths.list_data_names('fm-list-*.yml')


def test_llm_pick_of_another_metric_is_rejected():
    candidates = [{'code': 'L-RPM', 'name': 'Cross-region model inference requests per minute for X'},
                  {'code': 'L-TPM', 'name': 'Cross-region model inference tokens per minute for X'}]
    picked = {'tpm': {'code': 'L-RPM', 'name': ''}, 'rpm': {'code': 'L-RPM', 'name': ''}}
    cleaned = qm.QuotaMapper._drop_invalid_choices(picked, candidates, 'x.m', 'us')
    assert cleaned['tpm'] is None and cleaned['rpm']['code'] == 'L-RPM'


def test_quota_index_tells_unchecked_regions_from_api_errors(monkeypatch, tmp_path, no_bundle, commercial_creds, caplog):
    import logging
    caplog.set_level(logging.INFO)
    (tmp_path / 'data').mkdir()
    save_yaml(str(tmp_path / 'data' / 'fm-list-ap-southeast-7.yml'), {'models': [
        {'model_id': 'amazon.nova-lite-v1:0', 'endpoints': {'base': {'quotas': {
            'tpm': {'code': 'L-7', 'name': 'On-demand tokens per minute for Amazon Nova Lite'}}}}}]})
    monkeypatch.setattr(quota_index, '_account_regions', lambda partition: ['us-east-1'])
    quota_index.QuotaIndexGenerator().run()
    assert 'not enabled for this account' in caplog.text and 'API errors' not in caplog.text


def test_update_bundle_cleans_the_checkout_file_not_the_user_copy(monkeypatch, tmp_path, no_bundle, commercial_creds):
    (tmp_path / 'data').mkdir()
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    save_yaml(str(tmp_path / 'data' / 'regions.yml'), {'regions': ['us-east-1']})
    stale = {'model_id': 'amazon.nova-lite-v1:0', 'endpoints': {'base': {'quotas': {
        'tpm': {'code': 'L-X', 'name': 'Cross-region model inference tokens per minute for Amazon Nova Lite'}}}}}
    save_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'), {'models': [stale]})
    newer = dict(stale, endpoints={**stale['endpoints'], 'us': {'quotas': {}}})
    save_yaml(str(checkout / 'fm-list-us-east-1.yml'), {'models': [newer]})
    monkeypatch.setattr(quota_index, 'get_bundle_path', lambda: checkout)
    monkeypatch.setattr('bedrock_usage_analyzer.aws.servicequotas.check_quota', lambda code, region: ('ok', {'QuotaName': 'n'}))
    quota_index.QuotaIndexGenerator().run(update_bundle=True)
    written = load_yaml(str(checkout / 'fm-list-us-east-1.yml'))['models'][0]['endpoints']
    assert 'us' in written and written['base']['quotas']['tpm'] is None   # newer entry kept, code removed
    user_copy = load_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'))['models'][0]['endpoints']
    assert 'us' not in user_copy                                          # not overwritten with the checkout's list


def test_fm_quotas_update_bundle_reads_and_writes_the_checkout(monkeypatch, tmp_path):
    (tmp_path / 'data').mkdir()
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    save_yaml(str(tmp_path / 'data' / 'regions.yml'), {'regions': ['us-east-1']})
    user = {'models': [{'model_id': 'amazon.nova-lite-v1:0', 'provider': 'Amazon', 'endpoints': {'base': {'quotas': {}}}}]}
    save_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'), user)
    save_yaml(str(checkout / 'fm-list-us-east-1.yml'), {'models': [
        {'model_id': 'amazon.nova-lite-v1:0', 'provider': 'Amazon', 'endpoints': {'base': {'quotas': {}}, 'us': {'quotas': {}}}}]})
    monkeypatch.setattr(qm, 'get_bundle_path', lambda: checkout)
    monkeypatch.setattr(qm, 'list_quota_codes', lambda region: {})
    monkeypatch.setattr(qm, 'extract_common_name', lambda *a: 'nova')
    monkeypatch.setattr('bedrock_usage_analyzer.sync.regions.detect_partition', lambda _=None: ('aws', None))
    mapper = qm.QuotaMapper('us-east-1', 'model', 'us-east-1')
    monkeypatch.setattr(mapper, '_get_quota_mapping', lambda *a: {'tpm': {'code': 'L-N', 'name': 'n'}})
    mapper.run(update_bundle=True)
    assert 'us' in load_yaml(str(checkout / 'fm-list-us-east-1.yml'))['models'][0]['endpoints']
    assert load_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml')) == user          # user copy untouched


def test_quota_index_tolerates_non_mapping_quotas(monkeypatch, tmp_path, no_bundle, commercial_creds):
    (tmp_path / 'data').mkdir()
    save_yaml(str(tmp_path / 'data' / 'regions.yml'), {'regions': ['us-east-1', 'us-west-2']})
    for region in ('us-east-1', 'us-west-2'):
        save_yaml(str(tmp_path / 'data' / f'fm-list-{region}.yml'), {'models': [
            {'model_id': 'amazon.nova-lite-v1:0', 'endpoints': {'us': {'quotas': 'TODO'}, 'base': 'TODO'}}]})
    monkeypatch.setattr('bedrock_usage_analyzer.aws.servicequotas.check_quota', lambda code, region: ('ok', {}))
    quota_index.QuotaIndexGenerator().run()                  # no crash
    assert (tmp_path / 'data' / 'quota-index.csv').exists()


def test_fm_quotas_merge_tolerates_a_hand_edited_endpoint(monkeypatch, tmp_path):
    (tmp_path / 'data').mkdir()
    save_yaml(str(tmp_path / 'data' / 'regions.yml'), {'regions': ['us-east-1']})
    save_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'), {'models': [
        {'model_id': 'amazon.nova-lite-v1:0', 'provider': 'Amazon', 'endpoints': {'base': 'TODO'}}]})
    monkeypatch.setattr(qm, 'list_quota_codes', lambda region: {})
    monkeypatch.setattr(qm, 'extract_common_name', lambda *a: 'nova')
    monkeypatch.setattr('bedrock_usage_analyzer.sync.regions.detect_partition', lambda _=None: ('aws', None))
    mapper = qm.QuotaMapper('us-east-1', 'model', 'us-east-1')
    monkeypatch.setattr(mapper, '_get_quota_mapping', lambda *a: {'tpm': {'code': 'L-N', 'name': 'n'}})
    mapper.run()
    saved = load_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'))['models'][0]['endpoints']['base']
    assert saved == {'quotas': {'tpm': {'code': 'L-N', 'name': 'n'}}}
