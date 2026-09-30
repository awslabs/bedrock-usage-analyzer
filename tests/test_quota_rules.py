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


def test_quota_index_removes_saved_mismatches_in_every_region(monkeypatch, tmp_path):
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
