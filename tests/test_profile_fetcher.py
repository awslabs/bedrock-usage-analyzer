# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application inference profile source resolution (issue #7) and discovery."""

import pytest

from bedrock_usage_analyzer.aws.bedrock import (
    list_inference_profiles,
    model_id_from_arn,
    region_from_arn,
    region_group,
    split_profile_id,
)
from bedrock_usage_analyzer.core.profile_fetcher import InferenceProfileFetcher

from conftest import (AU_ARNS, GLOBAL_ARNS, HAIKU, JP_ARNS, NOVA, FakeBedrock, app_profile, arn,
                      system_profile)


@pytest.mark.parametrize('endpoint,expected', [
    ('us.amazon.nova-pro-v1:0', ('amazon.nova-pro-v1:0', 'us')),
    ('us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0', ('anthropic.claude-sonnet-4-5-20250929-v1:0', 'us-gov')),
    ('apac.amazon.nova-lite-v1:0', ('amazon.nova-lite-v1:0', 'apac')),
    ('ca.x.y', ('x.y', 'ca')),
    ('global.anthropic.claude-opus-5-5', ('anthropic.claude-opus-5-5', 'global')),
    ('amazon.nova-pro-v1:0', ('amazon.nova-pro-v1:0', None)),
    ('deepseek.v3.2', ('deepseek.v3.2', None)),          # two dots, still a base model
    ('moonshotai.kimi-k2.5', ('moonshotai.kimi-k2.5', None)),
])
def test_split_profile_id(endpoint, expected):
    assert split_profile_id(endpoint) == expected


def test_arn_helpers():
    assert model_id_from_arn(arn('us-east-1', NOVA)) == NOVA
    assert model_id_from_arn('arn:aws:bedrock:us-east-1:1:inference-profile/us.x') is None
    assert region_from_arn(arn('', HAIKU)) == ''
    assert region_from_arn('bad') == ''
    assert region_group('us-gov-west-1') == 'us-gov'
    assert region_group('eu-central-1') == 'eu'


def test_list_inference_profiles_paginates():
    client = FakeBedrock(system=[system_profile(f"p{i}.m", [arn('us-east-1', 'm')]) for i in range(5)],
                         page_size=2)
    assert len(list_inference_profiles(client, 'SYSTEM_DEFINED')) == 5
    assert [c[2] for c in client.calls] == [None, '2', '4']


def test_list_inference_profiles_without_api():
    assert list_inference_profiles(object(), 'APPLICATION') == []


def test_issue_7_au_application_profile_is_matched(sydney_bedrock):
    """An app profile copied from au.* must not be mistaken for apac.* (issue #7)."""
    fetcher = InferenceProfileFetcher(sydney_bedrock)
    ids, names, metadata = fetcher.find_profiles(HAIKU, 'au')
    assert ids == [f"au.{HAIKU}", 'auapp000001']
    assert names['auapp000001'] == 'team-a-au-haiku'
    assert metadata['auapp000001'] == {'id': 'auapp000001', 'tags': {'team': 'a', 'env': 'prod'}}
    assert metadata[f"au.{HAIKU}"] == {'id': 'N/A', 'tags': {}}


@pytest.mark.parametrize('prefix,expected_app', [
    ('global', 'glapp000001'),
    ('jp', 'jpapp000001'),
    (None, 'baseapp0001'),
])
def test_each_source_matches_only_its_profiles(sydney_bedrock, prefix, expected_app):
    fetcher = InferenceProfileFetcher(sydney_bedrock)
    ids, _, _ = fetcher.find_profiles(HAIKU, prefix)
    assert ids[1:] == [expected_app]


def test_apac_has_no_haiku_app_profiles(sydney_bedrock):
    ids, _, _ = InferenceProfileFetcher(sydney_bedrock).find_profiles(HAIKU, 'apac')
    assert ids == [f"apac.{HAIKU}"]


def test_list_application_profiles_resolves_sources(sydney_bedrock):
    apps = InferenceProfileFetcher(sydney_bedrock).list_application_profiles()
    sources = {a['id']: (a['source'], a['model_id'], a['profile_prefix']) for a in apps}
    assert sources == {
        'auapp000001': (f"au.{HAIKU}", HAIKU, 'au'),
        'glapp000001': (f"global.{HAIKU}", HAIKU, 'global'),
        'jpapp000001': (f"jp.{HAIKU}", HAIKU, 'jp'),
        'baseapp0001': (HAIKU, HAIKU, None),
        'novaapp0001': (f"apac.{NOVA}", NOVA, 'apac'),
    }


def test_profiles_are_listed_once(sydney_bedrock):
    fetcher = InferenceProfileFetcher(sydney_bedrock)
    fetcher.find_profiles(HAIKU, 'au')
    fetcher.find_profiles(HAIKU, 'global')
    fetcher.list_application_profiles()
    listing_calls = [c for c in sydney_bedrock.calls if c[0] == 'list_inference_profiles']
    # 5 system profiles (3 pages) + 5 application profiles (3 pages), fetched once each
    assert len(listing_calls) == 6


def test_scoped_find_profiles_returns_only_selected(sydney_bedrock):
    ids, names, _ = InferenceProfileFetcher(sydney_bedrock).find_profiles(
        HAIKU, 'au', application_profile_ids=['auapp000001'])
    assert ids == ['auapp000001']
    assert list(names) == ['auapp000001']


def test_resolve_application_profile_by_id_arn_or_name(sydney_bedrock):
    fetcher = InferenceProfileFetcher(sydney_bedrock)
    by_id = fetcher.resolve_application_profile('jpapp000001')
    assert by_id['source'] == f"jp.{HAIKU}"
    assert fetcher.resolve_application_profile(by_id['arn'])['id'] == 'jpapp000001'
    assert fetcher.resolve_application_profile(' team-c-jp-haiku ')['id'] == 'jpapp000001'
    assert fetcher.resolve_application_profile('missing') is None


def test_other_sources_for_model(sydney_bedrock):
    fetcher = InferenceProfileFetcher(sydney_bedrock)
    assert fetcher.other_sources_for_model(HAIKU, 'apac') == {'au': 1, 'global': 1, 'jp': 1, 'base': 1}
    assert fetcher.other_sources_for_model(NOVA, 'apac') == {}


def test_closest_match_when_routing_set_changed():
    """A profile created before a region was added to au.* still resolves to au.*."""
    system = [system_profile(f"au.{HAIKU}", AU_ARNS + [arn('ap-southeast-6', HAIKU)]),
              system_profile(f"jp.{HAIKU}", JP_ARNS)]
    apps = [app_profile('old00000001', 'old-au', AU_ARNS)]
    fetcher = InferenceProfileFetcher(FakeBedrock(system=system, application=apps))
    assert fetcher.list_application_profiles()[0]['source'] == f"au.{HAIKU}"


def test_heuristic_fallback_without_system_profiles():
    apps = [
        app_profile('g0000000001', 'g', GLOBAL_ARNS),
        app_profile('e0000000001', 'e', [arn('eu-west-1', NOVA), arn('eu-central-1', NOVA)]),
        app_profile('a0000000001', 'a', [arn('ap-southeast-1', NOVA), arn('ap-northeast-1', NOVA)]),
        app_profile('v0000000001', 'v', [arn('us-gov-west-1', NOVA, 'aws-us-gov'),
                                         arn('us-gov-east-1', NOVA, 'aws-us-gov')]),
        app_profile('m0000000001', 'm', [arn('us-east-1', NOVA), arn('eu-west-1', NOVA)]),
    ]
    fetcher = InferenceProfileFetcher(FakeBedrock(application=apps))
    sources = {a['id']: a['source'] for a in fetcher.list_application_profiles()}
    assert sources == {
        'g0000000001': f"global.{HAIKU}",
        'e0000000001': f"eu.{NOVA}",
        'a0000000001': f"apac.{NOVA}",
        'v0000000001': f"us-gov.{NOVA}",
        'm0000000001': f"global.{NOVA}",
    }


def test_profiles_without_models_are_skipped():
    apps = [app_profile('empty000001', 'empty', []),
            app_profile('weird000001', 'weird', ['arn:aws:bedrock:us-east-1:1:custom-model/x'])]
    assert InferenceProfileFetcher(FakeBedrock(application=apps)).list_application_profiles() == []


def test_tag_errors_do_not_break_discovery(sydney_bedrock):
    def fail(**_):
        raise RuntimeError('AccessDenied')

    sydney_bedrock.list_tags_for_resource = fail
    _, _, metadata = InferenceProfileFetcher(sydney_bedrock).find_profiles(HAIKU, 'au')
    assert metadata['auapp000001']['tags'] == {}


def test_listing_denied_still_analyzes_the_endpoint(sydney_bedrock):
    def denied(**_):
        raise RuntimeError('AccessDeniedException: not authorized to perform bedrock:ListInferenceProfiles')

    sydney_bedrock.list_inference_profiles = denied
    fetcher = InferenceProfileFetcher(sydney_bedrock)
    ids, _, _ = fetcher.find_profiles(HAIKU, 'au')
    assert ids == [f"au.{HAIKU}"]
    assert fetcher.other_sources_for_model(HAIKU, 'au') == {}
    with pytest.raises(RuntimeError):
        fetcher.find_profiles(HAIKU, 'au', application_profile_ids=['auapp000001'])


def test_user_prefix_file_from_older_version_keeps_new_prefixes(tmp_path):
    """An old user prefix-mapping.yml must not hide 'us-gov' or 'in' (TPD doubling depends on it)."""
    from bedrock_usage_analyzer.aws import bedrock
    from bedrock_usage_analyzer.utils.yaml_handler import save_yaml
    (tmp_path / 'data').mkdir()
    save_yaml(str(tmp_path / 'data' / 'prefix-mapping.yml'), {'prefixes': [
        {'prefix': 'us', 'quota_keyword': 'cross-region', 'description': 'custom', 'is_regional': True}]})
    bedrock._prefix_mapping_cache = None
    assert {'us-gov', 'in', 'au', 'us'} <= set(bedrock.get_regional_profile_prefixes())
    assert bedrock.get_endpoint_descriptions()['us'] == 'custom'   # user entry wins


@pytest.mark.parametrize('regions,prefix', [
    (['ap-northeast-1', 'ap-northeast-3'], 'jp'),
    (['ap-southeast-2', 'ap-southeast-4'], 'au'),
    (['ap-south-1', 'ap-south-2'], 'in'),
    (['ap-southeast-1', 'ap-northeast-2'], 'apac'),
])
def test_fallback_tells_country_profiles_from_apac(regions, prefix):
    apps = [app_profile('x0000000001', 'x', [arn(rg, NOVA) for rg in regions])]
    fetcher = InferenceProfileFetcher(FakeBedrock(application=apps))
    assert fetcher.list_application_profiles()[0]['source'] == f"{prefix}.{NOVA}"


def test_failed_listing_is_not_retried(sydney_bedrock):
    calls = []

    def denied(**kwargs):
        calls.append(kwargs)
        raise RuntimeError('AccessDeniedException')

    sydney_bedrock.list_inference_profiles = denied
    fetcher = InferenceProfileFetcher(sydney_bedrock)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            fetcher.list_application_profiles()
    fetcher.find_profiles(HAIKU, 'au')
    fetcher.other_sources_for_model(HAIKU, 'au')
    assert len(calls) == 1
