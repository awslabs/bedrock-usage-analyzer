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
    fetcher = InferenceProfileFetcher(sydney_bedrock, on_demand_models=[HAIKU])
    assert fetcher.other_sources_for_model(HAIKU, 'apac') == {}   # never triggers a listing
    fetcher.list_application_profiles()
    assert fetcher.other_sources_for_model(HAIKU, 'apac') == {'au': 1, 'global': 1, 'jp': 1, 'base': 1}
    assert fetcher.other_sources_for_model(NOVA, 'apac') == {}
    profile_only = InferenceProfileFetcher(sydney_bedrock)                  # no on-demand endpoint
    profile_only.list_application_profiles()
    assert 'base' not in profile_only.other_sources_for_model(HAIKU, 'apac')


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
        'm0000000001': None,                      # two families, no region-less ARN: not global
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


def test_shared_routing_set_belongs_to_every_matching_endpoint():
    """jp.X and apac.X route to the same two regions: a copy of either shows under both."""
    tokyo_osaka = [arn('ap-northeast-1', NOVA), arn('ap-northeast-3', NOVA)]
    system = [system_profile(f"apac.{NOVA}", tokyo_osaka), system_profile(f"jp.{NOVA}", tokyo_osaka)]
    apps = [app_profile('shared00001', 'shared', tokyo_osaka)]
    fetcher = InferenceProfileFetcher(FakeBedrock(system=system, application=apps))
    app = fetcher.list_application_profiles()[0]
    assert app['source'] == f"jp.{NOVA}" and app['sources'] == [f"jp.{NOVA}", f"apac.{NOVA}"]
    assert fetcher.find_profiles(NOVA, 'jp')[0][1:] == ['shared00001']
    assert fetcher.find_profiles(NOVA, 'apac')[0][1:] == ['shared00001']
    assert fetcher.other_sources_for_model(NOVA, 'jp') == {}


def test_country_fallback_learns_regions_from_system_profiles():
    """A new au region (here ap-southeast-9) is learned from the listed au.* profile."""
    au_new = [arn('ap-southeast-2', HAIKU), arn('ap-southeast-9', HAIKU)]
    system = [system_profile(f"au.{HAIKU}", au_new)]
    # Copy of an au profile for another model that has no system profile in this listing
    apps = [app_profile('newau000001', 'n', [arn('ap-southeast-9', NOVA), arn('ap-southeast-2', NOVA)])]
    fetcher = InferenceProfileFetcher(FakeBedrock(system=system, application=apps))
    assert fetcher.list_application_profiles()[0]['source'] == f"au.{NOVA}"


def test_throttled_listing_is_retried_but_denied_is_cached(sydney_bedrock):
    real = sydney_bedrock.list_inference_profiles
    state = {'fail': 'ThrottlingException: Rate exceeded', 'calls': 0}

    def flaky(**kwargs):
        state['calls'] += 1
        if state['fail']:
            state['fail'] = None                                   # one throttle, then fine
            raise RuntimeError('ThrottlingException: Rate exceeded')
        return real(**kwargs)

    sydney_bedrock.list_inference_profiles = flaky
    fetcher = InferenceProfileFetcher(sydney_bedrock)
    assert len(fetcher.list_application_profiles()) == 5          # retried within the same call

    denied = InferenceProfileFetcher(FakeBedrock())
    denied.bedrock_client.list_inference_profiles = lambda **k: (_ for _ in ()).throw(
        RuntimeError('AccessDeniedException: not authorized'))
    for _ in range(2):
        with pytest.raises(RuntimeError):
            denied.list_application_profiles()
    assert denied._listings['APPLICATION']['error'] is not None


def test_listing_failure_is_reported_as_a_warning(sydney_bedrock, caplog):
    import logging
    caplog.set_level(logging.WARNING)
    sydney_bedrock.list_inference_profiles = lambda **k: (_ for _ in ()).throw(RuntimeError('AccessDenied'))
    InferenceProfileFetcher(sydney_bedrock).find_profiles(HAIKU, 'au')
    assert 'without its application profiles' in caplog.text


def test_retired_country_profile_still_recognised():
    """jp.* retired, au.* still listed: a jp copy must not fall back to apac."""
    system = [system_profile(f"au.{HAIKU}", AU_ARNS)]
    apps = [app_profile('oldjp000001', 'old-jp', [arn('ap-northeast-1', NOVA), arn('ap-northeast-3', NOVA)])]
    fetcher = InferenceProfileFetcher(FakeBedrock(system=system, application=apps))
    assert fetcher.list_application_profiles()[0]['source'] == f"jp.{NOVA}"


def test_transient_listing_failure_gives_up_after_two_attempts(sydney_bedrock):
    calls = []

    def throttled(**k):
        calls.append(1)
        raise RuntimeError('ThrottlingException')

    sydney_bedrock.list_inference_profiles = throttled
    fetcher = InferenceProfileFetcher(sydney_bedrock)
    for _ in range(4):
        with pytest.raises(RuntimeError):
            fetcher.list_application_profiles()
    assert len(calls) == 2


def test_unknown_geography_keeps_profile_listed_without_a_source():
    """Two sa-* ARNs and no matching system profile: no invented 'sa.' endpoint."""
    apps = [app_profile('sa000000001', 'brazil', [arn('sa-east-1', NOVA), arn('sa-west-1', NOVA)])]
    fetcher = InferenceProfileFetcher(FakeBedrock(application=apps))
    app = fetcher.list_application_profiles()[0]
    assert app['source'] is None and app['sources'] == [] and app['model_id'] == NOVA
    assert fetcher.find_profiles(NOVA, None)[0] == [NOVA]
    assert fetcher.find_profiles(NOVA, 'unknown', application_profile_ids=['sa000000001'])[0] == ['sa000000001']


def test_new_system_prefix_is_taken_from_the_listed_profile():
    kr = [arn('ap-northeast-2', HAIKU), arn('ap-northeast-9', HAIKU)]
    fetcher = InferenceProfileFetcher(FakeBedrock(system=[system_profile(f"kr.{HAIKU}", kr)],
                                                  application=[app_profile('kr000000001', 'kr', kr)]))
    app = fetcher.list_application_profiles()[0]
    assert (app['model_id'], app['profile_prefix'], app['source']) == (HAIKU, 'kr', f"kr.{HAIKU}")


@pytest.mark.parametrize('region,group', [('us-iso-east-1', 'us-iso'), ('us-isob-east-1', 'us-isob'),
                                          ('eu-isoe-west-1', 'eu-isoe'), ('us-gov-east-1', 'us-gov'),
                                          ('ap-southeast-2', 'ap'), ('eusc-de-east-1', 'eusc')])
def test_region_group_keeps_partitions_apart(region, group):
    assert region_group(region) == group


def test_legacy_fm_list_entry_counts_as_on_demand():
    """An old entry (no 'endpoints', ON_DEMAND, model-level quotas) has the base endpoint, as the analyzer reads it."""
    from bedrock_usage_analyzer.utils.yaml_handler import endpoint_keys, has_endpoint
    legacy = {'model_id': NOVA, 'inference_types': ['ON_DEMAND'], 'quotas': {'tpm': {'code': 'L-1', 'name': 'n'}}}
    assert endpoint_keys(legacy) == {'base'} and has_endpoint([legacy], NOVA, None)
    us = [arn(r, NOVA) for r in ('us-east-1', 'us-east-2', 'us-west-2')]
    client = FakeBedrock(system=[system_profile(f"us.{NOVA}", us)],
                         application=[app_profile('legacy00001', 'l', [arn('us-east-1', NOVA)])])
    fetcher = InferenceProfileFetcher.for_region(client, [legacy], 'us-east-1')
    assert fetcher.list_application_profiles()[0]['sources'] == [NOVA]


def test_lone_arn_copy_of_a_model_newer_than_the_fm_list_is_a_base_copy():
    """A model missing from the fm-list may well be on demand: its lone in-region copy stays base."""
    us = [arn(r, NOVA) for r in ('us-east-1', 'us-east-2', 'us-west-2')]
    client = FakeBedrock(system=[system_profile(f"us.{NOVA}", us)],
                         application=[app_profile('newbase0001', 'n', [arn('us-east-1', NOVA)])])
    fetcher = InferenceProfileFetcher.for_region(client, [{'model_id': HAIKU, 'endpoints': {'base': {}}}], 'us-east-1')
    assert fetcher.list_application_profiles()[0]['sources'] == [NOVA]


def test_lone_arn_copy_of_a_profile_only_model_goes_to_a_listed_profile():
    """Tokyo lists jp.X and apac.X, X has no on-demand endpoint: a copy shrunk to Tokyo is jp, not base."""
    apac = JP_ARNS + [arn('ap-southeast-1', HAIKU)]
    client = FakeBedrock(system=[system_profile(f"jp.{HAIKU}", JP_ARNS), system_profile(f"apac.{HAIKU}", apac)],
                         application=[app_profile('jpshrunk001', 'j', [arn('ap-northeast-1', HAIKU)])])
    fetcher = InferenceProfileFetcher(client, on_demand_models=[], region='ap-northeast-1')
    app = fetcher.list_application_profiles()[0]
    assert (app['profile_prefix'], app['sources']) == ('jp', [f"jp.{HAIKU}"])


def test_single_region_system_profile_and_base_copy_are_both_candidates():
    one = [arn('ap-southeast-2', HAIKU)]
    fetcher = InferenceProfileFetcher(FakeBedrock(system=[system_profile(f"au.{HAIKU}", one)],
                                                  application=[app_profile('single00001', 's', one)]))
    app = fetcher.list_application_profiles()[0]
    assert app['sources'] == [f"au.{HAIKU}", HAIKU] and app['profile_prefix'] == 'au'
    assert fetcher.find_profiles(HAIKU, 'au')[0][1:] == ['single00001']
    assert fetcher.find_profiles(HAIKU, None)[0][1:] == ['single00001']


def test_prefixes_come_from_the_mapping_file(tmp_path, monkeypatch):
    from bedrock_usage_analyzer.aws import bedrock
    monkeypatch.setattr(bedrock, '_load_prefix_mapping', lambda: [
        {'prefix': 'zz', 'is_regional': True}, {'prefix': 'global', 'is_regional': False}])
    assert bedrock.get_regional_profile_prefixes() == ['zz']
    assert bedrock.get_profile_prefixes() == frozenset({'zz', 'global'})
    monkeypatch.setattr(bedrock, '_load_prefix_mapping', lambda: [])
    assert 'us-gov' in bedrock.get_regional_profile_prefixes() and 'global' not in bedrock.get_regional_profile_prefixes()


def test_copy_of_unlisted_country_profile_goes_to_listed_endpoint():
    """No au.* profile in the region: the source is the listed apac.* (an existing endpoint)."""
    apac_wide = [arn(r, HAIKU) for r in ('ap-northeast-1', 'ap-northeast-2', 'ap-south-1',
                                          'ap-southeast-1', 'ap-southeast-2', 'ap-southeast-4')]
    fetcher = InferenceProfileFetcher(FakeBedrock(system=[system_profile(f"apac.{HAIKU}", apac_wide)],
                                                  application=[app_profile('auonly00001', 'a', AU_ARNS)]))
    assert fetcher.list_application_profiles()[0]['source'] == f"apac.{HAIKU}"


def test_public_is_system_profile(sydney_bedrock):
    fetcher = InferenceProfileFetcher(sydney_bedrock)
    assert fetcher.is_system_profile(f"au.{HAIKU}") and not fetcher.is_system_profile('nope')


def test_unlisted_country_copy_also_shows_under_closest_endpoint():
    tokyo_osaka_plus = [arn(r, HAIKU) for r in ('ap-northeast-1', 'ap-northeast-2', 'ap-northeast-3', 'ap-southeast-1')]
    fetcher = InferenceProfileFetcher(FakeBedrock(system=[system_profile(f"apac.{HAIKU}", tokyo_osaka_plus)],
                                                  application=[app_profile('jponly00001', 'j', JP_ARNS)]))
    app = fetcher.list_application_profiles()[0]
    assert app['sources'] == [f"apac.{HAIKU}"]                                  # jp.* does not exist here
    assert fetcher.find_profiles(HAIKU, 'apac')[0][1:] == ['jponly00001']      # not lost from apac


def test_copy_goes_to_narrowest_listed_profile_containing_it():
    """A routing set grown since the copy: the narrowest listed superset is the source."""
    jp_now = [arn(r, HAIKU) for r in ('ap-northeast-1', 'ap-northeast-3', 'ap-northeast-9')]
    apac_now = jp_now + [arn('ap-southeast-1', HAIKU), arn('ap-south-1', HAIKU)]
    fetcher = InferenceProfileFetcher(FakeBedrock(
        system=[system_profile(f"jp.{HAIKU}", jp_now), system_profile(f"apac.{HAIKU}", apac_now)],
        application=[app_profile('jpold000001', 'j', JP_ARNS)]))
    assert fetcher.list_application_profiles()[0]['sources'] == [f"jp.{HAIKU}"]


def test_kr_copy_resolves_without_a_country_table():
    kr = [arn('ap-northeast-2', HAIKU), arn('ap-northeast-9', HAIKU)]
    kr_now = kr + [arn('ap-northeast-8', HAIKU)]
    apac = kr_now + [arn('ap-southeast-1', HAIKU), arn('ap-south-1', HAIKU)]
    fetcher = InferenceProfileFetcher(FakeBedrock(
        system=[system_profile(f"kr.{HAIKU}", kr_now), system_profile(f"apac.{HAIKU}", apac)],
        application=[app_profile('kr000000001', 'k', kr)]))
    app = fetcher.list_application_profiles()[0]
    assert (app['source'], app['profile_prefix']) == (f"kr.{HAIKU}", 'kr')


def test_regional_copy_is_never_attributed_to_global():
    """us.* dropped a region since the copy; global.* contains all of them, but is not the source."""
    us4 = [arn(r, HAIKU) for r in ('us-east-1', 'us-east-2', 'us-west-1', 'us-west-2')]
    us_now = [a for a in us4 if 'us-west-1' not in a]
    glob = us4 + [arn('eu-west-1', HAIKU), f"arn:aws:bedrock:::foundation-model/{HAIKU}"]
    fetcher = InferenceProfileFetcher(FakeBedrock(
        system=[system_profile(f"us.{HAIKU}", us_now), system_profile(f"global.{HAIKU}", glob)],
        application=[app_profile('usold000001', 'u', us4)]))
    assert fetcher.list_application_profiles()[0]['sources'] == [f"us.{HAIKU}"]


def test_country_copy_after_country_set_changed_stays_with_country():
    au_old = [arn('ap-southeast-2', HAIKU), arn('ap-southeast-4', HAIKU)]
    au_now = [arn('ap-southeast-2', HAIKU), arn('ap-southeast-6', HAIKU)]
    apac = au_old + [arn(r, HAIKU) for r in ('ap-northeast-1', 'ap-south-1', 'ap-southeast-1')]
    fetcher = InferenceProfileFetcher(FakeBedrock(
        system=[system_profile(f"au.{HAIKU}", au_now), system_profile(f"apac.{HAIKU}", apac)],
        application=[app_profile('auold000001', 'a', au_old)]))
    assert fetcher.list_application_profiles()[0]['sources'] == [f"au.{HAIKU}"]


def test_single_arn_copy_of_model_with_on_demand_endpoint_is_base_first():
    one = [arn('ap-southeast-2', HAIKU)]
    fetcher = InferenceProfileFetcher(FakeBedrock(system=[system_profile(f"au.{HAIKU}", one)],
                                                  application=[app_profile('single00001', 's', one)]),
                                      on_demand_models=[HAIKU])
    app = fetcher.list_application_profiles()[0]
    assert app['sources'] == [HAIKU, f"au.{HAIKU}"] and app['profile_prefix'] is None


def test_new_country_geography_is_learned_when_its_set_drifts():
    """kr.* is not in the defaults; it is a country because its regions sit inside apac.*."""
    kr_now = [arn('ap-northeast-2', HAIKU), arn('ap-northeast-8', HAIKU)]
    old_copy = [arn('ap-northeast-2', HAIKU), arn('ap-northeast-9', HAIKU)]
    apac = kr_now + [arn('ap-northeast-9', HAIKU), arn('ap-southeast-1', HAIKU)]
    fetcher = InferenceProfileFetcher(FakeBedrock(
        system=[system_profile(f"kr.{HAIKU}", kr_now), system_profile(f"apac.{HAIKU}", apac),
                system_profile(f"kr.{NOVA}", [arn('ap-northeast-2', NOVA), arn('ap-northeast-9', NOVA)]),
                system_profile(f"apac.{NOVA}", [arn('ap-northeast-2', NOVA), arn('ap-northeast-9', NOVA),
                                                arn('ap-southeast-1', NOVA)])],
        application=[app_profile('krold000001', 'k', old_copy)]))
    assert fetcher.list_application_profiles()[0]['sources'] == [f"kr.{HAIKU}"]


def test_failed_system_listing_is_not_retried_forever():
    calls = []

    class Down(FakeBedrock):
        def list_inference_profiles(self, **kwargs):
            calls.append(kwargs.get('typeEquals'))
            raise RuntimeError('Could not connect to the endpoint URL')

    fetcher = InferenceProfileFetcher(Down())
    for _ in range(4):
        with pytest.raises(RuntimeError):
            fetcher.is_system_profile('x')
    assert len(calls) == 2                                   # MAX_LISTING_ATTEMPTS, then cached


def test_learned_country_guess_keeps_the_real_model_id():
    """No system profile of this model is listed; the kr.* guess still splits into prefix and model."""
    other = 'anthropic.claude-sonnet-x-v1:0'
    kr = [arn('ap-northeast-2', HAIKU), arn('ap-northeast-9', HAIKU)]
    apac = kr + [arn('ap-southeast-1', HAIKU)]
    fetcher = InferenceProfileFetcher(FakeBedrock(
        system=[system_profile(f"kr.{HAIKU}", kr), system_profile(f"apac.{HAIKU}", apac)],
        application=[app_profile('krsonnet001', 's', [arn('ap-northeast-2', other), arn('ap-northeast-9', other)])]))
    app = fetcher.list_application_profiles()[0]
    assert (app['model_id'], app['profile_prefix']) == (other, 'kr')


def test_system_listing_failing_once_is_retried_without_listing_applications_again():
    calls = []

    class Flaky(FakeBedrock):
        def list_inference_profiles(self, **kwargs):
            calls.append(kwargs.get('typeEquals'))
            if kwargs.get('typeEquals') == 'SYSTEM_DEFINED' and calls.count('SYSTEM_DEFINED') == 1:
                raise RuntimeError('ThrottlingException')
            return super().list_inference_profiles(**kwargs)

    fetcher = InferenceProfileFetcher(Flaky(application=[app_profile('auapp000009', 'a', AU_ARNS)],
                                            system=[system_profile(f"au.{HAIKU}", AU_ARNS)]))
    resolved = fetcher.list_application_profiles()           # system listing retried at once
    assert resolved[0]['sources'] == [f"au.{HAIKU}"]
    assert calls.count('APPLICATION') == 1                    # not listed again


def test_learned_country_ranks_before_a_regional_profile_with_the_same_set():
    se = [arn('eu-north-1', HAIKU), arn('eu-north-9', HAIKU)]
    eu = se + [arn('eu-west-1', HAIKU)]
    fetcher = InferenceProfileFetcher(FakeBedrock(
        system=[system_profile(f"se.{HAIKU}", se), system_profile(f"eu.{HAIKU}", eu),
                system_profile(f"eu.{NOVA}", [arn('eu-north-1', NOVA), arn('eu-north-9', NOVA)])],
        application=[app_profile('sesame00001', 's', [arn('eu-north-1', NOVA), arn('eu-north-9', NOVA)])]))
    fetcher.list_application_profiles()
    from bedrock_usage_analyzer.core.profile_fetcher import _specific_first
    assert _specific_first([f"eu.{NOVA}", f"se.{NOVA}"], fetcher._country_regions)[0] == f"se.{NOVA}"


def test_narrowest_geography_wins_when_learned_sets_overlap():
    fetcher = InferenceProfileFetcher(FakeBedrock())
    fetcher._country_regions = {'jp': {'ap-northeast-1', 'ap-northeast-3'},
                                'aa': {'ap-northeast-1', 'ap-northeast-3', 'ap-northeast-2'}}
    assert fetcher._country_of({'ap-northeast-1', 'ap-northeast-3'}) == 'jp'


def test_global_copy_is_never_credited_to_a_regional_profile():
    us = [arn(r, HAIKU) for r in ('us-east-1', 'us-east-2', 'us-west-2')]
    glob = us + [arn('eu-west-1', HAIKU), arn('ap-northeast-1', HAIKU), f"arn:aws:bedrock:::foundation-model/{HAIKU}"]
    fetcher = InferenceProfileFetcher(FakeBedrock(system=[system_profile(f"us.{HAIKU}", us)],
                                                  application=[app_profile('globalold01', 'g', glob)]))
    assert fetcher.list_application_profiles()[0]['sources'] == [f"global.{HAIKU}"]   # not us.*


def test_copy_of_a_region_less_arn_is_global_not_base():
    """A lone region-less ARN only routes globally, even when the model is on demand."""
    only_global = [f"arn:aws:bedrock:::foundation-model/{HAIKU}"]
    listed = InferenceProfileFetcher(FakeBedrock(system=[system_profile(f"global.{HAIKU}", only_global)]),
                                     on_demand_models=[HAIKU])
    assert listed.resolve_endpoints(only_global) == [('global', HAIKU)]
    unlisted = InferenceProfileFetcher(FakeBedrock(), on_demand_models=[HAIKU])
    assert unlisted.resolve_endpoints(only_global) == [('global', HAIKU)]


def test_region_group_follows_the_partition_table():
    assert region_group('eusc-de-east-1') == 'eusc'
    assert region_group('us-isob-east-1') == 'us-isob'
    assert region_group('cn-north-1') == 'cn'
    assert region_group('ap-southeast-2') == 'ap'


def test_malformed_system_profile_is_skipped_not_half_indexed():
    fetcher = InferenceProfileFetcher(FakeBedrock(
        system=[{'models': [{'modelArn': arn('us-east-1', HAIKU)}]}, system_profile(f"au.{HAIKU}", AU_ARNS)],
        application=[app_profile('auapp000010', 'a', AU_ARNS)]))
    assert fetcher.list_application_profiles()[0]['sources'] == [f"au.{HAIKU}"]


def test_lone_arn_of_another_region_is_not_a_base_copy():
    """A base copy routes to the model in its own region; Tokyo-only routing seen from Sydney is not one."""
    client = FakeBedrock(system=[system_profile(f"jp.{HAIKU}", JP_ARNS)])
    client.meta = type('Meta', (), {'region_name': 'ap-southeast-2'})()
    fetcher = InferenceProfileFetcher(client, on_demand_models=[HAIKU])
    assert fetcher.resolve_endpoints([arn('ap-northeast-1', HAIKU)]) == [('jp', HAIKU)]
    assert fetcher.resolve_endpoints([arn('ap-southeast-2', HAIKU)])[0] == (None, HAIKU)


def test_summary_with_null_models_is_skipped():
    fetcher = InferenceProfileFetcher(FakeBedrock(
        system=[{'inferenceProfileId': f"us.{HAIKU}", 'models': None}, system_profile(f"au.{HAIKU}", AU_ARNS)],
        application=[{'inferenceProfileId': 'nullmodels1', 'models': None}, app_profile('auapp000011', 'a', AU_ARNS)]))
    assert [p['id'] for p in fetcher.list_application_profiles()] == ['auapp000011']


@pytest.mark.parametrize('content', ['prefixes: [kr]\n', '- prefix: kr\n', 'prefixes:\n  - {description: x}\n'])
def test_malformed_user_prefix_mapping_falls_back_to_bundled(content, tmp_path, monkeypatch):
    from bedrock_usage_analyzer.aws import bedrock
    from bedrock_usage_analyzer.utils import paths
    monkeypatch.setattr(paths, 'get_user_data_dir', lambda: tmp_path)
    (tmp_path / 'prefix-mapping.yml').write_text(content)
    bedrock.load_prefix_mapping(refresh=True)
    try:
        assert 'us' in bedrock.get_profile_prefixes()
        assert split_profile_id(f"us.{HAIKU}") == (HAIKU, 'us')
    finally:
        monkeypatch.undo()
        bedrock.load_prefix_mapping(refresh=True)


def test_profile_map_tolerates_null_models():
    from bedrock_usage_analyzer.aws.bedrock import build_profile_map, discover_prefix_mapping
    profiles = [{'inferenceProfileId': f"us.{HAIKU}", 'type': 'SYSTEM_DEFINED', 'models': None},
                system_profile(f"au.{HAIKU}", AU_ARNS)]
    assert build_profile_map(profiles) == {HAIKU: ['au']}
    assert [d['prefix'] for d in discover_prefix_mapping('ap-southeast-2', profiles)] == ['au']


def test_copy_goes_to_the_only_listed_profile_even_when_it_routes_one_region():
    """ap-northeast-1 lists only apac.X (shrunk to Tokyo); a copy routing Tokyo + Osaka is apac, not jp."""
    fetcher = InferenceProfileFetcher(FakeBedrock(
        system=[system_profile(f"apac.{HAIKU}", [arn('ap-northeast-1', HAIKU)])],
        application=[app_profile('apacold0001', 'a', JP_ARNS)]))
    assert fetcher.list_application_profiles()[0]['sources'] == [f"apac.{HAIKU}"]


def test_country_copy_matches_a_listed_single_region_country_profile():
    """au.X now routes to Sydney only; an older copy still routing to Sydney + Melbourne is au, not apac."""
    apac = [arn(r, HAIKU) for r in ('ap-northeast-1', 'ap-southeast-1', 'ap-southeast-2', 'ap-southeast-4')]
    fetcher = InferenceProfileFetcher(FakeBedrock(
        system=[system_profile(f"au.{HAIKU}", [arn('ap-southeast-2', HAIKU)]), system_profile(f"apac.{HAIKU}", apac)],
        application=[app_profile('auold000001', 'a', AU_ARNS)]))
    app = fetcher.list_application_profiles()[0]
    assert (app['profile_prefix'], app['sources']) == ('au', [f"au.{HAIKU}"])


def test_lone_copy_of_single_region_profile_is_not_base_when_model_is_not_on_demand():
    """The fm-list says X has no on-demand endpoint: a copy of the one-region au.X is au only."""
    one = [arn('ap-southeast-2', HAIKU)]
    fetcher = InferenceProfileFetcher(FakeBedrock(system=[system_profile(f"au.{HAIKU}", one)],
                                                  application=[app_profile('single00002', 's', one)]),
                                      on_demand_models=[], region='ap-southeast-2')
    assert fetcher.list_application_profiles()[0]['sources'] == [f"au.{HAIKU}"]
    assert fetcher.find_profiles(HAIKU, None)[0][1:] == []


def test_failed_listing_names_the_listing_and_is_not_announced_again(caplog):
    class SystemDenied(FakeBedrock):
        def list_inference_profiles(self, maxResults=1000, typeEquals='SYSTEM_DEFINED', nextToken=None):
            if typeEquals == 'SYSTEM_DEFINED':
                from botocore.exceptions import ClientError
                raise ClientError({'Error': {'Code': 'AccessDeniedException', 'Message': 'no'}}, 'List')
            return super().list_inference_profiles(maxResults, typeEquals, nextToken)

    fetcher = InferenceProfileFetcher(SystemDenied(application=[app_profile('a0000000009', 'a', AU_ARNS)]))
    for _ in range(2):
        with pytest.raises(Exception):
            fetcher.list_application_profiles()
    assert fetcher.failed_listing() == 'system'
    denied = InferenceProfileFetcher(FakeBedrock())
    denied._listings['APPLICATION']['error'] = RuntimeError('denied')
    caplog.clear()
    with caplog.at_level('INFO'), pytest.raises(RuntimeError):
        denied.list_application_profiles()
    assert 'Listing application inference profiles' not in caplog.text
    assert denied.failed_listing() == 'application'


def test_legacy_entries_are_migrated_and_malformed_endpoints_have_none():
    from bedrock_usage_analyzer.utils.yaml_handler import endpoint_keys, endpoint_quotas, valid_models
    legacy = {'model_id': NOVA, 'inference_types': ['ON_DEMAND'], 'quotas': {'tpd': {'code': 'L-2', 'name': 'n'}}}
    malformed = {'model_id': HAIKU, 'inference_types': ['ON_DEMAND'], 'endpoints': 'TODO'}
    models = valid_models({'models': [legacy, malformed]})
    assert dict(endpoint_quotas(models[0])) == {'base': {'tpd': {'code': 'L-2', 'name': 'n'}}}
    assert 'quotas' not in models[0]
    assert endpoint_keys(models[1]) == set()
