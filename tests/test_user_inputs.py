# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI model parsing, region checks, and interactive selection flows."""

import builtins

import pytest

from bedrock_usage_analyzer.core import user_inputs as ui_module
from bedrock_usage_analyzer.core.profile_fetcher import InferenceProfileFetcher
from bedrock_usage_analyzer.core.user_inputs import (
    UserInputs,
    group_application_profiles,
    merge_application_configs,
    parse_selection,
)

from bedrock_usage_analyzer.utils.partition import REGION_PATTERN

from conftest import HAIKU, NOVA


def feed(monkeypatch, answers):
    answers = iter(answers)
    monkeypatch.setattr(builtins, 'input', lambda *_: next(answers))


@pytest.fixture
def inputs(sydney_bedrock):
    ui = UserInputs()
    ui.region = 'ap-southeast-2'
    ui.profile_fetcher = InferenceProfileFetcher(sydney_bedrock)
    return ui


def test_picked_region_with_bad_format_exits_with_a_message(caplog):
    with pytest.raises(SystemExit):
        UserInputs()._ensure_fm_list('US-EAST-1')
    assert 'Invalid region format' in caplog.text


def test_unreadable_user_fm_list_names_the_file(tmp_path, caplog):
    (tmp_path / 'data').mkdir(exist_ok=True)
    (tmp_path / 'data' / 'fm-list-us-east-1.yml').write_text('models: [unclosed\n')
    with pytest.raises(SystemExit):
        UserInputs()._ensure_fm_list('us-east-1')
    assert 'fm-list-us-east-1.yml' in caplog.text and 'bua refresh fm-list us-east-1' in caplog.text


def test_non_utf8_user_fm_list_names_the_file(tmp_path, caplog):
    (tmp_path / 'data').mkdir(exist_ok=True)
    (tmp_path / 'data' / 'fm-list-us-east-1.yml').write_bytes('models:\n- model_id: caf\xe9\n'.encode('latin-1'))
    with pytest.raises(SystemExit):
        UserInputs()._ensure_fm_list('us-east-1')
    assert 'fm-list-us-east-1.yml' in caplog.text and 'bua refresh fm-list us-east-1' in caplog.text


@pytest.mark.parametrize('region', ['us-west-2', 'us-gov-west-1', 'cn-northwest-1', 'ap-southeast-7',
                                    'eusc-de-east-1', 'us-isob-east-1'])
def test_region_pattern_accepts(region):
    assert REGION_PATTERN.match(region)


@pytest.mark.parametrize('region', ['', 'us-west', '../etc', 'us-west-2/../x', 'US-WEST-2', 'us_west_2',
                                    'us-west-2 ', 'a-b-1'])
def test_region_pattern_rejects(region):
    assert not REGION_PATTERN.match(region)


@pytest.mark.parametrize('value,expected', [
    ('amazon.nova-premier-v1:0', {'model_id': 'amazon.nova-premier-v1:0', 'profile_prefix': None}),
    ('us.amazon.nova-premier-v1:0', {'model_id': 'amazon.nova-premier-v1:0', 'profile_prefix': 'us'}),
    ('ca.amazon.nova-lite-v1:0', {'model_id': 'amazon.nova-lite-v1:0', 'profile_prefix': 'ca'}),
    ('us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0',
     {'model_id': 'anthropic.claude-sonnet-4-5-20250929-v1:0', 'profile_prefix': 'us-gov'}),
    ('deepseek.v3.2', {'model_id': 'deepseek.v3.2', 'profile_prefix': None}),
    (' global.anthropic.claude-haiku-4-5-20251001-v1:0 ',
     {'model_id': HAIKU, 'profile_prefix': 'global'}),
    ('arn:aws:bedrock:ap-southeast-2:111122223333:inference-profile/au.' + HAIKU,
     {'model_id': HAIKU, 'profile_prefix': 'au'}),
    ('arn:aws:bedrock:ap-southeast-2::foundation-model/' + NOVA,
     {'model_id': NOVA, 'profile_prefix': None}),
])
def test_parse_model_id(inputs, value, expected):
    assert inputs._parse_model_id(value) == expected


def test_parse_application_profile_by_id_and_arn(inputs):
    expected = {'model_id': HAIKU, 'profile_prefix': 'au', 'application_profile_ids': ['auapp000001']}
    assert inputs._parse_model_id('auapp000001') == expected
    arn_value = 'arn:aws:bedrock:ap-southeast-2:111122223333:application-inference-profile/auapp000001'
    assert inputs._parse_model_id(arn_value) == expected


def test_parse_unknown_application_profile_exits(inputs):
    with pytest.raises(SystemExit):
        inputs._parse_model_id('doesnotexist')


def test_parse_arn_from_other_region_exits(inputs):
    with pytest.raises(SystemExit):
        inputs._parse_model_id('arn:aws:bedrock:us-east-1:1:application-inference-profile/auapp000001')


def test_parse_unsupported_arn_exits(inputs):
    with pytest.raises(SystemExit):
        inputs._parse_model_id('arn:aws:bedrock:ap-southeast-2:1:custom-model/x')


@pytest.mark.parametrize('text,count,expected', [
    ('1', 3, [0]), ('1,3', 3, [0, 2]), ('2-3', 3, [1, 2]), ('3, 1-2', 3, [0, 1, 2]),
    ('all', 2, [0, 1]), (' ALL ', 2, [0, 1]), ('1,1', 3, [0]),
])
def test_parse_selection(text, count, expected):
    assert parse_selection(text, count) == expected


@pytest.mark.parametrize('text', ['', '0', '4', '2-1', 'a', '1-', ','])
def test_parse_selection_rejects(text):
    with pytest.raises(ValueError):
        parse_selection(text, 3)


def test_group_application_profiles_by_source():
    apps = [
        {'id': 'a', 'model_id': 'm', 'profile_prefix': 'au'},
        {'id': 'b', 'model_id': 'm', 'profile_prefix': 'au'},
        {'id': 'c', 'model_id': 'm', 'profile_prefix': None},
        {'id': 'a', 'model_id': 'm', 'profile_prefix': 'au'},
    ]
    assert group_application_profiles(apps) == [
        {'model_id': 'm', 'profile_prefix': 'au', 'application_profile_ids': ['a', 'b']},
        {'model_id': 'm', 'profile_prefix': None, 'application_profile_ids': ['c']},
    ]


def test_profiles_of_unknown_source_are_not_aggregated():
    apps = [{'id': 'a', 'model_id': 'm', 'profile_prefix': 'unknown'},
            {'id': 'b', 'model_id': 'm', 'profile_prefix': 'unknown'}]
    assert [c['application_profile_ids'] for c in group_application_profiles(apps)] == [['a'], ['b']]
    configs = [{'model_id': 'm', 'profile_prefix': 'unknown', 'application_profile_ids': [i]} for i in 'ab']
    assert len(merge_application_configs(configs)) == 2


def test_interactive_application_profile_selection(inputs, monkeypatch):
    # "Specific application inference profiles", then pick #1 and #3
    feed(monkeypatch, ['2', 'x', '1,3'])
    configs = inputs._select_targets('ap-southeast-2')
    assert configs == [
        {'model_id': HAIKU, 'profile_prefix': 'au', 'application_profile_ids': ['auapp000001']},
        {'model_id': HAIKU, 'profile_prefix': 'jp', 'application_profile_ids': ['jpapp000001']},
    ]


def test_interactive_foundation_model_selection(inputs, monkeypatch):
    fm_list = [{'model_id': HAIKU, 'provider': 'Anthropic',
                'endpoints': {'au': {}, 'global': {}}},
               {'model_id': NOVA, 'provider': 'Amazon', 'endpoints': {'base': {}, 'apac': {}}}]
    monkeypatch.setattr(inputs, '_load_fm_list', lambda region: fm_list)
    # foundation model mode, provider Anthropic (2nd after sorting), model 1, prefix 'au' (1st)
    feed(monkeypatch, ['1', '2', '1', '1'])
    assert inputs._select_targets('ap-southeast-2') == [{'model_id': HAIKU, 'profile_prefix': 'au'}]
    # Amazon, base model is the last choice
    feed(monkeypatch, ['1', '1', '1', '2'])
    assert inputs._select_targets('ap-southeast-2') == [{'model_id': NOVA, 'profile_prefix': None}]


def test_model_without_endpoints_allows_manual_entry(inputs, monkeypatch):
    monkeypatch.setattr(inputs, '_load_fm_list',
                        lambda region: [{'model_id': 'x.y-v1:0', 'provider': 'X', 'endpoints': {}}])
    feed(monkeypatch, ['1', '1', 'us.x.y-v1:0'])
    assert inputs._select_model('ap-southeast-2') == {'model_id': 'x.y-v1:0', 'profile_prefix': 'us'}
    feed(monkeypatch, ['1', '1', ''])
    assert inputs._select_model('ap-southeast-2') is None


def test_no_application_profiles_skips_mode_question(monkeypatch):
    class Empty:
        def list_application_profiles(self):
            return []
    ui = UserInputs()
    ui.region = 'us-east-1'
    ui.profile_fetcher = Empty()
    monkeypatch.setattr(ui, '_load_fm_list',
                        lambda region: [{'model_id': NOVA, 'provider': 'Amazon', 'endpoints': {'base': {}}}])
    feed(monkeypatch, ['1', '1', '1'])
    assert ui._select_targets('us-east-1') == [{'model_id': NOVA, 'profile_prefix': None}]


def test_partition_mismatch_exits():
    ui = UserInputs()
    ui.partition = 'aws'
    with pytest.raises(SystemExit):
        ui._check_region_partition('us-gov-west-1')
    ui._check_region_partition('us-east-1')


def test_invalid_region_exits():
    with pytest.raises(SystemExit):
        UserInputs._validate_region('not a region')


def test_collect_scripted_with_repeated_models(inputs, monkeypatch):
    monkeypatch.setattr(ui_module, 'get_caller_identity',
                        lambda region=None, **_: {'Account': '111122223333', 'Arn': 'arn:aws:iam::1:user/a',
                                             'Partition': 'aws'})
    ui = inputs
    ui.collect(region='ap-southeast-2', model_id=['auapp000001', 'au.' + HAIKU, 'au.' + HAIKU],
               granularity_config={'1hour': 60, '1day': 60, '7days': 60, '14days': 60, '30days': 60},
               skip_confirm=True)
    assert ui.account == '111122223333'
    assert ui.models == [
        {'model_id': HAIKU, 'profile_prefix': 'au', 'application_profile_ids': ['auapp000001']},
        {'model_id': HAIKU, 'profile_prefix': 'au'},
    ]


def test_collect_rejects_region_in_other_partition(inputs, monkeypatch):
    def identity(region=None, **_):
        if region == 'us-gov-west-1':
            raise RuntimeError('InvalidClientTokenId')
        return {'Account': '1', 'Arn': 'arn:aws:iam::1:user/a', 'Partition': 'aws'}
    monkeypatch.setattr(ui_module, 'get_caller_identity', identity)
    with pytest.raises(SystemExit):
        inputs.collect(region='us-gov-west-1', model_id='x.y', skip_confirm=True)


def test_collect_checks_partition_when_sts_answers_for_another_partition(inputs, monkeypatch):
    """A custom STS endpoint answers for us-gov-west-1 with commercial credentials: still stopped."""
    monkeypatch.setattr(ui_module, 'get_caller_identity',
                        lambda region=None, **_: {'Account': '1', 'Arn': 'arn:aws:iam::1:user/a', 'Partition': 'aws'})
    monkeypatch.setattr(inputs, '_ensure_fm_list', lambda region: pytest.fail('must stop before the fm-list'))
    monkeypatch.setattr('builtins.input', lambda *_: pytest.fail('must stop before the Continue? prompt'))
    with pytest.raises(SystemExit):
        inputs.collect(region='us-gov-west-1', model_id='x.y')


def test_account_failure_exits_with_hint(monkeypatch, caplog):
    def fail(region=None, **_):
        raise RuntimeError('InvalidClientTokenId: The security token included in the request is invalid')
    monkeypatch.setattr(ui_module, 'get_caller_identity', fail)
    with pytest.raises(SystemExit):
        UserInputs()._get_current_account('us-gov-west-1')
    assert 'AWS GovCloud (US) uses separate accounts' in caplog.text


def test_commercial_credentials_with_govcloud_region_explain_mismatch(monkeypatch, caplog):
    def identity(region=None, **_):
        if region == 'us-gov-west-1':
            raise RuntimeError('InvalidClientTokenId: The security token included in the request is invalid')
        return {'Account': '1', 'Arn': 'arn:aws:iam::1:user/a', 'Partition': 'aws'}
    monkeypatch.setattr(ui_module, 'get_caller_identity', identity)
    with pytest.raises(SystemExit):
        UserInputs()._get_current_account('us-gov-west-1')
    assert 'credentials are for AWS Commercial' in caplog.text


def test_select_region_shows_only_credential_partition(monkeypatch):
    ui = UserInputs()
    ui.partition = 'aws-us-gov'          # set by the account check, which always runs first
    feed(monkeypatch, ['2'])
    assert ui._select_region() == 'us-gov-west-1'


def test_region_label():
    assert UserInputs._region_label('us-gov-west-1') == 'us-gov-west-1 (AWS GovCloud (US-West))'
    assert UserInputs._region_label('xx-new-1') == 'xx-new-1'


def test_ensure_fm_list(tmp_path):
    ui = UserInputs()
    ui._ensure_fm_list('us-gov-west-1')          # bundled
    with pytest.raises(SystemExit):
        ui._ensure_fm_list('xx-nowhere-1')        # missing
    with pytest.raises(SystemExit):                # rejected before it is used in a file name
        ui._ensure_fm_list('../../etc/passwd')


def test_output_dir_choice(monkeypatch, tmp_path):
    feed(monkeypatch, ['9', '3', '', '3', '~/reports'])
    path = UserInputs().select_output_dir()
    assert not path.startswith('~') and path.endswith('reports')


def test_repeated_application_profile_ids_are_aggregated(inputs, monkeypatch):
    """-m id1 -m id2 of the same endpoint gives one report, like interactive '1-2'."""
    from conftest import AU_ARNS, app_profile
    inputs.profile_fetcher.bedrock_client.application.append(app_profile('auapp000002', 'team-f-au', AU_ARNS))
    inputs.profile_fetcher._app_profiles = None
    monkeypatch.setattr(ui_module, 'get_caller_identity',
                        lambda region=None, **_: {'Account': '1', 'Arn': 'arn:aws:iam::1:user/a', 'Partition': 'aws'})
    inputs.collect(region='ap-southeast-2', model_id=['auapp000001', 'auapp000002', 'jpapp000001'],
                   granularity_config={p: 60 for p in ['1hour', '1day', '7days', '14days', '30days']},
                   skip_confirm=True)
    assert inputs.models == [
        {'model_id': HAIKU, 'profile_prefix': 'au', 'application_profile_ids': ['auapp000001', 'auapp000002']},
        {'model_id': HAIKU, 'profile_prefix': 'jp', 'application_profile_ids': ['jpapp000001']},
    ]


def test_closed_stdin_exits_cleanly(monkeypatch, caplog):
    import sys as _sys
    import bedrock_usage_analyzer.__main__ as cli

    def eof(*_):
        raise EOFError
    monkeypatch.setattr(builtins, 'input', eof)
    monkeypatch.setattr(_sys, 'argv', ['bua', 'analyze'])
    monkeypatch.setattr(ui_module, 'get_caller_identity',
                        lambda region=None, **_: {'Account': '1', 'Arn': 'arn:aws:iam::1:user/a', 'Partition': 'aws'})
    with pytest.raises(SystemExit):
        cli.main()
    assert 'Input ended before all prompts were answered' in caplog.text


def test_govcloud_credentials_with_commercial_region_explain_mismatch(monkeypatch, caplog):
    """No region configured: commercial STS rejects GovCloud creds, GovCloud STS identifies them."""
    def identity(region=None, **_):
        if region == 'us-gov-west-1':
            return {'Account': '1', 'Arn': 'arn:aws-us-gov:iam::1:user/a', 'Partition': 'aws-us-gov'}
        raise RuntimeError('InvalidClientTokenId')
    monkeypatch.setattr(ui_module, 'get_caller_identity', identity)
    with pytest.raises(SystemExit):
        UserInputs()._get_current_account('us-west-2')
    assert 'but the credentials are for AWS GovCloud (US)' in caplog.text


def test_network_errors_do_not_trigger_partition_probes(monkeypatch):
    calls = []

    def identity(region=None, **_):
        calls.append(region)
        raise RuntimeError('Could not connect to the endpoint URL')
    monkeypatch.setattr(ui_module, 'get_caller_identity', identity)
    with pytest.raises(SystemExit):
        UserInputs()._get_current_account('us-west-2')
    # Regional STS first; a network error triggers neither the home-region retry nor probes
    assert calls == ['us-west-2']


def test_mismatch_uses_configured_region_when_no_flag(monkeypatch, caplog):
    monkeypatch.setenv('AWS_REGION', 'us-west-2')

    def identity(region=None, **_):
        if region == 'us-gov-west-1':
            return {'Account': '1', 'Arn': 'arn:aws-us-gov:iam::1:user/a', 'Partition': 'aws-us-gov'}
        raise RuntimeError('InvalidClientTokenId')
    monkeypatch.setattr(ui_module, 'get_caller_identity', identity)
    # Only a default region: the run continues with the credentials' partition (the region
    # picker then lists GovCloud regions)
    inputs = UserInputs()
    assert inputs._get_current_account(None) == '1'
    assert inputs.partition == 'aws-us-gov'
    assert 'The credentials are for AWS GovCloud (US), but us-west-2 is in AWS Commercial' in caplog.text


def test_mismatch_with_region_flag_exits(monkeypatch, caplog):
    def identity(region=None, **_):
        if region == 'us-gov-west-1':
            return {'Account': '1', 'Arn': 'arn:aws-us-gov:iam::1:user/a', 'Partition': 'aws-us-gov'}
        raise RuntimeError('InvalidClientTokenId')
    monkeypatch.setattr(ui_module, 'get_caller_identity', identity)
    with pytest.raises(SystemExit):
        UserInputs()._get_current_account('us-west-2')
    assert 'but the credentials are for AWS GovCloud (US)' in caplog.text


def test_application_profile_name_with_dots(inputs):
    from conftest import AU_ARNS, app_profile
    inputs.profile_fetcher.bedrock_client.application.append(app_profile('dotted00001', 'team.prod:haiku', AU_ARNS))
    inputs.profile_fetcher._app_profiles = None
    assert inputs._parse_model_id('team.prod:haiku') == {
        'model_id': HAIKU, 'profile_prefix': 'au', 'application_profile_ids': ['dotted00001']}


def test_application_profile_id_lookup_bug_is_raised(inputs, monkeypatch):
    monkeypatch.setattr(inputs.profile_fetcher, 'resolve_application_profile', lambda identifier: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        inputs._application_profile_config('appprofile01')


def test_menu_marks_a_base_copy_of_a_model_no_longer_on_demand(inputs, monkeypatch, capsys):
    monkeypatch.setattr(inputs, '_load_fm_list', lambda region: [{'model_id': HAIKU, 'endpoints': {'global': {}}}])
    monkeypatch.setattr('builtins.input', lambda *_: '1')
    app = {'id': 'base0000001', 'name': 'b', 'model_id': HAIKU, 'profile_prefix': None, 'source': HAIKU,
           'sources': [HAIKU], 'arn': 'arn', 'status': 'ACTIVE', 'tags': {}}
    inputs._select_application_profiles([app])
    assert 'not offered in ap-southeast-2 any more' in capsys.readouterr().out


def test_dotted_name_of_a_custom_model_copy_stops_with_the_reason(inputs, caplog):
    from conftest import app_profile
    inputs.profile_fetcher.bedrock_client.application.append(
        app_profile('custom00002', 'team.custom.v1', ['arn:aws:bedrock:ap-southeast-2:1:custom-model/x']))
    inputs.profile_fetcher._app_profiles = None
    with pytest.raises(SystemExit):
        inputs._parse_model_id('team.custom.v1')
    assert 'routes to no foundation model' in caplog.text


def test_china_credentials_are_identified(monkeypatch, caplog):
    def identity(region=None, **_):
        if region == 'cn-north-1':
            return {'Account': '1', 'Arn': 'arn:aws-cn:iam::1:user/a', 'Partition': 'aws-cn'}
        raise RuntimeError('InvalidClientTokenId')
    monkeypatch.setattr(ui_module, 'get_caller_identity', identity)
    with pytest.raises(SystemExit):
        UserInputs()._get_current_account('us-east-1')
    assert 'but the credentials are for AWS China' in caplog.text


def test_manual_entry_typo_skips_model_instead_of_exiting(inputs, monkeypatch):
    monkeypatch.setattr(inputs, '_load_fm_list',
                        lambda region: [{'model_id': 'x.y-v1:0', 'provider': 'X', 'endpoints': {}}])
    feed(monkeypatch, ['1', '1', 'claude'])
    assert inputs._select_model('ap-southeast-2') is None


def test_system_profile_id_beats_application_profile_with_same_name(inputs, monkeypatch):
    from conftest import AU_ARNS, app_profile
    inputs.profile_fetcher.bedrock_client.application.append(app_profile('samename001', 'au.' + HAIKU, AU_ARNS))
    inputs.profile_fetcher._app_profiles = None
    assert inputs._parse_model_id('au.' + HAIKU) == {'model_id': HAIKU, 'profile_prefix': 'au'}


def test_partition_probes_use_fast_clients(monkeypatch):
    seen = []

    def identity(region=None, probe=False):
        seen.append((region, probe))
        raise RuntimeError('InvalidClientTokenId')
    monkeypatch.setattr(ui_module, 'get_caller_identity', identity)
    with pytest.raises(SystemExit):
        UserInputs()._get_current_account('us-east-1')
    assert seen[0] == ('us-east-1', False) and all(p for _, p in seen[1:]) and len(seen) == 3


def test_account_check_falls_back_to_partition_home_region(monkeypatch):
    """Regional STS first (VPC endpoints); a disabled opt-in region retries the home region."""
    seen = []

    def identity(region=None, **_):
        seen.append(region)
        if region == 'ap-east-1':
            raise RuntimeError('InvalidClientTokenId: The security token included in the request is invalid')
        return {'Account': '1', 'Arn': 'arn:aws:iam::1:user/a', 'Partition': 'aws'}

    monkeypatch.setattr(ui_module, 'get_caller_identity', identity)
    UserInputs()._get_current_account('ap-east-1')
    UserInputs()._get_current_account('ap-southeast-1')
    assert seen == ['ap-east-1', 'us-east-1', 'ap-southeast-1']

def test_interactive_rounds_merge_profiles_of_one_endpoint(inputs, monkeypatch):
    from conftest import AU_ARNS, app_profile
    inputs.profile_fetcher.bedrock_client.application.append(app_profile('auapp000002', 'team-f-au', AU_ARNS))
    inputs.profile_fetcher._app_profiles = None
    monkeypatch.setattr(ui_module, 'get_caller_identity',
                        lambda region=None, **_: {'Account': '1', 'Arn': 'arn:aws:iam::1:user/a', 'Partition': 'aws'})
    # round 1: profile #1 (auapp000001); round 2: profile #6 (auapp000002)
    feed(monkeypatch, ['2', '1', 'y', '2', '6', 'n'])
    inputs.collect(region='ap-southeast-2', granularity_config={p: 60 for p in
                   ['1hour', '1day', '7days', '14days', '30days']}, skip_confirm=True)
    assert inputs.models == [{'model_id': HAIKU, 'profile_prefix': 'au',
                              'application_profile_ids': ['auapp000001', 'auapp000002']}]


def test_fm_list_is_parsed_once(inputs, monkeypatch):
    calls = []
    real = ui_module.load_fm_list
    monkeypatch.setattr(ui_module, 'load_fm_list', lambda region: calls.append(region) or real(region))
    inputs._load_fm_list('us-east-1')
    inputs._load_fm_list('us-east-1')
    assert len(calls) == 1


def test_manual_entry_listing_error_skips_model(inputs, monkeypatch):
    monkeypatch.setattr(inputs, '_load_fm_list',
                        lambda region: [{'model_id': 'x.y-v1:0', 'provider': 'X', 'endpoints': {}}])
    def denied(identifier):
        raise RuntimeError('ThrottlingException')
    monkeypatch.setattr(inputs.profile_fetcher, 'resolve_application_profile', denied)
    feed(monkeypatch, ['1', '1', 'claude-haiku'])
    assert inputs._select_model('ap-southeast-2') is None


def test_cli_accepts_system_profile_with_new_prefix(inputs):
    from conftest import system_profile, arn
    inputs.profile_fetcher.bedrock_client.system.append(
        system_profile(f"kr.{HAIKU}", [arn('ap-northeast-2', HAIKU), arn('ap-northeast-9', HAIKU)]))
    inputs.profile_fetcher = type(inputs.profile_fetcher)(inputs.profile_fetcher.bedrock_client)
    assert inputs._parse_model_id(f"kr.{HAIKU}") == {'model_id': HAIKU, 'profile_prefix': 'kr'}
    # Base model IDs with two dots stay base models
    assert inputs._parse_model_id('deepseek.v3.2') == {'model_id': 'deepseek.v3.2', 'profile_prefix': None}


def test_unknown_dotted_model_warns(inputs, caplog):
    import logging
    caplog.set_level(logging.WARNING)
    assert inputs._parse_model_id('team.prd') == {'model_id': 'team.prd', 'profile_prefix': None}
    assert 'is not a model, inference profile or application inference profile known' in caplog.text


def test_listed_system_profile_newer_than_the_fm_list_is_not_doubted(inputs, caplog):
    import logging
    from conftest import arn, system_profile
    caplog.set_level(logging.INFO)
    new = 'vendor.brand-new-v1:0'
    inputs.profile_fetcher.bedrock_client.system.append(system_profile(f"au.{new}", [arn('ap-southeast-2', new)]))
    inputs.profile_fetcher._system_profiles = None
    assert inputs._parse_model_id(f"au.{new}") == {'model_id': new, 'profile_prefix': 'au'}
    assert 'is not a model, inference profile' not in caplog.text
    assert 'bua refresh fm-list ap-southeast-2' in caplog.text


def test_bare_id_of_profile_only_model_points_to_its_profiles(inputs, monkeypatch, caplog):
    import logging
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(inputs, '_load_fm_list', lambda region: [
        {'model_id': 'anthropic.claude-x-v1:0', 'endpoints': {'us': {}, 'global': {}}}])
    monkeypatch.setattr(inputs, '_find_application_profile', lambda value: None)
    monkeypatch.setattr(inputs, '_is_system_profile', lambda value: False)
    inputs._parse_model_id('anthropic.claude-x-v1:0')
    assert 'no on-demand endpoint' in caplog.text and 'global.anthropic.claude-x-v1:0' in caplog.text


def test_mismatch_found_without_any_region(monkeypatch, caplog):
    """GovCloud credentials, no --region and no AWS_REGION: found through the GovCloud probe."""
    import logging
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(ui_module, 'region_hint', lambda: None)

    def identity(region=None, **_):
        if region == 'us-gov-west-1':
            return {'Account': '1', 'Arn': 'arn:aws-us-gov:iam::1:user/a', 'Partition': 'aws-us-gov'}
        raise RuntimeError('InvalidClientTokenId')
    monkeypatch.setattr(ui_module, 'get_caller_identity', identity)
    inputs = UserInputs()
    assert inputs._get_current_account(None) == '1' and inputs.partition == 'aws-us-gov'
    assert 'the default STS endpoint' in caplog.text


def test_expired_token_is_not_probed_as_a_partition_mismatch(monkeypatch):
    asked = []

    def identity(region=None, probe=False, **_):
        asked.append((region, probe))
        raise RuntimeError('ExpiredToken: The security token included in the request is expired')
    monkeypatch.setattr(ui_module, 'get_caller_identity', identity)
    with pytest.raises(SystemExit):
        UserInputs()._get_current_account('us-east-1')
    assert not any(probe for _, probe in asked)


def test_zip_install_reads_bundled_fm_list_and_regions(monkeypatch, tmp_path):
    """No file path for bundled data (zip/egg): the lists are still found through resources."""
    from bedrock_usage_analyzer.utils import paths, yaml_handler
    monkeypatch.setattr(paths, 'get_bundled_file', lambda name: None)
    assert yaml_handler.load_fm_list('us-east-1')                      # read via load_bundled_yaml
    inputs = UserInputs()
    inputs._ensure_fm_list('us-east-1')                                # does not exit
    assert inputs._load_regions()


def test_missing_fm_list_still_exits_after_an_earlier_lookup(monkeypatch):
    inputs = UserInputs()
    inputs.region = 'xx-test-1'
    assert inputs._load_fm_list('xx-test-1') == []             # an earlier caller (no list)
    with pytest.raises(SystemExit):
        inputs._ensure_fm_list('xx-test-1')                   # the 'run bua refresh fm-list' exit


def test_picker_offers_the_base_endpoint_of_a_legacy_entry(monkeypatch):
    """No 'endpoints' but ON_DEMAND: the picker offers the base model instead of asking for an ID."""
    from bedrock_usage_analyzer.utils.yaml_handler import endpoint_keys
    keys = endpoint_keys({'model_id': 'x', 'inference_types': ['ON_DEMAND']})
    assert keys == {'base'}
    feed(monkeypatch, ['1'])
    assert UserInputs()._select_profile_prefix(keys, []) is None   # 'None (base model)', the only choice


def test_hand_edited_endpoints_value_is_no_endpoints():
    from bedrock_usage_analyzer.utils.yaml_handler import fm_endpoints, model_endpoints
    models = [{'model_id': 'x', 'endpoints': 'TODO'}, {'model_id': 'y', 'endpoints': ['us']}]
    assert fm_endpoints(models, 'x') == set() and fm_endpoints(models, 'y') == set()
    assert model_endpoints({'endpoints': {'us': None}}) == {'us': None}


def test_valid_models_turn_a_non_mapping_endpoints_value_into_none():
    from bedrock_usage_analyzer.core.profile_fetcher import InferenceProfileFetcher
    from bedrock_usage_analyzer.sync.quota_mapper import QuotaMapper
    from bedrock_usage_analyzer.utils.yaml_handler import valid_models
    models = valid_models({'models': [{'model_id': 'x', 'endpoints': 'database'}]})
    assert models[0]['endpoints'] == {}
    assert InferenceProfileFetcher.for_region(object(), [{'model_id': 'x', 'endpoints': 'database'}]).on_demand_models == set()
    mapper = QuotaMapper('us-east-1', 'm')
    assert mapper._get_endpoints_to_process({'model_id': 'x', 'endpoints': 'TODO'}) == []
    assert mapper._get_quota_mapping('us-east-1', 'x', 'x', 'T', []) is None   # unknown endpoint type
