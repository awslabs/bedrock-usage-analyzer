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
    monkeypatch.setattr(ui_module, 'regions_for_credentials',
                        lambda regions, region=None: (['us-gov-east-1', 'us-gov-west-1'], 'aws-us-gov'))
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
    with pytest.raises(ValueError):
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
    with pytest.raises(SystemExit):
        UserInputs()._get_current_account(None)
    assert 'but the credentials are for AWS GovCloud (US)' in caplog.text


def test_application_profile_name_with_dots(inputs):
    from conftest import AU_ARNS, app_profile
    inputs.profile_fetcher.bedrock_client.application.append(app_profile('dotted00001', 'team.prod:haiku', AU_ARNS))
    inputs.profile_fetcher._app_profiles = None
    assert inputs._parse_model_id('team.prod:haiku') == {
        'model_id': HAIKU, 'profile_prefix': 'au', 'application_profile_ids': ['dotted00001']}


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

def test_select_region_reuses_known_partition(monkeypatch):
    ui = UserInputs()
    ui.partition = 'aws-us-gov'
    monkeypatch.setattr(ui_module, 'regions_for_credentials',
                        lambda *a, **k: pytest.fail('must not call STS again'))
    feed(monkeypatch, ['1'])
    assert ui._select_region() == 'us-gov-east-1'


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
    real = ui_module.load_yaml
    monkeypatch.setattr(ui_module, 'load_yaml', lambda path: calls.append(path) or real(path))
    inputs._load_fm_list('us-east-1')
    inputs._load_fm_list('us-east-1')
    assert len(calls) == 1
