# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""On-demand custom model deployments: discovery, quota mapping rules and analysis targets."""

import pytest
from botocore.exceptions import ClientError

from bedrock_usage_analyzer.aws import bedrock
from bedrock_usage_analyzer.aws.custom_models import base_model_id, deployment_short_id, list_deployments, read_deployment
from bedrock_usage_analyzer.core.profile_fetcher import InferenceProfileFetcher
from bedrock_usage_analyzer.sync.quota_rules import mapping_conflict
from bedrock_usage_analyzer.utils.yaml_handler import invokable_endpoint_keys, load_yaml, profile_endpoints, save_yaml

from conftest import FakeBedrock

DEPLOYMENT = 'arn:aws:bedrock:us-east-1:111122223333:custom-model-deployment/dep0000001'
CUSTOM_MODEL = 'arn:aws:bedrock:us-east-1:111122223333:custom-model/amazon.nova-lite-v1:0:300k/cm00001'
BASE = 'amazon.nova-lite-v1:0:300k'
TPM = '(Model customization) Sum of on demand custom model deployment tokens per minute for Amazon Nova Lite'


def aws_error(code, operation):
    return ClientError({'Error': {'Code': code, 'Message': code}}, operation)


SUMMARY = {'customModelDeploymentArn': DEPLOYMENT, 'customModelDeploymentName': 'my-lite',
           'modelArn': CUSTOM_MODEL, 'status': 'Active'}


class FakeCustom(FakeBedrock):
    def __init__(self, deployments=(), bases=None, **kw):
        super().__init__(**kw)
        self.deployments = list(deployments)
        # custom model ARN -> its baseModelArn
        self.bases = bases or {CUSTOM_MODEL: f"arn:aws:bedrock:us-east-1::foundation-model/{BASE}"}
        self.calls = []

    def get_custom_model_deployment(self, customModelDeploymentIdentifier):
        self.calls.append('GetCustomModelDeployment')
        return {'customModelDeploymentArn': DEPLOYMENT, 'modelDeploymentName': 'my-lite',
                'modelArn': CUSTOM_MODEL, 'status': 'Active'}

    def get_custom_model(self, modelIdentifier):
        self.calls.append('GetCustomModel')
        return {'modelArn': modelIdentifier, 'baseModelArn': self.bases.get(modelIdentifier)}

    def list_custom_model_deployments(self, **kwargs):
        self.calls.append('ListCustomModelDeployments')
        return {'modelDeploymentSummaries': self.deployments}


def test_deployment_short_id():
    assert deployment_short_id(DEPLOYMENT) == 'dep0000001'


def test_a_deployment_is_read_like_a_listed_one():
    assert read_deployment(FakeCustom(), DEPLOYMENT) == {
        'arn': DEPLOYMENT, 'name': 'my-lite', 'status': 'Active', 'model_arn': CUSTOM_MODEL}
    assert list_deployments(FakeCustom([SUMMARY]))[0]['model_arn'] == CUSTOM_MODEL


def test_a_custom_model_resolves_to_its_base_model():
    assert base_model_id(FakeCustom(), CUSTOM_MODEL) == BASE


def test_a_model_fine_tuned_from_a_custom_model_resolves_to_the_foundation_model():
    parent = 'arn:aws:bedrock:us-east-1:111122223333:custom-model/amazon.nova-lite-v1:0:300k/cm00000'
    client = FakeCustom(bases={CUSTOM_MODEL: parent,
                               parent: f"arn:aws:bedrock:us-east-1::foundation-model/{BASE}"})
    assert base_model_id(client, CUSTOM_MODEL) == BASE


def test_a_deployed_foundation_model_needs_no_lookup():
    client = FakeCustom()
    assert base_model_id(client, f"arn:aws:bedrock:us-east-1::foundation-model/{BASE}") == BASE
    assert client.calls == []


def test_an_imported_model_has_no_base_model():
    assert base_model_id(FakeCustom(bases={CUSTOM_MODEL: None}), CUSTOM_MODEL) is None


def test_deployments_are_listed():
    assert list_deployments(FakeCustom([SUMMARY]))[0]['name'] == 'my-lite'


def test_custom_endpoint_has_its_own_quota_keyword_and_is_no_profile_prefix():
    assert bedrock.get_endpoint_quota_keywords()['custom'] == 'custom model deployment'
    assert 'custom' not in bedrock.get_profile_prefixes()
    models = [{'model_id': BASE, 'endpoints': {'custom': {'quotas': {}}, 'base': {'quotas': {}}}}]
    assert profile_endpoints(models, BASE) == []
    assert invokable_endpoint_keys(models[0]) == {'base'}


@pytest.mark.parametrize('endpoint,name,conflict', [
    ('custom', TPM, False),
    ('custom', 'On-demand model inference tokens per minute for Amazon Nova Lite', True),
    ('base', TPM, True),
    ('us', TPM, True),
])
def test_custom_deployment_quotas_belong_to_the_custom_endpoint_only(endpoint, name, conflict):
    assert bool(mapping_conflict(BASE, endpoint, name, {'us'})) is conflict


def test_fm_list_adds_the_custom_endpoint_for_customizable_models(monkeypatch, tmp_path):
    from bedrock_usage_analyzer.sync import fm_list
    (tmp_path / 'data').mkdir(exist_ok=True)
    monkeypatch.setattr(fm_list, 'discover_prefix_mapping', lambda region, profiles=None: [])
    monkeypatch.setattr(fm_list, 'list_system_profiles', lambda region: [])
    monkeypatch.setattr(fm_list, 'fetch_foundation_models', lambda region: [
        {'model_id': BASE, 'provider': 'Amazon', 'inference_types': ['PROVISIONED'], 'customizations': ['FINE_TUNING']},
        {'model_id': 'amazon.nova-pro-v1:0', 'provider': 'Amazon', 'inference_types': ['ON_DEMAND']}])
    fm_list.refresh_region('us-east-1')
    saved = {m['model_id']: m for m in load_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'))['models']}
    assert set(saved[BASE]['endpoints']) == {'custom'}
    assert set(saved['amazon.nova-pro-v1:0']['endpoints']) == {'base'}


def test_fm_list_keeps_mapped_custom_quotas_on_refresh(monkeypatch, tmp_path):
    from bedrock_usage_analyzer.sync import fm_list
    (tmp_path / 'data').mkdir(exist_ok=True)
    save_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'), {'models': [
        {'model_id': BASE, 'provider': 'Amazon', 'endpoints': {'custom': {'quotas': {'tpm': {'code': 'L-1', 'name': TPM}}}}}]})
    monkeypatch.setattr(fm_list, 'discover_prefix_mapping', lambda region, profiles=None: [])
    monkeypatch.setattr(fm_list, 'list_system_profiles', lambda region: [])
    monkeypatch.setattr(fm_list, 'fetch_foundation_models', lambda region: [
        {'model_id': BASE, 'provider': 'Amazon', 'inference_types': ['PROVISIONED'], 'customizations': ['FINE_TUNING']}])
    fm_list.refresh_region('us-east-1')
    saved = load_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'))['models'][0]
    assert saved['endpoints']['custom']['quotas']['tpm']['code'] == 'L-1'


def test_deployment_targets_are_the_deployments_by_name():
    fetcher = InferenceProfileFetcher(FakeCustom())
    ids, names, metadata = fetcher.find_profiles(BASE, 'custom', application_profile_ids=[DEPLOYMENT])
    assert ids == [DEPLOYMENT] and names[DEPLOYMENT] == 'my-lite' and metadata[DEPLOYMENT]['id'] == 'dep0000001'


def test_a_deployment_arn_with_m_becomes_a_custom_target(monkeypatch):
    from bedrock_usage_analyzer.core import user_inputs as ui_module
    monkeypatch.setattr(ui_module, 'create_client', lambda service, region=None, **_: FakeCustom())
    inputs = ui_module.UserInputs()
    inputs.region = 'us-east-1'
    assert inputs._parse_model_id(DEPLOYMENT) == {
        'model_id': BASE, 'profile_prefix': 'custom', 'application_profile_ids': [DEPLOYMENT]}


def test_a_deleted_deployment_arn_still_gives_its_usage(monkeypatch, caplog):
    class Missing(FakeCustom):
        def get_custom_model_deployment(self, customModelDeploymentIdentifier):
            raise aws_error('ResourceNotFoundException', 'GetCustomModelDeployment')
    inputs = _inputs(monkeypatch, Missing())
    assert inputs._parse_model_id(DEPLOYMENT) == {
        'model_id': 'dep0000001', 'profile_prefix': 'custom', 'application_profile_ids': [DEPLOYMENT]}
    assert 'if it was deleted' in caplog.text


def test_a_deployment_that_is_not_active_is_named(monkeypatch, caplog):
    class Failed(FakeCustom):
        def get_custom_model_deployment(self, customModelDeploymentIdentifier):
            return dict(super().get_custom_model_deployment(customModelDeploymentIdentifier), status='Failed')
    inputs = _inputs(monkeypatch, Failed())
    inputs._parse_model_id(DEPLOYMENT)
    assert 'my-lite is Failed, not Active' in caplog.text


def test_a_deployment_without_a_custom_model_says_so(monkeypatch, caplog):
    class NoModel(FakeCustom):
        def get_custom_model_deployment(self, customModelDeploymentIdentifier):
            found = dict(super().get_custom_model_deployment(customModelDeploymentIdentifier))
            found.pop('modelArn', None)
            return found
    inputs = _inputs(monkeypatch, NoModel())
    assert inputs._parse_model_id(DEPLOYMENT)['model_id'] == 'dep0000001'
    assert 'my-lite names no custom model' in caplog.text and 'no foundation base model' not in caplog.text


def test_an_application_profile_arn_needs_no_deployment_listing(monkeypatch):
    client = FakeCustom([SUMMARY])
    inputs = _inputs(monkeypatch, client)
    with pytest.raises(SystemExit):
        inputs._parse_model_id('arn:aws:bedrock:us-east-1:111122223333:application-inference-profile/missing00001')
    assert 'ListCustomModelDeployments' not in client.calls


def test_a_deployment_arn_that_cannot_be_read_is_analyzed_without_limits(monkeypatch, caplog):
    from bedrock_usage_analyzer.core import user_inputs as ui_module
    from botocore.exceptions import ClientError

    class Denied(FakeCustom):
        def get_custom_model_deployment(self, customModelDeploymentIdentifier):
            raise ClientError({'Error': {'Code': 'AccessDeniedException', 'Message': 'not authorized'}},
                              'GetCustomModelDeployment')
    monkeypatch.setattr(ui_module, 'create_client', lambda service, region=None, **_: Denied())
    inputs = ui_module.UserInputs()
    inputs.region = 'us-east-1'
    assert inputs._parse_model_id(DEPLOYMENT) == {
        'model_id': 'dep0000001', 'profile_prefix': 'custom', 'application_profile_ids': [DEPLOYMENT]}
    assert 'usage without limits' in caplog.text


def _inputs(monkeypatch, client):
    from bedrock_usage_analyzer.core import user_inputs as ui_module
    monkeypatch.setattr(ui_module, 'create_client', lambda service, region=None, **_: client)
    inputs = ui_module.UserInputs()
    inputs.region = 'us-east-1'
    return inputs


def test_a_bare_deployment_id_with_m_is_found_in_the_listing(monkeypatch):
    inputs = _inputs(monkeypatch, FakeCustom([SUMMARY]))
    assert inputs._parse_model_id('dep0000001') == {
        'model_id': BASE, 'profile_prefix': 'custom', 'application_profile_ids': [DEPLOYMENT]}


def test_an_unreadable_custom_model_falls_back_to_the_base_its_arn_names(monkeypatch, caplog):
    second = DEPLOYMENT.replace('dep0000001', 'dep0000002')
    third = DEPLOYMENT.replace('dep0000001', 'dep0000003')
    imported = 'arn:aws:bedrock:us-east-1:111122223333:custom-model/imported/cm00009'
    others = [dict(SUMMARY, customModelDeploymentArn=second, customModelDeploymentName='gone',
                   modelArn=CUSTOM_MODEL.replace('cm00001', 'cm00002')),
              dict(SUMMARY, customModelDeploymentArn=third, customModelDeploymentName='mine', modelArn=imported)]

    class Denied(FakeCustom):
        def get_custom_model(self, modelIdentifier):
            if modelIdentifier != CUSTOM_MODEL:
                raise aws_error('AccessDeniedException', 'GetCustomModel')
            return super().get_custom_model(modelIdentifier)
    inputs = _inputs(monkeypatch, Denied([SUMMARY] + others))
    caplog.set_level('INFO')
    monkeypatch.setattr('builtins.input', lambda prompt='': 'all')
    configs = inputs._select_custom_deployments(inputs._custom_deployments())
    assert [(c['model_id'], c['application_profile_ids']) for c in configs] == [
        (BASE, [DEPLOYMENT]), (BASE, [second]), ('dep0000003', [third])]
    assert 'using the base model its ARN names' in caplog.text
    assert 'mine has a custom model that could not be read' in caplog.text


def test_a_denied_deployment_read_uses_the_listing(monkeypatch):
    from botocore.exceptions import ClientError

    class Denied(FakeCustom):
        def get_custom_model_deployment(self, customModelDeploymentIdentifier):
            raise ClientError({'Error': {'Code': 'AccessDeniedException', 'Message': 'no'}}, 'GetCustomModelDeployment')
    inputs = _inputs(monkeypatch, Denied([SUMMARY]))
    assert inputs._parse_model_id(DEPLOYMENT)['model_id'] == BASE
    assert inputs.profile_fetcher._deployment_names[DEPLOYMENT] == 'my-lite'


def test_a_customizable_only_model_points_to_its_deployments(monkeypatch, caplog):
    inputs = _inputs(monkeypatch, FakeCustom())
    # Still analyzed (an entry kept for a model Bedrock no longer lists may have past usage)
    assert inputs._parse_model_id('amazon.nova-lite-v1:0:300k') == {'model_id': BASE, 'profile_prefix': None}
    assert 'under their custom model deployments' in caplog.text


def test_a_deployment_id_resolves_when_profiles_cannot_be_listed(monkeypatch):
    class NoProfiles(FakeCustom):
        def list_inference_profiles(self, **kwargs):
            raise RuntimeError('AccessDeniedException: not authorized to ListInferenceProfiles')
    inputs = _inputs(monkeypatch, NoProfiles([SUMMARY]))
    assert inputs._parse_model_id('my-lite')['application_profile_ids'] == [DEPLOYMENT]


def test_a_failed_deployment_listing_is_requested_once_and_reported(monkeypatch, caplog):
    from botocore.exceptions import ClientError

    class NoListing(FakeCustom):
        def list_custom_model_deployments(self, **kwargs):
            self.calls.append('ListCustomModelDeployments')
            raise ClientError({'Error': {'Code': 'AccessDeniedException', 'Message': 'not authorized'}},
                              'ListCustomModelDeployments')

        def get_custom_model_deployment(self, customModelDeploymentIdentifier):
            if customModelDeploymentIdentifier != 'my-lite':
                raise aws_error('ResourceNotFoundException', 'GetCustomModelDeployment')
            return super().get_custom_model_deployment(customModelDeploymentIdentifier)
    client = NoListing()
    inputs = _inputs(monkeypatch, client)
    with pytest.raises(SystemExit):
        inputs._parse_model_id('zzzzzzzzzzzz')
    assert client.calls.count('ListCustomModelDeployments') == 1
    # The direct read said there is no such deployment: no 'it may be one'
    assert 'Custom model deployments could not be listed either' not in caplog.text

    class Throttled(NoListing):
        def get_custom_model_deployment(self, customModelDeploymentIdentifier):
            raise aws_error('ThrottlingException', 'GetCustomModelDeployment')
    with pytest.raises(SystemExit):
        _inputs(monkeypatch, Throttled())._parse_model_id('zzzzzzzzzzzz')
    assert 'Custom model deployments could not be listed either' in caplog.text
    # A deployment name still resolves without the listing: GetCustomModelDeployment takes it
    assert inputs._parse_model_id('my-lite')['application_profile_ids'] == [DEPLOYMENT]


def test_a_failed_arn_read_falls_back_to_the_listed_arn_only(monkeypatch):
    # Another deployment is named like the requested deployment's ID: it must not be substituted
    impostor = dict(SUMMARY, customModelDeploymentArn=DEPLOYMENT.replace('dep0000001', 'dep0000009'),
                    customModelDeploymentName='dep0000001')

    class Missing(FakeCustom):
        def get_custom_model_deployment(self, customModelDeploymentIdentifier):
            raise aws_error('ResourceNotFoundException', 'GetCustomModelDeployment')
    inputs = _inputs(monkeypatch, Missing([impostor]))
    assert inputs._parse_model_id(DEPLOYMENT)['application_profile_ids'] == [DEPLOYMENT]


def test_an_old_boto3_says_what_it_needs(monkeypatch, caplog):
    class Old(FakeBedrock):
        pass  # no custom model deployment APIs
    inputs = _inputs(monkeypatch, Old())
    with pytest.raises(SystemExit):
        inputs._parse_model_id(DEPLOYMENT)
    assert 'need boto3 1.39.7' in caplog.text


@pytest.mark.parametrize('message,old', [
    ("'Bedrock' object has no attribute 'list_custom_model_deployments'", True),
    ("'Bedrock' object has no attribute 'get_custom_model'", True),
    # The tool's own typo is a bug to raise, not an old boto3
    ("'InferenceProfileFetcher' object has no attribute 'read_custom_model_deployments'", False),
    ("'NoneType' object has no attribute 'get'", False),
])
def test_only_a_missing_client_method_reads_as_an_old_boto3(message, old):
    from bedrock_usage_analyzer.core.profile_fetcher import missing_deployment_api
    assert missing_deployment_api(AttributeError(message)) is old
    assert not missing_deployment_api(ValueError(message))


def test_another_regions_deployments_are_refused_unread(monkeypatch):
    # The parallel pre-read skips them: the region check refuses them without API calls
    from bedrock_usage_analyzer.core import user_inputs as ui_module
    west = [DEPLOYMENT.replace('us-east-1', 'us-west-2').replace('dep0000001', d) for d in ('depA', 'depB')]
    client = FakeCustom()
    _inputs(monkeypatch, client)
    monkeypatch.setattr(ui_module, 'get_caller_identity',
                        lambda region=None, **_: {'Account': '111122223333', 'Arn': 'arn:aws:iam::1:user/a',
                                                  'Partition': 'aws'})
    with pytest.raises(SystemExit):
        ui_module.UserInputs().collect(region='us-east-1', model_id=west, skip_confirm=True,
                                       granularity_config={p: 60 for p in ['1hour', '1day', '7days', '14days', '30days']})
    assert 'GetCustomModelDeployment' not in client.calls


def test_deployments_read_at_once_are_not_read_again(monkeypatch):
    second = DEPLOYMENT.replace('dep0000001', 'dep0000002')
    client = FakeCustom()
    inputs = _inputs(monkeypatch, client)
    inputs._get_profile_fetcher().read_custom_deployments([DEPLOYMENT, second])
    before = list(client.calls)
    assert [inputs._parse_model_id(v)['model_id'] for v in (DEPLOYMENT, second)] == [BASE, BASE]
    assert client.calls == before  # answered from what was read in parallel
    assert before.count('GetCustomModelDeployment') == 2 and before.count('GetCustomModel') == 1


def test_deployments_are_listed_once_and_selected_names_are_reused(monkeypatch):
    client = FakeCustom([SUMMARY])
    inputs = _inputs(monkeypatch, client)
    inputs._custom_deployments()
    monkeypatch.setattr('builtins.input', lambda prompt='': '1')
    inputs._select_custom_deployments(inputs._custom_deployments())
    _, names, _ = inputs.profile_fetcher.find_profiles(BASE, 'custom', application_profile_ids=[DEPLOYMENT])
    assert names[DEPLOYMENT] == 'my-lite'
    assert [c for c in client.calls if isinstance(c, str)] == [
        'ListCustomModelDeployments', 'GetCustomModel']  # no GetCustomModelDeployment


def test_the_picker_offers_only_active_deployments(monkeypatch):
    failed = dict(SUMMARY, customModelDeploymentArn=DEPLOYMENT.replace('dep0000001', 'dep0000009'),
                  customModelDeploymentName='broken', status='Failed')
    inputs = _inputs(monkeypatch, FakeCustom([SUMMARY, failed]))
    offered = []
    monkeypatch.setattr(inputs, '_select_custom_deployments', lambda deployments: offered.extend(deployments) or [])
    monkeypatch.setattr('bedrock_usage_analyzer.core.user_inputs.select_from_list',
                        lambda prompt, modes, **_: next(m for m in modes if m.startswith('Custom')))
    inputs._select_targets('us-east-1')
    assert [d['name'] for d in offered] == ['my-lite']


def test_deployment_tags_are_reported(monkeypatch):
    class Tagged(FakeCustom):
        def list_tags_for_resource(self, resourceARN):
            return {'tags': [{'key': 'team', 'value': 'search'}]}
    _, _, metadata = InferenceProfileFetcher(Tagged()).find_profiles(BASE, 'custom', application_profile_ids=[DEPLOYMENT])
    assert metadata[DEPLOYMENT]['tags'] == {'team': 'search'}


def test_a_base_model_error_is_read_once_per_model(monkeypatch):
    class NoModel(FakeCustom):
        def get_custom_model(self, modelIdentifier):
            self.calls.append('GetCustomModel')
            raise aws_error('AccessDeniedException', 'GetCustomModel')
    client = NoModel()
    fetcher = InferenceProfileFetcher(client)
    for _ in range(3):
        with pytest.raises(ClientError):
            fetcher.deployment_base_model(CUSTOM_MODEL)
    assert client.calls == ['GetCustomModel']


def test_a_transient_listing_failure_is_retried_once():
    class Flaky(FakeCustom):
        def list_custom_model_deployments(self, **kwargs):
            self.calls.append('ListCustomModelDeployments')
            if len(self.calls) == 1:
                raise aws_error('ThrottlingException', 'ListCustomModelDeployments')
            return {'modelDeploymentSummaries': [SUMMARY]}
    assert InferenceProfileFetcher(Flaky()).list_custom_deployments()[0]['name'] == 'my-lite'


def test_fm_list_keeps_mapped_custom_quotas_when_customization_ends(monkeypatch, tmp_path):
    from bedrock_usage_analyzer.sync import fm_list
    (tmp_path / 'data').mkdir(exist_ok=True)
    save_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'), {'models': [
        {'model_id': BASE, 'provider': 'Amazon', 'endpoints': {'custom': {'quotas': {'tpm': {'code': 'L-1', 'name': TPM}}}}}]})
    monkeypatch.setattr(fm_list, 'discover_prefix_mapping', lambda region, profiles=None: [])
    monkeypatch.setattr(fm_list, 'list_system_profiles', lambda region: [])
    monkeypatch.setattr(fm_list, 'fetch_foundation_models', lambda region: [
        {'model_id': BASE, 'provider': 'Amazon', 'inference_types': ['PROVISIONED'], 'customizations': []}])
    fm_list.refresh_region('us-east-1')
    saved = load_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'))['models'][0]
    assert saved['endpoints']['custom']['quotas']['tpm']['code'] == 'L-1'


def test_a_bug_in_the_deployment_listing_is_raised(monkeypatch):
    inputs = _inputs(monkeypatch, FakeCustom())
    inputs.profile_fetcher = InferenceProfileFetcher(FakeCustom())
    inputs.profile_fetcher.list_custom_deployments = lambda: {}['missing']  # a KeyError, not an API error
    with pytest.raises(KeyError):
        inputs._custom_deployments()


def test_other_active_deployments_of_the_base_model_are_named(caplog):
    from bedrock_usage_analyzer.core.analyzer import BedrockAnalyzer
    sibling = dict(SUMMARY, customModelDeploymentArn=DEPLOYMENT.replace('dep0000001', 'dep0000002'),
                   customModelDeploymentName='other-lite', modelArn=CUSTOM_MODEL.replace('cm00001', 'cm00002'))
    failed = dict(sibling, customModelDeploymentArn=DEPLOYMENT.replace('dep0000001', 'dep0000003'), status='Failed')
    fetcher = InferenceProfileFetcher(FakeCustom([SUMMARY, sibling, failed]))
    fetcher.list_custom_deployments()
    analyzer = BedrockAnalyzer.__new__(BedrockAnalyzer)
    analyzer.profile_fetcher = fetcher
    caplog.set_level('INFO')
    analyzer._warn_other_deployments(BASE, [DEPLOYMENT])
    assert '1 other active custom model deployment(s)' in caplog.text and 'other-lite (dep0000002)' in caplog.text
    caplog.clear()
    analyzer._warn_other_deployments(BASE, [DEPLOYMENT, sibling['customModelDeploymentArn']])
    assert caplog.text == ''


def test_the_shared_quota_note_lists_deployments_for_an_arn_target(caplog):
    from bedrock_usage_analyzer.core.analyzer import BedrockAnalyzer
    sibling = dict(SUMMARY, customModelDeploymentArn=DEPLOYMENT.replace('dep0000001', 'dep0000002'),
                   customModelDeploymentName='other-lite')
    analyzer = BedrockAnalyzer.__new__(BedrockAnalyzer)
    analyzer.profile_fetcher = InferenceProfileFetcher(FakeCustom([SUMMARY, sibling]))  # nothing listed yet
    caplog.set_level('INFO')
    analyzer._warn_other_deployments(BASE, [DEPLOYMENT])
    assert 'other-lite (dep0000002)' in caplog.text


def test_a_sibling_whose_arn_names_no_base_is_read(caplog):
    from bedrock_usage_analyzer.core.analyzer import BedrockAnalyzer
    imported = 'arn:aws:bedrock:us-east-1:111122223333:custom-model/imported/cm00007'
    sibling = dict(SUMMARY, customModelDeploymentArn=DEPLOYMENT.replace('dep0000001', 'dep0000007'),
                   customModelDeploymentName='from-weights', modelArn=imported)
    client = FakeCustom([SUMMARY, sibling], bases={
        CUSTOM_MODEL: f"arn:aws:bedrock:us-east-1::foundation-model/{BASE}",
        imported: f"arn:aws:bedrock:us-east-1::foundation-model/{BASE}"})
    analyzer = BedrockAnalyzer.__new__(BedrockAnalyzer)
    analyzer.profile_fetcher = InferenceProfileFetcher(client)
    caplog.set_level('INFO')
    analyzer._warn_other_deployments(BASE, [DEPLOYMENT])
    assert 'from-weights (dep0000007)' in caplog.text


def test_a_bug_reading_a_base_model_is_raised(monkeypatch):
    class Broken(FakeCustom):
        def get_custom_model(self, modelIdentifier):
            return None  # .get on None: a TypeError/AttributeError, not an API error
    inputs = _inputs(monkeypatch, Broken())
    with pytest.raises(AttributeError):
        inputs._custom_deployment_config(DEPLOYMENT, list_deployments(FakeCustom([SUMMARY]))[0])


def test_selected_deployments_read_each_custom_model_once(monkeypatch):
    client = FakeCustom([SUMMARY, dict(SUMMARY, customModelDeploymentArn=DEPLOYMENT.replace('dep0000001', 'dep0000002'))])
    inputs = _inputs(monkeypatch, client)
    monkeypatch.setattr('builtins.input', lambda prompt='': 'all')
    configs = inputs._select_custom_deployments(inputs._custom_deployments())
    assert [c['model_id'] for c in configs] == [BASE, BASE]
    assert client.calls.count('GetCustomModel') == 1


@pytest.mark.parametrize('endpoint,mapped', [
    (None, False), ('TODO', False), ({'quotas': 'x'}, False), ({'quotas': {'tpm': None}}, False),
    ({'quotas': {'tpm': {'name': 'n'}}}, False), ({'quotas': {'tpm': {'code': 'L-1'}}}, True),
])
def test_a_custom_endpoint_counts_as_mapped_as_every_quota_reader_sees_it(endpoint, mapped):
    from bedrock_usage_analyzer.sync.fm_list import _has_mapped_quota
    assert _has_mapped_quota(endpoint) is mapped


def test_fm_list_keeps_mapped_custom_quotas_of_a_model_no_longer_listed(monkeypatch, tmp_path):
    from bedrock_usage_analyzer.sync import fm_list
    (tmp_path / 'data').mkdir(exist_ok=True)
    save_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'), {'models': [
        {'model_id': BASE, 'provider': 'Amazon', 'endpoints': {'custom': {'quotas': {'tpm': {'code': 'L-1', 'name': TPM}}}}},
        {'model_id': 'amazon.gone-v1:0', 'provider': 'Amazon', 'endpoints': {'base': {'quotas': {'tpm': None}}}}]})
    monkeypatch.setattr(fm_list, 'discover_prefix_mapping', lambda region, profiles=None: [])
    monkeypatch.setattr(fm_list, 'list_system_profiles', lambda region: [])
    monkeypatch.setattr(fm_list, 'fetch_foundation_models', lambda region: [
        {'model_id': 'amazon.nova-pro-v1:0', 'provider': 'Amazon', 'inference_types': ['ON_DEMAND']}])
    fm_list.refresh_region('us-east-1')
    saved = {m['model_id']: m for m in load_yaml(str(tmp_path / 'data' / 'fm-list-us-east-1.yml'))['models']}
    assert set(saved) == {'amazon.nova-pro-v1:0', BASE}
    assert saved[BASE]['endpoints'] == {'custom': {'quotas': {'tpm': {'code': 'L-1', 'name': TPM}}}}


IMPORTED = 'arn:aws:bedrock:us-east-1:111122223333:imported-model/imp0000001'


class FakeImported(FakeCustom):
    def __init__(self, imported=({'modelArn': IMPORTED, 'modelName': 'my-qwen'},), **kw):
        super().__init__(**kw)
        self.imported = list(imported)

    def list_imported_models(self, **kwargs):
        self.calls.append('ListImportedModels')
        return {'modelSummaries': self.imported}

    def get_imported_model(self, modelIdentifier):
        self.calls.append('GetImportedModel')
        return {'modelArn': IMPORTED, 'modelName': 'my-qwen'}


def test_imported_models_are_listed_and_read():
    from bedrock_usage_analyzer.aws.custom_models import is_imported, list_imported_models, read_imported_model

    class Paged(FakeImported):
        def list_imported_models(self, **kwargs):
            if 'nextToken' not in kwargs:
                return {'modelSummaries': [{'modelArn': IMPORTED}, {'modelName': 'no-arn'}], 'nextToken': 't'}
            return {'modelSummaries': [{'modelArn': IMPORTED.replace('imp0000001', 'imp2'), 'modelName': 'b'}]}
    assert list_imported_models(Paged()) == [{'arn': IMPORTED, 'name': 'imp0000001'},
                                             {'arn': IMPORTED.replace('imp0000001', 'imp2'), 'name': 'b'}]
    assert read_imported_model(FakeImported(), 'my-qwen') == {'arn': IMPORTED, 'name': 'my-qwen'}
    assert is_imported(IMPORTED) and not is_imported(DEPLOYMENT) and not is_imported(None)


def test_an_imported_model_is_analyzed_by_arn_id_or_name_without_limits(monkeypatch, caplog):
    caplog.set_level('INFO')
    expected = {'model_id': 'imp0000001', 'profile_prefix': 'imported', 'application_profile_ids': [IMPORTED]}
    for value in (IMPORTED, 'imp0000001', 'my-qwen'):
        inputs = _inputs(monkeypatch, FakeImported())
        assert inputs._parse_model_id(value) == expected
        assert inputs.profile_fetcher._deployment_names[IMPORTED] == 'my-qwen'
    assert 'Imported model my-qwen (imp0000001)' in caplog.text  # the analyzer says why it has no limits


def test_an_imported_model_id_wins_over_another_models_same_name(monkeypatch):
    other = IMPORTED.replace('imp0000001', 'imp2')
    # imp2 is named like imp0000001's ID and comes first in the listing
    client = FakeImported(imported=({'modelArn': other, 'modelName': 'imp0000001'},
                                    {'modelArn': IMPORTED, 'modelName': 'my-qwen'}))
    assert _inputs(monkeypatch, client)._parse_model_id('imp0000001')['application_profile_ids'] == [IMPORTED]


def test_without_the_listing_an_imported_id_is_read_by_its_arn_first(monkeypatch):
    class Unlisted(FakeImported):
        def list_imported_models(self, **kwargs):
            raise aws_error('AccessDeniedException', 'ListImportedModels')

        def get_imported_model(self, modelIdentifier):
            self.read.append(modelIdentifier)
            return {'modelArn': IMPORTED, 'modelName': 'my-qwen'}
    client = Unlisted()
    client.read = []
    inputs = _inputs(monkeypatch, client)
    inputs.account = '111122223333'
    assert inputs._parse_model_id('imp0000001')['application_profile_ids'] == [IMPORTED]
    assert client.read == [IMPORTED]  # the ARN its ID would have, before any name


def test_an_older_boto3_without_the_imported_apis_is_said_for_a_bare_name(monkeypatch, caplog):
    class Old(FakeImported):
        def list_imported_models(self, **kwargs):
            raise AttributeError("'Bedrock' object has no attribute 'list_imported_models'")

        def get_imported_model(self, modelIdentifier):
            raise AttributeError("'Bedrock' object has no attribute 'get_imported_model'")
    inputs = _inputs(monkeypatch, Old())
    assert inputs._deployment_target('my-qwen') is None
    assert isinstance(inputs.profile_fetcher.imported_models_error, AttributeError)
    inputs._report_deployment_listing_error('my-qwen')
    assert 'Imported models could not be checked' in caplog.text and 'boto3 1.39.7' in caplog.text


def test_an_older_boto3_without_the_deployment_apis_is_said_for_a_bare_name(monkeypatch, caplog):
    class Old(FakeImported):
        def list_custom_model_deployments(self, **kwargs):
            raise AttributeError("'Bedrock' object has no attribute 'list_custom_model_deployments'")
    inputs = _inputs(monkeypatch, Old(imported=()))
    assert inputs._deployment_target('my-lite') is None  # no direct read either: it would fail the same way
    inputs._report_deployment_listing_error('my-lite')
    assert 'Custom model deployments could not be checked' in caplog.text


def test_a_profile_id_shared_with_an_imported_models_name_is_warned_about(monkeypatch, caplog):
    inputs = _inputs(monkeypatch, FakeImported(imported=({'modelArn': IMPORTED, 'modelName': 'prof0000001'},)))
    inputs._note_imported_namesake('prof0000001', 'an application inference profile')
    assert 'WARNING: prof0000001 is analyzed as an application inference profile' in caplog.text


def test_a_deployment_id_wins_over_another_deployments_same_name(monkeypatch):
    other = DEPLOYMENT.replace('dep0000001', 'dep2')
    named_like_id = {**SUMMARY, 'customModelDeploymentArn': other, 'customModelDeploymentName': 'dep0000001'}
    inputs = _inputs(monkeypatch, FakeImported(deployments=[named_like_id, SUMMARY], imported=()))
    assert inputs._find_custom_deployment('dep0000001')['arn'] == DEPLOYMENT


def test_an_imported_model_arn_is_named_from_the_listing(monkeypatch):
    client = FakeImported()
    inputs = _inputs(monkeypatch, client)
    other = IMPORTED.replace('imp0000001', 'imp2')
    client.imported.append({'modelArn': other, 'modelName': 'b'})
    inputs._parse_model_id(IMPORTED)
    inputs._parse_model_id(other)
    assert inputs.profile_fetcher._deployment_names == {IMPORTED: 'my-qwen', other: 'b'}
    # One listing for both, no per-model read
    assert client.calls.count('ListImportedModels') == 1 and 'GetImportedModel' not in client.calls


def test_an_unreadable_imported_model_arn_is_still_analyzed(monkeypatch, caplog):
    class Denied(FakeImported):
        def list_imported_models(self, **kwargs):
            raise aws_error('AccessDeniedException', 'ListImportedModels')

        def get_imported_model(self, modelIdentifier):
            self.calls.append('GetImportedModel')
            raise aws_error('AccessDeniedException', 'GetImportedModel')
    caplog.set_level('INFO')
    client = Denied()
    inputs = _inputs(monkeypatch, client)
    assert inputs._parse_model_id(IMPORTED)['application_profile_ids'] == [IMPORTED]
    # The read permissions are optional: only the name is missing, said at info level
    assert 'imp0000001 is named by its ID' in caplog.text and 'WARNING' not in caplog.text
    assert inputs.profile_fetcher._deployment_names[IMPORTED] == 'imp0000001'
    # The failed read is not repeated
    inputs._parse_model_id(IMPORTED)
    assert client.calls.count('GetImportedModel') == 1


def test_an_imported_arn_the_listing_lacks_is_not_blamed_on_permissions(monkeypatch, caplog):
    class Unlisted(FakeImported):
        def get_imported_model(self, modelIdentifier):
            raise aws_error('AccessDeniedException', 'GetImportedModel')
    inputs = _inputs(monkeypatch, Unlisted(imported=()))  # listed fine: e.g. deleted, or another account's
    assert inputs._parse_model_id(IMPORTED)['application_profile_ids'] == [IMPORTED]
    assert 'is not among' in caplog.text and 'bedrock:ListImportedModels' not in caplog.text


def test_a_throttled_listing_does_not_blame_the_listing_permission(monkeypatch, caplog):
    class Throttled(FakeImported):
        def list_imported_models(self, **kwargs):
            raise aws_error('ThrottlingException', 'ListImportedModels')

        def get_imported_model(self, modelIdentifier):
            raise aws_error('AccessDeniedException', 'GetImportedModel')
    caplog.set_level('INFO')
    inputs = _inputs(monkeypatch, Throttled())
    assert inputs._parse_model_id(IMPORTED)['application_profile_ids'] == [IMPORTED]
    assert 'without bedrock:GetImportedModel its name' in caplog.text and 'ListImportedModels or' not in caplog.text


def test_a_bug_reading_an_imported_model_is_raised(monkeypatch):
    class Broken(FakeImported):
        def get_imported_model(self, modelIdentifier):
            raise KeyError('bug')
    with pytest.raises(KeyError):
        _inputs(monkeypatch, Broken(imported=()))._parse_model_id(IMPORTED)


def test_a_failed_imported_model_listing_is_retried_once():
    class Flaky(FakeImported):
        def list_imported_models(self, **kwargs):
            self.calls.append('ListImportedModels')
            if self.calls.count('ListImportedModels') == 1:
                raise aws_error('ThrottlingException', 'ListImportedModels')
            return {'modelSummaries': self.imported}
    client = Flaky()
    fetcher = InferenceProfileFetcher(client)
    assert fetcher.list_imported_models() == [{'arn': IMPORTED, 'name': 'my-qwen'}]
    assert fetcher.imported_models_error is None and client.calls.count('ListImportedModels') == 2


def test_an_imported_model_name_is_read_when_the_listing_fails(monkeypatch, caplog):
    class NoList(FakeImported):
        def list_imported_models(self, **kwargs):
            raise aws_error('AccessDeniedException', 'ListImportedModels')
    inputs = _inputs(monkeypatch, NoList())
    assert inputs._parse_model_id('my-qwen')['application_profile_ids'] == [IMPORTED]

    # A bare ID: GetImportedModel takes no ID, so it is read by the ARN it would have
    class ByArnOnly(NoList):
        def get_imported_model(self, modelIdentifier):
            if not modelIdentifier.startswith('arn:'):
                raise aws_error('ValidationException', 'GetImportedModel')
            return {'modelArn': modelIdentifier, 'modelName': 'my-qwen'}
    inputs = _inputs(monkeypatch, ByArnOnly())
    inputs.account = '111122223333'
    assert inputs._parse_model_id('imp0000001')['application_profile_ids'] == [IMPORTED]
    # An ARN, ID or name is analyzed without a 'not offered' note from the failed listing
    caplog.clear()
    caplog.set_level('INFO')
    assert _inputs(monkeypatch, NoList())._parse_model_id(IMPORTED)['application_profile_ids'] == [IMPORTED]
    assert _inputs(monkeypatch, NoList())._parse_model_id('my-qwen')['application_profile_ids'] == [IMPORTED]
    assert 'not offered' not in caplog.text

    # Read and not found: it is no imported model, so the error does not say it may be one
    class NotFound(NoList):
        def get_imported_model(self, modelIdentifier):
            raise aws_error('ResourceNotFoundException', 'GetImportedModel')
    inputs = _inputs(monkeypatch, NotFound())
    inputs.account = '111122223333'  # so the identifier is also read by its ARN
    with pytest.raises(SystemExit):
        inputs._parse_model_id('nothing-by-that-name')
    assert 'Imported models could not be listed either' not in caplog.text
    # Without the account it was never read by its ARN: it may still be one
    with pytest.raises(SystemExit):
        _inputs(monkeypatch, NotFound())._parse_model_id('nothing-by-that-name')
    assert 'Imported models could not be listed either' in caplog.text

    # A read by name that names no model ARN is no target (not a traceback later)
    class NoArn(NoList):
        def get_imported_model(self, modelIdentifier):
            if modelIdentifier.startswith('arn:'):
                raise aws_error('ResourceNotFoundException', 'GetImportedModel')
            return {'modelName': 'my-qwen'}
    caplog.clear()
    inputs = _inputs(monkeypatch, NoArn())
    inputs.account = '111122223333'
    with pytest.raises(SystemExit):
        inputs._parse_model_id('my-qwen')
    # Both reads answered, neither with a model: not 'it may be one'
    assert 'Imported models could not be listed either' not in caplog.text

    # A throttled read says nothing about whether it exists: it may be one
    caplog.clear()

    class Throttled(NoList):
        def get_imported_model(self, modelIdentifier):
            raise aws_error('ThrottlingException', 'GetImportedModel')
    with pytest.raises(SystemExit):
        _inputs(monkeypatch, Throttled())._parse_model_id('nothing-by-that-name')
    assert 'Imported models could not be listed either' in caplog.text
    caplog.clear()

    # Not readable either: it may be one
    class Neither(NoList):
        def get_imported_model(self, modelIdentifier):
            raise aws_error('AccessDeniedException', 'GetImportedModel')
    with pytest.raises(SystemExit):
        _inputs(monkeypatch, Neither())._parse_model_id('nothing-by-that-name')
    assert 'Imported models could not be listed either' in caplog.text
    # A denied listing is also what a region without Custom Model Import answers
    assert 'if us-east-1 offers Custom Model Import, it may be one' in caplog.text


def test_imported_models_are_offered_in_the_picker(monkeypatch):
    inputs = _inputs(monkeypatch, FakeImported())
    answers = iter(['2', 'all'])
    monkeypatch.setattr('builtins.input', lambda prompt='': next(answers))
    assert inputs._select_targets('us-east-1') == [
        {'model_id': 'imp0000001', 'profile_prefix': 'imported', 'application_profile_ids': [IMPORTED]}]


def test_a_failed_imported_model_listing_hides_the_choice_once(monkeypatch, caplog):
    caplog.set_level('INFO')

    class Denied(FakeImported):
        def list_imported_models(self, **kwargs):
            self.calls.append('ListImportedModels')
            raise aws_error('AccessDeniedException', 'ListImportedModels')
    client = Denied()
    inputs = _inputs(monkeypatch, client)
    assert inputs._imported_models() == [] and inputs._imported_models() == []
    assert client.calls.count('ListImportedModels') == 1
    # Denied is also what a region without Custom Model Import answers: said as either, once
    assert caplog.text.count('Custom Model Import is not available in us-east-1') == 1
    assert 'AccessDeniedException' not in caplog.text
    # Without the API (an older boto3) the choice is hidden too, and the boto3 needed is named
    assert _inputs(monkeypatch, FakeCustom())._imported_models() == []
    assert 'Imported models need boto3 1.39.7 or later' in caplog.text
    # An ARN passed with -m is still analyzed, named by its ID
    inputs = _inputs(monkeypatch, FakeCustom())
    assert inputs._parse_model_id(IMPORTED)['application_profile_ids'] == [IMPORTED]
    assert 'names imp0000001 by its ID' in caplog.text and 'could not be read' not in caplog.text


def test_a_bug_listing_imported_models_is_raised(monkeypatch):
    class Broken(FakeImported):
        def list_imported_models(self, **kwargs):
            raise KeyError('bug')
    with pytest.raises(KeyError):
        _inputs(monkeypatch, Broken())._imported_models()


def test_a_deployment_named_like_an_imported_model_says_so(monkeypatch, caplog):
    # Bedrock allows a deployment and an imported model of the same name: the deployment wins,
    # with a warning naming the imported model's ARN
    client = FakeImported(imported=({'modelArn': IMPORTED, 'modelName': 'my-lite'},), deployments=[SUMMARY])
    config = _inputs(monkeypatch, client)._parse_model_id('my-lite')
    assert config['application_profile_ids'] == [DEPLOYMENT]
    assert f'an imported model has the same name or ID. To analyze it, pass its ARN: -m {IMPORTED}' in caplog.text
    # No namesake: no warning
    caplog.clear()
    _inputs(monkeypatch, FakeImported(deployments=[SUMMARY]))._parse_model_id('my-lite')
    assert 'same name' not in caplog.text


def test_an_imported_model_target_keeps_its_noted_name(monkeypatch):
    client = FakeImported()
    fetcher = InferenceProfileFetcher(client)
    fetcher.note_deployment_name(IMPORTED, 'my-qwen')
    arns, names, _ = fetcher.find_profiles('imp0000001', 'imported', [IMPORTED])
    assert arns == [IMPORTED] and names == {IMPORTED: 'my-qwen'}
    # Not noted (an API caller): read with GetImportedModel, never as a deployment
    other = IMPORTED.replace('imp0000001', 'imp2')
    assert InferenceProfileFetcher(client).find_profiles('imp2', 'imported', [other])[1] == {other: 'my-qwen'}
    assert 'GetImportedModel' in client.calls

    class Unreadable(FakeImported):
        def get_imported_model(self, modelIdentifier):
            raise aws_error('AccessDeniedException', 'GetImportedModel')
    # Not readable: named by its ID
    assert InferenceProfileFetcher(Unreadable()).find_profiles('imp2', 'imported', [other])[1] == {other: 'imp2'}
    # Not noted but listed (an API caller that listed them): named from the listing
    listed = InferenceProfileFetcher(client)
    listed.list_imported_models()
    assert listed.find_profiles('imp0000001', 'imported', [IMPORTED])[1] == {IMPORTED: 'my-qwen'}
    assert 'GetCustomModelDeployment' not in client.calls
