# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""On-demand custom model deployments: discovery, quota mapping rules and analysis targets."""

import pytest

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
    assert read_deployment(FakeCustom(), DEPLOYMENT) == {'arn': DEPLOYMENT, 'name': 'my-lite', 'model_arn': CUSTOM_MODEL}
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


def test_a_missing_deployment_exits_with_its_arn(monkeypatch, caplog):
    from bedrock_usage_analyzer.core import user_inputs as ui_module

    class Missing(FakeCustom):
        def get_custom_model_deployment(self, customModelDeploymentIdentifier):
            raise RuntimeError('ResourceNotFoundException: no such deployment')
    monkeypatch.setattr(ui_module, 'create_client', lambda service, region=None, **_: Missing())
    inputs = ui_module.UserInputs()
    inputs.region = 'us-east-1'
    with pytest.raises(SystemExit):
        inputs._parse_model_id(DEPLOYMENT)
    assert DEPLOYMENT in caplog.text


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


def test_an_unreadable_base_model_leaves_out_only_the_limits(monkeypatch, caplog):
    second = DEPLOYMENT.replace('dep0000001', 'dep0000002')
    other = dict(SUMMARY, customModelDeploymentArn=second, customModelDeploymentName='gone',
                 modelArn=CUSTOM_MODEL + 'x')

    class Partial(FakeCustom):
        def get_custom_model(self, modelIdentifier):
            if modelIdentifier.endswith('x'):
                raise RuntimeError('ResourceNotFoundException: custom model deleted')
            return super().get_custom_model(modelIdentifier)
    inputs = _inputs(monkeypatch, Partial([SUMMARY, other]))
    monkeypatch.setattr('builtins.input', lambda prompt='': 'all')
    configs = inputs._select_custom_deployments(inputs._custom_deployments())
    assert [(c['model_id'], c['application_profile_ids']) for c in configs] == [
        (BASE, [DEPLOYMENT]), ('dep0000002', [second])]
    assert 'gone has a custom model that could not be read' in caplog.text


def test_a_deployment_id_resolves_when_profiles_cannot_be_listed(monkeypatch):
    class NoProfiles(FakeCustom):
        def list_inference_profiles(self, **kwargs):
            raise RuntimeError('AccessDeniedException: not authorized to ListInferenceProfiles')
    inputs = _inputs(monkeypatch, NoProfiles([SUMMARY]))
    assert inputs._parse_model_id('my-lite')['application_profile_ids'] == [DEPLOYMENT]


def test_a_failed_deployment_listing_is_requested_once_and_reported(monkeypatch, caplog):
    class NoListing(FakeCustom):
        def list_custom_model_deployments(self, **kwargs):
            self.calls.append('ListCustomModelDeployments')
            raise RuntimeError('AccessDeniedException: not authorized to ListCustomModelDeployments')
    client = NoListing()
    inputs = _inputs(monkeypatch, client)
    with pytest.raises(SystemExit):
        inputs._parse_model_id('zzzzzzzzzzzz')
    assert client.calls.count('ListCustomModelDeployments') == 1
    assert 'Custom model deployments could not be listed either' in caplog.text


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
            raise RuntimeError('AccessDeniedException: not authorized')
    client = NoModel()
    fetcher = InferenceProfileFetcher(client)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            fetcher.deployment_base_model(CUSTOM_MODEL)
    assert client.calls == ['GetCustomModel']


def test_a_transient_listing_failure_is_retried_once():
    class Flaky(FakeCustom):
        def list_custom_model_deployments(self, **kwargs):
            self.calls.append('ListCustomModelDeployments')
            if len(self.calls) == 1:
                raise RuntimeError('ThrottlingException: slow down')
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
