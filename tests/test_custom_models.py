# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""On-demand custom model deployments: discovery, quota mapping rules and analysis targets."""

import pytest

from bedrock_usage_analyzer.aws import bedrock
from bedrock_usage_analyzer.aws.custom_models import (
    deployment_short_id, is_deployment_arn, list_deployments, resolve_deployment)
from bedrock_usage_analyzer.core.profile_fetcher import InferenceProfileFetcher
from bedrock_usage_analyzer.sync.quota_rules import mapping_conflict
from bedrock_usage_analyzer.utils.yaml_handler import load_yaml, profile_endpoints, save_yaml

from conftest import FakeBedrock

DEPLOYMENT = 'arn:aws:bedrock:us-east-1:111122223333:custom-model-deployment/dep0000001'
CUSTOM_MODEL = 'arn:aws:bedrock:us-east-1:111122223333:custom-model/amazon.nova-lite-v1:0:300k/cm00001'
BASE = 'amazon.nova-lite-v1:0:300k'
TPM = '(Model customization) Sum of on demand custom model deployment tokens per minute for Amazon Nova Lite'


class FakeCustom(FakeBedrock):
    def __init__(self, deployments=(), **kw):
        super().__init__(**kw)
        self.deployments = list(deployments)

    def get_custom_model_deployment(self, customModelDeploymentIdentifier):
        return {'customModelDeploymentArn': customModelDeploymentIdentifier, 'modelDeploymentName': 'my-lite',
                'modelArn': CUSTOM_MODEL, 'status': 'Active'}

    def get_custom_model(self, modelIdentifier):
        return {'modelArn': modelIdentifier, 'baseModelArn': f"arn:aws:bedrock:us-east-1::foundation-model/{BASE}"}

    def list_custom_model_deployments(self, **kwargs):
        return {'modelDeploymentSummaries': self.deployments}


def test_deployment_arn_helpers():
    assert is_deployment_arn(DEPLOYMENT) and not is_deployment_arn(CUSTOM_MODEL)
    assert deployment_short_id(DEPLOYMENT) == 'dep0000001'


def test_a_deployment_resolves_to_its_base_model():
    info = resolve_deployment(FakeCustom(), DEPLOYMENT)
    assert (info['name'], info['base_model_id'], info['model_arn']) == ('my-lite', BASE, CUSTOM_MODEL)


def test_deployments_are_listed_and_an_unoffered_api_gives_none():
    summaries = [{'customModelDeploymentArn': DEPLOYMENT, 'customModelDeploymentName': 'my-lite',
                  'modelArn': CUSTOM_MODEL, 'status': 'Active'}]
    assert list_deployments(FakeCustom(summaries))[0]['name'] == 'my-lite'

    class NotOffered:
        def list_custom_model_deployments(self, **kwargs):
            raise RuntimeError('An error occurred (UnknownOperationException): Unknown Operation')
    assert list_deployments(NotOffered()) == []


def test_custom_endpoint_has_its_own_quota_keyword_and_is_no_profile_prefix():
    assert bedrock.get_endpoint_quota_keywords()['custom'] == 'custom model deployment'
    assert 'custom' not in bedrock.get_profile_prefixes()
    models = [{'model_id': BASE, 'endpoints': {'custom': {'quotas': {}}}}]
    assert profile_endpoints(models, BASE) == []


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


def test_an_unreadable_deployment_exits_with_its_arn(monkeypatch, caplog):
    from bedrock_usage_analyzer.core import user_inputs as ui_module

    class Denied(FakeCustom):
        def get_custom_model_deployment(self, customModelDeploymentIdentifier):
            raise RuntimeError('AccessDeniedException: not authorized')
    monkeypatch.setattr(ui_module, 'create_client', lambda service, region=None, **_: Denied())
    inputs = ui_module.UserInputs()
    inputs.region = 'us-east-1'
    with pytest.raises(SystemExit):
        inputs._parse_model_id(DEPLOYMENT)
    assert DEPLOYMENT in caplog.text
