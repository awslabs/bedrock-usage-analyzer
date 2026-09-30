# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures: no test may reach AWS or the user's real metadata directory."""

import pytest

from bedrock_usage_analyzer.aws import bedrock as bedrock_module
from bedrock_usage_analyzer.utils import partition


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Point metadata at an empty temp dir and use fake credentials."""
    monkeypatch.setenv('BEDROCK_ANALYZER_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'testing')
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'testing')
    monkeypatch.delenv('AWS_SESSION_TOKEN', raising=False)
    monkeypatch.delenv('AWS_PROFILE', raising=False)
    monkeypatch.setenv('AWS_DEFAULT_REGION', 'us-east-1')
    monkeypatch.delenv('AWS_REGION', raising=False)
    monkeypatch.setenv('AWS_CONFIG_FILE', str(tmp_path / 'aws_config'))
    monkeypatch.setenv('AWS_SHARED_CREDENTIALS_FILE', str(tmp_path / 'aws_credentials'))
    partition.clear_cache()
    bedrock_module._prefix_mapping_cache = None
    yield
    partition.clear_cache()
    bedrock_module._prefix_mapping_cache = None


def arn(region, model_id, partition_name='aws'):
    return f"arn:{partition_name}:bedrock:{region}::foundation-model/{model_id}"


class FakeBedrock:
    """Minimal Bedrock control-plane client for inference profile listing."""

    def __init__(self, system=(), application=(), tags=None, page_size=2):
        self.system = list(system)
        self.application = list(application)
        self.tags = tags or {}
        self.page_size = page_size
        self.calls = []

    def list_inference_profiles(self, maxResults=1000, typeEquals='SYSTEM_DEFINED', nextToken=None):
        self.calls.append(('list_inference_profiles', typeEquals, nextToken))
        items = self.system if typeEquals == 'SYSTEM_DEFINED' else self.application
        start = int(nextToken or 0)
        page = items[start:start + self.page_size]
        response = {'inferenceProfileSummaries': page}
        if start + self.page_size < len(items):
            response['nextToken'] = str(start + self.page_size)
        return response

    def list_tags_for_resource(self, resourceARN):
        self.calls.append(('list_tags_for_resource', resourceARN))
        return {'tags': [{'key': k, 'value': v} for k, v in self.tags.get(resourceARN, {}).items()]}


def system_profile(profile_id, arns):
    return {
        'inferenceProfileId': profile_id,
        'inferenceProfileName': profile_id,
        'type': 'SYSTEM_DEFINED',
        'models': [{'modelArn': a} for a in arns],
    }


def app_profile(profile_id, name, arns, region='ap-southeast-2', account='111122223333'):
    return {
        'inferenceProfileId': profile_id,
        'inferenceProfileName': name,
        'inferenceProfileArn': f"arn:aws:bedrock:{region}:{account}:application-inference-profile/{profile_id}",
        'status': 'ACTIVE',
        'type': 'APPLICATION',
        'models': [{'modelArn': a} for a in arns],
    }


HAIKU = 'anthropic.claude-haiku-4-5-20251001-v1:0'
NOVA = 'amazon.nova-lite-v1:0'

# Routing sets as returned by ap-southeast-2 on 2026-09-30
AU_ARNS = [arn('ap-southeast-2', HAIKU), arn('ap-southeast-4', HAIKU)]
GLOBAL_ARNS = [arn('', HAIKU), arn('ap-southeast-2', HAIKU)]
APAC_NOVA_ARNS = [arn(r, NOVA) for r in ('ap-southeast-2', 'ap-northeast-1', 'ap-south-1',
                                         'ap-northeast-2', 'ap-southeast-1', 'ap-northeast-3')]
JP_ARNS = [arn('ap-northeast-1', HAIKU), arn('ap-northeast-3', HAIKU)]
APAC_HAIKU_ARNS = [arn(r, HAIKU) for r in ('ap-northeast-1', 'ap-northeast-3', 'ap-southeast-2',
                                          'ap-southeast-1', 'ap-south-1')]


@pytest.fixture
def sydney_bedrock():
    """Profiles modelled on ap-southeast-2, including the issue #7 'au' case."""
    system = [
        system_profile(f"au.{HAIKU}", AU_ARNS),
        system_profile(f"global.{HAIKU}", GLOBAL_ARNS),
        system_profile(f"jp.{HAIKU}", JP_ARNS),
        system_profile(f"apac.{HAIKU}", APAC_HAIKU_ARNS),
        system_profile(f"apac.{NOVA}", APAC_NOVA_ARNS),
    ]
    application = [
        app_profile('auapp000001', 'team-a-au-haiku', AU_ARNS),
        app_profile('glapp000001', 'team-b-global-haiku', GLOBAL_ARNS),
        app_profile('jpapp000001', 'team-c-jp-haiku', JP_ARNS),
        app_profile('baseapp0001', 'team-d-base-haiku', [arn('ap-southeast-2', HAIKU)]),
        app_profile('novaapp0001', 'team-e-apac-nova', APAC_NOVA_ARNS),
    ]
    tags = {application[0]['inferenceProfileArn']: {'team': 'a', 'env': 'prod'}}
    return FakeBedrock(system=system, application=application, tags=tags)
