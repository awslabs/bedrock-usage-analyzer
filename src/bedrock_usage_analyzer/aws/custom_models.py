# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""On-demand custom model deployments: their base model, for quotas, and their metrics ID.

CloudWatch reports a deployment's usage with its deployment ARN as the ModelId. Its quotas
are the base model's "(Model customization) Sum of on demand custom model deployment ..."
quotas, mapped in the fm-list as the base model's 'custom' endpoint.
"""

import logging
from typing import Dict, List, Optional

from bedrock_usage_analyzer.aws.bedrock import model_id_from_arn

logger = logging.getLogger(__name__)

DEPLOYMENT_KIND = 'custom-model-deployment'
# A custom model fine-tuned from another custom model names that one as its base
_MAX_BASE_CHAIN = 10


def deployment_short_id(arn: str) -> str:
    """The ID part of a deployment ARN (for labels and file names)."""
    return arn.rsplit('/', 1)[-1]


def base_model_id_in_arn(model_arn: Optional[str]) -> Optional[str]:
    """The base model ID a custom model ARN names ('.../custom-model/<base model ID>/<id>'),
    or None (another ARN form). GetCustomModel is authoritative; this is the fallback."""
    resource = (model_arn or '').split(':custom-model/', 1)
    if len(resource) != 2 or resource[1].count('/') != 1:
        return None
    base = resource[1].split('/', 1)[0]
    return base if '.' in base else None


def base_model_id(bedrock_client, model_arn: Optional[str]) -> Optional[str]:
    """The foundation model a deployed model was trained from, following custom-model bases.

    None for a model without one (an imported model). Raises the API error of GetCustomModel.
    """
    model_arn = model_arn or ''
    for _ in range(_MAX_BASE_CHAIN):
        if model_id_from_arn(model_arn) or '/' not in model_arn:
            return model_id_from_arn(model_arn)
        model_arn = bedrock_client.get_custom_model(modelIdentifier=model_arn).get('baseModelArn') or ''
    return None


def read_deployment(bedrock_client, deployment_arn: str) -> Dict:
    """A deployment as list_deployments summarizes it (arn, name, model_arn). Raises the API
    error (a missing deployment, a missing permission)."""
    deployment = bedrock_client.get_custom_model_deployment(customModelDeploymentIdentifier=deployment_arn)
    return {'arn': deployment.get('customModelDeploymentArn') or deployment_arn,
            'name': deployment.get('modelDeploymentName') or deployment_short_id(deployment_arn),
            'model_arn': deployment.get('modelArn')}


def list_deployments(bedrock_client) -> List[Dict]:
    """The region's custom model deployments (summaries: arn, name, status, model_arn)."""
    deployments: List[Dict] = []
    kwargs: Dict = {'maxResults': 1000}
    while True:
        response = bedrock_client.list_custom_model_deployments(**kwargs)
        for summary in response.get('modelDeploymentSummaries') or []:
            arn = summary.get('customModelDeploymentArn')
            if arn:
                deployments.append({
                    'arn': arn,
                    'name': summary.get('customModelDeploymentName') or deployment_short_id(arn),
                    'status': summary.get('status'),
                    'model_arn': summary.get('modelArn'),
                })
        token = response.get('nextToken')
        if not token:
            return deployments
        kwargs['nextToken'] = token
