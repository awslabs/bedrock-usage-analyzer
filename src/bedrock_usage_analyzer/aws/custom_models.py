# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""On-demand custom model deployments: their base model, for quotas, and their metrics ID.

CloudWatch reports a deployment's usage with its deployment ARN as the ModelId. Its quotas
are the base model's "(Model customization) Sum of on demand custom model deployment ..."
quotas, mapped in the fm-list as the base model's 'custom' endpoint.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEPLOYMENT_KIND = 'custom-model-deployment'
# A custom model fine-tuned from another custom model names that one as its base
_MAX_BASE_CHAIN = 10


def deployment_short_id(arn: str) -> str:
    """The ID part of a deployment ARN (for labels and file names)."""
    return arn.rsplit('/', 1)[-1]


def _base_model_id(bedrock_client, model_arn: str) -> Optional[str]:
    """The foundation model a custom model was trained from, following custom-model bases."""
    for _ in range(_MAX_BASE_CHAIN):
        base_arn = bedrock_client.get_custom_model(modelIdentifier=model_arn).get('baseModelArn') or ''
        if 'foundation-model/' in base_arn:
            return base_arn.split('foundation-model/', 1)[1]
        if '/' not in base_arn:
            return None  # e.g. an imported model: no base model
        model_arn = base_arn
    return None


def resolve_deployment(bedrock_client, deployment_arn: str, summary: Optional[Dict] = None) -> Dict:
    """The deployment's ARN, name and base model ID.

    ``summary`` (from list_deployments) saves reading the deployment again. Raises the API
    error (a missing deployment, a missing permission): the caller says which deployment
    could not be resolved.
    """
    if summary is None:
        deployment = bedrock_client.get_custom_model_deployment(customModelDeploymentIdentifier=deployment_arn)
        summary = {'arn': deployment.get('customModelDeploymentArn') or deployment_arn,
                   'name': deployment.get('modelDeploymentName') or deployment_short_id(deployment_arn),
                   'model_arn': deployment.get('modelArn')}
    model_arn = summary.get('model_arn')
    return {'arn': summary['arn'], 'name': summary['name'],
            'base_model_id': _base_model_id(bedrock_client, model_arn) if model_arn else None}


def list_deployments(bedrock_client) -> List[Dict]:
    """The region's custom model deployments (summaries: arn, name, status, model_arn)."""
    deployments: List[Dict] = []
    kwargs: Dict = {}
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
