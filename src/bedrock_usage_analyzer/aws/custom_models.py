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


def is_deployment_arn(value: str) -> bool:
    """True for an on-demand custom model deployment ARN."""
    resource = value.split(':', 5)[-1] if value.startswith('arn:') and value.count(':') >= 5 else ''
    return resource.startswith(f"{DEPLOYMENT_KIND}/")


def deployment_short_id(arn: str) -> str:
    """The ID part of a deployment ARN (for labels and file names)."""
    return arn.rsplit('/', 1)[-1]


def _base_model_id(base_model_arn: Optional[str]) -> Optional[str]:
    if not base_model_arn or 'foundation-model/' not in base_model_arn:
        return None
    return base_model_arn.split('foundation-model/', 1)[1]


def resolve_deployment(bedrock_client, deployment_arn: str) -> Dict:
    """The deployment's name, custom model ARN and base model ID.

    Raises the API error (a missing deployment, a missing permission): the caller says
    which deployment could not be resolved.
    """
    deployment = bedrock_client.get_custom_model_deployment(customModelDeploymentIdentifier=deployment_arn)
    model_arn = deployment.get('modelArn')
    base = None
    if model_arn:
        model = bedrock_client.get_custom_model(modelIdentifier=model_arn)
        base = _base_model_id(model.get('baseModelArn'))
    return {
        'arn': deployment.get('customModelDeploymentArn') or deployment_arn,
        'name': deployment.get('modelDeploymentName') or deployment.get('customModelDeploymentName')
        or deployment_short_id(deployment_arn),
        'status': deployment.get('status'),
        'model_arn': model_arn,
        'base_model_id': base,
    }


def list_deployments(bedrock_client) -> List[Dict]:
    """The region's custom model deployments (summaries: arn, name, status, model_arn).

    Raises on a listing error; an API the region does not offer gives [].
    """
    deployments: List[Dict] = []
    kwargs: Dict = {}
    while True:
        try:
            response = bedrock_client.list_custom_model_deployments(**kwargs)
        except Exception as e:
            if 'UnknownOperation' in str(e) or 'UnknownOperationException' in type(e).__name__:
                return []  # not offered in this region
            raise
        for summary in response.get('modelDeploymentSummaries') or []:
            arn = summary.get('customModelDeploymentArn')
            if arn:
                deployments.append({
                    'arn': arn,
                    'name': summary.get('customModelDeploymentName') or deployment_short_id(arn),
                    'status': summary.get('status'),
                    'model_arn': summary.get('customModelArn') or summary.get('modelArn'),
                })
        token = response.get('nextToken')
        if not token:
            return deployments
        kwargs['nextToken'] = token
