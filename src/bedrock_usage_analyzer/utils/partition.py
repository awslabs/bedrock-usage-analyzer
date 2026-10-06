# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AWS partition detection and utilities for cross-partition support"""

import boto3
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Cache for partition detection
_cached_partition = None
_cached_account_id = None


def get_partition() -> str:
    """Detect AWS partition from caller identity

    Returns:
        str: Partition name ('aws', 'aws-us-gov', 'aws-cn', 'aws-iso', 'aws-iso-b')
    """
    global _cached_partition, _cached_account_id

    if _cached_partition is not None:
        return _cached_partition

    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        arn = identity['Arn']
        _cached_account_id = identity['Account']

        # Parse partition from ARN (format: arn:partition:service:region:account:resource)
        _cached_partition = arn.split(':')[1]

        logger.debug(f"Detected AWS partition: {_cached_partition}")
        return _cached_partition

    except Exception as e:
        logger.warning(f"Failed to detect partition, defaulting to 'aws': {e}")
        _cached_partition = 'aws'
        return _cached_partition


def get_account_id() -> Optional[str]:
    """Get cached account ID from partition detection

    Returns:
        str: AWS account ID or None if not yet detected
    """
    if _cached_account_id is None:
        # Trigger partition detection which also caches account ID
        get_partition()
    return _cached_account_id


def build_arn(service: str, region: str, account: str, resource: str) -> str:
    """Build ARN with correct partition

    Args:
        service: AWS service name (e.g., 'bedrock')
        region: AWS region (e.g., 'us-west-2')
        account: AWS account ID or empty string for no account
        resource: Resource identifier (e.g., 'foundation-model/model-id')

    Returns:
        str: Properly formatted ARN for the current partition
    """
    partition = get_partition()
    return f"arn:{partition}:{service}:{region}:{account}:{resource}"


def get_console_domain() -> str:
    """Get AWS Console domain for the current partition

    Returns:
        str: Console domain (e.g., 'console.aws.amazon.com' or 'console.amazonaws-us-gov.com')
    """
    partition = get_partition()

    domain_map = {
        'aws': 'console.aws.amazon.com',
        'aws-us-gov': 'console.amazonaws-us-gov.com',
        'aws-cn': 'console.amazonaws.cn',
        'aws-iso': 'console.c2s.ic.gov',
        'aws-iso-b': 'console.sc2s.sgov.gov'
    }

    return domain_map.get(partition, 'console.aws.amazon.com')


def get_service_quota_url(region: str, service_code: str, quota_code: str) -> str:
    """Build service quota console URL for the current partition

    Args:
        region: AWS region
        service_code: Service code (e.g., 'bedrock')
        quota_code: Quota code (e.g., 'L-xxx')

    Returns:
        str: Full console URL for the quota
    """
    console_domain = get_console_domain()
    return f"https://{region}.{console_domain}/servicequotas/home/services/{service_code}/quotas/{quota_code}"


def is_govcloud_region(region: str) -> bool:
    """Check if a region is a GovCloud region

    Args:
        region: AWS region name

    Returns:
        bool: True if region is a GovCloud region
    """
    return region.startswith('us-gov-')


def is_china_region(region: str) -> bool:
    """Check if a region is a China region

    Args:
        region: AWS region name

    Returns:
        bool: True if region is a China region
    """
    return region.startswith('cn-')
