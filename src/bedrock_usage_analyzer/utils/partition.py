# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AWS partition utilities (commercial, GovCloud, China and other partitions).

The partition of a region is resolved locally from botocore's endpoint data, so
building ARNs, console links and region names never needs an API call. The
partition of the *credentials* comes from the STS caller identity ARN and is
used to decide which regions to offer (GovCloud credentials cannot call
commercial regions and vice versa).
"""

import logging
import os
from typing import Dict, Iterable, List, Optional

import boto3
from botocore.loaders import create_loader

logger = logging.getLogger(__name__)

COMMERCIAL = 'aws'
GOVCLOUD = 'aws-us-gov'
CHINA = 'aws-cn'

# Fallback used only when botocore has no data for a region (e.g. a region
# launched after the installed botocore release). Longest prefix wins.
_REGION_PREFIX_PARTITIONS = [
    ('us-isob-', 'aws-iso-b'),
    ('us-isof-', 'aws-iso-f'),
    ('eu-isoe-', 'aws-iso-e'),
    ('us-iso-', 'aws-iso'),
    ('us-gov-', GOVCLOUD),
    ('eusc-', 'aws-eusc'),
    ('cn-', CHINA),
]

# AWS Management Console domain per partition. Partitions not listed here
# have no public console link, so reports show the quota code without a link.
_CONSOLE_DOMAINS = {
    COMMERCIAL: 'console.aws.amazon.com',
    GOVCLOUD: 'console.amazonaws-us-gov.com',
    CHINA: 'console.amazonaws.cn',
}

_PARTITION_NAMES = {
    COMMERCIAL: 'AWS Commercial',
    GOVCLOUD: 'AWS GovCloud (US)',
    CHINA: 'AWS China',
}

_endpoint_data = None
_caller_identity_cache: Dict[Optional[str], Dict[str, str]] = {}


def _load_endpoint_data() -> dict:
    global _endpoint_data
    if _endpoint_data is None:
        try:
            _endpoint_data = create_loader().load_data('endpoints')
        except Exception as e:  # pragma: no cover - botocore always ships this file
            logger.debug(f"Could not load botocore endpoint data: {e}")
            _endpoint_data = {'partitions': []}
    return _endpoint_data


def get_partition_for_region(region: Optional[str]) -> str:
    """Return the partition ('aws', 'aws-us-gov', 'aws-cn', ...) of a region."""
    if not region:
        return COMMERCIAL
    for partition in _load_endpoint_data().get('partitions', []):
        if region in partition.get('regions', {}):
            return partition['partition']
    for prefix, partition in _REGION_PREFIX_PARTITIONS:
        if region.startswith(prefix):
            return partition
    return COMMERCIAL


def is_govcloud_region(region: Optional[str]) -> bool:
    """True for AWS GovCloud (US) regions such as us-gov-west-1."""
    return get_partition_for_region(region) == GOVCLOUD


def is_china_region(region: Optional[str]) -> bool:
    """True for AWS China regions such as cn-north-1."""
    return get_partition_for_region(region) == CHINA


def get_partition_display_name(partition: str) -> str:
    return _PARTITION_NAMES.get(partition, partition)


def get_region_display_name(region: Optional[str]) -> str:
    """Human-readable region name, e.g. 'AWS GovCloud (US-West)'."""
    if not region:
        return 'Unknown Region'
    for partition in _load_endpoint_data().get('partitions', []):
        description = partition.get('regions', {}).get(region, {}).get('description')
        if description:
            return description
    return region


def get_region_info(region: str) -> Dict[str, object]:
    """Region metadata used by the reports and the region picker."""
    partition = get_partition_for_region(region)
    return {
        'name': region,
        'display_name': get_region_display_name(region),
        'partition': partition,
        'partition_name': get_partition_display_name(partition),
        'is_govcloud': partition == GOVCLOUD,
    }


def build_arn(service: str, region: str, account: str, resource: str) -> str:
    """Build an ARN in the partition that owns ``region``."""
    return f"arn:{get_partition_for_region(region)}:{service}:{region}:{account}:{resource}"


def get_console_domain(region: Optional[str] = None) -> Optional[str]:
    """Console domain for the region's partition, or None if there is no public console."""
    return _CONSOLE_DOMAINS.get(get_partition_for_region(region))


def get_service_quotas_console_url(region: Optional[str] = None) -> Optional[str]:
    """Link to the Service Quotas console home for the region's partition."""
    domain = get_console_domain(region)
    if not domain:
        return None
    if region:
        return f"https://{domain}/servicequotas/home?region={region}"
    return f"https://{domain}/servicequotas/home"


def get_service_quota_url(region: str, service_code: str, quota_code: str) -> Optional[str]:
    """Console URL of one quota, in the partition that owns ``region``."""
    domain = get_console_domain(region)
    if not domain:
        return None
    return (f"https://{domain}/servicequotas/home/services/{service_code}"
            f"/quotas/{quota_code}?region={region}")


def partition_regions(partition: str) -> List[str]:
    """All regions botocore knows for a partition (used as an offline fallback)."""
    for p in _load_endpoint_data().get('partitions', []):
        if p['partition'] == partition:
            return sorted(p.get('regions', {}).keys())
    return []


def filter_regions_by_partition(regions: Iterable[str], partition: Optional[str]) -> List[str]:
    """Keep only the regions that belong to ``partition`` (all regions if None)."""
    regions = list(regions)
    if not partition:
        return regions
    return [r for r in regions if get_partition_for_region(r) == partition]


def region_hint() -> Optional[str]:
    """Region configured in the environment or AWS config, if any."""
    region = os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION')
    if region:
        return region
    try:
        return boto3.session.Session().region_name
    except Exception:
        return None


def get_caller_identity(region: Optional[str] = None) -> Dict[str, str]:
    """Return {'Account', 'Arn', 'Partition'} for the current credentials (cached).

    ``region`` pins the STS endpoint. It matters for GovCloud and China
    credentials, which the global commercial STS endpoint rejects.
    """
    from bedrock_usage_analyzer.aws.client_factory import create_client

    key = region or region_hint()
    if key in _caller_identity_cache:
        return _caller_identity_cache[key]
    identity = create_client('sts', key).get_caller_identity()
    arn = identity.get('Arn', '')
    parts = arn.split(':')
    result = {
        'Account': identity.get('Account', ''),
        'Arn': arn,
        'Partition': parts[1] if len(parts) > 1 and parts[1] else get_partition_for_region(key),
    }
    _caller_identity_cache[key] = result
    return result


def detect_credentials_partition(region: Optional[str] = None) -> Optional[str]:
    """Partition of the current credentials, or None if it cannot be determined."""
    try:
        return get_caller_identity(region)['Partition']
    except Exception as e:
        logger.debug(f"Could not detect credentials partition: {e}")
        return None


def clear_cache() -> None:
    """Forget cached caller identities (used by tests)."""
    _caller_identity_cache.clear()
