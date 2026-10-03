# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AWS Service Quotas operations"""

import logging
import sys
import threading
from typing import List, Dict, Optional

from botocore.exceptions import ClientError

from bedrock_usage_analyzer.aws.client_factory import create_client

logger = logging.getLogger(__name__)


def fetch_service_quotas(region: str, service_code: str = 'bedrock') -> Optional[List[Dict]]:
    """Fetch all service quotas for Bedrock

    Args:
        region: AWS region
        service_code: AWS service code (default: bedrock)

    Returns:
        List of quota dictionaries, or None if the list could not be fetched
        (callers must not treat a failure as "no quotas")
    """
    try:
        client = create_client('service-quotas', region)
        quotas = []

        paginator = client.get_paginator('list_service_quotas')
        for page in paginator.paginate(ServiceCode=service_code):
            quotas.extend(page.get('Quotas', []))

        return quotas
    except Exception as e:
        print(f"Error fetching quotas for {region}: {e}", file=sys.stderr)
        return None


_clients: Dict[str, object] = {}
# boto3's default session is not thread-safe while it builds a client; lookups run in a pool
_clients_lock = threading.Lock()


def _client(region: str):
    """One Service Quotas client per region, reused across quota lookups (thread-safe)."""
    with _clients_lock:
        if region not in _clients:
            _clients[region] = create_client('service-quotas', region)
        return _clients[region]


def list_quota_codes(region: str, service_code: str = 'bedrock') -> Optional[Dict[str, Dict]]:
    """All quotas of the service in a region by code (one paginated listing), or None on error."""
    try:
        quotas = {}
        paginator = _client(region).get_paginator('list_service_quotas')
        for page in paginator.paginate(ServiceCode=service_code):
            quotas.update((q['QuotaCode'], q) for q in page.get('Quotas', []) if q.get('QuotaCode'))
        return quotas
    except Exception as e:
        logger.debug(f"Could not list {service_code} quotas in {region}: {e}")
        return None


QUOTA_OK = 'ok'
QUOTA_MISSING = 'missing'
QUOTA_ERROR = 'error'


def check_quota(quota_code: str, region: str, service_code: str = 'bedrock', client=None):
    """Look up one quota and say whether it exists.

    Returns:
        (QUOTA_OK, quota dict), (QUOTA_MISSING, None) when Service Quotas says the
        code does not exist, or (QUOTA_ERROR, None) for any other failure (throttling,
        network, permissions), which callers must not treat as "missing".
    """
    try:
        response = (client or _client(region)).get_service_quota(ServiceCode=service_code, QuotaCode=quota_code)
        return QUOTA_OK, response.get('Quota', {})
    except ClientError as e:
        if e.response.get('Error', {}).get('Code') == 'NoSuchResourceException':
            return QUOTA_MISSING, None
        print(f"Error fetching quota {quota_code}: {e}", file=sys.stderr)
        return QUOTA_ERROR, None
    except Exception as e:
        print(f"Error fetching quota {quota_code}: {e}", file=sys.stderr)
        return QUOTA_ERROR, None

