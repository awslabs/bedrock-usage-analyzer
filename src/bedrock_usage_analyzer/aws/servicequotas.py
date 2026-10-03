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
    quotas = list_quota_codes(region, service_code)  # logs the error itself
    return None if quotas is None else list(quotas.values())


_clients: Dict[str, object] = {}
# boto3's default session is not thread-safe while it builds a client; lookups run in a pool
_clients_lock = threading.Lock()


def regional_client(region: str):
    """One Service Quotas client per region, reused across quota lookups (thread-safe)."""
    with _clients_lock:
        if region not in _clients:
            _clients[region] = create_client('service-quotas', region)
        return _clients[region]


def list_quota_codes(region: str, service_code: str = 'bedrock') -> Optional[Dict[str, Dict]]:
    """All quotas of the service in a region by code (one paginated listing), or None on error."""
    try:
        quotas = {}
        paginator = regional_client(region).get_paginator('list_service_quotas')
        for page in paginator.paginate(ServiceCode=service_code):
            quotas.update((q['QuotaCode'], q) for q in page.get('Quotas', []) if q.get('QuotaCode'))
        return quotas
    except Exception as e:
        from bedrock_usage_analyzer.core.errors import is_access_denied
        # Without servicequotas:ListServiceQuotas, callers fall back to GetServiceQuota per code
        (logger.debug if is_access_denied(e) else logger.warning)(
            f"  Could not list {service_code} quotas in {region}: {e}")
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
        response = (client or regional_client(region)).get_service_quota(ServiceCode=service_code, QuotaCode=quota_code)
        return QUOTA_OK, response.get('Quota', {})
    except ClientError as e:
        if e.response.get('Error', {}).get('Code') == 'NoSuchResourceException':
            return QUOTA_MISSING, None
        print(f"Error fetching quota {quota_code}: {e}", file=sys.stderr)
        return QUOTA_ERROR, None
    except Exception as e:
        print(f"Error fetching quota {quota_code}: {e}", file=sys.stderr)
        return QUOTA_ERROR, None



def confirm_statuses(pairs, cache: Dict, lookup=None, workers: int = 8) -> Dict:
    """Look up, in parallel, the (code, region) pairs not yet in ``cache``; returns ``cache``.

    Each result is stored as cache[(code, region)] = QUOTA_OK / QUOTA_MISSING / QUOTA_ERROR.
    Used to confirm codes absent from a region's listing before they are removed.
    """
    from concurrent.futures import ThreadPoolExecutor
    lookup = lookup or check_quota
    todo = sorted(set(pairs) - set(cache))
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for key, result in zip(todo, pool.map(lambda k: lookup(*k), todo)):
                cache[key] = result[0]
    return cache


def lookup_quota(code: str, region: str, listings: Dict, check=None, lister=None, use_listing: bool = True):
    """(status, quota) of one code: from the region's listing (cached in ``listings``), else
    GetServiceQuota. The one lookup rule of the analyzer and `bua refresh quota-index`.

    ``check``/``lister`` default to check_quota/list_quota_codes; callers pass their own so a
    shared client or a test stub is used.
    """
    if use_listing:
        if region not in listings:
            listings[region] = (lister or list_quota_codes)(region)
        listed = (listings[region] or {}).get(code)
        if listed:
            return QUOTA_OK, listed
    return (check or check_quota)(code, region)
