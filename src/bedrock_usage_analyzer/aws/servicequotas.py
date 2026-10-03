# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AWS Service Quotas operations"""

import logging
import threading
from typing import List, Dict, Optional

from botocore.exceptions import ClientError

from bedrock_usage_analyzer.aws.client_factory import create_client

logger = logging.getLogger(__name__)


def fetch_service_quotas(region: str, service_code: str = 'bedrock') -> List[Dict]:
    """Fetch all service quotas for Bedrock (kept from earlier releases)

    Args:
        region: AWS region
        service_code: AWS service code (default: bedrock)

    Returns:
        List of quota dictionaries; [] if the list could not be fetched. Code that must tell
        a failure from "no quotas" uses list_quota_codes, which returns None on failure.
    """
    quotas = list_quota_codes(region, service_code)  # logs the error itself
    return [] if quotas is None else list(quotas.values())


_clients: Dict[str, object] = {}
# boto3's default session is not thread-safe while it builds a client; lookups run in a pool
_clients_lock = threading.Lock()


def regional_client(region: str):
    """One Service Quotas client per region, reused across quota lookups (thread-safe)."""
    with _clients_lock:
        if region not in _clients:
            _clients[region] = create_client('service-quotas', region)
        return _clients[region]


def list_quota_codes(region: str, service_code: str = 'bedrock',
                     quiet_denied: bool = False) -> Optional[Dict[str, Dict]]:
    """All quotas of the service in a region by code (one paginated listing), or None on error."""
    try:
        quotas = {}
        paginator = regional_client(region).get_paginator('list_service_quotas')
        for page in paginator.paginate(ServiceCode=service_code):
            quotas.update((q['QuotaCode'], q) for q in page.get('Quotas', []) if q.get('QuotaCode'))
        return quotas
    except Exception as e:
        from bedrock_usage_analyzer.core.errors import is_access_denied
        # quiet_denied: callers that fall back to GetServiceQuota per code (the analyzer and
        # quota-index) do not need servicequotas:ListServiceQuotas, so no warning there
        (logger.debug if quiet_denied and is_access_denied(e) else logger.warning)(
            f"  Could not list {service_code} quotas in {region}: {e}")
        return None


QUOTA_OK = 'ok'
QUOTA_MISSING = 'missing'
QUOTA_ERROR = 'error'


def check_quota(quota_code: str, region: str, service_code: str = 'bedrock'):
    """Look up one quota and say whether it exists.

    Returns:
        (QUOTA_OK, quota dict), (QUOTA_MISSING, None) when Service Quotas says the
        code does not exist, or (QUOTA_ERROR, None) for any other failure (throttling,
        network, permissions), which callers must not treat as "missing".
    """
    client = None
    try:
        client = regional_client(region)
        response = client.get_service_quota(ServiceCode=service_code, QuotaCode=quota_code)
        return QUOTA_OK, response.get('Quota', {})
    except ClientError as e:
        code = e.response.get('Error', {}).get('Code')
        if code == 'NoSuchResourceException':
            # GetServiceQuota also answers this for a quota whose applied value is not
            # available (only its default): the code is missing only if the default is too
            return _check_default_quota(client, quota_code, region, service_code)
        _report_lookup_error(quota_code, region, code, e)
        return QUOTA_ERROR, None
    except Exception as e:
        _report_lookup_error(quota_code, region, type(e).__name__, e)
        return QUOTA_ERROR, None


def _check_default_quota(client, quota_code: str, region: str, service_code: str):
    try:
        response = client.get_aws_default_service_quota(ServiceCode=service_code, QuotaCode=quota_code)
        return QUOTA_OK, response.get('Quota', {})
    except ClientError as e:
        code = e.response.get('Error', {}).get('Code')
        if code == 'NoSuchResourceException':
            return QUOTA_MISSING, None
        _report_lookup_error(quota_code, region, code, e)
        return QUOTA_ERROR, None
    except Exception as e:
        _report_lookup_error(quota_code, region, type(e).__name__, e)
        return QUOTA_ERROR, None


def get_quota_details(quota_code: str, region: str, service_code: str = 'bedrock') -> Optional[Dict]:
    """Details of one quota, or None when it does not exist or cannot be read (kept from earlier releases)."""
    status, quota = check_quota(quota_code, region, service_code)
    return quota if status == QUOTA_OK else None

_reported_errors = set()
_reported_lock = threading.Lock()


def _report_lookup_error(quota_code: str, region: str, kind, error) -> None:
    """Warn once per region and kind of error; bulk lookups would otherwise print one line per code."""
    key = (region, kind)
    with _reported_lock:
        first = key not in _reported_errors
        _reported_errors.add(key)
    if first:
        logger.warning(f"Error fetching quota {quota_code} in {region}: {error} "
                       f"(further {kind} errors in {region} are logged at debug level)")
    else:
        logger.debug(f"Error fetching quota {quota_code} in {region}: {error}")



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


def lookup_quota(code: str, region: str, listings: Dict, lister=None, use_listing: bool = True):
    """(status, quota) of one code: from the region's listing (cached in ``listings``), else
    GetServiceQuota. The one lookup rule of the analyzer and `bua refresh quota-index`.

    ``lister`` fills a missing listing (default list_quota_codes; quota-index passes one that
    never lists, because it fetched every listing up front).
    """
    if use_listing:
        if region not in listings:
            listings[region] = (lister or list_quota_codes)(region)
        listed = (listings[region] or {}).get(code)
        if listed:
            return QUOTA_OK, listed
    return check_quota(code, region)


def is_missing(code: str, region: str, cache: Dict, lookup=None) -> bool:
    """True when Service Quotas says ``code`` does not exist in ``region`` (result cached)."""
    key = (code, region)
    if key not in cache:
        cache[key] = (lookup or check_quota)(code, region)[0]
    return cache[key] == QUOTA_MISSING
