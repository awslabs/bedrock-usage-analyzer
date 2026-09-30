# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AWS Service Quotas operations"""

import sys
from typing import List, Dict, Optional

from botocore.exceptions import ClientError

from bedrock_usage_analyzer.aws.client_factory import create_client


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


def get_quota_details(quota_code: str, region: str, service_code: str = 'bedrock') -> Optional[Dict]:
    """Get details for a specific quota

    Args:
        quota_code: Quota code (L-xxx)
        region: AWS region
        service_code: AWS service code (default: bedrock)

    Returns:
        Quota details dictionary or None if not found
    """
    try:
        client = create_client('service-quotas', region)
        response = client.get_service_quota(
            ServiceCode=service_code,
            QuotaCode=quota_code
        )
        return response.get('Quota', {})
    except ClientError as e:
        if e.response.get('Error', {}).get('Code') != 'NoSuchResourceException':
            print(f"Error fetching quota {quota_code}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error fetching quota {quota_code}: {e}", file=sys.stderr)
        return None
