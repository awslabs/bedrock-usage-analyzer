# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AWS Service Quotas operations"""

import boto3
import sys
from typing import List, Dict, Optional


def fetch_service_quotas(region: str, service_code: str = 'bedrock') -> List[Dict]:
    """Fetch all service quotas for Bedrock
    
    Args:
        region: AWS region
        service_code: AWS service code (default: bedrock)
        
    Returns:
        List of quota dictionaries
    """
    try:
        client = boto3.client('service-quotas', region_name=region)
        quotas = []
        
        paginator = client.get_paginator('list_service_quotas')
        for page in paginator.paginate(ServiceCode=service_code):
            quotas.extend(page.get('Quotas', []))
        
        return quotas
    except Exception as e:
        print(f"Error fetching quotas for {region}: {e}", file=sys.stderr)
        return []


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
        client = boto3.client('service-quotas', region_name=region)
        response = client.get_service_quota(
            ServiceCode=service_code,
            QuotaCode=quota_code
        )
        return response.get('Quota', {})
    except client.exceptions.NoSuchResourceException:
        return None
    except Exception as e:
        print(f"Error fetching quota {quota_code}: {e}", file=sys.stderr)
        return None
