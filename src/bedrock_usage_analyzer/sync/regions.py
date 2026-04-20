# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AWS regions management"""

import boto3
import logging
from typing import List
import sys

from bedrock_usage_analyzer.utils.yaml_handler import save_yaml
from bedrock_usage_analyzer.utils.paths import get_writable_path

logger = logging.getLogger(__name__)

# Regions to skip due to region disruption
SKIP_REGIONS = {'me-south-1', 'me-central-1'}


def fetch_enabled_regions() -> List[str]:
    """Fetch enabled AWS regions for the account
    
    Returns:
        List of enabled region names
    """
    try:
        client = boto3.client('account')
        regions = []
        
        # Use paginator to get all regions
        paginator = client.get_paginator('list_regions')
        for page in paginator.paginate(RegionOptStatusContains=['ENABLED', 'ENABLED_BY_DEFAULT']):
            regions.extend([r['RegionName'] for r in page.get('Regions', [])])
        
        return sorted(regions)
    except Exception as e:
        logger.error(f"Error fetching regions: {e}")
        sys.exit(1)


def refresh_regions():
    """Refresh the regions list.
    
    Returns:
        dict: Regions data {'regions': [...]}
    """
    logger.info("Fetching enabled AWS regions...")
    
    regions = fetch_enabled_regions()
    regions = [r for r in regions if r not in SKIP_REGIONS]
    if SKIP_REGIONS:
        logger.info(f"Skipping regions: {', '.join(sorted(SKIP_REGIONS))}")
    
    if not regions:
        logger.error("No regions found")
        sys.exit(1)
    
    logger.info(f"Found {len(regions)} enabled regions")
    return {'regions': regions}


def main():
    """Main entry point"""
    refresh_regions()


if __name__ == "__main__":
    main()
