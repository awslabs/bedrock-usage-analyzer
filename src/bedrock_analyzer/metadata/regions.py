# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AWS regions management"""

import boto3
import logging
from typing import List
import sys

from bedrock_analyzer.utils.yaml_handler import save_yaml

logger = logging.getLogger(__name__)


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
    """Refresh the regions list and save to metadata/regions.yml"""
    logger.info("Fetching enabled AWS regions...")
    
    regions = fetch_enabled_regions()
    
    if not regions:
        logger.error("No regions found")
        sys.exit(1)
    
    output_file = 'metadata/regions.yml'
    save_yaml(output_file, {'regions': regions})
    
    logger.info(f"Saved {len(regions)} enabled regions to {output_file}")


def main():
    """Main entry point"""
    refresh_regions()


if __name__ == "__main__":
    main()
