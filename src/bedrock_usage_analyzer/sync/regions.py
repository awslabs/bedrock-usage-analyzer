# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AWS regions management with GovCloud support"""

import boto3
import logging
from typing import List
import sys

from bedrock_usage_analyzer.metadata.regions import refresh_regions
from bedrock_usage_analyzer.utils.yaml_handler import save_yaml
from bedrock_usage_analyzer.utils.paths import get_writable_path

logger = logging.getLogger(__name__)

# Regions to skip due to region disruption
SKIP_REGIONS = {'me-south-1', 'me-central-1'}


def fetch_enabled_regions() -> List[str]:
    """Fetch enabled AWS regions for the account (legacy function for compatibility)
    
    Returns:
        List of enabled region names
    """
    # Use the enhanced regions functionality
    regions_data = refresh_regions()
    regions = regions_data.get('regions', [])
    
    # Extract just the region names for backward compatibility
    if regions and isinstance(regions[0], dict):
        return [r['name'] for r in regions]
    else:
        return regions


def refresh_regions_legacy():
    """Legacy refresh function that saves to file
    
    Returns:
        dict: Regions data {'regions': [...]}
    """
    logger.info("Fetching enabled AWS regions (including GovCloud)...")
    
    # Use the enhanced regions functionality
    regions_data = refresh_regions()
    regions_data['regions'] = [r for r in regions_data.get('regions', []) if (r['name'] if isinstance(r, dict) else r) not in SKIP_REGIONS]
    if SKIP_REGIONS:
        logger.info(f"Skipping regions: {', '.join(sorted(SKIP_REGIONS))}")
    
    # Save to file
    regions_file = get_writable_path('regions.yml')
    save_yaml(regions_file, regions_data)
    logger.info(f"Saved regions to: {regions_file}")
    
    return regions_data


def main():
    """Main entry point"""
    refresh_regions_legacy()


if __name__ == "__main__":
    main()
