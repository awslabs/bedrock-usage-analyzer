# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enhanced AWS regions management with GovCloud support"""

import boto3
import logging
from typing import List, Dict, Any, Optional
import sys

from bedrock_usage_analyzer.utils.aws_partition import (
    detect_partition_from_sts,
    get_partition_display_info,
    get_detection_summary,
    is_govcloud_region,
    get_region_display_name,
    get_partition_from_region,
    normalize_region
)
from bedrock_usage_analyzer.aws.client_factory import EnhancedClientFactory
from bedrock_usage_analyzer.core.govcloud_errors import create_govcloud_error_handler
from bedrock_usage_analyzer.utils.yaml_handler import save_yaml
from bedrock_usage_analyzer.utils.paths import get_writable_path

logger = logging.getLogger(__name__)


def fetch_enabled_regions_with_govcloud() -> List[Dict[str, Any]]:
    """Fetch enabled AWS regions including GovCloud regions
    
    Returns:
        List of region dictionaries with metadata
    """
    regions = []
    
    # Detect current partition to optimize discovery
    logger.info("🔍 Detecting AWS partition for optimized region discovery...")
    partition_result = detect_partition_from_sts()
    detected_partition = partition_result['partition']
    confidence = partition_result['confidence']
    
    if confidence in ['high', 'medium']:
        partition_info = get_partition_display_info(partition_result)
        logger.info(f"   ✅ {partition_info['icon']} {partition_info['name']} detected")
        
        # Only check the detected partition - credentials are partition-specific
        if detected_partition == 'govcloud':
            logger.info("   📋 Discovering GovCloud regions only (GovCloud credentials detected)")
            try:
                logger.info("Discovering GovCloud regions...")
                govcloud_regions = _fetch_regions_for_partition('aws-us-gov')
                regions.extend(govcloud_regions)
                logger.info(f"Found {len(govcloud_regions)} GovCloud regions")
            except Exception as e:
                logger.error(f"Could not discover GovCloud regions: {e}")
                
        elif detected_partition == 'commercial':
            logger.info("   📋 Discovering standard AWS regions only (commercial credentials detected)")
            try:
                logger.info("Discovering standard AWS regions...")
                standard_regions = _fetch_regions_for_partition('aws')
                regions.extend(standard_regions)
                logger.info(f"Found {len(standard_regions)} standard AWS regions")
            except Exception as e:
                logger.error(f"Could not discover standard AWS regions: {e}")
    else:
        # Unknown partition - try both partitions (fallback behavior)
        logger.info("   ❓ Could not detect partition - trying both standard and GovCloud regions")
        logger.info("   ⚠️  Note: This may result in access errors if credentials are partition-specific")
        
        # Try to discover standard AWS regions
        try:
            logger.info("Discovering standard AWS regions...")
            standard_regions = _fetch_regions_for_partition('aws')
            regions.extend(standard_regions)
            logger.info(f"Found {len(standard_regions)} standard AWS regions")
        except Exception as e:
            logger.warning(f"Could not discover standard AWS regions: {e}")
            logger.info("This may be normal if you only have GovCloud access")
        
        # Try to discover GovCloud regions
        try:
            logger.info("Discovering GovCloud regions...")
            govcloud_regions = _fetch_regions_for_partition('aws-us-gov')
            regions.extend(govcloud_regions)
            logger.info(f"Found {len(govcloud_regions)} GovCloud regions")
        except Exception as e:
            logger.warning(f"Could not discover GovCloud regions: {e}")
            logger.info("This is normal if you don't have GovCloud credentials configured")
    
    if not regions:
        logger.error("No regions found. Please check your AWS credentials and permissions.")
        sys.exit(1)
    
    # Sort regions by name
    regions.sort(key=lambda x: x['name'])
    
    return regions


def _fetch_regions_for_partition(partition: str) -> List[Dict[str, Any]]:
    """Fetch regions for a specific AWS partition
    
    Args:
        partition: AWS partition ('aws' or 'aws-us-gov')
        
    Returns:
        List of region dictionaries
    """
    regions = []
    
    # For GovCloud, we know the regions and can validate access
    if partition == 'aws-us-gov':
        # Known GovCloud regions
        known_govcloud_regions = ['us-gov-east-1', 'us-gov-west-1']
        
        for region_name in known_govcloud_regions:
            try:
                # Test access by creating an STS client
                sts = boto3.client('sts', region_name=region_name)
                # Try to get caller identity to verify access
                sts.get_caller_identity()
                
                regions.append({
                    'name': region_name,
                    'type': 'govcloud',
                    'display_name': get_region_display_name(region_name),
                    'partition': partition
                })
                logger.debug(f"Verified access to GovCloud region: {region_name}")
            except Exception as e:
                logger.debug(f"No access to GovCloud region {region_name}: {e}")
                continue
        
        return regions
    
    # For standard AWS, try Account service first, then EC2 fallback
    try:
        client = boto3.client('account')
        paginator = client.get_paginator('list_regions')
        for page in paginator.paginate(RegionOptStatusContains=['ENABLED', 'ENABLED_BY_DEFAULT']):
            for region_data in page.get('Regions', []):
                region_name = region_data['RegionName']
                
                # Only include standard regions (not GovCloud)
                if not is_govcloud_region(region_name):
                    regions.append({
                        'name': region_name,
                        'type': 'standard',
                        'display_name': get_region_display_name(region_name),
                        'partition': partition
                    })
    
    except Exception as e:
        # Fallback to EC2 describe-regions if Account service fails
        logger.warning(f"Account service failed for {partition}, trying EC2 fallback: {e}")
        regions = _fetch_regions_ec2_fallback(partition)
    
    return regions


def _fetch_regions_ec2_fallback(partition: str) -> List[Dict[str, Any]]:
    """Fallback method using EC2 describe-regions
    
    Args:
        partition: AWS partition
        
    Returns:
        List of region dictionaries
    """
    regions = []
    
    try:
        if partition == 'aws-us-gov':
            # For GovCloud, we already handle this in the main function
            # This fallback shouldn't be needed for GovCloud
            return []
        else:
            # For standard AWS, use EC2 describe-regions
            client = boto3.client('ec2')
            response = client.describe_regions()
            
            for region_data in response['Regions']:
                region_name = region_data['RegionName']
                
                # Only include standard regions (not GovCloud)
                if not is_govcloud_region(region_name):
                    regions.append({
                        'name': region_name,
                        'type': 'standard',
                        'display_name': get_region_display_name(region_name),
                        'partition': partition
                    })
    
    except Exception as e:
        logger.error(f"EC2 fallback also failed for {partition}: {e}")
        raise
    
    return regions


def refresh_regions():
    """Refresh the regions list with GovCloud support
    
    Returns:
        dict: Enhanced regions data with GovCloud metadata
    """
    logger.info("Fetching enabled AWS regions (including GovCloud)...")
    
    regions = fetch_enabled_regions_with_govcloud()
    
    if not regions:
        logger.error("No regions found")
        sys.exit(1)
    
    # Count by type
    standard_count = len([r for r in regions if r['type'] == 'standard'])
    govcloud_count = len([r for r in regions if r['type'] == 'govcloud'])
    
    logger.info(f"Found {len(regions)} total regions:")
    logger.info(f"  - {standard_count} standard AWS regions")
    logger.info(f"  - {govcloud_count} GovCloud regions")
    
    if govcloud_count > 0:
        logger.info("🏛️ GovCloud regions discovered and will be marked with 🏛️ in the UI")
    
    return {'regions': regions}


def get_region_display_info(region_name: str) -> Dict[str, Any]:
    """Get display information for a region
    
    Args:
        region_name: AWS region name
        
    Returns:
        Dict with display information
    """
    is_govcloud = is_govcloud_region(region_name)
    
    return {
        'name': region_name,
        'type': 'govcloud' if is_govcloud else 'standard',
        'display_name': get_region_display_name(region_name),
        'partition': get_partition_from_region(region_name),
        'is_govcloud': is_govcloud,
        'indicator': '🏛️' if is_govcloud else ''
    }


def main():
    """Main entry point"""
    refresh_regions()


if __name__ == "__main__":
    main()