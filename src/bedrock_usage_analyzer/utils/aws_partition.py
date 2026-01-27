# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AWS partition detection and configuration utilities

This module consolidates all partition-related functionality including:
- Partition detection from STS credentials
- Region pattern matching
- Endpoint configuration
- Region format normalization
"""

import re
import boto3
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Partition Detection (from STS credentials)
# ============================================================================

class PartitionCache:
    """Cache for partition detection to avoid redundant STS calls"""
    _cached_result: Optional[Dict[str, Any]] = None
    
    @classmethod
    def get(cls) -> Optional[Dict[str, Any]]:
        """Get cached partition detection result"""
        return cls._cached_result
    
    @classmethod
    def set(cls, result: Dict[str, Any]) -> None:
        """Set cached partition detection result"""
        cls._cached_result = result
    
    @classmethod
    def clear(cls) -> None:
        """Clear cached partition detection result"""
        cls._cached_result = None


def detect_partition_from_sts() -> Dict[str, Any]:
    """Detect AWS partition from STS caller identity ARN
    
    Returns:
        Dict containing partition detection results:
        - partition: 'commercial', 'govcloud', or 'unknown'
        - confidence: 'high', 'medium', 'low', or 'none'
        - method: 'sts_arn'
        - arn: The caller identity ARN (if available)
        - error: Error message (if detection failed)
    """
    # Check cache first
    cached = PartitionCache.get()
    if cached is not None:
        logger.debug("Using cached partition detection result")
        return cached
    
    try:
        # Create STS client and get caller identity
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        
        # Extract ARN from response
        arn = identity.get('Arn', '')
        account_id = identity.get('Account', '')
        user_id = identity.get('UserId', '')
        
        logger.debug(f"STS Identity - ARN: {arn}, Account: {account_id}")
        
        if not arn:
            result = {
                'partition': 'unknown',
                'confidence': 'none',
                'method': 'sts_arn',
                'error': 'No ARN found in STS response',
                'account_id': account_id
            }
            PartitionCache.set(result)
            return result
        
        # Analyze ARN format to determine partition
        # Commercial ARN: arn:aws:iam::123456789012:user/username
        # GovCloud ARN:   arn:aws-us-gov:iam::123456789012:user/username
        
        if ':aws-us-gov:' in arn:
            result = {
                'partition': 'govcloud',
                'confidence': 'high',
                'method': 'sts_arn',
                'arn': arn,
                'account_id': account_id,
                'user_id': user_id
            }
        elif ':aws:' in arn:
            result = {
                'partition': 'commercial',
                'confidence': 'high', 
                'method': 'sts_arn',
                'arn': arn,
                'account_id': account_id,
                'user_id': user_id
            }
        else:
            # ARN format not recognized - could be a new partition or format
            result = {
                'partition': 'unknown',
                'confidence': 'none',
                'method': 'sts_arn',
                'arn': arn,
                'account_id': account_id,
                'error': f'Unrecognized ARN partition format: {arn}'
            }
        
        # Cache the result
        PartitionCache.set(result)
        return result
            
    except Exception as e:
        logger.debug(f"STS partition detection failed: {e}")
        result = {
            'partition': 'unknown',
            'confidence': 'none',
            'method': 'sts_arn',
            'error': str(e)
        }
        PartitionCache.set(result)
        return result


# ============================================================================
# Region Pattern Matching
# ============================================================================

# GovCloud region pattern - regions starting with 'us-gov-'
GOVCLOUD_PATTERN = re.compile(r'^us-gov-')


def is_govcloud_region(region: str) -> bool:
    """Determine if region is a GovCloud region
    
    Args:
        region: AWS region name (e.g., 'us-gov-east-1', 'us-east-1')
        
    Returns:
        bool: True if region is a GovCloud region, False otherwise
    """
    if not region or not isinstance(region, str):
        return False
    return bool(GOVCLOUD_PATTERN.match(region))


def get_region_type(region: str) -> str:
    """Return region type classification
    
    Args:
        region: AWS region name
        
    Returns:
        str: 'govcloud' for GovCloud regions, 'standard' for others
    """
    return 'govcloud' if is_govcloud_region(region) else 'standard'


def get_partition_from_region(region: str) -> str:
    """Get AWS partition for the region
    
    Args:
        region: AWS region name
        
    Returns:
        str: AWS partition ('aws-us-gov' for GovCloud, 'aws' for standard)
    """
    return 'aws-us-gov' if is_govcloud_region(region) else 'aws'


# ============================================================================
# Region Display Names
# ============================================================================

# Display name mapping for known regions
REGION_DISPLAY_NAMES = {
    'us-gov-east-1': 'AWS GovCloud (US-East)',
    'us-gov-west-1': 'AWS GovCloud (US-West)',
    'us-east-1': 'US East (N. Virginia)',
    'us-east-2': 'US East (Ohio)',
    'us-west-1': 'US West (N. California)',
    'us-west-2': 'US West (Oregon)',
    'eu-west-1': 'Europe (Ireland)',
    'eu-central-1': 'Europe (Frankfurt)',
    'ap-southeast-1': 'Asia Pacific (Singapore)',
    'ap-southeast-2': 'Asia Pacific (Sydney)',
}


def get_region_display_name(region: str) -> str:
    """Return user-friendly region name with GovCloud indicator
    
    Args:
        region: AWS region name
        
    Returns:
        str: Display name with GovCloud indicator if applicable
    """
    if not region:
        return "Unknown Region"
    
    if region in REGION_DISPLAY_NAMES:
        return REGION_DISPLAY_NAMES[region]
    
    # For unknown regions, generate display name with GovCloud indicator
    if is_govcloud_region(region):
        return f"AWS GovCloud ({region})"
    else:
        return region


# ============================================================================
# Endpoint Configuration
# ============================================================================

# GovCloud service endpoint mappings
GOVCLOUD_ENDPOINTS = {
    'bedrock': 'bedrock.{region}.amazonaws.com',
    'bedrock-runtime': 'bedrock-runtime.{region}.amazonaws.com',
    'cloudwatch': 'monitoring.{region}.amazonaws.com',
    'service-quotas': 'servicequotas.{region}.amazonaws.com',
    'sts': 'sts.{region}.amazonaws.com',
    'account': 'account.{region}.amazonaws.com'
}

# Services known to be available in GovCloud
GOVCLOUD_AVAILABLE_SERVICES = {
    'bedrock', 'bedrock-runtime', 'cloudwatch', 'service-quotas', 'sts', 'account'
}


def get_service_endpoint(service: str, region: str) -> Optional[str]:
    """Get service-specific endpoint URL for region
    
    Args:
        service: AWS service name (e.g., 'bedrock', 'cloudwatch')
        region: AWS region name
        
    Returns:
        Optional[str]: Service endpoint URL for GovCloud regions, None for standard regions
    """
    if not is_govcloud_region(region):
        # For standard regions, return None to use default endpoints
        return None
        
    if service not in GOVCLOUD_ENDPOINTS:
        logger.warning(f"Unknown service '{service}' for GovCloud endpoint configuration")
        return None
        
    endpoint_template = GOVCLOUD_ENDPOINTS[service]
    return f"https://{endpoint_template.format(region=region)}"


def get_client_config(service: str, region: str) -> Dict[str, Any]:
    """Get boto3 client configuration arguments
    
    Args:
        service: AWS service name
        region: AWS region name
        
    Returns:
        Dict[str, Any]: Configuration arguments for boto3 client creation
    """
    config = {
        'region_name': region
    }
    
    # Add endpoint URL for GovCloud regions
    endpoint_url = get_service_endpoint(service, region)
    if endpoint_url:
        config['endpoint_url'] = endpoint_url
        
    return config


def is_service_available(service: str, region: str) -> bool:
    """Check if service is available in the specified region
    
    Args:
        service: AWS service name
        region: AWS region name
        
    Returns:
        bool: True if service is available, False otherwise
    """
    if is_govcloud_region(region):
        return service in GOVCLOUD_AVAILABLE_SERVICES
    else:
        # For standard regions, assume all services are available
        return True


# ============================================================================
# Partition Detection Helpers
# ============================================================================

def get_partition_display_info(partition_result: Dict[str, Any]) -> Dict[str, Any]:
    """Get user-friendly partition information for display
    
    Args:
        partition_result: Result from detect_partition_from_sts()
        
    Returns:
        Dict with display information
    """
    partition = partition_result['partition']
    confidence = partition_result['confidence']
    
    if partition == 'govcloud':
        return {
            'name': 'AWS GovCloud (US)',
            'icon': '🏛️',
            'description': 'Government cloud partition',
            'confidence': confidence,
            'partition': partition
        }
    elif partition == 'commercial':
        return {
            'name': 'AWS Commercial',
            'icon': '🌍',
            'description': 'Standard AWS partition', 
            'confidence': confidence,
            'partition': partition
        }
    else:
        return {
            'name': 'Unknown Partition',
            'icon': '❓',
            'description': 'Could not determine partition',
            'confidence': 'none',
            'partition': 'unknown'
        }


def should_auto_filter(partition_result: Dict[str, Any]) -> bool:
    """Determine if regions should be automatically filtered based on detection confidence
    
    Args:
        partition_result: Result from detect_partition_from_sts()
        
    Returns:
        True if regions should be filtered, False to show all regions
    """
    return (
        partition_result['partition'] != 'unknown' and 
        partition_result['confidence'] in ['high', 'medium']
    )


def get_detection_summary(partition_result: Dict[str, Any]) -> str:
    """Get a formatted summary of the partition detection result
    
    Args:
        partition_result: Result from detect_partition_from_sts()
        
    Returns:
        Formatted summary string for user display
    """
    partition_info = get_partition_display_info(partition_result)
    
    if partition_result['confidence'] == 'high':
        return f"✅ {partition_info['icon']} {partition_info['name']} detected"
    elif partition_result['confidence'] == 'medium':
        return f"🔍 {partition_info['icon']} {partition_info['name']} detected (medium confidence)"
    else:
        error_msg = partition_result.get('error', 'Unknown error')
        return f"❓ Could not detect partition: {error_msg}"


def filter_regions_by_partition(regions: List[Dict[str, Any]], partition: str) -> List[Dict[str, Any]]:
    """Filter regions to show only accessible ones based on partition
    
    Args:
        regions: List of region dictionaries
        partition: Detected partition ('commercial', 'govcloud', or 'unknown')
        
    Returns:
        Filtered list of regions
    """
    if partition == 'govcloud':
        # Show only GovCloud regions
        return [r for r in regions if r.get('type') == 'govcloud']
    elif partition == 'commercial':
        # Show only standard/commercial regions
        return [r for r in regions if r.get('type') == 'standard']
    else:
        # Show all regions if partition is unknown
        return regions


# ============================================================================
# Region Format Normalization
# ============================================================================

def normalize_region(region) -> Dict[str, Any]:
    """Normalize a region to standard dictionary format
    
    Args:
        region: Region as string or dict
        
    Returns:
        Dict with standardized region information
    """
    if isinstance(region, dict):
        # Already in dict format, ensure all required fields
        return {
            'name': region['name'],
            'type': region.get('type', get_region_type(region['name'])),
            'display_name': region.get('display_name', get_region_display_name(region['name'])),
            'partition': region.get('partition', get_partition_from_region(region['name'])),
            'is_govcloud': region.get('is_govcloud', is_govcloud_region(region['name']))
        }
    else:
        # String format, convert to dict
        region_name = str(region)
        return {
            'name': region_name,
            'type': get_region_type(region_name),
            'display_name': get_region_display_name(region_name),
            'partition': get_partition_from_region(region_name),
            'is_govcloud': is_govcloud_region(region_name)
        }


def normalize_regions(regions: List) -> List[Dict[str, Any]]:
    """Normalize a list of regions to standard dictionary format
    
    Args:
        regions: List of regions (strings or dicts)
        
    Returns:
        List of dicts with standardized region information
    """
    return [normalize_region(region) for region in regions]


# ============================================================================
# Convenience Class (Backward Compatibility)
# ============================================================================

class PartitionDetector:
    """Utility class for AWS partition detection and region filtering
    
    This class provides a convenient interface for partition detection
    and is maintained for backward compatibility.
    """
    
    @staticmethod
    def detect_current_partition() -> Dict[str, Any]:
        """Detect AWS partition for current credentials"""
        return detect_partition_from_sts()
    
    @staticmethod
    def filter_regions_by_partition(regions: List[Dict[str, Any]], partition: str) -> List[Dict[str, Any]]:
        """Filter regions based on partition"""
        return filter_regions_by_partition(regions, partition)
    
    @staticmethod
    def get_partition_display_info(partition_result: Dict[str, Any]) -> Dict[str, Any]:
        """Get display information for partition"""
        return get_partition_display_info(partition_result)
    
    @staticmethod
    def should_auto_filter(partition_result: Dict[str, Any]) -> bool:
        """Determine if regions should be auto-filtered"""
        return should_auto_filter(partition_result)
    
    @staticmethod
    def get_detection_summary(partition_result: Dict[str, Any]) -> str:
        """Get formatted detection summary"""
        return get_detection_summary(partition_result)
