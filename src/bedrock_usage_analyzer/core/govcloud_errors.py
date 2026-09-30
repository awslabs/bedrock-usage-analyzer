# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""GovCloud-specific error handling and enhancement"""

import logging
from typing import Dict, Any, Optional
from bedrock_usage_analyzer.utils.aws_partition import is_govcloud_region

logger = logging.getLogger(__name__)


class GovCloudErrorHandler:
    """Enhanced error handler for GovCloud-specific issues"""
    
    def __init__(self, region: str):
        """Initialize error handler for a specific region
        
        Args:
            region: AWS region name
        """
        self.region = region
        self.is_govcloud = is_govcloud_region(region)
        self.region_type = "GovCloud" if self.is_govcloud else "standard"
    
    def enhance_error_message(self, error: Exception, context: Dict[str, Any]) -> str:
        """Enhance error message with GovCloud-specific guidance
        
        Args:
            error: Original exception
            context: Context information (service, operation, etc.)
            
        Returns:
            str: Enhanced error message with troubleshooting guidance
        """
        error_str = str(error)
        error_type = type(error).__name__
        service = context.get('service', 'unknown')
        operation = context.get('operation', 'unknown')
        
        # Base error message
        enhanced_msg = f"[{self.region_type}] {error_type} in {service}.{operation}: {error_str}"
        
        # Add GovCloud-specific guidance
        if self.is_govcloud:
            enhanced_msg += self._get_govcloud_guidance(error, context)
        else:
            enhanced_msg += self._get_standard_guidance(error, context)
        
        return enhanced_msg
    
    def _get_govcloud_guidance(self, error: Exception, context: Dict[str, Any]) -> str:
        """Get GovCloud-specific troubleshooting guidance
        
        Args:
            error: Original exception
            context: Context information
            
        Returns:
            str: GovCloud-specific guidance
        """
        error_str = str(error).lower()
        service = context.get('service', '')
        
        guidance = "\n\n🏛️ GovCloud Troubleshooting:"
        
        # Credential-related errors
        if any(keyword in error_str for keyword in ['credentials', 'access denied', 'unauthorized', 'forbidden']):
            guidance += """
• Ensure you're using GovCloud-specific AWS credentials
• GovCloud credentials are separate from standard AWS credentials
• Verify your credentials have access to the selected GovCloud region
• Check that your account has appropriate security clearance for GovCloud
• Test credentials: AWS_PROFILE=govcloud aws sts get-caller-identity"""
        
        # Service availability errors
        elif any(keyword in error_str for keyword in ['service not available', 'endpoint', 'not found']):
            guidance += f"""
• Service '{service}' may not be available in GovCloud region '{self.region}'
• Check AWS GovCloud service availability documentation
• Some services have limited availability in GovCloud regions
• Verify the service endpoint is accessible from your network"""
        
        # Network/connectivity errors
        elif any(keyword in error_str for keyword in ['timeout', 'connection', 'network']):
            guidance += f"""
• Check network connectivity to GovCloud endpoints
• GovCloud uses different service endpoints than standard AWS
• Verify firewall rules allow access to {self.region} endpoints
• Test connectivity: curl -I https://bedrock.{self.region}.amazonaws.com"""
        
        # Model/resource not found
        elif any(keyword in error_str for keyword in ['model not found', 'resource not found']):
            guidance += """
• GovCloud may have different foundation models available
• Some models may not be available due to compliance requirements
• Refresh model lists for GovCloud: AWS_PROFILE=govcloud bedrock-usage-analyzer refresh fm-list
• Check GovCloud-specific model catalog"""
        
        # Generic GovCloud guidance
        else:
            guidance += f"""
• Verify you're using GovCloud-specific credentials for region '{self.region}'
• Check AWS GovCloud documentation for service-specific requirements
• Ensure your account has appropriate GovCloud access and security clearance
• Contact AWS GovCloud support if the issue persists"""
        
        guidance += f"""

📚 Resources:
• GovCloud Setup Guide: See GOVCLOUD_SETUP.md
• AWS GovCloud Documentation: https://docs.aws.amazon.com/govcloud-us/
• Service Availability: https://aws.amazon.com/govcloud-us/details/"""
        
        return guidance
    
    def _get_standard_guidance(self, error: Exception, context: Dict[str, Any]) -> str:
        """Get standard AWS troubleshooting guidance
        
        Args:
            error: Original exception
            context: Context information
            
        Returns:
            str: Standard troubleshooting guidance
        """
        error_str = str(error).lower()
        service = context.get('service', '')
        
        guidance = "\n\n🔧 Troubleshooting:"
        
        # Credential-related errors
        if any(keyword in error_str for keyword in ['credentials', 'access denied', 'unauthorized']):
            guidance += """
• Check your AWS credentials configuration
• Verify IAM permissions for the required services
• Test credentials: aws sts get-caller-identity
• Ensure your credentials have access to the selected region"""
        
        # Service availability errors
        elif any(keyword in error_str for keyword in ['service not available', 'endpoint']):
            guidance += f"""
• Service '{service}' may not be available in region '{self.region}'
• Check AWS service availability documentation
• Try a different region where the service is available"""
        
        # Network/connectivity errors
        elif any(keyword in error_str for keyword in ['timeout', 'connection', 'network']):
            guidance += """
• Check your internet connection
• Verify network access to AWS endpoints
• Check firewall and proxy settings"""
        
        # Generic guidance
        else:
            guidance += """
• Verify your AWS credentials and permissions
• Check the AWS service status page
• Try again in a few minutes"""
        
        return guidance


def create_govcloud_error_handler(region: str) -> GovCloudErrorHandler:
    """Create a GovCloud-aware error handler for the specified region
    
    Args:
        region: AWS region name
        
    Returns:
        GovCloudErrorHandler: Configured error handler
    """
    return GovCloudErrorHandler(region)