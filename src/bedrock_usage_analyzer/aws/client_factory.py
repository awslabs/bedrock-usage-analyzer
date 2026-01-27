# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enhanced AWS client factory with GovCloud support"""

import boto3
import logging
from typing import Dict, Any, Optional

from bedrock_usage_analyzer.utils.aws_partition import (
    is_govcloud_region,
    get_client_config,
    is_service_available,
    get_region_type,
    get_region_display_name,
    get_partition_from_region
)

logger = logging.getLogger(__name__)


class EnhancedClientFactory:
    """Enhanced AWS client factory with GovCloud support
    
    This factory creates properly configured AWS service clients for both
    standard AWS regions and GovCloud regions, handling endpoint configuration
    and authentication context appropriately.
    """
    
    def __init__(self, region: str):
        """Initialize the client factory for a specific region
        
        Args:
            region: AWS region name (e.g., 'us-east-1', 'us-gov-east-1')
        """
        self.region = region
        self.is_govcloud = is_govcloud_region(region)
        
        logger.debug(f"Initialized EnhancedClientFactory for region {region} "
                    f"(type: {'GovCloud' if self.is_govcloud else 'standard'})")
    
    def _get_client_config(self, service: str) -> Dict[str, Any]:
        """Get boto3 client configuration for the service
        
        Args:
            service: AWS service name
            
        Returns:
            Dict[str, Any]: Configuration arguments for boto3 client creation
        """
        config = get_client_config(service, self.region)
        
        # Log endpoint configuration for debugging
        if 'endpoint_url' in config:
            logger.debug(f"Using GovCloud endpoint for {service}: {config['endpoint_url']}")
        else:
            logger.debug(f"Using default endpoint for {service} in {self.region}")
            
        return config
    
    def _validate_service_availability(self, service: str) -> None:
        """Validate that the service is available in the current region
        
        Args:
            service: AWS service name
            
        Raises:
            ValueError: If service is not available in the region
        """
        if not is_service_available(service, self.region):
            region_type = "GovCloud" if self.is_govcloud else "standard"
            error_msg = (
                f"Service '{service}' is not available in {region_type} region '{self.region}'. "
                f"Please check AWS service availability documentation for {region_type} regions."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def create_bedrock_client(self) -> boto3.client:
        """Create Bedrock client with appropriate configuration
        
        Returns:
            boto3.client: Configured Bedrock client
            
        Raises:
            ValueError: If Bedrock is not available in the region
        """
        service = 'bedrock'
        self._validate_service_availability(service)
        
        try:
            config = self._get_client_config(service)
            client = boto3.client(service, **config)
            
            logger.debug(f"Created Bedrock client for {self.region}")
            return client
        except Exception as e:
            logger.error(f"Failed to create Bedrock client: {e}")
            raise
    
    def create_bedrock_runtime_client(self) -> boto3.client:
        """Create Bedrock Runtime client with appropriate configuration
        
        Returns:
            boto3.client: Configured Bedrock Runtime client
            
        Raises:
            ValueError: If Bedrock Runtime is not available in the region
        """
        service = 'bedrock-runtime'
        self._validate_service_availability(service)
        
        try:
            config = self._get_client_config(service)
            client = boto3.client(service, **config)
            
            logger.debug(f"Created Bedrock Runtime client for {self.region}")
            return client
        except Exception as e:
            logger.error(f"Failed to create Bedrock Runtime client: {e}")
            raise
    
    def create_cloudwatch_client(self) -> boto3.client:
        """Create CloudWatch client with appropriate configuration
        
        Returns:
            boto3.client: Configured CloudWatch client
            
        Raises:
            ValueError: If CloudWatch is not available in the region
        """
        service = 'cloudwatch'
        self._validate_service_availability(service)
        
        try:
            config = self._get_client_config(service)
            client = boto3.client(service, **config)
            
            logger.debug(f"Created CloudWatch client for {self.region}")
            return client
        except Exception as e:
            logger.error(f"Failed to create CloudWatch client: {e}")
            raise
    
    def create_service_quotas_client(self) -> boto3.client:
        """Create Service Quotas client with appropriate configuration
        
        Returns:
            boto3.client: Configured Service Quotas client
            
        Raises:
            ValueError: If Service Quotas is not available in the region
        """
        service = 'service-quotas'
        self._validate_service_availability(service)
        
        try:
            config = self._get_client_config(service)
            client = boto3.client(service, **config)
            
            logger.debug(f"Created Service Quotas client for {self.region}")
            return client
        except Exception as e:
            logger.error(f"Failed to create Service Quotas client: {e}")
            raise
    
    def create_account_client(self) -> boto3.client:
        """Create Account client with appropriate configuration
        
        Returns:
            boto3.client: Configured Account client
            
        Raises:
            ValueError: If Account service is not available in the region
        """
        service = 'account'
        self._validate_service_availability(service)
        
        try:
            config = self._get_client_config(service)
            client = boto3.client(service, **config)
            
            logger.debug(f"Created Account client for {self.region}")
            return client
        except Exception as e:
            logger.error(f"Failed to create Account client: {e}")
            raise
    
    def create_sts_client(self) -> boto3.client:
        """Create STS client with appropriate configuration
        
        Returns:
            boto3.client: Configured STS client
            
        Raises:
            ValueError: If STS is not available in the region
        """
        service = 'sts'
        self._validate_service_availability(service)
        
        try:
            config = self._get_client_config(service)
            client = boto3.client(service, **config)
            
            logger.debug(f"Created STS client for {self.region}")
            return client
        except Exception as e:
            logger.error(f"Failed to create STS client: {e}")
            raise
    
    def create_client(self, service: str) -> boto3.client:
        """Create a generic client for any AWS service
        
        Args:
            service: AWS service name
            
        Returns:
            boto3.client: Configured client for the service
            
        Raises:
            ValueError: If service is not available in the region
        """
        self._validate_service_availability(service)
        
        try:
            config = self._get_client_config(service)
            client = boto3.client(service, **config)
            
            logger.debug(f"Created {service} client for {self.region}")
            return client
        except Exception as e:
            logger.error(f"Failed to create {service} client: {e}")
            raise
    
    def get_region_info(self) -> Dict[str, Any]:
        """Get information about the current region configuration
        
        Returns:
            Dict[str, Any]: Region information including type and partition
        """
        return {
            'region': self.region,
            'type': get_region_type(self.region),
            'display_name': get_region_display_name(self.region),
            'partition': get_partition_from_region(self.region),
            'is_govcloud': self.is_govcloud
        }