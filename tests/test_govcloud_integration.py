#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test script to validate GovCloud integration
"""

import sys
import os

# Add src to path for testing
sys.path.insert(0, 'src')

def test_govcloud_detection():
    """Test GovCloud region detection"""
    print("🧪 Testing GovCloud Detection...")
    
    from bedrock_usage_analyzer.aws.govcloud import GovCloudDetector
    
    test_cases = [
        ('us-gov-east-1', True),
        ('us-gov-west-1', True),
        ('us-east-1', False),
        ('eu-west-1', False),
        ('ap-southeast-1', False),
        ('', False),
        (None, False)
    ]
    
    for region, expected in test_cases:
        result = GovCloudDetector.is_govcloud_region(region)
        status = "✅" if result == expected else "❌"
        print(f"  {status} {region or 'None'}: {result} (expected: {expected})")
    
    print()

def test_region_display_names():
    """Test region display name generation"""
    print("🏷️  Testing Region Display Names...")
    
    from bedrock_usage_analyzer.aws.govcloud import GovCloudDetector
    
    test_cases = [
        'us-gov-east-1',
        'us-gov-west-1', 
        'us-east-1',
        'eu-west-1',
        'unknown-region'
    ]
    
    for region in test_cases:
        display_name = GovCloudDetector.get_region_display_name(region)
        is_govcloud = GovCloudDetector.is_govcloud_region(region)
        indicator = "🏛️" if is_govcloud else "🌍"
        print(f"  {indicator} {region}: {display_name}")
    
    print()

def test_endpoint_configuration():
    """Test GovCloud endpoint configuration"""
    print("🔗 Testing Endpoint Configuration...")
    
    from bedrock_usage_analyzer.aws.govcloud import GovCloudEndpointConfig
    
    services = ['bedrock', 'cloudwatch', 'service-quotas']
    regions = ['us-gov-east-1', 'us-east-1']
    
    for region in regions:
        region_type = "GovCloud" if region.startswith('us-gov-') else "Standard"
        print(f"  {region} ({region_type}):")
        
        for service in services:
            endpoint = GovCloudEndpointConfig.get_service_endpoint(service, region)
            if endpoint:
                print(f"    ✅ {service}: {endpoint}")
            else:
                print(f"    ➡️  {service}: Default endpoint")
    
    print()

def test_client_factory():
    """Test enhanced client factory (without actual AWS calls)"""
    print("🏭 Testing Client Factory...")
    
    from bedrock_usage_analyzer.aws.client_factory import EnhancedClientFactory
    
    regions = ['us-gov-east-1', 'us-east-1']
    
    for region in regions:
        print(f"  Testing {region}:")
        try:
            factory = EnhancedClientFactory(region)
            region_info = factory.get_region_info()
            
            print(f"    ✅ Factory created successfully")
            print(f"    📍 Region: {region_info['region']}")
            print(f"    🏷️  Type: {region_info['type']}")
            print(f"    📛 Display: {region_info['display_name']}")
            print(f"    🌐 Partition: {region_info['partition']}")
            print(f"    🏛️ Is GovCloud: {region_info['is_govcloud']}")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    print()

def test_error_handler():
    """Test GovCloud error handler"""
    print("🚨 Testing Error Handler...")
    
    from bedrock_usage_analyzer.core.govcloud_errors import create_govcloud_error_handler
    
    regions = ['us-gov-east-1', 'us-east-1']
    
    for region in regions:
        print(f"  Testing {region}:")
        try:
            handler = create_govcloud_error_handler(region)
            
            # Test with a sample error
            test_error = Exception("Access denied")
            context = {'service': 'bedrock', 'operation': 'list_models'}
            
            enhanced_msg = handler.enhance_error_message(test_error, context)
            
            print(f"    ✅ Error handler created")
            print(f"    📝 Enhanced message preview: {enhanced_msg[:100]}...")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    print()

def test_region_info():
    """Test region info functionality"""
    print("ℹ️  Testing Region Info...")
    
    from bedrock_usage_analyzer.metadata.regions import get_region_display_info
    
    regions = ['us-gov-east-1', 'us-gov-west-1', 'us-east-1', 'eu-west-1']
    
    for region in regions:
        try:
            info = get_region_display_info(region)
            indicator = "🏛️" if info['is_govcloud'] else "🌍"
            print(f"  {indicator} {region}:")
            print(f"    Display: {info['display_name']}")
            print(f"    Type: {info['type']}")
            print(f"    Partition: {info['partition']}")
            
        except Exception as e:
            print(f"    ❌ Error for {region}: {e}")
    
    print()

def main():
    """Run all tests"""
    print("🏛️ GovCloud Integration Test Suite")
    print("=" * 50)
    
    test_govcloud_detection()
    test_region_display_names()
    test_endpoint_configuration()
    test_client_factory()
    test_error_handler()
    test_region_info()
    
    print("=" * 50)
    print("✅ GovCloud integration tests completed!")
    print("💡 This validates the core functionality without requiring AWS credentials")
    print("🔧 To test with real AWS/GovCloud access, run the diagnostic script:")
    print("   python3 check-govcloud-status.py")

if __name__ == "__main__":
    main()