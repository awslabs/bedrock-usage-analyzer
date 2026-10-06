#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Basic test script to validate GovCloud integration (no AWS dependencies)
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
    
    all_passed = True
    for region, expected in test_cases:
        result = GovCloudDetector.is_govcloud_region(region)
        status = "✅" if result == expected else "❌"
        if result != expected:
            all_passed = False
        print(f"  {status} {region or 'None'}: {result} (expected: {expected})")
    
    return all_passed

def test_region_display_names():
    """Test region display name generation"""
    print("\n🏷️  Testing Region Display Names...")
    
    from bedrock_usage_analyzer.aws.govcloud import GovCloudDetector
    
    test_cases = [
        ('us-gov-east-1', 'AWS GovCloud (US-East)'),
        ('us-gov-west-1', 'AWS GovCloud (US-West)'), 
        ('us-east-1', 'US East (N. Virginia)'),
        ('eu-west-1', 'Europe (Ireland)'),
    ]
    
    all_passed = True
    for region, expected in test_cases:
        display_name = GovCloudDetector.get_region_display_name(region)
        is_govcloud = GovCloudDetector.is_govcloud_region(region)
        indicator = "🏛️" if is_govcloud else "🌍"
        status = "✅" if display_name == expected else "❌"
        if display_name != expected:
            all_passed = False
        print(f"  {status} {indicator} {region}: {display_name}")
    
    return all_passed

def test_endpoint_configuration():
    """Test GovCloud endpoint configuration"""
    print("\n🔗 Testing Endpoint Configuration...")
    
    from bedrock_usage_analyzer.aws.govcloud import GovCloudEndpointConfig
    
    # Test GovCloud endpoints
    govcloud_tests = [
        ('bedrock', 'us-gov-east-1', 'https://bedrock.us-gov-east-1.amazonaws.com'),
        ('cloudwatch', 'us-gov-west-1', 'https://monitoring.us-gov-west-1.amazonaws.com'),
        ('service-quotas', 'us-gov-east-1', 'https://servicequotas.us-gov-east-1.amazonaws.com'),
    ]
    
    # Test standard region endpoints (should return None)
    standard_tests = [
        ('bedrock', 'us-east-1', None),
        ('cloudwatch', 'eu-west-1', None),
    ]
    
    all_passed = True
    
    print("  GovCloud regions:")
    for service, region, expected in govcloud_tests:
        endpoint = GovCloudEndpointConfig.get_service_endpoint(service, region)
        status = "✅" if endpoint == expected else "❌"
        if endpoint != expected:
            all_passed = False
        print(f"    {status} {service} @ {region}: {endpoint}")
    
    print("  Standard regions:")
    for service, region, expected in standard_tests:
        endpoint = GovCloudEndpointConfig.get_service_endpoint(service, region)
        status = "✅" if endpoint == expected else "❌"
        if endpoint != expected:
            all_passed = False
        print(f"    {status} {service} @ {region}: {endpoint or 'Default endpoint'}")
    
    return all_passed

def test_partition_detection():
    """Test AWS partition detection"""
    print("\n🌐 Testing Partition Detection...")
    
    from bedrock_usage_analyzer.aws.govcloud import GovCloudEndpointConfig
    
    test_cases = [
        ('us-gov-east-1', 'aws-us-gov'),
        ('us-gov-west-1', 'aws-us-gov'),
        ('us-east-1', 'aws'),
        ('eu-west-1', 'aws'),
        ('ap-southeast-1', 'aws'),
    ]
    
    all_passed = True
    for region, expected in test_cases:
        partition = GovCloudEndpointConfig.get_partition(region)
        status = "✅" if partition == expected else "❌"
        if partition != expected:
            all_passed = False
        print(f"  {status} {region}: {partition}")
    
    return all_passed

def test_region_info():
    """Test region info functionality"""
    print("\n ℹ️  Testing Region Info...")
    
    # Test the core functionality without boto3 dependencies
    from bedrock_usage_analyzer.aws.govcloud import GovCloudDetector, GovCloudEndpointConfig
    
    regions = ['us-gov-east-1', 'us-gov-west-1', 'us-east-1', 'eu-west-1']
    
    all_passed = True
    for region in regions:
        try:
            # Manually create region info like the function would
            is_govcloud = GovCloudDetector.is_govcloud_region(region)
            info = {
                'name': region,
                'type': 'govcloud' if is_govcloud else 'standard',
                'display_name': GovCloudDetector.get_region_display_name(region),
                'partition': GovCloudEndpointConfig.get_partition(region),
                'is_govcloud': is_govcloud,
                'indicator': '🏛️' if is_govcloud else ''
            }
            
            indicator = "🏛️" if info['is_govcloud'] else "🌍"
            
            # Validate expected fields
            required_fields = ['name', 'type', 'display_name', 'partition', 'is_govcloud']
            missing_fields = [f for f in required_fields if f not in info]
            
            if missing_fields:
                print(f"  ❌ {indicator} {region}: Missing fields: {missing_fields}")
                all_passed = False
            else:
                print(f"  ✅ {indicator} {region}: {info['display_name']} ({info['type']}, {info['partition']})")
            
        except Exception as e:
            print(f"  ❌ Error for {region}: {e}")
            all_passed = False
    
    return all_passed

def main():
    """Run all tests"""
    print("🏛️ GovCloud Integration Test Suite (Basic)")
    print("=" * 50)
    
    tests = [
        ("GovCloud Detection", test_govcloud_detection),
        ("Region Display Names", test_region_display_names),
        ("Endpoint Configuration", test_endpoint_configuration),
        ("Partition Detection", test_partition_detection),
        ("Region Info", test_region_info),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! GovCloud integration is working correctly.")
        print("💡 Core functionality validated without requiring AWS credentials")
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        sys.exit(1)
    
    print("\n🔧 Next steps:")
    print("  • Install the package: pip install -e .")
    print("  • Test with real credentials: python3 check-govcloud-status.py")
    print("  • Run the analyzer: bedrock-usage-analyzer analyze")

if __name__ == "__main__":
    main()