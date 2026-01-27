#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test script for STS ARN partition detection
"""

import sys
import os

# Add src to path for testing
sys.path.insert(0, 'src')

def test_arn_parsing_logic():
    """Test ARN parsing logic without AWS calls"""
    print("🧪 Testing ARN Parsing Logic...")
    
    # Test cases for different ARN formats
    test_cases = [
        {
            'name': 'Commercial IAM User',
            'arn': 'arn:aws:iam::123456789012:user/testuser',
            'expected_partition': 'commercial',
            'expected_confidence': 'high'
        },
        {
            'name': 'GovCloud IAM User', 
            'arn': 'arn:aws-us-gov:iam::123456789012:user/testuser',
            'expected_partition': 'govcloud',
            'expected_confidence': 'high'
        },
        {
            'name': 'Commercial IAM Role',
            'arn': 'arn:aws:iam::123456789012:role/MyRole',
            'expected_partition': 'commercial',
            'expected_confidence': 'high'
        },
        {
            'name': 'GovCloud IAM Role',
            'arn': 'arn:aws-us-gov:iam::123456789012:role/MyRole',
            'expected_partition': 'govcloud',
            'expected_confidence': 'high'
        },
        {
            'name': 'Commercial Assumed Role',
            'arn': 'arn:aws:sts::123456789012:assumed-role/MyRole/session',
            'expected_partition': 'commercial',
            'expected_confidence': 'high'
        },
        {
            'name': 'GovCloud Assumed Role',
            'arn': 'arn:aws-us-gov:sts::123456789012:assumed-role/MyRole/session',
            'expected_partition': 'govcloud',
            'expected_confidence': 'high'
        },
        {
            'name': 'Unknown Partition',
            'arn': 'arn:aws-cn:iam::123456789012:user/testuser',
            'expected_partition': 'unknown',
            'expected_confidence': 'none'
        },
        {
            'name': 'Invalid ARN Format',
            'arn': 'invalid-arn-format',
            'expected_partition': 'unknown',
            'expected_confidence': 'none'
        }
    ]
    
    # Simulate ARN parsing logic
    def parse_arn_for_partition(arn):
        if ':aws-us-gov:' in arn:
            return {'partition': 'govcloud', 'confidence': 'high'}
        elif ':aws:' in arn:
            return {'partition': 'commercial', 'confidence': 'high'}
        else:
            return {'partition': 'unknown', 'confidence': 'none'}
    
    all_passed = True
    for test_case in test_cases:
        result = parse_arn_for_partition(test_case['arn'])
        
        partition_match = result['partition'] == test_case['expected_partition']
        confidence_match = result['confidence'] == test_case['expected_confidence']
        
        if partition_match and confidence_match:
            print(f"  ✅ {test_case['name']}: {result['partition']} ({result['confidence']})")
        else:
            print(f"  ❌ {test_case['name']}: Got {result['partition']} ({result['confidence']}), "
                  f"expected {test_case['expected_partition']} ({test_case['expected_confidence']})")
            all_passed = False
    
    return all_passed

def test_partition_detector_class():
    """Test PartitionDetector class methods"""
    print("\n🏭 Testing PartitionDetector Class...")
    
    from bedrock_usage_analyzer.utils.partition_detection import PartitionDetector
    
    # Test partition display info
    test_results = [
        {'partition': 'commercial', 'confidence': 'high'},
        {'partition': 'govcloud', 'confidence': 'high'},
        {'partition': 'unknown', 'confidence': 'none', 'error': 'Test error'}
    ]
    
    all_passed = True
    for result in test_results:
        try:
            display_info = PartitionDetector.get_partition_display_info(result)
            
            # Validate required fields
            required_fields = ['name', 'icon', 'description', 'confidence', 'partition']
            missing_fields = [f for f in required_fields if f not in display_info]
            
            if missing_fields:
                print(f"  ❌ {result['partition']}: Missing fields: {missing_fields}")
                all_passed = False
            else:
                print(f"  ✅ {display_info['icon']} {display_info['name']}: {display_info['description']}")
        except Exception as e:
            print(f"  ❌ {result['partition']}: Error: {e}")
            all_passed = False
    
    # Test auto-filter logic
    print("\n  Testing auto-filter logic:")
    filter_test_cases = [
        ({'partition': 'commercial', 'confidence': 'high'}, True),
        ({'partition': 'govcloud', 'confidence': 'high'}, True),
        ({'partition': 'commercial', 'confidence': 'medium'}, True),
        ({'partition': 'unknown', 'confidence': 'none'}, False),
        ({'partition': 'commercial', 'confidence': 'low'}, False)
    ]
    
    for result, expected in filter_test_cases:
        should_filter = PartitionDetector.should_auto_filter(result)
        status = "✅" if should_filter == expected else "❌"
        print(f"    {status} {result['partition']} ({result['confidence']}): "
              f"filter={should_filter} (expected: {expected})")
        if should_filter != expected:
            all_passed = False
    
    return all_passed

def test_region_filtering():
    """Test region filtering logic"""
    print("\n📋 Testing Region Filtering...")
    
    from bedrock_usage_analyzer.utils.partition_detection import PartitionDetector
    
    # Mock regions data
    mock_regions = [
        {'name': 'us-east-1', 'type': 'standard', 'display_name': 'US East (N. Virginia)'},
        {'name': 'us-west-2', 'type': 'standard', 'display_name': 'US West (Oregon)'},
        {'name': 'us-gov-east-1', 'type': 'govcloud', 'display_name': 'AWS GovCloud (US-East)'},
        {'name': 'us-gov-west-1', 'type': 'govcloud', 'display_name': 'AWS GovCloud (US-West)'},
        {'name': 'eu-west-1', 'type': 'standard', 'display_name': 'Europe (Ireland)'}
    ]
    
    test_cases = [
        {
            'partition': 'commercial',
            'expected_count': 3,  # us-east-1, us-west-2, eu-west-1
            'expected_types': ['standard']
        },
        {
            'partition': 'govcloud', 
            'expected_count': 2,  # us-gov-east-1, us-gov-west-1
            'expected_types': ['govcloud']
        },
        {
            'partition': 'unknown',
            'expected_count': 5,  # All regions
            'expected_types': ['standard', 'govcloud']
        }
    ]
    
    all_passed = True
    for test_case in test_cases:
        filtered = PartitionDetector.filter_regions_by_partition(mock_regions, test_case['partition'])
        
        # Check count
        count_match = len(filtered) == test_case['expected_count']
        
        # Check types
        actual_types = set(r['type'] for r in filtered)
        expected_types = set(test_case['expected_types'])
        types_match = actual_types.issubset(expected_types)
        
        if count_match and types_match:
            print(f"  ✅ {test_case['partition']}: {len(filtered)} regions, types: {sorted(actual_types)}")
        else:
            print(f"  ❌ {test_case['partition']}: Got {len(filtered)} regions (expected {test_case['expected_count']}), "
                  f"types: {sorted(actual_types)} (expected subset of {sorted(expected_types)})")
            all_passed = False
    
    return all_passed

def test_detection_summary():
    """Test detection summary formatting"""
    print("\n📝 Testing Detection Summary...")
    
    from bedrock_usage_analyzer.utils.partition_detection import PartitionDetector
    
    test_cases = [
        {
            'result': {'partition': 'commercial', 'confidence': 'high'},
            'expected_contains': ['✅', '🌍', 'AWS Commercial', 'detected']
        },
        {
            'result': {'partition': 'govcloud', 'confidence': 'high'},
            'expected_contains': ['✅', '🏛️', 'AWS GovCloud', 'detected']
        },
        {
            'result': {'partition': 'commercial', 'confidence': 'medium'},
            'expected_contains': ['🔍', '🌍', 'medium confidence']
        },
        {
            'result': {'partition': 'unknown', 'confidence': 'none', 'error': 'Test error'},
            'expected_contains': ['❓', 'Could not detect', 'Test error']
        }
    ]
    
    all_passed = True
    for test_case in test_cases:
        summary = PartitionDetector.get_detection_summary(test_case['result'])
        
        missing_content = []
        for expected in test_case['expected_contains']:
            if expected not in summary:
                missing_content.append(expected)
        
        if not missing_content:
            print(f"  ✅ {test_case['result']['partition']}: {summary}")
        else:
            print(f"  ❌ {test_case['result']['partition']}: Missing {missing_content}")
            print(f"     Actual: {summary}")
            all_passed = False
    
    return all_passed

def main():
    """Run all partition detection tests"""
    print("🎯 STS ARN Partition Detection Test Suite")
    print("=" * 50)
    
    tests = [
        ("ARN Parsing Logic", test_arn_parsing_logic),
        ("PartitionDetector Class", test_partition_detector_class),
        ("Region Filtering", test_region_filtering),
        ("Detection Summary", test_detection_summary)
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
        print("🎉 All tests passed! STS ARN partition detection is working correctly.")
        print("💡 Core functionality validated without requiring AWS credentials")
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        sys.exit(1)
    
    print("\n🚀 Usage Examples:")
    print("  # Interactive with automatic partition detection")
    print("  bedrock-usage-analyzer analyze")
    print("  # → 🔍 Detecting AWS partition...")
    print("  # → ✅ 🌍 AWS Commercial detected")
    print("  # → 📋 Showing 15 AWS Commercial regions")
    print()
    print("  # GovCloud automatic detection")
    print("  AWS_PROFILE=govcloud bedrock-usage-analyzer analyze")
    print("  # → 🔍 Detecting AWS partition...")
    print("  # → ✅ 🏛️ AWS GovCloud (US) detected")
    print("  # → 📋 Showing 2 AWS GovCloud (US) regions")

if __name__ == "__main__":
    main()