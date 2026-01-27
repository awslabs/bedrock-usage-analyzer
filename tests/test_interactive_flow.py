#!/usr/bin/env python3
"""
Test the interactive flow to see partition detection in action
"""

import sys
import os

# Add src to path for testing
sys.path.insert(0, 'src')

def test_interactive_flow():
    """Test the user input flow with partition detection"""
    from bedrock_usage_analyzer.core.user_inputs import UserInputs
    
    print("🎯 Testing Interactive Flow with Partition Detection")
    print("=" * 60)
    
    # Check current AWS profile
    current_profile = os.environ.get('AWS_PROFILE', 'default')
    print(f"Current AWS Profile: {current_profile}")
    
    try:
        user_inputs = UserInputs()
        
        # Test just the region selection part (which includes partition detection)
        print("\n🔍 Testing Region Selection with Partition Detection...")
        print("This will show the automatic partition detection in action:")
        print()
        
        # This will trigger the partition detection and region filtering
        region = user_inputs._select_region()
        
        print(f"\n✅ Selected region: {region}")
        print("🎉 Partition detection and region filtering worked!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Test cancelled by user")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_interactive_flow()