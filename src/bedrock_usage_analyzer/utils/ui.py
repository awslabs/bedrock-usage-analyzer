# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive UI for quota mapping parameter selection"""

import sys
from typing import Tuple



def select_from_list(
    prompt: str, 
    options: list, 
    allow_cancel: bool = True,
    display_fn=None,
    input_prompt: str = None
) -> str:
    """Generic numbered selection from list

    Args:
        prompt: Prompt message
        options: List of options
        allow_cancel: Allow cancellation with Ctrl+C
        display_fn: Optional function to format each option for display
        input_prompt: Optional custom input prompt (default: "Select (1-N):")

    Returns:
        Selected option
    """
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        display_text = display_fn(option) if display_fn else str(option)
        print(f"  {i}. {display_text}")
    
    default_prompt = f"\nSelect (1-{len(options)}): "
    actual_prompt = input_prompt if input_prompt else default_prompt
    
    while True:
        try:
            choice = int(input(actual_prompt))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            print(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("Please enter a valid number")
        except (KeyboardInterrupt, EOFError):
            if allow_cancel:
                print("\nSelection cancelled.", file=sys.stderr)
                sys.exit(1)
            raise


def _claude_endpoints_in(region: str) -> list:
    """Invokable Claude endpoint IDs listed in the region's fm-list, newest first."""
    import os
    from bedrock_usage_analyzer.utils.paths import get_data_path
    from bedrock_usage_analyzer.utils.yaml_handler import load_yaml
    path = get_data_path(f'fm-list-{region}.yml')
    if not os.path.exists(path):
        return []
    options = []
    for model in (load_yaml(path) or {}).get('models', []) or []:
        model_id = model.get('model_id', '')
        if not model_id.startswith('anthropic.claude') or model_id.count(':') > 1:
            continue  # skip context-window variants such as ...-v1:0:200k
        endpoints = model.get('endpoints') or {}
        for prefix in ('us-gov', 'global'):
            if prefix in endpoints:
                options.append(f"{prefix}.{model_id}")
        if 'base' in endpoints:
            options.append(model_id)
    return sorted(options, reverse=True)


def select_quota_mapping_params(target_region: str = None, bedrock_region: str = None, model_id: str = None) -> Tuple[str, str, str]:
    """Interactive selection for quota mapping parameters
    
    Args:
        target_region: Pre-filled target region (skips prompt if provided)
        bedrock_region: Pre-filled bedrock region (skips prompt if provided)
        model_id: Pre-filled model ID (skips prompt if provided)
    
    Returns:
        Tuple of (bedrock_region, model_id, target_region)
    """
    print("\n" + "="*60)
    print("Foundation Model Quota Mapping Tool")
    print("="*60)
    print("\nThis tool will:")
    print("  • Process ALL enabled regions automatically")
    print("  • Use a Bedrock LLM to intelligently map service quotas")
    print("  • Cache L-codes (same across regions)")
    print("="*60)
    
    # Show target region if provided
    if target_region is not None:
        print(f"\n✓ Using target region '{target_region}' as per input")
    
    # Load regions the current credentials can call (commercial or GovCloud, etc.)
    from bedrock_usage_analyzer.sync.regions import load_region_names, regions_for_credentials
    from bedrock_usage_analyzer.utils.partition import GOVCLOUD, get_partition_for_region
    # Any region the user already named pins STS to the right partition (GovCloud without AWS_REGION)
    all_regions, _ = regions_for_credentials(load_region_names(), bedrock_region or target_region)
    if not all_regions:
        print("\nNo regions in regions.yml for these credentials. Run: bua refresh regions", file=sys.stderr)
        sys.exit(1)

    # Step 1: Select Bedrock API region (skip if provided)
    if not bedrock_region:
        bedrock_region = select_from_list(
            "Step 1: Select AWS region to use for Bedrock API calls:",
            all_regions
        )
    print(f"\n✓ Bedrock calls will use region: {bedrock_region}")

    # Step 2: Select model for mapping (skip if provided)
    if not model_id:
        if get_partition_for_region(bedrock_region) == GOVCLOUD:
            # What each GovCloud region serves differs (base model in one, only a us-gov.
            # profile in the other), so offer the endpoints its own model list shows
            model_options = _claude_endpoints_in(bedrock_region) or [
                "us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0",
            ]
        else:
            model_options = [
                "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
                "au.anthropic.claude-haiku-4-5-20251001-v1:0",
                "jp.anthropic.claude-haiku-4-5-20251001-v1:0",
                "global.anthropic.claude-haiku-4-5-20251001-v1:0",
                "anthropic.claude-3-5-sonnet-20241022-v2:0",
                "anthropic.claude-3-5-haiku-20241022-v1:0"
            ]
        
        model_id = select_from_list(
            "Step 2: Select Claude model to use for intelligent mapping:",
            model_options
        )
    print(f"\n✓ Will use model: {model_id}")
    
    # Step 3: Optional target region filter (skip if provided)
    if target_region is None:
        print("\nStep 3: Target region filter (optional)")
        print("  1. Process ALL regions")
        print("  2. Process specific region only")
        
        while True:
            try:
                choice = int(input("\nSelect (1-2): "))
                if choice == 1:
                    target_region = None
                    print("\n✓ Will process all regions")
                    break
                elif choice == 2:
                    target_region = select_from_list(
                        "Select target region:",
                        all_regions
                    )
                    print(f"\n✓ Will process only: {target_region}")
                    break
                else:
                    print("Please enter 1 or 2")
            except ValueError:
                print("Please enter a valid number")
            except (KeyboardInterrupt, EOFError):
                print("\nSelection cancelled.", file=sys.stderr)
                sys.exit(1)
    
    return bedrock_region, model_id, target_region


def main():
    """Main entry point"""
    return select_quota_mapping_params()


if __name__ == "__main__":
    main()
