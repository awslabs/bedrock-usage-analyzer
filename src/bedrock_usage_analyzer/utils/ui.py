# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive UI for quota mapping parameter selection"""

import logging
import sys
from typing import Tuple

logger = logging.getLogger(__name__)


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


def _claude_endpoints_in(region: str, limit: int = 12) -> list:
    """Invokable Claude endpoint IDs listed in the region's fm-list (Haiku first, then newest)."""
    from bedrock_usage_analyzer.aws.bedrock import endpoint_id
    from bedrock_usage_analyzer.sync.quota_rules import model_version
    from bedrock_usage_analyzer.utils.yaml_handler import endpoint_keys, load_fm_list
    import yaml
    options = []
    try:
        models = load_fm_list(region) or []
    except yaml.YAMLError as e:
        # The fixed fallback list is offered instead; fm-quotas then skips the region with a hint
        logger.warning(f"Could not read fm-list-{region}.yml ({e}); fix or delete it, "
                       f"or run: bua refresh fm-list {region}")
        models = []
    for model in models:
        model_id = model['model_id']
        if not model_id.startswith('anthropic.claude') or model_id.count(':') > 1:
            continue  # skip context-window variants such as ...-v1:0:200k
        for prefix in sorted(endpoint_keys(model)):
            options.append(endpoint_id(model_id, prefix))

    def newest_first(option):
        # Haiku first (mapping makes many small calls), then model generation (4.5 > 3.7 > 3.5),
        # then profile endpoints before base models
        is_profile = not option.startswith('anthropic.')
        version = model_version(option.split('.', 1)[1] if is_profile else option)
        numbers = tuple(int(x) for x in version.split('.')) if version else ()
        return ('haiku' in option, numbers, is_profile)

    return sorted(options, key=newest_first, reverse=True)[:limit]


def require_credentials_partition(bedrock_region: str, credentials_partition: str,
                                  label: str = 'Bedrock calls: ') -> None:
    """Exit when ``bedrock_region`` is outside the credentials' partition.

    e.g. GovCloud credentials with a commercial Bedrock region: every call would fail.
    The one partition gate of the CLI, the analyzer's region choice included.
    """
    from bedrock_usage_analyzer.utils.partition import partition_mismatch
    problem = partition_mismatch(bedrock_region, credentials_partition)
    if problem:
        logger.error(f"\n{label}{problem}")
        sys.exit(1)


def select_quota_mapping_params(target_region: str = None, bedrock_region: str = None, model_id: str = None,
                                resolved: dict = None) -> Tuple[str, str, str]:
    """Interactive selection for quota mapping parameters
    
    Args:
        target_region: Pre-filled target region (skips prompt if provided)
        bedrock_region: Pre-filled bedrock region (skips prompt if provided)
        model_id: Pre-filled model ID (skips prompt if provided)
        resolved: Optional dict; gets 'regions', the credentials' regions read here

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
    all_regions, partition = regions_for_credentials(load_region_names(), target_region or bedrock_region)
    if not all_regions:
        print("\nNo regions in regions.yml for these credentials. Run: bua refresh regions", file=sys.stderr)
        sys.exit(1)
    if resolved is not None:
        resolved['regions'] = all_regions  # reused by the quota mapper (no second lookup)
    if target_region is not None:
        # Before any prompt: a target region of another partition cannot be refreshed
        require_credentials_partition(target_region, partition, label='Target region: ')
        if target_region not in all_regions:
            # (the mapper rejects it too, but only after the Step 1 and 2 prompts)
            logger.error(f"Region '{target_region}' is not among the regions these credentials can call "
                         f"(see regions.yml; run 'bua refresh regions' if it is new)")
            sys.exit(1)

    # Step 1: Select Bedrock API region (skip if provided)
    if not bedrock_region:
        bedrock_region = select_from_list(
            "Step 1: Select AWS region to use for Bedrock API calls:",
            all_regions
        )
    else:
        require_credentials_partition(bedrock_region, partition)
    print(f"\n✓ Bedrock calls will use region: {bedrock_region}")

    # Step 2: Select model for mapping (skip if provided)
    if not model_id:
        # Only endpoints the chosen region serves (base model, geography or global profile);
        # the fixed list is a fallback for regions without a bundled model list
        model_options = _claude_endpoints_in(bedrock_region)
        if not model_options and get_partition_for_region(bedrock_region) == GOVCLOUD:
            model_options = ["us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0"]
        elif not model_options:
            model_options = [
                "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
                "au.anthropic.claude-haiku-4-5-20251001-v1:0",
                "jp.anthropic.claude-haiku-4-5-20251001-v1:0",
                "global.anthropic.claude-haiku-4-5-20251001-v1:0",
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
