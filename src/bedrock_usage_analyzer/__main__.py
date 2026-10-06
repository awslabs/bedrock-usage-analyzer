# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified CLI entry point for bedrock-usage-analyzer."""

import sys
import logging
import traceback
import argparse

from bedrock_usage_analyzer.utils.paths import (
    get_metadata_location_message,
    get_refresh_location_message,
    get_writable_path,
    get_bundle_path,
    use_checkout_metadata,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def _parse_granularity(granularity_arg):
    """Parse granularity argument - single value or JSON.
    
    Args:
        granularity_arg: Either a single value (e.g., '1min') or JSON string
        
    Returns:
        dict: Granularity config with keys 1hour, 1day, 7days, 14days, 30days
        
    Raises:
        ValueError: If format is invalid or incomplete
    """
    import json
    
    GRANULARITY_MAP = {
        '1min': 60,
        '5min': 300,
        '1hour': 3600
    }
    
    TIME_PERIODS = ['1hour', '1day', '7days', '14days', '30days']
    
    # Try parsing as JSON first
    if granularity_arg.startswith('{'):
        try:
            parsed = json.loads(granularity_arg)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        
        # Validate all keys are present
        missing = [k for k in TIME_PERIODS if k not in parsed]
        if missing:
            raise ValueError(
                f"Incomplete granularity config. Missing: {', '.join(missing)}\n"
                f"Either specify a single value (e.g., -g 1min) for all periods,\n"
                f"or provide complete JSON with all keys: {', '.join(TIME_PERIODS)}\n"
                f"Example: -g '{{\"1hour\":\"1min\",\"1day\":\"5min\",\"7days\":\"1hour\",\"14days\":\"1hour\",\"30days\":\"1hour\"}}'"
            )
        
        # Convert string values to seconds
        config = {}
        for period in TIME_PERIODS:
            val = parsed[period]
            if val not in GRANULARITY_MAP:
                raise ValueError(f"Invalid granularity '{val}' for {period}. Must be one of: {', '.join(GRANULARITY_MAP.keys())}")
            config[period] = GRANULARITY_MAP[val]
        
        return config
    
    # Single value - apply to all periods
    if granularity_arg not in GRANULARITY_MAP:
        raise ValueError(f"Invalid granularity '{granularity_arg}'. Must be one of: {', '.join(GRANULARITY_MAP.keys())}")
    
    seconds = GRANULARITY_MAP[granularity_arg]
    return {period: seconds for period in TIME_PERIODS}


def cmd_analyze(args):
    """Run usage analysis."""
    from bedrock_usage_analyzer.core.user_inputs import UserInputs
    from bedrock_usage_analyzer.core.analyzer import BedrockAnalyzer
    
    print(get_metadata_location_message())
    print()
    
    # Parse granularity if provided
    granularity_config = None
    if args.granularity:
        try:
            granularity_config = _parse_granularity(args.granularity)
        except ValueError as e:
            logger.error(f"Error: {e}")
            sys.exit(1)
    
    user_inputs = UserInputs()
    try:
        user_inputs.collect(
            region=args.region,
            model_id=args.model_id,
            granularity_config=granularity_config,
            skip_confirm=args.yes
        )
    finally:
        # A region picked from the menu is not in args: keep it for the error hint in main(),
        # also when collect() fails after the pick
        args.region = args.region or user_inputs.region

    if not user_inputs.models:
        logger.error("No model selected; nothing to analyze.")
        sys.exit(1)

    # Get output directory (from arg, prompt, or default)
    output_dir = args.output_dir if args.output_dir else user_inputs.select_output_dir()

    analyzer = BedrockAnalyzer(user_inputs.region, user_inputs.granularity_config,
                               profile_fetcher=user_inputs.profile_fetcher,
                               fm_models=user_inputs.fm_models())
    analyzer.analyze(user_inputs.models, output_dir=output_dir)
    
    logger.info(f"\nCompleted! Results saved to: {output_dir}")


def _require_checkout(args):
    """The checkout's metadata path with --update-bundle (None without the flag).

    Exits before any AWS call when the flag is given outside a checkout, for every refresh
    command: otherwise the run would update user copies only and report success.
    """
    if not getattr(args, 'update_bundle', False):
        return None
    bundle_path = get_bundle_path()
    if bundle_path is None:
        _exit_not_in_checkout()
    # Every metadata read and write of the run (region lists, fm-lists, the picker, the
    # quota-index file listing) resolves to the checkout, not user copies or the installed package
    use_checkout_metadata(bundle_path.resolve())
    return bundle_path


def _exit_not_in_checkout():
    logger.error("\nError: --update-bundle requires a development environment.")
    logger.error("       Could not find: ./src/bedrock_usage_analyzer/metadata/")
    logger.error("\nThis flag is for maintainers in a cloned repository.")
    sys.exit(1)


def cmd_refresh_regions(args):
    """Refresh regions list."""
    from bedrock_usage_analyzer.sync.regions import discover_regions, read_region_file, refresh_regions
    from bedrock_usage_analyzer.utils.yaml_handler import save_yaml
    bundle_path = _require_checkout(args)

    print(get_refresh_location_message())
    print()

    # Only the credentials' partition is replaced; other partitions are kept
    # (e.g. GovCloud regions when refreshing with commercial credentials)
    discovered = discover_regions()
    if bundle_path is not None:
        # Maintainer mode writes the checkout only, as fm-list, fm-quotas and quota-index do:
        # a user copy written here would hide regions bundled in later releases
        bundle_file = bundle_path / "regions.yml"
        save_yaml(str(bundle_file), refresh_regions(existing=read_region_file(bundle_file), discovered=discovered))
        logger.info(f"✓ Saved: {bundle_file} (bundled)")
        return
    output_path = get_writable_path("regions.yml")
    # Merge into the user's own file only: copying bundled regions of other partitions into it
    # would freeze them against later releases (load_region_names adds them at read time)
    data = refresh_regions(existing=read_region_file(output_path), discovered=discovered)
    save_yaml(str(output_path), data)
    logger.info(f"✓ Saved: {output_path}")


def cmd_refresh_fm_list(args):
    """Refresh FM lists."""
    from bedrock_usage_analyzer.sync.fm_list import refresh_region, refresh_all_regions
    _require_checkout(args)

    print(get_refresh_location_message())
    print()
    
    from bedrock_usage_analyzer.sync.regions import load_region_names, regions_for_credentials
    from bedrock_usage_analyzer.utils.partition import is_valid_region_name

    if args.region:
        if not is_valid_region_name(args.region):
            logger.error(f"Invalid region format: {args.region}")
            sys.exit(1)
        # Same partition check as the other refresh commands (GovCloud region, commercial
        # credentials): a plain explanation instead of an 'invalid token' API error
        from bedrock_usage_analyzer.sync.regions import credentials_partition_or_exit
        from bedrock_usage_analyzer.utils.ui import require_credentials_partition
        require_credentials_partition(args.region, credentials_partition_or_exit(args.region))
        refresh_region(args.region, update_bundle=args.update_bundle)
    else:
        # Only the regions the current credentials can call
        regions, _ = regions_for_credentials(load_region_names(update_bundle=args.update_bundle))
        if not regions:
            logger.error("No regions found in regions.yml")
            logger.error("Please run: bua refresh regions")
            sys.exit(1)

        logger.info(f"Refreshing {len(regions)} regions...")
        refresh_all_regions(regions, update_bundle=args.update_bundle)
        logger.info("\n✓ All regions refreshed")


def cmd_refresh_fm_quotas(args):
    """Refresh quota mappings."""
    from bedrock_usage_analyzer.sync.quota_mapper import QuotaMapper
    from bedrock_usage_analyzer.utils.ui import select_quota_mapping_params
    _require_checkout(args)

    print(get_refresh_location_message())
    print()
    
    # Use provided arguments or interactive selection
    target_region = args.target_region
    bedrock_region = args.bedrock_region
    model_id = args.model_id
    from bedrock_usage_analyzer.utils.partition import is_valid_region_name
    for given in (target_region, bedrock_region):
        # A typo would otherwise surface as an STS connection error with credentials advice
        if given and not is_valid_region_name(given):
            logger.error(f"Invalid region format: {given}")
            sys.exit(1)

    credential_regions = None  # the credentials' regions, read once (by the picker or below)
    if not bedrock_region or not model_id or not target_region:
        resolved = {}
        bedrock_region, model_id, target_region = select_quota_mapping_params(
            target_region=target_region,
            bedrock_region=bedrock_region,
            model_id=model_id,
            resolved=resolved,
        )
        credential_regions = resolved.get('regions')
    else:
        from bedrock_usage_analyzer.sync.regions import load_region_names, regions_for_credentials
        from bedrock_usage_analyzer.utils.ui import require_credentials_partition
        credential_regions, partition = regions_for_credentials(
            load_region_names(update_bundle=args.update_bundle), target_region)
        # Both regions, as the interactive path does (the target first)
        require_credentials_partition(target_region, partition, label='Target region: ')
        require_credentials_partition(bedrock_region, partition)

    mapper = QuotaMapper(bedrock_region, model_id, target_region, credential_regions=credential_regions)
    mapper.run(update_bundle=args.update_bundle)
    
    logger.info("\n✓ Quota mapping complete")


def cmd_refresh_quota_index(args):
    """Generate quota index CSV."""
    from bedrock_usage_analyzer.sync.quota_index import QuotaIndexGenerator
    _require_checkout(args)

    print(get_refresh_location_message())
    print()
    
    generator = QuotaIndexGenerator()
    generator.run(update_bundle=args.update_bundle)
    
    logger.info("\n✓ Quota index generated")




def main():
    parser = argparse.ArgumentParser(
        prog='bua',
        description='Bedrock Usage Analyzer - Calculate token usage statistics for Amazon Bedrock'
    )
    subparsers = parser.add_subparsers(dest='command')
    
    # analyze
    p_analyze = subparsers.add_parser('analyze', help='Analyze token usage')
    p_analyze.add_argument('-o', '--output-dir', 
                          help='Directory to save results (default: prompt user)')
    p_analyze.add_argument('-r', '--region',
                          help='AWS region (e.g., us-west-2)')
    p_analyze.add_argument('-m', '--model-id', action='append',
                          help='Model ID, system inference profile ID, or application inference profile '
                               'ID/ARN (e.g., amazon.nova-premier-v1:0, us.amazon.nova-premier-v1:0, '
                               'or arn:aws:bedrock:us-west-2:111122223333:application-inference-profile/abc123). '
                               'Repeat to analyze several.')
    p_analyze.add_argument('-g', '--granularity',
                          help='Aggregation granularity: single value (1min, 5min, 1hour) for all periods, '
                               'or JSON for per-period config (e.g., \'{"1hour":"1min","1day":"5min","7days":"1hour","14days":"1hour","30days":"1hour"}\')')
    p_analyze.add_argument('-y', '--yes', action='store_true',
                          help='Skip account confirmation prompt')
    p_analyze.set_defaults(func=cmd_analyze)
    
    # refresh
    p_refresh = subparsers.add_parser('refresh', help='Refresh metadata')
    refresh_sub = p_refresh.add_subparsers(dest='refresh_command')
    
    # refresh regions
    p_regions = refresh_sub.add_parser('regions', help='Refresh regions list')
    p_regions.add_argument('--update-bundle', action='store_true',
                          help="Update the checkout's bundled metadata instead of user copies (maintainers only)")
    p_regions.set_defaults(func=cmd_refresh_regions)
    
    # refresh fm-list
    p_fm = refresh_sub.add_parser('fm-list', help='Refresh FM lists')
    p_fm.add_argument('region', nargs='?', help='Specific region (default: all)')
    p_fm.add_argument('--update-bundle', action='store_true',
                     help="Update the checkout's bundled metadata instead of user copies (maintainers only)")
    p_fm.set_defaults(func=cmd_refresh_fm_list)
    
    # refresh fm-quotas
    p_quotas = refresh_sub.add_parser('fm-quotas', help='Refresh quota mappings')
    p_quotas.add_argument('target_region', nargs='?', help='Target region')
    p_quotas.add_argument('bedrock_region', nargs='?', help='Bedrock API region')
    p_quotas.add_argument('model_id', nargs='?', help='Model ID for LLM calls')
    p_quotas.add_argument('--update-bundle', action='store_true',
                         help="Update the checkout's bundled metadata instead of user copies (maintainers only)")
    p_quotas.set_defaults(func=cmd_refresh_fm_quotas)
    
    # refresh quota-index
    p_index = refresh_sub.add_parser('quota-index', help='Generate quota index CSV')
    p_index.add_argument('--update-bundle', action='store_true',
                        help="Update the checkout's bundled metadata instead of user copies (maintainers only)")
    p_index.set_defaults(func=cmd_refresh_quota_index)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'refresh' and not getattr(args, 'refresh_command', None):
        p_refresh.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user.")
        sys.exit(1)
    except EOFError:
        logger.error("\nInput ended before all prompts were answered. "
                     "For scripted runs pass --region, --model-id, --granularity, --output-dir and -y.")
        sys.exit(1)
    except Exception as e:
        from bedrock_usage_analyzer.core.errors import troubleshooting_hint
        logger.error(f"Error: {e}")
        hint = troubleshooting_hint(e, getattr(args, 'region', None) or getattr(args, 'bedrock_region', None))
        if hint:
            logger.error(f"Hint: {hint}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        use_checkout_metadata(None)  # --update-bundle's redirect ends with the command


if __name__ == '__main__':
    main()
