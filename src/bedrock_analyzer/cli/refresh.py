# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI commands for refreshing metadata"""

import sys
import logging
import traceback
import argparse

from bedrock_analyzer.utils.yaml_handler import load_yaml, save_yaml
from bedrock_analyzer.metadata.regions import refresh_regions
from bedrock_analyzer.metadata.fm_list import refresh_region, refresh_all_regions
from bedrock_analyzer.metadata.quota_mapper import QuotaMapper
from bedrock_analyzer.metadata.quota_index import QuotaIndexGenerator
from bedrock_analyzer.utils.ui import select_quota_mapping_params
from bedrock_analyzer.aws.bedrock import discover_prefix_mapping, QUOTA_KEYWORD_ON_DEMAND, QUOTA_KEYWORD_GLOBAL

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def refresh_regions_command():
    """Refresh AWS regions list"""
    try:
        refresh_regions()
        logger.info("\n✓ Regions list refreshed")
    except Exception as e:
        logger.error(f"\n✗ Error: {e}")
        traceback.print_exc()
        sys.exit(1)


def refresh_fm_list_command(region: str = None):
    """Refresh foundation model lists
    
    Args:
        region: Specific region to refresh, or None for all regions
    """
    if region:
        # Refresh specific region
        refresh_region(region)
    else:
        # Refresh all regions
        try:
            regions_data = load_yaml('metadata/regions.yml')
            regions = regions_data.get('regions', [])
            
            if not regions:
                logger.error("No regions found in metadata/regions.yml")
                logger.error("Please run: refresh-regions first")
                sys.exit(1)
            
            logger.info(f"Refreshing {len(regions)} regions...")
            refresh_all_regions(regions)
            logger.info("\n✓ All regions refreshed")
            
        except FileNotFoundError:
            logger.error("Regions file not found: metadata/regions.yml")
            logger.error("Please run: refresh-regions first")
            sys.exit(1)


def refresh_quota_mapping_command(target_region: str = None, bedrock_region: str = None, model_id: str = None):
    """Refresh quota mappings for foundation models
    
    Args:
        target_region: Specific region to process (optional, prompts if not provided)
        bedrock_region: AWS region for Bedrock API calls (optional, prompts if not provided)
        model_id: Model ID to use for mapping (optional, prompts if not provided)
    """
    try:
        # Use provided arguments or interactive selection
        if not bedrock_region or not model_id or not target_region:
            bedrock_region, model_id, target_region = select_quota_mapping_params(
                target_region=target_region,
                bedrock_region=bedrock_region,
                model_id=model_id
            )
        
        # Run quota mapper
        mapper = QuotaMapper(bedrock_region, model_id, target_region)
        mapper.run()
        
        logger.info("\n✓ Quota mapping complete")
        
    except KeyboardInterrupt:
        logger.info("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗ Error: {e}")
        traceback.print_exc()
        sys.exit(1)


def refresh_quota_index_command():
    """Generate quota index CSV for validation"""
    try:
        generator = QuotaIndexGenerator()
        generator.run()
        logger.info("\n✓ Quota index generated")
    except Exception as e:
        logger.error(f"\n✗ Error: {e}")
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point for bedrock-refresh command"""

    
    parser = argparse.ArgumentParser(description='Refresh Bedrock metadata')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Regions refresh
    regions_parser = subparsers.add_parser('regions', help='Refresh AWS regions list')
    
    # FM list refresh
    fm_parser = subparsers.add_parser('fm-list', help='Refresh foundation model lists')
    fm_parser.add_argument('region', nargs='?', help='Specific region to refresh (optional)')
    
    # Quota mapping refresh
    quota_parser = subparsers.add_parser('fm-quotas', help='Refresh quota mappings')
    quota_parser.add_argument('target_region', nargs='?', help='Target region to process (optional)')
    quota_parser.add_argument('bedrock_region', nargs='?', help='Bedrock API region (e.g., us-west-2)')
    quota_parser.add_argument('model_id', nargs='?', help='Model ID for mapping (e.g., us.anthropic.claude-3-7-sonnet-20250219-v1:0)')
    
    # Quota index generation
    index_parser = subparsers.add_parser('quota-index', help='Generate quota index CSV')
    
    args = parser.parse_args()
    
    if args.command == 'regions':
        refresh_regions_command()
    elif args.command == 'fm-list':
        refresh_fm_list_command(args.region)
    elif args.command == 'fm-quotas':
        refresh_quota_mapping_command(args.target_region, args.bedrock_region, args.model_id)
    elif args.command == 'quota-index':
        refresh_quota_index_command()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
