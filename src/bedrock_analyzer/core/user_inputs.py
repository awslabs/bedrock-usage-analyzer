# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""User input collection for Bedrock usage analysis"""

import os
import sys
import logging
import boto3

from ..utils.yaml_handler import load_yaml
from ..utils.ui import select_from_list

logger = logging.getLogger(__name__)


class UserInputs:
    """Handles interactive user input collection"""
    
    def __init__(self):
        self.account = None
        self.region = None
        self.models = []
        self.granularity_config = {  # The aggregation granularity for different metrics window/period
            '1hour': 300,   # 5 minutes
            '1day': 300,    # 5 minutes
            '7days': 300,   # 5 minutes
            '14days': 300,  # 5 minutes
            '30days': 300   # 5 minutes
        }
    
    def collect(self):
        """Interactive dialog to collect user inputs"""
        logger.info("This tool calculates token usage statistics (p50, p90, TPM, TPD, RPM) and throttling metrics for Bedrock models in your AWS account across Bedrock application inference profiles for a given foundation model.")
        logger.info("Statistics will be generated for: 1 hour, 1 day, 7 days, 14 days, and 30 days.")
        print()

        self.account = self._get_current_account()
        confirm = input(f"AWS account: {self.account} - Continue? (y/n): ").lower()
        if confirm != 'y':
            sys.exit(1)
        
        # Region selection
        self.region = self._select_region()
        
        # Ensure FM list exists for selected region
        self._ensure_fm_list(self.region)
        
        # Granularity configuration
        self._configure_granularity()
        
        # Model selection loop
        while True:
            model_config = self._select_model(self.region)
            if model_config is not None:  
                self.models.append(model_config)
            
            add_more = input("\nAdd another model? (y/n): ").lower()
            if add_more != 'y':
                break
    
    def _get_current_account(self):
        """Get current AWS account ID"""

        logger.info("Getting AWS account ID...")
        try:
            sts = boto3.client('sts')
            account = sts.get_caller_identity()['Account']
            logger.info(f"  Account: {account}")
            return account
        except Exception as e:
            logger.error(f"Failed to get AWS account ID: {e}")
            logger.error("Please configure AWS credentials in your current machine.")
            sys.exit(1)
    
    def _select_region(self):
        """Select region with numbered list"""
        regions = self._load_regions()
        logger.info("\nHint: If your region is not listed, run ./bin/refresh-regions")
        return select_from_list(
            "Available regions:",
            regions,
            allow_cancel=False,
            input_prompt=f"\nSelect region (1-{len(regions)}): "
        )
    
    def _select_model(self, region):
        """Select model with numbered lists"""
        fm_list = self._load_fm_list(region)
        
        # Get unique providers
        providers = sorted(set(m['provider'] for m in fm_list))
        
        # Select provider
        logger.info(f"\nHint: To refresh models, run ./bin/refresh-fm-list {region}")
        logger.info(f"      then ./bin/refresh-fm-quotas-mapping {region}")
        provider = select_from_list(
            "Available providers:",
            providers,
            allow_cancel=False,
            input_prompt=f"\nSelect provider (1-{len(providers)}): "
        )
        
        # Filter models by provider
        provider_models = [m for m in fm_list if m['provider'] == provider]
        
        # Select model
        logger.info(f"\nHint: To refresh models, run ./bin/refresh-fm-list {region}")
        logger.info(f"      then ./bin/refresh-fm-quotas-mapping {region}")
        selected_model = select_from_list(
            f"Available {provider} models:",
            provider_models,
            allow_cancel=False,
            display_fn=lambda m: m['model_id'],
            input_prompt=f"\nSelect model (1-{len(provider_models)}): "
        )
        model_id = selected_model['model_id']
        
        # Get endpoints for selected model
        endpoints = selected_model.get('endpoints', {})  
        
        # Derive inference profiles from endpoints (exclude 'base')  
        inference_profiles = sorted([k for k in endpoints.keys() if k != 'base'])  
        
        # Determine profile prefix
        profile_prefix = self._select_profile_prefix(endpoints, inference_profiles)
        
        # Handle skipped model  
        if profile_prefix is None and not endpoints:  
            logger.info("Skipping this model.")  
            return None  
        
        return {
            'model_id': model_id,
            'profile_prefix': profile_prefix
        }
    
    def _select_profile_prefix(self, endpoints, inference_profiles):  
        """Select inference profile prefix based on supported types"""
        # Check if base model is available
        has_base = 'base' in endpoints  
        
        if not has_base:  
            # Only inference profiles available
            logger.info("\nThis model only supports inference profiles.")
            choices = inference_profiles if inference_profiles else []  
        else:
            # Add base model option if available
            choices = inference_profiles + ['None (base model)'] if inference_profiles else ['None (base model)']
        
        # Handle empty endpoints case  
        if not choices:  
            logger.error("\n⚠️  ERROR: This model has no on-demand or inference profile endpoints in metadata.")  
            logger.error("This may indicate incomplete metadata or the model is only available via provisioned throughput.")  
            logger.info("\nYou can either:")  
            logger.info("  1. Skip this model (press Enter)")  
            logger.info("  2. Manually enter the full model ID with prefix (e.g., 'us.claude-haiku-4-5-20251001-v1:0' or just 'claude-haiku-4-5-20251001-v1:0' for base)")  
            manual_input = input("\nEnter model ID (or press Enter to skip): ").strip()  
            if not manual_input:  
                return None  
            # Parse manual input  
            if '.' in manual_input:  
                return manual_input.split('.')[0]  # Return prefix  
            return None  # Return None for base model  
        
        choice = select_from_list(
            "Available inference profile prefixes:",
            choices,
            allow_cancel=False,
            input_prompt=f"\nSelect profile prefix (1-{len(choices)}): "
        )
        return None if 'None' in choice else choice
    
    def _configure_granularity(self):
        """Configure data granularity for each time period"""
        logger.info("\n" + "="*60)
        logger.info("DATA GRANULARITY CONFIGURATION")
        logger.info("="*60)
        logger.info("Default granularity settings:")
        logger.info("  1 hour:  5 minutes")
        logger.info("  1 day:   5 minutes")
        logger.info("  7 days:  5 minutes")
        logger.info("  14 days: 5 minutes")
        logger.info("  30 days: 5 minutes")
        print()
        
        use_default = input("Use default granularity settings? (y/n): ").lower()
        if use_default == 'y':
            return
        
        logger.info("\nConfigure granularity for each period:")
        logger.info("(Finer granularity = more detail but slower fetching)")
        logger.info("Note: Longer periods cannot use finer granularity than shorter periods")
        print()
        
        # Track minimum granularity and previous period info
        min_granularity = 60
        prev_period_name = None
        prev_granularity_label = None
        
        # Configure each period in order
        periods = [
            ('1 HOUR', '1hour', [('1 minute', 60), ('5 minutes', 300)]),
            ('1 DAY', '1day', [('1 minute', 60), ('5 minutes', 300), ('1 hour', 3600)]),
            ('7 DAYS', '7days', [('1 minute', 60), ('5 minutes', 300), ('1 hour', 3600)]),
            ('14 DAYS', '14days', [('1 minute', 60), ('5 minutes', 300), ('1 hour', 3600)]),
            ('30 DAYS', '30days', [('1 minute', 60), ('5 minutes', 300), ('1 hour', 3600)])
        ]
        
        for period_name, period_key, options in periods:
            selected_seconds = self._select_granularity(
                period_name, options, min_granularity, 
                prev_period_name, prev_granularity_label
            )
            self.granularity_config[period_key] = selected_seconds
            
            # Update tracking for next iteration
            min_granularity = max(min_granularity, selected_seconds)
            prev_period_name = period_name
            prev_granularity_label = next(label for label, sec in options if sec == selected_seconds)
        
        logger.info("\n" + "="*60)
        logger.info("Granularity configuration complete!")
        logger.info("="*60)
    
    def _select_granularity(self, period_name, options, min_granularity, prev_period_name=None, prev_granularity_label=None):
        """Select granularity with strikethrough for unavailable options"""
        logger.info(f"\n{period_name} period:")
        
        available_options = []
        for i, (label, seconds) in enumerate(options, 1):
            if seconds < min_granularity:
                # Strikethrough with descriptive message
                if prev_period_name and prev_granularity_label:
                    reason = f"not available as you picked {prev_granularity_label} for {prev_period_name} window"
                else:
                    reason = "unavailable - too fine"
                logger.info(f"  {i}. \033[9m{label}\033[0m ({reason})")
            else:
                logger.info(f"  {i}. {label}")
                available_options.append(i)
        
        # Get valid choice
        while True:
            try:
                choice = int(input(f"Select granularity (1-{len(options)}): "))
                if choice in available_options:
                    return options[choice - 1][1]  # Return seconds
                logger.info("Please select an available (non-strikethrough) option")
            except ValueError:
                logger.info("Please enter a valid number")
    
    def _get_choice(self, min_val, max_val, prompt):
        """Helper to get valid numeric choice"""
        while True:
            try:
                choice = int(input(prompt))
                if min_val <= choice <= max_val:
                    return choice
                logger.info(f"Please enter a number between {min_val} and {max_val}")
            except ValueError:
                logger.info("Please enter a valid number")
    
    def _load_regions(self):
        """Load regions from yml, refresh if needed"""
        regions_file = 'metadata/regions.yml'
        
        if not os.path.exists(regions_file) or os.path.getsize(regions_file) == 0:
            logger.error("Regions list not found: metadata/regions.yml")
            logger.error("Please run: ./bin/refresh-regions")
            sys.exit(1)
        
        data = load_yaml(regions_file)
        return data.get('regions', [])
    
    def _ensure_fm_list(self, region):
        """Ensure FM list exists for region"""
        # Validate region format (AWS regions are alphanumeric with hyphens)
        if not region or not all(c.isalnum() or c == '-' for c in region):
            raise ValueError(f"Invalid region format: {region}")
        
        fm_file = f'metadata/fm-list-{region}.yml'
        
        if not os.path.exists(fm_file) or os.path.getsize(fm_file) == 0:
            logger.error(f"Foundation model list not found: {fm_file}")
            logger.error(f"Please run: ./bin/refresh-fm-list {region}")
            sys.exit(1)
    
    def _load_fm_list(self, region):
        """Load foundation models for region"""
        fm_file = f'metadata/fm-list-{region}.yml'
        
        data = load_yaml(fm_file)
        return data.get('models', [])
