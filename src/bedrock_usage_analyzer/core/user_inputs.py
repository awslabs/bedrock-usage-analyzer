# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""User input collection for Bedrock usage analysis"""

import os
import sys
import logging
from typing import Dict, List, Optional, Sequence, Union

from ..aws.bedrock import region_from_arn, split_profile_id
from ..aws.client_factory import create_client
from ..core.errors import troubleshooting_hint
from ..core.profile_fetcher import InferenceProfileFetcher
from ..sync.regions import load_region_names
from ..utils.yaml_handler import load_yaml
from ..utils.ui import select_from_list
from ..utils.paths import get_data_path
from ..utils.partition import (
    filter_regions_by_partition,
    get_caller_identity,
    get_partition_display_name,
    get_partition_for_region,
    get_region_display_name,
    is_govcloud_region,
    is_token_rejection,
    is_valid_region_name,
    probe_other_partitions,
    region_hint,
    resolve_caller_identity,
)

logger = logging.getLogger(__name__)


def parse_selection(text: str, count: int) -> List[int]:
    """Parse '1,3-5' or 'all' into sorted 0-based indices; raises ValueError on bad input."""
    text = text.strip().lower()
    if text == 'all':
        return list(range(count))
    indices = set()
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start_s, end_s = part.split('-', 1)
            start, end = int(start_s), int(end_s)
            if start > end:
                raise ValueError(f"Invalid range: {part}")
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    if not indices or min(indices) < 1 or max(indices) > count:
        raise ValueError(f"Choose numbers between 1 and {count}")
    return sorted(i - 1 for i in indices)


def group_application_profiles(profiles: Sequence[Dict]) -> List[Dict]:
    """Turn selected application profiles into one model config per source endpoint."""
    groups: Dict[tuple, Dict] = {}
    for app in profiles:
        key = (app['model_id'], app['profile_prefix'])
        config = groups.setdefault(key, {
            'model_id': app['model_id'],
            'profile_prefix': app['profile_prefix'],
            'application_profile_ids': [],
        })
        if app['id'] not in config['application_profile_ids']:
            config['application_profile_ids'].append(app['id'])
    return list(groups.values())


def merge_application_configs(configs: Sequence[Dict]) -> List[Dict]:
    """Combine application-profile configs that share a source endpoint.

    `-m id1 -m id2` for two profiles of the same endpoint gives one aggregated
    report, the same as selecting both interactively.
    """
    merged: List[Dict] = []
    by_source: Dict[tuple, Dict] = {}
    for config in configs:
        ids = config.get('application_profile_ids')
        if not ids:
            if config not in merged:
                merged.append(config)
            continue
        key = (config['model_id'], config['profile_prefix'])
        if key in by_source:
            target = by_source[key]['application_profile_ids']
            target.extend(i for i in ids if i not in target)
        else:
            by_source[key] = {**config, 'application_profile_ids': list(ids)}
            merged.append(by_source[key])
    return merged


class UserInputs:
    """Handles interactive user input collection"""

    def __init__(self):
        self.account = None
        self.partition = None
        self.region = None
        self.models = []
        self.profile_fetcher: Optional[InferenceProfileFetcher] = None
        self._fm_lists: Dict[str, List[Dict]] = {}
        self.granularity_config = {  # The aggregation granularity for different metrics window/period
            '1hour': 300,   # 5 minutes
            '1day': 300,    # 5 minutes
            '7days': 300,   # 5 minutes
            '14days': 300,  # 5 minutes
            '30days': 300   # 5 minutes
        }

    def collect(self, region=None, model_id: Union[None, str, Sequence[str]] = None,
                granularity_config=None, skip_confirm=False):
        """Interactive dialog to collect user inputs, skipping prompts for provided values.

        Args:
            region: AWS region (skip region prompt if provided)
            model_id: One or more model IDs, system profile IDs, or application
                inference profile IDs/ARNs (skip model prompt if provided)
            granularity_config: Dict of time period to seconds (skip granularity prompt if provided)
            skip_confirm: Skip account confirmation prompt (for scripted usage)
        """
        logger.info("This tool calculates token usage statistics (p50, p90, TPM, TPD, RPM) and throttling metrics for Bedrock models in your AWS account across Bedrock application inference profiles for a given foundation model.")
        logger.info("Statistics will be generated for: 1 hour, 1 day, 7 days, 14 days, and 30 days.")
        print()

        if region:
            self._validate_region(region)

        self.account = self._get_current_account(region)
        if not skip_confirm:
            confirm = input(f"AWS account: {self.account} - Continue? ([y]/n): ").lower()
            if confirm not in ['', 'y']:
                sys.exit(1)
        else:
            logger.info(f"AWS account: {self.account}")

        # Region selection (skip if provided via CLI)
        if region:
            # A region of another partition was already rejected by the STS call above
            self.region = region
            logger.info(f"\nUsing region: {region}")
        else:
            self.region = self._select_region()

        # Ensure FM list exists for selected region
        self._ensure_fm_list(self.region)

        # Granularity configuration (skip if provided via CLI)
        if granularity_config:
            self.granularity_config = granularity_config
            logger.info(f"\nUsing granularity config from CLI")
        else:
            self._configure_granularity()

        # Model selection (skip if provided via CLI)
        if model_id:
            values = [model_id] if isinstance(model_id, str) else list(model_id)
            configs = []
            for value in values:
                configs.append(self._parse_model_id(value))
                logger.info(f"\nUsing model: {value}")
            self._add_models(merge_application_configs(configs))
        else:
            # Model selection loop
            while True:
                self._add_models(self._select_targets(self.region))

                add_more = input("\nAdd another model? (y/[n]): ").lower()
                if add_more != 'y':
                    break
            # Profiles of one endpoint picked in different rounds become one report, as with -m
            self.models = merge_application_configs(self.models)

    def _add_models(self, configs):
        for config in configs or []:
            if config and config not in self.models:
                self.models.append(config)

    # ------------------------------------------------------------ account/region

    @staticmethod
    def _validate_region(region):
        if not is_valid_region_name(region):
            logger.error(f"Invalid region format: {region!r} (expected e.g. us-west-2 or us-gov-west-1)")
            sys.exit(1)

    def _get_current_account(self, region=None):
        """Get current AWS account ID (and remember the credentials' partition)"""

        logger.info("Getting AWS account ID...")
        try:
            # Regional STS first (VPC endpoints), then the partition's home region
            identity = resolve_caller_identity(region, lookup=get_caller_identity)
        except Exception as e:
            if is_token_rejection(e):
                self._explain_partition_mismatch(region or region_hint())
            logger.error(f"Failed to get AWS account ID: {e}")
            hint = troubleshooting_hint(e, region or region_hint())
            logger.error(hint or "Please configure AWS credentials in your current machine.")
            if not region and not region_hint():
                logger.error("For GovCloud or China credentials, pass --region (e.g. --region us-gov-west-1) "
                             "or set AWS_REGION.")
            sys.exit(1)
        self.partition = identity['Partition']
        logger.info(f"  Account: {identity['Account']}")
        if self.partition != 'aws':
            logger.info(f"  Partition: {get_partition_display_name(self.partition)}")
        return identity['Account']

    def _explain_partition_mismatch(self, region):
        """If STS in ``region`` rejected the credentials, check whether they belong to another partition.

        STS of one partition rejects credentials of another with a generic
        'invalid token' error, so ask STS without the region pin and report the
        mismatch in plain terms.
        """
        if not region:
            return
        # Shared with the refresh commands; the lookup goes through this module's
        # get_caller_identity so it can be stubbed in tests
        identity = probe_other_partitions(region, lookup=lambda r: get_caller_identity(r, probe=True))
        if identity:
            self.partition = identity['Partition']
            self._check_region_partition(region)

    def _check_region_partition(self, region):
        """Stop early when the region belongs to a different partition than the credentials."""
        region_partition = get_partition_for_region(region)
        if self.partition and region_partition != self.partition:
            logger.error(f"\nRegion {region} is in {get_partition_display_name(region_partition)}, but the "
                         f"credentials are for {get_partition_display_name(self.partition)}.")
            logger.error("Use credentials for that partition (e.g. AWS_PROFILE=...) or pick a region in "
                         "the credentials' partition.")
            sys.exit(1)

    def _select_region(self):
        """Select a region, showing only regions in the credentials' partition"""
        # Known from the account check, which always runs first (no second STS call)
        partition = self.partition
        regions = filter_regions_by_partition(self._load_regions(), partition)
        if not regions:
            logger.error("No regions available for these credentials.")
            logger.error("Please run: bua refresh regions")
            sys.exit(1)
        if partition:
            logger.info(f"\nShowing {len(regions)} {get_partition_display_name(partition)} regions "
                        f"(detected from your credentials)")

        logger.info("Hint: If your region is not listed, run: bua refresh regions")
        return select_from_list(
            "Available regions:",
            regions,
            allow_cancel=False,
            display_fn=self._region_label,
            input_prompt=f"\nSelect region (1-{len(regions)}): "
        )

    @staticmethod
    def _region_label(region):
        name = get_region_display_name(region)
        label = region if name == region else f"{region} ({name})"
        return f"{label} [GovCloud]" if is_govcloud_region(region) and 'GovCloud' not in label else label

    # ------------------------------------------------------------------ models

    def _get_profile_fetcher(self) -> InferenceProfileFetcher:
        if self.profile_fetcher is None:
            self.profile_fetcher = InferenceProfileFetcher(create_client('bedrock', self.region))
        return self.profile_fetcher

    def _parse_model_id(self, model_id):
        """Parse model ID from CLI argument.

        Accepted formats:
        - Base model: 'amazon.nova-premier-v1:0'
        - System inference profile: 'us.amazon.nova-premier-v1:0', 'us-gov.anthropic...'
          or its ARN
        - Application inference profile: its ID (e.g. 'tqab5jqtywp7') or ARN; only
          that profile is analyzed

        The prefix (us, eu, apac, global, etc.) indicates cross-region inference profile.
        Provider names (amazon, anthropic, meta, etc.) are NOT prefixes.

        Args:
            model_id: Model ID with optional prefix

        Returns:
            dict: Model config with model_id and profile_prefix (and
                application_profile_ids for an application profile)
        """
        value = model_id.strip()

        if value.startswith('arn:'):
            resource = value.split(':', 5)[-1] if value.count(':') >= 5 else ''
            kind, _, ident = resource.partition('/')
            arn_region = region_from_arn(value)
            if arn_region and self.region and arn_region != self.region:
                logger.error(f"ARN region {arn_region} does not match the analysis region {self.region}")
                sys.exit(1)
            if kind == 'application-inference-profile':
                return self._application_profile_config(value)
            if kind in ('inference-profile', 'foundation-model') and ident:
                value = ident
            else:
                logger.error(f"Unsupported ARN: {model_id}")
                sys.exit(1)
        elif '.' not in value and ':' not in value:
            # No provider prefix: an application inference profile ID (or name)
            return self._application_profile_config(value)

        known_model = self._is_known_model(value)
        if not known_model and not model_id.strip().startswith('arn:'):
            # Application profile names may contain '.' and ':'. A known model or system
            # profile ID always wins, so '-m us.<model>' is never narrowed to one profile.
            profile = self._find_application_profile(value)
            if profile is not None:
                return self._application_profile_config(value, profile)

        base_model_id, prefix = split_profile_id(value)
        if prefix is None and value.count('.') >= 2 and not known_model and self._is_system_profile(value):
            # A system profile with a prefix newer than this release (e.g. 'kr.')
            prefix, base_model_id = value.split('.', 1)
        elif not known_model and self.region:
            # Still analyzed (the model list may predate a new model), but not silently
            logger.warning(f"  WARNING: {value} is not a model, inference profile or application "
                           f"inference profile known in {self.region}. If it is new, run "
                           f"'bua refresh fm-list {self.region}'; otherwise check the ID.")
        return {
            'model_id': base_model_id,
            'profile_prefix': prefix
        }

    def _is_system_profile(self, value: str) -> bool:
        """True when ``value`` is the ID of a system inference profile in the region."""
        try:
            return self._get_profile_fetcher().is_system_profile(value)
        except Exception as e:
            logger.debug(f"Could not list system inference profiles: {e}")
            return False

    def _is_known_model(self, value: str) -> bool:
        """True when ``value`` is a model or system profile ID listed for the region."""
        model_id, prefix = split_profile_id(value)
        if not self.region or not os.path.exists(get_data_path(f'fm-list-{self.region}.yml')):
            return prefix is not None
        for model in self._load_fm_list(self.region):
            if model.get('model_id') == model_id:
                return prefix is None or prefix in (model.get('endpoints') or {})
        return False

    def _find_application_profile(self, identifier):
        """Look up an application profile, or None if absent or the list cannot be read."""
        try:
            return self._get_profile_fetcher().resolve_application_profile(identifier)
        except Exception as e:
            logger.debug(f"Could not list application inference profiles: {e}")
            return None

    def _application_profile_config(self, identifier, profile=None):
        profile = profile or self._get_profile_fetcher().resolve_application_profile(identifier)
        if profile is None:
            logger.error(f"Application inference profile not found in {self.region}: {identifier}")
            sys.exit(1)
        logger.info(f"  Application inference profile {profile['name']} ({profile['id']}) "
                    f"is based on {profile['source'] or 'an unknown endpoint'}")
        return group_application_profiles([profile])[0]

    def _select_targets(self, region) -> List[Dict]:
        """Ask whether to analyze a foundation model or specific application profiles."""
        try:
            app_profiles = self._get_profile_fetcher().list_application_profiles()
        except Exception as e:
            logger.info(f"  Could not list application inference profiles: {e}")
            app_profiles = []

        if app_profiles:
            mode = select_from_list(
                "What do you want to analyze?",
                ['A foundation model (includes the application inference profiles created from it)',
                 f'Specific application inference profiles ({len(app_profiles)} in {region})'],
                allow_cancel=False,
                input_prompt="\nSelect (1-2): "
            )
            if mode.startswith('Specific'):
                return self._select_application_profiles(app_profiles)

        config = self._select_model(region)
        return [config] if config else []

    def _select_application_profiles(self, app_profiles) -> List[Dict]:
        """Pick one or more application inference profiles by number."""
        print("\nApplication inference profiles:")
        for i, app in enumerate(app_profiles, 1):
            print(f"  {i}. {app['name']} ({app['id']}) - based on {app['source'] or 'an unknown endpoint'}")
        while True:
            try:
                text = input(f"\nSelect profiles (e.g. 1,3-4 or all): ")
                indices = parse_selection(text, len(app_profiles))
                break
            except ValueError as e:
                print(f"Please enter valid numbers: {e}")
        return group_application_profiles([app_profiles[i] for i in indices])

    def _select_model(self, region):
        """Select model with numbered lists"""
        fm_list = self._load_fm_list(region)
        if not fm_list:
            logger.error(f"No foundation models listed for {region}. Run: bua refresh fm-list {region}")
            return None

        # Get unique providers
        providers = sorted(set(m['provider'] for m in fm_list))

        # Select provider
        logger.info(f"\nHint: To refresh models, run: bua refresh fm-list {region}")
        logger.info(f"      then: bua refresh fm-quotas {region}")
        provider = select_from_list(
            "Available providers:",
            providers,
            allow_cancel=False,
            input_prompt=f"\nSelect provider (1-{len(providers)}): "
        )

        # Filter models by provider
        provider_models = [m for m in fm_list if m['provider'] == provider]

        # Select model
        selected_model = select_from_list(
            f"Available {provider} models:",
            provider_models,
            allow_cancel=False,
            display_fn=lambda m: m['model_id'],
            input_prompt=f"\nSelect model (1-{len(provider_models)}): "
        )
        model_id = selected_model['model_id']

        # Get endpoints for selected model
        endpoints = selected_model.get('endpoints', {}) or {}

        # Derive inference profiles from endpoints (exclude 'base')
        inference_profiles = sorted([k for k in endpoints.keys() if k != 'base'])

        if not endpoints:
            return self._manual_model_entry()

        profile_prefix = self._select_profile_prefix(endpoints, inference_profiles)
        return {
            'model_id': model_id,
            'profile_prefix': profile_prefix
        }

    def _manual_model_entry(self):
        """Fallback when a model has no endpoints in metadata."""
        logger.error("\n⚠️  ERROR: This model has no on-demand or inference profile endpoints in metadata.")
        logger.error("This may indicate incomplete metadata or the model is only available via provisioned throughput.")
        logger.info("\nYou can either:")
        logger.info("  1. Skip this model (press Enter)")
        logger.info("  2. Manually enter the full model ID with prefix (e.g., 'us.anthropic.claude-haiku-4-5-20251001-v1:0' or 'anthropic.claude-haiku-4-5-20251001-v1:0' for base)")
        manual_input = input("\nEnter model ID (or press Enter to skip): ").strip()
        if not manual_input:
            logger.info("Skipping this model.")
            return None
        try:
            return self._parse_model_id(manual_input)
        except (SystemExit, Exception) as e:
            # A typo, or a failed profile lookup, skips this model; it does not end the
            # session and discard the models already selected
            if not isinstance(e, SystemExit):
                logger.info(f"Could not look up {manual_input!r}: {e}")
            logger.info("Skipping this model.")
            return None

    def _select_profile_prefix(self, endpoints, inference_profiles):
        """Select inference profile prefix based on supported types"""
        # Check if base model is available
        has_base = 'base' in endpoints

        if not has_base:
            # Only inference profiles available
            logger.info("\nThis model only supports inference profiles.")
            choices = list(inference_profiles)
        else:
            # Add base model option if available
            choices = list(inference_profiles) + ['None (base model)']

        choice = select_from_list(
            "Available inference profile prefixes:",
            choices,
            allow_cancel=False,
            input_prompt=f"\nSelect profile prefix (1-{len(choices)}): "
        )
        return None if choice == 'None (base model)' else choice

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
        
        use_default = input("Use default granularity settings? ([y]/n): ").lower()
        if use_default in ['y', '']:
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
        """Load region names from regions.yml (user copy, else bundled)"""
        if not os.path.exists(get_data_path('regions.yml')):
            logger.error("Regions list not found")
            logger.error("Please run: bua refresh regions")
            sys.exit(1)
        return load_region_names()

    def _ensure_fm_list(self, region):
        """Ensure FM list exists for region"""
        # Validate region format before using it in a file name
        if not is_valid_region_name(region):
            raise ValueError(f"Invalid region format: {region}")

        if not os.path.exists(get_data_path(f'fm-list-{region}.yml')):
            logger.error(f"Foundation model list not found for region: {region}")
            logger.error(f"Please run: bua refresh fm-list {region}")
            sys.exit(1)
    
    def _load_fm_list(self, region):
        """Load foundation models for region (parsed once per region)"""
        if region not in self._fm_lists:
            data = load_yaml(get_data_path(f'fm-list-{region}.yml')) or {}
            self._fm_lists[region] = data.get('models', []) or []
        return self._fm_lists[region]
    
    def select_output_dir(self) -> str:
        """Prompt user to select output directory for results."""
        from ..utils.paths import get_default_results_dir
        
        current_dir = "./results"
        user_data_dir = str(get_default_results_dir())
        
        print("\nWhere to save results?")
        print(f"  [1] Current directory ({current_dir})")
        print(f"  [2] User data directory ({user_data_dir})")
        print("  [3] Custom location")
        
        while True:
            choice = input("\nEnter choice [1]: ").strip()
            if choice == "" or choice == "1":
                return current_dir
            elif choice == "2":
                return user_data_dir
            elif choice == "3":
                custom_path = input("Enter custom path: ").strip()
                if custom_path:
                    return os.path.expanduser(custom_path)
                print("Please enter a valid path")
            else:
                print("Please enter 1, 2, or 3")
