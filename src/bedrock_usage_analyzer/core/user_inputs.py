# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""User input collection for Bedrock usage analysis"""

import os
import sys
import logging
from typing import Dict, List, Optional, Sequence, Union

from ..aws.bedrock import endpoint_id, region_from_arn, split_profile_id
from ..aws.client_factory import create_client
from ..core.errors import troubleshooting_hint
from ..core.profile_fetcher import UNKNOWN_SOURCE, InferenceProfileFetcher
from ..sync.regions import load_region_names
from ..utils.yaml_handler import endpoint_keys, has_endpoint, load_fm_list, profile_endpoints
from ..utils.ui import require_credentials_partition, select_from_list
from ..utils.partition import (
    filter_regions_by_partition,
    get_caller_identity,
    get_partition_display_name,
    get_region_display_name,
    is_govcloud_region,
    other_partition_message,
    is_valid_region_name,
    probe_if_rejected,
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


def _source_key(model_id: str, prefix, profile_ids) -> tuple:
    """Grouping key for application profiles: their source endpoint. Profiles whose source is
    unknown may come from different endpoints, so each one keeps a report of its own."""
    if prefix == UNKNOWN_SOURCE:
        return (model_id, prefix, tuple(profile_ids))
    return (model_id, prefix)


def group_application_profiles(profiles: Sequence[Dict]) -> List[Dict]:
    """Turn selected application profiles into one model config per source endpoint."""
    return merge_application_configs([
        {'model_id': app['model_id'], 'profile_prefix': app['profile_prefix'],
         'application_profile_ids': [app['id']]} for app in profiles])


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
        key = _source_key(config['model_id'], config['profile_prefix'], ids)
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
        self._region_given = False  # set by _get_current_account: a region was passed in
        self.models = []
        self.profile_fetcher: Optional[InferenceProfileFetcher] = None
        self._fm_lists: Dict[str, Optional[List[Dict]]] = {}  # None: the region has no fm-list
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
            # Its partition was checked with the account (_get_current_account)
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
        self._region_given = bool(region)
        try:
            # Regional STS first (VPC endpoints), then the partition's home region
            identity = resolve_caller_identity(region, lookup=get_caller_identity)
        except Exception as e:
            identity = self._explain_partition_mismatch(region or region_hint(), e)
            if identity is None:
                self._exit_no_identity(e, region)
            # Only a default region (AWS_REGION / config), or none, was in another partition:
            # the region picker below lists the credentials' partition
            logger.warning(f"  {other_partition_message(region_hint(), identity['Partition'])} "
                           f"Using the credentials' partition.")
        self.partition = identity['Partition']
        if region:
            # Before the 'Continue?' prompt. STS in a region of another partition usually
            # rejects the credentials (handled above), but a custom STS endpoint can answer
            self._check_region_partition(region)
        logger.info(f"  Account: {identity['Account']}")
        if self.partition != 'aws':
            logger.info(f"  Partition: {get_partition_display_name(self.partition)}")
        return identity['Account']

    @staticmethod
    def _exit_no_identity(e, region):
        logger.error(f"Failed to get AWS account ID: {e}")
        hint = troubleshooting_hint(e, region or region_hint())
        logger.error(hint or "Please configure AWS credentials in your current machine.")
        if not region and not region_hint():
            logger.error("For GovCloud or China credentials, pass --region (e.g. --region us-gov-west-1) "
                         "or set AWS_REGION.")
        sys.exit(1)

    def _explain_partition_mismatch(self, region, error):
        """If STS in ``region`` rejected the credentials, check whether they belong to another partition.

        STS of one partition rejects credentials of another with a generic
        'invalid token' error, so ask STS of the other partitions. Returns the
        identity when found and ``region`` was only a default (not --region);
        exits with a plain explanation when the user named that region.
        """
        # Shared with the refresh commands; the lookup goes through this module's
        # get_caller_identity so it can be stubbed in tests
        identity = probe_if_rejected(region, error, probe_lookup=lambda r: get_caller_identity(r, probe=True))
        if identity and self._region_given:
            self.partition = identity['Partition']
            self._check_region_partition(region)
        return identity

    def _check_region_partition(self, region):
        """Stop early when the region belongs to a different partition than the credentials."""
        require_credentials_partition(region, self.partition, label='')

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
            self.profile_fetcher = InferenceProfileFetcher.for_region(
                create_client('bedrock', self.region), self._load_fm_list(self.region), self.region)
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
            if isinstance(self.profile_fetcher, InferenceProfileFetcher) and \
                    self.profile_fetcher.routes_to_no_foundation_model(value):
                # A named copy of a custom model: stop, as for its ID, instead of an empty report
                self._application_profile_config(value)

        base_model_id, prefix = split_profile_id(value)
        profile_only = [] if known_model or prefix else self._profile_endpoints_of(base_model_id)
        if prefix is None and value.count('.') >= 2 and not known_model and self._is_system_profile(value):
            # A system profile with a prefix newer than this release (e.g. 'kr.')
            prefix, base_model_id = value.split('.', 1)
        elif profile_only:
            # Listed, but offered only through inference profiles: the bare ID has no usage
            options = ', '.join(endpoint_id(base_model_id, p) for p in profile_only)
            logger.warning(f"  WARNING: {value} has no on-demand endpoint in {self.region}; its "
                           f"usage is under its inference profiles: {options}")
        elif not known_model and self.region and prefix and self._is_system_profile(value):
            # Listed by Bedrock, only the model list is older: no reason to doubt the ID
            logger.info(f"  Note: {value} is listed in {self.region} but not in its model list; "
                        f"run 'bua refresh fm-list {self.region}' to map its quotas.")
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

    def _profile_endpoints_of(self, model_id: str) -> list:
        """Inference profile prefixes the region's fm-list lists for ``model_id``."""
        return profile_endpoints(self._load_fm_list(self.region), model_id)

    def _is_known_model(self, value: str) -> bool:
        """True when ``value`` is a model or system profile ID listed for the region."""
        model_id, prefix = split_profile_id(value)
        # The region's fm-list exists here: collect() exits earlier when it is missing
        # A bare model ID is an endpoint only when the model is invokable on demand
        return has_endpoint(self._load_fm_list(self.region), model_id, prefix)

    def _find_application_profile(self, identifier):
        """Look up an application profile, or None if absent or the list cannot be read."""
        try:
            return self._get_profile_fetcher().resolve_application_profile(identifier)
        except Exception as e:
            if not self._listing_error():
                raise
            # Not silent: an application profile name with '.' or ':' would otherwise be
            # analyzed as a model ID without saying why
            logger.warning(f"  WARNING: could not list {self._failed_listing()} inference profiles in {self.region}, "
                           f"so {identifier} is treated as a model or system profile ID: {e}")
            return None

    def _listing_error(self) -> bool:
        """True when the last profile-fetcher error was a listing (or client) failure, not a bug.

        A bug in source resolution is raised, not reported as missing permissions.
        """
        fetcher = self.profile_fetcher
        return not isinstance(fetcher, InferenceProfileFetcher) or fetcher.listing_failed()

    def _failed_listing(self) -> str:
        """'application' or 'system': which listing failed (see InferenceProfileFetcher.failed_listing)."""
        fetcher = self.profile_fetcher
        return fetcher.failed_listing() if isinstance(fetcher, InferenceProfileFetcher) else 'application'

    def _application_profile_config(self, identifier, profile=None):
        if profile is None:
            try:
                profile = self._get_profile_fetcher().resolve_application_profile(identifier)
            except Exception as e:
                # An application profile ID or ARN cannot be analyzed without the listings
                logger.error(f"Could not list {self._failed_listing()} inference profiles in {self.region}, "
                             f"so {identifier} cannot be resolved: {e}")
                sys.exit(1)
        if profile is None:
            fetcher = self.profile_fetcher
            if isinstance(fetcher, InferenceProfileFetcher) and fetcher.routes_to_no_foundation_model(identifier):
                # It exists (e.g. a copy of a custom model): say why it cannot be analyzed
                logger.error(f"Application inference profile {identifier} routes to no foundation model; "
                             f"this tool has no metrics or quotas for it")
            else:
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
            if not self._listing_error():
                raise
            logger.info(f"  Could not list {self._failed_listing()} inference profiles: {e}")
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
            # A guessed source the region no longer lists is marked here, before it is picked
            retired = app['profile_prefix'] not in (None, UNKNOWN_SOURCE) and app['source'] and \
                not self._is_system_profile(app['source'])
            note = f" (not offered in {self.region} any more)" if retired else ""
            print(f"  {i}. {app['name']} ({app['id']}) - based on {app['source'] or 'an unknown endpoint'}{note}")
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
        def provider_of(model):
            return str(model.get('provider') or 'Unknown')  # a hand-edited entry may have none

        providers = sorted(set(map(provider_of, fm_list)))

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
        provider_models = [m for m in fm_list if provider_of(m) == provider]

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
        # Endpoint keys, a legacy entry's 'base' included (as -m and the analyzer read it)
        endpoints = endpoint_keys(selected_model)

        # Derive inference profiles from endpoints (exclude 'base')
        inference_profiles = sorted(k for k in endpoints if k != 'base')

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
        # Read through the loaders (user copy, else bundled, also from a zipped package)
        names = load_region_names()
        if not names:
            logger.error("Regions list not found")
            logger.error("Please run: bua refresh regions")
            sys.exit(1)
        return names

    def _ensure_fm_list(self, region):
        """Ensure FM list exists for region"""
        # Validate region format before using it in a file name (a picked region comes from
        # a regions.yml that may be hand-edited, e.g. 'US-EAST-1')
        self._validate_region(region)

        import yaml
        try:
            models = self._load_fm_list_or_none(region)
        except yaml.YAMLError as e:
            # A hand-edited user copy with a syntax error: name the file and the way out
            from ..utils.paths import get_data_path
            logger.error(f"Could not read {get_data_path(f'fm-list-{region}.yml')}: {e}")
            logger.error(f"Fix or delete the file, or run: bua refresh fm-list {region}")
            sys.exit(1)
        if models is None:
            logger.error(f"Foundation model list not found for region: {region}")
            logger.error(f"Please run: bua refresh fm-list {region}")
            sys.exit(1)
    
    def fm_models(self):
        """Parsed fm-list of the selected region (shared with the analyzer)."""
        return self._load_fm_list(self.region)

    def _load_fm_list_or_none(self, region):
        """The region's fm-list models, None when there is no list (parsed once per region)."""
        if region not in self._fm_lists:
            self._fm_lists[region] = load_fm_list(region)
        return self._fm_lists[region]

    def _load_fm_list(self, region):
        """The region's fm-list models ([] when there is no list)."""
        return self._load_fm_list_or_none(region) or []
    
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
