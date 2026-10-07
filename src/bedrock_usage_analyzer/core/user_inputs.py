# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""User input collection for Bedrock usage analysis"""

import os
import sys
import logging
from typing import Dict, List, Optional, Sequence, Union

from ..aws.bedrock import arn_resource, endpoint_id, region_from_arn, split_profile_id
from ..aws.client_factory import create_client
from ..aws.custom_models import DEPLOYMENT_KIND, IMPORTED_KIND, base_model_id_in_arn, deployment_short_id, is_active
from ..aws.invocation_logs import (
    METADATA, PRINCIPAL, SESSION, TAG, Breakdown, BreakdownError, logging_destination)
from ..core.breakdown import CONFIG_DENIED_HINT, ENABLE_HINT
from ..core.errors import AWS_ERRORS, is_access_denied, troubleshooting_hint
from ..core.profile_fetcher import (
    UNKNOWN_SOURCE, InferenceProfileFetcher, deployment_read_error, missing_deployment_api)
from ..sync.regions import load_region_names
from ..utils.yaml_handler import CUSTOM_ENDPOINT, endpoint_keys, fm_endpoints, has_endpoint, invokable_endpoint_keys, load_fm_list, profile_endpoints
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
        self._inactive_deployments_noted = False
        self.breakdown: Optional[Breakdown] = None
        self._deployment_listing_noted = False
        self._imported_listing_noted = False
        self._fm_lists: Dict[str, Optional[List[Dict]]] = {}  # None: the region has no fm-list
        self.granularity_config = {  # The aggregation granularity for different metrics window/period
            '1hour': 300,   # 5 minutes
            '1day': 300,    # 5 minutes
            '7days': 300,   # 5 minutes
            '14days': 300,  # 5 minutes
            '30days': 300   # 5 minutes
        }

    def collect(self, region=None, model_id: Union[None, str, Sequence[str]] = None,
                granularity_config=None, skip_confirm=False, breakdown=None):
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
            # Deployment ARNs and their custom models are read all at once, in parallel
            # (only those of the region: another region's ARN is refused below, unread)
            deployment_arns = [v.strip() for v in values if f":{DEPLOYMENT_KIND}/" in v and v.strip().startswith('arn:')
                               and region_from_arn(v.strip()) == self.region]
            fetcher = self._get_profile_fetcher() if len(deployment_arns) > 1 else None
            if isinstance(fetcher, InferenceProfileFetcher):
                try:
                    fetcher.read_custom_deployments(deployment_arns)
                except AttributeError as e:
                    if not missing_deployment_api(e):
                        raise  # the per-ARN read says what boto3 is needed
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

        # Breakdown by caller (from the CLI; asked only in an interactive session)
        self.breakdown = breakdown
        if breakdown is None and not model_id and self.models:
            self.breakdown = self._select_breakdown()

    def _select_breakdown(self) -> Optional[Breakdown]:
        """Ask whether to break usage down by caller, when the region logs invocations to
        CloudWatch Logs (the only per-caller source of tokens and requests)."""
        fetcher = self._get_profile_fetcher()
        if not isinstance(fetcher, InferenceProfileFetcher):  # an API caller's own fetcher
            return None
        try:
            group, reason = logging_destination(fetcher.bedrock_client)
        except AWS_ERRORS as e:
            if is_access_denied(e):
                # Logging may well be on: the fix is the permission, or naming the group
                fix = CONFIG_DENIED_HINT
            else:  # throttled, unreachable, expired credentials: not a permission to add
                fix = troubleshooting_hint(e, self.region) or "Run again to be offered it."
            logger.info(f"\nUsage by caller (IAM principal) is not offered: the model invocation logging "
                        f"configuration could not be read ({e}). {fix}")
            return None
        if not group:
            logger.info(f"\nUsage by caller (IAM principal) is not available: {reason}. {ENABLE_HINT}")
            return None
        choices = ['No breakdown',
                   'By IAM principal (role or user; sessions of a role together)',
                   'By IAM principal session',
                   'By an IAM principal tag (e.g. team)',
                   'By a request metadata key (requestMetadata)']
        choice = select_from_list(
            f"\nBreak usage down by caller? (from the model invocation logs in {group}; "
            f"Logs Insights is charged per GB scanned)", choices, allow_cancel=False,
            input_prompt=f"\nSelect (1-{len(choices)}): ")
        kind = {choices[1]: PRINCIPAL, choices[2]: SESSION, choices[3]: TAG, choices[4]: METADATA}.get(choice)
        if kind is None:
            return None
        try:
            Breakdown.parse(PRINCIPAL, log_group=group)
        except BreakdownError as e:  # the log group: nothing the user types can fix it
            print(f"  {e}")
            return None
        # Without the group itself: the report reads it from the logging configuration again,
        # and so says what to fix there (not --log-group, never passed) if the group is gone
        if kind == PRINCIPAL:
            return Breakdown.parse(PRINCIPAL)
        while True:
            if kind in (TAG, METADATA):
                key = input(f"{'Tag' if kind == TAG else 'Metadata'} key (Enter for no breakdown): ").strip()
                if not key:  # a way out for a choice made by mistake, or a key not known
                    return None
            else:
                key = ''
            try:
                return Breakdown.parse(f"{kind}:{key}" if key else kind)
            except BreakdownError as e:  # only the key is left to be wrong
                print(f"  {e}")

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
        - On-demand custom model deployment: its ARN, ID or name
        - Custom Model Import model: its ARN, ID or name (analyzed without limits)

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
            resource = arn_resource(value)
            kind, _, ident = resource.partition('/')
            arn_region = region_from_arn(value)
            if arn_region and self.region and arn_region != self.region:
                logger.error(f"ARN region {arn_region} does not match the analysis region {self.region}")
                sys.exit(1)
            if kind == 'application-inference-profile':
                return self._application_profile_config(value)
            if kind == DEPLOYMENT_KIND and ident:
                return self._custom_deployment_config(value)
            if kind == IMPORTED_KIND and ident:
                return self._imported_model_config(value)
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
        elif not known_model and not prefix and \
                CUSTOM_ENDPOINT in (fm_endpoints(self._load_fm_list(self.region), base_model_id) or set()):
            # Listed only for its custom deployment quotas (customizable, or no longer listed by
            # Bedrock): still analyzed, as other unknown endpoints are, but not silently
            logger.warning(f"  WARNING: {value} has no on-demand endpoint in {self.region}; the usage of "
                           f"models customized from it is under their custom model deployments (pass a "
                           f"deployment ARN, ID or name with -m, or choose 'Custom model deployments')")
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

    def _custom_deployment_config(self, deployment_arn, summary=None):
        """Analysis target for an on-demand custom model deployment.

        Its usage is reported under the deployment ARN; its limits are the base model's
        custom model deployment quotas (the fm-list's 'custom' endpoint of that model).
        ``summary`` is its list_deployments entry; without it the deployment (passed by ARN)
        is read. When that fails (no permission, or a deleted deployment whose usage CloudWatch
        still keeps), the listing may still have it, and otherwise its ARN still gives the
        metrics, so it is analyzed without limits. Its base model only gives the limits: when
        that cannot be read, the base model ID in the custom model ARN is used, or none.
        """
        fetcher = self._get_profile_fetcher()
        base, reason = None, None
        if summary is None:
            try:
                summary = fetcher.read_custom_deployment(deployment_arn)
            except Exception as e:
                if missing_deployment_api(e):
                    logger.error(f"Custom model deployments need boto3 1.39.7 or later: {e}")
                    sys.exit(1)
                if not isinstance(e, AWS_ERRORS):
                    raise
                # The listing may still have it (same ARN: a name may equal another's ID)
                summary = next((d for d in self._custom_deployments() if d['arn'] == deployment_arn), None)
                if summary is None:
                    hint = troubleshooting_hint(e, self.region) if is_access_denied(e) else None
                    if hint:
                        logger.warning(f"  Hint: {hint}")
                    summary = {'arn': deployment_arn, 'name': deployment_short_id(deployment_arn), 'model_arn': None}
                    reason = (f"could not be read ({e}); if it was deleted, the report still shows the usage "
                              f"CloudWatch keeps for it")
        arn, name = summary['arn'], summary['name']
        fetcher.note_deployment_name(arn, name)
        if summary.get('status') and not is_active(summary):
            logger.warning(f"  WARNING: custom model deployment {name} is {summary['status']}, not Active; "
                           f"it serves no traffic, so its report may show no usage")
        if reason is None:
            try:
                base = fetcher.deployment_base_model(summary.get('model_arn'))
                # GetCustomModel names no base model, or the deployment names no custom model
                reason = "has no foundation base model" if summary.get('model_arn') else "names no custom model"
            except AWS_ERRORS as e:
                base = base_model_id_in_arn(summary.get('model_arn'))
                reason = f"has a custom model that could not be read ({e})"
                if base:
                    logger.info(f"  Could not read the custom model of deployment {name} ({e}); "
                                f"using the base model its ARN names")
        if base:
            logger.info(f"  Custom model deployment {name} ({deployment_short_id(arn)}) is based on {base}")
        else:
            logger.warning(f"  WARNING: custom model deployment {name} {reason}; "
                           f"the report shows its usage without limits")
            base = deployment_short_id(arn)
        return {'model_id': base, 'profile_prefix': CUSTOM_ENDPOINT, 'application_profile_ids': [arn]}

    def _imported_model_config(self, arn, summary=None):
        """Analysis target for a Custom Model Import model: its usage under its ARN, without
        limits (imported models have no per-model token or request quotas). Its ID stands in
        for the model ID, as for a deployment without a known base model. Without ``summary``
        (passed by ARN) its name comes from the region's listing (one request for all of them),
        or else from reading it."""
        fetcher = self._get_profile_fetcher()
        if summary is None and isinstance(fetcher, InferenceProfileFetcher):
            summary = next((m for m in self._imported_models() if m['arn'] == arn), None)
        if summary is None and isinstance(fetcher, InferenceProfileFetcher):
            try:
                summary = fetcher.read_imported_model(arn)
            except Exception as e:
                if not deployment_read_error(e):
                    raise
                hint = troubleshooting_hint(e, self.region) if is_access_denied(e) else None
                if hint:
                    logger.warning(f"  Hint: {hint}")
                # The name is cosmetic: the ARN still gives the metrics
                logger.warning(f"  WARNING: imported model {deployment_short_id(arn)} could not be read ({e}); "
                               f"if it was deleted, the report still shows the usage CloudWatch keeps for it")
        name = (summary or {}).get('name') or deployment_short_id(arn)
        if isinstance(fetcher, InferenceProfileFetcher):
            fetcher.note_deployment_name(arn, name)
        logger.info(f"  Imported model {name} ({deployment_short_id(arn)}): Custom Model Import models have no "
                    f"per-model token or request quotas (Bedrock scales the model copies that serve them), "
                    f"so the report shows its usage without limits")
        return {'model_id': deployment_short_id(arn), 'profile_prefix': CUSTOM_ENDPOINT, 'application_profile_ids': [arn]}

    def _application_profile_config(self, identifier, profile=None):
        if profile is None:
            try:
                profile = self._get_profile_fetcher().resolve_application_profile(identifier)
            except Exception as e:
                if not self._listing_error():
                    raise  # a bug, not missing permissions
                # The ID or name of a custom model deployment needs no profile listing
                config = self._deployment_target(identifier)
                if config:
                    return config
                # An application profile ID or ARN cannot be analyzed without the listings
                logger.error(f"Could not list {self._failed_listing()} inference profiles in {self.region}, "
                             f"so {identifier} cannot be resolved: {e}")
                self._report_deployment_listing_error(identifier)
                sys.exit(1)
        if profile is None:
            fetcher = self.profile_fetcher
            if isinstance(fetcher, InferenceProfileFetcher) and fetcher.routes_to_no_foundation_model(identifier):
                # It exists (e.g. a copy of a custom model): say why it cannot be analyzed
                logger.error(f"Application inference profile {identifier} routes to no foundation model; "
                             f"this tool has no metrics or quotas for it")
                sys.exit(1)
            config = self._deployment_target(identifier)
            if config:
                return config
            logger.error(f"Application inference profile not found in {self.region}: {identifier}")
            self._report_deployment_listing_error(identifier)
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

        # Only active deployments serve traffic: a Creating or Failed one has no usage to report
        listed = self._custom_deployments()
        deployments = [d for d in listed if is_active(d)]
        if len(deployments) < len(listed) and not self._inactive_deployments_noted:
            self._inactive_deployments_noted = True  # once per session: the listing is cached
            logger.info(f"  {len(listed) - len(deployments)} custom model deployment(s) in {region} are not "
                        f"active (Creating or Failed) and are not offered")
        modes = ['A foundation model (includes the application inference profiles created from it)']
        if app_profiles:
            modes.append(f'Specific application inference profiles ({len(app_profiles)} in {region})')
        if deployments:
            modes.append(f'Custom model deployments ({len(deployments)} in {region})')
        imported = self._imported_models()
        if imported:
            modes.append(f'Imported models (Custom Model Import, {len(imported)} in {region})')
        if len(modes) > 1:
            mode = select_from_list("What do you want to analyze?", modes, allow_cancel=False,
                                    input_prompt=f"\nSelect (1-{len(modes)}): ")
            if mode.startswith('Specific'):
                return self._select_application_profiles(app_profiles)
            if mode.startswith('Custom'):
                return self._select_custom_deployments(deployments)
            if mode.startswith('Imported'):
                indices = self._pick("Imported models", [
                    f"{m['name']} ({deployment_short_id(m['arn'])})" for m in imported], 'models')
                return [self._imported_model_config(imported[i]['arn'], imported[i]) for i in indices]

        config = self._select_model(region)
        return [config] if config else []

    def _custom_deployments(self) -> List[Dict]:
        """The region's custom model deployments ([] when there are none or they cannot be listed).

        A listing failure (API or network error, or a boto3 without the API) only hides the
        choice, and is said once per session; a bug is raised.
        """
        fetcher = self._get_profile_fetcher()
        if not isinstance(fetcher, InferenceProfileFetcher):
            return []  # another fetcher (an API caller's) knows no deployments
        try:
            return fetcher.list_custom_deployments()
        except Exception as e:
            if not deployment_read_error(e):
                raise
            if not self._deployment_listing_noted:
                self._deployment_listing_noted = True
                need = "need boto3 1.39.7 or later" if missing_deployment_api(e) else f"could not be listed ({e})"
                logger.info(f"  Custom model deployments {need}; they are not offered")
            return []

    def _deployment_target(self, identifier) -> Optional[Dict]:
        """The config of the custom model deployment with this ID or name (the ones the
        deployment list shows), or None. Without the listing it is read directly, as
        GetCustomModelDeployment also takes an ID or name."""
        deployment = self._find_custom_deployment(identifier)
        fetcher = self._get_profile_fetcher()
        if deployment is None and not identifier.startswith('arn:') and \
                isinstance(fetcher, InferenceProfileFetcher) and fetcher.custom_deployments_error is not None:
            try:
                deployment = fetcher.read_custom_deployment(identifier)
            except AWS_ERRORS as e:
                logger.debug(f"{identifier} is not a readable custom model deployment either: {e}")
        if deployment:
            return self._custom_deployment_config(deployment['arn'], deployment)
        if identifier.startswith('arn:'):
            return None
        imported = next(
            (m for m in self._imported_models() if identifier in (deployment_short_id(m['arn']), m['name'])), None)
        if imported is None and isinstance(fetcher, InferenceProfileFetcher) and fetcher.imported_models_error is not None:
            try:  # GetImportedModel also takes a name
                imported = fetcher.read_imported_model(identifier)
            except Exception as e:
                if not deployment_read_error(e):
                    raise
                logger.debug(f"{identifier} is not a readable imported model either: {e}")
        return self._imported_model_config(imported['arn'], imported) if imported else None

    def _imported_models(self) -> List[Dict]:
        """The region's Custom Model Import models ([] when there are none or they cannot be
        listed; a listing failure is said once per session, a bug is raised)."""
        fetcher = self._get_profile_fetcher()
        if not isinstance(fetcher, InferenceProfileFetcher):
            return []
        try:
            return fetcher.list_imported_models()
        except Exception as e:
            if not deployment_read_error(e):
                raise
            if not self._imported_listing_noted:
                self._imported_listing_noted = True
                logger.info(f"  Imported models could not be listed ({e}); they are not offered")
            return []

    def _report_deployment_listing_error(self, identifier):
        """Say so when ``identifier`` may be a deployment or imported model that could not be listed."""
        fetcher = self._get_profile_fetcher()
        if not isinstance(fetcher, InferenceProfileFetcher) or identifier.startswith('arn:'):
            return
        for what, error in (("Custom model deployments", fetcher.custom_deployments_error),
                            ("Imported models", fetcher.imported_models_error)):
            if error is not None:
                logger.error(f"  ({what} could not be listed either, so it may be one: {error})")

    def _find_custom_deployment(self, identifier) -> Optional[Dict]:
        """The region's custom model deployment with this ID or name, or None (an ARN needs
        no listing: deployment ARNs are read directly)."""
        if identifier.startswith('arn:'):
            return None
        return next((d for d in self._custom_deployments()
                     if identifier in (deployment_short_id(d['arn']), d['name'])), None)

    @staticmethod
    def _pick(title: str, lines: List[str], what: str) -> List[int]:
        """Print a numbered list and read a selection ('1,3-4' or 'all'): the chosen indices."""
        print(f"\n{title}:")
        for i, line in enumerate(lines, 1):
            print(f"  {i}. {line}")
        while True:
            try:
                return parse_selection(input(f"\nSelect {what} (e.g. 1,3-4 or all): "), len(lines))
            except ValueError as e:
                print(f"Please enter valid numbers: {e}")

    def _select_custom_deployments(self, deployments) -> List[Dict]:
        """Pick one or more custom model deployments by number."""
        indices = self._pick("Custom model deployments", [
            f"{d['name']} ({deployment_short_id(d['arn'])}) - {d['status']}" for d in deployments], 'deployments')
        chosen = [deployments[i] for i in indices]
        # One base-model read per custom model, in parallel (results and errors are kept)
        fetcher = self._get_profile_fetcher()
        if isinstance(fetcher, InferenceProfileFetcher):
            fetcher.read_base_models([d.get('model_arn') for d in chosen])
        # Listed: no read can fail and end the session (a base model that cannot be read only
        # leaves out the limits)
        return [self._custom_deployment_config(d['arn'], d) for d in chosen]

    def _select_application_profiles(self, app_profiles) -> List[Dict]:
        """Pick one or more application inference profiles by number."""
        lines = []
        for app in app_profiles:
            # A guessed source the region no longer lists is marked here, before it is picked
            if app['profile_prefix'] is None:
                # A base-model copy of a model the fm-list knows without an on-demand endpoint
                listed = fm_endpoints(self._load_fm_list(self.region), app['model_id'])
                retired = listed is not None and 'base' not in listed
            else:
                retired = app['profile_prefix'] != UNKNOWN_SOURCE and app['source'] and \
                    not self._is_system_profile(app['source'])
            note = f" (not offered in {self.region} any more)" if retired else ""
            lines.append(f"{app['name']} ({app['id']}) - based on {app['source'] or 'an unknown endpoint'}{note}")
        indices = self._pick("Application inference profiles", lines, 'profiles')
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
        # (not 'custom', custom model deployment quotas: deployments are picked on their own)
        endpoints = invokable_endpoint_keys(selected_model)

        # Derive inference profiles from endpoints (exclude 'base')
        inference_profiles = sorted(k for k in endpoints if k != 'base')

        if not endpoints:
            if CUSTOM_ENDPOINT in endpoint_keys(selected_model):
                # Listed for its custom deployment quotas: its usage is under the deployments
                logger.info(f"\n  {model_id} has no on-demand or inference profile endpoint in {region}. "
                            f"To analyze models customized from it, choose 'Custom model deployments' "
                            f"(offered when the region has an active one) or pass a deployment ARN with -m.")
                return None  # not incomplete metadata: the manual entry's warning would be wrong
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
