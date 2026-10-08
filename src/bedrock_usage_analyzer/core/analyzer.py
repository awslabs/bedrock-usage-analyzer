# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Main orchestrator for Bedrock token usage analysis"""

import hashlib
import numpy as np
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from bedrock_usage_analyzer.core.breakdown import BreakdownBuilder
from bedrock_usage_analyzer.core.profile_fetcher import UNKNOWN_SOURCE, InferenceProfileFetcher, in_parallel
from bedrock_usage_analyzer.sync.quota_rules import scrub_conflicting
from bedrock_usage_analyzer.core.metrics_fetcher import PERIOD_DAYS, CloudWatchMetricsFetcher
from bedrock_usage_analyzer.core.output_generator import (
    APPLICATION_PROFILE_SCOPE, DEPLOYMENT_SCOPE, IMPORTED_SCOPE, OutputGenerator)
from bedrock_usage_analyzer.aws.bedrock import (
    endpoint_id, get_endpoint_quota_keywords, get_regional_profile_prefixes)
from bedrock_usage_analyzer.aws.custom_models import deployment_short_id, is_imported
from bedrock_usage_analyzer.aws.client_factory import create_client
from bedrock_usage_analyzer.aws.servicequotas import (
    QUOTA_ERROR, QUOTA_MISSING, QUOTA_OK, list_quota_codes, lookup_quota)
from bedrock_usage_analyzer.utils.yaml_handler import (
    CUSTOM_ENDPOINT, IMPORTED_ENDPOINT, endpoint_quotas, fm_endpoints, has_endpoint, load_fm_list, profile_endpoints)
from bedrock_usage_analyzer.utils.partition import get_region_info, get_service_quota_url

logger = logging.getLogger(__name__)

# Above this many distinct quota codes per run, the region's quotas are listed once. The
# listing pages sequentially through every Bedrock quota of the region (hundreds), so it only
# beats parallel per-code lookups (4 at a time) for a few dozen codes
QUOTA_LISTING_THRESHOLD = 40


def _base_unknown(model_id, deployment_arns) -> bool:
    """True for a deployment target without a known base model: its deployment ID stands in
    for the model ID (UserInputs._custom_deployment_config)."""
    return model_id in {deployment_short_id(a) for a in deployment_arns}

class BedrockAnalyzer:
    """Main orchestrator for Bedrock token usage analysis"""
    
    TIME_PERIODS = ["1hour", "1day", "7days", "14days", "30days"]
    
    def __init__(self, region, granularity_config, profile_fetcher=None, fm_models=None,
                 breakdown=None, account=None):
        self.region = region
        self.granularity_config = granularity_config
        # Optional usage breakdown by caller, from the model invocation logs (core/breakdown.py)
        self.breakdown = breakdown
        self.account = account

        # Get local timezone - use system's local timezone
        local_dt = datetime.now().astimezone()
        self.local_tz = local_dt.tzinfo
        offset = local_dt.strftime('%z')
        self.tz_offset = f"{offset[:3]}:{offset[3:]}"  # +08:00 format
        self.tz_api_format = offset[:5]  # +0800 format for API

        # botocore picks the endpoint for the region's partition (commercial, GovCloud, China)
        self.cloudwatch_client = create_client('cloudwatch', region)
        # Region's fm-list: the one parsed during input collection, else read on first lookup
        self._fm_models = fm_models
        # One ListServiceQuotas pass pays off only for many codes; a short run uses direct
        # GetServiceQuota calls (set in analyze())
        self._quota_listings = {}
        self._quota_results = {}  # quota code -> (status, quota) of this run's region
        self._use_quota_listing = False
        # Reuse the fetcher from input collection so profiles are listed only once
        if profile_fetcher is not None:
            self.profile_fetcher = profile_fetcher
            self.bedrock_client = profile_fetcher.bedrock_client
        else:
            self.bedrock_client = create_client('bedrock', region)
            self.profile_fetcher = InferenceProfileFetcher.for_region(self.bedrock_client, self._fm_list(), region)
        self.metrics_fetcher = CloudWatchMetricsFetcher(self.cloudwatch_client, self.tz_api_format)
        self.output_generator = None  # Initialized in analyze() with output_dir
    
    def _system_profile_listed(self, profile_id) -> bool:
        """True unless the region's system profiles were listed and do not include it."""
        try:
            return self.profile_fetcher.is_system_profile(profile_id)
        except Exception:
            return True  # cannot tell; keep the refresh hint

    def _endpoint_listed(self, model_id, profile_prefix) -> bool:
        """True when the region's fm-list has this model with this endpoint."""
        return has_endpoint(self._fm_list(), model_id, profile_prefix)

    def _fm_list(self):
        """The region's fm-list models, parsed once per run (every target reads the same file)."""
        if self._fm_models is None:
            self._fm_models = load_fm_list(self.region) or []
        return self._fm_models

    def _load_quota_codes(self, model_id, profile_prefix=None, quiet=False):
        """Load quota codes for a model from FM list based on endpoint
        
        Args:
            model_id: Base model ID
            profile_prefix: Endpoint prefix (e.g., 'us', 'eu', 'global') or None for base endpoint
        
        Returns:
            dict: Quota codes for the specified endpoint (tpm, rpm, tpd, concurrent)
        """
        if profile_prefix == UNKNOWN_SOURCE:
            return {}
        endpoint_key = profile_prefix if profile_prefix else 'base'
        for model in self._fm_list():
            if model['model_id'] != model_id:
                continue
            # The guarded walk every reader uses (hand-edited 'us: null', 'quotas: TODO');
            # old-format model-level quotas were migrated to 'base' by load_fm_list
            quotas = dict(dict(endpoint_quotas(model)).get(endpoint_key) or {})
            # Skip codes that contradict this model or endpoint (e.g. saved by an older
            # version, before the mapping checks existed) instead of showing another limit
            for metric, quota, reason in scrub_conflicting(
                    model_id, endpoint_key, quotas, set(get_regional_profile_prefixes())):
                if not quiet:
                    logger.info(f"  Ignoring {metric} quota {quota.get('code')}: {reason}")
            return quotas

        return {}
    
    def _fetch_quotas(self, model_id, quota_codes, profile_prefix=None):
        """Fetch quota values from Service Quotas API
        
        Args:
            model_id: Model ID
            quota_codes: Dictionary of quota type to {code, name} or None
            profile_prefix: Endpoint prefix (e.g., 'us', 'eu', 'global') or None for base
        
        Returns:
            dict: Quota metadata (tpm, rpm, tpd) - each containing {value, code, name, url}
        """
        quotas = {'tpm': None, 'rpm': None, 'tpd': None, 'concurrent': None}

        if not quota_codes:
            return quotas

        logger.info(f"  Fetching quotas from Service Quotas API...")
        wanted = []
        for quota_type, quota_data in quota_codes.items():
            # Handle new structure: {code: L-xxx, name: "..."} or null
            if not (quota_data and isinstance(quota_data, dict) and quota_data.get('code')):
                continue
            key = next((k for k in quotas if k in quota_type.lower()), None)
            if key is not None:
                wanted.append((quota_type, quota_data, key))

        def lookup(code):
            # One lookup per code and run: targets sharing an endpoint reuse the result. A failed
            # lookup is not kept, so the next target sharing the code tries again
            if code in self._quota_results:
                return self._quota_results[code]
            result = lookup_quota(code, self.region, self._quota_listings, use_listing=self._use_quota_listing)
            if result[0] != QUOTA_ERROR:
                self._quota_results[code] = result
            return result

        if self._use_quota_listing and self.region not in self._quota_listings:
            # The region's listing once, before the pool (never listed again from the threads)
            self._quota_listings[self.region] = list_quota_codes(self.region, quiet_denied=True)
        # Per-code lookups (up to two calls each) run in parallel, as confirm_statuses does
        # (no pool when at most one code is not yet cached: nothing to run in parallel)
        uncached = sorted({item[1]['code'] for item in wanted} - set(self._quota_results))
        fetched = {}  # this target's results, failed lookups included (looked up once here)
        if len(uncached) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(uncached))) as pool:
                fetched = dict(zip(uncached, pool.map(lookup, uncached)))
        for code in {item[1]['code'] for item in wanted} - set(fetched):
            fetched[code] = lookup(code)
        results = [fetched[item[1]['code']] for item in wanted]
        for (quota_type, quota_data, key), (status, quota) in zip(wanted, results):
            code = quota_data['code']
            if status == QUOTA_OK and quota.get('Value') is None:
                logger.info(f"  Warning: {quota_type} quota {code} has no value; not shown")
            elif status == QUOTA_OK:
                quotas[key] = {'value': quota['Value'], 'code': code, 'name': quota_data.get('name'),
                               'url': get_service_quota_url(self.region, 'bedrock', code)}
            elif status == QUOTA_MISSING:
                # Not shown. 'bua refresh quota-index' removes it from a user copy of the list;
                # a bundled list is corrected in the next release
                logger.info(f"  Warning: {quota_type} quota {code} does not exist in {self.region}; "
                            f"shown without this limit")
            else:
                logger.info(f"  Warning: Could not fetch {quota_type} quota {code} for {model_id}")
        
        # Apply 2x multiplier for TPD on regional cross-region profiles
        regional_profile_prefixes = set(get_regional_profile_prefixes())
        if profile_prefix in regional_profile_prefixes and quotas['tpd'] and quotas['tpd']['value'] is not None:
            quotas['tpd']['value'] = quotas['tpd']['value'] * 2
        
        return quotas
    
    # This aggregates values within 1 Bedrock application profile
    # The aggregation across application inference profiles is implemented in metrics_fetcher.py
    def _calculate_stats_from_time_series(self, ts_data, time_period):
        """Calculate statistics from time series data"""
        stats = self.metrics_fetcher._initialize_metrics(time_period)
        
        for metric_name in ts_data:
            if 'values' in ts_data[metric_name] and ts_data[metric_name]['values']:
                # Filter out None values (from sparse data handling)
                values = [v for v in ts_data[metric_name]['values'] if v is not None]
                stats[metric_name] = {
                    'values': values,
                    'p50': np.percentile(values, 50) if values else 0.0,
                    'p90': np.percentile(values, 90) if values else 0.0,
                    'count': len(values),
                    'sum': sum(values),
                    'avg': np.mean(values) if values else 0.0
                }
        
        return stats
    
    def _calculate_contributions(self, model_results, time_series_data, profile_names, profile_metadata):
        """Calculate average contributions for each profile per period"""
        logger.info(f"  Calculating profile contributions...")
        contributions = {}
        
        for time_period in model_results.keys():
            period_contributions = []
            
            for profile_id, stats in model_results[time_period].items():
                if profile_id == '__AGGREGATED__':
                    continue
                
                profile_name = profile_names.get(profile_id, profile_id)
                metadata = profile_metadata.get(profile_id, {'id': 'N/A', 'tags': {}})
                
                # Get p50, p90, avg for each metric
                contribution = {
                    'profile_id': profile_id,
                    'profile_name': profile_name,
                    'profile_arn_id': metadata['id'],
                    'profile_tags': metadata['tags'],
                    'tpm_p50': stats.get('TPM', {}).get('p50', 0),
                    'tpm_p90': stats.get('TPM', {}).get('p90', 0),
                    'tpm_avg': stats.get('TPM', {}).get('avg', 0),
                    'rpm_p50': stats.get('RPM', {}).get('p50', 0),
                    'rpm_p90': stats.get('RPM', {}).get('p90', 0),
                    'rpm_avg': stats.get('RPM', {}).get('avg', 0),
                    'tpd_p50': stats.get('TPD', {}).get('p50', 0) if time_period != '1hour' else 0,
                    'tpd_p90': stats.get('TPD', {}).get('p90', 0) if time_period != '1hour' else 0,
                    'tpd_avg': stats.get('TPD', {}).get('avg', 0) if time_period != '1hour' else 0,
                    'throttles': stats.get('InvocationThrottles', {}).get('sum', 0)
                }
                
                period_contributions.append(contribution)
            
            # Sort by TPM average (descending)
            period_contributions.sort(key=lambda x: x['tpm_avg'], reverse=True)
            contributions[time_period] = period_contributions
        
        return contributions
    
    @staticmethod
    def _scope_key(model_config):
        """Cache key for one analysis target (model, endpoint and optional profile subset)."""
        app_ids = tuple(sorted(model_config.get('application_profile_ids') or ()))
        model_id = model_config['model_id']
        if model_config.get('profile_prefix') == IMPORTED_ENDPOINT and app_ids and all(map(is_imported, app_ids)):
            # Imported reports ignore the caller's model_id: the same models are one target
            model_id = deployment_short_id(app_ids[0])
        return (model_id, model_config.get('profile_prefix'), app_ids)

    @staticmethod
    def _imported_target(key) -> bool:
        """True when a target (its scope key) is Custom Model Import models (profile_prefix
        'imported'), reported under their ARNs without limits. Raises ValueError for imported
        model ARNs under any other kind, or other ARNs under 'imported': custom model
        deployments are measured against their base model's quotas, imported models against none."""
        model_id, profile_prefix, app_ids = key
        if profile_prefix == IMPORTED_ENDPOINT:
            if not app_ids or not all(is_imported(a) for a in app_ids):
                raise ValueError(f"target {model_id}: profile_prefix '{IMPORTED_ENDPOINT}' needs imported "
                                 f"model ARNs in application_profile_ids")
            return True
        if any(is_imported(a) for a in app_ids):
            raise ValueError(f"target {model_id}: pass imported model ARNs with profile_prefix "
                             f"'{IMPORTED_ENDPOINT}', as separate targets")
        return False

    def _warn_other_deployments(self, base, deployment_arns):
        """Tell the user when other active deployments share the base model's custom
        deployment quotas: those limits are account-wide sums, so this report's
        utilization leaves their usage out."""
        fetcher = self.profile_fetcher
        if not isinstance(fetcher, InferenceProfileFetcher) or _base_unknown(base, deployment_arns):
            return  # no base model: no shared quotas
        others = fetcher.other_deployments_of(base, deployment_arns)
        if others:
            names = ', '.join(f"{d['name']} ({deployment_short_id(d['arn'])})" for d in others)
            logger.info(f"  Note: {len(others)} other active custom model deployment(s) of {base} share its "
                        f"custom deployment quotas, so the utilization shown leaves their usage out: {names}. "
                        f"Select them too (e.g. 'all' under 'Custom model deployments') for the account-wide view.")

    def _warn_other_sources(self, model_id, profile_prefix, final_model_ids):
        """Tell the user when their application profiles sit under a different endpoint."""
        if len(final_model_ids) > 1:
            return
        others = self.profile_fetcher.other_sources_for_model(model_id, profile_prefix)
        if others:
            where = ', '.join(f"{count} on '{key}'" for key, count in sorted(others.items()))
            logger.info(f"  Note: no application inference profile of {model_id} is based on "
                        f"'{profile_prefix or 'base'}', but {where}. Choose that endpoint in the "
                        f"menu or pass its endpoint ID with -m, or pass the application profile "
                        f"ID/ARN with -m to analyze it directly.")

    def analyze(self, models, output_dir: str = 'results'):
        """Analyze token usage for given models

        Args:
            models: List of model configurations. Each has 'model_id' and
                'profile_prefix', and optionally 'application_profile_ids' to
                analyze only those application inference profiles.
            output_dir: Directory to save results
        """
        self.output_generator = OutputGenerator(output_dir)

        # Step 0: Discover all profiles once for all models
        logger.info(f"\n{'='*80}")
        logger.info(f"Discovering inference profiles for {len(models)} model(s)...")
        logger.info(f"{'='*80}")

        all_profiles_map = {}  # {scope_key: (final_model_ids, profile_names, profile_metadata)}
        for model_config in models:
            self._imported_target(self._scope_key(model_config))  # a wrong target fails before any AWS call

        for model_config in models:
            key = self._scope_key(model_config)
            if key in all_profiles_map:
                continue
            model_id, profile_prefix, app_ids = key
            final_model_ids, profile_names, profile_metadata = self.profile_fetcher.find_profiles(
                model_id, profile_prefix, application_profile_ids=list(app_ids) or None)
            all_profiles_map[key] = (final_model_ids, profile_names, profile_metadata)

            profile_list = [profile_names.get(pid, pid) for pid in final_model_ids]
            logger.info(f"  {model_id} ({profile_prefix or 'base'}): {len(final_model_ids)} profile(s) - {', '.join(profile_list)}")
            if not app_ids:
                self._warn_other_sources(model_id, profile_prefix, final_model_ids)
            elif profile_prefix == CUSTOM_ENDPOINT:
                self._warn_other_deployments(model_id, app_ids)

        logger.info(f"Profile discovery complete.\n")

        # Many quota codes to look up (several models): one paginated listing of the region's
        # quotas is cheaper than one GetServiceQuota call each
        # Counted with the same reader the lookups use (it also reads the legacy model-level
        # 'quotas' of a base endpoint)
        targets = {(model_id, prefix) for model_id, prefix, _ in all_profiles_map if prefix != IMPORTED_ENDPOINT}
        codes = {q['code'] for model_id, prefix in targets
                 for q in self._load_quota_codes(model_id, prefix, quiet=True).values()
                 if isinstance(q, dict) and q.get('code')}
        self._use_quota_listing = len(codes) > QUOTA_LISTING_THRESHOLD

        region_info = get_region_info(self.region)
        processed = set()
        builder = None
        if self.breakdown is not None:
            # One set of Logs Insights queries for every target of the run: the log group is
            # scanned once, whatever the number of reports
            builder = BreakdownBuilder(self.breakdown, self.region, self.bedrock_client, self.metrics_fetcher,
                                       self._calculate_stats_from_time_series, self.account,
                                       parallel=in_parallel,
                                       known_models=[m.get('model_id') for m in self._fm_list()])
            run_ids = sorted({cw_id for ids, _, _ in all_profiles_map.values() for cw_id in ids})
            if not run_ids:  # no target has a ModelId to report on: nothing to read the logs for
                builder = None
            else:
                # The real clock: log records expire by it (the breakdown's end is rounded down)
                builder.prepare(run_ids, datetime.now(timezone.utc), max(PERIOD_DAYS[p] for p in self.granularity_config))

        # Process each model
        for model_config in models:
            key = self._scope_key(model_config)
            if key in processed:
                continue
            processed.add(key)
            model_id, profile_prefix, app_ids = key
            imported = self._imported_target(key)

            logger.info(f"\n{'='*80}")
            logger.info(f"Processing model: {model_id}")
            logger.info(f"{'='*80}")

            # Step 1: Get profiles from cache
            final_model_ids, profile_names, profile_metadata = all_profiles_map[key]
            logger.info(f"Using {len(final_model_ids)} profile(s)")
            if not final_model_ids:
                logger.info("  No matching profiles; skipping.")
                continue

            # Step 2: Fetch quotas
            # Imported models have no quotas, whatever model ID an API caller gives them
            quota_codes = {} if imported else self._load_quota_codes(model_id, profile_prefix)
            retired = profile_prefix not in (None, UNKNOWN_SOURCE, CUSTOM_ENDPOINT, IMPORTED_ENDPOINT) and \
                not self._system_profile_listed(endpoint_id(model_id, profile_prefix))
            if profile_prefix is None and app_ids:
                # A base-model copy of a model the fm-list knows without an on-demand endpoint:
                # that endpoint was retired (no refresh can map its limits)
                listed = fm_endpoints(self._fm_list(), model_id)
                retired = listed is not None and 'base' not in listed
            if retired:
                # e.g. a copy of a retired au.* profile; an older fm-list may still map its quotas
                ending = "the limits shown come from the saved mapping" if any(quota_codes.values()) \
                    else "the report will show usage without limits"
                logger.info(f"  {endpoint_id(model_id, profile_prefix)} is not offered in "
                            f"{self.region}; {ending}")
            elif imported:
                # Said here, where CLI and API runs both pass
                logger.info("  Custom Model Import models have no per-model token or request quotas (Bedrock "
                            "scales the model copies that serve them), so the report shows usage without limits")
            elif not any(quota_codes.values()) and profile_prefix == CUSTOM_ENDPOINT:
                if _base_unknown(model_id, app_ids):
                    # No base model: the deployment ID stands in for it (said when it was
                    # selected), so no refresh can map limits
                    fix = ""
                elif self._endpoint_listed(model_id, profile_prefix):
                    # fm-quotas maps them only where Service Quotas lists custom deployment quotas
                    fix = (f" If Service Quotas lists custom model deployment quotas for it in {self.region}, "
                           f"bua refresh fm-quotas {self.region} maps them")
                else:
                    # A model list from before custom model deployments were mapped
                    fix = f" To map them: bua refresh fm-list {self.region}, then bua refresh fm-quotas {self.region}"
                logger.info(f"  No custom model deployment quotas are known for {model_id} in "
                            f"{self.region}; the report will show usage without limits.{fix}")
            elif not any(quota_codes.values()) and profile_prefix != UNKNOWN_SOURCE:
                profiles = profile_endpoints(self._fm_list(), model_id)
                if (profile_prefix or 'base') not in get_endpoint_quota_keywords():
                    # e.g. a one-region country prefix that prefix-mapping.yml does not know:
                    # fm-quotas has no quota keyword for it, so refreshing cannot map one
                    fix = None
                elif self._endpoint_listed(model_id, profile_prefix):
                    # A new model may have no quotas in Service Quotas yet (zai.glm-5.3): then
                    # refreshing maps none either
                    fix = (f"bua refresh fm-quotas {self.region} (if Service Quotas lists quotas for it; "
                           f"a new model may have none yet)")
                elif profile_prefix is None and profiles:
                    # No on-demand endpoint: refreshing cannot add one, its profiles have the limits
                    fix = "analyze one of its inference profiles instead: " + \
                        ', '.join(endpoint_id(model_id, p) for p in profiles)
                elif profile_prefix is None and fm_endpoints(self._fm_list(), model_id) == {CUSTOM_ENDPOINT}:
                    # Listed only for customization: no refresh adds an on-demand endpoint
                    fix = "analyze the custom model deployments of models customized from it instead"
                else:
                    # fm-quotas only maps endpoints already in the model list
                    fix = f"bua refresh fm-list {self.region}, then bua refresh fm-quotas {self.region}"
                if fix is None:
                    logger.info(f"  No quota codes mapped for this endpoint in {self.region}; the report will "
                                f"show usage without limits. '{profile_prefix}' endpoints have no quota type "
                                f"in prefix-mapping.yml, so bua refresh fm-quotas cannot map them yet.")
                else:
                    logger.info(f"  No quota codes mapped for this endpoint in {self.region}; the report will "
                                f"show usage without limits. To map them: {fix}")
            quotas = self._fetch_quotas(model_id, quota_codes, profile_prefix)
            if any(quotas.values()):
                logger.info(f"  Quotas: TPM={quotas['tpm']}, RPM={quotas['rpm']}, TPD={quotas['tpd']}, "
                            f"concurrent={quotas.get('concurrent')}")

            # Step 3: Fetch all data upfront with configured granularities
            # Data reuse optimization: if all periods use same granularity, only fetch once
            # If granularities differ, fetch separately for each unique granularity
            logger.info(f"  Fetching data with configured granularities (parallel)...")
            fetched_data_all_profiles = self.metrics_fetcher.fetch_all_data_mixed_granularity(
                final_model_ids,
                self.granularity_config
            )

            model_results = {}
            time_series_data = {}

            # Step 4: Process each time period
            for time_period in self.TIME_PERIODS:
                logger.info(f"  Processing {time_period}...")

                period_stats = {}
                period_time_series = {}

                try:
                    for final_model_id in final_model_ids:
                        # Slice data from fetched datasets
                        if final_model_id in fetched_data_all_profiles:
                            ts_data = self.metrics_fetcher.slice_and_process_data(
                                fetched_data_all_profiles[final_model_id],
                                time_period,
                                self.granularity_config
                            )
                            period_time_series[final_model_id] = ts_data

                            # Calculate statistics from time series data
                            stats = self._calculate_stats_from_time_series(ts_data, time_period)
                            period_stats[final_model_id] = stats

                    # Always create aggregated metrics for consistent template behavior
                    agg_stats = self.metrics_fetcher.aggregate_statistics(period_stats, time_period)
                    agg_ts = self.metrics_fetcher.aggregate_time_series(period_time_series, time_period)

                    period_stats['__AGGREGATED__'] = agg_stats
                    period_time_series['__AGGREGATED__'] = agg_ts

                    model_results[time_period] = period_stats
                    time_series_data[time_period] = period_time_series

                except Exception as e:
                    logger.info(f"\n  ERROR in {time_period} processing:")
                    logger.info(f"  Error type: {type(e).__name__}")
                    logger.info(f"  Error message: {e}")
                    logger.info(f"  Traceback:")
                    traceback.print_exc()
                    raise

            # Step 5: Calculate contributions
            contributions = self._calculate_contributions(model_results, time_series_data, profile_names, profile_metadata)
            breakdown_section = None
            if builder is not None:
                logger.info(f"  Breaking usage down by {self.breakdown.label}...")
                breakdown_section = builder.section(final_model_ids, profile_names, fetched_data_all_profiles,
                                                    self.granularity_config, self.TIME_PERIODS)

            # Step 6: Generate output
            logger.info(f"  Generating output files...")
            end_time_local = datetime.now(self.local_tz)
            scope_label = APPLICATION_PROFILE_SCOPE
            report_model = model_id
            if imported:
                # Named by the imported model IDs, whatever model ID an API caller passed
                ids = [deployment_short_id(a) for a in app_ids]
                report_model = ids[0] if len(ids) == 1 else ', '.join(ids)
                endpoint = f"{ids[0]} (imported model)" if len(ids) == 1 else f"{len(ids)} imported models"
                scope_label = IMPORTED_SCOPE
                file_label = f"imported-model.{ids[0]}" if len(ids) == 1 else \
                    self._file_label("imported", ids, marker='models')
            elif profile_prefix == CUSTOM_ENDPOINT:
                endpoint = f"{model_id} (custom model deployment)"
                scope_label = DEPLOYMENT_SCOPE
                # Deployment IDs, not their ARNs, in the file name
                file_label = self._file_label(f"custom-deployment.{model_id}",
                                              [deployment_short_id(a) for a in app_ids], marker='deployment')
            else:
                endpoint = f"{model_id} (source endpoint unknown)" if profile_prefix == UNKNOWN_SOURCE \
                    else endpoint_id(model_id, profile_prefix)
                file_label = self._file_label(endpoint, app_ids)
            scope = [profile_names.get(pid, pid) for pid in final_model_ids] if app_ids else []

            self.output_generator.generate({
                report_model: {
                    'stats': model_results,
                    'time_series': time_series_data,
                    'quotas': quotas,
                    'profile_names': profile_names,
                    'contributions': contributions,
                    'granularity_config': self.granularity_config,
                    'end_time': end_time_local,
                    'tz_offset': self.tz_offset,
                    'region': self.region,
                    'region_info': region_info,
                    'endpoint': endpoint,
                    'application_profile_scope': scope,
                    'scope_label': scope_label,
                    'imported': imported,
                    'file_label': file_label,
                    'breakdown': breakdown_section,
                }
            })

    @staticmethod
    def _file_label(endpoint, app_ids, marker='app'):
        """Distinct output name per target, so two endpoints of one model do not overwrite each other.

        ``marker`` names what the IDs are: 'app' (application profiles), 'deployment' or
        'models' (imported models: 'imported-models-<id>-<id>').
        """
        if app_ids:
            if len(app_ids) <= 3:
                return f"{endpoint}-{marker}-{'-'.join(app_ids)}"
            # Many targets: a short digest of the sorted IDs keeps different sets apart
            digest = hashlib.sha256('\n'.join(sorted(app_ids)).encode()).hexdigest()[:8]
            if marker == 'models':
                return f"{endpoint}-models-{len(app_ids)}-{digest}"
            kind = {'app': 'profiles'}.get(marker, 'deployments')
            return f"{endpoint}-{marker}-{len(app_ids)}{kind}-{digest}"
        return endpoint


def main():
    """Run the analysis as `bua analyze` does (kept for `python -m ...core.analyzer`)."""
    import sys
    from bedrock_usage_analyzer.__main__ import main as cli_main
    sys.argv = [sys.argv[0], 'analyze', *sys.argv[1:]]
    cli_main()


if __name__ == "__main__":
    main()
