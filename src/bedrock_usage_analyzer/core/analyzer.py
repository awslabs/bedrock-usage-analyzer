# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Main orchestrator for Bedrock token usage analysis"""

import numpy as np
import logging
import traceback
from datetime import datetime

from bedrock_usage_analyzer.core.profile_fetcher import UNKNOWN_SOURCE, InferenceProfileFetcher
from bedrock_usage_analyzer.sync.quota_rules import scrub_conflicting
from bedrock_usage_analyzer.core.metrics_fetcher import CloudWatchMetricsFetcher
from bedrock_usage_analyzer.core.output_generator import OutputGenerator
from bedrock_usage_analyzer.aws.bedrock import get_regional_profile_prefixes
from bedrock_usage_analyzer.aws.client_factory import create_client
from bedrock_usage_analyzer.aws.servicequotas import QUOTA_MISSING, QUOTA_OK, check_quota, regional_client
from bedrock_usage_analyzer.utils.yaml_handler import load_fm_list
from bedrock_usage_analyzer.utils.partition import get_region_info, get_service_quota_url

logger = logging.getLogger(__name__)

class BedrockAnalyzer:
    """Main orchestrator for Bedrock token usage analysis"""
    
    TIME_PERIODS = ["1hour", "1day", "7days", "14days", "30days"]
    
    def __init__(self, region, granularity_config, profile_fetcher=None, fm_models=None):
        self.region = region
        self.granularity_config = granularity_config

        # Get local timezone - use system's local timezone
        local_dt = datetime.now().astimezone()
        self.local_tz = local_dt.tzinfo
        offset = local_dt.strftime('%z')
        self.tz_offset = f"{offset[:3]}:{offset[3:]}"  # +08:00 format
        self.tz_api_format = offset[:5]  # +0800 format for API

        # botocore picks the endpoint for the region's partition (commercial, GovCloud, China)
        self.cloudwatch_client = create_client('cloudwatch', region)
        self.sq_client = regional_client(region)  # shared with the other quota lookups
        # Region's fm-list: the one parsed during input collection, else read on first lookup
        self._fm_models = fm_models
        # Reuse the fetcher from input collection so profiles are listed only once
        if profile_fetcher is not None:
            self.profile_fetcher = profile_fetcher
            self.bedrock_client = profile_fetcher.bedrock_client
        else:
            self.bedrock_client = create_client('bedrock', region)
            self.profile_fetcher = InferenceProfileFetcher.for_region(self.bedrock_client, self._fm_list())
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
        key = profile_prefix or 'base'
        return any(m['model_id'] == model_id and key in (m.get('endpoints') or {})
                   for m in self._fm_list())

    def _fm_list(self):
        """The region's fm-list models, parsed once per run (every target reads the same file)."""
        if self._fm_models is None:
            self._fm_models = load_fm_list(self.region) or []
        return self._fm_models

    def _load_quota_codes(self, model_id, profile_prefix=None):
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
            endpoints = model.get('endpoints') or {}
            if endpoint_key in endpoints:
                quotas = dict((endpoints[endpoint_key] or {}).get('quotas') or {})
            elif endpoint_key == 'base':
                # Old fm-list structure: model-level quotas were on-demand quotas
                quotas = dict(model.get('quotas') or {})
            else:
                return {}
            # Skip codes that contradict this model or endpoint (e.g. saved by an older
            # version, before the mapping checks existed) instead of showing another limit
            for metric, quota, reason in scrub_conflicting(
                    model_id, endpoint_key, quotas, set(get_regional_profile_prefixes())):
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
        for quota_type, quota_data in quota_codes.items():
            # Handle new structure: {code: L-xxx, name: "..."} or null
            if not (quota_data and isinstance(quota_data, dict) and quota_data.get('code')):
                continue
            key = next((k for k in quotas if k in quota_type.lower()), None)
            if key is None:
                continue
            code = quota_data['code']
            status, quota = check_quota(code, self.region, client=self.sq_client)
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
        return (model_config['model_id'], model_config.get('profile_prefix'), app_ids)

    def _warn_other_sources(self, model_id, profile_prefix, final_model_ids):
        """Tell the user when their application profiles sit under a different endpoint."""
        if len(final_model_ids) > 1:
            return
        others = self.profile_fetcher.other_sources_for_model(model_id, profile_prefix)
        if others:
            where = ', '.join(f"{count} on '{key}'" for key, count in sorted(others.items()))
            logger.info(f"  Note: no application inference profile of {model_id} is based on "
                        f"'{profile_prefix or 'base'}', but {where}. Select that endpoint, or "
                        f"pass the application profile ID/ARN with -m to analyze it directly.")

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

        logger.info(f"Profile discovery complete.\n")

        region_info = get_region_info(self.region)
        processed = set()

        # Process each model
        for model_config in models:
            key = self._scope_key(model_config)
            if key in processed:
                continue
            processed.add(key)
            model_id, profile_prefix, app_ids = key

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
            quota_codes = self._load_quota_codes(model_id, profile_prefix)
            if not any(quota_codes.values()) and profile_prefix not in (None, UNKNOWN_SOURCE) and \
                    not self._system_profile_listed(f"{profile_prefix}.{model_id}"):
                # e.g. a copy of a retired au.* profile: no quota exists to map
                logger.info(f"  {profile_prefix}.{model_id} is not offered in {self.region}; "
                            f"the report will show usage without limits")
            elif not any(quota_codes.values()) and profile_prefix != UNKNOWN_SOURCE:
                if self._endpoint_listed(model_id, profile_prefix):
                    fix = f"bua refresh fm-quotas {self.region}"
                else:
                    # fm-quotas only maps endpoints already in the model list
                    fix = f"bua refresh fm-list {self.region}, then bua refresh fm-quotas {self.region}"
                logger.info(f"  No quota codes mapped for this endpoint in {self.region}; the report will "
                            f"show usage without limits. To map them: {fix}")
            quotas = self._fetch_quotas(model_id, quota_codes, profile_prefix)
            if any(quotas.values()):
                logger.info(f"  Quotas: TPM={quotas['tpm']}, RPM={quotas['rpm']}, TPD={quotas['tpd']}")

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

            # Step 6: Generate output
            logger.info(f"  Generating output files...")
            end_time_local = datetime.now(self.local_tz)
            if profile_prefix == UNKNOWN_SOURCE:
                endpoint = f"{model_id} (source endpoint unknown)"
            else:
                endpoint = f"{profile_prefix}.{model_id}" if profile_prefix else model_id
            scope = [profile_names.get(pid, pid) for pid in final_model_ids] if app_ids else []

            self.output_generator.generate({
                model_id: {
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
                    'profile_prefix': profile_prefix,
                    'application_profile_scope': scope,
                    'file_label': self._file_label(endpoint, app_ids),
                }
            })

    @staticmethod
    def _file_label(endpoint, app_ids):
        """Distinct output name per target, so two endpoints of one model do not overwrite each other."""
        if app_ids:
            return f"{endpoint}-app-{'-'.join(app_ids)}" if len(app_ids) <= 3 else f"{endpoint}-app-{len(app_ids)}profiles"
        return endpoint
