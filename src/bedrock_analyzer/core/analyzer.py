# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Main orchestrator for Bedrock token usage analysis"""

import boto3
import numpy as np
import logging
import traceback
import os
import yaml
from datetime import datetime, timedelta, timezone

from bedrock_analyzer.core.user_inputs import UserInputs
from bedrock_analyzer.core.profile_fetcher import InferenceProfileFetcher
from bedrock_analyzer.core.metrics_fetcher import CloudWatchMetricsFetcher
from bedrock_analyzer.core.output_generator import OutputGenerator
from bedrock_analyzer.aws.bedrock import get_regional_profile_prefixes

logger = logging.getLogger(__name__)

class BedrockAnalyzer:
    """Main orchestrator for Bedrock token usage analysis"""
    
    TIME_PERIODS = ["1hour", "1day", "7days", "14days", "30days"]
    
    def __init__(self, region, granularity_config):
        self.region = region
        self.granularity_config = granularity_config
        
        # Get local timezone - use system's local timezone
        local_dt = datetime.now().astimezone()
        self.local_tz = local_dt.tzinfo
        offset = local_dt.strftime('%z')
        self.tz_offset = f"{offset[:3]}:{offset[3:]}"  # +08:00 format
        self.tz_api_format = offset[:5]  # +0800 format for API
        
        # Initialize clients
        self.bedrock_client = boto3.client('bedrock', region_name=region)
        self.cloudwatch_client = boto3.client('cloudwatch', region_name=region)
        self.sq_client = boto3.client('service-quotas', region_name=region)
        self.profile_fetcher = InferenceProfileFetcher(self.bedrock_client)
        self.metrics_fetcher = CloudWatchMetricsFetcher(self.cloudwatch_client, self.tz_api_format)
        self.output_generator = OutputGenerator()
    
    def _load_quota_codes(self, model_id, profile_prefix=None):
        """Load quota codes for a model from FM list based on endpoint
        
        Args:
            model_id: Base model ID
            profile_prefix: Endpoint prefix (e.g., 'us', 'eu', 'global') or None for base endpoint
        
        Returns:
            dict: Quota codes for the specified endpoint (tpm, rpm, tpd, concurrent)
        """
        fm_file = f'metadata/fm-list-{self.region}.yml'
        
        if not os.path.exists(fm_file):
            return {}
        
        with open(fm_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            models = data.get('models', [])
            
            for model in models:
                if model['model_id'] == model_id:
                    endpoints = model.get('endpoints', {})
                    
                    # Determine which endpoint to use
                    endpoint_key = profile_prefix if profile_prefix else 'base'
                    
                    # Get quotas from the specified endpoint
                    if endpoint_key in endpoints:
                        return endpoints[endpoint_key].get('quotas', {})
                    
                    # Fallback to old structure for backward compatibility
                    return model.get('quotas', {})
        
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
        quotas = {'tpm': None, 'rpm': None, 'tpd': None}
        
        if not quota_codes:
            return quotas
        
        logger.info(f"  Fetching quotas from Service Quotas API...")
        for quota_type, quota_data in quota_codes.items():
            # Handle new structure: {code: L-xxx, name: "..."} or null
            if quota_data and isinstance(quota_data, dict):
                code = quota_data.get('code')
                name = quota_data.get('name')
                
                if code:
                    try:
                        response = self.sq_client.get_service_quota(
                            ServiceCode='bedrock',
                            QuotaCode=code
                        )
                        value = response['Quota']['Value']
                        url = f"https://{self.region}.console.aws.amazon.com/servicequotas/home/services/bedrock/quotas/{code}"
                        
                        quota_info = {'value': value, 'code': code, 'name': name, 'url': url}
                        
                        if 'tpm' in quota_type.lower():
                            quotas['tpm'] = quota_info
                        elif 'rpm' in quota_type.lower():
                            quotas['rpm'] = quota_info
                        elif 'tpd' in quota_type.lower():
                            quotas['tpd'] = quota_info
                    
                    except Exception as e:
                        logger.info(f"  Warning: Could not fetch {quota_type} quota for {model_id}: {e}")
        
        # Apply 2x multiplier for TPD on regional cross-region profiles
        regional_profile_prefixes = get_regional_profile_prefixes()
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
    
    def analyze(self, models):
        """Analyze token usage for given models"""
        
        # Step 0: Discover all profiles once for all models
        logger.info(f"\n{'='*80}")
        logger.info(f"Discovering inference profiles for {len(models)} model(s)...")
        logger.info(f"{'='*80}")
        
        all_profiles_map = {}  # {model_id: {profile_prefix: (final_model_ids, profile_names, profile_metadata)}}
        
        for model_config in models:
            model_id = model_config['model_id']
            profile_prefix = model_config['profile_prefix']
            
            if model_id not in all_profiles_map:
                all_profiles_map[model_id] = {}
            
            if profile_prefix not in all_profiles_map[model_id]:
                final_model_ids, profile_names, profile_metadata = self.profile_fetcher.find_profiles(model_id, profile_prefix)
                all_profiles_map[model_id][profile_prefix] = (final_model_ids, profile_names, profile_metadata)
                
                # Display profiles for this model
                profile_list = [profile_names.get(pid, pid) for pid in final_model_ids]
                logger.info(f"  {model_id} ({profile_prefix or 'base'}): {len(final_model_ids)} profile(s) - {', '.join(profile_list)}")
        
        logger.info(f"Profile discovery complete.\n")
        
        # Process each model
        for model_config in models:
            model_id = model_config['model_id']
            profile_prefix = model_config['profile_prefix']
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing model: {model_id}")
            logger.info(f"{'='*80}")
            
            # Step 1: Get profiles from cache
            final_model_ids, profile_names, profile_metadata = all_profiles_map[model_id][profile_prefix]
            logger.info(f"Using {len(final_model_ids)} profile(s)")
            
            # Step 2: Fetch quotas
            quota_codes = self._load_quota_codes(model_id, profile_prefix)
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
                    'region': self.region
                }
            })


def main():
    try:
        user_inputs = UserInputs()
        user_inputs.collect()
        
        analyzer = BedrockAnalyzer(user_inputs.region, user_inputs.granularity_config)
        analyzer.analyze(user_inputs.models)
        
        logger.info(f"\nCompleted! Check the 'results' directory for output files.")
        
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user.")
    except Exception as e:
        logger.info(f"Error: {e}")


if __name__ == "__main__":
    main()
