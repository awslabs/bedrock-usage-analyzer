# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CloudWatch metrics fetching for Bedrock usage analysis"""

import os
import numpy as np
from datetime import datetime, timedelta, timezone
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import defaultdict

logger = logging.getLogger(__name__)

class CloudWatchMetricsFetcher:
    """Handles CloudWatch metrics retrieval"""
    
    def __init__(self, cloudwatch_client, tz_api_format='+0000'):
        self.cloudwatch_client = cloudwatch_client
        self.tz_api_format = tz_api_format
        self.progress_lock = Lock()
        self.chunks_completed = 0
        self.total_chunks = 0
    
    def _process_combined_time_series(self, all_data, timestamps, period, time_period, target_period=None, end_time=None):
        """Process combined time series data from multiple chunks
        
        Args:
            all_data: Dict of metric data arrays
            timestamps: List of timestamps
            period: Source data period in seconds (e.g., 60 for 1-min data)
            time_period: Time period name (e.g., '7days')
            target_period: Target aggregation period in seconds (e.g., 300 for 5-min peaks)
                          If None or same as period, no aggregation needed
            end_time: datetime object for TPD window reference (optional)
        """
        # If target_period not specified or same as source, no aggregation needed
        if target_period is None or target_period == period:
            target_period = period
            
        period_minutes = period / 60
        
        result = {}
        
        # Sort timestamps and align all data arrays
        # Technique: Create sorted indices from timestamps, then apply same reordering to all metric arrays
        if timestamps:
            sorted_indices = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
            timestamps = [timestamps[i] for i in sorted_indices]
            
            # Sort all data arrays using the same indices
            for key in all_data:
                if all_data[key] and len(all_data[key]) == len(timestamps):
                    all_data[key] = [all_data[key][i] for i in sorted_indices]
        
        # Process each metric
        input_tokens = all_data['input_tokens']
        output_tokens = all_data['output_tokens']
        
        if input_tokens and output_tokens:
            # Calculate TPM only for timestamps where BOTH input and output exist (not None)
            total_tokens = []
            valid_timestamps = []
            
            for i, ts in enumerate(timestamps):
                inp_val = input_tokens[i] if i < len(input_tokens) else None
                out_val = output_tokens[i] if i < len(output_tokens) else None
                
                if inp_val is not None and out_val is not None:
                    total_tokens.append(inp_val + out_val)
                    valid_timestamps.append(ts)
            
            if total_tokens:
                tpm_values_1min = [t / period_minutes for t in total_tokens]
                
                # Store 1-min values for statistics (always per-minute)
                ts_strings_1min = [ts.isoformat() for ts in valid_timestamps]
                result['TPM_1min'] = {
                    'timestamps': ts_strings_1min,
                    'values': tpm_values_1min
                }
                
                # Aggregate to peak if target_period > source period (for charts)
                tpm_values_chart = tpm_values_1min
                valid_timestamps_chart = valid_timestamps
                if target_period > period:
                    valid_timestamps_chart, tpm_values_chart = self._aggregate_to_peak(valid_timestamps, tpm_values_1min, period, target_period)
                
                # Fill missing timestamps for TPM chart
                ts_strings = [ts.isoformat() for ts in valid_timestamps_chart]
                filled_ts, filled_tpm = self._fill_missing_timestamps(ts_strings, tpm_values_chart, target_period if target_period > period else period)
                
                result['TPM'] = {
                    'timestamps': filled_ts,
                    'values': filled_tpm
                }
                
                # Also include raw token counts (with None values preserved)
                ts_strings_all = [ts.isoformat() for ts in timestamps]
                filled_ts_input, filled_input = self._fill_missing_timestamps(ts_strings_all, input_tokens, period)
                filled_ts_output, filled_output = self._fill_missing_timestamps(ts_strings_all, output_tokens, period)
                
                result['InputTokenCount'] = {
                    'timestamps': filled_ts_input,
                    'values': filled_input
                }
                result['OutputTokenCount'] = {
                    'timestamps': filled_ts_output,
                    'values': filled_output
                }
            
            if time_period != "1hour" and total_tokens:
                # TPD: Aggregate tokens by day (sum all tokens within each day)
                # Note: TPD uses daily aggregation, not granularity-based filling
                ts_strings_valid = [ts.isoformat() for ts in valid_timestamps]
                # Use end_time if provided, otherwise fall back to datetime.now()
                reference_time = end_time if end_time else datetime.now(timezone.utc)
                daily_timestamps, daily_totals = self._aggregate_tokens_by_day(ts_strings_valid, total_tokens, reference_time)
                result['TPD'] = {
                    'timestamps': daily_timestamps,
                    'values': daily_totals
                }
        
        if all_data['invocations']:
            # Filter out None values for RPM calculation
            rpm_values = []
            rpm_timestamps = []
            for i, inv in enumerate(all_data['invocations']):
                if inv is not None:
                    rpm_values.append(inv / period_minutes)
                    rpm_timestamps.append(timestamps[i])
            
            if rpm_values:
                # Store 1-min values for statistics (always per-minute)
                ts_strings_1min = [ts.isoformat() for ts in rpm_timestamps]
                result['RPM_1min'] = {
                    'timestamps': ts_strings_1min,
                    'values': rpm_values
                }
                
                # Aggregate to peak if target_period > source period (for charts)
                rpm_values_chart = rpm_values
                rpm_timestamps_chart = rpm_timestamps
                if target_period > period:
                    rpm_timestamps_chart, rpm_values_chart = self._aggregate_to_peak(rpm_timestamps, rpm_values, period, target_period)
                
                ts_strings = [ts.isoformat() for ts in rpm_timestamps_chart]
                filled_ts_rpm, filled_rpm = self._fill_missing_timestamps(ts_strings, rpm_values_chart, target_period if target_period > period else period)
                result['RPM'] = {
                    'timestamps': filled_ts_rpm,
                    'values': filled_rpm
                }
                
                # Also include raw invocations count (with None preserved)
                ts_strings_all = [ts.isoformat() for ts in timestamps]
                filled_ts_inv, filled_inv = self._fill_missing_timestamps(ts_strings_all, all_data['invocations'], period)
                result['Invocations'] = {
                    'timestamps': filled_ts_inv,
                    'values': filled_inv
                }
        
        if all_data['throttles']:
            ts_strings = [ts.isoformat() for ts in timestamps]
            filled_ts, filled_vals = self._fill_missing_timestamps(ts_strings, all_data['throttles'], period)
            result['InvocationThrottles'] = {
                'timestamps': filled_ts,
                'values': filled_vals
            }
        
        if all_data['client_errors']:
            ts_strings = [ts.isoformat() for ts in timestamps]
            filled_ts, filled_vals = self._fill_missing_timestamps(ts_strings, all_data['client_errors'], period)
            result['InvocationClientErrors'] = {
                'timestamps': filled_ts,
                'values': filled_vals
            }
        
        if all_data['server_errors']:
            ts_strings = [ts.isoformat() for ts in timestamps]
            filled_ts, filled_vals = self._fill_missing_timestamps(ts_strings, all_data['server_errors'], period)
            result['InvocationServerErrors'] = {
                'timestamps': filled_ts,
                'values': filled_vals
            }
        
        if all_data['latency']:
            ts_strings = [ts.isoformat() for ts in timestamps]
            filled_ts, filled_vals = self._fill_missing_timestamps(ts_strings, all_data['latency'], period)
            result['InvocationLatency'] = {
                'timestamps': filled_ts,
                'values': filled_vals
            }
        
        # If no data was processed, return properly structured empty time series
        if not result:
            return self._empty_time_series(time_period)
        
        return result
    
    def _align_to_period_boundary(self, dt, period_seconds):
        """Align datetime to period boundary by rounding down
        
        Args:
            dt: datetime to align
            period_seconds: period in seconds (60, 300, 3600, etc.)
        
        Returns:
            datetime aligned to period boundary
        
        Examples:
            - 03:08:23 with 300s period -> 03:05:00
            - 03:08:23 with 3600s period -> 03:00:00
            - 03:08:23 with 60s period -> 03:08:00
        """
        # Remove seconds and microseconds
        dt = dt.replace(second=0, microsecond=0)
        
        if period_seconds >= 3600:  # 1 hour or more
            # Round down to top of hour
            dt = dt.replace(minute=0)
        elif period_seconds >= 60:  # 1 minute or more
            # Round down to nearest period boundary in minutes
            period_minutes = period_seconds // 60
            minutes_offset = dt.minute % period_minutes
            dt = dt - timedelta(minutes=minutes_offset)
        
        return dt
    
    def fetch_all_data_mixed_granularity(self, model_ids, granularity_config):
        """Fetch data at configured granularities for all periods (parallel fetching)
        Returns data that can be sliced for different periods
        
        Strategy:
        - Token metrics (InputTokenCount, OutputTokenCount, Invocations) always fetched at 1-min
          for accurate TPM/RPM peak detection
        - Other metrics (Throttles, Errors, Latency) fetched at configured granularity
        
        Args:
            model_ids: List of model IDs to fetch
            granularity_config: Dict mapping time_period to granularity in seconds
        """
        logger.info(f"  Starting parallel CloudWatch data fetch...")
        logger.info(f"  Granularity config: {granularity_config}")
        
        # Align end_time to 1-minute boundary (finest granularity for token metrics)
        end_time = datetime.now(timezone.utc)
        end_time = self._align_to_period_boundary(end_time, 60)
        
        # Build fetch configs for token metrics (always 1-min) and other metrics (configured)
        max_days = max({'1hour': 1/24, '1day': 1, '7days': 7, '14days': 14, '30days': 30}[tp] 
                       for tp in granularity_config.keys())
        target_start = end_time - timedelta(days=max_days)
        
        # Config 1: Token metrics at 1-min (for TPM/RPM peak detection)
        token_metrics_config = {
            'start_time': target_start,
            'end_time': end_time,
            'period': 60,
            'metrics': ['input_tokens', 'output_tokens', 'invocations']
        }
        
        # Config 2: Other metrics at configured granularities
        other_metrics_configs = {}
        period_ranges = {}
        for time_period, period in granularity_config.items():
            if period not in period_ranges:
                period_ranges[period] = []
            days = {'1hour': 1/24, '1day': 1, '7days': 7, '14days': 14, '30days': 30}[time_period]
            period_ranges[period].append(days)
        
        for period, day_list in period_ranges.items():
            max_days_for_period = max(day_list)
            period_start = end_time - timedelta(days=max_days_for_period)
            other_metrics_configs[period] = {
                'start_time': period_start,
                'end_time': end_time,
                'metrics': ['throttles', 'client_errors', 'server_errors', 'latency']
            }
        
        # Calculate total chunks for progress tracking
        self.chunks_completed = 0
        self.total_chunks = 0
        for model_id in model_ids:
            # Token metrics at 1-min
            chunks = self._chunk_time_range(token_metrics_config['start_time'], 
                                           token_metrics_config['end_time'], 60)
            self.total_chunks += len(chunks)
            # Other metrics at configured granularities
            for period, config in other_metrics_configs.items():
                chunks = self._chunk_time_range(config['start_time'], config['end_time'], period)
                self.total_chunks += len(chunks)
        
        logger.info(f"  Fetching {len(model_ids)} model(s): token metrics at 1-min + other metrics at configured granularities")
        logger.info(f"  Total chunks: {self.total_chunks}")
        
        all_fetched_data = {}
        
        # Parallel fetching across all model IDs
        max_workers = os.cpu_count() or 4
        logger.info(f"  Using {max_workers} parallel workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for model_id in model_ids:
                # Fetch 1: Token metrics at 1-min
                future = executor.submit(
                    self._fetch_token_metrics,
                    model_id,
                    token_metrics_config['start_time'],
                    token_metrics_config['end_time'],
                    60
                )
                futures.append((future, model_id, 60, 'token'))
                
                # Fetch 2: Other metrics at configured granularities
                for period, config in other_metrics_configs.items():
                    future = executor.submit(
                        self._fetch_other_metrics,
                        model_id,
                        config['start_time'],
                        config['end_time'],
                        period
                    )
                    futures.append((future, model_id, period, 'other'))
            
            for future, model_id, period, fetch_type in futures:
                if model_id not in all_fetched_data:
                    logger.info(f"  Fetching data for {model_id}...")
                    all_fetched_data[model_id] = {'end_time': end_time}
                
                try:
                    new_data = future.result()
                    
                    # Store token metrics separately with '60_token' key
                    if fetch_type == 'token':
                        all_fetched_data[model_id]['60_token'] = new_data
                    else:
                        # Merge other metrics into the period's data
                        if period not in all_fetched_data[model_id]:
                            all_fetched_data[model_id][period] = new_data
                        else:
                            # Merge if period already exists
                            existing = all_fetched_data[model_id][period]
                            for key in new_data['data']:
                                existing['data'][key] = new_data['data'][key]
                        
                except Exception as e:
                    logger.info(f"    Warning: Failed to fetch {fetch_type} data (period={period}s) for {model_id}: {e}")
                    # Create empty data structure
                    empty_data = {
                        'timestamps': [], 
                        'data': {
                            'invocations': [], 
                            'input_tokens': [], 
                            'output_tokens': [], 
                            'throttles': [],
                            'client_errors': [],
                            'server_errors': [],
                            'latency': []
                        }, 
                        'period': period
                    }
                    if fetch_type == 'token':
                        all_fetched_data[model_id]['60_token'] = empty_data
                    else:
                        all_fetched_data[model_id][period] = empty_data
        
        logger.info(f"  Parallel fetch complete")
        
        return all_fetched_data
    
    def _fetch_token_metrics(self, model_id, start_time, end_time, period):
        """Fetch only token-related metrics (for TPM/RPM calculation)
        
        Always fetched at 1-minute granularity for accurate peak detection
        """
        try:
            chunks = self._chunk_time_range(start_time, end_time, period)
            
            all_data_with_timestamps = {
                'invocations': {'timestamps': [], 'values': []},
                'input_tokens': {'timestamps': [], 'values': []},
                'output_tokens': {'timestamps': [], 'values': []}
            }
            
            for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
                response = self.cloudwatch_client.get_metric_data(
                    MetricDataQueries=[
                        self._create_query('invocations', 'Invocations', model_id, period),
                        self._create_query('input_tokens', 'InputTokenCount', model_id, period),
                        self._create_query('output_tokens', 'OutputTokenCount', model_id, period)
                    ],
                    StartTime=chunk_start,
                    EndTime=chunk_end,
                    LabelOptions={'Timezone': self.tz_api_format}
                )
                
                with self.progress_lock:
                    self.chunks_completed += 1
                    pct = int(self.chunks_completed / self.total_chunks * 100)
                    logger.info(f"    Progress: {self.chunks_completed}/{self.total_chunks} chunks ({pct}%)")
                
                for result in response['MetricDataResults']:
                    metric_id = result['Id']
                    if result['Values'] and result['Timestamps']:
                        all_data_with_timestamps[metric_id]['values'].extend(result['Values'])
                        all_data_with_timestamps[metric_id]['timestamps'].extend(result['Timestamps'])
            
            # Align data by timestamps
            all_timestamps_set = set()
            for metric_data in all_data_with_timestamps.values():
                all_timestamps_set.update(metric_data['timestamps'])
            
            all_timestamps = sorted(list(all_timestamps_set))
            
            all_data = {}
            for metric_id, metric_data in all_data_with_timestamps.items():
                ts_to_value = {ts: val for ts, val in zip(metric_data['timestamps'], metric_data['values'])}
                all_data[metric_id] = [ts_to_value.get(ts) for ts in all_timestamps]
            
            # Sort chronologically
            if all_timestamps:
                sorted_indices = sorted(range(len(all_timestamps)), key=lambda i: all_timestamps[i])
                all_timestamps = [all_timestamps[i] for i in sorted_indices]
                for key in all_data:
                    if all_data[key] and len(all_data[key]) == len(all_timestamps):
                        all_data[key] = [all_data[key][i] for i in sorted_indices]
            
            return {
                'timestamps': all_timestamps,
                'data': all_data,
                'period': period
            }
        except Exception as e:
            logger.info(f"    Warning: Could not fetch token metrics: {e}")
            return {
                'timestamps': [],
                'data': {'invocations': [], 'input_tokens': [], 'output_tokens': []},
                'period': period
            }
    
    def _fetch_other_metrics(self, model_id, start_time, end_time, period):
        """Fetch non-token metrics (throttles, errors, latency)
        
        Fetched at configured granularity
        """
        try:
            chunks = self._chunk_time_range(start_time, end_time, period)
            
            all_data_with_timestamps = {
                'throttles': {'timestamps': [], 'values': []},
                'client_errors': {'timestamps': [], 'values': []},
                'server_errors': {'timestamps': [], 'values': []},
                'latency': {'timestamps': [], 'values': []}
            }
            
            for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
                response = self.cloudwatch_client.get_metric_data(
                    MetricDataQueries=[
                        self._create_query('throttles', 'InvocationThrottles', model_id, period),
                        self._create_query('client_errors', 'InvocationClientErrors', model_id, period),
                        self._create_query('server_errors', 'InvocationServerErrors', model_id, period),
                        self._create_query('latency', 'InvocationLatency', model_id, period, stat='Average')
                    ],
                    StartTime=chunk_start,
                    EndTime=chunk_end,
                    LabelOptions={'Timezone': self.tz_api_format}
                )
                
                with self.progress_lock:
                    self.chunks_completed += 1
                    pct = int(self.chunks_completed / self.total_chunks * 100)
                    logger.info(f"    Progress: {self.chunks_completed}/{self.total_chunks} chunks ({pct}%)")
                
                for result in response['MetricDataResults']:
                    metric_id = result['Id']
                    if result['Values'] and result['Timestamps']:
                        all_data_with_timestamps[metric_id]['values'].extend(result['Values'])
                        all_data_with_timestamps[metric_id]['timestamps'].extend(result['Timestamps'])
            
            # Align data by timestamps
            all_timestamps_set = set()
            for metric_data in all_data_with_timestamps.values():
                all_timestamps_set.update(metric_data['timestamps'])
            
            all_timestamps = sorted(list(all_timestamps_set))
            
            all_data = {}
            for metric_id, metric_data in all_data_with_timestamps.items():
                ts_to_value = {ts: val for ts, val in zip(metric_data['timestamps'], metric_data['values'])}
                all_data[metric_id] = [ts_to_value.get(ts) for ts in all_timestamps]
            
            # Sort chronologically
            if all_timestamps:
                sorted_indices = sorted(range(len(all_timestamps)), key=lambda i: all_timestamps[i])
                all_timestamps = [all_timestamps[i] for i in sorted_indices]
                for key in all_data:
                    if all_data[key] and len(all_data[key]) == len(all_timestamps):
                        all_data[key] = [all_data[key][i] for i in sorted_indices]
            
            return {
                'timestamps': all_timestamps,
                'data': all_data,
                'period': period
            }
        except Exception as e:
            logger.info(f"    Warning: Could not fetch other metrics: {e}")
            return {
                'timestamps': [],
                'data': {'throttles': [], 'client_errors': [], 'server_errors': [], 'latency': []},
                'period': period
            }
    
    def _fetch_raw_data(self, model_id, start_time, end_time, period):
        """Fetch raw CloudWatch data for a time range"""
        try:
            chunks = self._chunk_time_range(start_time, end_time, period)
            
            # Store timestamps per metric for proper alignment
            all_data_with_timestamps = {
                'invocations': {'timestamps': [], 'values': []},
                'input_tokens': {'timestamps': [], 'values': []},
                'output_tokens': {'timestamps': [], 'values': []},
                'throttles': {'timestamps': [], 'values': []},
                'client_errors': {'timestamps': [], 'values': []},
                'server_errors': {'timestamps': [], 'values': []},
                'latency': {'timestamps': [], 'values': []}
            }
            
            for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
                # All metrics are using "Sum" statistic aggregation method, except InvocationLatency which is using "Average"
                response = self.cloudwatch_client.get_metric_data(
                    MetricDataQueries=[
                        self._create_query('invocations', 'Invocations', model_id, period),
                        self._create_query('input_tokens', 'InputTokenCount', model_id, period),
                        self._create_query('output_tokens', 'OutputTokenCount', model_id, period),
                        self._create_query('throttles', 'InvocationThrottles', model_id, period),
                        self._create_query('client_errors', 'InvocationClientErrors', model_id, period),
                        self._create_query('server_errors', 'InvocationServerErrors', model_id, period),
                        self._create_query('latency', 'InvocationLatency', model_id, period, stat='Average')
                    ],
                    StartTime=chunk_start,
                    EndTime=chunk_end,
                    LabelOptions={'Timezone': self.tz_api_format}
                )
                
                # Update progress
                with self.progress_lock:
                    self.chunks_completed += 1
                    pct = int(self.chunks_completed / self.total_chunks * 100)
                    logger.info(f"    Progress: {self.chunks_completed}/{self.total_chunks} chunks ({pct}%)")
                
                # Collect timestamps AND values for each metric separately
                for result in response['MetricDataResults']:
                    metric_id = result['Id']
                    if result['Values'] and result['Timestamps']:
                        all_data_with_timestamps[metric_id]['values'].extend(result['Values'])
                        all_data_with_timestamps[metric_id]['timestamps'].extend(result['Timestamps'])
            
            # Align data by timestamps: collect all unique timestamps and map values
            all_timestamps_set = set()
            for metric_data in all_data_with_timestamps.values():
                all_timestamps_set.update(metric_data['timestamps'])
            
            all_timestamps = sorted(list(all_timestamps_set))
            
            # Create timestamp-to-value mapping for each metric
            all_data = {}
            for metric_id, metric_data in all_data_with_timestamps.items():
                # Build mapping
                ts_to_value = {ts: val for ts, val in zip(metric_data['timestamps'], metric_data['values'])}
                # Align to all_timestamps (use None for missing timestamps)
                all_data[metric_id] = [ts_to_value.get(ts) for ts in all_timestamps]
            
            # Sort timestamps chronologically and reorder all metrics to match
            # CloudWatch doesn't guarantee order, especially across multiple chunks
            # This ensures data integrity: each metric value stays aligned with its timestamp
            # Technique: Create sorted indices from timestamps, then apply same reordering to all metric arrays
            if all_timestamps:
                sorted_indices = sorted(range(len(all_timestamps)), key=lambda i: all_timestamps[i])
                all_timestamps = [all_timestamps[i] for i in sorted_indices]
                for key in all_data:
                    if all_data[key] and len(all_data[key]) == len(all_timestamps):
                        all_data[key] = [all_data[key][i] for i in sorted_indices]
            
            return {
                'timestamps': all_timestamps,
                'data': all_data,
                'period': period
            }
        except Exception as e:
            logger.info(f"    Warning: Could not fetch data: {e}")
            return {
                'timestamps': [], 
                'data': {
                    'invocations': [], 
                    'input_tokens': [], 
                    'output_tokens': [], 
                    'throttles': [],
                    'client_errors': [],
                    'server_errors': [],
                    'latency': []
                }, 
                'period': period
            }
    
    def slice_and_process_data(self, fetched_data, time_period, granularity_config):
        """
        Slice fetched data for a specific time period and process into time series.
        
        Strategy:
        - Token metrics (input_tokens, output_tokens, invocations) come from '60_token' (1-min data)
        - Other metrics (throttles, errors, latency) come from configured granularity
        - Merge both datasets for processing
        """
        end_time = fetched_data['end_time']
        period = granularity_config[time_period]
        
        if time_period == '1hour':
            start_time = end_time - timedelta(hours=1)
        elif time_period == '1day':
            start_time = end_time - timedelta(days=1)
        elif time_period == '7days':
            start_time = end_time - timedelta(days=7)
        elif time_period == '14days':
            start_time = end_time - timedelta(days=14)
        elif time_period == '30days':
            start_time = end_time - timedelta(days=30)
        else:
            return self._empty_time_series(time_period)
        
        # Get token metrics from 1-min data
        if '60_token' not in fetched_data:
            logger.info(f"    Warning: No 1-min token data for {time_period}")
            return self._empty_time_series(time_period)
        
        token_dataset = fetched_data['60_token']
        
        # Get other metrics from configured granularity
        other_dataset = None
        if period in fetched_data:
            other_dataset = fetched_data[period]
        
        # Slice and merge both datasets
        return self._slice_and_merge_datasets(token_dataset, other_dataset, start_time, end_time, time_period, period)
    
    def _slice_and_merge_datasets(self, token_dataset, other_dataset, start_time, end_time, time_period, target_period):
        """Slice and merge token metrics (1-min) with other metrics (configured granularity)
        
        Args:
            token_dataset: Dataset with 1-min token metrics
            other_dataset: Dataset with other metrics at configured granularity (or None)
            start_time: Start of time range
            end_time: End of time range
            time_period: Time period name (e.g., '7days')
            target_period: Target granularity in seconds (e.g., 300 for 5-min)
        """
        # Slice token metrics from 1-min data
        token_timestamps = token_dataset['timestamps']
        token_data = token_dataset['data']
        
        # Find indices within time range
        token_indices = [i for i, ts in enumerate(token_timestamps) if start_time <= ts <= end_time]
        
        if not token_indices:
            return self._empty_time_series(time_period)
        
        filtered_token_timestamps = [token_timestamps[i] for i in token_indices]
        filtered_token_data = {}
        for key in ['invocations', 'input_tokens', 'output_tokens']:
            if key in token_data and token_data[key]:
                valid_indices = [i for i in token_indices if i < len(token_data[key])]
                filtered_token_data[key] = [token_data[key][i] for i in valid_indices]
            else:
                filtered_token_data[key] = []
        
        # Slice other metrics if available
        filtered_other_data = {}
        if other_dataset:
            other_timestamps = other_dataset['timestamps']
            other_data = other_dataset['data']
            other_indices = [i for i, ts in enumerate(other_timestamps) if start_time <= ts <= end_time]
            
            for key in ['throttles', 'client_errors', 'server_errors', 'latency']:
                if key in other_data and other_data[key]:
                    valid_indices = [i for i in other_indices if i < len(other_data[key])]
                    filtered_other_data[key] = [other_data[key][i] for i in valid_indices]
                else:
                    filtered_other_data[key] = []
        else:
            # No other metrics available
            filtered_other_data = {
                'throttles': [],
                'client_errors': [],
                'server_errors': [],
                'latency': []
            }
        
        # Merge datasets: token metrics at 1-min, other metrics at configured granularity
        # Process with special handling for TPM/RPM peak aggregation
        return self._process_combined_time_series(
            {**filtered_token_data, **filtered_other_data},
            filtered_token_timestamps,
            60,  # Token metrics are at 1-min
            time_period,
            target_period,  # Target granularity for aggregation
            end_time  # Pass end_time for TPD window consistency
        )
    
    def _aggregate_to_peak(self, timestamps, values, source_period, target_period):
        """Aggregate fine-grained data to coarser granularity using peak (max) values
        
        Args:
            timestamps: List of datetime objects
            values: List of values (TPM or RPM)
            source_period: Source period in seconds (e.g., 60 for 1-min)
            target_period: Target period in seconds (e.g., 300 for 5-min, 3600 for 1-hour)
            
        Returns:
            Tuple of (aggregated_timestamps, aggregated_values) where each value is the peak within the window
        """
        if not timestamps or not values:
            return timestamps, values
        
        # Group data points by target period windows
        from collections import defaultdict
        windows = defaultdict(list)
        
        for ts, val in zip(timestamps, values):
            # Align timestamp to target period boundary
            window_start = self._align_to_period_boundary(ts, target_period)
            windows[window_start].append(val)
        
        # Calculate peak for each window
        aggregated_timestamps = sorted(windows.keys())
        aggregated_values = [max(windows[ts]) for ts in aggregated_timestamps]
        
        return aggregated_timestamps, aggregated_values
    
    def _slice_from_dataset(self, dataset, start_time, end_time, time_period):
        """Slice data from a single dataset by time range
        
        Dataset structure:
        {
            'timestamps': [datetime, datetime, ...],
            'data': {
                'invocations': [value, value, ...],
                'input_tokens': [value, value, ...],
                'output_tokens': [value, value, ...],
                'throttles': [value, value, ...],
                'client_errors': [value, value, ...],
                'server_errors': [value, value, ...],
                'latency': [value, value, ...]
            },
            'period': 60|300|3600  # granularity in seconds
        }
        """
        timestamps = dataset['timestamps']
        data = dataset['data']
        period = dataset['period']
        
        # Find timestamps that fall within the requested time range
        indices = [i for i, ts in enumerate(timestamps) if start_time <= ts <= end_time]
        
        # If empty
        if not indices:
            return self._empty_time_series(time_period)
        
        filtered_timestamps = [timestamps[i] for i in indices]
        filtered_data = {}
        
        # Safely filter data arrays, ensuring indices are within bounds
        for key in data:
            if data[key]:
                # Only use indices that are valid for this data array
                valid_indices = [i for i in indices if i < len(data[key])]
                filtered_data[key] = [data[key][i] for i in valid_indices]
            else:
                filtered_data[key] = []
        
        return self._process_combined_time_series(filtered_data, filtered_timestamps, period, time_period)
    
    def _chunk_time_range(self, start_time, end_time, period):
        """Split time range into chunks to respect CloudWatch data point limit
        
        CloudWatch limit: 100,800 data points per request
        With Period=300 (5 min), that's 100,800 * 5 min = 504,000 minutes = 350 days
        So we can fetch 30 days in a single request with 5-min granularity
        """
        # Calculate max duration based on period
        # CloudWatch limit: 100,800 data points per request
        max_data_points = 100800
        max_duration_seconds = max_data_points * period
        max_duration = timedelta(seconds=max_duration_seconds)
        
        chunks = []
        current_start = start_time
        
        while current_start < end_time:
            current_end = min(current_start + max_duration, end_time)
            chunks.append((current_start, current_end))
            current_start = current_end
        
        return chunks
    
    def _initialize_metrics(self, time_period):
        """Initialize metrics with empty defaults (no fake data points)"""
        metrics = {
            'Invocations': {'values': [], 'p50': 0.0, 'p90': 0.0, 'count': 0, 'sum': 0.0, 'avg': 0.0},
            'InputTokenCount': {'values': [], 'p50': 0.0, 'p90': 0.0, 'count': 0, 'sum': 0.0, 'avg': 0.0},
            'OutputTokenCount': {'values': [], 'p50': 0.0, 'p90': 0.0, 'count': 0, 'sum': 0.0, 'avg': 0.0},
            'InvocationLatency': {'values': [], 'p50': 0.0, 'p90': 0.0, 'count': 0, 'sum': 0.0, 'avg': 0.0},
            'InvocationThrottles': {'values': [], 'p50': 0.0, 'p90': 0.0, 'count': 0, 'sum': 0.0, 'avg': 0.0},
            'InvocationClientErrors': {'values': [], 'p50': 0.0, 'p90': 0.0, 'count': 0, 'sum': 0.0, 'avg': 0.0},
            'InvocationServerErrors': {'values': [], 'p50': 0.0, 'p90': 0.0, 'count': 0, 'sum': 0.0, 'avg': 0.0},
            'TPM': {'values': [], 'p50': 0.0, 'p90': 0.0, 'count': 0, 'sum': 0.0, 'avg': 0.0},
            'RPM': {'values': [], 'p50': 0.0, 'p90': 0.0, 'count': 0, 'sum': 0.0, 'avg': 0.0},
            'TPM_1min': {'values': [], 'p50': 0.0, 'p90': 0.0, 'count': 0, 'sum': 0.0, 'avg': 0.0},
            'RPM_1min': {'values': [], 'p50': 0.0, 'p90': 0.0, 'count': 0, 'sum': 0.0, 'avg': 0.0}
        }
        if time_period != "1hour":
            metrics['TPD'] = {'values': [], 'p50': 0.0, 'p90': 0.0, 'count': 0, 'sum': 0.0, 'avg': 0.0}
        return metrics
    
    # The default stat = "Sum" is crucial so that by default it is aggregating the data points by summing them within the period (e.g. 1 min, 5 mins, 1 hour)
    def _create_query(self, query_id, metric_name, model_id, period, stat='Sum'):
        """Create a metric query"""
        return {
            'Id': query_id,
            'MetricStat': {
                'Metric': {
                    'Namespace': 'AWS/Bedrock',
                    'MetricName': metric_name,
                    'Dimensions': [{'Name': 'ModelId', 'Value': model_id}]
                },
                'Period': period,
                'Stat': stat
            }
        }
    
    def _empty_time_series(self, time_period):
        """Return empty time series data"""
        metrics = {
            'RPM': {'timestamps': [], 'values': []},
            'TPM': {'timestamps': [], 'values': []},
            'InvocationThrottles': {'timestamps': [], 'values': []}
        }
        if time_period != "1hour":
            metrics['TPD'] = {'timestamps': [], 'values': []}
        return metrics
    
    def _fill_missing_timestamps(self, timestamps, values, period):
        """Fill missing timestamps with null values to create gaps in charts
        
        Args:
            timestamps: List of ISO timestamp strings (already sorted)
            values: List of values corresponding to timestamps
            period: Granularity period in seconds (60, 300, 3600)
        
        Returns:
            tuple: (filled_timestamps, filled_values) with nulls for missing data points
        """
        if not timestamps or not values:
            return timestamps, values
        
        # Convert ISO strings to datetime objects
        dt_timestamps = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in timestamps]
        
        # Generate complete sequence from first to last timestamp
        start_time = dt_timestamps[0]
        end_time = dt_timestamps[-1]
        
        filled_timestamps = []
        filled_values = []
        
        # Create a map of existing timestamps to values for quick lookup
        timestamp_map = {dt: val for dt, val in zip(dt_timestamps, values)}
        
        # Generate expected timestamps at period intervals
        current_time = start_time
        while current_time <= end_time:
            filled_timestamps.append(current_time.isoformat())
            # Use actual value if exists, otherwise None (becomes null in JSON)
            filled_values.append(timestamp_map.get(current_time, None))
            current_time += timedelta(seconds=period)
        
        return filled_timestamps, filled_values
    
    def _aggregate_tokens_by_day(self, timestamps, token_values, reference_time):
        """Aggregate token values by day using 24-hour backward windows from reference time
        
        Args:
            timestamps: List of ISO timestamp strings
            token_values: List of token counts (raw sums from CloudWatch)
            reference_time: datetime object to use as reference for window boundaries
        
        Returns:
            tuple: (daily_timestamps, daily_totals) where each entry represents one 24-hour window
        """
        
        if not timestamps or not token_values:
            return [], []
        
        # Use reference time for consistent window boundaries across all profiles
        now = reference_time
        
        # Create 24-hour windows going backward from now
        # Determine how many days we need based on the oldest timestamp
        oldest_ts = datetime.fromisoformat(timestamps[0].replace('Z', '+00:00'))
        days_needed = int((now - oldest_ts).total_seconds() / 86400) + 1
        
        # Create windows: each window is [window_start, window_end)
        windows = []
        for day_offset in range(days_needed):
            window_end = now - timedelta(days=day_offset)
            window_start = window_end - timedelta(days=1)
            windows.append((window_start, window_end))
        
        # Aggregate tokens into windows
        window_totals = defaultdict(int)
        for ts_str, tokens in zip(timestamps, token_values):
            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            # Find which window this timestamp belongs to
            for window_start, window_end in windows:
                if window_start <= ts < window_end:
                    window_totals[(window_start, window_end)] += tokens
                    break
        
        # Sort windows by start time and create output lists
        # Only include windows with data (sparse output for individual profiles)
        sorted_windows = sorted(window_totals.keys(), key=lambda w: w[0])
        daily_timestamps = [window_start.isoformat() for window_start, _ in sorted_windows]
        daily_totals = [window_totals[window] for window in sorted_windows]
        
        return daily_timestamps, daily_totals
    
    def aggregate_statistics(self, all_stats, time_period):
        """Aggregate statistics across multiple profiles"""
        if not all_stats:
            return {}
        
        aggregated = self._initialize_metrics(time_period)
        
        for metric_name in aggregated.keys():
            all_values = []
            for profile_stats in all_stats.values():
                if metric_name in profile_stats and profile_stats[metric_name]['values']:
                    all_values.extend(profile_stats[metric_name]['values'])
            
            if all_values:
                aggregated[metric_name] = {
                    'values': all_values,
                    'p50': np.percentile(all_values, 50),
                    'p90': np.percentile(all_values, 90),
                    'count': len(all_values),
                    'sum': sum(all_values),
                    'avg': np.mean(all_values)
                }
        
        return aggregated
    
    def aggregate_time_series(self, all_ts, time_period):
        """Aggregate time series across multiple profiles by summing values at each timestamp"""
        if not all_ts:
            return {}
        
        logger.info(f"    Aggregating time series for {len(all_ts)} profiles...")
        
        # Determine period for filling based on time_period
        period_map = {'1hour': 60, '1day': 300, '7days': 3600, '14days': 3600, '30days': 3600}
        fill_period = period_map.get(time_period, 300)
        
        aggregated = {}
        
        for metric_name in ['TPM', 'RPM', 'InvocationThrottles']:
            # Collect only timestamps with non-None values from all profiles
            values_by_ts = {}
            
            for profile_ts in all_ts.values():
                if metric_name in profile_ts:
                    ts_list = profile_ts[metric_name]['timestamps']
                    val_list = profile_ts[metric_name]['values']
                    for ts, val in zip(ts_list, val_list):
                        if val is not None:  # Only collect non-None values
                            if ts not in values_by_ts:
                                values_by_ts[ts] = 0
                            values_by_ts[ts] += val
            
            if values_by_ts:
                # Sort timestamps and get values
                sorted_timestamps = sorted(values_by_ts.keys())
                sorted_values = [values_by_ts[ts] for ts in sorted_timestamps]
                
                # Fill missing timestamps to create gaps in chart
                filled_ts, filled_vals = self._fill_missing_timestamps(sorted_timestamps, sorted_values, fill_period)
                
                aggregated[metric_name] = {
                    'timestamps': filled_ts,
                    'values': filled_vals
                }
            else:
                aggregated[metric_name] = {
                    'timestamps': [],
                    'values': []
                }
        
        # Aggregate TPD separately using only TPD timestamps (daily granularity)
        if time_period != "1hour":
            tpd_timestamps = set()
            for profile_ts in all_ts.values():
                if 'TPD' in profile_ts and profile_ts['TPD']['timestamps']:
                    tpd_timestamps.update(profile_ts['TPD']['timestamps'])
            
            if tpd_timestamps:
                sorted_tpd_timestamps = sorted(tpd_timestamps)
                tpd_values_by_ts = {ts: 0 for ts in sorted_tpd_timestamps}
                
                for profile_ts in all_ts.values():
                    if 'TPD' in profile_ts:
                        ts_list = profile_ts['TPD']['timestamps']
                        val_list = profile_ts['TPD']['values']
                        for ts, val in zip(ts_list, val_list):
                            tpd_values_by_ts[ts] += val
                
                # Fill in missing days with 0s for aggregated TPD (dense timeline)
                # This ensures continuous timeline even when no profiles had data for certain days
                if sorted_tpd_timestamps:
                    first_ts = datetime.fromisoformat(sorted_tpd_timestamps[0].replace('Z', '+00:00'))
                    last_ts = datetime.fromisoformat(sorted_tpd_timestamps[-1].replace('Z', '+00:00'))
                    
                    # Generate all daily timestamps from first to last
                    complete_timestamps = []
                    current = first_ts
                    while current <= last_ts:
                        ts_str = current.isoformat()
                        complete_timestamps.append(ts_str)
                        if ts_str not in tpd_values_by_ts:
                            tpd_values_by_ts[ts_str] = 0
                        current += timedelta(days=1)
                    
                    aggregated['TPD'] = {
                        'timestamps': complete_timestamps,
                        'values': [tpd_values_by_ts[ts] for ts in complete_timestamps]
                    }
        
        return aggregated


