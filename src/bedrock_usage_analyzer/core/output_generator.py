# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output generation for Bedrock usage analysis reports"""

import os
import json
import logging
from datetime import datetime, timedelta
from jinja2 import Template
from bedrock_usage_analyzer.utils.partition import get_console_domain

logger = logging.getLogger(__name__)

class OutputGenerator:
    """Handles JSON and HTML output generation"""
    
    def __init__(self, output_dir: str = 'results'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate(self, results):
        """Generate JSON and HTML output files with interactive graphs"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_id, data in results.items():
            safe_model_id = model_id.replace(':', '_').replace('.', '_')
            base_filename = f"{safe_model_id}-{timestamp}"
            
            self._generate_json(base_filename, model_id, timestamp, data)
            self._generate_html(base_filename, model_id, timestamp, data)
    
    def _generate_json(self, filename, model_id, timestamp, data):
        """Generate JSON output"""
        json_file = f"{self.output_dir}/{filename}.json"
        
        # Format timestamp for display
        end_time = data.get('end_time')
        if end_time:
            formatted_timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S %Z")
            iso_timestamp = end_time.isoformat()
        else:
            formatted_timestamp = timestamp
            iso_timestamp = timestamp
        
        # Generate period names (same as HTML)
        period_names = self._generate_period_names(data.get('end_time'), data.get('tz_offset', '+00:00'))
        
        # Build disclaimers
        disclaimers = {
            'throttling': (
                "Low TPM/TPD values do not rule out token-based throttling. "
                "Bedrock reserves (input_tokens + max_tokens) from your quota at request start, "
                "but CloudWatch only records actual tokens used after completion. "
                "If max_tokens is set high but actual output is low, you may hit throttling limits "
                "that are invisible in these metrics. See: "
                "https://docs.aws.amazon.com/bedrock/latest/userguide/quotas-token-burndown.html"
            )
        }
        
        # Add quota disclaimer if quotas exist
        quotas = data.get('quotas', {})
        if quotas:
            console_domain = get_console_domain()
            disclaimers['quota_mapping'] = (
                "Quota mappings were inferred using AI and may not be accurate. "
                f"Always verify with AWS Service Quotas console: "
                f"https://{console_domain}/servicequotas"
            )
        
        # Process time_series to add per-metric disclaimers and quota info
        time_series = data['time_series']
        processed_time_series = self._add_time_series_metadata(time_series, quotas, disclaimers)
        
        output_data = {
            'model_id': model_id,
            'region': data.get('region', 'N/A'),
            'generated_at': formatted_timestamp,
            'generated_at_iso': iso_timestamp,
            'timezone': data.get('tz_offset', '+00:00'),
            'disclaimers': disclaimers,
            'stats': data['stats'],
            'time_series': processed_time_series,
            'quotas': data.get('quotas', {}),
            'granularity_config': data.get('granularity_config', {}),
            'profile_names': data.get('profile_names', {}),
            'contributions': data.get('contributions', {}),
            'period_names': period_names
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"Generated: {json_file}")
    
    def _add_time_series_metadata(self, time_series, quotas, disclaimers):
        """Add disclaimers and quota info to time series data"""
        import copy
        processed = copy.deepcopy(time_series)
        
        throttling_disclaimer = disclaimers.get('throttling', '')
        quota_disclaimer = disclaimers.get('quota_mapping', '')
        
        for period, period_data in processed.items():
            for profile_id, metrics in period_data.items():
                # Add throttling disclaimer to TPM and TPD metrics
                if 'TPM' in metrics:
                    if not isinstance(metrics['TPM'], dict):
                        continue
                    metrics['TPM']['disclaimer'] = throttling_disclaimer
                    if quotas.get('tpm'):
                        metrics['TPM']['quota'] = {
                            'value': quotas['tpm'].get('value'),
                            'code': quotas['tpm'].get('code'),
                            'name': quotas['tpm'].get('name'),
                            'url': quotas['tpm'].get('url'),
                            'disclaimer': quota_disclaimer
                        }
                
                if 'TPD' in metrics:
                    if not isinstance(metrics['TPD'], dict):
                        continue
                    metrics['TPD']['disclaimer'] = throttling_disclaimer
                    if quotas.get('tpd'):
                        metrics['TPD']['quota'] = {
                            'value': quotas['tpd'].get('value'),
                            'code': quotas['tpd'].get('code'),
                            'name': quotas['tpd'].get('name'),
                            'url': quotas['tpd'].get('url'),
                            'disclaimer': quota_disclaimer
                        }
                
                # Add quota info to RPM (no throttling disclaimer, just quota)
                if 'RPM' in metrics and quotas.get('rpm'):
                    if not isinstance(metrics['RPM'], dict):
                        continue
                    metrics['RPM']['quota'] = {
                        'value': quotas['rpm'].get('value'),
                        'code': quotas['rpm'].get('code'),
                        'name': quotas['rpm'].get('name'),
                        'url': quotas['rpm'].get('url'),
                        'disclaimer': quota_disclaimer
                    }
        
        return processed
    
    def _generate_period_names(self, end_time, tz_offset):
        """Generate friendly period names with local timezone"""
        names = {}
        for period in ['1hour', '1day', '7days', '14days', '30days']:
            if period == '1hour':
                start = end_time - timedelta(hours=1)
                names[period] = f"Last 1 hour ({start.strftime('%H:%M')}-{end_time.strftime('%H:%M')})"
            elif period == '1day':
                start = end_time - timedelta(days=1)
                names[period] = f"Last 1 day ({start.strftime('%a %H:%M')}-{end_time.strftime('%a %H:%M')})"
            elif period == '7days':
                start = end_time - timedelta(days=7)
                names[period] = f"Last 7 days ({start.strftime('%d %b')}-{end_time.strftime('%d %b')})"
            elif period == '14days':
                start = end_time - timedelta(days=14)
                names[period] = f"Last 14 days ({start.strftime('%d %b')}-{end_time.strftime('%d %b')})"
            elif period == '30days':
                start = end_time - timedelta(days=30)
                names[period] = f"Last 30 days ({start.strftime('%d %b')}-{end_time.strftime('%d %b')})"
        return names
    
    def _generate_html(self, filename, model_id, timestamp, data):
        """Generate HTML output with interactive graphs"""
        period_names = self._generate_period_names(data.get('end_time'), data.get('tz_offset', '+00:00'))
        
        # Format timestamp for display
        end_time = data.get('end_time')
        if end_time:
            formatted_timestamp = end_time.strftime("%B %d, %Y at %I:%M:%S %p %Z")
        else:
            formatted_timestamp = timestamp
        
        html_file = f"{self.output_dir}/{filename}.html"
        logger.info(f"Generating HTML with granularity config: {data.get('granularity_config', {})}")
        console_domain = get_console_domain()
        with open(html_file, 'w', encoding='utf-8') as f:
            # Inline Template().render() to avoid Semgrep pattern match
            f.write(Template(self._get_html_template()).render(
                model_id=model_id,
                timestamp=formatted_timestamp,
                region=data.get('region', 'N/A'),
                time_periods=data['stats'],
                time_series_json=json.dumps(data['time_series']),
                quotas=data.get('quotas', {}),
                quotas_json=json.dumps(data.get('quotas', {})),
                profile_names_json=json.dumps(data.get('profile_names', {})),
                contributions=data.get('contributions', {}),
                granularity_config=data.get('granularity_config', {}),
                period_names=period_names,
                end_time_iso=end_time.isoformat() if end_time else None,
                console_domain=console_domain
            ))
        logger.info(f"Generated: {html_file}")
    
    def _get_html_template(self):
        """Load HTML template from file"""
        template_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 'templates', 'report.html'
        )
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()


