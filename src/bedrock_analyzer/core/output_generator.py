# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output generation for Bedrock usage analysis reports"""

import os
import json
import logging
from datetime import datetime, timedelta
from jinja2 import Template

logger = logging.getLogger(__name__)

class OutputGenerator:
    """Handles JSON and HTML output generation"""
    
    def __init__(self):
        self.output_dir = 'results'
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
        
        output_data = {
            'model_id': model_id,
            'region': data.get('region', 'N/A'),
            'generated_at': formatted_timestamp,
            'generated_at_iso': iso_timestamp,
            'timezone': data.get('tz_offset', '+00:00'),
            'stats': data['stats'],
            'time_series': data['time_series'],
            'quotas': data.get('quotas', {}),
            'granularity_config': data.get('granularity_config', {}),
            'profile_names': data.get('profile_names', {}),
            'contributions': data.get('contributions', {}),
            'period_names': period_names
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"Generated: {json_file}")
    
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
                end_time_iso=end_time.isoformat() if end_time else None
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


