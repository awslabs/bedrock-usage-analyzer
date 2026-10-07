# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output generation for Bedrock usage analysis reports"""

import os
import re
import functools
import json
import logging
from datetime import datetime, timedelta
from jinja2 import Environment, PackageLoader, select_autoescape

from bedrock_usage_analyzer.aws.invocation_logs import UNATTRIBUTED
from bedrock_usage_analyzer.core.metrics_fetcher import PERIOD_DAYS
from bedrock_usage_analyzer.utils.partition import get_region_info, get_service_quotas_console_url

logger = logging.getLogger(__name__)

# What the names in a report's application_profile_scope are
APPLICATION_PROFILE_SCOPE = 'Application inference profiles analyzed'
DEPLOYMENT_SCOPE = 'Custom model deployments analyzed'
IMPORTED_SCOPE = 'Imported models analyzed'


def safe_filename(label: str) -> str:
    """Turn a model/profile label into a file name (no path separators or odd characters)."""
    return re.sub(r'[^A-Za-z0-9_-]+', '_', label).strip('_') or 'report'


def https_url(value) -> str:
    """Template filter: keep only https:// links (no javascript: or data: URIs in href)."""
    text = str(value or '')
    return text if text.startswith('https://') else ''


def _local_time(value, tz=None) -> str:
    """Template helper: an ISO time from the breakdown in the report's timezone, as the
    period headings are (the value as given if it is not an ISO time)."""
    try:
        return datetime.fromisoformat(value).astimezone(tz).strftime('%Y-%m-%d %H:%M %Z')
    except (TypeError, ValueError):
        return '' if value is None else str(value)


def _breakdown_tpm(breakdown) -> list:
    """The breakdown chart's data: [[period, [[caller, TPM series, folded], ...]], ...].
    Lists, not objects, as callers are names chosen by whoever calls Bedrock (an object key
    such as "__proto__" would be dropped by the script); TPM only, the one series it draws;
    folded marks the '(N smaller groups)' row, whose name changes with N."""
    breakdown = breakdown or {}
    series = breakdown.get('time_series') or {}
    periods = breakdown.get('periods') or {}
    folded = {period: {r['name'] for r in (p.get('rows') or []) if r.get('folded')} for period, p in periods.items()}
    return [[period, [[name, s['TPM'], name in folded.get(period, ())] for name, s in rows.items() if s.get('TPM')]]
            for period, rows in series.items()]


class OutputGenerator:
    """Handles JSON and HTML output generation"""

    def __init__(self, output_dir: str = 'results'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Autoescape protects the report from profile names, tags or model IDs that contain markup.
        # This is a static report writer, not a Flask app; escaping is covered by
        # tests/test_analyzer_and_output.py::test_report_escapes_markup_and_script_breakout.
        self._env = Environment(  # nosemgrep: python.flask.security.xss.audit.direct-use-of-jinja2.direct-use-of-jinja2
            loader=PackageLoader('bedrock_usage_analyzer', 'templates'),  # works from a zip install too
            autoescape=select_autoescape(['html']),
        )
        self._env.filters['https_url'] = https_url
        # |tojson sorts keys by default; keep discovery order so the analyzed endpoint stays
        # the first chart series (colour and legend order)
        self._env.policies['json.dumps_kwargs'] = {'sort_keys': False}

    def generate(self, results):
        """Generate JSON and HTML output files with interactive graphs"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for model_id, data in results.items():
            base_filename = self._unique_basename(f"{safe_filename(data.get('file_label') or model_id)}-{timestamp}")

            self._generate_json(base_filename, model_id, timestamp, data)
            self._generate_html(base_filename, model_id, timestamp, data)

    def _unique_basename(self, base: str) -> str:
        """Append a counter when a report with the same name already exists."""
        candidate, n = base, 1
        while (os.path.exists(os.path.join(self.output_dir, f"{candidate}.json"))
               or os.path.exists(os.path.join(self.output_dir, f"{candidate}.html"))):
            n += 1
            candidate = f"{base}-{n}"
        return candidate

    @staticmethod
    def _region_info(data):
        region_name = data.get('region', 'N/A')
        return data.get('region_info') or (get_region_info(region_name) if region_name != 'N/A' else {})
    
    def _generate_json(self, filename, model_id, timestamp, data):
        """Generate JSON output"""
        json_file = os.path.join(self.output_dir, f"{filename}.json")
        
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
        quotas = data.get('quotas', {}) or {}
        if any(quotas.values()):
            console_url = get_service_quotas_console_url(data.get('region'))
            disclaimers['quota_mapping'] = (
                "Quota mappings were inferred using AI and may not be accurate. "
                "Always verify with AWS Service Quotas console"
                + (f": {console_url}" if console_url else ".")
            )

        # Process time_series to add per-metric disclaimers and quota info
        time_series = data['time_series']
        processed_time_series = self._add_time_series_metadata(time_series, quotas, disclaimers)

        output_data = {
            'model_id': model_id,
            'endpoint': data.get('endpoint', model_id),
            'application_profile_scope': data.get('application_profile_scope', []),
            # What the scope names are: application inference profiles or custom model deployments
            'scope_label': data.get('scope_label') or APPLICATION_PROFILE_SCOPE,
            'region': data.get('region', 'N/A'),
            'region_info': self._region_info(data),
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
            # Usage by caller from the model invocation logs (None: no breakdown asked for)
            'breakdown': data.get('breakdown'),
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
        if end_time is None:
            end_time = datetime.now().astimezone()
        names = {}
        for period, days in PERIOD_DAYS.items():  # the one period table
            start = end_time - timedelta(days=days)
            if days < 1:
                label, fmt = f"{round(days * 24)} hour", '%H:%M'
            else:
                label, fmt = f"{days:g} day{'s' if days != 1 else ''}", '%a %H:%M' if days == 1 else '%d %b'
            names[period] = f"Last {label} ({start.strftime(fmt)}-{end_time.strftime(fmt)})"
        return names
    
    def _generate_html(self, filename, model_id, timestamp, data):
        """Generate HTML output with interactive graphs"""
        period_names = self._generate_period_names(data.get('end_time'), data.get('tz_offset', '+00:00'))
        # The breakdown's periods end where the logs were read up to, before the report's end
        breakdown_end = ((data.get('breakdown') or {}).get('coverage') or {}).get('end')
        breakdown_period_names = period_names
        local = data.get('end_time').tzinfo if data.get('end_time') else None
        if breakdown_end:
            breakdown_period_names = self._generate_period_names(
                datetime.fromisoformat(breakdown_end).astimezone(local), data.get('tz_offset', '+00:00'))

        # Format timestamp for display
        end_time = data.get('end_time')
        if end_time:
            formatted_timestamp = end_time.strftime("%B %d, %Y at %I:%M:%S %p %Z")
        else:
            formatted_timestamp = timestamp

        html_file = os.path.join(self.output_dir, f"{filename}.html")
        logger.info(f"Generating HTML with granularity config: {data.get('granularity_config', {})}")
        region_name = data.get('region', 'N/A')

        # JSON blobs are passed as Python objects and serialised in the template with
        # |tojson, which escapes <, > and & so data cannot close the <script> element.
        template = self._env.get_template('report.html')
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(template.render(  # nosemgrep: python.flask.security.xss.audit.direct-use-of-jinja2.direct-use-of-jinja2
                model_id=model_id,
                endpoint=data.get('endpoint', model_id),
                application_profile_scope=data.get('application_profile_scope', []),
                scope_label=data.get('scope_label') or APPLICATION_PROFILE_SCOPE,
                imported=data.get('scope_label') == IMPORTED_SCOPE,
                timestamp=formatted_timestamp,
                region=region_name,
                region_info=self._region_info(data),
                time_periods=data['stats'],
                time_series=data['time_series'],
                quotas=data.get('quotas', {}),
                profile_names=data.get('profile_names', {}),
                contributions=data.get('contributions', {}),
                breakdown=data.get('breakdown'),
                breakdown_tpm=_breakdown_tpm(data.get('breakdown')),
                unattributed_label=UNATTRIBUTED,
                period_ms={period: days * 86400 * 1000 for period, days in PERIOD_DAYS.items()},
                granularity_config=data.get('granularity_config', {}),
                period_names=period_names,
                breakdown_period_names=breakdown_period_names,
                local_time=functools.partial(_local_time, tz=local),
                end_time_iso=end_time.isoformat() if end_time else None,
                service_quotas_console_url=get_service_quotas_console_url(
                    region_name if region_name != 'N/A' else None),
            ))
        logger.info(f"Generated: {html_file}")
