# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Breakdown of an endpoint's usage by IAM principal, session, principal tag or request
metadata key, from the model invocation logs.

The report's totals, quotas and throttles stay CloudWatch's. The breakdown rows come from
the invocation logs and are scaled to nothing: each row is that caller's own logged usage,
and what CloudWatch counted but the logs do not have (logging off for part of the window,
undelivered records) is shown as its own row, so the shares add up to the endpoint total.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from bedrock_usage_analyzer.aws.client_factory import create_client
from bedrock_usage_analyzer.aws.invocation_logs import (
    AWS_ERRORS, METADATA, PRINCIPAL, SESSION, TAG, UNATTRIBUTED, Breakdown, InvocationLogFetcher, LogsQueryError,
    logging_destination, model_id_forms, principal_tags)
from bedrock_usage_analyzer.core.errors import troubleshooting_hint

logger = logging.getLogger(__name__)

PERIOD_DAYS = {'1hour': 1 / 24, '1day': 1, '7days': 7, '14days': 14, '30days': 30}
ENABLE_HINT = ("To attribute usage to callers, enable model invocation logging to CloudWatch Logs in this "
               "region (Bedrock console > Settings, or PutModelInvocationLoggingConfiguration); turning "
               "off text, image, embedding and video delivery keeps only metadata.")


def _empty_minute():
    return [0.0, 0.0, 0.0]  # input tokens, output tokens, requests


class BreakdownBuilder:
    """Fetches the run's invocation-log rows once (for every target of the run) and builds
    each report's breakdown section from them."""

    def __init__(self, breakdown: Breakdown, region: str, bedrock_client, metrics_fetcher, stats_fn: Callable,
                 local_tz, account: Optional[str], parallel: Callable = None,
                 logs_client=None, iam_client=None):
        self.breakdown = breakdown
        self.region = region
        self.bedrock_client = bedrock_client
        self.metrics_fetcher = metrics_fetcher
        self._stats = stats_fn
        self.local_tz = local_tz
        self.account = account
        self._parallel = parallel
        self._logs_client = logs_client
        self._iam_client = iam_client
        self._rows: Optional[List[Dict]] = None
        self._unavailable: Optional[str] = None
        self.log_group: Optional[str] = breakdown.log_group
        self.coverage: Optional[Tuple[datetime, datetime]] = None
        self._tags: Optional[Dict[str, Dict[str, str]]] = None
        self._tag_error: Optional[Exception] = None
        self.notes: List[str] = []

    # ------------------------------------------------------------------ fetching

    def prepare(self, cw_ids: Iterable[str], end: datetime, days: float) -> Optional[str]:
        """Query the logs for every ModelId of the run, once. Returns why the breakdown is
        unavailable, or None."""
        if self._rows is not None or self._unavailable is not None:
            return self._unavailable
        try:
            if not self.log_group:
                self.log_group, reason = logging_destination(self.bedrock_client)
                if not self.log_group:
                    return self._give_up(f"{reason}. {ENABLE_HINT}")
            logs = self._logs_client or create_client('logs', self.region)
            fetcher = InvocationLogFetcher(logs, self.log_group)
            start = end - timedelta(days=days)
            covered_from = fetcher.coverage_start(start, end)
            if covered_from is None:
                return self._give_up(f"log group {self.log_group} does not exist in {self.region}. {ENABLE_HINT}")
            self.coverage = (covered_from, end)
            forms = model_id_forms(cw_ids, self.region, self.account)
            logger.info(f"  Reading model invocation logs from {self.log_group} "
                        f"({covered_from:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} UTC)...")
            self._rows = fetcher.fetch(forms, self.breakdown, covered_from, end)
            logger.info(f"  Invocation logs: {len(self._rows)} per-minute rows from {fetcher.queries_run} "
                        f"Logs Insights quer{'y' if fetcher.queries_run == 1 else 'ies'}, "
                        f"{fetcher.bytes_scanned / 1e9:.2f} GB scanned")
        except AWS_ERRORS + (LogsQueryError,) as e:
            hint = troubleshooting_hint(e, self.region)
            return self._give_up(f"could not read the model invocation logs: {e}" + (f" ({hint})" if hint else ""))
        if self.breakdown.kind == TAG or self.breakdown.kind == PRINCIPAL:
            self._read_tags({r['principal'] for r in self._rows})
        return None

    def _give_up(self, reason: str) -> str:
        self._unavailable = reason
        logger.info(f"  Breakdown by {self.breakdown.label} unavailable: {reason}")
        return reason

    def _read_tags(self, principals):
        iam = self._iam_client or create_client('iam', self.region)
        self._tags, self._tag_error = principal_tags(iam, principals, self._parallel)
        if self._tag_error is not None:
            message = f"some IAM principal tags could not be read ({self._tag_error})"
            if self.breakdown.kind == TAG:
                message += f"; principals without readable tags are grouped under '(tags not readable)'"
            self.notes.append(message)
            logger.info(f"  Note: {message}")

    # ------------------------------------------------------------------ grouping

    def _group_of(self, row) -> str:
        kind = self.breakdown.kind
        if kind == PRINCIPAL:
            return row['principal']
        if kind == SESSION:
            return row['key'] or row['principal']
        if kind == METADATA:
            return row['key'] if row['key'] not in (None, '') else f"(no {self.breakdown.key})"
        tags = (self._tags or {}).get(row['principal'])
        if tags is None and row['principal'].startswith(('role/', 'user/')) and self._tag_error is not None:
            return '(tags not readable)'
        value = (tags or {}).get(self.breakdown.key)
        return value if value not in (None, '') else f"(no {self.breakdown.key} tag)"

    def _selected(self, row) -> bool:
        wanted = self.breakdown.principals
        return not wanted or row['principal'] in wanted or (row['key'] or '') in wanted

    # ------------------------------------------------------------------ per report

    def section(self, final_model_ids, profile_names, fetched_cw: Dict, granularity_config: Dict,
                time_periods: Iterable[str]) -> Optional[Dict]:
        """The breakdown section of one report, or None when no breakdown was asked for."""
        base = {'kind': self.breakdown.kind, 'key': self.breakdown.key, 'label': self.breakdown.label,
                'source': 'model invocation logs', 'log_group': self.log_group,
                'principal_filter': list(self.breakdown.principals), 'notes': list(self.notes)}
        if self._unavailable is not None or self._rows is None or self.coverage is None:
            return {**base, 'unavailable': self._unavailable or 'not fetched', 'periods': {}, 'time_series': {}}
        ids = set(final_model_ids)
        end = next((d['end_time'] for d in fetched_cw.values() if d.get('end_time')), self.coverage[1])
        covered_from = self.coverage[0]

        groups: Dict[str, Dict] = {}
        logged_all = defaultdict(_empty_minute)  # every logged caller, before the principal filter
        for row in self._rows:
            if row['cw_id'] not in ids:
                continue
            minute = row['minute']
            totals = logged_all[minute]
            totals[0] += row['input']; totals[1] += row['output']; totals[2] += row['requests']
            if not self._selected(row):
                continue
            name = self._group_of(row)
            group = groups.setdefault(name, {'minutes': defaultdict(_empty_minute), 'principals': set(), 'via': set()})
            values = group['minutes'][minute]
            values[0] += row['input']; values[1] += row['output']; values[2] += row['requests']
            group['principals'].add(row['principal'])
            group['via'].add(profile_names.get(row['cw_id'], row['cw_id']))

        cw_minutes = self._cloudwatch_minutes(fetched_cw, final_model_ids, covered_from)
        remainder = {}
        others = {}
        for minute, (inp, out, req) in cw_minutes.items():
            logged = logged_all.get(minute, (0.0, 0.0, 0.0))
            gap = (max(inp - logged[0], 0.0), max(out - logged[1], 0.0), max(req - logged[2], 0.0))
            if any(gap):
                remainder[minute] = list(gap)
        if self.breakdown.principals:
            selected = defaultdict(_empty_minute)
            for group in groups.values():
                for minute, values in group['minutes'].items():
                    for i in range(3):
                        selected[minute][i] += values[i]
            for minute, values in logged_all.items():
                rest = [max(values[i] - selected.get(minute, (0, 0, 0))[i], 0.0) for i in range(3)]
                if any(rest):
                    others[minute] = rest

        series = {name: (g['minutes'], sorted(g['principals']), sorted(g['via'])) for name, g in groups.items()}
        if others:
            series['(other principals)'] = (others, [], [])
        if remainder:
            series[UNATTRIBUTED] = (remainder, [], [])

        periods, time_series = {}, {}
        for period in time_periods:
            period_start = end - timedelta(days=PERIOD_DAYS[period])
            window_start = max(period_start, covered_from)
            total = _window_sum(cw_minutes, window_start, end)
            logged = _window_sum(logged_all, window_start, end)
            rows, period_series = [], {}
            for name, (minutes, principals, via) in series.items():
                ts_data = self.metrics_fetcher.slice_and_process_data(
                    self._dataset(minutes, end), period, granularity_config)
                stats = self._stats(ts_data, period)
                if name == UNATTRIBUTED:
                    # Over the whole window, not minute by minute: a record logged in the minute
                    # after CloudWatch counted it is not missing (its TPM series stays per minute)
                    tokens = max(total[0] + total[1] - logged[0] - logged[1], 0.0)
                    requests = max(total[2] - logged[2], 0.0)
                else:
                    tokens = _sum(stats, 'InputTokenCount') + _sum(stats, 'OutputTokenCount')
                    requests = _sum(stats, 'Invocations')
                if not tokens and not requests:
                    continue
                rows.append(self._row(name, principals, via, stats, tokens, requests, total, period))
                period_series[name] = {k: ts_data[k] for k in ('TPM', 'RPM') if k in ts_data}
            rows.sort(key=lambda r: (r['name'] == UNATTRIBUTED, r['name'] == '(other principals)', -r['tokens']))
            periods[period] = {'rows': rows, 'total_tokens': total[0] + total[1], 'total_requests': total[2],
                               'covered_from': window_start.isoformat(),
                               'partial': window_start > period_start}
            time_series[period] = period_series
        return {**base, 'coverage': {'start': covered_from.isoformat(), 'end': end.isoformat()},
                'periods': periods, 'time_series': time_series}

    def _row(self, name, principals, via, stats, tokens, requests, total, period) -> Dict:
        tags = {}
        if self.breakdown.kind == PRINCIPAL and self._tags:
            tags = self._tags.get(name, {})
        return {
            'name': name,
            'principals': principals if self.breakdown.kind != PRINCIPAL else [],
            'via': via,
            'tags': tags,
            'tokens': tokens,
            'requests': requests,
            'share_tokens': tokens / (total[0] + total[1]) if total[0] + total[1] else None,
            'share_requests': requests / total[2] if total[2] else None,
            'tpm_p50': _stat(stats, 'TPM_1min', 'p50'), 'tpm_p90': _stat(stats, 'TPM_1min', 'p90'),
            'tpm_max': _max(stats, 'TPM_1min'),
            'rpm_p50': _stat(stats, 'RPM_1min', 'p50'), 'rpm_p90': _stat(stats, 'RPM_1min', 'p90'),
            'rpm_max': _max(stats, 'RPM_1min'),
            'tpd_avg': _stat(stats, 'TPD', 'avg') if period != '1hour' else 0,
            'tpd_max': _max(stats, 'TPD') if period != '1hour' else 0,
        }

    def _dataset(self, minutes: Dict[datetime, List[float]], end: datetime) -> Dict:
        """Per-minute values in the shape the CloudWatch fetcher returns, for its slicing."""
        stamps = sorted(minutes)
        return {'end_time': end, '60_token': {
            'timestamps': [m.astimezone(self.local_tz) for m in stamps],
            'data': {'input_tokens': [minutes[m][0] for m in stamps],
                     'output_tokens': [minutes[m][1] for m in stamps],
                     'invocations': [minutes[m][2] for m in stamps]},
            'period': 60}}

    @staticmethod
    def _cloudwatch_minutes(fetched_cw: Dict, final_model_ids, covered_from: datetime) -> Dict[datetime, List[float]]:
        """CloudWatch's per-minute totals of the report's ModelIds, from the covered time on."""
        minutes = defaultdict(_empty_minute)
        for cw_id in final_model_ids:
            token = (fetched_cw.get(cw_id) or {}).get('60_token') or {}
            data = token.get('data') or {}
            for i, stamp in enumerate(token.get('timestamps') or []):
                if stamp < covered_from:
                    continue
                values = minutes[stamp]
                for j, key in enumerate(('input_tokens', 'output_tokens', 'invocations')):
                    column = data.get(key) or []
                    if i < len(column) and column[i] is not None:
                        values[j] += column[i]
        return minutes


def _window_sum(minutes: Dict[datetime, List[float]], start: datetime, end: datetime) -> List[float]:
    """Input tokens, output tokens and requests of the minutes in [start, end]."""
    total = [0.0, 0.0, 0.0]
    for minute, values in minutes.items():
        if start <= minute <= end:
            for i in range(3):
                total[i] += values[i]
    return total


def _sum(stats, metric) -> float:
    return float((stats.get(metric) or {}).get('sum') or 0)


def _stat(stats, metric, name) -> float:
    return float((stats.get(metric) or {}).get(name) or 0)


def _max(stats, metric) -> float:
    values = (stats.get(metric) or {}).get('values') or []
    return float(max(values)) if values else 0.0
