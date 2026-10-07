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
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from bedrock_usage_analyzer.aws.client_factory import create_client
from bedrock_usage_analyzer.aws.sts import get_account_id
from bedrock_usage_analyzer.aws.invocation_logs import (
    MAX_ROWS, METADATA, PRINCIPAL, SESSION, TAG, UNATTRIBUTED, UNKNOWN_CALLER, Breakdown, InvocationLogFetcher,
    LogsQueryError, has_tags, logging_destination, main_error, model_id_forms, next_minute, principal_tags)
from bedrock_usage_analyzer.core.errors import AWS_ERRORS, troubleshooting_hint
from bedrock_usage_analyzer.core.metrics_fetcher import PERIOD_DAYS

logger = logging.getLogger(__name__)

# Groups shown as their own rows (and chart lines); smaller ones are summed into one row
MAX_GROUPS = 30
OTHER_PRINCIPALS = '(other principals)'
# Invocation log records reach the log group seconds after the call (minutes at worst), so
# the breakdown ends this long before the run started; its periods end there too, a window
# CloudWatch's data of every report covers
LOG_DELIVERY_DELAY = timedelta(minutes=5)
# CloudWatch keeps 1-minute metric data for 15 days
CLOUDWATCH_MINUTE_DAYS = 15
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
                 logs_client=None, iam_client=None, known_models: Iterable[str] = ()):
        self.breakdown = breakdown
        self.region = region
        self.known_models = set(known_models)  # the region's foundation model IDs
        self._wanted = {p.lower() for p in breakdown.principals}
        self.bedrock_client = bedrock_client
        self.metrics_fetcher = metrics_fetcher
        self._stats = stats_fn
        self.local_tz = local_tz
        self.account = account
        self._parallel = parallel
        self._logs_client = logs_client
        self._iam_client = iam_client
        # The logged per-minute rows by CloudWatch ModelId; None until they are read
        self._rows_by_id: Optional[Dict[str, List[Dict]]] = None
        self._unavailable: Optional[str] = None
        self._truncated: List[Tuple[datetime, datetime, frozenset]] = []  # query windows cut at the row limit
        self.log_group: Optional[str] = breakdown.log_group
        self.coverage: Optional[Tuple[datetime, datetime]] = None
        self._tags: Dict[str, Dict[str, str]] = {}
        self._tags_read: Set[str] = set()
        self._tag_errors: Dict[str, Exception] = {}  # principals whose tags could not be read

    # ------------------------------------------------------------------ fetching

    def prepare(self, cw_ids: Iterable[str], now: datetime, days: float) -> Optional[str]:
        """Query the logs for every ModelId of the run, once, up to LOG_DELIVERY_DELAY before
        now (records reach the log group seconds to minutes after the call). Returns why the
        breakdown is unavailable, or None. Never raises: the CloudWatch reports go on."""
        if self._rows_by_id is not None or self._unavailable is not None:
            return self._unavailable
        try:
            return self._prepare(cw_ids, now, days)
        except Exception as e:  # an unexpected error must not end the run
            logger.debug("Reading the invocation logs failed", exc_info=True)
            return self._give_up(f"could not read the model invocation logs: {e}")

    def _prepare(self, cw_ids: Iterable[str], now: datetime, days: float) -> Optional[str]:
        # On a minute (the per-minute rows end there); now itself stays the real clock,
        # which log retention counts back from
        end = (now - LOG_DELIVERY_DELAY).replace(second=0, microsecond=0)
        try:
            if not self.log_group:
                self.log_group, reason = logging_destination(self.bedrock_client)
                if not self.log_group:
                    return self._give_up(f"{reason}. {ENABLE_HINT}")
            if self.account is None:
                # Application and system inference profile ARNs in the logs carry the account
                try:
                    self.account = get_account_id(self.region)
                except Exception as e:  # STS unreachable: those spellings are not matched
                    logger.info(f"  Note: account ID unavailable ({e}); usage logged under profile ARNs "
                                f"is shown as '{UNATTRIBUTED}'")
            logs = self._logs_client or create_client('logs', self.region)
            starter = self._logs_client or create_client('logs', self.region, single_attempt=True)
            fetcher = InvocationLogFetcher(logs, self.log_group, start_client=starter)
            # No further back than CloudWatch keeps 1-minute data (counted from now), which the
            # shares and the not-logged row are compared with; longer periods are marked partly
            # covered
            start = now - timedelta(days=min(days, CLOUDWATCH_MINUTE_DAYS))
            covered_from = fetcher.coverage_start(start, end, now)
            if covered_from is None:
                return self._give_up(f"log group {self.log_group} does not exist in {self.region}. {ENABLE_HINT}")
            if covered_from >= end:
                # Created after the breakdown's end: nothing to read yet
                return self._give_up(f"log group {self.log_group} holds no records from before "
                                     f"{end:%Y-%m-%d %H:%M} UTC yet: it was created after that "
                                     f"(logging started recently); run again later")
            forms = model_id_forms(cw_ids, self.region, self.account, self.known_models)
            logger.info(f"  Reading model invocation logs from {self.log_group} "
                        f"({covered_from:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} UTC)...")
            rows = fetcher.fetch(forms, self.breakdown, covered_from, end)
            logger.info(f"  Invocation logs: {len(rows)} per-minute rows from {fetcher.queries_run} "
                        f"Logs Insights quer{'y' if fetcher.queries_run == 1 else 'ies'}, "
                        f"{fetcher.bytes_scanned / 1e9:.2f} GB scanned")
        except AWS_ERRORS + (LogsQueryError,) as e:
            hint = troubleshooting_hint(e, self.region)
            return self._give_up(f"could not read the model invocation logs: {e}" + (f" ({hint})" if hint else ""))
        self.coverage = (covered_from, end)
        self._truncated = sorted(fetcher.truncated, key=lambda w: (w[0], w[1]))
        if self._truncated:
            logger.info(f"  Warning: {len(self._truncated)} invocation-log query window(s) returned the "
                        f"{MAX_ROWS}-row limit at the smallest split; some callers there are missing")
        self._rows_by_id = {}
        for row in rows:
            self._rows_by_id.setdefault(row['cw_id'], []).append(row)
        if self.breakdown.kind == TAG:
            # Grouping needs every selected principal's tags (--principal limits them); a
            # principal breakdown reads only its rows' tags, when the sections are built
            self._read_tags({r['principal'] for r in rows if self._selected(r)})
        return None

    def _give_up(self, reason: str) -> str:
        self._unavailable = reason
        logger.info(f"  Breakdown by {self.breakdown.label} unavailable: {reason}")
        return reason

    def _read_tags(self, principals):
        principals = {p for p in principals if has_tags(p)}
        if not principals:
            return
        self._tags_read |= principals
        try:
            if self._iam_client is None:  # one client for every report's tag reads
                self._iam_client = create_client('iam', self.region)
            tags, errors = principal_tags(self._iam_client, principals, self._parallel)
        except Exception as e:  # tags are never worth the rows: these are '(tags not readable)'
            tags, errors = {}, {p: e for p in principals}
        self._tags.update(tags)
        # Reported in each report's notes, with the error of that report's principals
        self._tag_errors.update(errors)

    def _notes(self, principals: Set[str], start: datetime, no_cloudwatch: List[str],
               cw_ids: Iterable[str] = (), left_out: Iterable[str] = ()) -> List[str]:
        """This report's notes: query windows of its ModelIds cut at the row limit, ModelIds
        left out for want of CloudWatch data, principals whose tags could not be read, and
        the --principal values none of its logged calls (from start on) came from."""
        notes = []
        cut = [(a, b) for a, b, ids in self._truncated if b > start and ids & set(cw_ids)]
        if cut:
            # The tool cut these, not the logs: say so, or the remainder row would blame logging
            shortest = min(b - a for a, b in cut)
            notes.append(f"{len(cut)} invocation-log query window(s) between {max(cut[0][0], start):%Y-%m-%d %H:%M} and "
                         f"{cut[-1][1]:%Y-%m-%d %H:%M} UTC returned the Logs Insights limit of {MAX_ROWS} rows "
                         f"even when split down to {int(shortest.total_seconds() // 60)} minutes; the usage "
                         f"of callers left out there is counted in '{UNATTRIBUTED}'")
        if no_cloudwatch:
            notes.append(f"CloudWatch's 1-minute data could not be fetched for {', '.join(no_cloudwatch)}, "
                         f"so their logged calls are left out of the breakdown; run again to include them")
        failed = sorted(principals & set(self._tag_errors))
        if failed:
            error = main_error(self._tag_errors[p] for p in failed)
            message = f"some IAM principal tags could not be read ({error})"
            if self.breakdown.kind == TAG:
                message += "; principals without readable tags are grouped under '(tags not readable)'"
            notes.append(message)
        if self._wanted:
            # A --principal that no logged caller matches (a typo, or no calls) shows only
            # '(other principals)': say so
            # (one whose calls are only in rows left out for want of CloudWatch data did call)
            seen = {p.lower() for p in principals} | {p.lower() for p in left_out}
            missing = [p for p in self.breakdown.principals if p.lower() not in seen]
            if missing:
                end = self.coverage[1]
                notes.append(f"no logged call from {', '.join(missing)} in "
                             f"{start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} UTC")
        return notes

    # ------------------------------------------------------------------ grouping

    def _group_of(self, row) -> str:
        kind = self.breakdown.kind
        if kind == PRINCIPAL:
            return row['principal']
        if kind == SESSION:
            return row['key'] or row['principal']
        if kind == METADATA:
            return _caller_value(row['key']) if row['key'] not in (None, '') else f"(no {self.breakdown.key})"
        if row['principal'] == UNKNOWN_CALLER:  # no caller: no tags, but not '(no <key> tag)' either
            return UNKNOWN_CALLER
        if row['principal'] in self._tag_errors:
            return '(tags not readable)'
        tags = self._tags.get(row['principal'])
        # IAM tag keys are case-insensitive (a principal cannot have both Team and team)
        wanted = self.breakdown.key.lower()
        value = next((v for k, v in (tags or {}).items() if k.lower() == wanted), None)
        return _caller_value(value) if value not in (None, '') else f"(no {self.breakdown.key} tag)"

    def _selected(self, row) -> bool:
        # IAM role and user names are case-insensitive; the logs carry their real case
        return not self._wanted or row['principal'].lower() in self._wanted

    # ------------------------------------------------------------------ per report

    def section(self, final_model_ids, profile_names, fetched_cw: Dict, granularity_config: Dict,
                time_periods: Iterable[str]) -> Optional[Dict]:
        """The breakdown section of one report. A section that cannot be built says why
        instead of failing the report."""
        empty = {'notes': [], 'periods': {}, 'time_series': {}}
        if self._unavailable is not None or self._rows_by_id is None:
            built = {**empty, 'unavailable': self._unavailable or 'not fetched'}
        else:
            try:
                built = self._build(final_model_ids, profile_names, fetched_cw, granularity_config, time_periods)
            except Exception as e:  # never lose the CloudWatch report to the breakdown
                logger.debug("Breakdown section failed", exc_info=True)
                logger.info(f"  Breakdown by {self.breakdown.label} unavailable for this report: {e}")
                built = {**empty, 'unavailable': f"the breakdown could not be built: {e}"}
        for note in built['notes']:
            logger.info(f"  Note: {note}")
        return {'kind': self.breakdown.kind, 'key': self.breakdown.key, 'label': self.breakdown.label,
                'source': 'model invocation logs', 'log_group': self.log_group,
                'principal_filter': list(self.breakdown.principals), **built}

    def _build(self, final_model_ids, profile_names, fetched_cw: Dict, granularity_config: Dict,
               time_periods: Iterable[str]) -> Dict:
        # Every period of the breakdown ends where the logs were read up to (CloudWatch,
        # read later for each report, covers that window too), and so do its totals
        logs_from, end = self.coverage
        # This report's CloudWatch 1-minute data starts 15 days before it was fetched (after
        # the logs were read): the breakdown starts where both sources have data
        cw_end = max((d['end_time'] for d in fetched_cw.values() if d.get('end_time')), default=end)
        # The minute after: cw_end is rounded down to a minute, and the data ages out from the
        # real fetch time (up to a minute later), so the boundary minute may be partly gone
        cw_from = next_minute(cw_end - timedelta(days=CLOUDWATCH_MINUTE_DAYS)) + timedelta(minutes=1)
        covered_from = max(logs_from, cw_from)

        # name -> (its minutes, the last minute of each of its principals, and of each profile)
        groups: Dict[str, Tuple[Dict, Dict, Dict]] = {}
        logged_all = defaultdict(_empty_minute)  # every logged caller, before the principal filter
        others = defaultdict(_empty_minute)  # callers left out by --principal
        report_principals: Set[str] = set()
        # A ModelId whose CloudWatch 1-minute fetch failed has no total to compare its logged
        # calls with (they would push the shares past 100%): its rows are left out, with a note
        no_cloudwatch = [cw_id for cw_id in dict.fromkeys(final_model_ids) if self._rows_by_id.get(cw_id)
                         and ((fetched_cw.get(cw_id) or {}).get('60_token') or {}).get('fetch_failed')]
        used_ids = [cw_id for cw_id in dict.fromkeys(final_model_ids) if cw_id not in no_cloudwatch]
        # Their principals did call (only the rows are left out): not 'no logged call from'
        left_out = {r['principal'] for cw_id in no_cloudwatch for r in self._rows_by_id[cw_id]
                    if r['minute'] >= covered_from}
        for row in (r for cw_id in used_ids for r in self._rows_by_id.get(cw_id, ())):
            minute = row['minute']
            if minute < covered_from:
                continue
            _add(logged_all[minute], row)
            if not self._selected(row):
                _add(others[minute], row)
                continue
            report_principals.add(row['principal'])
            name = self._group_of(row)
            minutes, principals, via = groups.setdefault(name, (defaultdict(_empty_minute), {}, {}))
            _add(minutes[minute], row)
            # The last minute of each principal and profile, so that each period lists only
            # those it has calls from (every period ends at the same time)
            _latest(principals, row['principal'], minute)
            _latest(via, profile_names.get(row['cw_id'], row['cw_id']), minute)

        cw_minutes = self._cloudwatch_minutes(fetched_cw, final_model_ids, covered_from, end)
        remainder = {}
        for minute, (inp, out, req) in cw_minutes.items():
            logged = logged_all.get(minute, (0.0, 0.0, 0.0))
            gap = (max(inp - logged[0], 0.0), max(out - logged[1], 0.0), max(req - logged[2], 0.0))
            if any(gap):
                remainder[minute] = list(gap)

        periods, time_series = {}, {}
        for period in time_periods:
            period_start = end - timedelta(days=PERIOD_DAYS[period])
            window_start = max(period_start, covered_from)
            total = _window_sum(cw_minutes, window_start, end)
            logged = _window_sum(logged_all, window_start, end)
            # The period's own largest callers keep their rows (a caller that started today
            # leads the last hour even if it is small over 30 days)
            series, folded_name = _fold_small_groups(groups, window_start, end)
            # Only when they have usage in this period, like the groups
            if any(_window_sum(others, window_start, end)):
                series[OTHER_PRINCIPALS] = (others, {}, {})
            if any(_window_sum(remainder, window_start, end)):
                series[UNATTRIBUTED] = (remainder, {}, {})
            rows, period_series = [], {}
            for name, (minutes, principal_last, via_last) in series.items():
                # Only the period's own minutes (the fetcher slices to the same start), so the
                # 1-hour period does not sort and convert 15 days of minutes
                dataset = self._dataset({m: v for m, v in minutes.items() if m >= period_start}, end)
                ts_data = self.metrics_fetcher.slice_and_process_data(dataset, period, granularity_config)
                principals, via = _active(principal_last, window_start), _active(via_last, window_start)
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
                rows.append(self._row(name, principals, via, stats, tokens, requests, total, period,
                                      folded=name == folded_name))
                period_series[name] = {k: ts_data[k] for k in ('TPM', 'RPM') if k in ts_data}
            rows.sort(key=lambda r: (r['name'] == UNATTRIBUTED, r['name'] == OTHER_PRINCIPALS,
                                     r['name'] == folded_name, -r['tokens']))
            # Partly covered: the period starts before CloudWatch's 1-minute data, or before
            # the logs (retention, a newer log group); the report says which
            partial = window_start > period_start
            periods[period] = {'rows': rows, 'total_tokens': total[0] + total[1], 'total_requests': total[2],
                               'covered_from': window_start.isoformat(), 'covered_to': end.isoformat(),
                               'partial': partial,
                               'partial_reason': (None if not partial else
                                                  'cloudwatch' if cw_from >= logs_from else 'logs')}
            time_series[period] = period_series
        if self.breakdown.kind == PRINCIPAL:
            self._add_tags(periods)
        return {'coverage': {'start': covered_from.isoformat(), 'end': end.isoformat()},
                'notes': self._notes(report_principals, covered_from, no_cloudwatch, final_model_ids, left_out),
                'periods': periods, 'time_series': time_series}

    def _add_tags(self, periods: Dict) -> None:
        """The IAM tags of the principals shown as rows, read once each (not those of every
        logged caller: only the rows show them)."""
        names = {r['name'] for p in periods.values() for r in p['rows']}
        self._read_tags(names - self._tags_read)
        for p in periods.values():
            for r in p['rows']:
                r['tags'] = self._tags.get(r['name'], {})

    def _row(self, name, principals, via, stats, tokens, requests, total, period, folded=False) -> Dict:
        return {
            'name': name,
            # A principal row is its own principal; the summed row lists the ones it holds
            'principals': principals if self.breakdown.kind != PRINCIPAL or folded else [],
            'folded': folded,  # the '(N smaller groups)' row: one chart color in every period
            'via': via,
            'tags': {},
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
    def _cloudwatch_minutes(fetched_cw: Dict, final_model_ids, covered_from: datetime,
                            covered_to: datetime) -> Dict[datetime, List[float]]:
        """CloudWatch's per-minute totals of the report's ModelIds over the logs' window."""
        minutes = defaultdict(_empty_minute)
        for cw_id in final_model_ids:
            token = (fetched_cw.get(cw_id) or {}).get('60_token') or {}
            data = token.get('data') or {}
            for i, stamp in enumerate(token.get('timestamps') or []):
                if stamp < covered_from or stamp >= covered_to:
                    continue
                values = minutes[stamp]
                for j, key in enumerate(('input_tokens', 'output_tokens', 'invocations')):
                    column = data.get(key) or []
                    if i < len(column) and column[i] is not None:
                        values[j] += column[i]
        return minutes


def _fold_small_groups(series: Dict, start: datetime, end: datetime) -> Tuple[Dict, Optional[str]]:
    """The groups used in [start, end], the MAX_GROUPS largest (tokens, then requests) as they
    are and the rest summed into one row, so that thousands of sessions stay a readable report.
    Returns (series, the summed row's name or None)."""
    sizes = {}
    for name, (minutes, _, _) in series.items():
        used = _window_sum(minutes, start, end)
        if any(used):
            sizes[name] = (used[0] + used[1], used[2], name)
    if len(sizes) <= MAX_GROUPS:
        return {name: series[name] for name in sizes}, None
    ranked = sorted(sizes, key=sizes.get, reverse=True)
    kept = {name: series[name] for name in ranked[:MAX_GROUPS - 1]}
    rest = ranked[MAX_GROUPS - 1:]
    folded = defaultdict(_empty_minute)
    principals: Dict[str, datetime] = {}
    via: Dict[str, datetime] = {}
    for name in rest:
        minutes, names, routes = series[name]
        for minute, values in minutes.items():
            for i in range(3):
                folded[minute][i] += values[i]
        for target, source in ((principals, names), (via, routes)):
            for item, last in source.items():
                _latest(target, item, last)
    folded_name = f"({len(rest)} smaller groups)"
    kept[folded_name] = (folded, principals, via)
    return kept, folded_name


def _latest(last_seen: Dict[str, datetime], item: str, minute: datetime) -> None:
    """Keep the latest minute item was seen at."""
    if item not in last_seen or minute > last_seen[item]:
        last_seen[item] = minute


def _active(last_seen: Dict[str, datetime], start: datetime) -> List[str]:
    """The items seen at or after start (every period ends at the breakdown's end)."""
    return sorted(item for item, last in last_seen.items() if last >= start)


def _caller_value(value: str) -> str:
    """A metadata or tag value as a row name. The tool's own rows are named '(...)', so a
    value starting with '(' is quoted and can never take one of their places; one starting
    with a quote is quoted too, so that two different values never share a name."""
    return f'"{value}"' if value.startswith(('(', '"')) else value


def _add(values: List[float], row: Dict) -> None:
    values[0] += row['input']; values[1] += row['output']; values[2] += row['requests']


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
