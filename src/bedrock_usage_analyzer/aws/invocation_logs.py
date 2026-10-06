# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Usage per caller from Amazon Bedrock model invocation logs (CloudWatch Logs Insights).

CloudWatch's AWS/Bedrock metrics have one dimension, ModelId, so every caller of one
endpoint shares one series. The model invocation log records each request's caller
(identity.arn), its model or profile ID and its token counts, so a Logs Insights query
can split an endpoint's per-minute tokens and requests by IAM principal, by session, or
by a requestMetadata key. Grouping by an IAM principal tag is done here from the
principals' tags (iam:ListRoleTags / iam:ListUserTags), as the logs carry no tags.

Only metadata fields are queried: prompts and completions are never read.
"""

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from bedrock_usage_analyzer.aws.bedrock import endpoint_id, split_profile_id
from bedrock_usage_analyzer.core.errors import AWS_ERRORS, is_access_denied
from bedrock_usage_analyzer.utils.partition import build_arn

logger = logging.getLogger(__name__)


PRINCIPAL = 'principal'
SESSION = 'session'
TAG = 'tag'
METADATA = 'metadata'
KINDS = (PRINCIPAL, SESSION, TAG, METADATA)

# Logs Insights returns at most this many rows per query
MAX_ROWS = 10000
# Queries run at once (the account allows 100 across all users, and 10 StartQuery per second)
MAX_CONCURRENT_QUERIES = 4
QUERY_TIMEOUT_SECONDS = 900
# A query window is never split below this, even if it returns MAX_ROWS rows
MIN_WINDOW = timedelta(minutes=10)
# StartQuery accepts query strings of up to 10,000 characters; each window adds its end
# filter (QUERY_END_FILTER_LENGTH characters at most) to the query
MAX_QUERY_LENGTH = 10000
QUERY_END_FILTER_LENGTH = 40
_QUERY_HEAD = 'fields @timestamp\n'

# requestMetadata keys (Bedrock allows letters, digits, whitespace and :_@$#=/+,-.) go into
# the query inside backticks, which none of these characters can close; IAM tag keys are
# only looked up in ListRoleTags/ListUserTags results
_KEY_PATTERN = re.compile(r'^[A-Za-z0-9 :_@$#=/+,.-]{1,256}$')
_TAG_KEY_PATTERN = re.compile(r'^[A-Za-z0-9 _.:/=+@-]{1,128}$')
# modelId values in a query string literal: model, profile and deployment IDs and ARNs
_MODEL_ID_PATTERN = re.compile(r'^[A-Za-z0-9_.:/-]{1,2048}$')
# An IAM principal given with --principal: 'role/<name>', 'user/<name>' or an ARN
_PRINCIPAL_PATTERN = re.compile(r'^[A-Za-z0-9_.:/=+,@-]{1,2048}$')

UNATTRIBUTED = '(not in the invocation logs)'


class BreakdownError(ValueError):
    """An invalid --breakdown or --principal value."""


@dataclass(frozen=True)
class Breakdown:
    """How usage is attributed: by IAM principal (role or user), session, principal tag or
    requestMetadata key, optionally limited to some principals."""
    kind: str = PRINCIPAL
    key: Optional[str] = None
    principals: Tuple[str, ...] = ()
    log_group: Optional[str] = None

    @classmethod
    def parse(cls, value: str, principals: Iterable[str] = (), log_group: Optional[str] = None) -> 'Breakdown':
        """'principal', 'session', 'tag:<key>' or 'metadata:<key>' (raises BreakdownError)."""
        kind, _, key = (value or '').strip().partition(':')
        kind = kind.lower()
        if kind not in KINDS:
            raise BreakdownError(f"unknown breakdown '{value}': use principal, session, tag:<key> or metadata:<key>")
        if kind == TAG:
            if not _TAG_KEY_PATTERN.match(key) or not key.strip():
                raise BreakdownError(f"'{value}' needs a key of letters, digits, spaces and _.:/=+@- (e.g. tag:team)")
        elif kind == METADATA:
            if not _KEY_PATTERN.match(key) or not key.strip():
                raise BreakdownError(f"'{value}' needs a key of letters, digits, spaces and :_@$#=/+,-. "
                                     f"(e.g. metadata:team)")
        elif key:
            raise BreakdownError(f"'{kind}' takes no key: '{value}'")
        names = tuple(normalize_principal(p.strip()) for p in principals if p and p.strip())
        for name in names:
            if not _PRINCIPAL_PATTERN.match(name) or not name.startswith(('role/', 'user/', 'arn:')):
                raise BreakdownError(f"invalid principal '{name}': use role/<name>, user/<name> or an IAM ARN")
        if log_group is not None and not re.match(r'^[A-Za-z0-9_./#-]{1,512}$', log_group):
            raise BreakdownError(f"invalid log group name '{log_group}'")
        return cls(kind, key or None, names, log_group)

    @property
    def label(self) -> str:
        """What the breakdown rows are, for report headings."""
        return {PRINCIPAL: 'IAM principal', SESSION: 'IAM principal session',
                TAG: f"IAM principal tag '{self.key}'", METADATA: f"request metadata '{self.key}'"}[self.kind]


def normalize_principal(arn: str) -> str:
    """The IAM principal a caller ARN belongs to, as 'role/<name>' or 'user/<name>'.

    'arn:aws:sts::111122223333:assumed-role/Billing/session-1' -> 'role/Billing' (every
    session of a role is that role), 'arn:aws:iam::111122223333:user/ops/alice' ->
    'user/alice'. IAM paths are dropped (role and user names are unique in an account, and
    assumed-role ARNs carry no path). Other callers (root, federated users) keep their ARN.
    """
    value = (arn or '').strip()
    resource = value
    if value.startswith('arn:'):
        resource = value.split(':', 5)[5] if value.count(':') >= 5 else ''
    if resource.startswith('assumed-role/'):
        return 'role/' + resource.split('/')[1]
    if resource.startswith(('role/', 'user/')):
        kind, _, rest = resource.partition('/')
        return f"{kind}/{rest.rsplit('/', 1)[-1]}"
    return value


def _from_query_principal(value: str) -> str:
    """The query's principal column ('assumed-role/<name>', 'user/...', or an ARN) in the
    normalize_principal form."""
    if value.startswith('assumed-role/'):
        return 'role/' + value.split('/', 2)[1]
    return normalize_principal(value)


def logging_destination(bedrock_client) -> Tuple[Optional[str], str]:
    """The region's invocation log group, and why there is none.

    Returns (log group, '') or (None, reason). Raises AWS errors other than the ones that
    mean "not configured".
    """
    config = (bedrock_client.get_model_invocation_logging_configuration() or {}).get('loggingConfig') or {}
    group = (config.get('cloudWatchConfig') or {}).get('logGroupName')
    if group:
        return group, ''
    if config.get('s3Config'):
        return None, ("model invocation logs go to S3 only; the breakdown reads them from "
                      "CloudWatch Logs, so add a CloudWatch Logs destination")
    return None, "model invocation logging is not enabled in this region"


def model_id_forms(cw_ids: Iterable[str], region: str, account: Optional[str]) -> Dict[str, str]:
    """Every spelling of each CloudWatch ModelId value that invocation logs may record.

    The log's modelId is what the caller passed: a model ID or its ARN, an inference profile
    ID or ARN, an application profile ARN, or a deployment ARN. CloudWatch reports all of them
    under one ModelId value (the profile or model ID, the application profile ID, the
    deployment ARN). Returns {log spelling: CloudWatch ModelId value}.
    """
    forms: Dict[str, str] = {}
    for cw_id in cw_ids:
        forms[cw_id] = cw_id
        if cw_id.startswith('arn:'):
            continue  # a deployment ARN is recorded as it is
        model_id, prefix = split_profile_id(cw_id)
        if '.' not in cw_id and ':' not in cw_id:
            # An application inference profile ID
            if account:
                forms[build_arn('bedrock', region, account, f"application-inference-profile/{cw_id}")] = cw_id
        elif prefix:
            if account:
                forms[build_arn('bedrock', region, account, f"inference-profile/{endpoint_id(model_id, prefix)}")] = cw_id
        else:
            forms[build_arn('bedrock', region, '', f"foundation-model/{model_id}")] = cw_id
    return {form: cw_id for form, cw_id in forms.items() if _MODEL_ID_PATTERN.match(form)}


def build_query(forms: Iterable[str], breakdown: Breakdown) -> str:
    """Logs Insights query: per-minute input/output tokens and requests by modelId,
    principal and (for session and metadata breakdowns) the breakdown key."""
    values = sorted(set(forms))
    for value in values:
        if not _MODEL_ID_PATTERN.match(value):
            raise ValueError(f"unsafe modelId for a query: {value!r}")
    in_list = ', '.join(f'"{value}"' for value in values)
    keys = ['principal']
    if breakdown.kind == SESSION:
        keys.append('identity.arn as session')
    elif breakdown.kind == METADATA:
        # Backticks: keys may hold characters (- : / = + @) that are not field-name characters
        keys.append(f'`requestMetadata.{breakdown.key}` as meta')
    return _QUERY_HEAD + '\n'.join([
        f'| filter modelId in [{in_list}]',
        # Failed calls are logged too (with an errorCode); CloudWatch's Invocations counts
        # only successful ones
        '| filter not ispresent(errorCode)',
        r'| parse identity.arn /:(?<p_role>assumed-role\/[^\/]+)\// ',
        r'| parse identity.arn /:(?<p_user>user\/.+)$/',
        '| fields coalesce(p_role, p_user, identity.arn) as principal',
        '| stats sum(input.inputTokenCount) as i, sum(output.outputTokenCount) as o, count(*) as n'
        f' by bin(1m) as minute, modelId, {", ".join(keys)}',
        f'| limit {MAX_ROWS}',
    ])


class LogsQueryError(RuntimeError):
    """A Logs Insights query that failed, was cancelled or timed out."""


def _parse_minute(value: str) -> datetime:
    """Logs Insights bin(1m) value ('2026-10-06 10:26:00.000', UTC) as an aware datetime."""
    try:
        return datetime.strptime(value[:19], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    except ValueError:
        raise LogsQueryError(f"unexpected minute value in the query results: {value!r}") from None


def query_batches(forms: Iterable[str], breakdown: Breakdown) -> List[str]:
    """The breakdown queries for these modelId spellings: one, or several when one query
    string would pass the StartQuery limit of MAX_QUERY_LENGTH characters."""
    queries: List[str] = []
    batch: List[str] = []
    for value in sorted(set(forms)):
        if batch and len(build_query(batch + [value], breakdown)) > MAX_QUERY_LENGTH - QUERY_END_FILTER_LENGTH:
            queries.append(build_query(batch, breakdown))
            batch = []
        batch.append(value)
    if batch:
        queries.append(build_query(batch, breakdown))
    return queries


def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class InvocationLogFetcher:
    """Runs the breakdown query over a time window, split into day-long queries that run a
    few at a time; a window that returns the row limit is split in half and run again."""

    def __init__(self, logs_client, log_group: str, sleep: Callable[[float], None] = time.sleep,
                 poll_seconds: float = 1.0, max_concurrent: int = MAX_CONCURRENT_QUERIES):
        self.logs_client = logs_client
        self.log_group = log_group
        self._sleep = sleep
        self._poll = poll_seconds
        self._max_concurrent = max_concurrent
        self.bytes_scanned = 0.0
        self.queries_run = 0
        self._cancel = threading.Event()

    def coverage_start(self, start: datetime, end: datetime) -> Optional[datetime]:
        """The earliest time of [start, end] the log group can hold (its creation time and
        retention), or None when it does not exist."""
        response = self.logs_client.describe_log_groups(logGroupNamePrefix=self.log_group)
        group = next((g for g in response.get('logGroups') or [] if g.get('logGroupName') == self.log_group), None)
        if group is None:
            return None
        earliest = start
        created = group.get('creationTime')
        if created:
            earliest = max(earliest, datetime.fromtimestamp(created / 1000, tz=timezone.utc))
        retention = group.get('retentionInDays')
        if retention:
            earliest = max(earliest, end - timedelta(days=retention))
        # On a minute: window boundaries then fall on whole seconds (startTime is in seconds)
        return min(earliest.replace(second=0, microsecond=0), end)

    def fetch(self, forms: Dict[str, str], breakdown: Breakdown, start: datetime, end: datetime) -> List[Dict]:
        """Rows of the window: {'minute', 'cw_id', 'principal', 'key', 'input', 'output',
        'requests'}, 'key' being the session ARN or metadata value (None otherwise)."""
        if not forms:
            return []
        windows = []
        cursor = start
        while cursor < end:
            windows.append((cursor, min(cursor + timedelta(days=1), end)))
            cursor = windows[-1][1]
        # Each modelId spelling is in exactly one query, so the results simply add up
        jobs = [(query, w_start, w_end) for query in query_batches(forms, breakdown) for w_start, w_end in windows]
        rows: List[Dict] = []
        self._cancel.clear()
        pool = ThreadPoolExecutor(max_workers=max(1, min(self._max_concurrent, len(jobs))))
        try:
            # In completion order, so that the first failure cancels the rest at once
            for future in as_completed([pool.submit(self._run_window, *job) for job in jobs]):
                rows.extend(future.result())
        except BaseException:
            # A failed query, or Ctrl-C: the queries still running stop at their next poll
            # (and are stopped in the account) instead of running to completion
            self._cancel.set()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        return [self._row(raw, forms, breakdown) for raw in rows]

    def _run_window(self, query: str, start: datetime, end: datetime) -> List[Dict]:
        if self._cancel.is_set():
            raise LogsQueryError("cancelled")
        results = self._run_query(query, start, end)
        if len(results) < MAX_ROWS or end - start <= MIN_WINDOW:
            if len(results) >= MAX_ROWS:
                logger.info(f"  Warning: the invocation-log query for {start:%Y-%m-%d %H:%M} returned the "
                            f"{MAX_ROWS}-row limit; some callers of that window may be missing")
            return results
        middle = start + (end - start) / 2
        middle = middle.replace(second=0, microsecond=0)
        return self._run_window(query, start, middle) + self._run_window(query, middle, end)

    def _run_query(self, query: str, start: datetime, end: datetime) -> List[Dict]:
        # startTime and endTime are whole seconds and both inclusive: the window's end is
        # cut at the millisecond in the query, so a record in the last second before a
        # window boundary is in exactly one window
        end_ms = int(end.timestamp() * 1000)
        query = query.replace(_QUERY_HEAD, f"{_QUERY_HEAD}| filter @timestamp < {end_ms}\n", 1)
        response = self.logs_client.start_query(
            logGroupName=self.log_group, queryString=query,
            startTime=int(start.timestamp()), endTime=int(end.timestamp()), limit=MAX_ROWS)
        query_id = response.get('queryId')
        if not query_id:
            raise LogsQueryError("StartQuery returned no query ID")
        self.queries_run += 1
        waited = 0.0
        try:
            while True:
                if self._cancel.is_set():
                    raise LogsQueryError("cancelled")
                result = self.logs_client.get_query_results(queryId=query_id)
                status = result.get('status')
                if status == 'Complete':
                    self.bytes_scanned += _number((result.get('statistics') or {}).get('bytesScanned'))
                    return [{field['field']: field.get('value') for field in row}
                            for row in result.get('results') or []]
                if status in ('Failed', 'Cancelled', 'Timeout', 'Unknown'):
                    raise LogsQueryError(f"the Logs Insights query {status.lower()}")
                if waited >= QUERY_TIMEOUT_SECONDS:
                    raise LogsQueryError(f"the Logs Insights query did not finish in {QUERY_TIMEOUT_SECONDS} s")
                self._sleep(self._poll)
                waited += self._poll
        except BaseException:
            # Interrupted or failed: do not leave it running (and counting) in the account
            try:
                self.logs_client.stop_query(queryId=query_id)
            except Exception as e:  # already finished, or no logs:StopQuery permission
                logger.debug(f"Could not stop query {query_id}: {e}")
            raise

    @staticmethod
    def _row(raw: Dict, forms: Dict[str, str], breakdown: Breakdown) -> Dict:
        principal = _from_query_principal(raw.get('principal') or '')
        if breakdown.kind == SESSION:
            key = raw.get('session') or principal
        elif breakdown.kind == METADATA:
            key = raw.get('meta')
        else:
            key = None
        return {'minute': _parse_minute(raw.get('minute') or ''),
                'cw_id': forms.get(raw.get('modelId') or '', raw.get('modelId')),
                'principal': principal, 'key': key,
                'input': _number(raw.get('i')), 'output': _number(raw.get('o')),
                'requests': _number(raw.get('n'))}


def principal_tags(iam_client, principals: Iterable[str],
                   parallel: Callable = None) -> Tuple[Dict[str, Dict[str, str]], Optional[Exception]]:
    """IAM tags of 'role/<name>' and 'user/<name>' principals.

    Returns ({principal: tags}, the first error), principals whose tags could not be read
    being left out. Other principals (root, federated users) have no tags to read.
    """
    def read(principal):
        kind, _, name = principal.partition('/')
        name = name.rsplit('/', 1)[-1]  # a user's path is not part of its name
        try:
            if kind == 'role':
                response = iam_client.list_role_tags(RoleName=name)
            else:
                response = iam_client.list_user_tags(UserName=name)
            return principal, {t['Key']: t['Value'] for t in response.get('Tags') or []}, None
        except AWS_ERRORS as e:
            return principal, None, e

    names = [p for p in dict.fromkeys(principals) if p.startswith(('role/', 'user/'))]
    results = parallel(read, names) if parallel else [read(p) for p in names]
    tags: Dict[str, Dict[str, str]] = {}
    error = None
    for principal, value, e in results:
        if value is not None:
            tags[principal] = value
        elif error is None or (is_access_denied(e) and not is_access_denied(error)):
            error = e
    return tags, error
