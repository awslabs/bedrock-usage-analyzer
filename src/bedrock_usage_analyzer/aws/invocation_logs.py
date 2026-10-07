# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Usage per caller from Amazon Bedrock model invocation logs (CloudWatch Logs Insights).

CloudWatch's AWS/Bedrock metrics have one dimension, ModelId, so every caller of one
endpoint shares one series. The model invocation log records each request's caller
(identity.arn), its model or profile ID and its token counts, so a Logs Insights query
can split an endpoint's per-minute tokens and requests by IAM principal, by session, or
by a requestMetadata key. Grouping by an IAM principal tag is done here from the
principals' tags (iam:ListRoleTags / iam:ListUserTags), as the logs carry no tags.

Only metadata fields are returned: a record's prompt and completion bodies are scanned
server-side (and billed) by the query, which reads the token counts and caller from the
raw record when they come after large bodies, but they are never returned or stored.
Metadata-only delivery keeps the bodies out of the logs altogether.
"""

import logging
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from botocore.exceptions import ClientError, ConnectTimeoutError, EndpointConnectionError

from bedrock_usage_analyzer.aws.bedrock import arn_resource, split_profile_id
from bedrock_usage_analyzer.core.errors import AWS_ERRORS, is_access_denied
from bedrock_usage_analyzer.utils.partition import build_arn

logger = logging.getLogger(__name__)


PRINCIPAL = 'principal'
SESSION = 'session'
TAG = 'tag'
METADATA = 'metadata'
KINDS = (PRINCIPAL, SESSION, TAG, METADATA)

# Logs Insights returns at most this many rows per query, in GetQueryResults pages of
# PAGE_ROWS; with an SDK that cannot ask for the next page or for that limit, one page
MAX_ROWS = 100000
PAGE_ROWS = 10000
# Queries run at once (the account allows 100 across all users, and 10 StartQuery per second)
MAX_CONCURRENT_QUERIES = 4
QUERY_TIMEOUT_SECONDS = 900
MAX_POLL_SECONDS = 10  # polling slows from poll_seconds to this as a query runs
# A query window is never split below this, even if it returns MAX_ROWS rows
MIN_WINDOW = timedelta(minutes=10)
# A window that returns MAX_ROWS rows is run again as this many parts
SPLIT_PARTS = 4
# StartQuery refusals that started no query, retried with backoff (1, 2, 4, 8, 16, 32 s):
# about a minute, so other users' queries holding the account's concurrency can finish
START_RETRY_CODES = ('ThrottlingException', 'LimitExceededException', 'TooManyRequestsException')
START_ATTEMPTS = 7
# StartQuery accepts query strings of up to 10,000 characters; each window adds its end
# filter (QUERY_END_FILTER_LENGTH characters at most) to the query
MAX_QUERY_LENGTH = 10000
QUERY_END_FILTER_LENGTH = 40
_QUERY_HEAD = 'fields @timestamp\n'

# requestMetadata keys (Bedrock allows letters, digits, whitespace and :_@$#=/+,-.) go into
# the query inside backticks, which none of these characters can close; IAM tag keys are
# only looked up in ListRoleTags/ListUserTags results
_KEY_PATTERN = re.compile(r'^[A-Za-z0-9 :_@$#=/+,.-]{1,256}\Z')
_TAG_KEY_PATTERN = re.compile(r'^[\w .:/=+@-]{1,128}\Z')  # \w: letters and digits of any language
# modelId values in a query string literal: model, profile and deployment IDs and ARNs
_MODEL_ID_PATTERN = re.compile(r'^[A-Za-z0-9_.:/-]{1,2048}\Z')
# An IAM principal given with --principal: 'role/<name>', 'user/<name>' or an ARN
_PRINCIPAL_PATTERN = re.compile(r'^[A-Za-z0-9_.:/=+,@-]{1,2048}\Z')

UNATTRIBUTED = '(not in the invocation logs)'
# A record whose caller could not be read (a principal is an ARN or 'role/'/'user/', never '(...)')
UNKNOWN_CALLER = '(caller not recorded)'


def has_tags(principal: str) -> bool:
    """Whether IAM tags can be read for a principal: roles and users (not root, federated
    users or a caller not recorded)."""
    return principal.startswith(('role/', 'user/'))


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
        kind, sep, key = (value or '').strip().partition(':')
        kind, key = kind.strip().lower(), key.strip()  # spaces around the separator
        if kind not in KINDS:
            raise BreakdownError(f"unknown breakdown '{value}': use principal, session, tag:<key> or metadata:<key>")
        if kind == TAG:
            if not _TAG_KEY_PATTERN.match(key):
                raise BreakdownError(f"'{value}' needs a key of letters (any language), digits, spaces and _.:/=+@- (e.g. tag:team)")
        elif kind == METADATA:
            if not _KEY_PATTERN.match(key):
                raise BreakdownError(f"'{value}' needs a key of letters, digits, spaces and :_@$#=/+,-. "
                                     f"(e.g. metadata:team)")
        elif sep:  # 'session:' too, as 'tag:' with no key is an error
            raise BreakdownError(f"'{kind}' takes no key: '{value}'")
        principals = tuple(principals)
        if any(not (p or '').strip() for p in principals):
            # An empty --principal "$SVC" would otherwise read every caller's usage
            raise BreakdownError("empty principal: use role/<name>, user/<name> or an IAM ARN")
        names = tuple(normalize_principal(p.strip()) for p in principals)
        for name in names:
            if not _PRINCIPAL_PATTERN.match(name) or not name.startswith(('role/', 'user/', 'arn:')) \
                    or name in ('role/', 'user/') or (name.startswith('arn:') and not _arn_names_principal(name)):
                raise BreakdownError(f"invalid principal '{name}': use role/<name>, user/<name> or an IAM ARN")
        if log_group is not None and not re.match(r'^[A-Za-z0-9_./#-]{1,512}\Z', log_group):
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
        resource = arn_resource(value)
    if resource.startswith('assumed-role/'):
        return 'role/' + resource.split('/')[1]
    if resource.startswith(('role/', 'user/')):
        kind, _, rest = resource.partition('/')
        return f"{kind}/{rest.rsplit('/', 1)[-1]}"
    return value


def _arn_names_principal(arn: str) -> bool:
    """Whether an ARN left as it is by normalize_principal (roles and users are not) names a
    caller of the logs: the root user or a federated user. Not 'arn:', a role without a
    name, or an IAM group, a policy or a model, which never call."""
    service = arn.split(':')[2] if arn.count(':') >= 5 else ''
    resource = arn_resource(arn)
    return (service in ('iam', 'sts')
            and (resource == 'root' or bool(re.match(r'^federated-user/[^/]+\Z', resource))))


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


def model_id_forms(cw_ids: Iterable[str], region: str, account: Optional[str],
                   known_models: Iterable[str] = ()) -> Dict[str, str]:
    """Every spelling of each CloudWatch ModelId value that invocation logs may record.

    The log's modelId is what the caller passed: a model ID or its ARN, an inference profile
    ID or ARN, an application profile ARN, or a deployment ARN. CloudWatch reports all of them
    under one ModelId value (the profile or model ID, the application profile ID, the
    deployment ARN). known_models are the region's foundation model IDs. Returns {log
    spelling: CloudWatch ModelId value}.
    """
    known_models = set(known_models)
    forms: Dict[str, str] = {}
    for cw_id in cw_ids:
        forms[cw_id] = cw_id
        if cw_id.startswith('arn:'):
            continue  # a deployment or imported model ARN is recorded as it is
        model_id, prefix = split_profile_id(cw_id)
        if '.' not in cw_id and ':' not in cw_id:
            # An application inference profile ID
            if account:
                forms[build_arn('bedrock', region, account, f"application-inference-profile/{cw_id}")] = cw_id
        else:
            # A profile ID, or a model ID. A profile whose prefix the bundled mapping does not
            # have yet ('kr.anthropic.x') looks like a model ID: an ID that is not one of the
            # region's foundation models gets both ARN spellings
            if account and (prefix or cw_id not in known_models):
                forms[build_arn('bedrock', region, account, f"inference-profile/{cw_id}")] = cw_id
            if not prefix:
                forms[build_arn('bedrock', region, '', f"foundation-model/{model_id}")] = cw_id
    return {form: cw_id for form, cw_id in forms.items() if _MODEL_ID_PATTERN.match(form)}


def build_query(forms: Iterable[str], breakdown: Breakdown, limit: Optional[int] = None) -> str:
    """Logs Insights query: per-minute input/output tokens and requests by modelId,
    principal and (for session and metadata breakdowns) the breakdown key."""
    values = sorted(set(forms))
    for value in values:
        if not _MODEL_ID_PATTERN.match(value):
            raise ValueError(f"unsafe modelId for a query: {value!r}")
    in_list = ', '.join(f'"{value}"' for value in values)
    keys = ['principal']
    if breakdown.kind == SESSION:
        keys.append('arn as session')
    elif breakdown.kind == METADATA:
        # Backticks: keys may hold characters (- : / = + @) that are not field-name characters
        keys.append(f'`requestMetadata.{breakdown.key}` as meta')
    return _QUERY_HEAD + '\n'.join([
        f'| filter modelId in [{in_list}]',
        # Failed calls are logged too (with an errorCode); CloudWatch's Invocations counts
        # only successful ones. A failed call's record has no bodies, so its few fields are
        # always discovered
        '| filter not ispresent(errorCode)',
        # The token counts and the caller come after the request and response bodies, so in a
        # record with large bodies they can be past the first 200 fields Logs Insights
        # discovers. Then they are read from the raw record, anchored on the structure Bedrock
        # writes around them (bodies are embedded JSON and may hold keys of the same names,
        # e.g. a streamed response's own metrics): the input count closes the input object
        # before the output object, the output count and the caller end the record
        r'| parse @message /"inputTokenCount":(?<i_tok>\d+)[^{}]*\},"output":\{"outputContentType"/',
        r'| parse @message /"outputTokenCount":(?<o_tok>\d+)[^{}]*\},"identity":\{"arn":"[^"]+"\}[^{}]*\}\s*$/',
        r'| parse @message /"identity":\{"arn":"(?<caller>[^"]+)"\}[^{}]*\}\s*$/',
        '| fields coalesce(input.inputTokenCount, i_tok) as in_tokens,'
        ' coalesce(output.outputTokenCount, o_tok) as out_tokens, coalesce(identity.arn, caller) as arn',
        r'| parse arn /:(?<p_role>assumed-role\/[^\/]+)\// ',
        r'| parse arn /:(?<p_user>user\/.+)$/',
        '| fields coalesce(p_role, p_user, arn) as principal',
        '| stats sum(in_tokens) as i, sum(out_tokens) as o, count(*) as n'
        f' by bin(1m) as minute, modelId, {", ".join(keys)}',
        f'| limit {limit or MAX_ROWS}',
    ])


class LogsQueryError(RuntimeError):
    """A Logs Insights query that failed, was cancelled or timed out."""


class QueryTimeoutError(LogsQueryError):
    """A query that ran too long (here or in Logs Insights): a shorter window may finish."""


def _parse_minute(value: str) -> datetime:
    """Logs Insights bin(1m) value ('2026-10-06 10:26:00.000', UTC) as an aware datetime."""
    try:
        return datetime.strptime(value[:19], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    except ValueError:
        raise LogsQueryError(f"unexpected minute value in the query results: {value!r}") from None


def query_batches(forms: Iterable[str], breakdown: Breakdown) -> List[str]:
    """The breakdown queries for these modelId spellings: one, or several when one query
    string would pass the StartQuery limit of MAX_QUERY_LENGTH characters."""
    return [build_query(batch, breakdown) for batch in _value_batches(forms, breakdown)]


def _value_batches(forms: Iterable[str], breakdown: Breakdown) -> List[List[str]]:
    """The modelId spellings of each breakdown query (see query_batches)."""
    values = sorted(set(forms))
    if not values:
        return []
    # A query is its fixed text plus '"value", ' per spelling: count instead of rebuilding
    fixed = len(build_query(values[:1], breakdown)) - len(values[0]) - 2
    budget = MAX_QUERY_LENGTH - QUERY_END_FILTER_LENGTH
    batches: List[List[str]] = []
    batch: List[str] = []
    length = fixed
    for value in values:
        added = len(value) + 2 + (2 if batch else 0)
        if batch and length + added > budget:
            batches.append(batch)
            batch, length, added = [], fixed, len(value) + 2
        batch.append(value)
        length += added
    batches.append(batch)
    return batches


def next_minute(moment: datetime) -> datetime:
    """The first whole minute at or after moment."""
    floor = moment.replace(second=0, microsecond=0)
    return floor if floor == moment else floor + timedelta(minutes=1)


def _row_limit(logs_client) -> int:
    """Rows a query may return with this SDK: MAX_ROWS when it can ask GetQueryResults for
    the next page and its StartQuery accepts that limit (both came in different releases),
    else one page."""
    page = min(MAX_ROWS, PAGE_ROWS)
    try:
        model = logs_client.meta.service_model
        if 'nextToken' not in model.operation_model('GetQueryResults').input_shape.members:
            return page
        limit = model.operation_model('StartQuery').input_shape.members['limit']
        return max(page, min(MAX_ROWS, int(limit.metadata.get('max', page))))
    except Exception:  # not a botocore client (a test double), or no such operation
        return page


def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class InvocationLogFetcher:
    """Runs the breakdown query over a time window, split into day-long queries that run a
    few at a time; a window that returns the row limit is run again in SPLIT_PARTS parts."""

    def __init__(self, logs_client, log_group: str, sleep: Optional[Callable[[float], object]] = None,
                 poll_seconds: float = 1.0, max_concurrent: int = MAX_CONCURRENT_QUERIES,
                 start_client=None):
        self.logs_client = logs_client
        # StartQuery is not idempotent: a client that never retries it (start_client), so a
        # lost reply cannot leave a second, unseen query scanning; throttling is retried here
        self.start_client = start_client or logs_client
        self.log_group = log_group
        # Rows a query may return: a window at this many is split
        self.max_rows = _row_limit(logs_client)
        self._poll = poll_seconds
        self._max_concurrent = max_concurrent
        self.bytes_scanned = 0.0
        self.queries_run = 0
        # Windows cut at the row limit: (start, end, the report ModelIds of their query)
        self.truncated: List[Tuple[datetime, datetime, frozenset]] = []
        self._batch_ids: Dict[str, frozenset] = {}
        self._cancel = threading.Event()
        self._count_lock = threading.Lock()
        # Waits (backoff, polling) end at once when the run is cancelled
        self._sleep = sleep or self._cancel.wait

    def coverage_start(self, start: datetime, end: datetime,
                       now: datetime) -> Optional[datetime]:
        """The earliest time of [start, end] the log group can hold (its creation time and
        retention, counted back from now: records expire by the clock, not by the window's
        end), or None when it does not exist."""
        group = self._log_group()
        if group is None:
            return None
        # On a minute: window boundaries then fall on whole seconds (startTime is in seconds)
        earliest = start.replace(second=0, microsecond=0)
        created = group.get('creationTime')
        if created:
            created_at = datetime.fromtimestamp(created / 1000, tz=timezone.utc)
            earliest = max(earliest, created_at.replace(second=0, microsecond=0))
        retention = group.get('retentionInDays')
        if retention:
            # Up to the next minute: the minute retention cuts into may be partly deleted
            kept_from = now - timedelta(days=retention)
            earliest = max(earliest, next_minute(kept_from))
        return min(earliest, end)

    def _log_group(self) -> Optional[Dict]:
        """The log group's description. The name is a prefix filter, so other groups whose
        names start with it may fill the first pages: read on until the exact name."""
        pages = self.logs_client.get_paginator('describe_log_groups').paginate(logGroupNamePrefix=self.log_group)
        try:
            for page in pages:
                for group in page.get('logGroups') or []:
                    if group.get('logGroupName') == self.log_group:
                        return group
        except ClientError as e:
            if not is_access_denied(e):
                raise
            # It only narrows the start (creation, retention): queries may still be allowed
            logger.info(f"  Note: {self.log_group} could not be described ({e}); reading from the "
                        f"requested start")
            return {}
        return None

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
        # Each query and the report ModelIds its spellings belong to
        self._batch_ids = {build_query(values, breakdown, self.max_rows): frozenset(forms[v] for v in values)
                           for values in _value_batches(forms, breakdown)}
        batches = list(self._batch_ids)
        jobs = [(query, w_start, w_end, False) for query in batches for w_start, w_end in windows]
        rows: List[Dict] = []
        self._cancel.clear()
        pool = ThreadPoolExecutor(max_workers=self._max_concurrent)
        try:
            # In completion order, so that the first failure cancels the rest at once; the
            # parts of a split window go back into the pool and run beside the other windows
            pending = {pool.submit(self._run_window, *job) for job in jobs}
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    window_rows, parts = future.result()
                    rows.extend(window_rows)
                    pending |= {pool.submit(self._run_window, *part) for part in parts}
        except BaseException:
            # A failed query, or Ctrl-C: the queries still running stop at their next poll
            # (and are stopped in the account) instead of running to completion
            self._cancel.set()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        return [self._row(raw, forms, breakdown) for raw in rows]

    def _run_window(self, query: str, start: datetime, end: datetime,
                    after_timeout: bool) -> Tuple[List[Dict], List[Tuple]]:
        """(rows, []) for a window that fits, or ([], its parts) for one to split: at the
        row limit, or after a timeout (once: a part that times out again ends the
        breakdown, so a stuck log group costs one more timeout, not one per split level)."""
        if self._cancel.is_set():
            raise LogsQueryError("cancelled")
        try:
            results = self._run_query(query, start, end)
        except QueryTimeoutError:
            if after_timeout or end - start <= MIN_WINDOW:
                raise
            # Too much to scan in one query (a busy log group with bodies): smaller windows
            logger.debug(f"Invocation-log query for {start:%Y-%m-%d %H:%M} timed out; splitting it")
            return [], self._parts(query, start, end, True)
        if len(results) < self.max_rows or end - start <= MIN_WINDOW:
            if len(results) >= self.max_rows:
                # Summed up once by the caller and in the reports it affects, not per window
                logger.debug(f"Invocation-log query for {start:%Y-%m-%d %H:%M} returned the "
                             f"{self.max_rows}-row limit")
                with self._count_lock:
                    self.truncated.append((start, end, self._batch_ids.get(query, frozenset())))
            return results, []
        return [], self._parts(query, start, end, after_timeout)

    @staticmethod
    def _parts(query: str, start: datetime, end: datetime, after_timeout: bool) -> List[Tuple]:
        # Four parts, not two: each split scans the window's bytes again, and a busy window
        # (many sessions per minute) then reaches a size that fits in fewer rounds
        step = (end - start) / SPLIT_PARTS
        edges = [start] + [(start + step * i).replace(second=0, microsecond=0) for i in range(1, SPLIT_PARTS)] + [end]
        edges = sorted(set(edges))
        return [(query, a, b, after_timeout) for a, b in zip(edges, edges[1:])]

    def _run_query(self, query: str, start: datetime, end: datetime) -> List[Dict]:
        # startTime and endTime are whole seconds and both inclusive: the window's end is
        # cut at the millisecond in the query, so a record in the last second before a
        # window boundary is in exactly one window
        end_ms = int(end.timestamp() * 1000)
        query = f"filter @timestamp < {end_ms}\n| {query}"  # its own first stage
        for attempt in range(START_ATTEMPTS):
            if self._cancel.is_set():  # another window failed while this one was backing off
                raise LogsQueryError("cancelled")
            try:
                response = self.start_client.start_query(
                    logGroupName=self.log_group, queryString=query,
                    startTime=int(start.timestamp()), endTime=int(end.timestamp()), limit=self.max_rows)
                break
            except (ClientError, EndpointConnectionError, ConnectTimeoutError) as e:
                # Refused (throttled, or too many queries at once), or no connection was made:
                # no query was started. Any other failure may have started one, so no retry
                code = e.response.get('Error', {}).get('Code') if isinstance(e, ClientError) else None
                retry = code in START_RETRY_CODES or not isinstance(e, ClientError)
                if not retry or attempt == START_ATTEMPTS - 1:
                    raise
                self._sleep(2 ** attempt)
        query_id = response.get('queryId')
        if not query_id:
            raise LogsQueryError("StartQuery returned no query ID")
        with self._count_lock:  # queries run in several threads
            self.queries_run += 1
        waited, polls, started = 0.0, 0, time.monotonic()
        scanned = 0.0  # so far, from the latest poll: a stopped query's scan is billed too
        try:
            while True:
                if self._cancel.is_set():
                    raise LogsQueryError("cancelled")
                result = self.logs_client.get_query_results(queryId=query_id)
                status = result.get('status')
                scanned = _number((result.get('statistics') or {}).get('bytesScanned')) or scanned
                if status == 'Complete':
                    results = list(result.get('results') or [])
                    token = result.get('nextToken')
                    while token:  # rows past the first page
                        if self._cancel.is_set():
                            raise LogsQueryError("cancelled")
                        page = self.logs_client.get_query_results(queryId=query_id, nextToken=token)
                        results.extend(page.get('results') or [])
                        token = page.get('nextToken')
                    rows = [{field['field']: field.get('value') for field in row} for row in results]
                    with self._count_lock:  # after the rows: a bad row counts it once, below
                        self.bytes_scanned += scanned
                    return rows
                if status == 'Timeout':
                    raise QueryTimeoutError("the Logs Insights query timed out")
                if status in ('Failed', 'Cancelled', 'Unknown'):
                    raise LogsQueryError(f"the Logs Insights query {status.lower()}")
                # Wall time too: slow (retried) polls count, not only the waits between them
                if max(waited, time.monotonic() - started) >= QUERY_TIMEOUT_SECONDS:
                    raise QueryTimeoutError(f"the Logs Insights query did not finish in {QUERY_TIMEOUT_SECONDS} s")
                # Each poll returns the partial results so far: poll less often as a query
                # runs longer, so a long query's rows are not fetched again every second
                delay = min(self._poll * 1.5 ** polls, MAX_POLL_SECONDS)
                polls += 1
                self._sleep(delay)
                waited += delay
        except BaseException:
            with self._count_lock:  # what it scanned until now is in the bill and the cost line
                self.bytes_scanned += scanned
            # Interrupted or failed: do not leave it running (and counting) in the account
            try:
                self.logs_client.stop_query(queryId=query_id)
            except Exception as e:  # already finished, or no logs:StopQuery permission
                logger.debug(f"Could not stop query {query_id}: {e}")
            raise

    @staticmethod
    def _row(raw: Dict, forms: Dict[str, str], breakdown: Breakdown) -> Dict:
        # 'assumed-role/<name>', 'user/...' or an ARN; a record without a readable caller gets a named row
        principal = normalize_principal(raw.get('principal') or '') or UNKNOWN_CALLER
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
                   parallel: Callable = None) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Exception]]:
    """IAM tags of 'role/<name>' and 'user/<name>' principals.

    Returns ({principal: tags}, {principal: error}) for the principals whose tags were read
    and those whose tags could not be. Other principals (root, federated users) have no
    tags to read.
    """
    def read(principal):
        kind, _, name = principal.partition('/')  # normalized: no IAM path
        try:
            if kind == 'role':
                response = iam_client.list_role_tags(RoleName=name)
            else:
                response = iam_client.list_user_tags(UserName=name)
            return principal, {t['Key']: t['Value'] for t in response.get('Tags') or []}, None
        except AWS_ERRORS as e:
            return principal, None, e

    names = [p for p in dict.fromkeys(principals) if has_tags(p)]
    results = parallel(read, names) if parallel else [read(p) for p in names]
    tags: Dict[str, Dict[str, str]] = {}
    errors: Dict[str, Exception] = {}
    for principal, value, e in results:
        if value is not None:
            tags[principal] = value
        else:
            errors[principal] = e
    return tags, errors


def main_error(errors: Iterable[Exception]) -> Optional[Exception]:
    """The error to quote for several failed reads: a denial (the fix is a permission)
    before others such as a deleted principal."""
    errors = list(errors)
    return next((e for e in errors if is_access_denied(e)), errors[0] if errors else None)
