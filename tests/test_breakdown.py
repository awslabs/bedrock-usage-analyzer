# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Usage breakdown by caller from the model invocation logs: query building, principal
normalization, chunked Logs Insights queries, grouping, remainder rows and the report."""

import json
import os
import re
from datetime import datetime, timedelta, timezone

import pytest
from botocore.exceptions import ClientError

from bedrock_usage_analyzer.aws import invocation_logs as il
from bedrock_usage_analyzer.aws.invocation_logs import (
    UNATTRIBUTED, Breakdown, BreakdownError, InvocationLogFetcher, LogsQueryError, build_query,
    logging_destination, model_id_forms, normalize_principal, principal_tags)
from bedrock_usage_analyzer.core.breakdown import BreakdownBuilder
from bedrock_usage_analyzer.core.metrics_fetcher import CloudWatchMetricsFetcher

from conftest import HAIKU, FakeBedrock

REGION = 'us-east-1'
ACCOUNT = '111122223333'
US_HAIKU = f"us.{HAIKU}"
END = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)
ROLE_ARN = f"arn:aws:sts::{ACCOUNT}:assumed-role/OrdersService/session-1"
ROLE2_ARN = f"arn:aws:sts::{ACCOUNT}:assumed-role/OrdersService/session-2"
USER_ARN = f"arn:aws:iam::{ACCOUNT}:user/ops/alice"
GRANULARITY = {'1hour': 60, '1day': 300, '7days': 3600, '14days': 3600, '30days': 3600}
PERIODS = ['1hour', '1day', '7days', '14days', '30days']


def aws_error(code, operation='StartQuery'):
    return ClientError({'Error': {'Code': code, 'Message': code}}, operation)


def minute(dt):
    return dt.strftime('%Y-%m-%d %H:%M:%S.000')


class FakeLogs:
    """Logs Insights: returns scripted per-minute rows inside each query's time window."""

    def __init__(self, rows=(), statuses=('Running', 'Complete'), groups=None, fail=None):
        self.rows = list(rows)  # dicts with minute (datetime), modelId, principal, i, o, n, extra fields
        self.statuses = list(statuses)
        self.groups = groups if groups is not None else [{'logGroupName': '/bedrock/logs', 'creationTime': 0}]
        self.fail = fail
        self.queries = []
        self.stopped = []
        self._polls = {}

    def describe_log_groups(self, logGroupNamePrefix):
        return {'logGroups': [g for g in self.groups if g['logGroupName'].startswith(logGroupNamePrefix)]}

    def start_query(self, logGroupName, queryString, startTime, endTime, limit):
        if self.fail:
            raise self.fail
        query_id = f"q{len(self.queries)}"
        self.queries.append({'id': query_id, 'group': logGroupName, 'query': queryString,
                             'start': startTime, 'end': endTime, 'limit': limit})
        self._polls[query_id] = 0
        return {'queryId': query_id}

    def get_query_results(self, queryId):
        query = next(q for q in self.queries if q['id'] == queryId)
        status = self.statuses[min(self._polls[queryId], len(self.statuses) - 1)]
        self._polls[queryId] += 1
        if status != 'Complete':
            return {'status': status}
        # startTime/endTime whole seconds, both inclusive; the query's own end filter in ms
        end_ms = int(re.search(r'@timestamp < (\d+)', query['query']).group(1))
        rows = [r for r in self.rows
                if query['start'] <= r['minute'].timestamp() <= query['end'] and r['minute'].timestamp() * 1000 < end_ms]
        rows = rows[:query['limit']]
        return {'status': 'Complete', 'statistics': {'bytesScanned': 1000.0},
                'results': [[{'field': k, 'value': minute(v) if k == 'minute' else str(v)} for k, v in r.items()]
                            for r in rows]}

    def stop_query(self, queryId):
        self.stopped.append(queryId)
        return {'success': True}


class FakeIam:
    def __init__(self, role_tags=None, user_tags=None, deny=()):
        self.role_tags = role_tags or {}
        self.user_tags = user_tags or {}
        self.deny = set(deny)
        self.calls = []

    def list_role_tags(self, RoleName):
        self.calls.append(('role', RoleName))
        if RoleName in self.deny:
            raise aws_error('AccessDenied', 'ListRoleTags')
        return {'Tags': [{'Key': k, 'Value': v} for k, v in self.role_tags.get(RoleName, {}).items()]}

    def list_user_tags(self, UserName):
        self.calls.append(('user', UserName))
        if UserName in self.deny:
            raise aws_error('AccessDenied', 'ListUserTags')
        return {'Tags': [{'Key': k, 'Value': v} for k, v in self.user_tags.get(UserName, {}).items()]}


def row(at, principal, i=100, o=50, n=1, model=US_HAIKU, **extra):
    return {'minute': at, 'modelId': model, 'principal': principal, 'i': i, 'o': o, 'n': n, **extra}


# ------------------------------------------------------------------ parsing and helpers

@pytest.mark.parametrize('value,kind,key', [
    ('principal', 'principal', None), ('session', 'session', None), ('PRINCIPAL', 'principal', None),
    ('tag:team', 'tag', 'team'), ('metadata:cost-center', 'metadata', 'cost-center'),
    ('tag:Cost Center', 'tag', 'Cost Center'), ('tag: team ', 'tag', 'team'), (' metadata : app', 'metadata', 'app'),
    # IAM tag keys may hold letters of any language
    ('tag:Équipe', 'tag', 'Équipe'), ('tag:部署', 'tag', '部署'),
    # requestMetadata keys may hold what Bedrock allows: spaces, $ # , and up to 256 characters
    ('metadata:cost center', 'metadata', 'cost center'), ('metadata:app#$,x', 'metadata', 'app#$,x'),
    ('metadata:' + 'k' * 256, 'metadata', 'k' * 256),
])
def test_breakdown_parsing(value, kind, key):
    parsed = Breakdown.parse(value)
    assert (parsed.kind, parsed.key) == (kind, key)


@pytest.mark.parametrize('value', ['owner', 'tag', 'tag:', 'tag:a"b', 'tag:   ', 'metadata:x|z', 'metadata:a`b',
                                   'metadata:' + 'k' * 257, 'principal:x', 'session:y'])
def test_invalid_breakdowns_are_refused(value):
    with pytest.raises(BreakdownError):
        Breakdown.parse(value)


def test_a_trailing_newline_is_never_valid():
    # '$' would match before it; the log group would then not be found, and a modelId would
    # pass the query guard
    with pytest.raises(BreakdownError):
        Breakdown.parse('principal', log_group='/aws/bedrock\n')
    with pytest.raises(ValueError):
        build_query([US_HAIKU + '\n'], Breakdown())
    assert model_id_forms([US_HAIKU + '\n'], REGION, ACCOUNT) == {}


def test_principals_and_log_group_are_validated():
    parsed = Breakdown.parse('principal', [ROLE_ARN, ' user/bob '], '/aws/bedrock/logs')
    assert parsed.principals == ('role/OrdersService', 'user/bob') and parsed.log_group == '/aws/bedrock/logs'
    # An empty --principal "$SVC" is an error, not "every caller"
    for empty in ('', '  ', None):
        with pytest.raises(BreakdownError, match='empty principal'):
            Breakdown.parse('principal', [ROLE_ARN, empty])
    with pytest.raises(BreakdownError):
        Breakdown.parse('principal', ['role/x"; drop'])
    with pytest.raises(BreakdownError, match='use role/<name>'):
        Breakdown.parse('principal', ['OrdersService'])  # a bare name would match nothing
    with pytest.raises(BreakdownError):
        Breakdown.parse('principal', ['role/'])  # no name
    # A user's IAM path is dropped, as in the logged principals
    assert Breakdown.parse('principal', ['user/ops/alice']).principals == ('user/alice',)
    with pytest.raises(BreakdownError):
        Breakdown.parse('principal', log_group='bad group name!')


def test_labels():
    assert Breakdown.parse('principal').label == 'IAM principal'
    assert Breakdown.parse('session').label == 'IAM principal session'
    assert Breakdown.parse('tag:team').label == "IAM principal tag 'team'"
    assert Breakdown.parse('metadata:app').label == "request metadata 'app'"


@pytest.mark.parametrize('arn,expected', [
    (ROLE_ARN, 'role/OrdersService'),
    (f"arn:aws-us-gov:sts::{ACCOUNT}:assumed-role/Gov/s", 'role/Gov'),
    (USER_ARN, 'user/alice'),
    # A role's path is not in its assumed-role ARNs, so it is dropped
    (f"arn:aws:iam::{ACCOUNT}:role/aws-reserved/sso.amazonaws.com/AWSReservedSSO_Admin_ab", 'role/AWSReservedSSO_Admin_ab'),
    (f"arn:aws:iam::{ACCOUNT}:role/Direct", 'role/Direct'),
    (f"arn:aws:iam::{ACCOUNT}:root", f"arn:aws:iam::{ACCOUNT}:root"),
    (f"arn:aws:sts::{ACCOUNT}:federated-user/bob", f"arn:aws:sts::{ACCOUNT}:federated-user/bob"),
    ('role/Already', 'role/Already'), ('role/service-role/X', 'role/X'), ('user/ops/bob', 'user/bob'),
    ('assumed-role/A/s', 'role/A'), ('arn:bad', 'arn:bad'), ('', ''),
])
def test_principal_normalization(arn, expected):
    assert normalize_principal(arn) == expected


def test_logging_destination():
    assert logging_destination(FakeBedrock(logging_config={'cloudWatchConfig': {'logGroupName': '/g'}})) == ('/g', '')
    group, reason = logging_destination(FakeBedrock(logging_config={'s3Config': {'bucketName': 'b'}}))
    assert group is None and 'S3 only' in reason
    group, reason = logging_destination(FakeBedrock())
    assert group is None and 'not enabled' in reason


def test_every_model_id_spelling_maps_to_its_cloudwatch_value():
    deployment = f"arn:aws:bedrock:{REGION}:{ACCOUNT}:custom-model-deployment/dep0000001"
    forms = model_id_forms([US_HAIKU, HAIKU, 'app0000001', deployment], REGION, ACCOUNT)
    assert forms[US_HAIKU] == US_HAIKU
    assert forms[f"arn:aws:bedrock:{REGION}:{ACCOUNT}:inference-profile/{US_HAIKU}"] == US_HAIKU
    # Inference profile ARNs always carry the account
    assert f"arn:aws:bedrock:{REGION}::inference-profile/{US_HAIKU}" not in forms
    # A prefix the bundled mapping does not know yet: both ARN spellings
    # A known foundation model gets no inference-profile spelling (no caller can use one)
    known = model_id_forms([HAIKU], REGION, ACCOUNT, known_models=[HAIKU])
    assert set(known) == {HAIKU, f"arn:aws:bedrock:{REGION}::foundation-model/{HAIKU}"}
    unknown = model_id_forms(['kr.anthropic.x-v1:0'], REGION, ACCOUNT, known_models=[HAIKU])
    assert unknown[f"arn:aws:bedrock:{REGION}:{ACCOUNT}:inference-profile/kr.anthropic.x-v1:0"] == 'kr.anthropic.x-v1:0'
    assert unknown[f"arn:aws:bedrock:{REGION}::foundation-model/kr.anthropic.x-v1:0"] == 'kr.anthropic.x-v1:0'
    assert forms[HAIKU] == HAIKU and forms[f"arn:aws:bedrock:{REGION}::foundation-model/{HAIKU}"] == HAIKU
    assert forms[f"arn:aws:bedrock:{REGION}:{ACCOUNT}:application-inference-profile/app0000001"] == 'app0000001'
    assert forms[deployment] == deployment
    # Without the account, only the account-free spellings
    assert not any(ACCOUNT in f for f in model_id_forms([US_HAIKU, 'app0000001'], REGION, None))


def test_unsafe_model_ids_never_reach_the_query():
    assert model_id_forms(['us.x"] | delete'], REGION, ACCOUNT) == {}
    with pytest.raises(ValueError):
        build_query(['a"b'], Breakdown())


def test_query_per_kind():
    principal = build_query([US_HAIKU], Breakdown.parse('principal'))
    assert f'filter modelId in ["{US_HAIKU}"]' in principal and 'by bin(1m) as minute, modelId, principal' in principal
    assert 'inputBodyJson' not in principal and 'outputBodyJson' not in principal  # metadata fields only
    # Failed calls are left out, as CloudWatch's Invocations counts only successful ones
    assert '| filter not ispresent(errorCode)\n' in principal
    assert 'arn as session' in build_query([US_HAIKU], Breakdown.parse('session'))
    assert '`requestMetadata.team` as meta' in build_query([US_HAIKU], Breakdown.parse('metadata:team'))
    # Keys with '-' (or : / = + @) are read as one field name
    assert '`requestMetadata.cost-center` as meta' in build_query([US_HAIKU], Breakdown.parse('metadata:cost-center'))
    assert ', principal\n' in build_query([US_HAIKU], Breakdown.parse('tag:team')) + '\n'


def test_counts_and_caller_are_read_from_the_raw_record_by_bedrocks_own_structure():
    # Past Logs Insights' 200 discovered fields (large bodies), the counts and the caller
    # are read from @message (the discovered fields come first when present). The bodies are
    # embedded JSON and may hold the same keys, e.g. a streamed response's own metrics or
    # keys a caller or a model put there; only Bedrock's own fields match
    query = build_query([US_HAIKU], Breakdown())
    assert 'coalesce(input.inputTokenCount, i_tok)' in query and 'coalesce(identity.arn, caller)' in query
    patterns = {name: re.compile(pattern.replace(f'(?<{name}>', f'(?P<{name}>'))
                for pattern, name in re.findall(r'\| parse @message /(.*\(\?<(\w+)>.*)/\n', query)}
    assert set(patterns) == {'i_tok', 'o_tok', 'caller'}
    spoof = '{"inputTokenCount":999999},"identity":{"arn":"arn:aws:iam::1:user/victim"},"text":"say \\"x\\""'
    streamed = ('[{"chunk":{"amazon-bedrock-invocationMetrics":{"inputTokenCount":7,"outputTokenCount":8}}},'
                '{"tool":{"outputTokenCount":5},"identity":{"arn":"arn:aws:iam::1:user/model"}}]')
    message = ('{"modelId":"m","input":{"inputContentType":"application/json","inputBodyJson":'
               '{"additionalModelRequestFields":' + spoof + '},"inputTokenCount":12,'
               '"cacheReadInputTokenCount":0},"output":{"outputContentType":"application/json",'
               '"outputBodyJson":' + streamed + ',"outputTokenCount":3},'
               '"identity":{"arn":"arn:aws:sts::1:assumed-role/Real/s"},"inferenceRegion":"us-east-1",'
               '"schemaType":"ModelInvocationLog","schemaVersion":"1.0"}')
    found = {name: p.search(message).group(name) for name, p in patterns.items()}
    assert found == {'i_tok': '12', 'o_tok': '3', 'caller': 'arn:aws:sts::1:assumed-role/Real/s'}
    # An embedding: no output count, the input count and the caller are still found
    embedding = ('{"modelId":"e","input":{"inputContentType":"application/json","inputBodyJson":{"inputText":"a"},'
                 '"inputTokenCount":5},"output":{"outputContentType":"application/json"},'
                 '"identity":{"arn":"arn:aws:iam::1:user/u"},"schemaType":"ModelInvocationLog","schemaVersion":"1.0"}')
    assert patterns['i_tok'].search(embedding).group('i_tok') == '5' and not patterns['o_tok'].search(embedding)
    assert patterns['caller'].search(embedding).group('caller') == 'arn:aws:iam::1:user/u'


def test_long_model_id_lists_are_split_across_queries(monkeypatch):
    monkeypatch.setattr(il, 'MAX_QUERY_LENGTH', 1100)
    ids = [f"us.vendor.model-{n:03d}-v1:0" for n in range(40)]
    queries = il.query_batches(ids, Breakdown())
    assert len(queries) > 1 and all(len(q) <= 1100 for q in queries)
    # Every spelling is in exactly one query
    assert sorted(i for i in ids for q in queries if f'"{i}"' in q) == sorted(ids)
    assert il.query_batches([US_HAIKU], Breakdown()) == [build_query([US_HAIKU], Breakdown())]
    assert il.query_batches([], Breakdown()) == []
    # The counted lengths are the real ones: each batch is as full as the budget allows
    for query, nxt in zip(queries, queries[1:]):
        first_next = re.search(r'\["([^"]+)"', nxt).group(1)
        assert len(query) + len(first_next) + 4 > 1100 - il.QUERY_END_FILTER_LENGTH

    logs = FakeLogs([row(END - timedelta(minutes=5), 'assumed-role/A', model=ids[0]),
                     row(END - timedelta(minutes=5), 'assumed-role/A', model=ids[-1])])
    rows = fetcher_for(logs).fetch({i: i for i in ids}, Breakdown(), END - timedelta(hours=1), END)
    assert len(logs.queries) == len(queries) and {r['cw_id'] for r in rows} == {ids[0], ids[-1]}


# ------------------------------------------------------------------ fetcher

def fetcher_for(logs, **kw):
    return InvocationLogFetcher(logs, '/bedrock/logs', sleep=lambda s: None, **kw)


def test_rows_are_fetched_per_day_and_normalized():
    logs = FakeLogs([row(END - timedelta(hours=2), 'assumed-role/OrdersService'),
                     row(END - timedelta(days=2), 'user/ops/alice', i=10, o=None, n=2, model=HAIKU)])
    fetcher = fetcher_for(logs)
    forms = model_id_forms([US_HAIKU, HAIKU], REGION, ACCOUNT)
    rows = fetcher.fetch(forms, Breakdown(), END - timedelta(days=3), END)
    assert len(logs.queries) == 3 and fetcher.queries_run == 3 and fetcher.bytes_scanned == 3000.0
    assert sorted(r['principal'] for r in rows) == ['role/OrdersService', 'user/alice']
    alice = next(r for r in rows if r['principal'] == 'user/alice')
    assert (alice['cw_id'], alice['input'], alice['output'], alice['requests']) == (HAIKU, 10.0, 0.0, 2.0)
    assert alice['minute'].tzinfo is not None
    # Windows meet: each query's end is cut at the millisecond where the next one starts
    assert all(q['end'] - q['start'] == 86400 for q in logs.queries)
    assert all(q['query'].startswith(f"filter @timestamp < {q['end'] * 1000}\n| fields @timestamp\n") for q in logs.queries)


def test_a_record_in_the_last_second_before_a_window_boundary_is_counted_once():
    boundary = END - timedelta(days=1)
    logs = FakeLogs([row(boundary - timedelta(milliseconds=400), 'assumed-role/A'), row(boundary, 'assumed-role/B')])
    rows = fetcher_for(logs).fetch({US_HAIKU: US_HAIKU}, Breakdown(), END - timedelta(days=2), END)
    assert sorted(r['principal'] for r in rows) == ['role/A', 'role/B']


def test_a_full_window_is_split_until_it_fits(monkeypatch):
    monkeypatch.setattr(il, 'MAX_ROWS', 3)
    stamps = [END - timedelta(minutes=m) for m in range(5, 45, 5)]  # 8 rows, 5 minutes apart
    logs = FakeLogs([row(s, 'assumed-role/A') for s in stamps])
    rows = fetcher_for(logs).fetch({US_HAIKU: US_HAIKU}, Breakdown(), END - timedelta(hours=1), END)
    assert len(rows) == 8 and len(logs.queries) > 1


def test_a_full_window_is_run_again_in_quarters():
    stamps = [END - timedelta(minutes=m) for m in range(5, 60, 10)]  # 6 rows, at most 2 per quarter-hour
    logs = FakeLogs([row(s, 'assumed-role/A') for s in stamps])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(il, 'MAX_ROWS', 3)
        rows = fetcher_for(logs).fetch({US_HAIKU: US_HAIKU}, Breakdown(), END - timedelta(hours=1), END)
    # The full hour is run again as four quarter-hours (not halves), which fit
    assert len(rows) == 6 and len(logs.queries) == 1 + il.SPLIT_PARTS
    assert [q['end'] - q['start'] for q in logs.queries[1:]] == [15 * 60] * il.SPLIT_PARTS


def test_a_window_at_the_minimum_keeps_its_rows_and_warns(monkeypatch, caplog):
    caplog.set_level('INFO')
    monkeypatch.setattr(il, 'MAX_ROWS', 2)
    logs = FakeLogs([row(END - timedelta(minutes=m), 'assumed-role/A') for m in range(1, 5)])
    rows = fetcher_for(logs).fetch({US_HAIKU: US_HAIKU}, Breakdown(), END - timedelta(minutes=5), END)
    assert len(rows) == 2 and 'row limit' in caplog.text


@pytest.mark.parametrize('status', ['Failed', 'Cancelled', 'Timeout'])
def test_a_failed_query_raises_and_is_stopped(status):
    logs = FakeLogs(statuses=[status])
    with pytest.raises(LogsQueryError):
        fetcher_for(logs).fetch({US_HAIKU: US_HAIKU}, Breakdown(), END - timedelta(hours=1), END)
    assert logs.stopped == ['q0']


def test_a_query_that_never_finishes_times_out(monkeypatch):
    monkeypatch.setattr(il, 'QUERY_TIMEOUT_SECONDS', 3)
    logs = FakeLogs(statuses=['Running'])
    with pytest.raises(LogsQueryError, match='did not finish'):
        fetcher_for(logs).fetch({US_HAIKU: US_HAIKU}, Breakdown(), END - timedelta(hours=1), END)
    assert logs.stopped == ['q0']


def test_start_query_retries_only_refusals_that_started_nothing():
    class Throttled(FakeLogs):
        def __init__(self, refusals, code):
            super().__init__()
            self.refusals, self.code, self.calls = refusals, code, 0

        def start_query(self, **kwargs):
            self.calls += 1
            if self.calls <= self.refusals:
                raise aws_error(self.code)
            return super().start_query(**kwargs)
    # StartQuery goes to its own client (one that never retries it); the rest to the other
    assert InvocationLogFetcher(FakeLogs(), '/g', start_client='starter').start_client == 'starter'
    starter = Throttled(2, 'ThrottlingException')
    starter.rows = [row(END - timedelta(minutes=5), 'assumed-role/A')]
    fetcher = InvocationLogFetcher(starter, '/bedrock/logs', sleep=lambda s: None, start_client=starter)
    # Throttling is retried here, with backoff
    assert len(fetcher.fetch({US_HAIKU: US_HAIKU}, Breakdown(), END - timedelta(hours=1), END)) == 1
    assert starter.calls == 3 and len(starter.queries) == 1
    # Anything else (e.g. a validation error) is not retried
    starter = Throttled(1, 'InvalidParameterException')
    fetcher = InvocationLogFetcher(FakeLogs(), '/bedrock/logs', sleep=lambda s: None, start_client=starter)
    with pytest.raises(ClientError):
        fetcher.fetch({US_HAIKU: US_HAIKU}, Breakdown(), END - timedelta(hours=1), END)
    assert starter.calls == 1
    # Throttled every time: gives up after START_ATTEMPTS
    starter = Throttled(99, 'LimitExceededException')
    fetcher = InvocationLogFetcher(FakeLogs(), '/bedrock/logs', sleep=lambda s: None, start_client=starter)
    with pytest.raises(ClientError):
        fetcher.fetch({US_HAIKU: US_HAIKU}, Breakdown(), END - timedelta(hours=1), END)
    assert starter.calls == il.START_ATTEMPTS


def test_start_query_retries_a_connection_that_was_never_made():
    from botocore.exceptions import ConnectTimeoutError, EndpointConnectionError, ReadTimeoutError

    class Unreachable(FakeLogs):
        def __init__(self, errors):
            super().__init__()
            self.errors, self.calls = list(errors), 0

        def start_query(self, **kwargs):
            self.calls += 1
            if self.errors:
                raise self.errors.pop(0)
            return super().start_query(**kwargs)
    starter = Unreachable([EndpointConnectionError(endpoint_url='https://logs'),
                           ConnectTimeoutError(endpoint_url='https://logs')])
    fetcher = InvocationLogFetcher(starter, '/bedrock/logs', sleep=lambda s: None, start_client=starter)
    fetcher.fetch({US_HAIKU: US_HAIKU}, Breakdown(), END - timedelta(hours=1), END)
    assert starter.calls == 3 and len(starter.queries) == 1
    # A read timeout: the request was sent and may have started a query, so no retry
    starter = Unreachable([ReadTimeoutError(endpoint_url='https://logs')])
    fetcher = InvocationLogFetcher(FakeLogs(), '/bedrock/logs', sleep=lambda s: None, start_client=starter)
    with pytest.raises(ReadTimeoutError):
        fetcher.fetch({US_HAIKU: US_HAIKU}, Breakdown(), END - timedelta(hours=1), END)
    assert starter.calls == 1


def test_waits_end_when_cancelled_and_polling_slows_down():
    fetcher = InvocationLogFetcher(FakeLogs(), '/bedrock/logs')
    fetcher._cancel.set()
    assert fetcher._sleep(30) is True  # returns at once once cancelled, not after 30 s
    waits = []
    logs = FakeLogs(statuses=['Running'] * 12 + ['Complete'])
    InvocationLogFetcher(logs, '/bedrock/logs', sleep=waits.append).fetch(
        {US_HAIKU: US_HAIKU}, Breakdown(), END - timedelta(hours=1), END)
    assert waits[0] == 1.0 and waits == sorted(waits) and waits[-1] == il.MAX_POLL_SECONDS


def test_start_query_backoff_stops_when_the_run_is_cancelled():
    class Throttled(FakeLogs):
        calls = 0

        def start_query(self, **kwargs):
            self.calls += 1
            raise aws_error('ThrottlingException')
    starter = Throttled()
    fetcher = InvocationLogFetcher(FakeLogs(), '/bedrock/logs', start_client=starter,
                                   sleep=lambda s: fetcher._cancel.set())  # another window failed
    with pytest.raises(LogsQueryError, match='cancelled'):
        fetcher._run_query(il._QUERY_HEAD, END - timedelta(hours=1), END)
    assert starter.calls == 1  # no new (billed) query after the cancel


def test_a_stop_failure_does_not_hide_the_query_error():
    class NoStop(FakeLogs):
        def stop_query(self, queryId):
            raise aws_error('AccessDeniedException', 'StopQuery')
    with pytest.raises(LogsQueryError):
        fetcher_for(NoStop(statuses=['Failed'])).fetch({US_HAIKU: US_HAIKU}, Breakdown(), END - timedelta(hours=1), END)


def test_one_failed_query_stops_the_others():
    class OneFails(FakeLogs):
        def get_query_results(self, queryId):
            return {'status': 'Failed' if queryId == 'q0' else 'Running'}
    logs = OneFails()
    fetcher = fetcher_for(logs, max_concurrent=2)
    with pytest.raises(LogsQueryError):
        fetcher.fetch({US_HAIKU: US_HAIKU}, Breakdown(), END - timedelta(days=2), END)
    # The query still running was stopped instead of polled until its timeout
    assert sorted(logs.stopped) == sorted(q['id'] for q in logs.queries)
    # A window not started yet once the fetch is cancelled never starts a query
    started = len(logs.queries)
    with pytest.raises(LogsQueryError, match='cancelled'):
        fetcher._run_window('q', END - timedelta(hours=1), END)
    assert len(logs.queries) == started


def test_a_running_query_stops_at_its_next_poll_once_cancelled():
    class CancelWhilePolling(FakeLogs):
        def get_query_results(self, queryId):
            fetcher._cancel.set()  # e.g. Ctrl-C in the main thread
            return {'status': 'Running'}
    logs = CancelWhilePolling()
    fetcher = fetcher_for(logs)
    with pytest.raises(LogsQueryError, match='cancelled'):
        fetcher._run_query('q', END - timedelta(hours=1), END)
    assert logs.stopped == ['q0']


def test_unexpected_query_responses_are_query_errors():
    class NoId(FakeLogs):
        def start_query(self, **kwargs):
            return {}

    class BadMinute(FakeLogs):
        def get_query_results(self, queryId):
            return {'status': 'Complete', 'results': [[{'field': 'minute', 'value': 'not a time'}]]}
    for logs, match in ((NoId(), 'no query ID'), (BadMinute(), 'unexpected minute')):
        with pytest.raises(LogsQueryError, match=match):
            fetcher_for(logs).fetch({US_HAIKU: US_HAIKU}, Breakdown(), END - timedelta(hours=1), END)


def test_no_forms_means_no_query():
    logs = FakeLogs()
    assert fetcher_for(logs).fetch({}, Breakdown(), END - timedelta(days=1), END) == [] and logs.queries == []


def test_coverage_follows_creation_and_retention():
    created = END - timedelta(days=3)
    # A creation time inside a minute starts the coverage on that minute, so that window
    # boundaries fall on whole seconds
    logs = FakeLogs(groups=[{'logGroupName': '/bedrock/logs', 'creationTime': int(created.timestamp() * 1000) + 37123},
                            {'logGroupName': '/bedrock/logs-other'}])
    assert fetcher_for(logs).coverage_start(END - timedelta(days=30), END) == created
    logs.groups[0] = {'logGroupName': '/bedrock/logs', 'retentionInDays': 7}
    assert fetcher_for(logs).coverage_start(END - timedelta(days=30), END) == END - timedelta(days=7)
    assert fetcher_for(FakeLogs(groups=[])).coverage_start(END - timedelta(days=1), END) is None


def test_session_and_metadata_keys_are_kept():
    logs = FakeLogs([row(END - timedelta(minutes=5), 'assumed-role/OrdersService', session=ROLE_ARN),
                     row(END - timedelta(minutes=4), 'assumed-role/OrdersService', meta='checkout')])
    session = fetcher_for(logs).fetch({US_HAIKU: US_HAIKU}, Breakdown.parse('session'), END - timedelta(hours=1), END)
    assert {r['key'] for r in session} == {ROLE_ARN, 'role/OrdersService'}
    meta = fetcher_for(logs).fetch({US_HAIKU: US_HAIKU}, Breakdown.parse('metadata:app'), END - timedelta(hours=1), END)
    assert {r['key'] for r in meta} == {None, 'checkout'}


def test_principal_tags_reads_roles_and_users_and_keeps_each_error():
    iam = FakeIam(role_tags={'OrdersService': {'team': 'orders'}}, user_tags={'alice': {'team': 'ops'}}, deny={'Locked'})
    tags, errors = principal_tags(iam, ['role/OrdersService', 'user/ops/alice', 'role/Locked',
                                        f"arn:aws:iam::{ACCOUNT}:root", 'role/OrdersService'])
    assert tags == {'role/OrdersService': {'team': 'orders'}, 'user/ops/alice': {'team': 'ops'}}
    assert set(errors) == {'role/Locked'} and ('user', 'alice') in iam.calls and len(iam.calls) == 3


def test_iam_tag_reads_back_off_adaptively():
    # Hundreds of principals' tags, several at a time, against IAM's low request rate
    from bedrock_usage_analyzer.aws.client_factory import create_client
    assert create_client('iam', 'us-east-1').meta.config.retries['mode'] == 'adaptive'


def test_the_quoted_tag_error_is_a_denial_first():
    from bedrock_usage_analyzer.aws.invocation_logs import main_error
    gone, denied = aws_error('NoSuchEntity', 'ListRoleTags'), aws_error('AccessDenied', 'ListRoleTags')
    assert main_error([gone, denied]) is denied and main_error([gone]) is gone and main_error([]) is None


def test_each_report_quotes_the_tag_error_of_its_own_principals():
    class Iam(FakeIam):
        def list_role_tags(self, RoleName):
            raise aws_error('NoSuchEntity' if RoleName == 'Gone' else 'AccessDenied', 'ListRoleTags')
    logs = FakeLogs([row(T1, 'assumed-role/Gone', model=US_HAIKU), row(T1, 'assumed-role/Locked', model=f"eu.{HAIKU}")])
    builder = builder_for(Breakdown(), logs, iam=Iam())
    builder.prepare([US_HAIKU, f"eu.{HAIKU}"], END, 1)
    first = builder.section([US_HAIKU], {}, {US_HAIKU: cloudwatch({T1: (300, 0, 2)})}, GRANULARITY, ['1hour'])
    second = builder.section([f"eu.{HAIKU}"], {}, {f"eu.{HAIKU}": cloudwatch({T1: (300, 0, 2)})}, GRANULARITY, ['1hour'])
    assert 'NoSuchEntity' in ' '.join(first['notes']) and 'AccessDenied' in ' '.join(second['notes'])


# ------------------------------------------------------------------ builder

def stats_fn(ts_data, time_period):
    from bedrock_usage_analyzer.core.analyzer import BedrockAnalyzer
    analyzer = BedrockAnalyzer.__new__(BedrockAnalyzer)
    analyzer.metrics_fetcher = CloudWatchMetricsFetcher(None)
    return analyzer._calculate_stats_from_time_series(ts_data, time_period)


def cloudwatch(minutes):
    """CloudWatch fetch result for one ModelId: {minute: (input, output, invocations)}."""
    stamps = sorted(minutes)
    return {'end_time': END, '60_token': {
        'timestamps': stamps, 'period': 60,
        'data': {'input_tokens': [minutes[s][0] for s in stamps], 'output_tokens': [minutes[s][1] for s in stamps],
                 'invocations': [minutes[s][2] for s in stamps]}}}


def builder_for(breakdown, logs, iam=None, bedrock=None):
    bedrock = bedrock or FakeBedrock(logging_config={'cloudWatchConfig': {'logGroupName': '/bedrock/logs'}})
    return BreakdownBuilder(breakdown, REGION, bedrock, CloudWatchMetricsFetcher(None), stats_fn, timezone.utc,
                            ACCOUNT, logs_client=logs, iam_client=iam or FakeIam())


T1, T2, T3 = (END - timedelta(minutes=m) for m in (30, 20, 10))


def test_principal_breakdown_with_remainder():
    logs = FakeLogs([row(T1, 'assumed-role/OrdersService', i=600, o=400, n=2),
                     row(T2, 'user/ops/alice', i=200, o=100, n=1)])
    iam = FakeIam(role_tags={'OrdersService': {'team': 'orders'}})
    builder = builder_for(Breakdown(), logs, iam)
    assert builder.prepare([US_HAIKU], END, 30) is None
    cw = {US_HAIKU: cloudwatch({T1: (600, 400, 2), T2: (200, 100, 1), T3: (300, 400, 1)})}
    section = builder.section([US_HAIKU], {US_HAIKU: US_HAIKU}, cw, GRANULARITY, PERIODS)
    hour = section['periods']['1hour']
    names = [r['name'] for r in hour['rows']]
    assert names == ['role/OrdersService', 'user/alice', UNATTRIBUTED]
    orders = hour['rows'][0]
    assert orders['tokens'] == 1000 and orders['requests'] == 2 and orders['tags'] == {'team': 'orders'}
    assert orders['share_tokens'] == pytest.approx(1000 / 2000) and orders['via'] == [US_HAIKU]
    assert orders['tpm_max'] == 1000 and orders['rpm_max'] == 2
    remainder = hour['rows'][-1]
    assert remainder['tokens'] == 700 and remainder['requests'] == 1
    assert sum(r['share_tokens'] for r in hour['rows']) == pytest.approx(1.0)
    assert hour['total_tokens'] == 2000 and not hour['partial']
    assert set(section['time_series']['1hour']) == set(names)
    assert section['coverage']['end'] == (END - timedelta(minutes=5)).isoformat() and section['log_group'] == '/bedrock/logs'


def test_a_record_logged_a_minute_late_is_not_missing():
    # CloudWatch counted it at T1, the log binned it a minute later: no remainder over the window
    logs = FakeLogs([row(T1 + timedelta(minutes=1), 'assumed-role/A', i=100, o=0, n=1)])
    builder = builder_for(Breakdown(), logs)
    builder.prepare([US_HAIKU], END, 1)
    section = builder.section([US_HAIKU], {}, {US_HAIKU: cloudwatch({T1: (100, 0, 1)})}, GRANULARITY, ['1hour'])
    assert [r['name'] for r in section['periods']['1hour']['rows']] == ['role/A']


def test_tag_breakdown_groups_principals_and_marks_unreadable_tags():
    logs = FakeLogs([row(T1, 'assumed-role/OrdersService'), row(T1, 'assumed-role/Billing'),
                     row(T2, 'assumed-role/Locked'), row(T2, 'assumed-role/Untagged'),
                     row(T3, f"arn:aws:iam::{ACCOUNT}:root")])
    iam = FakeIam(role_tags={'OrdersService': {'team': 'shop'}, 'Billing': {'team': 'shop'}}, deny={'Locked'})
    builder = builder_for(Breakdown.parse('tag:team'), logs, iam)
    builder.prepare([US_HAIKU], END, 1)
    section = builder.section([US_HAIKU], {}, {US_HAIKU: cloudwatch({})}, GRANULARITY, ['1hour'])
    rows = {r['name']: r for r in section['periods']['1hour']['rows']}
    assert rows['shop']['principals'] == ['role/Billing', 'role/OrdersService'] and rows['shop']['requests'] == 2
    assert set(rows) == {'shop', '(tags not readable)', '(no team tag)'}
    assert rows['(no team tag)']['principals'] == [f"arn:aws:iam::{ACCOUNT}:root", 'role/Untagged']
    assert any('could not be read' in n for n in section['notes'])
    assert rows['shop']['share_tokens'] is None  # CloudWatch had no data for the window


def test_metadata_and_session_breakdowns():
    logs = FakeLogs([row(T1, 'assumed-role/A', meta='checkout', session=ROLE_ARN),
                     row(T2, 'assumed-role/A', session=ROLE2_ARN)])
    builder = builder_for(Breakdown.parse('metadata:app'), logs)
    builder.prepare([US_HAIKU], END, 1)
    rows = builder.section([US_HAIKU], {}, {US_HAIKU: cloudwatch({})}, GRANULARITY, ['1hour'])['periods']['1hour']['rows']
    assert {r['name'] for r in rows} == {'checkout', '(no app)'}
    builder = builder_for(Breakdown.parse('session'), logs)
    builder.prepare([US_HAIKU], END, 1)
    rows = builder.section([US_HAIKU], {}, {US_HAIKU: cloudwatch({})}, GRANULARITY, ['1hour'])['periods']['1hour']['rows']
    assert {r['name'] for r in rows} == {ROLE_ARN, ROLE2_ARN} and all(r['principals'] == ['role/A'] for r in rows)


def test_a_principal_filter_shows_the_others_as_one_row():
    logs = FakeLogs([row(T1, 'assumed-role/OrdersService', i=300, o=0), row(T1, 'assumed-role/Billing', i=100, o=0)])
    builder = builder_for(Breakdown.parse('principal', ['role/OrdersService']), logs)
    builder.prepare([US_HAIKU], END, 1)
    section = builder.section([US_HAIKU], {}, {US_HAIKU: cloudwatch({T1: (400, 0, 2)})}, GRANULARITY, ['1hour'])
    rows = section['periods']['1hour']['rows']
    assert [(r['name'], r['tokens']) for r in rows] == [('role/OrdersService', 300), ('(other principals)', 100)]
    assert section['principal_filter'] == ['role/OrdersService']


def test_tag_keys_and_principal_names_match_regardless_of_case():
    # IAM tag keys and role/user names are case-insensitive
    logs = FakeLogs([row(T1, 'assumed-role/Billing'), row(T1, 'assumed-role/Orders')])
    iam = FakeIam(role_tags={'Billing': {'Team': 'finance'}})
    builder = builder_for(Breakdown.parse('tag:team', ['role/billing']), logs, iam)
    builder.prepare([US_HAIKU], END, 1)
    rows = builder.section([US_HAIKU], {}, {US_HAIKU: cloudwatch({})}, GRANULARITY, ['1hour'])['periods']['1hour']['rows']
    assert [(r['name'], r['principals']) for r in rows] == [('finance', ['role/Billing']), ('(other principals)', [])]


def test_tags_are_read_only_for_the_selected_principals():
    logs = FakeLogs([row(T1, 'assumed-role/OrdersService'), row(T1, 'assumed-role/Billing')])
    iam = FakeIam()
    builder_for(Breakdown.parse('tag:team', ['role/OrdersService']), logs, iam).prepare([US_HAIKU], END, 1)
    assert iam.calls == [('role', 'OrdersService')]


def test_a_principal_breakdown_reads_only_its_rows_tags_once(monkeypatch):
    from bedrock_usage_analyzer.core import breakdown as breakdown_module
    monkeypatch.setattr(breakdown_module, 'MAX_GROUPS', 2)
    logs = FakeLogs([row(T1, 'assumed-role/Big', i=900), row(T1, 'assumed-role/Small1', i=1),
                     row(T1, 'assumed-role/Small2', i=1)])
    iam = FakeIam(role_tags={'Big': {'team': 'core'}}, deny={'Small1'})
    builder = builder_for(Breakdown(), logs, iam)
    builder.prepare([US_HAIKU], END, 1)
    assert iam.calls == []  # nothing read before the rows are known
    for _ in range(2):  # two reports of the run
        section = builder.section([US_HAIKU], {}, {US_HAIKU: cloudwatch({})}, GRANULARITY, ['1hour'])
    assert iam.calls == [('role', 'Big')]  # the folded callers' tags are never read
    rows = section['periods']['1hour']['rows']
    assert rows[0]['tags'] == {'team': 'core'} and rows[1]['tags'] == {}
    assert section['notes'] == []


def test_many_groups_are_folded_into_one_row(monkeypatch):
    from bedrock_usage_analyzer.core import breakdown as breakdown_module
    monkeypatch.setattr(breakdown_module, 'MAX_GROUPS', 3)
    logs = FakeLogs([row(T1, f"assumed-role/R{n}", i=100 * (n + 1), o=0) for n in range(5)])
    builder = builder_for(Breakdown(), logs)
    builder.prepare([US_HAIKU], END, 1)
    section = builder.section([US_HAIKU], {}, {US_HAIKU: cloudwatch({T1: (1500, 0, 5)})}, GRANULARITY, ['1hour'])
    rows = section['periods']['1hour']['rows']
    assert [(r['name'], r['tokens']) for r in rows] == [('role/R4', 500), ('role/R3', 400), ('(3 smaller groups)', 600)]
    assert rows[-1]['principals'] == [] and set(section['time_series']['1hour']) == {r['name'] for r in rows}
    # Under the cap nothing is folded
    monkeypatch.setattr(breakdown_module, 'MAX_GROUPS', 5)
    builder = builder_for(Breakdown(), logs)
    builder.prepare([US_HAIKU], END, 1)
    section = builder.section([US_HAIKU], {}, {US_HAIKU: cloudwatch({T1: (1500, 0, 5)})}, GRANULARITY, ['1hour'])
    assert len(section['periods']['1hour']['rows']) == 5


def test_the_breakdown_ends_before_records_still_being_delivered():
    # Run at END: the logs are read up to END - 5 min (records arrive seconds to minutes
    # late); CloudWatch's later minutes are not compared with logs that may not have them yet
    logs = FakeLogs([row(T1, 'assumed-role/A', i=100, o=0, n=1), row(END - timedelta(minutes=2), 'assumed-role/A')])
    builder = builder_for(Breakdown(), logs)
    builder.prepare([US_HAIKU], END, 1)
    late = END - timedelta(minutes=2)
    section = builder.section([US_HAIKU], {}, {US_HAIKU: cloudwatch({T1: (100, 0, 1), late: (900, 0, 3)})},
                              GRANULARITY, ['1hour'])
    hour = section['periods']['1hour']
    assert [r['name'] for r in hour['rows']] == ['role/A'] and hour['total_tokens'] == 100
    # The breakdown's hour ends there, so it is fully covered
    logs_end = (END - timedelta(minutes=5)).isoformat()
    assert section['coverage']['end'] == logs_end and hour['covered_to'] == logs_end and not hour['partial']
    assert logs.queries[-1]['end'] == int((END - timedelta(minutes=5)).timestamp())


def test_without_an_account_it_is_read_from_sts(monkeypatch, caplog):
    from bedrock_usage_analyzer.core import breakdown as breakdown_module
    caplog.set_level('INFO')
    monkeypatch.setattr(breakdown_module, 'get_account_id', lambda region: ACCOUNT)
    profile_arn = f"arn:aws:bedrock:{REGION}:{ACCOUNT}:application-inference-profile/app0000001"
    logs = FakeLogs([row(T1, 'assumed-role/A', model=profile_arn)])
    builder = builder_for(Breakdown(), logs)
    builder.account = None
    builder.prepare(['app0000001'], END, 1)
    assert builder.account == ACCOUNT
    section = builder.section(['app0000001'], {}, {'app0000001': cloudwatch({})}, GRANULARITY, ['1hour'])
    assert [r['name'] for r in section['periods']['1hour']['rows']] == ['role/A']

    def no_sts(region):
        raise RuntimeError('no STS')
    monkeypatch.setattr(breakdown_module, 'get_account_id', no_sts)
    builder = builder_for(Breakdown(), FakeLogs())
    builder.account = None
    assert builder.prepare(['app0000001'], END, 1) is None and 'account ID unavailable' in caplog.text


def test_each_period_keeps_its_own_largest_callers(monkeypatch):
    from bedrock_usage_analyzer.core import breakdown as breakdown_module
    monkeypatch.setattr(breakdown_module, 'MAX_GROUPS', 2)
    logs = FakeLogs([row(END - timedelta(days=2), 'assumed-role/Old', i=1000, o=0),
                     row(T1, 'assumed-role/New', i=10, o=0), row(T1, 'assumed-role/Newer', i=5, o=0)])
    builder = builder_for(Breakdown(), logs)
    builder.prepare([US_HAIKU], END, 7)
    section = builder.section([US_HAIKU], {}, {US_HAIKU: cloudwatch({})}, GRANULARITY, ['1hour', '7days'])
    assert [r['name'] for r in section['periods']['7days']['rows']] == ['role/Old', '(2 smaller groups)']
    assert [r['name'] for r in section['periods']['1hour']['rows']] == ['role/New', 'role/Newer']


def test_a_section_that_cannot_be_built_keeps_the_report(caplog):
    caplog.set_level('INFO')
    builder = builder_for(Breakdown(), FakeLogs([row(T1, 'assumed-role/A')]))
    builder.prepare([US_HAIKU], END, 1)
    section = builder.section([US_HAIKU], {}, {US_HAIKU: cloudwatch({})}, GRANULARITY, ['2hours'])
    assert 'could not be built' in section['unavailable'] and section['periods'] == {}
    assert section['label'] == 'IAM principal' and 'unavailable for this report' in caplog.text


def test_rows_of_other_reports_and_old_minutes_are_left_out():
    deployment = f"arn:aws:bedrock:{REGION}:{ACCOUNT}:custom-model-deployment/dep0000001"
    logs = FakeLogs([row(T1, 'assumed-role/A', model=deployment), row(T1, 'assumed-role/B'),
                     row(END - timedelta(days=2), 'assumed-role/Old')])
    builder = builder_for(Breakdown(), logs)
    builder.prepare([US_HAIKU, deployment], END, 7)
    section = builder.section([deployment], {deployment: 'my-lite'}, {deployment: cloudwatch({})}, GRANULARITY,
                              ['1hour', '7days'])
    assert [r['name'] for r in section['periods']['1hour']['rows']] == ['role/A']
    assert section['periods']['1hour']['rows'][0]['via'] == ['my-lite']
    assert [r['name'] for r in section['periods']['7days']['rows']] == ['role/A']


def test_a_partly_covered_period_is_marked():
    logs = FakeLogs([row(T1, 'assumed-role/A')], groups=[{'logGroupName': '/bedrock/logs', 'retentionInDays': 3}])
    builder = builder_for(Breakdown(), logs)
    builder.prepare([US_HAIKU], END, 30)
    before = END - timedelta(days=10)
    section = builder.section([US_HAIKU], {}, {US_HAIKU: cloudwatch({before: (999, 0, 9), T1: (100, 50, 1)})},
                              GRANULARITY, ['7days', '30days'])
    assert section['periods']['30days']['partial'] and section['periods']['7days']['partial']
    assert section['periods']['7days']['partial_reason'] == 'logs'
    # CloudWatch usage from before the logs start is not counted as unattributed
    assert section['periods']['30days']['total_tokens'] == 150


def test_the_logs_are_read_no_further_back_than_cloudwatch_keeps_minutes():
    # CloudWatch keeps 1-minute data for 15 days: a 30-day period is compared over those only
    logs = FakeLogs([row(END - timedelta(days=20), 'assumed-role/Old'), row(T1, 'assumed-role/A')])
    builder = builder_for(Breakdown(), logs)
    builder.prepare([US_HAIKU], END, 30)
    assert builder.coverage[0] == END - timedelta(days=15)  # counted from now, as CloudWatch's
    section = builder.section([US_HAIKU], {}, {US_HAIKU: cloudwatch({T1: (100, 50, 1)})}, GRANULARITY,
                              ['14days', '30days'])
    assert section['periods']['30days']['partial'] and not section['periods']['14days']['partial']
    assert section['periods']['30days']['partial_reason'] == 'cloudwatch'
    assert section['periods']['14days']['partial_reason'] is None
    assert [r['name'] for r in section['periods']['30days']['rows']] == ['role/A']


def test_a_failed_one_minute_token_fetch_is_flagged_not_reported_as_no_usage():
    # The breakdown leaves a ModelId out only if the fetcher says its fetch failed
    class Failing:
        def get_metric_data(self, **kwargs):
            raise aws_error('ThrottlingException', 'GetMetricData')
    fetcher = CloudWatchMetricsFetcher(Failing())
    fetcher.total_chunks = 1
    failed = fetcher._fetch_token_metrics(US_HAIKU, END - timedelta(hours=1), END, 60)
    assert failed['fetch_failed'] is True and failed['timestamps'] == []

    class Empty:
        def get_metric_data(self, **kwargs):
            return {'MetricDataResults': []}
    fetcher = CloudWatchMetricsFetcher(Empty())
    fetcher.total_chunks = 1
    assert 'fetch_failed' not in fetcher._fetch_token_metrics(US_HAIKU, END - timedelta(hours=1), END, 60)


def test_minutes_logged_before_this_reports_cloudwatch_data_starts_are_left_out():
    # The report's CloudWatch data was fetched 10 minutes after the logs were read: its
    # 1-minute data starts 10 minutes later too, and so does the breakdown
    first = END - timedelta(days=15) + timedelta(minutes=2)
    logs = FakeLogs([row(first, 'assumed-role/Early'), row(T1, 'assumed-role/A')])
    builder = builder_for(Breakdown(), logs)
    builder.prepare([US_HAIKU], END, 30)
    data = cloudwatch({T1: (150, 0, 1)})
    data['end_time'] = END + timedelta(minutes=10, seconds=30)
    section = builder.section([US_HAIKU], {}, {US_HAIKU: data}, GRANULARITY, ['30days'])
    assert section['coverage']['start'] == (END - timedelta(days=15) + timedelta(minutes=11)).isoformat()
    assert [r['name'] for r in section['periods']['30days']['rows']] == ['role/A']


def test_caller_values_never_take_the_place_of_the_tools_own_rows():
    logs = FakeLogs([row(T1, 'assumed-role/A', meta='(other principals)'), row(T1, 'assumed-role/B', meta='x')])
    builder = builder_for(Breakdown.parse('metadata:app', ['role/A']), logs)
    builder.prepare([US_HAIKU], END, 1)
    rows = builder.section([US_HAIKU], {}, {US_HAIKU: cloudwatch({})}, GRANULARITY, ['1hour'])['periods']['1hour']['rows']
    assert [r['name'] for r in rows] == ['"(other principals)"', '(other principals)']
    # Distinct values keep distinct names, even one that is already quoted
    logs = FakeLogs([row(T1, 'assumed-role/A', meta='(batch)'), row(T1, 'assumed-role/A', meta='"(batch)"')])
    builder = builder_for(Breakdown.parse('metadata:app'), logs)
    builder.prepare([US_HAIKU], END, 1)
    rows = builder.section([US_HAIKU], {}, {US_HAIKU: cloudwatch({})}, GRANULARITY, ['1hour'])['periods']['1hour']['rows']
    assert sorted(r['name'] for r in rows) == ['""(batch)""', '"(batch)"']


def test_each_period_lists_only_its_own_principals_and_profiles():
    logs = FakeLogs([row(END - timedelta(days=2), 'assumed-role/A', model='app0000001'),
                     row(T1, 'assumed-role/B', model='app0000002')])
    iam = FakeIam(role_tags={'A': {'team': 'x'}, 'B': {'team': 'x'}})
    builder = builder_for(Breakdown.parse('tag:team'), logs, iam)
    builder.prepare(['app0000001', 'app0000002'], END, 7)
    names = {'app0000001': 'P1', 'app0000002': 'P2'}
    section = builder.section(['app0000001', 'app0000002'], names, {'app0000001': cloudwatch({})}, GRANULARITY,
                              ['1hour', '7days'])
    hour, week = section['periods']['1hour']['rows'][0], section['periods']['7days']['rows'][0]
    assert (hour['principals'], hour['via']) == (['role/B'], ['P2'])
    assert (week['principals'], week['via']) == (['role/A', 'role/B'], ['P1', 'P2'])


def test_a_model_id_whose_cloudwatch_fetch_failed_is_left_out():
    logs = FakeLogs([row(T1, 'assumed-role/A', i=100, o=0, model='app0000001'),
                     row(T1, 'assumed-role/B', i=900, o=0, model='app0000002')])
    builder = builder_for(Breakdown(), logs)
    builder.prepare(['app0000001', 'app0000002'], END, 1)
    failed = cloudwatch({})
    failed['60_token']['fetch_failed'] = True
    section = builder.section(['app0000001', 'app0000002'], {},
                              {'app0000001': cloudwatch({T1: (100, 0, 1)}), 'app0000002': failed},
                              GRANULARITY, ['1hour'])
    rows = section['periods']['1hour']['rows']
    assert [(r['name'], r['share_tokens']) for r in rows] == [('role/A', 1.0)]
    assert any('1-minute data could not be fetched for app0000002' in n for n in section['notes'])


def test_notes_belong_to_the_report_they_are_about():
    # Report 1's principal has unreadable tags; report 2's principal is fine
    logs = FakeLogs([row(T1, 'assumed-role/Deleted', model='app0000001'),
                     row(T1, 'assumed-role/Other', model='app0000002')])
    iam = FakeIam(deny={'Deleted'})
    builder = builder_for(Breakdown(), logs, iam)
    builder.prepare(['app0000001', 'app0000002'], END, 1)
    first = builder.section(['app0000001'], {}, {'app0000001': cloudwatch({})}, GRANULARITY, ['1hour'])
    second = builder.section(['app0000002'], {}, {'app0000002': cloudwatch({})}, GRANULARITY, ['1hour'])
    assert any('could not be read' in n for n in first['notes']) and second['notes'] == []


def test_an_unexpected_error_while_reading_the_logs_keeps_the_reports(monkeypatch):
    from bedrock_usage_analyzer.core import breakdown as breakdown_module

    def broken(*args, **kwargs):
        raise TypeError('unexpected value')
    monkeypatch.setattr(breakdown_module, 'model_id_forms', broken)
    builder = builder_for(Breakdown(), FakeLogs())
    assert 'unexpected value' in builder.prepare([US_HAIKU], END, 1)
    assert 'unexpected value' in builder.section([US_HAIKU], {}, {}, GRANULARITY, ['1hour'])['unavailable']


def test_a_principal_filter_that_matches_no_logged_caller_is_noted():
    logs = FakeLogs([row(T1, 'assumed-role/Orders')])
    builder = builder_for(Breakdown.parse('principal', ['role/Orders', 'role/Typo']), logs)
    builder.prepare([US_HAIKU], END, 1)
    section = builder.section([US_HAIKU], {}, {US_HAIKU: cloudwatch({})}, GRANULARITY, ['1hour'])
    assert any('no logged call from role/Typo' in n for n in section['notes'])
    assert not any('role/Orders' in n for n in section['notes'])


@pytest.mark.parametrize('bedrock,logs,expected', [
    (FakeBedrock(), FakeLogs(), 'not enabled'),
    (FakeBedrock(logging_config={'s3Config': {'bucketName': 'b'}}), FakeLogs(), 'S3 only'),
    (None, FakeLogs(groups=[]), 'does not exist'),
    (None, FakeLogs(fail=aws_error('AccessDeniedException')), 'could not read'),
    (None, FakeLogs(statuses=['Failed']), 'could not read'),
])
def test_an_unavailable_breakdown_says_why_and_keeps_the_report(bedrock, logs, expected):
    builder = builder_for(Breakdown(), logs, bedrock=bedrock)
    reason = builder.prepare([US_HAIKU], END, 1)
    assert expected in reason and builder.prepare([US_HAIKU], END, 1) == reason  # asked once
    section = builder.section([US_HAIKU], {}, {}, GRANULARITY, ['1hour'])
    assert expected in section['unavailable'] and section['periods'] == {}


def test_the_log_group_is_found_past_the_first_page_of_prefix_matches():
    class Paged(FakeLogs):
        def describe_log_groups(self, logGroupNamePrefix, nextToken=None):
            self.pages = getattr(self, 'pages', 0) + 1
            if nextToken is None:  # a page of other groups sharing the prefix
                return {'logGroups': [{'logGroupName': f'/bedrock/logs-{n}'} for n in range(50)], 'nextToken': 't'}
            return {'logGroups': [{'logGroupName': '/bedrock/logs', 'creationTime': 0}]}
    logs = Paged()
    assert InvocationLogFetcher(logs, '/bedrock/logs').coverage_start(END - timedelta(days=1), END) == END - timedelta(days=1)
    assert logs.pages == 2
    missing = FakeLogs(groups=[{'logGroupName': '/bedrock/logs-other'}])
    assert InvocationLogFetcher(missing, '/bedrock/logs').coverage_start(END - timedelta(days=1), END) is None


def test_a_query_times_out_on_wall_time_too(monkeypatch):
    # Slow (retried) polls count, not only the waits between them
    clock = iter(range(0, 10 ** 6, 400))
    monkeypatch.setattr(il.time, 'monotonic', lambda: next(clock))
    logs = FakeLogs(statuses=['Running'])
    with pytest.raises(LogsQueryError, match='did not finish'):
        InvocationLogFetcher(logs, '/bedrock/logs', sleep=lambda s: None).fetch(
            {US_HAIKU: US_HAIKU}, Breakdown(), END - timedelta(hours=1), END)
    assert logs.stopped == ['q0']


def test_a_tag_read_failure_keeps_the_rows():
    class Broken(FakeIam):
        def list_role_tags(self, RoleName):
            raise RuntimeError('no iam')
    logs = FakeLogs([row(T1, 'assumed-role/A')])
    builder = builder_for(Breakdown(), logs, iam=Broken())
    builder.prepare([US_HAIKU], END, 1)
    section = builder.section([US_HAIKU], {}, {US_HAIKU: cloudwatch({T1: (150, 0, 1)})}, GRANULARITY, ['1hour'])
    assert [r['name'] for r in section['periods']['1hour']['rows']] == ['role/A']
    assert section['periods']['1hour']['rows'][0]['tags'] == {}
    assert any('tags could not be read (no iam)' in n for n in section['notes'])


def test_a_log_group_newer_than_the_breakdown_end_says_why_it_is_empty():
    # Logging enabled a moment ago: no window to read, and the report says so
    logs = FakeLogs(groups=[{'logGroupName': '/bedrock/logs', 'creationTime': int(END.timestamp() * 1000) + 60000}])
    builder = builder_for(Breakdown(), logs)
    reason = builder.prepare([US_HAIKU], END, 1)
    assert 'holds no records from before' in reason and logs.queries == []
    assert 'holds no records' in builder.section([US_HAIKU], {}, {}, GRANULARITY, ['1hour'])['unavailable']


def test_a_given_log_group_skips_the_logging_configuration():
    bedrock = FakeBedrock()
    builder = builder_for(Breakdown.parse('principal', log_group='/bedrock/logs'), FakeLogs(), bedrock=bedrock)
    assert builder.prepare([US_HAIKU], END, 1) is None and bedrock.calls == []


# ------------------------------------------------------------------ report and CLI

def test_report_renders_the_breakdown_escaped(analyzer, tmp_path, monkeypatch):
    from bedrock_usage_analyzer.core import breakdown as breakdown_module
    hostile = 'assumed-role/<img src=x onerror=alert(1)>'
    logs = FakeLogs([row(datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=8),
                         hostile, model=f"au.{HAIKU}")])
    monkeypatch.setattr(breakdown_module, 'create_client',
                        lambda service, region=None, **_: logs if service == 'logs' else FakeIam())
    analyzer.breakdown = Breakdown.parse('principal', log_group='/bedrock/logs')
    analyzer.account = ACCOUNT
    out = tmp_path / 'results'
    analyzer.analyze([{'model_id': HAIKU, 'profile_prefix': 'au'}], output_dir=str(out))
    files = sorted(os.listdir(out))
    data = json.loads((out / files[1]).read_text())
    assert data['breakdown']['label'] == 'IAM principal'
    names = [r['name'] for r in data['breakdown']['periods']['1hour']['rows']]
    assert 'role/<img src=x onerror=alert(1)>' in names
    html = (out / files[0]).read_text()
    assert 'Usage by IAM principal' in html and '<img src=x' not in html
    assert 'role/&lt;img src=x onerror=alert(1)&gt;' in html
    assert 'breakdown_tpm_1hour' in html


def test_the_chart_gets_tpm_pairs_so_no_caller_name_is_an_object_key():
    from bedrock_usage_analyzer.core.output_generator import _breakdown_tpm
    tpm = {'timestamps': ['t'], 'values': [1]}
    breakdown = {'time_series': {'1hour': {'__proto__': {'TPM': tpm, 'RPM': tpm}, 'constructor': {'TPM': tpm},
                                           'no-tpm': {}}}}
    assert _breakdown_tpm(breakdown) == [['1hour', [['__proto__', tpm], ['constructor', tpm]]]]
    assert _breakdown_tpm(None) == [] and _breakdown_tpm({'unavailable': 'x'}) == []


def test_no_breakdown_keeps_the_report_as_before(analyzer, tmp_path):
    out = tmp_path / 'results'
    analyzer.analyze([{'model_id': HAIKU, 'profile_prefix': 'au'}], output_dir=str(out))
    files = sorted(os.listdir(out))
    assert json.loads((out / files[1]).read_text())['breakdown'] is None
    assert 'class="breakdown-section"' not in (out / files[0]).read_text()


def test_unavailable_breakdown_is_shown_in_the_report(analyzer, tmp_path):
    analyzer.breakdown = Breakdown()
    out = tmp_path / 'results'
    analyzer.analyze([{'model_id': HAIKU, 'profile_prefix': 'au'}], output_dir=str(out))
    files = sorted(os.listdir(out))
    assert 'not enabled' in json.loads((out / files[1]).read_text())['breakdown']['unavailable']
    assert 'Not available: model invocation logging is not enabled' in (out / files[0]).read_text()


@pytest.fixture
def analyzer(sydney_bedrock, monkeypatch):
    from test_analyzer_and_output import FakeCloudWatch, FakeQuotas
    from bedrock_usage_analyzer.core import analyzer as analyzer_module
    from bedrock_usage_analyzer.core.profile_fetcher import InferenceProfileFetcher
    cw = FakeCloudWatch({'auapp000001': (2, 100, 50), f"au.{HAIKU}": (1, 10, 5)})
    clients = {'bedrock': sydney_bedrock, 'cloudwatch': cw, 'service-quotas': FakeQuotas()}
    monkeypatch.setattr(analyzer_module, 'create_client', lambda service, region=None, **_: clients[service])
    monkeypatch.setattr(analyzer_module, 'list_quota_codes', lambda region, **_: None)
    monkeypatch.setattr('bedrock_usage_analyzer.aws.servicequotas.regional_client', lambda region: clients['service-quotas'])
    a = analyzer_module.BedrockAnalyzer('ap-southeast-2', GRANULARITY, profile_fetcher=InferenceProfileFetcher(sydney_bedrock))
    a.cw = cw
    return a


@pytest.mark.parametrize('argv,expected', [
    (['--breakdown', 'tag:team'], ('tag', 'team', ())),
    (['--principal', 'role/A'], ('principal', None, ('role/A',))),
    (['--breakdown', 'session', '--log-group', '/g'], ('session', None, ())),
])
def test_cli_options_become_a_breakdown(monkeypatch, tmp_path, argv, expected):
    from bedrock_usage_analyzer import __main__ as cli
    from bedrock_usage_analyzer.core import user_inputs, analyzer
    seen = {}

    def fake_collect(self, **kwargs):
        seen['breakdown'] = kwargs['breakdown']
        self.region, self.account, self.models = REGION, ACCOUNT, [{'model_id': HAIKU, 'profile_prefix': 'us'}]
        self.breakdown = kwargs['breakdown']
    monkeypatch.setattr(user_inputs.UserInputs, 'collect', fake_collect)
    monkeypatch.setattr(user_inputs.UserInputs, 'fm_models', lambda self: [])

    class FakeAnalyzer:
        def __init__(self, region, granularity, **kwargs):
            seen['analyzer'] = kwargs

        def analyze(self, models, output_dir):
            seen['ran'] = True
    monkeypatch.setattr(analyzer, 'BedrockAnalyzer', FakeAnalyzer)
    monkeypatch.setattr('sys.argv', ['bua', 'analyze', '-r', REGION, '-m', HAIKU, '-y', '-o', str(tmp_path), *argv])
    cli.main()
    b = seen['breakdown']
    assert (b.kind, b.key, b.principals) == expected
    assert b.log_group == ('/g' if '--log-group' in argv else None)
    assert seen['analyzer']['breakdown'] is b and seen['analyzer']['account'] == ACCOUNT and seen['ran']


@pytest.mark.parametrize('extra', [[], ['--log-group', '/g']])
@pytest.mark.parametrize('value', ['owner', ''])
def test_an_invalid_cli_breakdown_exits(monkeypatch, caplog, value, extra):
    # An empty --breakdown "$BY" too: it is an error, not a run without a breakdown
    from bedrock_usage_analyzer import __main__ as cli
    monkeypatch.setattr('sys.argv', ['bua', 'analyze', '-r', REGION, '-m', HAIKU, '--breakdown', value, *extra])
    with pytest.raises(SystemExit):
        cli.main()
    assert 'unknown breakdown' in caplog.text


def test_a_log_group_alone_does_not_start_a_breakdown(monkeypatch, caplog):
    from bedrock_usage_analyzer import __main__ as cli
    monkeypatch.setattr('sys.argv', ['bua', 'analyze', '-r', REGION, '-m', HAIKU, '--log-group', '/g'])
    with pytest.raises(SystemExit):
        cli.main()
    assert '--log-group needs --breakdown or --principal' in caplog.text


# ------------------------------------------------------------------ interactive question

def inputs_with(monkeypatch, bedrock, answers):
    from bedrock_usage_analyzer.core import user_inputs as ui_module
    from bedrock_usage_analyzer.core.profile_fetcher import InferenceProfileFetcher
    inputs = ui_module.UserInputs()
    inputs.region = REGION
    inputs.profile_fetcher = InferenceProfileFetcher(bedrock)
    feed = iter(answers)
    monkeypatch.setattr('builtins.input', lambda prompt='': next(feed))
    return inputs


LOGGING = {'cloudWatchConfig': {'logGroupName': '/bedrock/logs'}}


@pytest.mark.parametrize('answers,expected', [
    (['1'], None),
    (['2'], ('principal', None)),
    (['3'], ('session', None)),
    (['4', 'bad key!', 'team'], ('tag', 'team')),
    (['5', 'app'], ('metadata', 'app')),
    # Enter at the key prompt: no breakdown (a way out of a mistaken choice)
    (['4', 'bad key!', ''], None), (['5', ''], None),
])
def test_interactive_breakdown_choice(monkeypatch, answers, expected):
    inputs = inputs_with(monkeypatch, FakeBedrock(logging_config=LOGGING), answers)
    chosen = inputs._select_breakdown()
    assert (None if chosen is None else (chosen.kind, chosen.key)) == expected
    if chosen:
        assert chosen.log_group == '/bedrock/logs'


@pytest.mark.parametrize('answers', [['2'], ['3'], ['4']])
def test_interactive_breakdown_with_a_log_group_it_cannot_use_asks_nothing_more(monkeypatch, capsys, answers):
    # No key the user types can fix the log group: no breakdown instead of asking forever
    logging = {'cloudWatchConfig': {'logGroupName': 'bad group!'}}
    inputs = inputs_with(monkeypatch, FakeBedrock(logging_config=logging), answers)
    assert inputs._select_breakdown() is None
    assert 'log group' in capsys.readouterr().out


def test_interactive_breakdown_is_not_offered_without_logging(monkeypatch, caplog):
    caplog.set_level('INFO')
    inputs = inputs_with(monkeypatch, FakeBedrock(), [])
    assert inputs._select_breakdown() is None and 'not available' in caplog.text

    class Denied(FakeBedrock):
        def get_model_invocation_logging_configuration(self):
            raise aws_error('AccessDeniedException', 'GetModelInvocationLoggingConfiguration')
    inputs = inputs_with(monkeypatch, Denied(), [])
    caplog.clear()
    assert inputs._select_breakdown() is None and 'could not be read' in caplog.text
    # The advice is the missing permission (logging may well be on), not to enable logging
    assert 'bedrock:GetModelInvocationLoggingConfiguration' in caplog.text and 'enable model' not in caplog.text
