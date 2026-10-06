# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Usage breakdown by caller from the model invocation logs: query building, principal
normalization, chunked Logs Insights queries, grouping, remainder rows and the report."""

import json
import os
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
        rows = [r for r in self.rows if query['start'] <= r['minute'].timestamp() <= query['end']]
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
    ('tag:Cost Center', 'tag', 'Cost Center'),
])
def test_breakdown_parsing(value, kind, key):
    parsed = Breakdown.parse(value)
    assert (parsed.kind, parsed.key) == (kind, key)


@pytest.mark.parametrize('value', ['owner', 'tag', 'tag:', 'tag:a"b', 'tag:   ', 'metadata:x y', 'metadata:x|z',
                                   'principal:x', 'session:y'])
def test_invalid_breakdowns_are_refused(value):
    with pytest.raises(BreakdownError):
        Breakdown.parse(value)


def test_principals_and_log_group_are_validated():
    parsed = Breakdown.parse('principal', [ROLE_ARN, ' user/bob ', ''], '/aws/bedrock/logs')
    assert parsed.principals == ('role/OrdersService', 'user/bob') and parsed.log_group == '/aws/bedrock/logs'
    with pytest.raises(BreakdownError):
        Breakdown.parse('principal', ['role/x"; drop'])
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
    (USER_ARN, 'user/ops/alice'),
    # A role's path is not in its assumed-role ARNs, so it is dropped
    (f"arn:aws:iam::{ACCOUNT}:role/aws-reserved/sso.amazonaws.com/AWSReservedSSO_Admin_ab", 'role/AWSReservedSSO_Admin_ab'),
    (f"arn:aws:iam::{ACCOUNT}:role/Direct", 'role/Direct'),
    (f"arn:aws:iam::{ACCOUNT}:root", f"arn:aws:iam::{ACCOUNT}:root"),
    (f"arn:aws:sts::{ACCOUNT}:federated-user/bob", f"arn:aws:sts::{ACCOUNT}:federated-user/bob"),
    ('role/Already', 'role/Already'), ('role/service-role/X', 'role/X'), ('user/ops/bob', 'user/ops/bob'),
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
    assert forms[f"arn:aws:bedrock:{REGION}::inference-profile/{US_HAIKU}"] == US_HAIKU
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
    assert 'identity.arn as session' in build_query([US_HAIKU], Breakdown.parse('session'))
    assert '`requestMetadata.team` as meta' in build_query([US_HAIKU], Breakdown.parse('metadata:team'))
    # Keys with '-' (or : / = + @) are read as one field name
    assert '`requestMetadata.cost-center` as meta' in build_query([US_HAIKU], Breakdown.parse('metadata:cost-center'))
    assert ', principal\n' in build_query([US_HAIKU], Breakdown.parse('tag:team')) + '\n'


def test_long_model_id_lists_are_split_across_queries(monkeypatch):
    monkeypatch.setattr(il, 'MAX_QUERY_LENGTH', 700)
    ids = [f"us.vendor.model-{n:03d}-v1:0" for n in range(40)]
    queries = il.query_batches(ids, Breakdown())
    assert len(queries) > 1 and all(len(q) <= 700 for q in queries)
    # Every spelling is in exactly one query
    assert sorted(i for i in ids for q in queries if f'"{i}"' in q) == sorted(ids)
    assert il.query_batches([US_HAIKU], Breakdown()) == [build_query([US_HAIKU], Breakdown())]

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
    assert sorted(r['principal'] for r in rows) == ['role/OrdersService', 'user/ops/alice']
    alice = next(r for r in rows if r['principal'] == 'user/ops/alice')
    assert (alice['cw_id'], alice['input'], alice['output'], alice['requests']) == (HAIKU, 10.0, 0.0, 2.0)
    assert alice['minute'].tzinfo is not None
    # Windows do not overlap: each query ends one second before the next starts
    assert all(q['end'] - q['start'] == 86399 for q in logs.queries)


def test_a_full_window_is_split_until_it_fits(monkeypatch):
    monkeypatch.setattr(il, 'MAX_ROWS', 3)
    stamps = [END - timedelta(minutes=m) for m in range(5, 45, 5)]  # 8 rows, 5 minutes apart
    logs = FakeLogs([row(s, 'assumed-role/A') for s in stamps])
    rows = fetcher_for(logs).fetch({US_HAIKU: US_HAIKU}, Breakdown(), END - timedelta(hours=1), END)
    assert len(rows) == 8 and len(logs.queries) > 1


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
    logs = FakeLogs(groups=[{'logGroupName': '/bedrock/logs', 'creationTime': int(created.timestamp() * 1000)},
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


def test_principal_tags_reads_roles_and_users_and_keeps_the_first_denial():
    iam = FakeIam(role_tags={'OrdersService': {'team': 'orders'}}, user_tags={'alice': {'team': 'ops'}}, deny={'Locked'})
    tags, error = principal_tags(iam, ['role/OrdersService', 'user/ops/alice', 'role/Locked',
                                       f"arn:aws:iam::{ACCOUNT}:root", 'role/OrdersService'])
    assert tags == {'role/OrdersService': {'team': 'orders'}, 'user/ops/alice': {'team': 'ops'}}
    assert error is not None and ('user', 'alice') in iam.calls and len(iam.calls) == 3


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
    assert names == ['role/OrdersService', 'user/ops/alice', UNATTRIBUTED]
    orders = hour['rows'][0]
    assert orders['tokens'] == 1000 and orders['requests'] == 2 and orders['tags'] == {'team': 'orders'}
    assert orders['share_tokens'] == pytest.approx(1000 / 2000) and orders['via'] == [US_HAIKU]
    assert orders['tpm_max'] == 1000 and orders['rpm_max'] == 2
    remainder = hour['rows'][-1]
    assert remainder['tokens'] == 700 and remainder['requests'] == 1
    assert sum(r['share_tokens'] for r in hour['rows']) == pytest.approx(1.0)
    assert hour['total_tokens'] == 2000 and not hour['partial']
    assert set(section['time_series']['1hour']) == set(names)
    assert section['coverage']['end'] == END.isoformat() and section['log_group'] == '/bedrock/logs'


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


def test_tags_are_read_only_for_the_selected_principals():
    logs = FakeLogs([row(T1, 'assumed-role/OrdersService'), row(T1, 'assumed-role/Billing')])
    iam = FakeIam()
    builder_for(Breakdown.parse('principal', ['role/OrdersService']), logs, iam).prepare([US_HAIKU], END, 1)
    assert iam.calls == [('role', 'OrdersService')]


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


def test_cloudwatch_minutes_after_the_logs_were_read_are_not_unattributed():
    # The logs were read up to END - 5 min; CloudWatch, read later, also has END - 2 min
    logs = FakeLogs([row(T1, 'assumed-role/A', i=100, o=0, n=1)])
    builder = builder_for(Breakdown(), logs)
    builder.prepare([US_HAIKU], END - timedelta(minutes=5), 1)
    late = END - timedelta(minutes=2)
    section = builder.section([US_HAIKU], {}, {US_HAIKU: cloudwatch({T1: (100, 0, 1), late: (900, 0, 3)})},
                              GRANULARITY, ['1hour'])
    hour = section['periods']['1hour']
    assert [r['name'] for r in hour['rows']] == ['role/A'] and hour['total_tokens'] == 100
    # The section says where the breakdown stops
    assert section['coverage']['end'] == (END - timedelta(minutes=5)).isoformat()


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
    # CloudWatch usage from before the logs start is not counted as unattributed
    assert section['periods']['30days']['total_tokens'] == 150


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


def test_a_given_log_group_skips_the_logging_configuration():
    bedrock = FakeBedrock()
    builder = builder_for(Breakdown.parse('principal', log_group='/bedrock/logs'), FakeLogs(), bedrock=bedrock)
    assert builder.prepare([US_HAIKU], END, 1) is None and bedrock.calls == []


# ------------------------------------------------------------------ report and CLI

def test_report_renders_the_breakdown_escaped(analyzer, tmp_path, monkeypatch):
    from bedrock_usage_analyzer.core import breakdown as breakdown_module
    hostile = 'assumed-role/<img src=x onerror=alert(1)>'
    logs = FakeLogs([row(datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=3),
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
    (['--log-group', '/g'], ('principal', None, ())),
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
    assert seen['analyzer']['breakdown'] is b and seen['analyzer']['account'] == ACCOUNT and seen['ran']


def test_an_invalid_cli_breakdown_exits(monkeypatch, caplog):
    from bedrock_usage_analyzer import __main__ as cli
    monkeypatch.setattr('sys.argv', ['bua', 'analyze', '-r', REGION, '-m', HAIKU, '--breakdown', 'owner'])
    with pytest.raises(SystemExit):
        cli.main()
    assert 'unknown breakdown' in caplog.text


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
])
def test_interactive_breakdown_choice(monkeypatch, answers, expected):
    inputs = inputs_with(monkeypatch, FakeBedrock(logging_config=LOGGING), answers)
    chosen = inputs._select_breakdown()
    assert (None if chosen is None else (chosen.kind, chosen.key)) == expected
    if chosen:
        assert chosen.log_group == '/bedrock/logs'


def test_interactive_breakdown_is_not_offered_without_logging(monkeypatch, caplog):
    caplog.set_level('INFO')
    inputs = inputs_with(monkeypatch, FakeBedrock(), [])
    assert inputs._select_breakdown() is None and 'not available' in caplog.text

    class Denied(FakeBedrock):
        def get_model_invocation_logging_configuration(self):
            raise aws_error('AccessDeniedException', 'GetModelInvocationLoggingConfiguration')
    inputs = inputs_with(monkeypatch, Denied(), [])
    assert inputs._select_breakdown() is None and 'could not be read' in caplog.text
