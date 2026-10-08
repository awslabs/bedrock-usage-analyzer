# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analyzer pipeline with fake AWS clients, and report generation."""

import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone

import pytest

from bedrock_usage_analyzer.core import analyzer as analyzer_module
from bedrock_usage_analyzer.core.analyzer import BedrockAnalyzer
from bedrock_usage_analyzer.core.output_generator import OutputGenerator, safe_filename
from bedrock_usage_analyzer.core.profile_fetcher import InferenceProfileFetcher

from conftest import HAIKU

GRANULARITY = {'1hour': 60, '1day': 300, '7days': 3600, '14days': 3600, '30days': 3600}


class FakeCloudWatch:
    """Returns one data point per minute for the last 10 minutes, per requested ModelId."""

    def __init__(self, per_model):
        self.per_model = per_model  # {model_id: (invocations, input, output)}
        self.dimensions = []

    def get_metric_data(self, MetricDataQueries, StartTime, EndTime, LabelOptions):
        results = []
        for q in MetricDataQueries:
            model = q['MetricStat']['Metric']['Dimensions'][0]['Value']
            self.dimensions.append(model)
            period = q['MetricStat']['Period']
            inv, inp, out = self.per_model.get(model, (0, 0, 0))
            values = {'invocations': inv, 'input_tokens': inp, 'output_tokens': out,
                      'throttles': 1, 'client_errors': 0, 'server_errors': 0, 'latency': 250}
            if model not in self.per_model:
                results.append({'Id': q['Id'], 'Timestamps': [], 'Values': []})
                continue
            end = EndTime.replace(second=0, microsecond=0)
            stamps = [end - timedelta(seconds=period * i) for i in range(1, 11)]
            stamps = [s for s in stamps if s >= StartTime]
            results.append({'Id': q['Id'], 'Timestamps': stamps, 'Values': [values[q['Id']]] * len(stamps)})
        return {'MetricDataResults': results}


class FakeQuotas:
    def get_service_quota(self, ServiceCode, QuotaCode):
        return {'Quota': {'Value': 1000000.0}}


@pytest.fixture
def analyzer(sydney_bedrock, monkeypatch):
    cw = FakeCloudWatch({'auapp000001': (2, 100, 50), f"au.{HAIKU}": (1, 10, 5)})
    clients = {'bedrock': sydney_bedrock, 'cloudwatch': cw, 'service-quotas': FakeQuotas()}
    monkeypatch.setattr(analyzer_module, 'create_client', lambda service, region=None, **_: clients[service])
    monkeypatch.setattr(analyzer_module, 'list_quota_codes', lambda region, **_: None)
    monkeypatch.setattr('bedrock_usage_analyzer.aws.servicequotas.regional_client', lambda region: clients['service-quotas'])
    a = BedrockAnalyzer('ap-southeast-2', GRANULARITY, profile_fetcher=InferenceProfileFetcher(sydney_bedrock))
    a.cw = cw
    return a


def reports(path):
    return sorted(os.listdir(path))


def test_full_pipeline_for_au_endpoint(analyzer, tmp_path):
    out = tmp_path / 'results'
    analyzer.analyze([{'model_id': HAIKU, 'profile_prefix': 'au'}], output_dir=str(out))
    files = reports(out)
    assert len(files) == 2 and files[0].startswith('au_anthropic_claude-haiku-4-5-20251001-v1_0-')
    data = json.loads((out / files[1]).read_text())
    assert data['endpoint'] == f"au.{HAIKU}"
    assert data['region_info']['display_name'] == 'Asia Pacific (Sydney)'
    agg = data['stats']['1hour']['__AGGREGATED__']
    # Both the system profile and the au application profile contribute
    assert agg['InputTokenCount']['sum'] == pytest.approx(10 * 100 + 10 * 10)
    names = {c['profile_name'] for c in data['contributions']['1hour']}
    assert names == {'team-a-au-haiku', f"au.{HAIKU}"}
    assert set(analyzer.cw.dimensions) == {'auapp000001', f"au.{HAIKU}"}


def test_application_profile_scope_only(analyzer, tmp_path):
    out = tmp_path / 'results'
    analyzer.analyze([{'model_id': HAIKU, 'profile_prefix': 'au', 'application_profile_ids': ['auapp000001']}],
                     output_dir=str(out))
    files = reports(out)
    assert 'app-auapp000001' in files[0]
    data = json.loads((out / files[1]).read_text())
    assert data['application_profile_scope'] == ['team-a-au-haiku']
    assert set(analyzer.cw.dimensions) == {'auapp000001'}
    html = (out / files[0]).read_text()
    assert 'Application inference profiles analyzed:</strong> team-a-au-haiku' in html


def test_a_custom_model_deployment_report(analyzer, tmp_path, caplog):
    deployment = 'arn:aws:bedrock:ap-southeast-2:111122223333:custom-model-deployment/dep0000001'
    analyzer.cw.per_model[deployment] = (3, 30, 15)
    caplog.set_level('INFO')
    out = tmp_path / 'results'
    # No base model: the deployment ID stands in for it, and no refresh can map limits
    analyzer.analyze([{'model_id': 'dep0000001', 'profile_prefix': 'custom', 'application_profile_ids': [deployment]}],
                     output_dir=str(out))
    files = reports(out)
    assert set(analyzer.cw.dimensions) == {deployment}
    assert '-deployment-dep0000001-' in files[0] and '-app-' not in files[0]
    assert json.loads((out / files[1]).read_text())['scope_label'] == 'Custom model deployments analyzed'
    html = (out / files[0]).read_text()
    assert 'Custom model deployments analyzed:</strong> dep0000001' in html
    assert 'No custom model deployment quotas are known for dep0000001' in caplog.text
    assert 'To map them' not in caplog.text


def test_an_imported_model_report(analyzer, tmp_path, caplog):
    imported = 'arn:aws:bedrock:ap-southeast-2:111122223333:imported-model/imp0000001'
    analyzer.cw.per_model[imported] = (3, 30, 15)
    analyzer.profile_fetcher.note_deployment_name(imported, 'my-qwen')
    caplog.set_level('INFO')
    out = tmp_path / 'results'
    analyzer.analyze([{'model_id': 'imp0000001', 'profile_prefix': 'imported', 'application_profile_ids': [imported]}],
                     output_dir=str(out))
    files = reports(out)
    assert set(analyzer.cw.dimensions) == {imported}
    assert files[0].startswith('imported-model_imp0000001-')
    data = json.loads((out / files[1]).read_text())
    assert data['scope_label'] == 'Imported models analyzed' and data['endpoint'] == 'imp0000001 (imported model)'
    assert data['stats']['1hour']['__AGGREGATED__']['InputTokenCount']['sum'] == pytest.approx(10 * 30)
    assert 'Imported models analyzed:</strong> my-qwen' in (out / files[0]).read_text()
    # No quotas exist for imported models: no deployment quota hint, and the run log says why
    assert 'custom model deployment quotas' not in caplog.text
    assert caplog.text.count('Custom Model Import models have no per-model token or request quotas') == 1


def test_the_same_imported_models_under_two_model_ids_are_one_target(analyzer, tmp_path):
    imported = 'arn:aws:bedrock:ap-southeast-2:111122223333:imported-model/imp0000001'
    analyzer.cw.per_model[imported] = (3, 30, 15)
    out = tmp_path / 'results'
    analyzer.analyze([{'model_id': 'imp0000001', 'profile_prefix': 'imported', 'application_profile_ids': [imported]},
                      {'model_id': HAIKU, 'profile_prefix': 'imported', 'application_profile_ids': [imported, imported]}],
                     output_dir=str(out))
    assert len(reports(out)) == 2  # one HTML and one JSON report
    assert reports(out)[0].startswith('imported-model_imp0000001-')  # one model, though passed twice


def test_an_imported_model_gets_no_quotas_whatever_its_model_id(analyzer, tmp_path, monkeypatch):
    imported = 'arn:aws:bedrock:ap-southeast-2:111122223333:imported-model/imp0000001'
    analyzer.cw.per_model[imported] = (3, 30, 15)
    monkeypatch.setattr(analyzer, '_load_quota_codes', lambda *a: pytest.fail('quotas looked up'))
    monkeypatch.setattr(analyzer, '_warn_other_deployments', lambda *a: pytest.fail('deployment note'))
    out = tmp_path / 'results'
    # An API caller naming a base model: still no limits, and named by the imported model
    analyzer.analyze([{'model_id': HAIKU, 'profile_prefix': 'imported', 'application_profile_ids': [imported]}],
                     output_dir=str(out))
    data = json.loads((out / reports(out)[1]).read_text())
    assert data['scope_label'] == 'Imported models analyzed' and not any(data['quotas'].values())
    assert data['model_id'] == 'imp0000001'
    # No max_tokens throttling note in the JSON either: the imported-model note instead
    assert 'throttling' not in data['disclaimers'] and 'Custom Model Import' in data['disclaimers']['imported']
    assert 'max_tokens' not in json.dumps(data['time_series'])


def test_imported_models_must_be_their_own_targets(analyzer, tmp_path):
    imported = 'arn:aws:bedrock:ap-southeast-2:111122223333:imported-model/imp0000001'
    deployment = 'arn:aws:bedrock:ap-southeast-2:111122223333:custom-model-deployment/dep0000001'
    for prefix, ids, message in (('custom', [deployment, imported], "with profile_prefix 'imported'"),
                                 ('imported', [deployment, imported], 'needs imported model ARNs'),
                                 ('imported', [], 'needs imported model ARNs')):
        with pytest.raises(ValueError, match=message):
            analyzer.analyze([{'model_id': HAIKU, 'profile_prefix': prefix, 'application_profile_ids': ids}],
                             output_dir=str(tmp_path / 'r'))
    assert not analyzer.cw.dimensions  # refused before any metric is read
    # The error names the target as the caller passed it, not an ID taken from a wrong ARN
    with pytest.raises(ValueError, match=re.escape(HAIKU)):
        analyzer.analyze([{'model_id': HAIKU, 'profile_prefix': 'imported', 'application_profile_ids': [deployment]}],
                         output_dir=str(tmp_path / 'r'))


def test_several_imported_models_in_one_report_are_named_by_their_ids(analyzer, tmp_path):
    ids = [f'arn:aws:bedrock:ap-southeast-2:111122223333:imported-model/imp{i}' for i in range(4)]
    for arn in ids:
        analyzer.cw.per_model[arn] = (1, 10, 5)
    out = tmp_path / 'results'
    analyzer.analyze([{'model_id': HAIKU, 'profile_prefix': 'imported', 'application_profile_ids': ids}],
                     output_dir=str(out))
    files = reports(out)
    assert files[0].startswith('imported-models-4-')
    data = json.loads((out / files[1]).read_text())
    assert data['endpoint'] == '4 imported models' and data['imported'] is True
    html = (out / files[0]).read_text()
    # Its own note, and the per-chart max_tokens note switched off
    assert 'no token or request quotas' in html and '&& !true)' in html


def test_two_endpoints_of_one_model_do_not_overwrite(analyzer, tmp_path):
    out = tmp_path / 'results'
    analyzer.analyze([{'model_id': HAIKU, 'profile_prefix': 'au'},
                      {'model_id': HAIKU, 'profile_prefix': 'global'},
                      {'model_id': HAIKU, 'profile_prefix': 'au'}], output_dir=str(out))
    assert len(reports(out)) == 4


def test_no_data_still_produces_report(analyzer, tmp_path):
    out = tmp_path / 'results'
    analyzer.analyze([{'model_id': HAIKU, 'profile_prefix': 'jp'}], output_dir=str(out))
    data = json.loads((out / reports(out)[1]).read_text())
    assert data['stats']['1hour']['__AGGREGATED__']['InputTokenCount']['sum'] == 0


def test_warns_when_app_profiles_are_under_another_endpoint(analyzer, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    analyzer.analyze([{'model_id': HAIKU, 'profile_prefix': 'apac'}], output_dir=str(tmp_path / 'r'))
    assert "no application inference profile of" in caplog.text
    assert "on 'au'" in caplog.text


def test_quota_urls_follow_partition(analyzer, monkeypatch):
    analyzer.region = 'us-gov-west-1'
    monkeypatch.setattr(analyzer_module, 'get_regional_profile_prefixes', lambda: ['us-gov'])
    quotas = analyzer._fetch_quotas(HAIKU, {'tpm': {'code': 'L-1', 'name': 'x'},
                                            'tpd': {'code': 'L-2', 'name': 'y'}}, 'us-gov')
    assert quotas['tpm']['url'].startswith('https://console.amazonaws-us-gov.com/')
    assert quotas['tpd']['value'] == 2000000.0   # regional profile TPD doubling


def base_data(**extra):
    now = datetime.now(timezone.utc)
    data = {
        'stats': {p: {'__AGGREGATED__': {}} for p in GRANULARITY},
        'time_series': {p: {} for p in GRANULARITY},
        'quotas': {'tpm': None, 'rpm': None, 'tpd': None},
        'profile_names': {},
        'contributions': {},
        'granularity_config': GRANULARITY,
        'end_time': now,
        'tz_offset': '+00:00',
        'region': 'us-west-2',
    }
    data.update(extra)
    return data


def test_report_escapes_markup_and_script_breakout(tmp_path):
    evil = '</script><img src=x onerror=alert(1)>'
    data = base_data(profile_names={'p1': evil},
                     contributions={'1hour': [{'profile_name': evil, 'profile_arn_id': 'id', 'profile_tags': {'k': evil},
                                               'tpm_p50': 1, 'tpm_p90': 1, 'tpm_avg': 1, 'rpm_p50': 1,
                                               'rpm_p90': 1, 'rpm_avg': 1, 'throttles': 0}]})
    OutputGenerator(str(tmp_path)).generate({evil: data})
    html_file = next(f for f in os.listdir(tmp_path) if f.endswith('.html'))
    html = (tmp_path / html_file).read_text()
    assert '<img src=x' not in html
    assert html.count('</script>') == html.count('<script')  # nothing closes a script early
    assert '&lt;/script&gt;' in html
    assert '\\u003c/script\\u003e' in html
    assert '/' not in html_file and '<' not in html_file


def test_report_without_console_for_partition(tmp_path, monkeypatch):
    # A partition without a public console: no links rather than wrong ones
    monkeypatch.setattr('bedrock_usage_analyzer.core.output_generator.get_service_quotas_console_url',
                        lambda region=None: None)
    quotas = {'tpm': {'value': 5.0, 'code': 'L-9', 'name': 'TPM', 'url': None}, 'rpm': None, 'tpd': None}
    OutputGenerator(str(tmp_path)).generate({'m': base_data(quotas=quotas)})
    html = (tmp_path / next(f for f in os.listdir(tmp_path) if f.endswith('.html'))).read_text()
    assert 'href="None"' not in html and '[L-9]' in html
    data = json.loads((tmp_path / next(f for f in os.listdir(tmp_path) if f.endswith('.json'))).read_text())
    assert data['disclaimers']['quota_mapping'].endswith('console.')


def test_govcloud_report_labels_partition(tmp_path):
    quotas = {'tpm': {'value': 5.0, 'code': 'L-9', 'name': 'TPM',
                      'url': 'https://console.amazonaws-us-gov.com/x'}, 'rpm': None, 'tpd': None}
    OutputGenerator(str(tmp_path)).generate({'m': base_data(region='us-gov-west-1', quotas=quotas)})
    html = (tmp_path / next(f for f in os.listdir(tmp_path) if f.endswith('.html'))).read_text()
    assert 'AWS GovCloud (US-West)' in html and 'AWS GovCloud (US)' in html
    assert 'https://console.amazonaws-us-gov.com/servicequotas/home?region=us-gov-west-1' in html
    assert 'integrity="sha384-' in html


def test_no_quota_disclaimer_when_no_quotas(tmp_path):
    OutputGenerator(str(tmp_path)).generate({'m': base_data()})
    data = json.loads((tmp_path / next(f for f in os.listdir(tmp_path) if f.endswith('.json'))).read_text())
    assert 'quota_mapping' not in data['disclaimers']


def test_missing_end_time_is_handled(tmp_path):
    OutputGenerator(str(tmp_path)).generate({'m': base_data(end_time=None)})
    assert len(os.listdir(tmp_path)) == 2


@pytest.mark.parametrize('label,expected', [
    ('us.amazon.nova-pro-v1:0', 'us_amazon_nova-pro-v1_0'),
    ('../../etc/passwd', 'etc_passwd'),
    ('', 'report'),
    ('a/b\\c', 'a_b_c'),
])
def test_safe_filename(label, expected):
    assert safe_filename(label) == expected


@pytest.mark.parametrize('value,expected', [
    ('https://console.aws.amazon.com/x', 'https://console.aws.amazon.com/x'),
    ('javascript:alert(1)', ''),
    ('data:text/html,x', ''),
    ('http://example.com', ''),
    (None, ''),
])
def test_https_url_filter(value, expected):
    from bedrock_usage_analyzer.core.output_generator import https_url
    assert https_url(value) == expected


def test_report_drops_non_https_quota_links(tmp_path):
    quotas = {'tpm': {'value': 5.0, 'code': 'L-9', 'name': 'TPM', 'url': 'javascript:alert(1)'},
              'rpm': None, 'tpd': None}
    OutputGenerator(str(tmp_path)).generate({'m': base_data(quotas=quotas)})
    html = (tmp_path / next(f for f in os.listdir(tmp_path) if f.endswith('.html'))).read_text()
    # No link is rendered; the value only survives as inert JSON data for the charts
    assert 'href="javascript:' not in html and '[L-9]' in html


def test_concurrent_quota_is_fetched_and_missing_codes_explained(analyzer, caplog, monkeypatch):
    import logging
    caplog.set_level(logging.INFO)

    class Quotas:
        def get_service_quota(self, ServiceCode, QuotaCode):
            if QuotaCode == 'L-GONE':
                from botocore.exceptions import ClientError
                raise ClientError({'Error': {'Code': 'NoSuchResourceException', 'Message': 'x'}}, 'GetServiceQuota')
            return {'Quota': {'Value': 7.0}}

        def get_aws_default_service_quota(self, ServiceCode, QuotaCode):
            from botocore.exceptions import ClientError
            raise ClientError({'Error': {'Code': 'NoSuchResourceException', 'Message': 'x'}}, 'GetAWSDefaultServiceQuota')

    monkeypatch.setattr('bedrock_usage_analyzer.aws.servicequotas.regional_client', lambda region: Quotas())
    quotas = analyzer._fetch_quotas(HAIKU, {'concurrent': {'code': 'L-C', 'name': 'c'},
                                            'tpm': {'code': 'L-GONE', 'name': 't'}}, None)
    assert quotas['concurrent']['value'] == 7.0 and quotas['tpm'] is None
    assert "does not exist in ap-southeast-2; shown without this limit" in caplog.text


def test_report_is_well_formed_utf8(tmp_path):
    """Charset declared first (symbols render in every browser) and every tag balanced."""
    import html.parser
    OutputGenerator(str(tmp_path)).generate({'m': base_data()})
    src = (tmp_path / next(f for f in os.listdir(tmp_path) if f.endswith('.html'))).read_text(encoding='utf-8')
    assert src.index('<meta charset="utf-8">') < src.index('<title>')
    void = {'meta', 'link', 'br', 'img', 'input', 'hr'}

    class Checker(html.parser.HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack, self.errors = [], []

        def handle_starttag(self, tag, attrs):
            if tag not in void:
                self.stack.append(tag)

        def handle_endtag(self, tag):
            if tag in void:
                return
            if self.stack and self.stack[-1] == tag:
                self.stack.pop()
            else:
                self.errors.append(tag)

    checker = Checker()
    checker.feed(src)
    assert checker.errors == [] and checker.stack == []


def test_unknown_source_endpoint_label(analyzer, tmp_path):
    out = tmp_path / 'results'
    analyzer.analyze([{'model_id': HAIKU, 'profile_prefix': 'unknown', 'application_profile_ids': ['auapp000001']}],
                     output_dir=str(out))
    data = json.loads((out / reports(out)[1]).read_text())
    assert data['endpoint'] == f"{HAIKU} (source endpoint unknown)"
    assert not reports(out)[0].startswith('unknown.')


def test_chart_quota_note_is_built_without_innerhtml():
    template = open(os.path.join(os.path.dirname(__file__), '..', 'src', 'bedrock_usage_analyzer',
                                 'templates', 'report.html'), encoding='utf-8').read()
    assert 'quotaInfo.url}' not in template and 'quotaInfo.name}' not in template
    assert "quotaInfo.url.startsWith('https://')" in template


def test_quota_without_value_is_skipped(analyzer, monkeypatch):
    class Quotas:
        def get_service_quota(self, ServiceCode, QuotaCode):
            return {'Quota': {'QuotaName': 'x'}}
    monkeypatch.setattr('bedrock_usage_analyzer.aws.servicequotas.regional_client', lambda region: Quotas())
    assert analyzer._fetch_quotas(HAIKU, {'tpm': {'code': 'L-1', 'name': 'n'}}, None)['tpm'] is None


def test_report_json_keeps_discovery_order(tmp_path):
    from bedrock_usage_analyzer.core.output_generator import OutputGenerator
    env = OutputGenerator(str(tmp_path))._env
    rendered = env.from_string('{{ d|tojson }}').render(d={'us.model': 1, 'abc000000001': 2})
    assert rendered.index('us.model') < rendered.index('abc000000001')


def test_quotas_come_from_the_region_listing(analyzer, monkeypatch):
    monkeypatch.setattr(analyzer_module, 'list_quota_codes',
                        lambda region, **_: {'L-1': {'QuotaCode': 'L-1', 'Value': 123.0}})
    asked = []
    monkeypatch.setattr('bedrock_usage_analyzer.aws.servicequotas.check_quota', lambda code, region, client=None: asked.append(code) or ('missing', None))
    analyzer._use_quota_listing = True                        # set by analyze() for many codes
    quotas = analyzer._fetch_quotas(HAIKU, {'tpm': {'code': 'L-1', 'name': 'x'}, 'rpm': {'code': 'L-2', 'name': 'y'}})
    assert quotas['tpm']['value'] == 123.0 and asked == ['L-2']                # one lookup, for the unlisted code


def test_failed_quota_lookup_is_retried_by_the_next_target(analyzer, monkeypatch):
    from bedrock_usage_analyzer.aws.servicequotas import QUOTA_ERROR, QUOTA_OK
    answers = [(QUOTA_ERROR, None), (QUOTA_OK, {'Value': 7.0})]
    calls = []
    monkeypatch.setattr(analyzer_module, 'lookup_quota',
                        lambda code, region, listings, use_listing=False: calls.append(code) or answers.pop(0))
    codes = {'tpm': {'code': 'L-1', 'name': 'x'}}
    assert analyzer._fetch_quotas(HAIKU, codes)['tpm'] is None               # throttled: no limit
    assert analyzer._fetch_quotas(HAIKU, codes)['tpm']['value'] == 7.0       # tried again
    assert calls == ['L-1', 'L-1']


def test_report_names_tell_large_profile_sets_apart():
    a = BedrockAnalyzer._file_label('us.x', ['a1', 'a2', 'a3', 'a4'])
    b = BedrockAnalyzer._file_label('us.x', ['b1', 'b2', 'b3', 'b4'])
    assert a != b and a.startswith('us.x-app-4profiles-')
    assert BedrockAnalyzer._file_label('us.x', ['a4', 'a3', 'a2', 'a1']) == a       # order does not matter
