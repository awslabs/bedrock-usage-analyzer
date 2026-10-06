# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic checks that veto wrong quota mappings.

The LLM that maps models to quotas can pick a quota of a neighbouring model
('Claude Sonnet 4.6' for Sonnet 4) or of another endpoint type ('Global
cross-region ...' for a 'us.' profile). These rules catch both from the quota
name alone. They only reject clear contradictions; when a rule cannot tell,
it lets the mapping through.
"""

import re
from typing import Optional, Set

from bedrock_usage_analyzer.utils.yaml_handler import CUSTOM_ENDPOINT

_VERSION_TOKEN = re.compile(r'^\d{1,2}(?:\.\d{1,2})?$')
_LETTERS_THEN_VERSION = re.compile(r'^([a-z]+)(\d{1,2}(?:\.\d{1,2})?)$')
_API_VERSION_TOKEN = re.compile(r'^v\d+(?:\.\d+)?$')
_SIZE_TOKEN = re.compile(r'^\d+(?:\.\d+)?[bmk]$')
# A version in a quota name: '4.6' in 'Sonnet 4.6', '6' in 'GPT-6 Sol'; not '20B', 'V2' or 'K2.5'
_NAME_VERSION = re.compile(r'(?<![\w.])(\d{1,2}(?:\.\d{1,2})?)(?![\w.])')


def model_version(model_id: str) -> Optional[str]:
    """The model generation encoded in a model ID, e.g. '4.6' for claude-sonnet-4-6.

    'anthropic.claude-sonnet-4-20250514-v1:0' -> '4'
    'anthropic.claude-3-5-sonnet-20241022-v2:0' -> '3.5'
    'meta.llama3-2-1b-instruct-v1:0' -> '3.2'
    'openai.gpt-6.1-sol' -> '6.1'
    'amazon.nova-lite-v1:0' -> None
    """
    rest = model_id.split('.', 1)[1] if '.' in model_id else model_id
    rest = rest.lower().split(':', 1)[0]
    tokens = re.split(r'[-_]', rest)
    run = []
    for i, token in enumerate(tokens):
        if _SIZE_TOKEN.match(token):
            # Versions precede sizes: 'llama3-2-1b', 'gpt-oss-20b-1' (the trailing 1 is a revision)
            break
        if _API_VERSION_TOKEN.match(token):
            if run:
                break
            # 'rerank-v3-5' is version 3.5; a trailing 'v1'/'v2' is the API version
            if i + 1 < len(tokens) and _VERSION_TOKEN.match(tokens[i + 1]):
                run.append(token[1:])
            continue
        match = _LETTERS_THEN_VERSION.match(token)
        if match and not run:
            run.append(match.group(2))
            continue
        if _VERSION_TOKEN.match(token):
            run.append(token)
        elif run:
            break
    return '.'.join(run) if run else None


def _version_inside_name(model_id: str) -> bool:
    """True when the ID's version stands between name words, as in 'nova-2-5-sonic'.

    Such a version is part of the model's name ('Nova 2 Sonic'), so a quota that names the
    family without a version ('Amazon Nova Sonic') is another model's. A trailing version
    ('pegasus-1-2-v1:0') or one glued to a word ('qwen3', 'kimi-k2.5') is often left out of
    quota names, so it says nothing.
    """
    rest = model_id.split('.', 1)[1] if '.' in model_id else model_id
    tokens = re.split(r'[-_]', rest.lower().split(':', 1)[0])
    for i in range(1, len(tokens)):
        if _VERSION_TOKEN.match(tokens[i]) and tokens[i - 1].isalpha():
            j = i
            while j < len(tokens) and _VERSION_TOKEN.match(tokens[j]):
                j += 1
            return j < len(tokens) and tokens[j].isalpha() and not _API_VERSION_TOKEN.match(tokens[j])
    return False


def _canonical(version: str) -> str:
    """'3.0' and '3' are the same generation."""
    return version[:-2] if version.endswith('.0') else version


def quota_versions(quota_name: str) -> Set[str]:
    """Stand-alone version numbers in a quota name."""
    return {_canonical(v) for v in _NAME_VERSION.findall(quota_name)}


# What each metric's quota name says (every bundled mapping matches)
METRIC_KEYWORDS = {'tpm': 'tokens per minute', 'rpm': 'requests per minute',
                   'tpd': 'tokens per day', 'concurrent': 'concurrent'}


def measures_metric(metric: str, quota_name: Optional[str]) -> bool:
    """True when ``quota_name`` is a quota of ``metric`` (an RPM quota is not a TPM one)."""
    keyword = METRIC_KEYWORDS.get(metric)
    return not keyword or keyword in (quota_name or '').lower()


def mapping_conflict(model_id: str, endpoint_type: str, quota_name: Optional[str],
                     regional_prefixes) -> Optional[str]:
    """Return why ``quota_name`` cannot be a quota of this model endpoint, or None."""
    if not quota_name:
        return None
    # The shared tokens-per-day quota ('... (doubled for cross-region calls)') is also the
    # on-demand limit, so that note does not make it a cross-region quota
    name = quota_name.lower().replace('(doubled for cross-region calls)', '')
    custom_quota = 'custom model' in name
    if endpoint_type == CUSTOM_ENDPOINT and not custom_quota:
        return "not a custom model deployment quota for a custom model endpoint"
    if endpoint_type != CUSTOM_ENDPOINT and custom_quota:
        return "custom model deployment quota for a foundation model endpoint"
    if endpoint_type in regional_prefixes and 'global' in name:
        return "global quota for a regional (geographic) cross-region endpoint"
    if endpoint_type == 'base' and 'cross-region' in name:
        return "cross-region quota for an on-demand endpoint"
    if endpoint_type == 'global' and 'global' not in name:
        return "non-global quota for a global endpoint"
    if 'context length' in name:
        # e.g. '... Claude Sonnet 4.5 V1 1M Context Length': a separate limit for long-context
        # requests on the same model ID; the standard quota is the one usage is measured against
        return "long-context variant quota"
    if 'latency-optimized' in name:
        # e.g. 'On-Demand, latency-optimized model inference tokens per minute for Amazon Nova
        # Pro V1': a separate limit for requests made with performanceConfig latency=optimized
        return "latency-optimized inference quota"
    version = model_version(model_id)
    version = _canonical(version) if version else None
    versions = quota_versions(quota_name)
    if version and versions and version not in versions:
        return f"quota is for version {'/'.join(sorted(versions))}, model is {version}"
    if version and not versions and _version_inside_name(model_id):
        return f"quota names no version, model is {version}"
    return None


def slot_conflict(model_id: str, endpoint_type: str, metric: str, quota_name: Optional[str],
                  regional_prefixes) -> Optional[str]:
    """Why a saved ``metric`` slot cannot hold ``quota_name``, or None: it contradicts the
    model endpoint, or measures another metric (an RPM quota saved as the TPM one)."""
    reason = mapping_conflict(model_id, endpoint_type, quota_name, regional_prefixes)
    if reason or not quota_name or metric not in METRIC_KEYWORDS or measures_metric(metric, quota_name):
        return reason
    # Only a name that names another metric is a conflict: a name without any metric keyword
    # (hand-edited, or shortened) says nothing about what it measures
    other = [m for m in METRIC_KEYWORDS if m != metric and measures_metric(m, quota_name)]
    return f"a {METRIC_KEYWORDS[other[0]]} quota saved as {metric}" if other else None


def scrub_conflicting(model_id: str, endpoint_type: str, quotas: Optional[dict], regional_prefixes):
    """Null, in place, the codes in ``quotas`` that contradict this model endpoint or metric.

    Returns [(metric, quota, reason)] for each code removed, so callers can log it.
    """
    removed = []
    for metric, quota in list((quotas or {}).items()):
        if isinstance(quota, dict):
            reason = slot_conflict(model_id, endpoint_type, metric, quota.get('name'), regional_prefixes)
            if reason:
                removed.append((metric, quota, reason))
                quotas[metric] = None
    return removed
