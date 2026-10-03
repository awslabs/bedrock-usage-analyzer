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


def _canonical(version: str) -> str:
    """'3.0' and '3' are the same generation."""
    return version[:-2] if version.endswith('.0') else version


def quota_versions(quota_name: str) -> Set[str]:
    """Stand-alone version numbers in a quota name."""
    return {_canonical(v) for v in _NAME_VERSION.findall(quota_name)}


def mapping_conflict(model_id: str, endpoint_type: str, quota_name: Optional[str],
                     regional_prefixes) -> Optional[str]:
    """Return why ``quota_name`` cannot be a quota of this model endpoint, or None."""
    if not quota_name:
        return None
    # The shared tokens-per-day quota ('... (doubled for cross-region calls)') is also the
    # on-demand limit, so that note does not make it a cross-region quota
    name = quota_name.lower().replace('(doubled for cross-region calls)', '')
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
    version = model_version(model_id)
    version = _canonical(version) if version else None
    versions = quota_versions(quota_name)
    if version and versions and version not in versions:
        return f"quota is for version {'/'.join(sorted(versions))}, model is {version}"
    return None


def scrub_conflicting(model_id: str, endpoint_type: str, quotas: Optional[dict], regional_prefixes):
    """Null, in place, the codes in ``quotas`` that contradict this model endpoint.

    Returns [(metric, quota, reason)] for each code removed, so callers can log it.
    """
    removed = []
    for metric, quota in list((quotas or {}).items()):
        if isinstance(quota, dict):
            reason = mapping_conflict(model_id, endpoint_type, quota.get('name'), regional_prefixes)
            if reason:
                removed.append((metric, quota, reason))
                quotas[metric] = None
    return removed
