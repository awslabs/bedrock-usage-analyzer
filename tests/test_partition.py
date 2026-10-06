#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for partition detection and cross-partition support (offline: no AWS calls)"""

import os
import sys

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bedrock_usage_analyzer.utils import partition as partition_module  # noqa: E402
from bedrock_usage_analyzer.utils.partition import (  # noqa: E402
    build_arn,
    get_console_domain,
    get_service_quota_url,
    is_china_region,
    is_govcloud_region,
    partition_for_region,
)


@pytest.fixture(autouse=True)
def no_sts(monkeypatch):
    """Every partition here comes from the region name: STS must not be called."""
    monkeypatch.setattr(partition_module, '_cached_partition', None)
    monkeypatch.setattr(partition_module.boto3, 'client',
                        lambda *a, **k: pytest.fail('STS must not be called for a known region'))


@pytest.mark.parametrize('region,partition,console', [
    ('us-west-2', 'aws', 'console.aws.amazon.com'),
    ('us-gov-west-1', 'aws-us-gov', 'console.amazonaws-us-gov.com'),
    ('cn-north-1', 'aws-cn', 'console.amazonaws.cn'),
])
def test_arn_console_and_quota_url_follow_the_region(region, partition, console):
    assert partition_for_region(region) == partition
    assert build_arn('bedrock', region, '', 'foundation-model/amazon.titan-text-express-v1') == \
        f"arn:{partition}:bedrock:{region}::foundation-model/amazon.titan-text-express-v1"
    assert get_console_domain(region) == console
    assert get_service_quota_url(region, 'bedrock', 'L-1234') == \
        f"https://{region}.{console}/servicequotas/home/services/bedrock/quotas/L-1234"


def test_region_detection():
    for region in ('us-gov-west-1', 'us-gov-east-1'):
        assert is_govcloud_region(region)
    for region in ('cn-north-1', 'cn-northwest-1'):
        assert is_china_region(region)
    for region in ('us-west-2', 'us-east-1', 'eu-west-1'):
        assert not is_govcloud_region(region) and not is_china_region(region)


def test_credentials_partition_comes_from_the_caller_identity(monkeypatch):
    class FakeSts:
        def get_caller_identity(self):
            return {'Arn': 'arn:aws-us-gov:iam::123456789012:user/a', 'Account': '123456789012'}

    monkeypatch.setattr(partition_module.boto3, 'client', lambda *a, **k: FakeSts())
    assert partition_module.get_partition() == 'aws-us-gov'
    assert partition_module.get_account_id() == '123456789012'


def test_failed_detection_is_not_cached(monkeypatch):
    def failing(*a, **k):
        raise RuntimeError('no credentials')

    monkeypatch.setattr(partition_module.boto3, 'client', failing)
    assert partition_module.get_partition() == 'aws'
    assert partition_module._cached_partition is None      # retried once credentials work
