# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Troubleshooting hints for common AWS errors, aware of the partition in use."""

from typing import Optional

from bedrock_usage_analyzer.utils.partition import (
    COMMERCIAL,
    TOKEN_REJECTION_CODES,
    get_partition_display_name,
    get_partition_for_region,
)

# The STS token-rejection codes plus other signs of missing or stale credentials
_CREDENTIAL_MARKERS = tuple(code.lower() for code in TOKEN_REJECTION_CODES) + (
    'security token', 'unable to locate credentials', 'nocredentials', 'expiredtoken',
    'expired token', 'credentials')
_ACCESS_MARKERS = ('accessdenied', 'access denied', 'not authorized', 'unauthorizedoperation')
_NETWORK_MARKERS = ('could not connect', 'endpoint url', 'connecttimeout', 'read timeout',
                    'connection was closed', 'name or service not known')


def _error_text(error: Exception) -> str:
    return f"{type(error).__name__} {error}"


def is_access_denied(error: Exception) -> bool:
    """True for permission errors, which do not go away when the call is retried."""
    text = _error_text(error).lower()
    return any(m in text for m in _ACCESS_MARKERS)


def troubleshooting_hint(error: Exception, region: Optional[str] = None) -> Optional[str]:
    """Return a short hint for ``error``, or None when there is nothing useful to add."""
    text = _error_text(error).lower()
    partition = get_partition_for_region(region) if region else None
    partition_name = get_partition_display_name(partition) if partition else None

    if any(m in text for m in _CREDENTIAL_MARKERS):
        command = f"aws sts get-caller-identity --region {region}" if region else "aws sts get-caller-identity"
        hint = f"Check your AWS credentials: run '{command}'."
        if partition and partition != COMMERCIAL:
            hint += (f" {partition_name} uses separate accounts and credentials from"
                     f" commercial AWS; use a profile for that partition (AWS_PROFILE=...)"
                     f" and pass --region {region}.")
        return hint
    if is_access_denied(error):
        return ("The credentials lack a required permission. See the IAM permissions"
                " section of the README for the actions this tool needs.")
    if any(m in text for m in _NETWORK_MARKERS):
        target = f" for {region}" if region else ""
        return (f"Could not reach the AWS endpoint{target}. Check network or proxy access,"
                " and that the region name is correct.")
    return None
