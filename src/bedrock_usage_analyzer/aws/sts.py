# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AWS STS (Security Token Service) operations"""

from typing import Optional

from bedrock_usage_analyzer.utils.partition import resolve_caller_identity


def get_account_id(region: Optional[str] = None) -> str:
    """Get current AWS account ID (cached; same STS fallback as the account check).

    Args:
        region: Region whose STS endpoint to try first; required for GovCloud or
            China credentials when no region is configured.

    Raises:
        Exception: If unable to get account ID
    """
    return resolve_caller_identity(region)['Account']
