# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bedrock Usage Analyzer - Token usage statistics for Amazon Bedrock"""

try:
    from importlib.metadata import version
    __version__ = version("bedrock-usage-analyzer")
except Exception:
    # Fallback for development/testing
    __version__ = "0.5.0-beta-dev"

__all__ = ["__version__"]
