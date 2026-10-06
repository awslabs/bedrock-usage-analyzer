# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bedrock Usage Analyzer - Token usage statistics for Amazon Bedrock"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bedrock-usage-analyzer")
except PackageNotFoundError:
    # Running from a source tree without installed package metadata
    __version__ = "unknown"

__all__ = ["__version__"]
