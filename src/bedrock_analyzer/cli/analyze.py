# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Main CLI entry point for Bedrock usage analysis"""

import logging
import sys
import traceback

from bedrock_analyzer.core.user_inputs import UserInputs
from bedrock_analyzer.core.analyzer import BedrockAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for bedrock-analyze command"""
    try:
  
        user_inputs = UserInputs()
        user_inputs.collect()
        
        analyzer = BedrockAnalyzer(user_inputs.region, user_inputs.granularity_config)
        analyzer.analyze(user_inputs.models)
        
        logger.info(f"\nCompleted! Check the 'results' directory for output files.")
        
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
