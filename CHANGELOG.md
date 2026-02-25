# Changelog

All notable changes to the Bedrock Usage Analyzer will be documented in this file.

## [Unreleased]

### Added
- **AWS GovCloud and Multi-Partition Support**: The tool now automatically detects and supports AWS GovCloud, China, ISO, and ISO-B partitions in addition to commercial AWS
  - Automatic partition detection from STS GetCallerIdentity
  - Partition-aware ARN construction
  - Correct console URLs for each partition (console.amazonaws-us-gov.com for GovCloud, etc.)
  - Region detection utilities for GovCloud and China regions
  - See [GOVCLOUD_SUPPORT.md](GOVCLOUD_SUPPORT.md) for detailed documentation

### Changed
- ARN construction now uses detected partition instead of hardcoded `aws` partition
- Console URLs in reports and output now use correct domain based on partition
- Service quota URLs now point to correct console domain for the partition

### Technical Details
- New module: `src/bedrock_usage_analyzer/utils/partition.py`
- Updated modules: `aws/bedrock.py`, `aws/sts.py`, `core/analyzer.py`, `core/output_generator.py`, `templates/report.html`

## [0.5.0-beta] - 2025-02-25

### Added
- CLI arguments for non-interactive usage (--region, --model-id, --granularity, --output-dir)
- -y/--yes flag to skip account confirmation prompts
- Support for scripted/automated workflows and CI/CD pipelines

### Changed
- Interactive prompts can now be skipped by providing CLI arguments
- Account confirmation can be bypassed with -y flag

## [Previous versions]

(See git history for changes prior to multi-partition support)
