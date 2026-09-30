# Changelog

All notable changes to the Bedrock Usage Analyzer will be documented in this file.

## [0.6.0-beta] - 2026-09-30

Combines the AWS GovCloud work from #5 and #6 into one partition layer and fixes #7.

### Added
- **AWS GovCloud (US) support** (#5, #6): partition of each region is resolved from botocore
  endpoint data; ARNs, endpoints and Service Quotas console links follow it. The region picker
  lists only regions of the credentials' partition (from the STS caller identity). Bundled
  metadata includes `us-gov-east-1`, `us-gov-west-1` and the `us-gov` profile prefix.
- **Application inference profile selection** (#7): choose specific application profiles
  interactively (`1,3-4` or `all`), or pass an application profile ID or ARN with `-m`.
- `-m/--model-id` can be repeated; it also accepts system inference profile and
  foundation-model ARNs.
- A note when an endpoint has no application profiles but the same model has profiles under
  another endpoint.
- Error hints for credential, permission and network errors, with partition-specific advice.
- pytest suite (`tests/`) that runs offline against stubbed AWS clients.

### Fixed
- Application profiles copied from `au.*` or `jp.*` profiles were attributed to `apac.*`, so
  the analysis showed no data for them (#7). Sources are now matched on the exact set of model
  ARNs of the system profiles.
- `bua refresh fm-quotas` dropped endpoints that got no quota mapping from the model list.
- Two reports for the same model (different endpoints) in one run overwrote each other.
- Missing `regions.yml` or `fm-list-<region>.yml` crashed instead of printing the refresh hint.
- CloudWatch fetch warned "Connection pool is full" on hosts with many CPUs.
- Python 3.9: `Path | None` and `list[Path]` annotations failed at import (#5).

### Security
- HTML report rendered with Jinja2 autoescaping; embedded data uses `|tojson`, so profile names,
  tags or model IDs cannot inject markup or close the `<script>` block.
- Chart.js and plugins pinned to exact versions with Subresource Integrity hashes.
- Region names are validated before they are used in metadata file names.

### Changed
- `regions.yml` stays a plain list of region names; entries in the `{name: ...}` form are
  still read.
- `refresh regions` keeps regions of other partitions when it updates the list.
- All AWS clients are created in one place (`aws/client_factory.py`) with adaptive retries.

### Removed
- Hardcoded GovCloud endpoint URLs and the service allowlist from #5 (botocore resolves them).
- Duplicate partition modules and the documentation files added by #5 and #6; the README has one
  GovCloud section.

## [0.5.1-beta]

### Changed
- Skip `me-south-1` and `me-central-1` during region refresh; refreshed bundled metadata.

## [0.5.0-beta] - 2025-02-25

### Added
- CLI arguments for non-interactive usage (--region, --model-id, --granularity, --output-dir)
- -y/--yes flag to skip account confirmation prompts
- Support for scripted/automated workflows and CI/CD pipelines

### Changed
- Interactive prompts can now be skipped by providing CLI arguments
- Account confirmation can be bypassed with -y flag

## [Previous versions]

(See git history for changes prior to 0.5.0)
