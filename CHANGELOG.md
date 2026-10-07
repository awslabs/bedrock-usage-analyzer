# Changelog

All notable changes to the Bedrock Usage Analyzer will be documented in this file.

## [0.7.0-beta] - 2026-10-07

### Added
- **Usage by IAM principal** (`--breakdown principal|session|tag:<key>|metadata:<key>`, `--principal`,
  or the interactive question; `--log-group` names another log group): for services that share an endpoint instead of having their
  own application inference profiles, each report breaks its usage down by calling IAM role or user, role
  session, IAM principal tag or `requestMetadata` key, read from the model invocation logs with CloudWatch
  Logs Insights (metadata fields only). Rows show tokens, requests, shares of the endpoint total, TPM/RPM
  (P50, P90, max) and TPD; usage the logs do not hold is its own row. Totals and quotas stay CloudWatch's.
- **Custom model deployments**: analyze on-demand custom model deployments, picked
  interactively or passed with `-m` (deployment ARN, ID or name). Usage comes from CloudWatch
  under the deployment ARN; limits are the base model's "(Model customization) Sum of on demand
  custom model deployment ..." quotas, mapped in the fm-list as the base model's `custom`
  endpoint (added by `refresh fm-list` for models that support customization). This covers
  customized Amazon Nova models (Nova 2 Lite, Lite, Micro, Pro), whether trained in Bedrock or
  in SageMaker AI.
- **Imported models (Custom Model Import)**: analyze models brought in with Custom Model Import,
  picked interactively ("Imported models", offered when the region has one) or passed with `-m`
  as the imported model ARN, ID or name. Usage comes from CloudWatch under the imported model ARN;
  reports show usage and throttles without limits, as imported models have no per-model token or
  request quotas. Optional permissions: `bedrock:ListImportedModels` (picker, ID or name) and
  `bedrock:GetImportedModel` (the name of an ARN passed with `-m`).

### Fixed
- Quota mapping no longer gives an on-demand endpoint the latency-optimized quotas, a model
  whose version is part of its name (Nova 2.5 Sonic) the quota of the unversioned family, or a
  model the quota of another API version of it (a "... Claude 3.5 Sonnet V2" quota for
  `anthropic.claude-3-5-sonnet-20240620-v1:0`).
- Bundled quota mappings: `nvidia.nemotron-nano-12b-v2` now uses the "NVIDIA Nemotron Nano 2 VL"
  quotas (it had the 9B model's "Nemotron Nano 2" ones, 10 regions); the `us` profiles of
  Stable Image Outpaint, Search and Recolor and Style Transfer (us-east-1, us-east-2, us-west-2)
  get their RPM quota, and Stable Diffusion 3.5 Large on demand (us-west-2) its RPM quota.

### Changed
- Refreshed bundled fm-lists and quota mappings: adds `amazon.nova-2-5-sonic`, `zai.glm-5.3` and
  new inference profile endpoints. Newly mapped where Service Quotas lists them: Grok 4.7
  `global` TPM (12 regions), Claude Sonnet 5.5 `global` TPM (15 regions), GPT-6.1 Sol `global`
  TPM and TPD (15 regions), the `in` profiles of Claude Haiku 4.5, Sonnet 5 and Opus 5
  (ap-south-1, ap-south-2), Claude Sonnet 5 on demand TPM (ap-northeast-2, ap-southeast-1),
  Claude Opus 5 on demand TPM (ap-northeast-2) and Claude 3.5 Sonnet V2 `apac` TPD
  (ap-south-1). Kimi K3 keeps its `us` and `global` mappings. GLM 5.3 has no Service Quotas
  yet, so it is reported without limits.
- Requires boto3 1.39.7 or later (custom model deployment APIs).

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
- `--update-bundle` (maintainers) reads and writes only the checkout's
  `src/bedrock_usage_analyzer/metadata` for every refresh command; user copies are no longer
  updated alongside it. Outside a checkout the commands exit before any AWS call.
- All AWS clients are created in one place (`aws/client_factory.py`): adaptive retries (up to 9 attempts) for CloudWatch, Service Quotas and Bedrock Runtime, standard retries (up to 4 attempts) for the others, and a single quick attempt for the cross-partition STS probes.

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
