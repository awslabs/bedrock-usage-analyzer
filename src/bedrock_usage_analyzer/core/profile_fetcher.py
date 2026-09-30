# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference profile discovery for Bedrock models"""

import logging
from typing import Dict, FrozenSet, Iterable, List, Optional

from bedrock_usage_analyzer.aws.bedrock import (
    get_default_region_prefix_map,
    list_inference_profiles,
    model_id_from_arn,
    region_from_arn,
    region_group,
    split_profile_id,
)

logger = logging.getLogger(__name__)


class InferenceProfileFetcher:
    """Discovers the endpoints (base model, system profile, application profiles) to analyze.

    An application inference profile does not record which endpoint it was
    copied from; the API only returns the foundation-model ARNs it routes to.
    The source is resolved by matching that ARN set against the system-defined
    profiles in the region, which tells an 'au.' copy apart from an 'apac.' or
    'jp.' copy even though all of them route to ap-* regions.
    """

    def __init__(self, bedrock_client):
        self.bedrock_client = bedrock_client
        self.prefix_map = get_default_region_prefix_map()
        self._system_profiles: Optional[List[Dict]] = None
        self._system_by_arns: Dict[FrozenSet[str], str] = {}
        self._app_profiles: Optional[List[Dict]] = None
        self._tags_cache: Dict[str, Dict[str, str]] = {}

    # ------------------------------------------------------------------ listing

    def _load_system_profiles(self) -> List[Dict]:
        if self._system_profiles is None:
            self._system_profiles = list_inference_profiles(self.bedrock_client, 'SYSTEM_DEFINED')
            for profile in self._system_profiles:
                arns = frozenset(m.get('modelArn', '') for m in profile.get('models', []))
                if arns:
                    self._system_by_arns.setdefault(arns, profile['inferenceProfileId'])
        return self._system_profiles

    def list_application_profiles(self) -> List[Dict]:
        """All application inference profiles in the region, with their resolved source.

        Each entry: id, name, arn, status, model_id, profile_prefix (None for a
        base-model copy) and source (the endpoint ID it was copied from).
        """
        if self._app_profiles is None:
            logger.info("  Listing application inference profiles...")
            raw = list_inference_profiles(self.bedrock_client, 'APPLICATION')
            self._load_system_profiles()
            profiles = []
            for profile in raw:
                arns = [m.get('modelArn', '') for m in profile.get('models', [])]
                source = self.resolve_source(arns)
                if source is None:
                    continue
                model_id, prefix = split_profile_id(source)
                profiles.append({
                    'id': profile['inferenceProfileId'],
                    'name': profile.get('inferenceProfileName', profile['inferenceProfileId']),
                    'arn': profile.get('inferenceProfileArn'),
                    'status': profile.get('status'),
                    'model_id': model_id,
                    'profile_prefix': prefix,
                    'source': source,
                })
            self._app_profiles = profiles
            logger.info(f"  Found {len(profiles)} application inference profile(s)")
        return self._app_profiles

    # --------------------------------------------------------------- resolution

    def resolve_source(self, model_arns: Iterable[str]) -> Optional[str]:
        """Return the endpoint ID an application profile was copied from.

        Order: exact match with a multi-region system profile, then a single
        model ARN (base model copy), then the closest system profile for the
        same model, then a region-prefix heuristic.
        """
        arns = [a for a in model_arns if a]
        if not arns:
            return None
        model_ids = sorted({m for m in (model_id_from_arn(a) for a in arns) if m})
        if not model_ids:
            return None
        model_id = model_ids[0]
        arn_set = frozenset(arns)

        self._load_system_profiles()
        exact = self._system_by_arns.get(arn_set)
        if exact and len(arn_set) > 1:
            return exact

        if len(arn_set) == 1:
            return model_id

        # Closest system profile for the same model (routing sets change over time)
        best, best_score = None, 0.0
        for system_arns, profile_id in self._system_by_arns.items():
            if len(system_arns) < 2 or profile_id.split('.', 1)[-1] != model_id:
                continue
            overlap = len(arn_set & system_arns)
            if not overlap:
                continue
            score = overlap / len(arn_set | system_arns)
            if score > best_score:
                best, best_score = profile_id, score
        if best:
            return best

        return self._infer_from_regions(arns, model_id)

    def _infer_from_regions(self, model_arns: List[str], model_id: str) -> str:
        """Fallback when no system profile matches: guess from the ARN regions."""
        regions = [region_from_arn(a) for a in model_arns]
        if any(not r for r in regions):
            return f"global.{model_id}"  # global profiles include a region-less ARN
        groups = {region_group(r) for r in regions}
        if len(groups) == 1:
            group = groups.pop()
            return f"{self.prefix_map.get(group, group)}.{model_id}"
        return f"global.{model_id}"

    def resolve_application_profile(self, identifier: str) -> Optional[Dict]:
        """Find an application profile by ID, ARN or name."""
        identifier = identifier.strip()
        for profile in self.list_application_profiles():
            if identifier in (profile['id'], profile['arn'], profile['name']):
                return profile
        return None

    # ---------------------------------------------------------------- discovery

    def _get_tags(self, profile_arn: Optional[str], profile_id: str) -> Dict[str, str]:
        if not profile_arn:
            return {}
        if profile_arn not in self._tags_cache:
            tags = {}
            try:
                response = self.bedrock_client.list_tags_for_resource(resourceARN=profile_arn)
                tags = {t['key']: t['value'] for t in response.get('tags', [])}
            except Exception as e:
                logger.info(f"  Warning: Could not fetch tags for {profile_id}: {e}")
            self._tags_cache[profile_arn] = tags
        return self._tags_cache[profile_arn]

    def find_profiles(self, model_id, profile_prefix, application_profile_ids=None):
        """Find the endpoints to analyze for a model.

        Without ``application_profile_ids``: the base model or system profile
        plus every application profile copied from it. With it: only those
        application profiles.

        Returns:
            tuple: (profiles list, profile_names dict, profile_metadata dict)
                   profile_metadata contains 'id' and 'tags' for each profile
        """
        logger.info("  Discovering inference profiles...")
        target_endpoint = model_id if profile_prefix is None else f"{profile_prefix}.{model_id}"

        profiles: List[str] = []
        profile_names: Dict[str, str] = {}
        profile_metadata: Dict[str, Dict] = {}

        if not application_profile_ids:
            profiles.append(target_endpoint)
            profile_names[target_endpoint] = target_endpoint
            profile_metadata[target_endpoint] = {'id': 'N/A', 'tags': {}}

        if not hasattr(self.bedrock_client, 'list_inference_profiles'):
            logger.info("No application profiles found (API not available)")
            return profiles, profile_names, profile_metadata

        wanted = set(application_profile_ids or [])
        try:
            app_profiles = self.list_application_profiles()
        except Exception as e:
            if wanted:
                raise
            # e.g. no bedrock:ListInferenceProfiles permission: still analyze the endpoint itself
            logger.info(f"  Warning: Could not list application inference profiles: {e}")
            app_profiles = []
        matched = 0
        for app in app_profiles:
            if wanted:
                if app['id'] not in wanted:
                    continue
            elif app['source'] != target_endpoint:
                continue
            matched += 1
            profiles.append(app['id'])
            profile_names[app['id']] = app['name']
            profile_metadata[app['id']] = {'id': app['id'], 'tags': self._get_tags(app['arn'], app['id'])}

        logger.info(f"  Profile discovery: {matched} application profiles matched")
        return profiles, profile_names, profile_metadata

    def other_sources_for_model(self, model_id: str, profile_prefix: Optional[str]) -> Dict[str, int]:
        """Count application profiles of ``model_id`` that come from other endpoints.

        Used to warn when a selection matches no application profile but the
        account does have some for the same model under a different endpoint.
        """
        counts: Dict[str, int] = {}
        try:
            app_profiles = self.list_application_profiles()
        except Exception:
            return counts
        for app in app_profiles:
            if app['model_id'] == model_id and app['profile_prefix'] != profile_prefix:
                key = app['profile_prefix'] or 'base'
                counts[key] = counts.get(key, 0) + 1
        return counts
