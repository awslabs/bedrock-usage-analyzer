# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference profile discovery for Bedrock models"""

import logging
from typing import Dict, FrozenSet, Iterable, List, Optional

from bedrock_usage_analyzer.core.errors import is_access_denied
from bedrock_usage_analyzer.aws.bedrock import (
    get_default_region_prefix_map,
    list_inference_profiles,
    model_id_from_arn,
    region_from_arn,
    region_group,
    split_profile_id,
)

logger = logging.getLogger(__name__)

# profile_prefix of an application profile whose source endpoint cannot be determined;
# no fm-list endpoint has this key, so no quota is attached to it
UNKNOWN_SOURCE = 'unknown'

# Attempts at listing application profiles per run before giving up on transient errors
MAX_LISTING_ATTEMPTS = 2

# Regions behind the country-level Asia Pacific profiles, extended at run time from the
# listed profiles. Used only for copies of a country profile the region does not list.
COUNTRY_PROFILE_REGIONS = {
    'jp': {'ap-northeast-1', 'ap-northeast-3'},
    'au': {'ap-southeast-2', 'ap-southeast-4', 'ap-southeast-6'},
    'in': {'ap-south-1', 'ap-south-2'},
}


def _specific_first(profile_ids: List[str]) -> List[str]:
    """Order candidates so geography-specific prefixes (jp, au) come before apac/global."""
    broad = ('apac.', 'global.')
    return sorted(dict.fromkeys(profile_ids), key=lambda p: (p.startswith(broad), p))


class InferenceProfileFetcher:
    """Discovers the endpoints (base model, system profile, application profiles) to analyze.

    An application inference profile does not record which endpoint it was
    copied from; the API only returns the foundation-model ARNs it routes to.
    The source is resolved by matching that ARN set against the system-defined
    profiles in the region, which tells an 'au.' copy apart from an 'apac.' or
    'jp.' copy even though all of them route to ap-* regions.
    """

    @classmethod
    def for_region(cls, bedrock_client, fm_models: Optional[List[Dict]]) -> 'InferenceProfileFetcher':
        """Fetcher that knows which models of the region's fm-list have an on-demand endpoint."""
        on_demand = [m['model_id'] for m in fm_models or [] if 'base' in (m.get('endpoints') or {})]
        return cls(bedrock_client, on_demand)

    def __init__(self, bedrock_client, on_demand_models: Optional[Iterable[str]] = None):
        self.bedrock_client = bedrock_client
        # Models with an on-demand ('base') endpoint in the region's fm-list
        self.on_demand_models = set(on_demand_models or ())
        self.prefix_map = get_default_region_prefix_map()
        self._system_profiles: Optional[List[Dict]] = None
        self._system_by_arns: Dict[FrozenSet[str], List[str]] = {}
        self._system_ids: set = set()
        self._country_regions = {p: set(r) for p, r in COUNTRY_PROFILE_REGIONS.items()}
        self._app_profiles: Optional[List[Dict]] = None
        # Per listing type: the result, the error the run gave up on, and failed attempts
        self._listings: Dict[str, Dict] = {kind: {'result': None, 'error': None, 'failures': 0}
                                           for kind in ('SYSTEM_DEFINED', 'APPLICATION')}
        self._by_model: Dict[str, List] = {}   # model ID -> [(routing set, profile IDs)]
        self._listed_prefixes: set = set()
        self._tags_cache: Dict[str, Dict[str, str]] = {}

    # ------------------------------------------------------------------ listing

    def _list_once(self, kind: str) -> List[Dict]:
        """List inference profiles of ``kind`` once per run.

        A permission error is permanent. A throttle or network blip gets one more try from
        the next caller, then the run stops retrying (each try can take a while with
        retries and timeouts). A successful listing is kept even if a later one fails.
        """
        state = self._listings[kind]
        if state['error'] is not None:
            raise state['error']
        if state['result'] is None:
            try:
                state['result'] = list_inference_profiles(self.bedrock_client, kind)
            except Exception as e:
                state['failures'] += 1
                if is_access_denied(e) or state['failures'] >= MAX_LISTING_ATTEMPTS:
                    state['error'] = e
                raise
        return state['result']

    def _load_system_profiles(self) -> List[Dict]:
        if self._system_profiles is None:
            profiles = self._list_once('SYSTEM_DEFINED')
            self._system_profiles = profiles
            prefix_regions: Dict[str, set] = {}
            for profile in profiles:
                self._system_ids.add(profile['inferenceProfileId'])
                arns = frozenset(m.get('modelArn', '') for m in profile.get('models', []))
                if arns:
                    # Several profiles can share one routing set (jp.X and apac.X when a model
                    # is offered only in Tokyo and Osaka), so keep every candidate
                    self._system_by_arns.setdefault(arns, []).append(profile['inferenceProfileId'])
                    regions = {region_from_arn(a) for a in arns}
                    self._listed_prefixes.add(profile['inferenceProfileId'].split('.', 1)[0])
                    if all(regions):  # region-bound (global profiles have a region-less ARN)
                        prefix = profile['inferenceProfileId'].split('.', 1)[0]
                        prefix_regions.setdefault(prefix, set()).update(regions)
            self._learn_country_geographies(prefix_regions)
            # Candidates per model, so resolving one application profile only looks at its model
            for arn_set, ids in self._system_by_arns.items():
                for profile_id in ids:
                    model = profile_id.split('.', 1)[-1]
                    entry = next((e for e in self._by_model.setdefault(model, []) if e[0] == arn_set), None)
                    if entry is None:
                        self._by_model[model].append((arn_set, [profile_id]))
                    else:
                        entry[1].append(profile_id)
        return self._system_profiles

    def _learn_country_geographies(self, prefix_regions: Dict[str, set]) -> None:
        """Country geographies (jp, au, kr, ...) from the listed profiles.

        A regional prefix whose regions are a strict subset of another regional prefix's
        (kr.* inside apac.*) is a country geography, so a new one needs no code change.
        The defaults also cover a country whose profiles are not listed for every model.
        """
        for prefix, regions in prefix_regions.items():
            if prefix in self._country_regions or any(
                    regions < other for p, other in prefix_regions.items() if p != prefix):
                self._country_regions.setdefault(prefix, set()).update(regions)

    def list_application_profiles(self) -> List[Dict]:
        """All application inference profiles in the region, with their resolved source.

        Each entry: id, name, arn, status, model_id, profile_prefix (None for a
        base-model copy) and source (the endpoint ID it was copied from).
        """
        if self._app_profiles is None:
            if self._listings['APPLICATION']['result'] is None:
                logger.info("  Listing application inference profiles...")
            raw = self._list_once('APPLICATION')
            self._load_system_profiles()
            profiles = []
            for profile in raw:
                arns = [m.get('modelArn', '') for m in profile.get('models', [])]
                sources = self.resolve_sources(arns)
                if sources:
                    source = sources[0]
                    first = source.split('.', 1)[0]
                    if '.' in source and (source in self._system_ids or first in self._listed_prefixes
                                          or first in self._country_regions):
                        # A listed system profile or a known geography: the first segment is
                        # the prefix, even one launched after this release (e.g. 'kr.')
                        prefix, model_id = source.split('.', 1)
                    else:
                        model_id, prefix = split_profile_id(source)
                else:
                    # Still listed (it can be analyzed by ID), but no endpoint or quotas are implied
                    model_ids = sorted({m for m in (model_id_from_arn(a) for a in arns) if m})
                    if not model_ids:
                        continue
                    source, model_id, prefix = None, model_ids[0], UNKNOWN_SOURCE
                    logger.info(f"  Note: could not tell which endpoint {profile['inferenceProfileId']} "
                                f"was copied from")
                profiles.append({
                    'id': profile['inferenceProfileId'],
                    'name': profile.get('inferenceProfileName', profile['inferenceProfileId']),
                    'arn': profile.get('inferenceProfileArn'),
                    'status': profile.get('status'),
                    'model_id': model_id,
                    'profile_prefix': prefix,
                    'source': source,
                    'sources': sources,
                })
            self._app_profiles = profiles
            logger.info(f"  Found {len(profiles)} application inference profile(s)")
        return self._app_profiles

    # --------------------------------------------------------------- resolution

    def resolve_sources(self, model_arns: Iterable[str]) -> List[str]:
        """Return the endpoint IDs an application profile may have been copied from.

        Usually one. Several when system profiles share the exact routing set, in which
        case the API gives no way to tell them apart and the profile belongs to each.
        Order: exact match with a multi-region system profile, then a single model ARN
        (base model copy), then the narrowest system profile containing every routed
        region, then the closest system profile for the same model, then a region-prefix
        heuristic.
        """
        arns = [a for a in model_arns if a]
        if not arns:
            return []
        model_ids = sorted({m for m in (model_id_from_arn(a) for a in arns) if m})
        if not model_ids:
            return []
        model_id = model_ids[0]
        arn_set = frozenset(arns)

        self._load_system_profiles()
        exact = self._system_by_arns.get(arn_set)
        if exact and len(arn_set) > 1:
            return _specific_first(exact)

        if len(arn_set) == 1:
            # A base-model copy, unless a system profile routes to exactly this one ARN: the
            # two cannot be told apart, so the profile belongs to both endpoints. The first
            # one decides the quotas: the on-demand endpoint when the model has one, else the
            # system profile (the model may have no on-demand endpoint at all).
            if model_id in self.on_demand_models:
                return [model_id] + _specific_first(exact or [])
            return _specific_first(exact or []) + [model_id]

        # The routing set no longer equals any listed profile (sets change over time). Only
        # listed profiles of this model are candidates, so the source always exists in the
        # region; global.* only for a copy that routes globally (it has a region-less ARN).
        regions = {region_from_arn(a) for a in arns}
        routes_globally = '' in regions
        candidates = [(system_arns, ids) for system_arns, ids in (
            (s, [p for p in ids if routes_globally or not p.startswith('global.')])
            for s, ids in self._by_model.get(model_id, [])) if len(system_arns) > 1 and ids]

        # A copy inside one country geography (au, jp, in) whose profile is listed is a copy of
        # that profile, even when it also fits inside the wider apac.* set (issue #7)
        country = None if routes_globally else self._country_of(regions)
        country_id = f"{country}.{model_id}" if country else None
        if country_id and any(country_id in ids for _, ids in candidates):
            return [country_id]

        # Sets usually grow: the narrowest listed profile that contains every routed region
        supersets = {}
        for system_arns, ids in candidates:
            if arn_set < system_arns:
                supersets.setdefault(len(system_arns), []).extend(ids)
        if supersets:
            return _specific_first(supersets[min(supersets)])

        # Closest system profile for the same model (a region was also removed)
        best, best_score = [], 0.0
        for system_arns, ids in candidates:
            overlap = len(arn_set & system_arns)
            if not overlap:
                continue
            score = overlap / len(arn_set | system_arns)
            if score > best_score:
                best, best_score = ids, score
            elif score == best_score:
                best = best + ids
        if best:
            return _specific_first(best)

        # Nothing comparable is listed for this model: guess from the regions
        inferred = self._infer_from_regions(arns, model_id)
        return [inferred] if inferred else []

    def _country_of(self, regions) -> Optional[str]:
        """The country prefix (jp, au, kr, ...) whose regions contain all of ``regions``."""
        for prefix, members in sorted(self._country_regions.items()):
            if regions and set(regions) <= members and \
                    (prefix in self.prefix_map or prefix in self._listed_prefixes):
                return prefix
        return None

    def _infer_from_regions(self, model_arns: List[str], model_id: str) -> Optional[str]:
        """Fallback when no system profile matches: guess from the ARN regions."""
        regions = [region_from_arn(a) for a in model_arns]
        if any(not r for r in regions):
            return f"global.{model_id}"  # global profiles include a region-less ARN
        # No listed profile of this model to compare with: a set inside one country is a copy
        # of that country's profile, not of apac.* (issue #7)
        country = self._country_of(regions)
        if country:
            return f"{country}.{model_id}"
        groups = {region_group(r) for r in regions}
        if len(groups) == 1:
            group = groups.pop()
            if group not in self.prefix_map:
                # No system profile family for this geography (sa, me, mx, ...): an ID like
                # 'sa.<model>' would not exist, so leave the source unknown
                return None
            return f"{self.prefix_map[group]}.{model_id}"
        return f"global.{model_id}"

    def is_system_profile(self, profile_id: str) -> bool:
        """True when ``profile_id`` is a system-defined inference profile in the region."""
        self._load_system_profiles()
        return profile_id in self._system_ids

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
            # e.g. no bedrock:ListInferenceProfiles permission: still analyze the endpoint itself,
            # but say plainly that the report is missing its application profiles
            logger.warning(f"  WARNING: Could not list application inference profiles ({e}). "
                           f"This report covers {target_endpoint} only, without its application profiles.")
            app_profiles = []
        matched = 0
        for app in app_profiles:
            if wanted:
                if app['id'] not in wanted:
                    continue
            elif target_endpoint not in app['sources']:
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
        # Only a hint: use a listing that already succeeded, never trigger a new one
        app_profiles = self._app_profiles or []
        target = model_id if profile_prefix is None else f"{profile_prefix}.{model_id}"
        for app in app_profiles:
            # Profiles with an unknown source are not pointed to: there is no endpoint to pick
            if app['model_id'] == model_id and app['sources'] and target not in app['sources']:
                key = app['profile_prefix'] or 'base'
                counts[key] = counts.get(key, 0) + 1
        return counts
