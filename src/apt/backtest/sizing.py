"""Phase 2B sizing primitives (asset-agnostic).

§3.2 fixed-fractional sizing:
    notional / capital = risk_frac / stop_distance,
where stop_distance = (z_stop − z_entry) · rolling_sigma_at_entry, i.e. the
spread move at which the position would hit its z-stop and lose ``risk_frac``
of capital. All inputs are unitless (z-scores) or in log-spread units —
no equity-specific assumptions.

§3.3 / §3.4 / §3.5 / §3.6 caps:
    apply_per_pair_cap, apply_cluster_scaledown, has_shared_leg.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping


def compute_risk_based_weight(
    *,
    z_entry: float,
    z_stop: float,
    sigma_at_entry: float,
    risk_frac: float,
) -> float:
    """1%-rule notional (as a fraction of capital).

    ``z_stop`` is the positive z-magnitude of the stop threshold (e.g. 3.5);
    the signed stop position is inferred from the sign of ``z_entry``.
    """
    if sigma_at_entry <= 0:
        return 0.0
    if not (z_stop > 0 and abs(z_entry) > 0):
        return 0.0
    direction = 1 if z_entry < 0 else -1
    stop_z = -z_stop * direction  # long ⇒ stop at -z_stop ; short ⇒ at +z_stop
    distance_in_z = abs(z_entry - stop_z)
    distance_in_spread = distance_in_z * abs(sigma_at_entry)
    if distance_in_spread <= 0:
        return 0.0
    return float(risk_frac) / float(distance_in_spread)


def apply_per_pair_cap(weight: float, per_pair_cap: float) -> float:
    """Clamp a per-pair notional weight at the per-pair cap."""
    if per_pair_cap is None or per_pair_cap <= 0:
        return weight
    return min(float(weight), float(per_pair_cap))


def cluster_usage(
    open_weights: Mapping[str, float],
    sector_map: Mapping[str, str],
) -> dict[str, float]:
    """Sum of weights per cluster (sector)."""
    out: dict[str, float] = defaultdict(float)
    for pkey, w in open_weights.items():
        if w == 0:
            continue
        sec = sector_map.get(pkey, "OTHER")
        out[sec] += abs(float(w))
    return dict(out)


def cluster_room(
    sector: str,
    current_usage: Mapping[str, float],
    cluster_cap: float,
) -> float:
    """Available headroom in a cluster before hitting the cap (0 if at/above)."""
    used = float(current_usage.get(sector, 0.0))
    return max(0.0, float(cluster_cap) - used)


def has_shared_leg(
    new_pair_key: str,
    open_pair_keys: list[str],
) -> bool:
    """True if ``new_pair_key`` shares ANY symbol with any currently-open pair.

    Pair key format: ``"Y/X"`` (slash-separated). De-dup is direction-agnostic
    here — entering a same-sym overlap is always blocked. The conservative
    interpretation of §3.5: stack 2× on a single name and you've concentrated
    risk on the underlying, not on the spread.
    """
    y_new, x_new = new_pair_key.split("/")
    new_syms = {y_new, x_new}
    for k in open_pair_keys:
        ky, kx = k.split("/")
        if new_syms & {ky, kx}:
            return True
    return False


def apply_cluster_scaledown(
    open_weights: dict[str, float],
    sector_map: Mapping[str, str],
    cluster_cap: float,
) -> dict[str, float]:
    """Pro-rata scale-down: for any cluster whose summed weight exceeds the
    cap, scale every member's weight by ``cap / cluster_sum`` so the cluster
    is exactly at the cap. Returns the mutated dict for convenience.

    Pure side-effect on the input dict — mutates in place AND returns it.
    """
    if cluster_cap is None or cluster_cap <= 0:
        return open_weights
    usage = cluster_usage(open_weights, sector_map)
    for sec, total in usage.items():
        if total <= cluster_cap:
            continue
        scale = cluster_cap / total
        for pkey in list(open_weights.keys()):
            if open_weights[pkey] == 0:
                continue
            if sector_map.get(pkey, "OTHER") == sec:
                open_weights[pkey] *= scale
    return open_weights
