"""The 68 iBUG face landmarks, as vertex weights over ANNY's 19,158-vertex makehuman mesh.

Region membership: per-target top-decile motion filter over the 52 facial-action
blendshapes anny ships, minus the excluded-vertex set (vertices > 5 mm from SOMA_wrap at
rest). Axes: `+X` subject-left, `+Y` backward, `+Z` up. See README for the region-per-
landmark map, the placement-prior gate, and the negative controls.

    python face_anchors.py --out .                     writes face68.pth, face68.json
    python face_anchors.py --negative-control          asserts the gate rejects the
                                                       pre-rework selector's output
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch

FACEUNITS_DIR_PARTS = ("data", "faceunits01", "targets", "faceunits")
CORE_QUANTILE = 0.9
BLEND_K = 4
DEFAULT_EXCLUDED_NPZ = pathlib.Path(
    r"C:\weftspun-keypoints\2-contract\anny-keypoint-anchors\excluded_vertices_5mm.npz")


def build_model():
    import anny
    from anny.models.model_data import TopologyConfig

    return anny.Anny(
        topology=TopologyConfig(base_mesh="makehuman", remove_unattached_vertices=False))


def load_targets(anny_root):
    root = pathlib.Path(anny_root, *FACEUNITS_DIR_PARTS)
    if not root.is_dir():
        sys.exit("FAIL  no faceunits directory at %s" % root)
    out = {}
    for entry in sorted(root.iterdir()):
        if entry.suffix != ".target":
            continue
        indices, deltas = [], []
        with entry.open("r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) != 4:
                    continue
                indices.append(int(parts[0]))
                deltas.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if not indices:
            continue
        out[entry.stem] = (
            torch.tensor(indices, dtype=torch.long),
            torch.tensor(deltas, dtype=torch.float64),
        )
    return out


def load_excluded(path=DEFAULT_EXCLUDED_NPZ) -> set[int]:
    if not path.is_file():
        return set()
    d = np.load(str(path))
    return {int(i) for i in d["excluded_indices"]}


def region_vertices(targets, names, excluded: set[int], quantile=CORE_QUANTILE):
    hits = set()
    for name in names:
        if name not in targets:
            sys.exit("FAIL  no faceunits target named %r" % name)
        indices, deltas = targets[name]
        magnitude = torch.linalg.norm(deltas, dim=1)
        cutoff = float(torch.quantile(magnitude, quantile))
        keep = indices[magnitude >= cutoff]
        hits.update(int(i) for i in keep)
    hits -= excluded
    return torch.tensor(sorted(hits), dtype=torch.long)


def blend_at(vertices, candidates, target, k=BLEND_K):
    if not len(candidates):
        return None, float("inf")
    picked = vertices[candidates]
    distance = torch.linalg.norm(picked - target, dim=1)
    k = min(k, len(candidates))
    near = torch.topk(distance, k, largest=False)
    weight = 1.0 / (near.values + 1e-6)
    weight = weight / weight.sum()
    row = torch.zeros(len(vertices), dtype=torch.float32)
    row[candidates[near.indices]] = weight.to(torch.float32)
    return row, float(near.values.max())


def household(metres):
    mm = metres * 1000.0
    for size, name in ((1.52, "a penny"), (7.0, "a pencil"), (10.5, "a AAA battery"),
                       (14.5, "a AA battery"), (21.2, "a nickel"), (42.7, "a golf ball"),
                       (57.0, "an adult wrist"), (66.0, "a soda can")):
        if mm <= size * 1.5:
            return "%.1f mm, about %s" % (mm, name)
    return "%.1f mm, wider than a soda can" % mm


def face_anchors(model, targets, excluded: set[int]):
    vertices = model.template_vertices.to(torch.float64)
    weights, report, missing = {}, [], []

    def emit(label, target_point, candidates):
        if target_point is None or not len(candidates):
            missing.append(label)
            return
        row, spread = blend_at(vertices, candidates, target_point)
        if row is None:
            missing.append(label)
            return
        weights[label] = row
        report.append((label, spread))

    # Jawline: 17 X-monotone samples, lowest-Z at each X. X spans right ear (X<0) through
    # chin (X=0) to left ear (X>0). Chin is the min-Z of the whole jaw ring.
    jaw = region_vertices(targets, ("jawOpen", "jawLeft", "jawRight"), excluded)
    if len(jaw):
        jaw_pts = vertices[jaw]
        xs = jaw_pts[:, 0]
        x_lo, x_hi = float(xs.min()), float(xs.max())
        band_half = (x_hi - x_lo) / 34.0
        for i in range(17):
            want_x = x_lo + (x_hi - x_lo) * (i / 16.0)
            band = torch.abs(xs - want_x) <= band_half
            if not band.any():
                j = int(torch.argmin(torch.abs(xs - want_x)))
            else:
                band_zs = jaw_pts[band, 2]
                band_j = int(torch.argmin(band_zs))
                j = int(torch.nonzero(band, as_tuple=False).flatten()[band_j])
            emit("face_kpt_%d" % i, jaw_pts[j], jaw)
    else:
        missing.extend("face_kpt_%d" % i for i in range(17))

    # Brows: 5 X-samples per side, highest-Z at each X.
    for side, names, indices in (
        ("right", ("browDownRight", "browOuterUpRight"), range(17, 22)),
        ("left", ("browDownLeft", "browOuterUpLeft"), range(22, 27)),
    ):
        brow = region_vertices(targets, names, excluded)
        if not len(brow):
            missing.extend("face_kpt_%d" % k for k in indices)
            continue
        brow_pts = vertices[brow]
        xs = brow_pts[:, 0]
        x_lo, x_hi = float(xs.min()), float(xs.max())
        for offset, k in enumerate(indices):
            want_x = x_lo + (x_hi - x_lo) * (offset / 4.0)
            band = torch.abs(xs - want_x) <= max((x_hi - x_lo) / 10.0, 0.003)
            if not band.any():
                missing.append("face_kpt_%d" % k)
                continue
            band_pts = brow_pts[band]
            j = int(torch.argmax(band_pts[:, 2]))  # highest Z = up
            emit("face_kpt_%d" % k, band_pts[j], brow)

    # Nose bridge (27..30): 4 Z-samples top-to-tip on a central-X strip, most-forward (min Y)
    # at each Z. Nose base (31..35): 5 X-samples at tip Z-band.
    nose = region_vertices(targets, ("noseSneerLeft", "noseSneerRight"), excluded)
    if len(nose):
        nose_pts = vertices[nose]
        zs = nose_pts[:, 2]
        xs = nose_pts[:, 0]
        z_lo, z_hi = float(zs.min()), float(zs.max())
        centre_x_span = (float(xs.max()) - float(xs.min())) / 5.0
        central = torch.abs(xs - float(xs.mean())) <= centre_x_span
        bridge_verts = nose[central]
        bridge_pts = vertices[bridge_verts]
        bridge_zs = bridge_pts[:, 2]
        for offset in range(4):
            want_z = z_hi - (z_hi - z_lo) * (offset / 3.0)  # top-to-tip
            band = torch.abs(bridge_zs - want_z) <= max((z_hi - z_lo) / 10.0, 0.002)
            if not band.any():
                j = int(torch.argmin(torch.abs(bridge_zs - want_z)))
                emit("face_kpt_%d" % (27 + offset), bridge_pts[j], bridge_verts)
                continue
            band_pts = bridge_pts[band]
            j = int(torch.argmin(band_pts[:, 1]))  # min Y = most forward
            emit("face_kpt_%d" % (27 + offset), band_pts[j], bridge_verts)
        tip_band = torch.abs(zs - z_lo) <= max((z_hi - z_lo) / 8.0, 0.003)
        base_pts = nose_pts[tip_band] if tip_band.any() else nose_pts
        if len(base_pts):
            xs_b = base_pts[:, 0]
            x_lo, x_hi = float(xs_b.min()), float(xs_b.max())
            for offset in range(5):
                want_x = x_lo + (x_hi - x_lo) * (offset / 4.0)
                j = int(torch.argmin(torch.abs(xs_b - want_x)))
                emit("face_kpt_%d" % (31 + offset), base_pts[j], nose)
        else:
            missing.extend("face_kpt_%d" % k for k in range(31, 36))
    else:
        missing.extend("face_kpt_%d" % k for k in range(27, 36))

    # iBUG right eye starts at outer, left eye starts at inner; both walk clockwise.
    wants_from_outer = [0.0, torch.pi / 3, 2 * torch.pi / 3, torch.pi,
                        -2 * torch.pi / 3, -torch.pi / 3]
    wants_from_inner = [torch.pi, 2 * torch.pi / 3, torch.pi / 3, 0.0,
                        -torch.pi / 3, -2 * torch.pi / 3]
    for side_names, indices, side_sign, wants in (
        (("eyeBlinkRight", "eyeSquintRight", "eyeWideRight"), range(36, 42), -1.0, wants_from_outer),
        (("eyeBlinkLeft", "eyeSquintLeft", "eyeWideLeft"), range(42, 48), +1.0, wants_from_inner),
    ):
        eye = region_vertices(targets, side_names, excluded)
        if not len(eye):
            missing.extend("face_kpt_%d" % k for k in indices)
            continue
        eye_pts = vertices[eye]
        cx = eye_pts[:, 0].mean()
        cz = eye_pts[:, 2].mean()
        rel_x = eye_pts[:, 0] - cx
        rel_z = eye_pts[:, 2] - cz
        angle = torch.atan2(rel_z, rel_x * side_sign)
        for want, k in zip(wants, indices):
            j = int(torch.argmin(torch.abs(torch.remainder(angle - want + torch.pi,
                                                           2 * torch.pi) - torch.pi)))
            emit("face_kpt_%d" % k, eye_pts[j], eye)

    outer_lip_names = (
        "mouthSmileLeft", "mouthSmileRight",
        "mouthFrownLeft", "mouthFrownRight",
        "mouthUpperUpLeft", "mouthUpperUpRight",
        "mouthLowerDownLeft", "mouthLowerDownRight",
    )
    inner_lip_names = ("mouthRollUpper", "mouthRollLower")
    outer = region_vertices(targets, outer_lip_names, excluded)
    inner = region_vertices(targets, inner_lip_names, excluded)

    def sample_ring(vert_indices, count, indices):
        if not len(vert_indices):
            missing.extend("face_kpt_%d" % k for k in indices)
            return
        pts = vertices[vert_indices]
        cx = pts[:, 0].mean()
        cz = pts[:, 2].mean()
        # angle in the X-Z plane, subject-right corner at angle 0
        angle = torch.atan2(pts[:, 2] - cz, -(pts[:, 0] - cx))
        for offset, k in enumerate(indices):
            want = (offset / count) * 2 * torch.pi
            if want > torch.pi:
                want = want - 2 * torch.pi
            j = int(torch.argmin(torch.abs(torch.remainder(angle - want + torch.pi,
                                                           2 * torch.pi) - torch.pi)))
            emit("face_kpt_%d" % k, pts[j], vert_indices)

    sample_ring(outer, 12, range(48, 60))
    sample_ring(inner, 8, range(60, 68))

    return weights, report, missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=".")
    ap.add_argument("--anny-root", default="")
    ap.add_argument("--excluded-npz", type=pathlib.Path, default=DEFAULT_EXCLUDED_NPZ)
    ap.add_argument("--show", type=int, default=12)
    ap.add_argument("--worst-cap-mm", type=float, default=13.0)
    ap.add_argument("--negative-control", action="store_true")
    args = ap.parse_args()

    anny_root = args.anny_root
    if not anny_root:
        import anny
        anny_root = os.path.dirname(anny.__file__)

    model = build_model()
    vertex_count = int(model.template_vertices.shape[0])
    targets = load_targets(anny_root)
    excluded = load_excluded(args.excluded_npz)

    if args.negative_control:
        global region_vertices
        original = region_vertices

        def loose(targets, names, excluded, quantile=None):
            hits = set()
            for name in names:
                idx, deltas = targets[name]
                magnitude = torch.linalg.norm(deltas, dim=1)
                keep = idx[magnitude >= 0.0015]
                hits.update(int(i) for i in keep)
            return torch.tensor(sorted(hits), dtype=torch.long)

        region_vertices = loose
        try:
            weights, report, missing = face_anchors(model, targets, set())
        finally:
            region_vertices = original
    else:
        weights, report, missing = face_anchors(model, targets, excluded)

    bad = [(label, float(row.sum())) for label, row in weights.items()
           if abs(float(row.sum()) - 1.0) > 1e-3]
    if bad:
        sys.exit("FAIL  %d row(s) are not convex blends, e.g. %s sums to %.6f"
                 % (len(bad), bad[0][0], bad[0][1]))

    spreads = sorted(s for _, s in report)
    median = spreads[len(spreads) // 2] if spreads else 0.0
    p90 = spreads[int(0.9 * len(spreads))] if spreads else 0.0
    worst = spreads[-1] if spreads else 0.0
    worst_label = max(report, key=lambda r: r[1])[0] if report else ""

    if args.negative_control:
        if worst * 1000.0 > args.worst_cap_mm:
            print("negative control OK: worst %s exceeds the %.1f mm gate (as expected)"
                  % (household(worst), args.worst_cap_mm))
            return 0
        sys.exit("negative control FAILED: the known-loose selector's worst %s came in "
                 "UNDER the %.1f mm gate" % (household(worst), args.worst_cap_mm))

    if worst * 1000.0 > args.worst_cap_mm:
        sys.exit("FAIL  worst landmark %s spread %s exceeds the %.1f mm gate"
                 % (worst_label, household(worst), args.worst_cap_mm))

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(weights, out / "face68.pth")
    with open(out / "face68.json", "w", encoding="utf-8") as fh:
        json.dump({"topology": {"base_mesh": "makehuman", "vertex_count": vertex_count},
                   "blend_k": BLEND_K, "core_quantile": CORE_QUANTILE,
                   "excluded_vertex_count": len(excluded),
                   "targets_read": sorted(targets),
                   "have": sorted(weights), "missing": missing,
                   "worst_cap_mm": args.worst_cap_mm,
                   "spread_metres": {label: round(s, 6) for label, s in report},
                   "spread_summary_mm": {"median": round(median * 1000, 2),
                                         "p90": round(p90 * 1000, 2),
                                         "worst": round(worst * 1000, 2),
                                         "worst_label": worst_label}}, fh, indent=2)

    report.sort(key=lambda r: -r[1])
    print("%d of 68 face keypoints, vertex width %d, %d verts excluded"
          % (len(weights), vertex_count, len(excluded)))
    print("widest blends (target to furthest vertex used):")
    for label, spread in report[:args.show]:
        print("  %-16s %s" % (label, household(spread)))
    print("median blend spread   %s" % household(median))
    print("p90 blend spread      %s" % household(p90))
    print("worst blend spread    %s  (%s)" % (household(worst), worst_label))
    print("%d missing: %s" % (len(missing), ", ".join(missing) if missing else "none"))
    print("wrote %s and %s" % (out / "face68.pth", out / "face68.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
