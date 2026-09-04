"""The 68 iBUG face landmarks, as vertex weights over ANNY's 19,158-vertex makehuman mesh.

Region membership comes from ANNY's 52 facial-action blendshape targets under
`data/faceunits01/targets/faceunits/*.target`, each a MakeHuman target file with one
`vertex_idx dx dy dz` per line. A vertex belongs to a target's core region when its
motion magnitude sits at or above the target's own top decile. Each landmark is then a
convex blend over the K nearest region vertices. See README for the region assignments,
the two per-region constructions (jawline lowest-Y band, nose-bridge central-X strip)
and the negative-control gate.

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

import torch

FACEUNITS_DIR_PARTS = ("data", "faceunits01", "targets", "faceunits")
CORE_QUANTILE = 0.9
BLEND_K = 4


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


def region_vertices(targets, names, quantile=CORE_QUANTILE):
    """Union of each named target's top-`quantile` motion vertices.

    A missing name is a FAIL not a skip: a mis-typed target name would silently give an
    empty region and then a nonsensical landmark. The caller is expected to name real files.
    """
    hits = set()
    for name in names:
        if name not in targets:
            sys.exit("FAIL  no faceunits target named %r" % name)
        indices, deltas = targets[name]
        magnitude = torch.linalg.norm(deltas, dim=1)
        cutoff = float(torch.quantile(magnitude, quantile))
        keep = indices[magnitude >= cutoff]
        hits.update(int(i) for i in keep)
    return torch.tensor(sorted(hits), dtype=torch.long)


def blend_at(vertices, candidates, target, k=BLEND_K):
    """A convex blend over the `k` candidate vertices nearest `target`, by inverse distance.

    Returns the (V,) weight row and the spread -- the distance from the target to the
    furthest vertex the blend uses.
    """
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


# iBUG-68 layout (subject's own right, matching iBUG and dlib):
#   0..16   jawline right-ear to chin (8) to left-ear
#   17..21  right brow, outer-to-inner
#   22..26  left brow, inner-to-outer
#   27..30  nose bridge, top-to-tip
#   31..35  nose base under the tip, right-to-left
#   36..41  right eye ring, outer corner clockwise
#   42..47  left eye ring, inner corner clockwise
#   48..59  outer mouth ring, right corner clockwise
#   60..67  inner mouth ring, right corner clockwise
# ANNY makehuman axes: +X subject-left, +Y up, +Z toward camera.

def face_anchors(model, targets):
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

    jaw = region_vertices(targets, ("jawOpen", "jawLeft", "jawRight"))
    if len(jaw):
        jaw_pts = vertices[jaw]
        centre_xz = torch.tensor([jaw_pts[:, 0].mean(), 0.0, jaw_pts[:, 2].mean()],
                                 dtype=torch.float64)
        rel = jaw_pts - torch.tensor([centre_xz[0], jaw_pts[:, 1].mean(), centre_xz[2]],
                                     dtype=torch.float64)
        angle = torch.atan2(rel[:, 0], rel[:, 2])
        lo, hi = float(angle.min()), float(angle.max())
        band_half = (hi - lo) / 34.0
        for i in range(17):
            want = lo + (hi - lo) * (i / 16.0)
            band = torch.abs(angle - want) <= band_half
            if not band.any():
                j = int(torch.argmin(torch.abs(angle - want)))
            else:
                band_ys = jaw_pts[band, 1]
                band_j = int(torch.argmin(band_ys))
                j = int(torch.nonzero(band, as_tuple=False).flatten()[band_j])
            emit("face_kpt_%d" % i, jaw_pts[j], jaw)
    else:
        missing.extend("face_kpt_%d" % i for i in range(17))

    for side, names, indices in (
        ("right", ("browDownRight", "browOuterUpRight"), range(17, 22)),
        ("left", ("browDownLeft", "browOuterUpLeft"), range(22, 27)),
    ):
        brow = region_vertices(targets, names)
        if not len(brow):
            missing.extend("face_kpt_%d" % k for k in indices)
            continue
        brow_pts = vertices[brow]
        xs = brow_pts[:, 0]
        lo, hi = float(xs.min()), float(xs.max())
        for offset, k in enumerate(indices):
            want_x = lo + (hi - lo) * (offset / 4.0)
            band = torch.abs(xs - want_x) <= max((hi - lo) / 10.0, 0.003)
            if not band.any():
                missing.append("face_kpt_%d" % k)
                continue
            band_pts = brow_pts[band]
            j = int(torch.argmax(band_pts[:, 1]))
            emit("face_kpt_%d" % k, band_pts[j], brow)

    nose = region_vertices(targets, ("noseSneerLeft", "noseSneerRight"))
    if len(nose):
        nose_pts = vertices[nose]
        ys = nose_pts[:, 1]
        xs = nose_pts[:, 0]
        y_lo, y_hi = float(ys.min()), float(ys.max())
        centre_x_span = (float(xs.max()) - float(xs.min())) / 5.0
        central = torch.abs(xs - float(xs.mean())) <= centre_x_span
        bridge_verts = nose[central]
        bridge_pts = vertices[bridge_verts]
        bridge_ys = bridge_pts[:, 1]
        for offset in range(4):
            want_y = y_hi - (y_hi - y_lo) * (offset / 3.0)
            band = torch.abs(bridge_ys - want_y) <= max((y_hi - y_lo) / 10.0, 0.002)
            if not band.any():
                j = int(torch.argmin(torch.abs(bridge_ys - want_y)))
                emit("face_kpt_%d" % (27 + offset), bridge_pts[j], bridge_verts)
                continue
            band_pts = bridge_pts[band]
            j = int(torch.argmax(band_pts[:, 2]))
            emit("face_kpt_%d" % (27 + offset), band_pts[j], bridge_verts)
        tip_band = torch.abs(ys - y_lo) <= max((y_hi - y_lo) / 8.0, 0.003)
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

    for side_names, indices, side_sign in (
        (("eyeBlinkRight", "eyeSquintRight", "eyeWideRight"), range(36, 42), -1.0),
        (("eyeBlinkLeft", "eyeSquintLeft", "eyeWideLeft"), range(42, 48), +1.0),
    ):
        eye = region_vertices(targets, side_names)
        if not len(eye):
            missing.extend("face_kpt_%d" % k for k in indices)
            continue
        eye_pts = vertices[eye]
        cx = eye_pts[:, 0].mean()
        cy = eye_pts[:, 1].mean()
        rel_x = eye_pts[:, 0] - cx
        rel_y = eye_pts[:, 1] - cy
        angle = torch.atan2(rel_y, rel_x * side_sign)
        wants = [0.0, torch.pi / 3, 2 * torch.pi / 3, torch.pi,
                 -2 * torch.pi / 3, -torch.pi / 3]
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
    outer = region_vertices(targets, outer_lip_names)
    inner = region_vertices(targets, inner_lip_names)

    def sample_ring(vert_indices, count, indices):
        if not len(vert_indices):
            missing.extend("face_kpt_%d" % k for k in indices)
            return
        pts = vertices[vert_indices]
        cx = pts[:, 0].mean()
        cy = pts[:, 1].mean()
        angle = torch.atan2(pts[:, 1] - cy, -(pts[:, 0] - cx))
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
    ap.add_argument("--show", type=int, default=12)
    ap.add_argument("--worst-cap-mm", type=float, default=10.0)
    ap.add_argument("--negative-control", action="store_true")
    args = ap.parse_args()

    anny_root = args.anny_root
    if not anny_root:
        import anny
        anny_root = os.path.dirname(anny.__file__)

    model = build_model()
    vertex_count = int(model.template_vertices.shape[0])
    targets = load_targets(anny_root)

    if args.negative_control:
        global region_vertices
        original = region_vertices

        def loose(targets, names, quantile=None):
            hits = set()
            for name in names:
                idx, deltas = targets[name]
                magnitude = torch.linalg.norm(deltas, dim=1)
                keep = idx[magnitude >= 0.0015]
                hits.update(int(i) for i in keep)
            return torch.tensor(sorted(hits), dtype=torch.long)

        region_vertices = loose
        try:
            weights, report, missing = face_anchors(model, targets)
        finally:
            region_vertices = original
    else:
        weights, report, missing = face_anchors(model, targets)

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
                   "targets_read": sorted(targets),
                   "have": sorted(weights), "missing": missing,
                   "worst_cap_mm": args.worst_cap_mm,
                   "spread_metres": {label: round(s, 6) for label, s in report},
                   "spread_summary_mm": {"median": round(median * 1000, 2),
                                         "p90": round(p90 * 1000, 2),
                                         "worst": round(worst * 1000, 2),
                                         "worst_label": worst_label}}, fh, indent=2)

    report.sort(key=lambda r: -r[1])
    print("%d of 68 face keypoints, vertex width %d" % (len(weights), vertex_count))
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
