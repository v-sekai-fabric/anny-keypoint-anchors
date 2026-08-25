"""The 42 COCO-WholeBody hand keypoints, as vertex weights over ANNY's base mesh.

WHY THESE CANNOT BE COPIED AND CANNOT COME FROM BONES ALONE. ANNY carries 20 bones per hand --
`wrist`, fifteen phalanx joints `finger1-1` to `finger5-3`, and four metacarpals -- while
COCO-WholeBody wants 21 per hand: a root, then four points per digit. The phalanx joints supply
three of the four (MCP, PIP, DIP). **Every fingertip is missing**, because a tip is the far end
of the last bone rather than a joint, and there is no fifth metacarpal to borrow.

AND THE OBVIOUS FIX IS NOT AVAILABLE HERE. `get_bone_ends` takes rest bone tails and would give
a posed tip directly, but on the makehuman topology `template_bone_tails` is **None** --
measured, not assumed -- so there are no tails to pass it. The tip has to come off the surface.

SO A TIP IS THE FURTHEST SKINNED POINT ALONG THE DIGIT'S OWN AXIS. For the distal phalanx of
each digit, take the vertices the rig actually skins to that bone, project them onto the axis
running from the middle phalanx's head through the distal phalanx's head, and keep the extreme.
That is a construction from the mesh and the rig rather than a hand-placed guess, it moves
correctly when the hand is posed because the vertices are skinned, and it is reproducible.

EVERY OUTPUT IS A CONVEX BLEND, WHICH IS WHAT MAKES IT A KEYPOINT. `KeypointsRegressor`
multiplies a (K, V) weight matrix by the model's vertices, so a row that sums to one returns a
point inside the hull of the vertices it touches -- on the skin, following the skin. A row that
does not sum to one returns a point that is nowhere in particular, and nothing about its shape
says so, which is why `build_wholebody_anchors.py` refuses those.

CANDIDATES ARE RESTRICTED BY THE RIG'S OWN SKINNING, not by distance. `vertex_bone_weights` is
(19158, 12) -- up to twelve bones per vertex -- so "which vertices belong to this finger" is a
question the rig already answers. Restricting by distance instead would let a fingertip anchor
onto the neighbouring finger whenever two digits touch, and in the rest pose they very nearly
do.

    python hand_anchors.py --out .        writes hands42.pth and prints the spread per point
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

# COCO-WholeBody digit order, and ANNY's finger numbering. `finger1` is the thumb.
DIGITS = (("thumb", 1), ("forefinger", 2), ("middle_finger", 3),
          ("ring_finger", 4), ("pinky_finger", 5))

# How many vertices a keypoint blends over. One vertex would be exact and brittle -- it moves
# with a single skinning weight and lands on whatever the mesh happens to have there. Eight is
# enough to average out that particular vertex without smearing the point across a joint; the
# spread printed per keypoint is what says whether it was too many.
BLEND_K = 8

# A vertex counts as belonging to a bone above this skinning weight. Below it the vertex is
# mostly driven by something else and would drag the keypoint toward the neighbouring segment.
MIN_SKIN_WEIGHT = 0.2


def build_model():
    """The makehuman topology, because that is the space `coco.pth`'s weights are in."""
    import anny
    from anny.models.model_data import TopologyConfig

    return anny.Anny(
        topology=TopologyConfig(base_mesh="makehuman", remove_unattached_vertices=False))


def skinned_to(model, bone_index, minimum=MIN_SKIN_WEIGHT):
    """Vertices the rig drives with this bone, by the rig's own weights."""
    hit = (model.vertex_bone_indices == bone_index) & (model.vertex_bone_weights >= minimum)
    return torch.nonzero(hit.any(dim=1), as_tuple=False).flatten()


def blend_at(vertices, candidates, target, k=BLEND_K):
    """A convex blend over the `k` candidate vertices nearest `target`, by inverse distance.

    Returns the weight row and the spread -- the distance from the target to the furthest
    vertex the blend uses. The spread is reported rather than swallowed because it is the
    number that says whether the keypoint is a point or a smear: a fingertip whose blend
    reaches two centimetres, about a stacked penny and a half, is not a fingertip.
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


def digit_tip(vertices, model, labels, side, digit):
    """The far end of the distal phalanx, along the digit's own axis.

    The axis is middle-phalanx head to distal-phalanx head, which points down the finger by
    construction. Projecting the distal phalanx's skinned vertices onto it and taking the
    maximum gives the tip without needing a bone tail, and without needing a threshold: it is
    an extreme, not a cutoff.
    """
    distal = labels.index("finger%d-3.%s" % (digit, side))
    middle = labels.index("finger%d-2.%s" % (digit, side))
    heads = model.template_bone_heads
    axis = heads[distal] - heads[middle]
    norm = torch.linalg.norm(axis)
    if float(norm) < 1e-9:
        return None, None
    axis = axis / norm
    candidates = skinned_to(model, distal)
    if not len(candidates):
        return None, None
    along = (vertices[candidates] - heads[distal]) @ axis
    return vertices[candidates[int(torch.argmax(along))]], candidates


def hand_anchors(model):
    """The 42 rows, and a report row per keypoint. Nothing is emitted for a point that has no
    candidate vertices; a named gap is recoverable and a filled one is not."""
    labels = list(model.bone_labels)
    vertices = model.template_vertices.to(torch.float64)
    weights, report, missing = {}, [], []

    for coco_side, anny_side in (("left", "L"), ("right", "R")):
        wrist = "wrist.%s" % anny_side
        if wrist not in labels:
            missing.append("%s_hand_root" % coco_side)
        else:
            index = labels.index(wrist)
            row, spread = blend_at(vertices, skinned_to(model, index),
                                   model.template_bone_heads[index].to(torch.float64))
            if row is None:
                missing.append("%s_hand_root" % coco_side)
            else:
                weights["%s_hand_root" % coco_side] = row
                report.append(("%s_hand_root" % coco_side, spread))

        for name, digit in DIGITS:
            # Joints 1 to 3 are the heads of the three phalanges, in order down the digit.
            for joint in (1, 2, 3):
                label = "%s_%s%d" % (coco_side, name, joint)
                bone = "finger%d-%d.%s" % (digit, joint, anny_side)
                if bone not in labels:
                    missing.append(label)
                    continue
                index = labels.index(bone)
                row, spread = blend_at(vertices, skinned_to(model, index),
                                       model.template_bone_heads[index].to(torch.float64))
                if row is None:
                    missing.append(label)
                    continue
                weights[label] = row
                report.append((label, spread))

            # Joint 4 is the tip, which no bone head reaches.
            label = "%s_%s4" % (coco_side, name)
            target, candidates = digit_tip(vertices, model, labels, anny_side, digit)
            if target is None:
                missing.append(label)
                continue
            row, spread = blend_at(vertices, candidates, target)
            if row is None:
                missing.append(label)
                continue
            weights[label] = row
            report.append((label, spread))

    return weights, report, missing


def household(metres):
    """CLAUDE.md's reporting rule: a millimetre figure does not say whether an error matters."""
    mm = metres * 1000.0
    for size, name in ((1.52, "a penny"), (7.0, "a pencil"), (10.5, "a AAA battery"),
                       (14.5, "a AA battery"), (21.2, "a nickel"), (42.7, "a golf ball"),
                       (57.0, "an adult wrist"), (66.0, "a soda can")):
        if mm <= size * 1.5:
            return "%.1f mm, about %s" % (mm, name)
    return "%.1f mm, wider than a soda can" % mm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=".")
    ap.add_argument("--show", type=int, default=8, help="how many spreads to print")
    args = ap.parse_args()

    model = build_model()
    vertex_count = model.template_vertices.shape[0]
    weights, report, missing = hand_anchors(model)

    bad = [(label, float(row.sum())) for label, row in weights.items()
           if abs(float(row.sum()) - 1.0) > 1e-3]
    if bad:
        sys.exit("FAIL  %d row(s) are not convex blends, e.g. %s sums to %.6f"
                 % (len(bad), bad[0][0], bad[0][1]))

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(weights, out / "hands42.pth")
    with open(out / "hands42.json", "w", encoding="utf-8") as fh:
        json.dump({"topology": {"base_mesh": "makehuman", "vertex_count": vertex_count},
                   "blend_k": BLEND_K, "min_skin_weight": MIN_SKIN_WEIGHT,
                   "have": sorted(weights), "missing": missing,
                   "spread_metres": {label: round(s, 6) for label, s in report}}, fh, indent=2)

    report.sort(key=lambda r: -r[1])
    print("%d of 42 hand keypoints, vertex width %d" % (len(weights), vertex_count))
    print("widest blends (target to furthest vertex used):")
    for label, spread in report[:args.show]:
        print("  %-24s %s" % (label, household(spread)))
    if report:
        median = sorted(s for _, s in report)[len(report) // 2]
        print("median blend spread   %s" % household(median))
    print("%d missing: %s" % (len(missing), ", ".join(missing) if missing else "none"))
    print("wrote %s and %s" % (out / "hands42.pth", out / "hands42.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
