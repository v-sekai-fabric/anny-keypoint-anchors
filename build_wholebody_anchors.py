"""Write the COCO-WholeBody 133 label order, and the weights that exist for it.

TWO OUTPUTS, AND THE SECOND IS DELIBERATELY INCOMPLETE.

  wholebody133.json   the wire format: 133 labels in COCO-WholeBody's published order.
  wholebody133.pth    label -> (19158,) regression weights, in the format
                      `anny.keypoints.KeypointsRegressor.load_precomputed` reads.

The `.pth` carries **23 of the 133**, because that is how many ANNY ships weights for, and the
file names the 110 it cannot fill rather than emitting a plausible guess. A landmark with no
defensible vertex left out and counted is a gap; one silently filled is a defect that nothing
downstream can find again.

THE ORDER IS NOT THE SAME ORDER, WHICH IS THE WHOLE REASON THIS COPIES BY NAME.
ANNY's `coco.pth` stores its six foot points as

    left_big_toe, right_big_toe, left_small_toe, right_small_toe, left_heel, right_heel

and COCO-WholeBody's indices 17 to 22 are

    left_big_toe, left_small_toe, left_heel, right_big_toe, right_small_toe, right_heel

Position 18 is `right_big_toe` in one and `left_small_toe` in the other. Copying the tensor by
slice would put four of the six feet on the wrong point, and nothing would raise: the shapes
agree, the weights still sum to one, and the error only appears as a body whose feet are
crossed. Every copy here is keyed by label, and the labels are asserted to exist.

THE NAMES ARE BUILT BY RULE, NOT TYPED. 133 strings typed by hand is 133 chances to
transpose one, and a transposed label in a wire format is not caught by any shape check.

    python build_wholebody_anchors.py --out .
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

# Indices 0 to 16, COCO's own order. `loop1_fit.COCO17` holds the same tuple and asserts the
# regressor agrees with it rather than trusting `coco.pth`.
BODY = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

# Indices 17 to 22. Left foot complete, then right foot: NOT left/right interleaved.
FEET = (
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
)

# Indices 23 to 90. The face is 68 landmarks with no individual names in the convention, so
# they are numbered, and the numbering is the identity: face_kpt_0 is the right jaw corner in
# the 68-point layout and nothing else.
FACE_COUNT = 68

# Indices 91 to 132. Root, then five digits of four joints each, per side.
FINGERS = ("thumb", "forefinger", "middle_finger", "ring_finger", "pinky_finger")


def wholebody_labels() -> list[str]:
    labels = list(BODY) + list(FEET)
    labels += ["face_kpt_%d" % i for i in range(FACE_COUNT)]
    for side in ("left", "right"):
        labels.append("%s_hand_root" % side)
        for finger in FINGERS:
            labels += ["%s_%s%d" % (side, finger, j) for j in range(1, 5)]
    return labels


def check_layout(labels) -> None:
    """The published ranges, asserted rather than trusted. A wire format that is one index
    out is a wire format that trains a model on the wrong point."""
    problems = []
    if len(labels) != 133:
        problems.append("%d labels, not 133" % len(labels))
    if labels[:17] != list(BODY):
        problems.append("indices 0 to 16 are not COCO's body order")
    if labels[17:23] != list(FEET):
        problems.append("indices 17 to 22 are not the published foot order")
    if len(labels[23:91]) != 68:
        problems.append("indices 23 to 90 are not 68 face landmarks")
    if labels[91] != "left_hand_root" or labels[112] != "right_hand_root":
        problems.append("the hand roots are not at 91 and 112")
    if len(labels) != len(set(labels)):
        problems.append("a label is repeated, so two indices name one point")
    if problems:
        raise ValueError("; ".join(problems))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=".")
    ap.add_argument("--source", default="", help="path to anny's coco.pth; found if omitted")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    labels = wholebody_labels()
    check_layout(labels)

    import torch

    if args.source:
        source_path = pathlib.Path(args.source)
    else:
        from anny.paths import get_anny_root_dir

        source_path = get_anny_root_dir() / "data" / "keypoints" / "coco.pth"
    if not source_path.is_file():
        sys.exit("FAIL  no coco.pth at %s" % source_path)

    have = torch.load(source_path, weights_only=True)
    width = len(next(iter(have.values())))

    # THE HANDS AND THE FACE COME FROM SIBLING BUILDERS, NOT FROM ANNY. `coco.pth` ships
    # weights for 23 labels; none of them is a hand or a face landmark, so those 42+68 are
    # built by `hand_anchors.py` and `face_anchors.py` and merged here by name. The width is
    # checked against `coco.pth`'s -- a 13,718-wide row would be the other topology and
    # would fail at a vertex index rather than loudly.
    for source in ("hands42.pth", "face68.pth"):
        source_pth = out / source
        if not source_pth.is_file():
            continue
        extra = torch.load(source_pth, weights_only=True)
        for label, vector in extra.items():
            if len(vector) != width:
                sys.exit("FAIL  %s is %d wide against coco.pth's %d: that is the other "
                         "topology, and multiplying it by this mesh fails at a vertex index"
                         % (label, len(vector), width))
            if label in have:
                sys.exit("FAIL  %s is defined in both coco.pth and %s; two sources for one "
                         "point is the drift this repository exists to prevent"
                         % (label, source))
            have[label] = vector

    # THE COPY, BY NAME. `have` is keyed by label and so is this, so a name that moved between
    # versions raises here instead of silently landing on a neighbour.
    weights, missing = {}, []
    for label in labels:
        if label in have:
            vector = have[label]
            total = float(vector.sum())
            if abs(total - 1.0) > 1e-3:
                sys.exit("FAIL  %s sums to %.6f, not 1: it is not a convex blend and the "
                         "regressed point would not lie on the mesh" % (label, total))
            weights[label] = vector
        else:
            missing.append(label)

    json_path = out / "wholebody133.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({
            "convention": "COCO-WholeBody",
            "source": "open-mmlab/mmpose configs/_base_/datasets/coco_wholebody.py",
            "count": len(labels),
            "ranges": {"body": [0, 16], "feet": [17, 22], "face": [23, 90],
                       "left_hand": [91, 111], "right_hand": [112, 132]},
            "labels": labels,
            "topology": {"base_mesh": "makehuman", "vertex_count": width},
            "have_weights": sorted(weights),
            "missing_weights": missing,
        }, fh, indent=2)

    pth_path = out / "wholebody133.pth"
    torch.save(weights, pth_path)

    by_region = {"body": 0, "feet": 0, "face": 0, "left_hand": 0, "right_hand": 0}
    bounds = {"body": (0, 16), "feet": (17, 22), "face": (23, 90),
              "left_hand": (91, 111), "right_hand": (112, 132)}
    for i, label in enumerate(labels):
        if label in weights:
            for region, (lo, hi) in bounds.items():
                if lo <= i <= hi:
                    by_region[region] += 1

    print("%d labels, vertex width %d" % (len(labels), width))
    for region, (lo, hi) in bounds.items():
        total = hi - lo + 1
        print("  %-11s %3d of %3d have weights" % (region, by_region[region], total))
    print("wrote %s" % json_path)
    print("wrote %s  (%d of %d labels)" % (pth_path, len(weights), len(labels)))
    # NAMED AND COUNTED, NOT OMITTED. An unmet precondition reported as a number is a gap
    # somebody can close; the same gap left out of the output is one nobody knows about.
    print("%d label(s) have no weights yet: %s%s"
          % (len(missing), ", ".join(missing[:6]),
             " and %d more" % (len(missing) - 6) if len(missing) > 6 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
