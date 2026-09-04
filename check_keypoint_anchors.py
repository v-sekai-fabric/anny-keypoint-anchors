"""Re-derive every claim in README.md from the installed package, and try to break each one.

A proof about a mapping cannot notice that the mapping disagrees with a file it never reads.
`hm08-partition` learned that the expensive way -- its Lean file proved every theorem it
stated while being wrong about the size of the mesh -- so this re-derives the numbers rather
than restating them, and ships negative controls that must each fail.

    python check_keypoint_anchors.py [--anchors .] [--self-test]

Exit code 1 if a claim is wrong or a control passes.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from build_wholebody_anchors import BODY, FEET, wholebody_labels  # noqa: E402

# The numbers README.md states. Each is re-derived below; none is trusted from here.
CLAIMED_COCO_ENTRIES = 23
CLAIMED_WIDTH = 19158
CLAIMED_BODY_TOPOLOGY_WIDTH = 13718
CLAIMED_HAND_BONES_PER_SIDE = 20
CLAIMED_WHOLEBODY_HAND_POINTS = 21


def anny_coco():
    from anny.paths import get_anny_root_dir

    path = get_anny_root_dir() / "data" / "keypoints" / "coco.pth"
    return path, torch.load(path, weights_only=True)


def check_claims() -> list[str]:
    bad = []
    _path, coco = anny_coco()

    if len(coco) != CLAIMED_COCO_ENTRIES:
        bad.append("coco.pth has %d entries, README says %d" % (len(coco), CLAIMED_COCO_ENTRIES))

    width = len(next(iter(coco.values())))
    if width != CLAIMED_WIDTH:
        bad.append("weight vectors are %d wide, README says %d" % (width, CLAIMED_WIDTH))

    for label, vector in coco.items():
        total = float(vector.sum())
        if abs(total - 1.0) > 1e-3:
            bad.append("%s sums to %.6f rather than 1" % (label, total))
            break

    # THE ORDER CLAIM, WHICH IS THE ONE WITH TEETH. README says ANNY's file order and
    # COCO-WholeBody's index order differ across the feet. If that ever stops being true the
    # copy-by-name in the builder becomes unnecessary, and if it silently starts being false
    # in the other direction a copy-by-slice would look correct. Either way it is checked.
    file_order = list(coco.keys())
    if file_order[:17] != list(BODY):
        bad.append("coco.pth's first 17 are not COCO's body order")
    file_feet = file_order[17:23]
    if file_feet == list(FEET):
        bad.append("coco.pth's foot order now MATCHES COCO-WholeBody (%s). README claims they "
                   "differ, and the builder's copy-by-name is justified by that claim."
                   % ", ".join(file_feet))

    labels = wholebody_labels()
    if len(labels) != 133:
        bad.append("the built layout has %d labels, not 133" % len(labels))
    if labels[17:23] != list(FEET):
        bad.append("built indices 17 to 22 are not the published foot order")

    return bad


def check_bones() -> list[str]:
    """The hand-bone claim: 20 per side, and no fingertip among them."""
    bad = []
    try:
        # THE PACKAGE DIRECTLY, NOT ANOTHER REPOSITORY'S HELPER. The first version of this
        # imported `anny_rig`, which lives in `6-datasource/anny-render-corpus`, so the check
        # only ran from inside that directory and reported its own claims as UNCHECKED
        # everywhere else. A contract repository that cannot be checked on its own is not a
        # contract. `anny.Anny()` with no topology argument is the body topology, which is the
        # 13,718-vertex one this claim is about.
        import anny

        model = anny.Anny()
    except Exception as error:  # noqa: BLE001
        # AN UNMET PRECONDITION IS A FAIL, NOT A SKIP. A silent skip reads exactly like a pass.
        return ["could not build an ANNY model to count bones, so the 20-per-hand and "
                "13,718-vertex claims are UNCHECKED: %s: %s" % (type(error).__name__, error)]

    names = list(model.bone_labels)
    for side in ("L", "R"):
        hand = [n for n in names if n.endswith("." + side) and
                (n.startswith(("finger", "metacarpal")) or n.startswith("wrist"))]
        if len(hand) != CLAIMED_HAND_BONES_PER_SIDE:
            bad.append("%d hand bones on side %s, README says %d"
                       % (len(hand), side, CLAIMED_HAND_BONES_PER_SIDE))

    phalanges = [n for n in names if n.startswith("finger") and n.endswith(".L")]
    per_finger = {}
    for n in phalanges:
        per_finger.setdefault(n.split("-")[0], []).append(n)
    if any(len(v) != 3 for v in per_finger.values()):
        bad.append("a left finger does not have exactly 3 phalanx joints: %s"
                   % {k: len(v) for k, v in per_finger.items()})
    # 1 wrist + 15 phalanx joints = 16, against 21 wanted. The 5 missing are the tips.
    reachable = 1 + sum(len(v) for v in per_finger.values())
    if CLAIMED_WHOLEBODY_HAND_POINTS - reachable != 5:
        bad.append("bones reach %d of %d hand points, so the shortfall is %d and README says 5"
                   % (reachable, CLAIMED_WHOLEBODY_HAND_POINTS,
                      CLAIMED_WHOLEBODY_HAND_POINTS - reachable))

    verts = model()["vertices"].shape[1]
    if verts != CLAIMED_BODY_TOPOLOGY_WIDTH:
        bad.append("build_corpus_model returns %d vertices, README says %d"
                   % (verts, CLAIMED_BODY_TOPOLOGY_WIDTH))
    # THE TRAP ITSELF, ASSERTED. If these two ever became equal the topology warning would be
    # obsolete, and leaving an obsolete warning in place is its own defect.
    if verts == CLAIMED_WIDTH:
        bad.append("the two topologies now have the same vertex count, so README's warning "
                   "about multiplying 19,158-wide weights by a 13,718-vertex mesh is stale")
    return bad


def self_test() -> int:
    """Each control breaks one thing the checks above are supposed to catch."""
    _path, coco = anny_coco()
    labels = wholebody_labels()
    fails = []

    print("negative controls")
    cases = []

    # a. A weight vector that does not sum to 1 is not a convex blend, so the regressed point
    #    is not on the mesh. Nothing about its shape says so.
    broken = {k: v.clone() for k, v in coco.items()}
    first = next(iter(broken))
    broken[first] = broken[first] * 2.0
    cases.append(("a weight vector scaled by 2",
                  lambda: abs(float(broken[first].sum()) - 1.0) <= 1e-3))

    # b. The width of the other topology. Same dtype, same sum, wrong mesh.
    wrong = torch.zeros(CLAIMED_BODY_TOPOLOGY_WIDTH)
    wrong[0] = 1.0
    cases.append(("a 13,718-wide weight vector",
                  lambda: len(wrong) == CLAIMED_WIDTH))

    # c. The foot order copied by position rather than by name. This is the exact bug the
    #    builder exists to prevent, so it must be visible.
    by_slice = list(coco.keys())[17:23]
    cases.append(("feet copied by slice from coco.pth",
                  lambda: by_slice == list(FEET)))

    # d. A duplicated label. Two indices naming one point passes every count.
    dup = list(labels)
    dup[100] = dup[99]
    cases.append(("a repeated label at index 100",
                  lambda: len(dup) == len(set(dup))))

    # e. A layout one index short. The ranges still look plausible.
    short = labels[:-1]
    cases.append(("132 labels instead of 133", lambda: len(short) == 133))

    face_short = [l for l in labels if not l.startswith("face_kpt_")] + \
        ["face_kpt_%d" % i for i in range(67)]
    cases.append(("67 face landmarks instead of 68 (iBUG is 68 exactly)",
                  lambda: len([l for l in face_short if l.startswith("face_kpt_")]) == 68))

    for label, passes in cases:
        if passes():
            fails.append(label)
            print("  BAD %s: accepted" % label)
        else:
            print("  ok  %s: rejected" % label)

    print("positive control")
    bad = check_claims() + check_bones()
    if bad:
        fails.append("the anchors as built")
        for b in bad:
            print("  BAD the anchors as built: %s" % b)
    else:
        print("  ok  the anchors as built: every README claim re-derived")

    print("\n%d failed" % len(fails))
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchors", default=str(pathlib.Path(__file__).parent))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    bad = check_claims() + check_bones()

    built = pathlib.Path(args.anchors) / "wholebody133.json"
    if built.is_file():
        meta = json.loads(built.read_text(encoding="utf-8"))
        have, missing = len(meta["have_weights"]), len(meta["missing_weights"])
        if have + missing != 133:
            bad.append("%d with weights plus %d without is not 133" % (have, missing))
        print("wholebody133.json: %d of 133 labels have weights, %d named as missing"
              % (have, missing))
    else:
        bad.append("no wholebody133.json at %s: run build_wholebody_anchors.py" % args.anchors)

    for b in bad:
        print("  BAD  %s" % b)
    print("%d problem(s)" % len(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
