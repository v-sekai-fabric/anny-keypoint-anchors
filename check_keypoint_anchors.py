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

    # Positional-prior control for the face landmarks. Spread was a proxy; small blend
    # spread does not certify that a landmark lands where iBUG says it should. v2 shipped
    # geographically-clustered anchors (jawline compressed to chin height, eyes/mouth
    # X-compressed) with median spread 4.0 mm and 6/6 controls passing. Assert placement
    # directly: jawline monotone in X with chin at min Z; eyes lateral of a threshold;
    # left-mouth-corner lateral of nose tip and right-mouth-corner on the opposite side.
    v2_pth = pathlib.Path(__file__).parent / "face68_v2_snapshot.pth"
    print("positional-prior control")
    v3 = _positions_for(pathlib.Path(__file__).parent / "face68.pth")
    v3_bad = _positional_priors_fail(v3)
    if v3_bad:
        fails.append("v3 positions fail placement priors")
        for b in v3_bad:
            print("  BAD v3 face68.pth: %s" % b)
    else:
        print("  ok  v3 face68.pth: jawline monotone-X + chin-min-Z, eyes lateral, mouth corners flank nose")

    wholebody_pth = pathlib.Path(__file__).parent / "wholebody133.pth"
    if wholebody_pth.is_file():
        wb = _positions_for(wholebody_pth)
        wb_bad = _body_hand_priors_fail(wb)
        if wb_bad:
            fails.append("wholebody133 body/hand placement priors")
            for b in wb_bad:
                print("  BAD wholebody133.pth: %s" % b)
        else:
            print("  ok  wholebody133.pth: 17 body + 6 feet + 42 hands placement priors "
                  "(L/R mirror, Z hierarchy, fingertip distal, heel behind toes)")
    if v2_pth.is_file():
        v2 = _positions_for(v2_pth)
        v2_bad = _positional_priors_fail(v2)
        if not v2_bad:
            fails.append("v2 snapshot passes placement priors — control cannot certify v3")
            print("  BAD v2 snapshot: passed placement priors — v3 gate cannot distinguish "
                  "known-broken from known-good, and rule 2 is violated")
        else:
            print("  ok  v2 snapshot: %d placement priors fail as expected "
                  "(the known-broken input)" % len(v2_bad))
    else:
        print("  NOT-MEASURED v2 snapshot not at %s: skipping the v3-vs-v2 negative control "
              "for placement priors. Ship the snapshot as face68_v2_snapshot.pth alongside "
              "face68.pth so this control runs." % v2_pth)

    print("\n%d failed" % len(fails))
    return 1 if fails else 0


def _positions_for(pth_path: pathlib.Path) -> dict[str, tuple[float, float, float]]:
    import anny
    from anny.models.model_data import TopologyConfig

    m = anny.Anny(topology=TopologyConfig(base_mesh="makehuman",
                                          remove_unattached_vertices=False))
    V = m.template_vertices.to(torch.float64)
    pth = torch.load(pth_path, weights_only=True)
    out = {}
    for k, w in pth.items():
        pos = (w.to(torch.float64).unsqueeze(-1) * V).sum(dim=0)
        out[k] = (float(pos[0]), float(pos[1]), float(pos[2]))
    return out


def _positional_priors_fail(pos: dict[str, tuple[float, float, float]]) -> list[str]:
    """Rules taken from iBUG's own geometry: jawline is right-ear-to-left-ear with chin at
    the lowest Z, eyes sit lateral of ~20 mm from midline, mouth corners flank the nose tip
    by at least a pencil's width. Returns the list of failures; empty list is a pass."""
    bad = []
    jaw_xs = [pos["face_kpt_%d" % i][0] for i in range(17)]
    for i in range(16):
        if jaw_xs[i + 1] <= jaw_xs[i]:
            bad.append("jawline X not monotone at kpt_%d -> kpt_%d (%.4f -> %.4f)"
                       % (i, i + 1, jaw_xs[i], jaw_xs[i + 1]))
    jaw_zs = [pos["face_kpt_%d" % i][2] for i in range(17)]
    min_z_i = jaw_zs.index(min(jaw_zs))
    if min_z_i != 8:
        bad.append("chin (min Z of jawline) is at kpt_%d, not kpt_8" % min_z_i)

    # Eyes lateral: right eye (36-41) all X < 0 and |X| > 20 mm; left eye (42-47) mirror.
    for k in range(36, 42):
        x = pos["face_kpt_%d" % k][0]
        if x >= 0 or abs(x) < 0.020:
            bad.append("right eye kpt_%d X=%.4f is not lateral (X<0 and |X|>=0.020)" % (k, x))
    for k in range(42, 48):
        x = pos["face_kpt_%d" % k][0]
        if x <= 0 or abs(x) < 0.020:
            bad.append("left eye kpt_%d X=%.4f is not lateral (X>0 and |X|>=0.020)" % (k, x))

    # Mouth corners kpt_48 (subject-right) and kpt_54 (subject-left) sit farther from the
    # midline than nose tip kpt_30 by at least a pencil (7 mm).
    nose_x = pos["face_kpt_30"][0]
    r_corner_x = pos["face_kpt_48"][0]
    l_corner_x = pos["face_kpt_54"][0]
    if r_corner_x >= 0 or abs(r_corner_x - nose_x) < 0.007:
        bad.append("right mouth corner kpt_48 X=%.4f is not lateral of nose_x=%.4f by 7 mm"
                   % (r_corner_x, nose_x))
    if l_corner_x <= 0 or abs(l_corner_x - nose_x) < 0.007:
        bad.append("left mouth corner kpt_54 X=%.4f is not lateral of nose_x=%.4f by 7 mm"
                   % (l_corner_x, nose_x))
    return bad


def _body_hand_priors_fail(pos: dict[str, tuple[float, float, float]]) -> list[str]:
    """Placement priors for the 65 non-face anchors (17 body + 6 feet + 42 hands). Rule 7
    generalisation of the face bug: same axis-map assumption on the sibling scripts, so the
    hand and body positions get a placement gate too, not just the face."""
    bad = []
    if not pos:
        return bad
    for side, sign in (("left", +1.0), ("right", -1.0)):
        for name in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle",
                     "big_toe", "small_toe", "heel"):
            key = "%s_%s" % (side, name)
            if key not in pos:
                continue
            x = pos[key][0]
            if x * sign <= 0:
                bad.append("%s X=%.4f is not on subject-%s side" % (key, x, side))

    z_order = ("nose", "left_shoulder", "left_hip", "left_knee", "left_ankle", "left_heel")
    ordered = [k for k in z_order if k in pos]
    for a, b in zip(ordered, ordered[1:]):
        if pos[a][2] <= pos[b][2]:
            bad.append("%s Z=%.4f is not above %s Z=%.4f" % (a, pos[a][2], b, pos[b][2]))

    for side in ("left", "right"):
        s = "%s_shoulder" % side
        e = "%s_elbow" % side
        w = "%s_wrist" % side
        if all(k in pos for k in (s, e, w)):
            if not (pos[s][2] > pos[e][2] > pos[w][2]):
                bad.append("%s arm Z order shoulder>elbow>wrist violated: %.3f/%.3f/%.3f"
                           % (side, pos[s][2], pos[e][2], pos[w][2]))
        h = "%s_hip" % side
        k = "%s_knee" % side
        a = "%s_ankle" % side
        if all(x in pos for x in (h, k, a)):
            if not (pos[h][2] > pos[k][2] > pos[a][2]):
                bad.append("%s leg Z order hip>knee>ankle violated: %.3f/%.3f/%.3f"
                           % (side, pos[h][2], pos[k][2], pos[a][2]))

    for side in ("left", "right"):
        big = pos.get("%s_big_toe" % side)
        heel = pos.get("%s_heel" % side)
        if big and heel and big[1] >= heel[1]:
            bad.append("%s_big_toe Y=%.4f is not forward of %s_heel Y=%.4f"
                       % (side, big[1], side, heel[1]))

    for side, sign in (("left", +1.0), ("right", -1.0)):
        for finger in ("thumb", "forefinger", "middle_finger", "ring_finger", "pinky_finger"):
            root = pos.get("%s_hand_root" % side)
            tip = pos.get("%s_%s4" % (side, finger))
            j3 = pos.get("%s_%s3" % (side, finger))
            if root and tip and j3:
                d_tip = sum((tip[i] - root[i]) ** 2 for i in range(3)) ** 0.5
                d_j3 = sum((j3[i] - root[i]) ** 2 for i in range(3)) ** 0.5
                if d_tip <= d_j3:
                    bad.append("%s_%s tip is not distal of joint 3 (root-dist %.3f vs %.3f)"
                               % (side, finger, d_tip, d_j3))
            if tip and tip[0] * sign <= 0:
                bad.append("%s_%s4 X=%.4f is not on subject-%s side"
                           % (side, finger, tip[0], side))
    return bad


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
