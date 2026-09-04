# anny-keypoint-anchors

Where the COCO-WholeBody 133 keypoints sit on ANNY's mesh, as vertex weights, checked
against the data.

## Why this is a repository and not a function

ANNY already ships the mapper. `anny.keypoints.KeypointsRegressor` regresses named keypoints
from vertices by a linear blend, and `KeypointsRegressor.coco(model)` loads
`anny/data/keypoints/coco.pth`. Nothing here reimplements it, and reimplementing it would be
the mistake: the regressor is Apache-2.0, it is the interface the fit already uses, and a
second copy of a mapper is a copy that drifts.

What is missing is the **data it loads**. Measured, rather than assumed, from the installed
package:

| property | value |
| --- | --- |
| entries in `coco.pth` | **23** |
| labels | `nose … right_ankle` (17), then `left_big_toe, right_big_toe, left_small_toe, right_small_toe, left_heel, right_heel` (6) |
| weight vector width | **19,158**, float32, dense |
| weights per label | sum to 1.0 |
| non-zero vertices | 78 (`left_eye`) to 4,105 (`left_ear`) |

Those 23 are exactly COCO-WholeBody indices **0–22** — body then feet — in its published
order. So the gap is precise and it is not the mapper: indices **23–90** (68 face landmarks)
and **91–132** (21 per hand) have no weights, and this repository is where they get built and
checked.

It sits on `2-contract` for the same reason [`hm08-partition`](../hm08-partition) does. Both
are definitions over the same 19,158-vertex base mesh that other repositories have to agree
on, rather than models that run or corpora that are consumed.

## Why the 133 cannot come from bones

Measured on ANNY's 104 bones:

- **40 hand bones**, twenty per hand: `wrist`, `finger1-1 … finger5-3` (fifteen phalanx
  joints), and `metacarpal1 … metacarpal4`.
- COCO-WholeBody wants **21 per hand**: the wrist, then four points per finger — MCP, PIP,
  DIP and **TIP**.

The phalanx joints supply MCP, PIP and DIP. **Every fingertip is missing**, because a tip is
the far end of the last bone rather than a joint, and there are only four metacarpals for five
fingers. So bones give sixteen of the twenty-one and the remaining five are surface points.
`get_bone_ends` can produce a posed tail, but a tail is a rig construct that moves with a
convention, and a vertex weight is a point on the skin that does not — which is the whole
reason `KEYPOINT_VERTEX_ANCHOR` exists beside `KEYPOINT_BONE_ANCHOR` in the corpus schema.

The 68 face landmarks are surface points for a blunter reason, and an earlier version of this
paragraph got it wrong. It said ANNY's facial action bones drive the face. **There are no
facial action bones.** Measured on the 104: the only head bones are `neck01`, `neck02`,
`neck03`, `head`, `eye.L` and `eye.R`. `jaw`, `lip`, `brow` and the rest are blendshape
targets under `data/faceunits01`, not joints. So the face cannot be anchored to a rig at all,
and the 68 landmarks need vertex correspondence against the hm08 topology.

The retraction stays next to what it retracts, because a reader who knows the rig has no jaw
bone will not go looking for one.

## The topology trap

A `vertex_id` means nothing without the mesh it indexes. ANNY's two topologies share **zero**
vertices — 19,158 for `base_mesh="makehuman"` against 13,718 for the body topology — so a
weight vector of the wrong width does not fail loudly, it fails at a vertex index.

`coco.pth` is 19,158 wide. `anny_rig.build_corpus_model()` returns **13,718** vertices;
`loop1_fit.build_model()` returns 19,158. Those are different meshes and only one of them can
be multiplied by these weights. Every file here records its topology, and
`check_keypoint_anchors.py` fails if a weight vector's width disagrees with the model it is
given.

## Contents

- `wholebody133.json` — the wire format: the 133 labels in COCO-WholeBody's published order,
  the skeleton links, and the OpenPose/mmpose colours. Not OKHSL. A pose control is consumed
  by a model trained on one specific rendering, so the order, the links and the colours are
  interface, not design.
- `build_wholebody_anchors.py` — writes `wholebody133.pth` in the format
  `KeypointsRegressor.load_precomputed` reads: `{label: (V,) weights summing to 1}`. It copies
  indices 0–22 from ANNY's own `coco.pth` rather than re-deriving them, so the body and feet
  cannot drift from the set the fit already verifies, and merges `hands42.pth` and
  `face68.pth` by name.
- `hand_anchors.py` — builds the 42 hand points and writes `hands42.pth` and `hands42.json`.
  Candidate vertices are restricted by the rig's own skinning weights rather than by distance,
  because two fingers nearly touch in the rest pose and a distance-restricted blend would
  anchor a fingertip onto its neighbour.
- `face_anchors.py` — builds the 68 iBUG face landmarks and writes `face68.pth` and
  `face68.json`. Candidate vertices come from the 52 facial-action blendshape targets ANNY
  ships (`anny/data/faceunits01/targets/faceunits/*.target`) — each target lists the
  vertices it moves, so a facial region is a documented vertex set rather than a distance
  cutoff on the raw mesh, and no landmark can drift onto a neighbouring region.
- `check_keypoint_anchors.py` — re-derives every number in this README from the installed
  package, with negative controls that must each fail.

## What is built

**133 of 133.** Body 17, feet 6, face 68, and both hands at 21 each. The 68 face landmarks
are the iBUG 68-point layout in COCO-WholeBody's order (`face_kpt_0` at the subject-right
jaw corner, `face_kpt_67` at the inner-mouth lower-right).

Face landmark spreads, measured as the distance from the target to the furthest vertex the
blend uses:

| | spread |
| --- | --- |
| median across the 68 | 4.0 mm, about a pencil |
| p90 across the 68 | 7.8 mm, about a pencil |
| worst, nose tip `face_kpt_29` | 9.4 mm, about a pencil |

The face uses `BLEND_K = 4` where the hand uses 8 because the face mesh is denser than
the hand and a K=8 blend drags the widest landmarks out past 20 mm. Region membership is
gated by a per-target top-decile motion filter (`CORE_QUANTILE = 0.9`) rather than a
fixed millimetre floor: a fixed floor across a region's target union admits the loudest
flesh of a large target (a brow-raiser propagates deltas into forehead and cheek) alongside
the ridge of a small one, and reducing K alone cannot recover a landmark whose candidate
set covers half the upper face. The top-decile filter is per-target and self-scaling, so
a target that mostly moves a small ridge keeps its ridge and a target that mostly moves
flesh keeps only the loudest of that flesh.

Two per-region constructions ride on top of that filter:

- **Jawline (0-16):** for each of 17 evenly spaced angles around the Y axis, the target
  is the lowest-Y vertex in the jaw region within a narrow angle band. Without the
  lowest-Y constraint the mid-jaw landmarks climb up to the cheek where the mesh happens
  to be dense; the jawbone edge is the lowest Y at each angle.
- **Nose bridge (27-30):** candidates are restricted to a central-X strip one fifth of
  the nose region's X-extent. The bridge is a midline structure, and the nose region's
  top decile reaches out to the nose-wing edges where the four-nearest blend for a bridge
  point drags in flank vertices.

`face_anchors.py --negative-control` re-runs the build with the pre-rework selector
(a per-target 1.5 mm floor over the union rather than the top-decile filter) and asserts
the spread gate REJECTS the result; a gate that passes on both good and bad inputs
certifies the defect it was written to catch. The pre-rework selector's worst comes in
at 37.5 mm, about a golf ball.

Hand blend spreads, measured as the distance from the target to the furthest vertex the blend
uses:

| | spread |
| --- | --- |
| median across the 42 | 7.8 mm, about a pencil |
| widest, `left_thumb1` and `right_thumb1` | 18.1 mm, about a AA battery |
| next, both hand roots | 14.7 mm, about a AAA battery |

The thumb base is the widest because it is a broad region rather than a joint, and the left
and right figures are identical, which is what a symmetric rest pose should give.

`template_bone_tails` is **None** on this topology, so `get_bone_ends` has no tails to work
from and a fingertip is instead the furthest vertex skinned to the distal phalanx, measured
along the axis from the middle phalanx's head through the distal phalanx's head. That is an
extreme rather than a threshold, so there is nothing to tune.

## Licence

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

`SPDX-License-Identifier: Apache-2.0 OR MIT`

The weights in `wholebody133.pth` are partly derived from ANNY's own `coco.pth`, which is
Apache-2.0 (NAVER Corp.), and the index order is checked against MMPose's dataset config,
also Apache-2.0. Both are named in `CITATION.cff`.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in this work by you shall be dual licensed as above, without any additional terms or
conditions.
