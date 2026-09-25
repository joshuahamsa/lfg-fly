# REPORT: the fly's taste, `fly-v1`

Written 2026-09-25T08:20:30+00:00 by `fly train --round r1` on `cuda`. Every accuracy is pairwise, on held-out look families the head never saw; every interval is a 95% family-bootstrap CI (2,000 resamples) and every fly − model difference is a paired bootstrap over the same families (spec §3.4).

## Gate B

**Gate B: PASS** — the taste head's held-out accuracy is 0.649 (95% CI 0.607–0.689, n = 584 pairs over 200 families). The go-live gate needs the CI lower bound above 0.55; 0.607 is above it. The fly goes live as a chooser (spec §3.4).

Best-of-6 agreement against the 1/6 chance rate, the gate's second criterion, is round 2's measurement and has not been made; this verdict rests on the CI criterion alone.

## The fly

- Brain: rate, g_syn 1.0, bias 0.1, 60 steps, code `all-sensory-brain`, g_in 2.0, simulated at c_ref 0.7 (taste is rarity-invariant by construction, spec §2).
- Graph hash `f2f62e0a0bbf145e…`, ≥3 synapses, 165,122 neurons, 10,511,038 edges.
- Held-out 0.649 (0.607–0.689); training accuracy 0.766; CV accuracy 0.618; L2 λ = 0.1; readout features 2129.
- The shipped head (fitted on the training pairs, the model the protocol scored) gets 0.649 (0.607–0.689) on the same held-out pairs and agrees with the protocol's per-pair verdicts on 100.0% of them; taste sd 2.262 (β's unit in the decision score).
- Activity: mean active fraction 0.824, readout active 0.984; 34.28 s of simulation.

## Labels

- Source `data/critic/r1.jsonl` (round r1): 3,000 pairs judged, 78 dropped as position-bias flips, 2,922 kept over 1000 look families (3920 distinct looks); near pairs 0.584.
- Split by family: 2,338 training pairs, 584 held-out pairs in 200 families (≥ 20% of families).
- The critic's strength (1–3) is the sample weight; the L2 strength is chosen by 5-fold cross-validation grouped by family.

## Controls (spec §3.4)

| # | Model | Held-out | 95% CI | Fly − model | 95% CI | Fly beats it |
|---|---|---|---|---|---|---|
| 1 | Random mover | 0.500 | — | +0.149 | fly's own CI 0.607–0.689 | yes |
| 2 | No brain: logistic regression on one-hot traits + mean RGB + 8-bin luminance histogram | 0.668 | 0.628–0.706 | -0.019 | -0.047 to +0.009 | no |
| 3 | MLP (256 hidden) on the same raw features | 0.644 | 0.602–0.683 | +0.005 | -0.028 to +0.037 | no |
| 4 | Rewired fly (degree-preserving; 22,153 repair swaps) | 0.642 | 0.602–0.681 | +0.007 | -0.022 to +0.036 | no |
| 5 | Sign-shuffled fly (NT labels permuted across neurons) | 0.651 | 0.611–0.689 | -0.002 | -0.031 to +0.027 | no |
| 6 | Eyes-only (on the tonic baseline): Phase 0's best `eyes` setting, rate, g_syn 2.0, bias 0.1, 60 steps, code `eyes`, g_in 0.5, planted-taste held-out 0.593 | 0.608 | 0.565–0.652 | +0.041 | -0.005 to +0.086 | no |
| 6 | Nose-only: Phase 0's best `nose` setting, rate, g_syn 1.0, bias 0.1, 60 steps, code `nose`, g_in 2.0, planted-taste held-out 0.648 | 0.539 | 0.500–0.579 | +0.110 | +0.060 to +0.161 | yes |
| 7 | ≥5-synapse graph (hash 3ce5d7f715b5af35…, 6,235,682 edges) | 0.644 | 0.600–0.683 | +0.005 | -0.012 to +0.024 | no |
| 8 | Critic agreement a / implied ceiling | a = 0.740 → ceiling 0.846 | — | — | — | — |

## What the controls say

A control is beaten when the paired difference's 95% CI lower bound is above 0. Beating the controls is not required for the gate (spec §3.4); whether the fly does is stated here either way.

- **Does the fly beat its own raw inputs?** No: the fly does not beat its own raw inputs. The no-brain regression scores 0.668 (0.628–0.706) against the fly's 0.649 (0.607–0.689); paired difference fly − model -0.019 (95% CI -0.047 to +0.009).
- **Against a conventional net:** the MLP scores 0.644 (0.602–0.683) against the fly's 0.649 (0.607–0.689); paired difference fly − model +0.005 (95% CI -0.028 to +0.037). The fly does not beat it.
- **Does the real wiring matter?** No: the wiring does not matter for this taste. The rewired twin (every neuron's in- and out-degree preserved) scores 0.642 (0.602–0.681) against the fly's 0.649 (0.607–0.689); paired difference fly − model +0.007 (95% CI -0.022 to +0.036). Every published fly-game control so far found no advantage for real wiring, and neither does this one.
- **Do the signs matter?** No: the signs do not matter. The sign-shuffled twin scores 0.651 (0.611–0.689) against the fly's 0.649 (0.607–0.689); paired difference fly − model -0.002 (95% CI -0.031 to +0.027).
- **Which sense carries taste?** Eyes-only 0.608 (0.565–0.652), nose-only 0.539 (0.500–0.579), the fly (all senses) 0.649: the eyes carry more of it than the nose. Paired differences fly − eyes +0.041, fly − nose +0.110.
- **Threshold sensitivity:** the ≥5-synapse graph scores 0.644 (0.600–0.683) against the fly's 0.649 (0.607–0.689); paired difference fly − model +0.005 (95% CI -0.012 to +0.024). The threshold moves the number by -0.005.

The fly beats 1 of the 7 control(s) that ran.

## The critic's labels (spec §3.2 quality control)

- Pairs judged: 3000; A chosen in 0.501 of them (position bias shows as a share far from 0.5).
- Flip rate 0.260: 78 of 300 flip-checked pairs got the opposite verdict with the sides swapped; those pairs were dropped from training and scoring.
- Agreement a = 0.740 over 300 pairs judged twice; implied ceiling (1+√(2a−1))/2 = 0.846. Agreement is a lower bound on what a model can reach against one critic; the ceiling is exact under uniform label noise and an upper estimate when pair difficulty varies.
- The fly's held-out accuracy 0.649 is below the implied ceiling 0.846.
- Strength histogram (1–3): 1: 1636, 2: 1355, 3: 9.

## The rarity head

Not trained here. The rarity head (spec §2) is fitted nightly by `fly retrain` from a supply snapshot and learns an additive, already-solved function (Σ n_live/freq over the nine traits). Rarity is the fly's incentive; taste is where the fly is tested, and this report scores taste only.

## Checkpoint identity (spec §2 Checkpoint and versioning)

- Checkpoint: `checkpoints/fly-v1` (version `fly-v1`), created 2026-09-25T16:26:50+00:00.
- Graph hash `f2f62e0a0bbf145e285cfe6cd3b6ec9a02d328cd36639b675d491b2efc3bed1b`; threshold ≥3 synapses; 502 neurons without an NT prediction (treated as `unclear`).
- Feathers (sha256): annotations `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2`, neurotransmitters `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621`, weights `9b3beab17bad5f618be3f2c02d3139a8d07b822565919c013f1e5506d93e604b`.
- Column assignment hash: `7ce7d9fa4895d00301b09a01509bd785ea7f92bc790a3bef0c023578007624b1`; catalog hash `cbd9184be0c73a20cc18dc711220d2afd8bfafa7b0ef2dd89744911eef62c70f`.
- Brain setting: rate, g_syn 1.0, bias 0.1, 60 steps, code `all-sensory-brain`, g_in 2.0; c_ref 0.7; sense codes {'allsens': 'v1', 'odor': 'v2'}; LFG trait_config commit `0415a63afdcea11156b57169d34e38bb8d377adc`.
- Sign map and the head's standardization stats and weights are in the checkpoint (`manifest.json`, `head.npz`, `head.json`).

## Reproduce

`fly train --round r1 --version fly-v1 --device cuda --min-syn 3` (seed 0; results in `data/train/fly-v1/results.json`).

---

Connectome: Janelia FlyEM MaleCNS v1.0 (CC BY 4.0), Berg et al., Cell 189(18):5504–5526.e15 (2026), doi:10.1016/j.cell.2026.08.015. Changes: thresholded at ≥3 synapses, signed by predicted transmitter, photoreceptor columns derived from synaptic partners, simulated. No endorsement by HHMI/Janelia, Cambridge, MRC-LMB or Google is implied.
