# Phase 0b: does the fly learn trait interactions?

Pre-registered in spec §3.0b before any run. This probe is a diagnostic, not a gate: Phase 0's verdict stands. Every number below is on the **confirmation set**, an independent instance of each taste that played no part in choosing the setting. Differences are paired (both models on the same test pairs), with 95% family-bootstrap CIs, and a reading holds iff its CI lies above 0.

## Readings

| Taste | Learns it (CI lower bound > 0.55) | Uses interactions (fly − one-hot) | Wiring matters (fly − rewired) | Beats no-brain (fly − MLP) |
|---|---|---|---|---|
| additive (Phase 0's taste, for reference) | **yes**: CI lo 0.730 | no: -0.008 (-0.030 to +0.015) | no: -0.003 (-0.027 to +0.022) | no: +0.010 (-0.018 to +0.038) |
| latent | no: CI lo 0.540 | no: -0.088 (-0.130 to -0.042) | no: -0.082 (-0.122 to -0.040) | no: -0.075 (-0.118 to -0.030) |
| visual | **yes**: CI lo 0.642 | no: -0.028 (-0.060 to +0.003) | no: +0.002 (-0.028 to +0.033) | no: -0.007 (-0.045 to +0.032) |

## Held-out accuracy

| Model | additive | latent | visual |
|---|---|---|---|
| Bayes ceiling | 0.819 | 0.835 | 0.824 |
| Additive oracle | 0.819 | 0.694 | 0.745 |
| **The fly** | **0.762 (0.730–0.795)** | **0.582 (0.540–0.625)** | **0.680 (0.642–0.720)** |
| Rewired twin | 0.765 (0.732–0.795) | 0.663 (0.627–0.702) | 0.678 (0.642–0.715) |
| Sign-shuffled twin | 0.755 (0.723–0.785) | 0.655 (0.617–0.693) | 0.693 (0.653–0.732) |
| One-hot logistic regression | 0.770 (0.738–0.800) | 0.670 (0.632–0.708) | 0.708 (0.670–0.745) |
| One-hot + pixel stats logistic regression | 0.770 (0.738–0.798) | 0.662 (0.623–0.700) | 0.705 (0.667–0.742) |
| MLP (256 hidden) on one-hot + pixel stats | 0.752 (0.717–0.783) | 0.657 (0.618–0.693) | 0.687 (0.648–0.723) |

The additive oracle is the best any additive model could do with unlimited data. The gap between it and the Bayes ceiling is the accuracy that only the interactions carry.

## Paired differences (fly − model)

| Model | additive | latent | visual |
|---|---|---|---|
| Rewired twin | -0.003 (-0.027 to +0.022) | -0.082 (-0.122 to -0.040) | +0.002 (-0.028 to +0.033) |
| Sign-shuffled twin | +0.007 (-0.017 to +0.032) | -0.073 (-0.115 to -0.030) | -0.013 (-0.042 to +0.015) |
| One-hot logistic regression | -0.008 (-0.030 to +0.015) | -0.088 (-0.130 to -0.042) | -0.028 (-0.060 to +0.003) |
| One-hot + pixel stats logistic regression | -0.008 (-0.030 to +0.015) | -0.080 (-0.123 to -0.035) | -0.025 (-0.057 to +0.005) |
| MLP (256 hidden) on one-hot + pixel stats | +0.010 (-0.018 to +0.038) | -0.075 (-0.118 to -0.030) | -0.007 (-0.045 to +0.032) |

## The best setting per taste

Chosen on each taste's selection set, from every Stage A passer.

| Taste | kind | code | steps | trials | noise | g_syn | bias | g_in | selection |
|---|---|---|---|---|---|---|---|---|---|
| additive | rate | all-sensory+vnc | 60 | 1 | 0.0 | 0.5 | 0.1 | 2.0 | 0.753 |
| latent | rate | all-sensory+vnc | 60 | 1 | 0.0 | 2.0 | 0.1 | 0.5 | 0.645 |
| visual | rate | all-sensory+vnc | 60 | 1 | 0.0 | 0.5 | 0.1 | 0.5 | 0.678 |

## Stage B: the best selection score per brain and code

130 settings probed on every taste's selection set.

| Brain | Code | additive | latent | visual |
|---|---|---|---|---|
| lif | all-sensory-brain | 0.533 | 0.532 | 0.510 |
| lif | eyes+nose | 0.457 | 0.513 | 0.542 |
| lif | nose | 0.467 | 0.490 | 0.542 |
| lif-volley | all-sensory+vnc | 0.605 | 0.578 | 0.560 |
| lif-volley | all-sensory-brain | 0.597 | 0.567 | 0.583 |
| lif-volley | eyes | 0.547 | 0.575 | 0.590 |
| lif-volley | eyes+nose | 0.582 | 0.572 | 0.580 |
| lif-volley | nose | 0.557 | 0.557 | 0.550 |
| rate | all-sensory+vnc | 0.753 | 0.645 | 0.678 |
| rate | all-sensory-brain | 0.752 | 0.622 | 0.672 |
| rate | eyes | 0.588 | 0.577 | 0.650 |
| rate | eyes+nose | 0.662 | 0.593 | 0.648 |
| rate | nose | 0.647 | 0.573 | 0.582 |

## Calibration (selection set)

| Taste | Bayes | Additive oracle | Gap | One-hot | + pixels | MLP |
|---|---|---|---|---|---|---|
| additive | 0.832 | 0.832 | +0.000 | 0.768 | 0.768 | 0.772 |
| latent | 0.810 | 0.682 | +0.129 | 0.662 | 0.663 | 0.665 |
| visual | 0.819 | 0.745 | +0.074 | 0.688 | 0.690 | 0.677 |

## Reproduction

Additive-taste score minus Phase 0's, over 130 settings: mean -0.0001, sd 0.0021, largest |difference| 0.0150; 3 chose a different λ than in Phase 0. The labels are Phase 0's exactly; single scores move by up to ~0.01 between identical runs because GPU rounding can flip the readout's CV-chosen λ (spec §3.0b amendment 1). Stage C requires |mean| ≤ 0.005.
Rewired twin: 22153 repair swaps.

## Attribution

Connectome: Janelia FlyEM MaleCNS v1.0 (CC BY 4.0), Berg et al., Cell 189(18):5504–5526.e15 (2026), doi:10.1016/j.cell.2026.08.015. Changes: thresholded at ≥3 synapses, signed by predicted transmitter, photoreceptor columns derived from synaptic partners, simulated. No endorsement by HHMI/Janelia, Cambridge, MRC-LMB or Google is implied.
