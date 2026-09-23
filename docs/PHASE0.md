# Phase 0: can the fly learn a taste at all?

**Verdict: PASS** (gate: held-out ≥ 0.60 on the confirmation set, a planted additive taste that no selection saw, by the setting that passes every constraint and scores best on the selection set).

| Reference (confirmation set) | Held-out |
|---|---|
| Bayes ceiling | 0.819 |
| One-hot, no brain | 0.770 |
| **Best fly setting (confirmation)** | **0.758** (0.727–0.788) |
| rewired twin of the best | 0.777 |
| sign-shuffled twin of the best | 0.733 |

Best setting's selection score: 0.760 (0.727–0.793), selection (max over 130 probed; optimistic, not the gate).

## The best setting

| kind | code | steps | trials | noise | g_syn | bias | g_in |
|---|---|---|---|---|---|---|---|
| rate | all-sensory-brain | 60 | 1 | 0.0 | 1.0 | 0.1 | 2.0 |

## Stage A: screening

130 of 420 settings passed every check.

| Brain | Code | Screened | Passed |
|---|---|---|---|
| lif | all-sensory+vnc | 18 | 0 |
| lif | all-sensory-brain | 18 | 2 |
| lif | eyes | 18 | 0 |
| lif | eyes+nose | 18 | 1 |
| lif | nose | 18 | 1 |
| lif-avg | all-sensory+vnc | 18 | 0 |
| lif-avg | all-sensory-brain | 18 | 0 |
| lif-avg | eyes | 18 | 0 |
| lif-avg | eyes+nose | 18 | 0 |
| lif-avg | nose | 18 | 0 |
| lif-volley | all-sensory+vnc | 36 | 24 |
| lif-volley | all-sensory-brain | 36 | 24 |
| lif-volley | eyes | 36 | 10 |
| lif-volley | eyes+nose | 36 | 13 |
| lif-volley | nose | 36 | 20 |
| rate | all-sensory+vnc | 12 | 8 |
| rate | all-sensory-brain | 12 | 8 |
| rate | eyes | 12 | 5 |
| rate | eyes+nose | 12 | 6 |
| rate | nose | 12 | 8 |

## Stage B: planted-taste probe

Every setting scored on the selection probe set, the set the best is picked on.

| Brain | Code | steps | trials | g_syn | bias | g_in | eligible | held-out | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| rate | all-sensory-brain | 60 | 1 | 1.0 | 0.1 | 2.0 | yes | 0.760 | 0.727–0.793 |
| rate | all-sensory+vnc | 60 | 1 | 0.5 | 0.1 | 2.0 | yes | 0.753 | 0.720–0.787 |
| rate | all-sensory+vnc | 60 | 1 | 1.0 | 0.1 | 2.0 | yes | 0.753 | 0.718–0.787 |
| rate | all-sensory-brain | 60 | 1 | 2.0 | 0.1 | 2.0 | yes | 0.752 | 0.718–0.785 |
| rate | all-sensory+vnc | 60 | 1 | 0.5 | 0.1 | 0.5 | yes | 0.750 | 0.717–0.783 |
| rate | all-sensory-brain | 60 | 1 | 0.5 | 0.1 | 2.0 | yes | 0.750 | 0.717–0.783 |
| rate | all-sensory-brain | 60 | 1 | 1.0 | 0.1 | 0.5 | yes | 0.747 | 0.712–0.782 |
| rate | all-sensory+vnc | 60 | 1 | 1.0 | 0.1 | 0.5 | yes | 0.747 | 0.715–0.778 |
| rate | all-sensory-brain | 60 | 1 | 0.5 | 0.1 | 0.5 | yes | 0.740 | 0.707–0.773 |
| rate | all-sensory-brain | 60 | 1 | 2.0 | 0.1 | 0.5 | yes | 0.722 | 0.685–0.760 |
| rate | all-sensory+vnc | 60 | 1 | 2.0 | 0.1 | 2.0 | yes | 0.703 | 0.665–0.742 |
| rate | all-sensory+vnc | 60 | 1 | 2.0 | 0.1 | 0.5 | yes | 0.692 | 0.653–0.730 |
| rate | eyes+nose | 60 | 1 | 1.0 | 0.1 | 2.0 | yes | 0.663 | 0.623–0.702 |
| rate | eyes+nose | 60 | 1 | 1.0 | 0.1 | 0.5 | yes | 0.658 | 0.620–0.697 |
| rate | eyes+nose | 60 | 1 | 2.0 | 0.1 | 0.5 | yes | 0.657 | 0.617–0.695 |
| rate | eyes+nose | 60 | 1 | 2.0 | 0.1 | 2.0 | yes | 0.653 | 0.617–0.690 |
| rate | all-sensory-brain | 60 | 1 | 2.0 | 0.0 | 2.0 | yes | 0.653 | 0.612–0.692 |
| rate | nose | 60 | 1 | 1.0 | 0.1 | 2.0 | yes | 0.648 | 0.608–0.687 |
| rate | nose | 60 | 1 | 1.0 | 0.1 | 0.5 | yes | 0.647 | 0.607–0.687 |
| rate | all-sensory+vnc | 60 | 1 | 2.0 | 0.0 | 2.0 | yes | 0.647 | 0.608–0.685 |
| rate | nose | 60 | 1 | 2.0 | 0.1 | 0.5 | yes | 0.632 | 0.590–0.672 |
| rate | nose | 60 | 1 | 0.5 | 0.1 | 2.0 | yes | 0.630 | 0.590–0.670 |
| rate | nose | 60 | 1 | 2.0 | 0.1 | 2.0 | yes | 0.622 | 0.582–0.660 |
| rate | eyes+nose | 60 | 1 | 0.5 | 0.1 | 2.0 | yes | 0.620 | 0.582–0.658 |
| rate | all-sensory-brain | 60 | 1 | 2.0 | 0.0 | 0.5 | yes | 0.620 | 0.582–0.658 |
| rate | eyes+nose | 60 | 1 | 0.5 | 0.1 | 0.5 | yes | 0.617 | 0.578–0.653 |
| rate | all-sensory+vnc | 60 | 1 | 2.0 | 0.0 | 0.5 | yes | 0.608 | 0.570–0.647 |
| rate | nose | 60 | 1 | 2.0 | 0.0 | 2.0 | yes | 0.605 | 0.565–0.645 |
| lif-volley | all-sensory+vnc | 10 | 1 | 6.0 | 0.1 | 0.5 | yes | 0.605 | 0.565–0.647 |
| rate | nose | 60 | 1 | 0.5 | 0.1 | 0.5 | yes | 0.603 | 0.565–0.642 |
| lif-volley | all-sensory+vnc | 10 | 1 | 4.0 | 0.1 | 2.0 | yes | 0.598 | 0.560–0.637 |
| lif-volley | all-sensory-brain | 10 | 1 | 6.0 | 0.1 | 2.0 | yes | 0.597 | 0.555–0.637 |
| lif-volley | all-sensory+vnc | 10 | 1 | 8.0 | 0.05 | 0.5 | yes | 0.597 | 0.555–0.638 |
| lif-volley | all-sensory+vnc | 6 | 1 | 6.0 | 0.05 | 2.0 | yes | 0.595 | 0.557–0.635 |
| rate | eyes | 60 | 1 | 2.0 | 0.1 | 0.5 | yes | 0.593 | 0.553–0.635 |
| lif-volley | all-sensory-brain | 10 | 1 | 4.0 | 0.05 | 2.0 | yes | 0.590 | 0.548–0.630 |
| lif-volley | all-sensory+vnc | 10 | 1 | 8.0 | 0.1 | 2.0 | yes | 0.587 | 0.547–0.627 |
| lif-volley | all-sensory+vnc | 6 | 1 | 4.0 | 0.05 | 2.0 | yes | 0.585 | 0.543–0.625 |
| lif-volley | all-sensory-brain | 10 | 1 | 8.0 | 0.05 | 2.0 | yes | 0.585 | 0.542–0.628 |
| lif-volley | all-sensory+vnc | 10 | 1 | 8.0 | 0.05 | 2.0 | yes | 0.585 | 0.545–0.623 |
| lif-volley | all-sensory+vnc | 10 | 1 | 4.0 | 0.05 | 2.0 | yes | 0.583 | 0.545–0.623 |
| lif-volley | all-sensory-brain | 10 | 1 | 4.0 | 0.1 | 2.0 | yes | 0.583 | 0.542–0.623 |
| lif-volley | eyes+nose | 6 | 1 | 4.0 | 0.1 | 2.0 | yes | 0.582 | 0.542–0.618 |
| lif-volley | all-sensory+vnc | 10 | 1 | 8.0 | 0.1 | 0.5 | yes | 0.582 | 0.542–0.622 |
| lif-volley | all-sensory-brain | 10 | 1 | 4.0 | 0.05 | 0.5 | yes | 0.580 | 0.538–0.620 |
| lif-volley | all-sensory-brain | 10 | 1 | 8.0 | 0.1 | 2.0 | yes | 0.580 | 0.538–0.622 |
| rate | nose | 60 | 1 | 2.0 | 0.0 | 0.5 | yes | 0.578 | 0.537–0.620 |
| lif-volley | all-sensory+vnc | 6 | 1 | 8.0 | 0.1 | 0.5 | yes | 0.578 | 0.535–0.622 |
| lif-volley | all-sensory+vnc | 6 | 1 | 8.0 | 0.05 | 2.0 | yes | 0.577 | 0.537–0.620 |
| lif-volley | all-sensory-brain | 6 | 1 | 8.0 | 0.1 | 2.0 | yes | 0.577 | 0.535–0.618 |
| lif-volley | all-sensory+vnc | 10 | 1 | 6.0 | 0.1 | 2.0 | yes | 0.577 | 0.537–0.615 |
| rate | eyes | 60 | 1 | 2.0 | 0.1 | 2.0 | yes | 0.570 | 0.532–0.610 |
| lif-volley | all-sensory-brain | 10 | 1 | 6.0 | 0.05 | 2.0 | yes | 0.570 | 0.532–0.612 |
| rate | eyes | 60 | 1 | 1.0 | 0.1 | 0.5 | yes | 0.567 | 0.527–0.607 |
| lif-volley | all-sensory+vnc | 10 | 1 | 4.0 | 0.05 | 0.5 | yes | 0.567 | 0.525–0.607 |
| lif-volley | all-sensory+vnc | 6 | 1 | 4.0 | 0.1 | 2.0 | yes | 0.565 | 0.523–0.605 |
| lif-volley | all-sensory-brain | 10 | 1 | 4.0 | 0.1 | 0.5 | yes | 0.565 | 0.525–0.603 |
| lif-volley | all-sensory+vnc | 10 | 1 | 6.0 | 0.05 | 2.0 | yes | 0.565 | 0.523–0.603 |
| lif-volley | all-sensory-brain | 6 | 1 | 4.0 | 0.05 | 2.0 | yes | 0.563 | 0.525–0.603 |
| rate | eyes | 60 | 1 | 1.0 | 0.1 | 2.0 | yes | 0.563 | 0.522–0.605 |
| lif-volley | all-sensory+vnc | 6 | 1 | 8.0 | 0.05 | 0.5 | yes | 0.563 | 0.520–0.607 |
| lif-volley | all-sensory-brain | 6 | 1 | 6.0 | 0.1 | 2.0 | yes | 0.562 | 0.518–0.607 |
| lif-volley | all-sensory+vnc | 6 | 1 | 6.0 | 0.1 | 2.0 | yes | 0.560 | 0.518–0.603 |
| lif-volley | all-sensory-brain | 10 | 1 | 8.0 | 0.05 | 0.5 | yes | 0.558 | 0.518–0.598 |
| lif-volley | all-sensory+vnc | 6 | 1 | 4.0 | 0.1 | 0.5 | yes | 0.557 | 0.517–0.598 |
| lif-volley | nose | 6 | 1 | 6.0 | 0.05 | 0.5 | yes | 0.557 | 0.513–0.598 |
| lif-volley | all-sensory-brain | 6 | 1 | 8.0 | 0.05 | 2.0 | yes | 0.557 | 0.515–0.597 |
| lif-volley | all-sensory-brain | 6 | 1 | 6.0 | 0.05 | 2.0 | yes | 0.555 | 0.513–0.597 |
| lif-volley | all-sensory+vnc | 10 | 1 | 4.0 | 0.1 | 0.5 | yes | 0.555 | 0.512–0.597 |
| lif-volley | all-sensory-brain | 6 | 1 | 8.0 | 0.1 | 0.5 | yes | 0.552 | 0.510–0.593 |
| lif-volley | all-sensory-brain | 6 | 1 | 4.0 | 0.05 | 0.5 | yes | 0.548 | 0.507–0.588 |
| lif-volley | eyes | 6 | 1 | 4.0 | 0.1 | 0.5 | yes | 0.547 | 0.505–0.588 |
| lif-volley | eyes | 6 | 1 | 8.0 | 0.1 | 0.5 | yes | 0.547 | 0.503–0.588 |
| lif-volley | all-sensory-brain | 6 | 1 | 4.0 | 0.1 | 2.0 | yes | 0.547 | 0.507–0.585 |
| lif-volley | all-sensory+vnc | 6 | 1 | 6.0 | 0.05 | 0.5 | yes | 0.543 | 0.502–0.585 |
| lif-volley | all-sensory+vnc | 10 | 1 | 6.0 | 0.05 | 0.5 | yes | 0.543 | 0.500–0.587 |
| lif-volley | nose | 6 | 1 | 4.0 | 0.1 | 0.5 | yes | 0.542 | 0.502–0.580 |
| lif-volley | all-sensory-brain | 6 | 1 | 6.0 | 0.05 | 0.5 | yes | 0.542 | 0.502–0.580 |
| lif-volley | nose | 6 | 1 | 6.0 | 0.1 | 0.5 | yes | 0.540 | 0.500–0.582 |
| lif-volley | nose | 6 | 1 | 8.0 | 0.1 | 2.0 | yes | 0.540 | 0.498–0.580 |
| lif-volley | all-sensory-brain | 10 | 1 | 8.0 | 0.1 | 0.5 | yes | 0.540 | 0.498–0.583 |
| lif-volley | eyes+nose | 6 | 1 | 4.0 | 0.1 | 0.5 | yes | 0.538 | 0.498–0.578 |
| lif-volley | nose | 6 | 1 | 4.0 | 0.05 | 2.0 | yes | 0.538 | 0.497–0.578 |
| lif-volley | eyes | 6 | 1 | 4.0 | 0.1 | 2.0 | yes | 0.538 | 0.498–0.580 |
| lif-volley | nose | 6 | 1 | 6.0 | 0.05 | 2.0 | yes | 0.538 | 0.497–0.578 |
| lif-volley | eyes+nose | 10 | 1 | 4.0 | 0.1 | 2.0 | yes | 0.538 | 0.497–0.580 |
| lif-volley | all-sensory-brain | 6 | 1 | 4.0 | 0.1 | 0.5 | yes | 0.537 | 0.495–0.577 |
| lif-volley | nose | 10 | 1 | 4.0 | 0.1 | 0.5 | yes | 0.537 | 0.493–0.577 |
| lif-volley | nose | 10 | 1 | 4.0 | 0.1 | 2.0 | yes | 0.535 | 0.495–0.575 |
| lif-volley | nose | 10 | 1 | 6.0 | 0.05 | 0.5 | yes | 0.535 | 0.492–0.580 |
| lif-volley | nose | 10 | 1 | 8.0 | 0.1 | 2.0 | yes | 0.535 | 0.492–0.578 |
| lif | all-sensory-brain | 60 | 1 | 8.0 | 0.1 | 0.5 | yes | 0.533 | 0.492–0.575 |
| lif-volley | eyes | 10 | 1 | 4.0 | 0.1 | 2.0 | yes | 0.533 | 0.493–0.575 |
| lif-volley | all-sensory-brain | 6 | 1 | 6.0 | 0.1 | 0.5 | yes | 0.533 | 0.492–0.575 |
| lif-volley | all-sensory-brain | 10 | 1 | 6.0 | 0.05 | 0.5 | yes | 0.533 | 0.492–0.575 |
| lif-volley | nose | 10 | 1 | 8.0 | 0.1 | 0.5 | yes | 0.533 | 0.490–0.575 |
| rate | eyes | 60 | 1 | 0.5 | 0.1 | 2.0 | yes | 0.532 | 0.492–0.573 |
| lif-volley | nose | 10 | 1 | 6.0 | 0.1 | 2.0 | yes | 0.532 | 0.493–0.570 |
| lif-volley | eyes | 10 | 1 | 4.0 | 0.05 | 0.5 | yes | 0.530 | 0.490–0.570 |
| lif-volley | eyes | 10 | 1 | 6.0 | 0.1 | 2.0 | yes | 0.530 | 0.487–0.570 |
| lif-volley | all-sensory+vnc | 6 | 1 | 8.0 | 0.1 | 2.0 | yes | 0.530 | 0.488–0.570 |
| lif-volley | eyes+nose | 6 | 1 | 6.0 | 0.1 | 0.5 | yes | 0.527 | 0.485–0.570 |
| lif-volley | nose | 10 | 1 | 8.0 | 0.05 | 0.5 | yes | 0.527 | 0.487–0.568 |
| lif-volley | nose | 10 | 1 | 4.0 | 0.05 | 0.5 | yes | 0.525 | 0.485–0.565 |
| lif-volley | eyes+nose | 10 | 1 | 8.0 | 0.1 | 0.5 | yes | 0.525 | 0.485–0.567 |
| lif-volley | eyes+nose | 10 | 1 | 8.0 | 0.1 | 2.0 | yes | 0.525 | 0.485–0.568 |
| lif-volley | nose | 10 | 1 | 6.0 | 0.05 | 2.0 | yes | 0.525 | 0.483–0.563 |
| lif-volley | eyes | 10 | 1 | 8.0 | 0.1 | 0.5 | yes | 0.523 | 0.482–0.567 |
| lif-volley | nose | 6 | 1 | 4.0 | 0.05 | 0.5 | yes | 0.522 | 0.482–0.563 |
| lif-volley | eyes+nose | 10 | 1 | 6.0 | 0.05 | 0.5 | yes | 0.522 | 0.478–0.565 |
| lif-volley | all-sensory-brain | 6 | 1 | 8.0 | 0.05 | 0.5 | yes | 0.522 | 0.482–0.562 |
| lif-volley | nose | 10 | 1 | 4.0 | 0.05 | 2.0 | yes | 0.522 | 0.478–0.565 |
| lif-volley | nose | 6 | 1 | 6.0 | 0.1 | 2.0 | yes | 0.518 | 0.477–0.560 |
| lif | all-sensory-brain | 60 | 1 | 8.0 | 0.05 | 0.5 | yes | 0.517 | 0.477–0.557 |
| lif-volley | all-sensory+vnc | 6 | 1 | 4.0 | 0.05 | 0.5 | yes | 0.517 | 0.475–0.558 |
| lif-volley | eyes+nose | 10 | 1 | 6.0 | 0.1 | 0.5 | yes | 0.515 | 0.475–0.552 |
| lif-volley | nose | 10 | 1 | 6.0 | 0.1 | 0.5 | yes | 0.513 | 0.472–0.553 |
| lif-volley | eyes+nose | 10 | 1 | 8.0 | 0.05 | 0.5 | yes | 0.510 | 0.472–0.553 |
| lif-volley | eyes | 6 | 1 | 8.0 | 0.1 | 2.0 | yes | 0.507 | 0.463–0.550 |
| lif-volley | all-sensory+vnc | 6 | 1 | 6.0 | 0.1 | 0.5 | yes | 0.507 | 0.467–0.545 |
| lif-volley | eyes+nose | 10 | 1 | 8.0 | 0.05 | 2.0 | yes | 0.507 | 0.465–0.550 |
| lif-volley | eyes | 10 | 1 | 8.0 | 0.05 | 2.0 | yes | 0.503 | 0.458–0.547 |
| lif-volley | eyes | 10 | 1 | 8.0 | 0.05 | 0.5 | yes | 0.502 | 0.460–0.545 |
| lif-volley | nose | 6 | 1 | 4.0 | 0.1 | 2.0 | yes | 0.500 | 0.460–0.540 |
| lif-volley | all-sensory-brain | 10 | 1 | 6.0 | 0.1 | 0.5 | yes | 0.500 | 0.453–0.545 |
| lif-volley | eyes+nose | 10 | 1 | 4.0 | 0.1 | 0.5 | yes | 0.498 | 0.458–0.538 |
| lif-volley | eyes+nose | 10 | 1 | 6.0 | 0.1 | 2.0 | yes | 0.495 | 0.457–0.533 |
| lif-volley | eyes+nose | 10 | 1 | 4.0 | 0.05 | 0.5 | yes | 0.493 | 0.452–0.535 |
| lif | nose | 60 | 1 | 8.0 | 0.1 | 0.5 | yes | 0.467 | 0.422–0.508 |
| lif | eyes+nose | 60 | 1 | 8.0 | 0.1 | 0.5 | yes | 0.457 | 0.417–0.495 |

## Per-slot decodability by input code

For each input code, the Stage B setting with the highest held-out taste accuracy (eligible or not). A linear readout is trained to name each slot's value on the training looks and scored on the held-out looks; chance is the most common value's share of the training looks.

| Code | Brain | g_syn | bias | g_in | eligible | taste held-out |
|---|---|---|---|---|---|---|
| nose | rate | 1.0 | 0.1 | 2.0 | yes | 0.648 |
| eyes | rate | 2.0 | 0.1 | 0.5 | yes | 0.593 |
| eyes+nose | rate | 1.0 | 0.1 | 2.0 | yes | 0.663 |
| all-sensory-brain | rate | 1.0 | 0.1 | 2.0 | yes | 0.760 |
| all-sensory+vnc | rate | 0.5 | 0.1 | 2.0 | yes | 0.753 |

| Slot | Chance | nose | eyes | eyes+nose | all-sensory-brain | all-sensory+vnc |
|---|---|---|---|---|---|---|
| Background | 5.3% | 48.1% | 81.7% | 74.4% | 74.8% | 87.6% |
| Back | 59.8% | 92.5% | 84.6% | 96.0% | 100.0% | 99.9% |
| Body | 24.6% | 79.4% | 51.4% | 83.0% | 93.9% | 95.7% |
| Clothing | 3.2% | 30.0% | 85.7% | 67.0% | 64.9% | 75.2% |
| Mouth | 5.0% | 39.7% | 11.2% | 42.5% | 67.6% | 85.5% |
| Eyebrows | 7.0% | 61.3% | 4.1% | 58.6% | 80.4% | 93.9% |
| Eyes | 5.6% | 52.0% | 21.6% | 52.9% | 80.1% | 87.6% |
| Head | 2.1% | 23.0% | 29.7% | 32.9% | 52.5% | 65.9% |
| Accessory | 29.6% | 56.7% | 45.9% | 65.1% | 90.6% | 87.0% |

## Stage C: per-slot decodability of the best setting

On the selection probe set's held-out looks, like Stage B's. The confirmation scores are in the table at the top.

| Slot | Held-out | Chance |
|---|---|---|
| Background | 74.6% | 5.3% |
| Back | 100.0% | 59.8% |
| Body | 93.9% | 24.6% |
| Clothing | 64.9% | 3.2% |
| Mouth | 67.6% | 5.0% |
| Eyebrows | 80.4% | 7.0% |
| Eyes | 80.1% | 5.6% |
| Head | 52.4% | 2.1% |
| Accessory | 90.6% | 29.6% |

Rewire repair swaps: 22,153. Confirmation set: seed 1000.

## How to read the checks

1. **Smoothness can pass trivially.** Deterministic spiking brains often pass the smoothness check without being smooth: a 0.01 concentration change flips no spike, so the readout moves not at all (distance 0), and 0 is at most 10% of any one-slot swap's distance.
2. **Each code's minimal perturbation.** The `eyes` code's minimal perturbation is image brightness × 1.01, not a concentration change. `eyes+nose` perturbs only the nose, and both senses share one `g_in`, so the grid never drives the eyes at another gain than the nose.
3. **The sense check is weak.** A sense check passes on any nonzero readout change between two sets of looks, however small.
4. **`lif-avg` is not reproducible across batch layouts.** Its trial noise stream is per batch, so a look's features depend on its batch position: the same look in another batch layout gets different features.
5. **The gate uses the confirmation set.** Stage B scores every passer on the same selection probe set, and the best setting's selection score is the maximum of those scores, so it is optimistic (the winner's curse). The gate is that one setting's score on the confirmation set: an independent planted taste and independent pairs, drawn from seed 1000 (the run's seed 0 + 1000), which no selection ever saw. The twins and the one-hot control are scored on it too.

---

Connectome: Janelia FlyEM MaleCNS v1.0 (CC BY 4.0), Berg et al., Cell 189(18):5504–5526.e15 (2026), doi:10.1016/j.cell.2026.08.015. Changes: thresholded at ≥3 synapses, signed by predicted transmitter, photoreceptor columns derived from synaptic partners, simulated. No endorsement by HHMI/Janelia, Cambridge, MRC-LMB or Google is implied.
