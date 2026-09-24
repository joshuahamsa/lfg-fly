# lfg-fly

A fruit-fly connectome (Janelia FlyEM **MaleCNS v1.0**) that learns to dress its own
[LFG](https://build.letseffinggo.com) NFT, as an ordinary outside user of the platform.
Inspired by [Fly vs Chess](https://opsmavix.com/blog/fly-vs-chess/). The wiring is never
changed; only linear readouts of the fly's descending and motor neurons are trained.

Design: [`docs/specs/2026-09-22-fly-design.md`](docs/specs/2026-09-22-fly-design.md).

## Status: Phase 0 passed, and the wiring didn't matter

Phase 0 asks the easiest question first: can the fly learn a **planted additive taste**
over LFG traits at all? The planted taste gives every trait value a hidden score; a look's
score is the sum. This question comes before any critic tokens, wallet or LFG work. Full
tables are in [`docs/PHASE0.md`](docs/PHASE0.md), raw results in [`data/probe/`](data/probe/).

- **420 settings screened** (4 brain types × 5 input codes × gains) on the full connectome:
  165,122 traced neurons and 10,511,038 synapse-weighted edges, simulated on one RTX 2060 SUPER.
  130 settings passed the activity and smoothness checks. Every one of those 130 was then
  probed on 3,000 labelled pairs.
- **The best setting passes the ≥ 0.60 gate:** a rate-model brain, with each trait injected
  into a fixed random group of brain sensory neurons. It scores **0.758** held-out
  (95% CI 0.727–0.788) on an *independent* confirmation taste that played no part in choosing it.
- **The real wiring did not help.** On that same confirmation set:

  | Model | Held-out |
  |---|---|
  | Bayes ceiling | 0.819 |
  | Rewired twin (same neurons and degrees, wiring shuffled) | 0.777 |
  | No brain (logistic regression on the trait list) | 0.770 |
  | **The fly (real MaleCNS wiring)** | **0.758** |
  | Sign-shuffled twin | 0.733 |

  The confidence intervals overlap. A degree-preserving random rewiring does at least as well
  as the real connectome, and so does skipping the brain entirely. This matches every
  published control in the fly-game space so far. On this task the connectome works as a
  rich random-feature machine, and its particular wiring earns nothing.
- **The full-window spiking brain is chaotic and stays at chance** (≤ 0.53). The smooth rate
  model is what learns.
- **The senses carry trait identity, each its own way.** From the readout of the eyes-only
  code, Background (82%) and Clothing (86%) are decodable, but not Eyebrows (4%) or Mouth
  (11%): a ~28×29 compound eye sees big regions, not fine features. Odour codes carry the
  fine slots.

**These numbers have about ±0.006 of run-to-run noise.** Phase 0 was run before the
simulator was deterministic on the GPU. torch's sparse matmul (cuSPARSE) sums in a different
order on every run, so simulating the best setting twice gave readout features that differed
by up to ~4e-7, and the readout fit turned that into held-out scores between 0.750 and 0.762.
The stored 0.758 (confirmation) and 0.760 (selection) are single draws from that spread, and so
are the rewired and sign-shuffled twins' scores. The no-brain and Bayes rows involve no
simulation. The committed results have not been rerun. The simulator now uses its own CSR
kernel ([`lfg_fly/brain/spmm.py`](lfg_fly/brain/spmm.py)), so repeated runs on one GPU give
bit-identical features.

## Phase 0b: the fly doesn't learn trait interactions, and nothing else does either

Phase 0's taste was *additive*, and for an additive task a model on the plain trait list is
already the right tool. The question Phase 0 left open is whether the connectome captures trait
**interactions** better than a plain model. That is where "an unexpected combination that works"
lives. Phase 0b asks it before any critic tokens are spent. It plants two tastes that are half
interaction and half additive, then re-probes every one of Phase 0's 130 passing settings.

- **latent:** hidden trait "harmony" vectors, where pairs of traits score by how well they fit.
- **visual:** how well the character stands out from the background, plus hue harmony.

The tests and the four readings were pre-registered before any run (design spec §3.0b), and the
simulator is now deterministic, so every number reproduces exactly. Full tables are in
[`docs/PHASE0B.md`](docs/PHASE0B.md), raw results in [`data/probe-interact/`](data/probe-interact/).

| Held-out, confirmation set | additive | latent | visual |
|---|---|---|---|
| Bayes ceiling | 0.819 | 0.835 | 0.824 |
| Best any additive model can do | 0.819 | 0.694 | 0.745 |
| **The fly** | **0.762** | **0.582** | **0.680** |
| Rewired twin | 0.765 | 0.663 | 0.678 |
| No brain (logistic regression on the trait list) | 0.770 | 0.670 | 0.708 |
| MLP (256 hidden) on traits + pixel stats | 0.752 | 0.657 | 0.687 |

- **No model learns the interactions from about 3,000 pairs.** On the latent taste, 0.14 of
  accuracy is available *only* through interactions (0.835 − 0.694). Every model lands below
  the best purely additive score, the MLP included.
- **The fly does no better, and on the latent taste it does significantly worse.** It trails
  every control there, including its own rewired twin (−0.082, 95% CI −0.122 to −0.040). Its
  best setting scored 0.645 on the selection set and fell to 0.582 on a fresh instance.
- **On the visual taste the fly learns (0.680), but only as well as a plain model.** Its eyes
  don't help: on the selection set, the best eyes-only setting (0.650) trails the
  all-sensory ones (0.678).
- **Pre-registered readings:**
  - *Learns it* (CI lower bound above 0.55): holds for additive (0.730) and visual (0.642); fails for latent (0.540).
  - *Uses interactions*, *wiring matters*, *beats no-brain*: hold for **no** taste.

**What this means for the critic phase.** The critic's taste will be learned from a similar
number of pairs, and at that scale "an unexpected combination that works" is not learnable here,
by the fly or by anything else we tried. The fly can learn the additive and broadly visual parts
of a taste, as well as a plain model can. If the critic's taste is mostly that, it would likely
clear Gate B's bar. If it is interaction-heavy, it likely won't.

## Reproduce

```bash
scripts/setup_venv.sh                       # uv venv, torch 2.5.1+cu121
nice -n 10 ionice -c3 .venv/bin/fly fetch   # three public feathers, ~566 MB, sha256-checked
nice -n 10 ionice -c3 .venv/bin/fly build   # integer-CSR graph on the GPU
nice -n 10 ionice -c3 .venv/bin/fly columns # photoreceptor columns from synaptic partners
nice -n 10 ionice -c3 .venv/bin/fly catalog --api https://<lfg-api-base>
nice -n 10 ionice -c3 .venv/bin/fly grid --device cuda   # Stages A, B, C and verdict (resumable)
nice -n 10 ionice -c3 .venv/bin/fly interact --device cuda   # Phase 0b: calibration, B, C (resumable)
.venv/bin/fly report                        # docs/PHASE0.md and docs/PHASE0B.md
```

Data lives in `FLY_DATA_DIR` (default `~/fly-data`), never in this repo. Everything runs
niced, capped at one CPU thread, and on the GPU.

## Attribution and licence

Connectome: Janelia FlyEM MaleCNS v1.0 (CC BY 4.0). Berg et al., *Cell* 189(18):5504–5526.e15
(2026), doi:10.1016/j.cell.2026.08.015. Changes: thresholded at ≥ 3 synapses, signed by
predicted transmitter, photoreceptor columns derived from synaptic partners, simulated. No
endorsement by HHMI/Janelia, Cambridge, MRC-LMB or Google is implied. The connectome is
downloaded at build time, never redistributed.

Code: MIT. Nothing is copied from the unlicensed chess and tic-tac-toe fly repos, or from
nfly. Fly vs Chess (@ErnestoSOFTWARE) is credited as the inspiration.
