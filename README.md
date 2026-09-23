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

**What Phase 0 does not answer:** the taste was *additive*, and for an additive task a model
on the plain trait list is already the right tool. Whether the connectome captures trait
**interactions** better than a plain model is the open question. That is where "an unexpected
combination that works" lives. The critic phase tests it with the same controls, and the
report will state that answer as plainly as this one.

## Reproduce

```bash
scripts/setup_venv.sh                       # uv venv, torch 2.5.1+cu121
nice -n 10 ionice -c3 .venv/bin/fly fetch   # three public feathers, ~566 MB, sha256-checked
nice -n 10 ionice -c3 .venv/bin/fly build   # integer-CSR graph on the GPU
nice -n 10 ionice -c3 .venv/bin/fly columns # photoreceptor columns from synaptic partners
nice -n 10 ionice -c3 .venv/bin/fly catalog --api https://<lfg-api-base>
nice -n 10 ionice -c3 .venv/bin/fly grid --device cuda   # Stages A, B, C and verdict (resumable)
.venv/bin/fly report                        # docs/PHASE0.md
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
