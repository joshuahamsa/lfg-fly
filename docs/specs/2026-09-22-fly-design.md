# The fly: a fruit-fly connectome that dresses its own LFG NFT

Status: design approved in brainstorming 2026-09-22, then revised after an
adversarial review. That review measured the first-draft brain on the real
connectome, and it **failed the easiest learnability test** (§3.0). The
project therefore starts with a **Phase 0 feasibility probe**: GPU only, no
critic tokens, no LFG work. Everything after Phase 0 is conditional on its
verdict. LFG-side companion spec: `Team-Hamsa/LFG`
`docs/superpowers/specs/2026-09-22-agent-users-design.md`.

## What this is

[Fly vs Chess](https://opsmavix.com/blog/fly-vs-chess/) showed a simulated
fruit-fly nervous system choosing chess moves. Its wiring comes from a real
connectome and is never changed; only a linear readout of its motor neurons is
trained. This project gives the same kind of fly an LFG character and a
wardrobe. Every day it changes its look, chasing **visually appealing,
surprising ideas** built from trait combinations, and it posts the result from
its own X account.

The fly is **literally another LFG user**. It has its own wallet and key, and
signs in through LFG's public API. It earns the BRIX drip, appears on every
leaderboard and in SourceTag tracking, and anyone can compete with it. Its
only visible difference is honest provenance: its sessions use LFG's `agent`
sign-in provider, so its on-chain memos read `platform=agent`.

### Decisions made in brainstorming

| Topic | Decision |
|---|---|
| Purpose | Spectacle: the fly owns its NFT and restyles it on a schedule, announced on X |
| Objective | Visual appeal + a nameable idea + **surprise** (an unexpected combination that works) |
| Teacher | An offline **Claude Code workflow** critic. No API key anywhere, and nothing at runtime calls Claude |
| On-chain mechanic (v1) | The Builder: Closet **equip**, which is free, needs no signature, and applies all changes in one `NFTokenModify` |
| Wardrobe (v1) | A **donor batch** of characters harvested into the fly's Closet; Closet Market buys and community gifts come later |
| Compute | **GPU** (RTX 2060 SUPER); the CPU stays with the XRPL validator |
| Brain | Fly-vs-Chess-style (fixed wiring, trained linear readouts only), written from scratch, with the variant chosen by Phase 0 |
| Home | This repo (`joshuahamsa/lfg-fly`, public, MIT); LFG gets only the generic agent-users PR |
| Keys | The fly holds only a **RegularKey**; the master key stays in the operator's Xaman |
| Hero | A **male** character (the largest catalog; male↔female share every slot except Clothing) |
| Rarity | Enters as **smell concentration**, read live; its own readout head, retrained nightly |
| Sequencing | **Phase 0 probe first**. The critic, the LFG agent PR and everything else wait for its verdict |

### Non-goals

- Claiming the connectome "understands" fashion. The fly's learned parts are
  two linear readouts. Control runs report whether the real wiring beats a
  rewired one or a no-brain baseline, whatever the answer.
- Any LFG privilege. If the public API can't do it, the fly can't either.
- Running anything on the CPU that the GPU can do.
- Posting as "the fly chose this" when its taste head has failed its gate.

## 1. Architecture

```
 lfg-fly (this repo, own venv: torch cu121 on the 2060)        LFG (public API)
 ┌────────────────────────────────────────────────┐          ┌──────────────────────────┐
 │ brain/  connectome + senses + sim + heads        │  HTTPS   │ /api/web/signin (+proof) │
 │ teacher/ probe, renderer, sampler, train, report │◄────────►│ /api/nfts  /api/economy  │
 │ body/   LFG client, signer (RegularKey),         │          │ /api/layer /api/rarity/* │
 │         legality mirror, candidates, decide      │          │ /api/equip /api/harvest  │
 │ voice/  naming, before→after card, X poster      │          │ /api/closet /api/brix/*  │
 │ .claude/workflows/critic.js (offline teacher)    │          │ /api/sign/{id}(+result)  │
 └────────────────────────────────────────────────┘          └──────────────────────────┘
```

- **The brain knows nothing about the chain.** `brain/` is a pure function:
  looks in, activity and scores out. `body/` owns every network call and every
  signature.
- **LFG's own rules are the real guard.** Everything the fly submits is
  re-validated by the server, as for any user. The fly's legality mirror
  exists only to avoid wasting moves.
- **Its own venv** (Python 3.10, matching the box). The base deps are numpy,
  pillow, pyarrow, aiohttp, and `xrpl-py` pinned to LFG's version. The `[gpu]`
  extra is `torch` from the cu121 index, which works under the box's driver 535
  (CUDA ≤ 12.2), so no driver upgrade on the validator box is needed. Turing
  has no bf16, so everything is fp32.
- **CPU discipline.**
  - The CLI entrypoint sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
    `MKL_NUM_THREADS` to 1 before importing numpy, torch or pyarrow. It then
    calls `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)` and
    `pyarrow.set_cpu_count(1)`.
  - Every command, one-off or scheduled, runs under `nice -n 10 ionice -c3`.
    pm2 has no nice option, so each pm2 entry uses `script: "/usr/bin/nice"`,
    `args: ["-n", "10", "ionice", "-c3", <venv python>, "-m", "lfg_fly.cli", …]`
    and `interpreter: "none"`.
  - Graph filtering, CSR construction, simulation and training run on CUDA.
- **Data** lives in `FLY_DATA_DIR` (default `~/fly-data`), outside every repo,
  and is **scoped per network**:
  - Mainnet and testnet each get their own `FLY_DATA_DIR/<network>/{records,
    sims, snapshots, renders, outbox}`; the connectome in `raw/` and `graph/`
    is shared.
  - Every record, snapshot and rarity-head artifact is stamped with
    `{network, lfg_api_base, wallet}`. Loaders refuse a mismatch, and the loop
    stops with an outbox alert. This prevents the 2026-09-08 fixture-leak
    failure class: a stray artifact from another network resumed by prod.
- **Committed artifacts** (small, auditable): the critic's judgments
  (`data/critic/*.jsonl`), the taste head + manifest
  (`checkpoints/fly-v<N>/`), the Phase 0 results, and `REPORT.md`.

### Repo layout

```
lfg_fly/
  connectome/  fetch.py  build.py  neurons.py  columns.py
  brain/       sim.py (torch + numpy reference)  senses.py  readout.py
               calibrate.py  checkpoint.py
  teacher/     probe.py  render.py  sample.py  dataset.py  train.py
               controls.py  report.py
  body/        client.py  signer.py  policy.py  legality.py  candidates.py
               decide.py  loop.py  claim.py  records.py  reconcile.py  setup.py
  voice/       naming.py  card.py  x_post.py
  cli.py       # fly fetch|build|probe|calibrate|render|train|report|setup|move|claim|retrain
.claude/workflows/critic.js
checkpoints/  data/critic/  data/probe/  docs/specs/  docs/plans/  REPORT.md
ecosystem.config.js   # pm2: fly-move, fly-claim, fly-retrain (nice + ionice wrapped)
.gitignore  .gitleaks.toml   # committed before any code (§4.4)
tests/
```

## 2. The brain

### Connectome

- **Source:** Janelia FlyEM **MaleCNS v1.0** (CC BY 4.0), three public
  feathers under
  `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`,
  no login needed:
  - `body-annotations-male-cns-v1.0-minconf-0.5.feather` (14.5 MB)
  - `body-neurotransmitters-male-cns-v1.0.feather` (43.3 MB)
  - `connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather`
    (508 MB, 25,563,197 edges between `status=='Traced'` bodies)
  `fly fetch` downloads them, verifies their sha256 and records it in the
  manifest.
- **Neurons:** the 165,122 `Traced` bodies.
- **Edges:** synapse count ≥ 3 (10,511,038 edges; ≥5 is a control).
- **Weights, two-step form.** First an **integer** CSR of `sign × synapse
  count`, then a per-row scale `g_syn / Σ syn(·→post)`, computed over the
  thresholded edges. Every neuron's thresholded input sum is ≤ 119,325 < 2²⁴,
  so fp32 partial sums of integers are exact in any order. That makes the GPU
  simulation **bitwise reproducible**, even though cuSPARSE's reduction order
  isn't fixed. `torch.use_deterministic_algorithms` doesn't cover sparse CSR
  matmul and is not relied on. Nothing in the weights is learned.
- **Sign map**, keyed on the pre neuron's `consensus_nt`, whose v1.0 values are
  the lower-case strings acetylcholine, gaba, glutamate, histamine, dopamine,
  serotonin, octopamine and unclear:
  - `acetylcholine` +1; `gaba`, `glutamate` and `histamine` −1;
    `dopamine`, `serotonin`, `octopamine` and `unclear` +1.
  - A traced body with **no NT row** (502 in v1.0, including one `vnc_motor`
    readout neuron) is treated as `unclear` (+1) and counted in the manifest.
  - Only an unknown non-null string fails the build.
- **The eyes need a tonic baseline.** Every photoreceptor is histaminergic,
  so every photoreceptor out-edge is inhibitory (40,036 of 40,036 at ≥3).
  Over excitatory edges alone, photoreceptors reach **0** of the 2,129 readout
  neurons. A network started from silence can never pass light downstream.
  - Every candidate brain adds a global **tonic bias** `b` to every
    non-sensory neuron.
  - Each decision window starts from a cached **resting state**, reached after
    a 200-step burn-in with a mid-grey retina and no odour.
  - In a spiking brain, `b` must keep the photoreceptors' targets tonically
    firing (`b ≥ 1 − e^(−dt/τ) ≈ 0.049` on them). Only then can a
    light-driven *decrease* show up in spikes, as it does in the real fly.
  - Without that baseline, light could act only by suppressing odour-evoked
    activity, so what the fly saw would depend on what it smelled.
- **Depth:** in the ≥3 graph the shortest paths from photoreceptors and from
  ORNs to the readout neurons have a median of 3 synapses and a max of 5.

### Candidate brains (Phase 0 chooses)

All candidates share the connectome, the weights above, `dt` = 1 ms and the
readout set, and run on the GPU in batches of candidate looks.

| Id | Dynamics | Notes |
|---|---|---|
| `lif` | LIF: `v ← v·e^(−dt/τ) + W·s + I + b`, spike at `v ≥ 1`, reset 0, refractory 2, τ = 20 ms | the first draft's brain; measured chaotic (§3.0) |
| `lif-avg` | `lif` with seeded independent input noise, spike counts averaged over R ≥ 8 trials | smooths the chaotic map; untested |
| `lif-volley` | `lif` read over a short first-volley window (T ≈ 5–12 steps) | the near-feed-forward regime, before recurrence mixes; untested |
| `rate` | `h ← (1−α)h + α·clip(ReLU(W·h + I + b), 0, 10)`, α = 0.7, from `h_rest` | measured 0.54–0.55 as drafted; untested with glomerulus codes and all-sensory input |

The decision window `T` defaults to 60 steps, except for `lif-volley`.

### Senses (fixed; never trained)

Phase 0 also chooses the **input code**:

| Id | Code |
|---|---|
| `eyes+nose` | the two senses below, together |
| `nose` / `eyes` | one sense alone (the eyes on the tonic baseline) |
| `all-sensory` | Fly-vs-Chess-style random injection. Every `(slot, value)` drives a fixed random group of ~14 neurons, seeded by the trait hash, drawn from **every traced sensory neuron** (the `ol_sensory`, `cb_sensory` and `vnc_sensory` superclasses). Fly vs Chess injects into 8,865 brain sensory neurons only. Adding the 6,365 VNC sensory neurons, which feed the VNC motor circuits directly, is this project's variant; the probe also runs the brain-only set |

**Eyes.** Each look is rendered at 64×64 RGB (the §3 renderer, then
downscaled). They use the 4,107 traced photoreceptors (R1–R6 1,394; R7 1,299;
R8 1,329; R7R8_unclear 85).
- **Columns:** no photoreceptor carries `assignedOlHex1/2` in v1.0; only the
  columnar `ol_intrinsic` cells do (L1–L3, L5, C2, C3, Mi1/4/9,
  Tm1/2/4/9/20, T1). And `somaSide` is empty for 4,078 of them, because their
  somata lie outside the imaged volume. So `fly build` assigns each
  photoreceptor the column `(somaSide, hex1, hex2)` that receives the most of
  its output synapses among hex-labelled cells, using the traced-only edge
  list at any weight. On v1.0 that assigns 3,936 of 4,107, with a median
  top-column share of 1.0.
  - The photoreceptor's eye is its column's `somaSide` (fallback `rootSide`).
  - Unassigned photoreceptors are dropped and counted, and the build fails if
    fewer than 90% are assigned.
- **Lattice:** hex (q, r) → (q + r/2, r·√3/2), normalized per eye; each eye
  sees the whole image.
- **Sampling:** each photoreceptor takes a Gaussian 3×3 average of one channel.
  - R1–R6: luminance.
  - R8y: G.
  - R8p: B.
  - R7*: B, standing in for UV.
  - R8d and unclear types: luminance.
- **Drive:** `I = g_eye · x`, on top of the tonic baseline.

**Nose: identity, with rarity as concentration.**
- **Receptors:** the traced `ORN_*` neurons, 2,635 across 53 glomerular types
  (14–204 each), all cholinergic. ORNs converge by type: 83% of ORN→PN
  synapses land on the same-glomerulus PN. So odour codes are defined at the
  **glomerulus level**; a random per-ORN code would collapse at the first
  synapse (mean trait-pair cosine 0.42 after pooling, 0.86 between looks that
  differ in all nine traits).
- **Identity:** a trait's code is **k = 4 of the 53 glomerular types**, chosen
  without replacement from a stream seeded by
  `SHA-256("lfg-fly/odor/v2|" + slot + "|" + value)`. Each gets a weight
  u ∈ U(0.5, 1), applied to every ORN of that type and scaled by
  1/√(type size). Measured: single-trait cosine 0.08; looks that differ in all
  nine traits 0.42. (Disjoint per-slot glomerulus blocks were rejected: two
  values of one slot would have cosine ≈ 0.97.) Codes are golden-hashed in
  tests, so identities never drift.
- **Concentration = rarity:** `c = 0.4 + 0.6·p`, where p is the trait's
  percentile of `n_live / freq` within its slot in the live supply snapshot
  (`GET /api/rarity/supply`). A value with zero live supply gets p = 1.
- **Drive:** `I_nose = g_nose · Σ_traits c(t) · code(t)`.

Which sense actually carries identity is **measured, not assumed**: Phase 0
and REPORT.md give per-slot value decodability from the readout on held-out
looks for each input code.

### Readout: the only trained part

- **Features:** spike counts (or rates, for `rate`) over the window of the same
  2,129 neurons Fly vs Chess reads: every Traced `descending_neuron` (1,314),
  `vnc_motor` (708) and `cb_motor` (107). They're standardized with
  training-set statistics.
- **Taste head** (`w_t`): Bradley–Terry logistic regression,
  `P(A ≻ B) = σ(w_t · (ẑ_A − ẑ_B))`, with L2 chosen by 5-fold cross-validation
  grouped by look family, and the critic's strength (1–3) as the sample weight.
  **Rarity-invariant by construction:** taste simulations pin every trait's
  concentration to a constant **c_ref = 0.7**, both in training and at
  decision time, so taste cannot depend on rarity.
- **Rarity head** (`w_r`): ridge regression from ẑ to the look's standardized
  `nft_rarity` score (`Σ n_live / freq`). It reads a **second** simulation of
  each look under live concentrations: one extra GPU batch per 64 candidates,
  about 0.6 s measured. It's retrained nightly on a fresh snapshot. Today's
  economy reaches the decision **only** through this term. REPORT.md says
  plainly that this head learns an additive, already-solved function: rarity
  is the fly's incentive, while taste is where the fly is tested.

### Decision score

```
score(look) = taste(look) + β · rarity(look) − λ · cost_BRIX(action)
```

β is the fly's disclosed "greed" (default 0.3, in units of taste's standard
deviation) and λ is its price sensitivity (default 0.05 per BRIX).
- **Costs:** a swap costs `/api/nfts` → `swap_fee.per_nft` × the NFTs it
  touches. Equip costs **0**, a constant in `body/decide.py`, because LFG
  publishes no Builder price.
- **Builder pricing (operator's v2):** when it arrives, it needs an LFG field
  and a fly change to read it; the score already has the term.

### Checkpoint and versioning

A `fly-v<N>` manifest pins everything behind a decision:
- the feather sha256s and the graph hash;
- the threshold, the sign map and the NT-missing count;
- the column assignment and the brain id;
- every constant (`g_syn`, `g_eye`, `g_nose`, `b`, `T`, R, `c_ref`);
- the input code and the sense versions;
- the LFG commit whose `trait_config.yaml` the renderer used;
- the taste weights and normalization stats.

Changing any of them makes a new version. The nightly rarity head is a
per-network data-dir artifact identified by its snapshot hash, and each
decision record names it. `fly reproduce <record>` re-simulates a stored
decision and asserts identical scores.

## 3. The teacher

### 3.0 Phase 0: feasibility probe (before any critic tokens)

**Why.** The review simulated the first-draft brain on the real graph (GPU,
integer CSR) against a **planted additive taste**:
- U(look) = Σ_slot θ[slot, value], with θ ~ N(0, 1);
- 3,000 pairs from the real sampler, labelled with P(A ≻ B) = σ(1.2·ΔU);
- 600 held-out pairs split by family (95% CI ±0.040).

Any real taste is at least this hard to learn: the critic's taste has
interactions, while the planted one is purely additive. Results:

| Model | Held-out |
|---|---|
| Bayes ceiling | 0.837 |
| **One-hot, no brain** | **0.770** |
| `lif`, g 8 / g 10 (train 0.716) | 0.498 / 0.497 |
| `rate`, bias 0 / bias 0.1 | 0.552 / 0.540 |
| Projection-neuron readout (LIF / rate) | 0.535 / 0.583 |
| Kenyon-cell readout (LIF / rate) | 0.497 / 0.578 |
| Glomerulus-level codes | 0.513–0.535 |

The LIF is **chaotic**. At g = 8, readout distance from a base look is 235
for a 0.01 concentration change, 306 for a one-slot swap and 311 for all nine
traits new. A one-slot swap moves the readout as far as a full re-draw, so
nothing generalizes. The probe scripts and data are preserved in
`~/fly-data/review-2026-09-22/`.

**The probe.** `fly probe` runs that exact protocol over the grid of candidate
brains (§2) × input codes (§2) × gains, including `b`. Each setting must
first pass the constraints:
- no step has > 10% of neurons firing;
- on average ≥ 5% of readout neurons spike;
- eyes-only (on the tonic baseline) and nose-only stimulation each change the
  readout across looks;
- **smoothness:** the readout distance for a 0.01 concentration change is
  ≤ 10% of the distance for a one-slot swap, and a one-slot swap is closer
  than a full re-draw.

Settings that pass are ranked by held-out planted-taste accuracy. **Readout
variance is never the criterion**, because maximizing it selects chaos. The
report also gives per-slot decodability and the no-brain one-hot control for
comparison.

**The gate.** The best setting reaches **≥ 0.60 held-out**, and its
**rewired twin** (§3.4) is reported beside it.
- **If a setting passes:** it becomes `fly-v1`'s brain, and the build
  continues (§5).
- **If none passes:** the operator chooses between **publishing a negative
  result** (REPORT.md + probe data, the honest outcome the fly-game field
  still lacks) and **re-brainstorming**. The latter might mean relaxing
  "only the readout is trained", with the claims that change as a result.

Nothing downstream (critic tokens, the LFG PR, wallets) starts before this
verdict.

### 3.1 Looks and pairs (after the gate)

- **Catalog:** the union of `/api/rarity?body=<b>` values over all five
  bodies, filtered by `/api/layer?body=male` returning 200. That includes
  female art outside Clothing, and the universal Accessory/Back art from every
  body (e.g. milady-only Accessory values that render on a male).
  - `None` is a value in every non-Body slot. It needs no layer and the
    renderer skips it, as LFG's compose does. `/api/layer` answers 404 for
    some `None`s, so it's never used to test `None`.
- **Sampling:** 40% realistic (each slot drawn from the male mint odds), 30%
  uniform, 30% hybrids (a realistic base with 3 uniform slots).
- **Pairs:** 60% **near** (1–2 slots differ) and 40% far. Every pair belongs
  to a *look family* (its base look), and splits are by family.
- **Renderer** (`teacher/render.py`): stacks cached `/api/layer` thumbs, using
  the first frame for animated art. Z-order comes from **`trait_config.yaml`
  (`layers` + `z_overrides`) at a pinned Team-Hamsa/LFG commit**, recorded in
  the manifest. `/api/economy` needs a wallet session, which the teacher
  phase doesn't have. At run time the body cross-checks that order against
  `/api/economy`'s `z_order` and refuses to decide on a mismatch. Pairs render
  at 1024×512 (A | B, labelled), retina renders at 64×64.

### 3.2 The critic (a Claude Code workflow; no API key)

`.claude/workflows/critic.js` fans out agents that each view 40 rendered pairs
(Read tool) and return, per pair, a structured verdict:
`{pair_id, winner: "A"|"B", strength: 1-3, idea_A, idea_B, reason}`. Idea
names are at most 5 words and reasons at most 15.

**The rubric, which is the fly's taste:**
1. **Appeal:** colour harmony and contrast, a readable silhouette, nothing
   muddy or cluttered, and a background that supports the character.
2. **Idea:** the combination reads as a character concept you could name
   ("Retired Space Pirate").
3. **Surprise:** an unexpected combination that works beats a predictable
   matched set.

It's a forced choice. The critic judges only what it sees and is never told
rarity, price, or which look the fly chose.

**Quality control:**
- **Position bias:** 10% of pairs are shown again with the sides swapped. The
  flip rate is reported, and pairs the critic flips on are dropped.
- **Agreement:** 10% of pairs are judged by a second agent, and their
  agreement **a** is reported. Agreement is a *lower bound* on what a model can
  reach against one critic, not a ceiling. The **implied ceiling**
  (1+√(2a−1))/2 is also reported: exact under uniform label noise, an upper
  estimate when pair difficulty varies.

**Rounds:**
- **Round 1:** 3,000 pairs (+20% QC), about 90 agents.
- **Round 2:** about 1,000 "homework" pairs (the fly's top pick against its
  runner-up or a random legal alternative) plus 100 **best-of-6** grids, about
  25 agents.
- **Token budget:** about 4–5M tokens in total.

Judgments are committed as `data/critic/r<N>.jsonl` together with the look
definitions.

**Naming at runtime, with no API.** A post names the fly's look after the
nearest critic-named look by slot-weighted overlap, labelled "critic's read".
Below a similarity threshold, the look posts as untitled. The optional weekly
review (v1.1) names those.

### 3.3 Training

The looks are simulated with the Phase 0 brain. The taste head uses `c_ref`;
the rarity head uses a second, live-concentration simulation. Training is as
described in §2.

### 3.4 Controls and REPORT.md

**Metrics:**
- Held-out pairwise accuracy, with 95% CIs from a bootstrap that **resamples
  look families** (≥ 2,000 resamples).
- The test split holds ≥ 20% of families, about 600 pairs.
- Best-of-6 agreement against the 1/6 chance rate.

| # | Model | Question it answers |
|---|---|---|
| 1 | Random mover | chance level |
| 2 | **No brain:** logistic regression on one-hot traits + mean RGB + an 8-bin luminance histogram | does the fly beat its own raw inputs? |
| 3 | MLP (256 hidden) on the same raw features | what a conventional net gets, for reference |
| 4 | **Rewired fly** (below) | does the *real wiring* matter? |
| 5 | Sign-shuffled fly (NT labels permuted across neurons) | do the signs matter? |
| 6 | Eyes-only (on the tonic baseline) / nose-only | which sense carries taste? |
| 7 | ≥5-synapse graph | threshold sensitivity |
| 8 | Critic agreement a and implied ceiling | how good the labels are |

**The rewired fly** preserves every neuron's in- and out-degree exactly:
- The thresholded edge list's `post` column is permuted.
- Each self-loop or duplicate the permutation creates (≈ 0.2% on v1.0) is
  repaired: swap its `post` with a randomly chosen edge's `post`, accepting the
  swap only if it creates no new self-loop or duplicate, and repeat until none
  remain.
- Edges keep their synapse counts and their pre neuron's sign, and the
  per-post normalization is recomputed.
- The number of repair swaps is reported.

REPORT.md states the answers, including if the answer is "the wiring doesn't
matter". Every published fly-game control so far found no advantage for real
wiring (firefly, flyt3's satellite project, NeuroTerrarium), and this report
won't be shaded to say otherwise.

**Go-live gate:**
- The taste head's held-out accuracy has a family-bootstrap 95% CI lower
  bound above 0.55. With n ≈ 600, that takes about 0.59–0.60 observed.
- Its best-of-6 agreement beats 1/6 at binomial p < 0.01.
- Beating the controls isn't required, but the report must say whether it
  does.
- **Failing the gate means the fly does not go live as a chooser.** The
  operator then chooses as in §3.0.

## 4. The fly's day (after the gates)

Three pm2 cron jobs (`ecosystem.config.js`, nice + ionice wrapped), clear of
LFG's cron slots (00:10–03:40 UTC):

| Job | When (UTC) | What |
|---|---|---|
| `fly-claim` | 04:10 | claim the BRIX drip (`POST /api/brix/claim`), then poll its status |
| `fly-retrain` | 04:30 | supply snapshot → live-concentration re-simulation → new rarity head |
| `fly-move` | 15:00 | the move loop |

### 4.1 The move loop

1. **Sign in fresh**, with the `agent` provider and the proof signed by the
   RegularKey. The token is kept **in memory only**, never under
   `FLY_DATA_DIR`, and every run ends with `POST /api/logout` in a `finally`
   block.
2. **Reconcile** any non-terminal record first (§4.2). Never proceed while an
   outcome is unknown.
3. **Read state:**
   - the hero (`/api/nfts`);
   - the wardrobe and `z_order` (`/api/economy`, cross-checked against the
     manifest's pinned order);
   - live supply (`/api/rarity/supply`);
   - costs (`swap_fee` from `/api/nfts`; equip 0).
4. **Generate candidates.** This is a disclosed non-fly component: every legal
   single-slot Builder change from the current look, plus **stay**. The hero
   must be mutable and not blank. A change is legal when:
   - its slot is one of the 8 non-Body slots;
   - the value is in the Closet with count > 0 and isn't listed;
   - a value other than `None` resolves on the hero's body
     (`/api/layer?body=male` returns 200, the same `resolve_layer` the
     server's equip gate uses).

   `None` is legal whenever the Closet holds that slot's `None` unit; harvest
   credits one for every empty donor slot, and equip skips the affinity check
   for it. All candidates are simulated; any cap is disclosed as "considered
   N of M".
5. **Decide.** Each candidate gets a taste simulation (`c_ref`) and a rarity
   simulation (live), in GPU batches. A choice is sampled at temperature
   τ = 0.15 with seed `SHA-256(date | fly version)`, so it's reproducible. A
   day's look is up to **3 greedy steps**, stopping early when **stay** wins.
   One disclosed assist: the fly **never re-wears a complete look from the
   last 30 days**.
6. **Record before submitting.** `records/<date>.json` holds:
   - hashes of the inputs;
   - every candidate with its taste, rarity, cost and score;
   - the chosen changes and the neuron stats;
   - the checkpoint and rarity-head versions, and the seed;
   - `{network, lfg_api_base, wallet}`.
7. **Submit:** one `POST /api/equip` with all the changes (one
   `NFTokenModify`, no signature), then poll `/api/equip/{id}`.
8. **Post** (§4.5).

### 4.2 Reconcile rule

- **How an equip session's result maps:**

  | Equip session | Recorded as |
  |---|---|
  | `done` | done |
  | `failed`, `resolution` reverted or null | failed |
  | `failed`, `resolution` **uncertain** (modify, sync or revert indeterminate) | **UNKNOWN** |

- **Resolving UNKNOWN, or a session that no longer exists:**
  1. Wait at least 10 minutes, past any LastLedgerSequence the pending modify
     could carry.
  2. Read the hero's **current URI from the ledger**: clio `nft_info`, or
     `account_nfts` on a public endpoint. Never use LFG's index
     (`/api/nfts`/`/api/economy`); it lags an indeterminate modify.
  3. Fetch that metadata and compare its attributes. The intended look means
     done, the original look means failed, and anything else stays UNKNOWN.
- **A record left UNKNOWN** stops the loop and writes an outbox alert until
  the operator clears it.
- **Never** blindly re-submit a day's move.

### 4.3 Signer policy

The signer covers LFG sign requests plus the few transactions the fly builds
itself during setup. Every signature first runs a `simulate` pre-flight (LFG
skips it on client-signed paths) and the chain-identity check (§4.4).

| Tx | Allowed in | Signer requires |
|---|---|---|
| Payment (proof) | sign-in | Destination `rrrrrrrrrrrrrrrrrNAMEtxvNvQ`, Amount `"1"`, Fee ≤ 10,000 drops, LastLedgerSequence set, canonical memos; **never submitted** by the fly |
| Payment (mint) | `fly setup mint` only, never the loop | the mint session (or bulk job) reports `pay_with == "XRP"`; XRP drops only; Amount equals the session's `pay_amount` × quantity; Destination equals LFG's signing account, pinned per network in the fly's config; no SendMax, Paths or DestinationTag; a setup-wide spend cap (`FLY_SETUP_MAX_XRP`) not exceeded |
| NFTokenAcceptOffer | setup only | `NFTokenSellOffer` only (buy offers and `NFTokenBrokerFee` refused). The offer, **read on-ledger**, has Amount `"0"` and Destination = the fly. Its Owner is the NFT issuer encoded in the NFTokenID (LFG delivery and Closet offers) or a wallet in `FLY_DONOR_SOURCES` (operator transfers, §5) |
| TrustSet | setup only | LimitAmount currency and issuer equal the BRIX pair; no other flags |

Anything else is refused. The autonomous loop signs nothing but the sign-in
proof.

### 4.4 Safety

- **Chain identity.**
  - Check `server_info.network_id` (mainnet 0, testnet 1) plus the
    ledger-32570 hash, asked of full-history public endpoints and never the
    box's validator.
  - **Mainnet** uses s2.ripple.com / xrplcluster.com, and the expected hash is
    a constant,
    `4109C6F2045FC7EFF4CDE8F9905D19C28820D86304080FF886B299F0206E42B5`.
  - **Testnet** uses `clio.altnet.rippletest.net:51233`; the public testnet
    rippled nodes are pruned and answer `lgrNotFound`. The expected hash is
    pinned in the testnet config at setup (`18D82E2D…FC53` today). A testnet
    reset changes it, and the check then fails closed until it's re-pinned.
  - Endpoints that answer `lgrNotFound` or are unreachable are skipped. At
    least one must return the expected hash; any mismatch refuses. The check
    runs once per process start.
  - A fly pointed at the wrong LFG stack also fails at sign-in, because its
    RegularKey proof must match the AccountRoot on that stack's ledger.
- **What a live session can do.** Any live session for the fly's wallet can,
  **with no signature**, equip, harvest, post Closet asks, and fill any
  standing Closet bid. A bid fill sells a Closet unit at the bidder's price,
  and any price above 0 BRIX is valid. A stolen token could drain the
  wardrobe. Three things contain this:
  - fresh per-run tokens held in memory only;
  - an explicit logout at the end of every run;
  - LFG's per-request RegularKey re-check (companion spec §2): once the key
    is removed, every RegularKey-minted session dies within 60 s.
- **Kill procedure:**
  1. Remove the RegularKey in Xaman. Sessions die within 60 s and no new
     sign-in is possible.
  2. `FLY_ENABLED=0`.
- **Caps:** one look change per day, at most 3 slots, a single-flight lock,
  and `FLY_SETUP_MAX_XRP` for setup.
- **The key:**
  - `FLY_REGULAR_SEED` sits in `~/lfg-fly/.env` (gitignored, chmod 600).
  - The **first commit, before any code**, adds `.gitignore` (`.env*`,
    `secrets/`, any data paths) and `.gitleaks.toml`, which has
    `[extend] useDefault = true` plus a rule `xrpl-family-seed`, regex
    `\bs(?:Ed)?[1-9A-HJ-NP-Za-km-z]{28}\b`. The default gitleaks rules do
    **not** detect XRPL seeds.
  - A CI test generates an unfunded seed in a temp repo and asserts that
    gitleaks exits non-zero.
  - The wallet holds only the fly's characters, a small XRP balance and its
    BRIX.

### 4.5 Posts: the fly's own X account

- **What:** a link-free before → after card (1200×675) built from the hero's
  real on-chain images, taken from the metadata `image` field before and after
  the move. Example text:
  > 🪰 Day 12. Tried 147 outfits in 61 ms of fly-brain time; 9,812 neurons fired. Head: Crown → Pirate Hat · Eyes: Laser → Monocle. Critic's read: "Retired Space Pirate".
- **Account:** the bio links REPORT.md, and X's "automated account" label is
  turned on.
- **Budget and plumbing:** OAuth 1.0a (the fly's own X dev app), with
  `FLY_X_MONTHLY_BUDGET` defaulting to 40.
- **Without credentials,** posts go to `outbox/`.

## 5. Rollout

**Build order** (by dependency; each gate blocks what follows):
0. **Phase 0 probe** (§3.0): fetch, build (columns and the integer CSR), the
   candidate brains, the probe, and a verdict. **Gate A.**
1. **LFG agent-users PR** (companion spec) → staging → prod.
2. **Teacher:** renderer, sampler, critic round 1, training, controls,
   REPORT.md, round 2. **Gate B** (§3.4).
3. **Testnet rehearsal on LFG staging** (`:8177`, testnet, economy on,
   `AGENT_SIGNIN_ENABLED=1`):
   - **Wallet:** the fly's testnet wallet is faucet-funded as often as the
     spend cap needs. Its seed lives in a state file refused inside any repo
     (the `fee_cover_rehearsal` pattern), and a RegularKey is set by script.
   - **Donors:** minted through LFG's mint or bulk-mint flow (the only donor
     path LFG's public API can complete; see below).
     - LFG picks each minted donor's body by **live-share odds**, and the
       minter can't choose it. On staging testnet, ape dominates (ape 15 /
       male 9 / female 0 live), so setup keeps minting until the male-legal
       wardrobe reaches its target or the spend cap is hit. Alternatively,
       staging's odds can be adjusted first (rarity dashboard, staging only).
     - Setup harvests everything that arrives.
   - **Setup:** BRIX line, Closet accept, harvests; then the daily loop, with
     posts going to the outbox.
   - **BRIX claims need staging fixes first.** Staging can't accrue or pay
     BRIX today:
     - `~/LFG-staging/.env` has no `BRIX_DISTRIBUTOR_SEED`, so claims return
       503;
     - `stg-brix-accrue` fails its chain-identity check, because no configured
       testnet JSON-RPC endpoint serves ledger 32570;
     - the staging archive has never recorded an accrual.

     The prerequisites are:
     1. Add `https://clio.altnet.rippletest.net:51234/` to staging's
        `XRPL_JSON_RPC_FALLBACK_URLS`.
     2. Create and BRIX-fund a testnet distributor, and set the distributor
        seed and address.
     3. Certify the testnet history archive (`docs/ops/sponsored-free-mint.md`).
     4. Confirm one `stg-brix-accrue` run writes rows.

     If the operator skips these, claims drop out of the rehearsal criteria
     and are first verified on the mainnet fly's first claim.
   - **Exit criteria:** 7 consecutive daily moves with no intervention; a
     deliberately killed mid-move run reconciles cleanly; claims work, if the
     staging prerequisites were done; posts read right.
4. **Mainnet go-live (operator-driven):**
   1. Create the fly wallet in Xaman (the master key stays on the phone).
      **Funding:** about 3 XRP of reserves (1 base + 0.2 per owned object:
      NFTokenPages, the BRIX line, the Closet) plus fees. If the fly mints its
      donors, add the number minted × the mint price (`MINT_PRICE_XRP`,
      10 XRP today; about 150 XRP for 15). Bulk mint takes one Payment for up
      to `BULK_MINT_MAX`.
   2. `fly setup keygen` makes a RegularKey on the box. Set it on the fly
      wallet from Xaman. The rehearsal confirms Xaman's menu path; the
      fallback is any wallet that can sign `SetRegularKey`.
   3. **Donors**, choosing either:
      - **(a) The fly mints them** through LFG. The bodies come by live-share
        odds, about 90% male/female on mainnet.
      - **(b) The operator transfers them.** The operator creates a
        destination-locked, zero-price sell offer to the fly for each donor
        from a wallet listed in `FLY_DONOR_SOURCES`. The fly accepts each one
        **directly on-ledger** with its own key (`fly setup accept`, under the
        §4.3 policy). LFG's pending-offer tray lists only LFG's own offers,
        so this bypasses LFG, exactly as any wallet owner may.
   4. The fly sets its BRIX line, accepts its Closet, and harvests the donors
      (`fly setup …`, via the public API). Donors must be **mutable**;
      fly-minted ones always are.
   5. Choose the hero (a mutable male) and run `fly move --dry-run`. The
      operator reviews the record.
   6. `FLY_ENABLED=1`, then add the X credentials.

**Later phases** (not v1; nothing in v1 blocks them):
- **Already true in v1:** the fly is on `users_swaps`, and its hero on
  `nft_swaps`, from day one. Those boards count every `NFTokenModify` of a
  collection character, and equips and harvests are modifies: about 15 at
  setup, then one per daily move.
- **v2:**
  - Paid trait swaps, taken only when the score clears the BRIX cost, funded by
    the drip. They add to the same swap counts.
  - Closet Market buys, which need LFG to open Closet Market bids/ask-buys to
    client-signed providers.
  - Builder pricing, once LFG publishes it.
- **v3:** community gifts.
- **v1.1:** the weekly critic review post.

## 6. Attribution and licence

- **Data:** MaleCNS v1.0 is CC BY 4.0. Cite Berg et al., *Cell* 189(18):
  5504–5526.e15 (2026), doi:10.1016/j.cell.2026.08.015. State the changes
  (thresholded at ≥3 synapses, signed by predicted transmitter, photoreceptor
  columns derived from synaptic partners, simulated). State that no
  endorsement by HHMI/Janelia, Cambridge, MRC-LMB or Google is implied. This
  goes in the README, REPORT.md and the X bio link. The connectome is
  downloaded, never redistributed.
- **Code:** this repo is MIT. None of it is copied from the chess or
  tic-tac-toe fly repos, which have no licence, or from nfly (MIT, not used).
  The review's probe scripts are throwaway references, not project code. Fly
  vs Chess (@ErnestoSOFTWARE) is credited as the inspiration.

## 7. Testing

- **CPU unit tests, on tiny synthetic graphs:**
  - numpy-reference determinism and refractory behaviour;
  - the integer-CSR two-step form matching a float reference;
  - the sign map (lower-case keys; a missing NT row treated as `unclear`; an
    unknown string fails the build);
  - **eyes-only input on the tonic baseline changing a downstream neuron**, on
    a graph with an inhibitory photoreceptor;
  - partner-derived photoreceptor column assignment, including the < 90%
    build failure;
  - the hex lattice and channel mapping;
  - glomerulus-level odour codes: golden hashes, cosine bounds and the
    1/√size scaling;
  - the concentration curve;
  - Bradley–Terry recovering a planted readout, and ridge recovering a planted
    rarity map;
  - the rewire preserving both degree sequences exactly with no self-loops or
    duplicates, on a graph built to force collisions;
  - the probe protocol end to end on a toy brain, with a known-learnable toy
    that passes and a known-chaotic toy that fails.
- **GPU (skipped without CUDA; CI is CPU-only):**
  - exact spike-train equality between the torch simulator and the numpy
    reference on the tiny graph;
  - a manual real-graph batch check;
  - `fly reproduce`.
- **Body, against a fake LFG API** (an aiohttp test server):
  - agent sign-in with a RegularKey proof;
  - the signer-policy table, both permitted and refused cases per row;
  - chain identity: pinned hash, `lgrNotFound` skip, mismatch refusal;
  - per-network stamps refusing a cross-network record;
  - legality, including `None` and a `/api/layer` 404;
  - greedy multi-step decisions, the 30-day tabu, and the date seed;
  - every reconcile branch, including `uncertain` → UNKNOWN and a
    ledger-URI comparison;
  - in-memory tokens and logout in `finally`;
  - the outbox, when there are no X credentials.
- **gitleaks:** the seed-detection CI test (§4.4).
- **Staging smoke:** the testnet rehearsal.
- **Review:** Greptile and CodeRabbit get installed on this repo (an operator
  step), with the same close-every-finding discipline as LFG.
