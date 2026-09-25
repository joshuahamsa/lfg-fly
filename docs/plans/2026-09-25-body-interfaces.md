# fly-v1 body, brain heads, voice, trainer: the interface contract

Implementers work in parallel on separate files, so every cross-module surface
is fixed here. The design is `docs/specs/2026-09-22-fly-design.md` (§2 Readout,
§2 Decision score, §4, §7); the LFG API is `docs/lfg-api.md`. Where this file
and the spec disagree, the spec wins and the difference is reported.

Conventions: Python 3.10, ruff (line length 100, rules E/F/I/B/UP), pytest.
Tests never touch the network, `~/fly-data` (the autouse fixture sets
`FLY_DATA_DIR` to a tmp dir), or the GPU (skip CUDA-only tests when
`torch.cuda.is_available()` is false; CI is CPU-only). Async code uses
`asyncio`; HTTP is `aiohttp` (3.14); XRPL is `xrpl-py` 5.2
(`xrpl.asyncio.clients.AsyncJsonRpcClient`, `xrpl.wallet.Wallet`,
`xrpl.core.keypairs`, `xrpl.core.binarycodec`, `xrpl.transaction`).
`Look = tuple[str, ...]`: nine values in `lfg_fly.brain.senses.SLOTS` order,
`"None"` for an empty non-Body slot. Nobody edits `lfg_fly/cli.py`; each
command module exposes `register(sub: argparse._SubParsersAction) -> None` and
the integrator wires it. Nobody runs `git commit`.

## Data directory (`lfg_fly/paths.py`, existing)

`FLY_DATA_DIR/<network>/` holds `records/`, `sims/`, `snapshots/`, `renders/`,
`outbox/`, `catalog/` (mainnet has the catalog today; testnet reads mainnet's
layer cache through `catalog_dir(network)` which falls back to mainnet's) and
`wallet.json` (testnet only; chmod 600). The connectome (`raw/`, `graph/`) is
shared. Add to `paths.py`: `records_dir(network)`, `outbox_dir(network)`,
`snapshots_dir(network)`, `catalog_dir(network)` (mainnet fallback),
`wallet_path(network)`, `checkpoint_dir(version)` (= repo `checkpoints/<version>`),
`rarity_head_path(network, snapshot_hash)`.

## `lfg_fly/body/config.py`

```python
@dataclass(frozen=True)
class FlyConfig:
    network: str                 # "testnet" | "mainnet"  (FLY_NETWORK, default "testnet")
    api_base: str                # FLY_API_BASE (default testnet http://localhost:8177, mainnet http://localhost:8176)
    version: str = "fly-v1"      # FLY_VERSION
    beta: float = 0.3            # FLY_BETA, greed, in units of taste sd
    lam: float = 0.05            # FLY_LAMBDA, per BRIX
    temperature: float = 0.15    # FLY_TEMPERATURE
    max_steps: int = 3
    tabu_days: int = 30
    enabled: bool = False        # FLY_ENABLED ("1")
    setup_max_xrp: float = 0.0   # FLY_SETUP_MAX_XRP; setup refuses to exceed it
    donor_sources: tuple[str, ...] = ()   # FLY_DONOR_SOURCES, comma separated
    rpc_urls: tuple[str, ...]    # FLY_RPC_URLS or the per-network defaults below
    expected_ledger_hash: str | None   # mainnet: the §4.4 constant; testnet: FLY_TESTNET_LEDGER_HASH (default the 2026-09-25 pin)
    signing_account: str | None  # FLY_LFG_SIGNING_ACCOUNT: LFG's mint destination for this network (None = unknown → mint refused)
    regular_seed: str | None     # FLY_REGULAR_SEED (mainnet, from ~/lfg-fly/.env); testnet reads wallet.json instead
    x_monthly_budget: int = 40
def load_config(env: Mapping[str, str] | None = None) -> FlyConfig
```
Defaults: testnet RPC `https://s.altnet.rippletest.net:51234/`,
`https://testnet.xrpl-labs.com/`, clio `https://clio.altnet.rippletest.net:51234/`
(the only one that serves ledger 32570); mainnet `https://s2.ripple.com:51234/`,
`https://xrplcluster.com/`. Mainnet hash
`4109C6F2045FC7EFF4CDE8F9905D19C28820D86304080FF886B299F0206E42B5`; testnet
pin `18D82E2D616C76A960DB76D514ED39BCC377381A037DFC0F5AF4F0E22E14FC53`
(verified against clio 2026-09-25; network_id 1). `load_config` never reads
`.env` itself; the CLI loads `~/lfg-fly/.env` into the environment if present.

## `lfg_fly/body/outbox.py`

```python
def write(network: str, kind: str, payload: dict, when: datetime | None = None) -> Path
# FLY_DATA_DIR/<network>/outbox/<UTC timestamp>-<kind>.json; kind in {"alert", "post"}
def alerts(network) -> list[dict]   # every alert not yet cleared
def clear(network, path) -> None     # renames to .cleared
```

## `lfg_fly/body/records.py` (spec §4.1 step 6, §4.2)

```python
STATES = ("dry_run", "pending", "submitted", "done", "failed", "UNKNOWN")
@dataclass
class Stamp: network: str; lfg_api_base: str; wallet: str
@dataclass
class Record:
    date: str                     # YYYY-MM-DD (UTC)
    stamp: Stamp
    version: str                  # checkpoint version
    rarity_head: str | None       # snapshot hash
    seed: str                     # hex of SHA-256(date | version)
    hero: str                     # nft_id
    before: list[str]             # Look
    after: list[str]              # Look
    changes: list[dict]           # [{"slot","value"}], in order
    candidates: list[dict]        # [{"look","changes","taste","rarity","cost","score","considered": bool}]
    considered: int; total: int   # "considered N of M"
    neuron_stats: dict            # from the brain: neurons_fired, ms, etc.
    inputs_hash: dict             # sha256 of supply snapshot, economy state, closet
    state: str
    equip_id: str | None = None
    resolution: str | None = None
    error: str | None = None
    post: dict | None = None      # what was posted / put in the outbox
class StampMismatch(RuntimeError)
def path_for(network, date) -> Path
def write(record: Record) -> Path            # atomic (tmp + rename)
def read(path, expect: Stamp) -> Record      # raises StampMismatch on any field differing
def non_terminal(network, expect: Stamp) -> list[Record]   # pending/submitted/UNKNOWN
def recent_looks(network, expect: Stamp, days: int, today: date) -> set[Look]   # done records' `after`
def date_seed(date: str, version: str) -> int   # int.from_bytes(sha256(f"{date}|{version}")[:8], "little")
```

## `lfg_fly/body/chain.py` (spec §4.4 chain identity; ledger reads; submit)

```python
class ChainIdentityError(RuntimeError)
@dataclass
class ChainIdentity: network_id: int; ledger_32570: str; endpoint: str
class Ledger:
    def __init__(self, urls: Sequence[str], network: str, timeout: float = 20.0)
    async def identity_check(self, expected_hash: str | None) -> ChainIdentity
        # for each url: server_info.network_id must be 0 (mainnet) / 1 (testnet); ledger(32570).ledger_hash
        # lgrNotFound or unreachable → skip; the first url returning the expected hash wins;
        # any url returning a DIFFERENT hash → ChainIdentityError; none returning it → ChainIdentityError
    async def request(self, req) -> dict           # first url that answers; raises LedgerUnavailable
    async def account_info(self, address) -> dict | None       # None on actNotFound
    async def account_nfts(self, address) -> list[dict]
    async def nft_info(self, nft_id) -> dict | None             # clio nft_info; None if unsupported
    async def nft_uri(self, nft_id, owner) -> str | None        # nft_info, else account_nfts scan
    async def sell_offers(self, nft_id) -> list[dict]           # nft_sell_offers, [] on objectNotFound
    async def autofill(self, tx: dict) -> dict                  # Fee, Sequence, LastLedgerSequence (+20)
    async def simulate(self, tx_json: dict) -> dict             # `simulate` RPC; raises SimulateFailed unless engine_result == "tesSUCCESS"
    async def submit_and_wait(self, tx_blob: str) -> dict       # validated result incl. meta, hash; raises SubmitFailed
    async def tx(self, tx_hash) -> dict | None
    async def validated_ledger_index(self) -> int
```
Tests use an aiohttp fake JSON-RPC server (`tests/fake_rpc.py`: a dict of
method → responder; `lgrNotFound`, unreachable, mismatch cases).

## `lfg_fly/body/signer.py` (spec §4.3)

```python
class PolicyError(RuntimeError)
@dataclass(frozen=True)
class Purpose:
    kind: str    # "proof" | "mint_payment" | "accept_offer" | "trustset"
    # mint_payment: pay_amount_xrp: Decimal, quantity: int, destination: str (LFG's signing account, pinned)
    # accept_offer: allowed_owners: frozenset[str] (issuer from the NFTokenID is always allowed; plus donor sources)
    # trustset: currency: str, issuer: str (the BRIX pair)
class Signer:
    def __init__(self, seed: str, ledger: Ledger, config: FlyConfig, spend_ledger: SpendLedger | None = None)
    address: str; public_key: str
    def sign_proof(self, tx_json: dict, source_tag: int) -> dict
        # §4.3 row 1: Payment, Destination rrrrrrrrrrrrrrrrrNAMEtxvNvQ, Amount "1", Fee ≤ 10000, LastLedgerSequence set,
        # SourceTag == source_tag, Memos present and unchanged; returns tx with SigningPubKey + TxnSignature. NEVER submitted.
    async def sign_and_submit(self, tx_json: dict, purpose: Purpose) -> dict
        # autofill → policy(tx, purpose) → ledger.simulate → sign → ledger.submit_and_wait; returns {"hash", "result"}
    def policy(self, tx: dict, purpose: Purpose) -> None   # raises PolicyError; pure; unit-tested per §4.3 row, permitted and refused
class SpendLedger:   # setup-wide XRP cap (FLY_SETUP_MAX_XRP), persisted at FLY_DATA_DIR/<network>/setup-spend.json
    def charge(self, drops: int) -> None   # raises PolicyError past the cap
```
Policy per row (§4.3): Payment(mint): Amount in XRP drops (a string), equals
`pay_amount × quantity` in drops, Destination == purpose.destination, no
`SendMax`/`Paths`/`DestinationTag`, spend cap. NFTokenAcceptOffer:
`NFTokenSellOffer` only (no `NFTokenBuyOffer`, no `NFTokenBrokerFee`); the
offer read on-ledger (`ledger.sell_offers`) has `Amount == "0"`, `Destination ==
signer.address`, and `owner` ∈ {issuer encoded in the NFTokenID
(`xrpl.utils`/manual: bytes 4..24 of the id → classic address)} ∪
purpose.allowed_owners. TrustSet: `LimitAmount.currency/issuer` == the BRIX
pair, `Flags` ⊆ {tfSetNoRipple 0x20000, tfFullyCanonicalSig}. Anything else,
including any other TransactionType, is refused. The signer holds only the
RegularKey seed; `Account` in every tx must equal `signer.address`'s account,
which is the fly's wallet: the signer is constructed with `account: str` (the
wallet whose RegularKey it is) — add that parameter; `Account` must equal it.

## `lfg_fly/body/client.py` (docs/lfg-api.md)

```python
class LfgError(RuntimeError): status: int; code: str | None; body: dict | str
class LfgClient:
    def __init__(self, api_base: str, session: aiohttp.ClientSession | None = None, timeout: float = 30.0)
    async def __aenter__(self) -> LfgClient; async def __aexit__(...)   # logout if signed in, always
    token: str | None      # memory only; never written anywhere
    wallet: str | None
    async def sign_in(self, signer: Signer, ledger: Ledger) -> str   # POST /api/web/signin {provider:"agent"} → build the proof
        # tx: {TransactionType Payment, Account signer.account, Destination PROOF_DESTINATION, Amount "1", Fee "12",
        #  Sequence 1, LastLedgerSequence validated+20, SourceTag source_tag, Memos (verbatim)} → signer.sign_proof → POST proof
    async def logout(self) -> None
    async def me(self), health(self), nfts(self), economy(self), rarity_supply(self), rarity(self, body)
    async def layer_resolves(self, body, slot, value) -> bool          # GET /api/layer 200 vs 404; memoized per client
    async def equip(self, nft_id, changes: list[dict]) -> dict; async def equip_status(self, sid) -> dict
    async def harvest(self, nft_id) -> dict; async def harvest_status(self, sid) -> dict
    async def closet(self) -> dict                                       # POST /api/closet
    async def brix(self), brix_claim(self), brix_claim_status(self, cid), brix_trustline(self), brix_trustline_status(self, uuid)
    async def mint(self, ref=None), mint_status(self, sid), mint_active(self), bulk_mint(self, quantity), bulk_status(self, jid), bulk_unit_accept(self, jid, index)
    async def pending_offers(self), accept_offer(self, offer_index)
    async def sign_request(self, sid) -> dict
    async def sign_result(self, sid, *, tx_hash: str | None = None, rejected=False, error=None) -> dict
        # retries 202/503 with backoff until the row's expires_at, then raises LfgError
    async def wait(self, getter, sid, terminal: set[str], timeout: float, every: float = 3.0) -> dict   # poll a session
    @staticmethod def sign_id(link: str) -> str    # "lfg-wc://wc-…" → "wc-…"; raises ValueError otherwise
```
Every non-2xx raises `LfgError(status, code, body)`. `tests/fake_lfg.py`: an
aiohttp app with in-memory state (flags: `agent_enabled`, `economy`; a
`characters` list; `closet`; `equip_outcome` in {done, failed_reverted,
failed_uncertain, hang}; sign requests store; mint sessions that advance on
`sign_result`), verifying the proof's signature and memos like LFG does
(`xrpl.core.keypairs.is_valid_message`), and a pytest fixture `fake_lfg` →
`(base_url, state)`. It is shared by the loop, setup and reconcile tests.

## `lfg_fly/brain/readout.py`, `lfg_fly/brain/checkpoint.py` (spec §2 Readout, Checkpoint)

```python
@dataclass
class TasteHead: mu: np.ndarray; sd: np.ndarray; w: np.ndarray; lam: float; taste_sd: float
    def score(self, X: np.ndarray) -> np.ndarray        # ((X-mu)/sd) @ w
def fit_taste(X, a_idx, b_idx, y, weight, family, device) -> TasteHead      # weighted Bradley–Terry via probe.fit_bt(..., w=)
@dataclass
class RarityHead: mu; sd; w; b; snapshot_hash: str; target_mu: float; target_sd: float
    def score(self, X) -> np.ndarray                     # standardized nft_rarity prediction
def fit_rarity(X, target, snapshot_hash, lam=1.0) -> RarityHead               # ridge
def concentrations(looks, supply: dict, catalog) -> np.ndarray [n, 9]         # §2: c = 0.4 + 0.6·p, p = percentile of n_live/freq within slot; zero supply → p = 1
def nft_rarity(look, supply) -> float                                         # Σ n_live/freq over the 9 traits (freq 0 → n_live)
def save_head(path, head) / load_head(path)                                   # npz + json

@dataclass
class Manifest: version; graph_hash; feathers: dict[str, str]; min_syn; sign_map: dict; nt_missing: int; columns_hash: str | None;
                setting: dict (brain params, code, g_in); c_ref; codes: dict (odor "v2", allsens "v1"); lfg_trait_config_commit;
                catalog_hash; taste: dict (heldout, ci_lo, ci_hi, lam, n_train, n_test, gate_pass); created_at
def write_checkpoint(dir, manifest, head) / read_checkpoint(dir) -> (Manifest, TasteHead)
class FlyBrain:
    @classmethod
    def load(cls, checkpoint_dir: Path, device: str, catalog_dir: Path) -> FlyBrain   # graph, pops, retina, Simulator, InputBuilder, head
    def features(self, looks: list[Look], conc: np.ndarray | None = None) -> tuple[np.ndarray, dict]   # via teacher.grid.features_for
    def taste(self, looks) -> np.ndarray                 # c_ref simulation → head.score
    def rarity(self, looks, conc, head: RarityHead) -> np.ndarray
    stats of the last run: neurons_fired, ms
```

## `lfg_fly/body/legality.py`, `candidates.py`, `decide.py` (spec §4.1 steps 4–5)

```python
def legal_changes(hero: Look, closet_assets: list[dict], listed: set[tuple[str,str]], resolves: Callable[[str, str], bool]) -> list[tuple[str, str]]
    # non-Body slots; value in the Closet with count > 0 and not listed; value != hero's current; "None" legal iff the Closet holds that slot's None unit;
    # value other than None must resolve (resolves(slot, value) True)
@dataclass
class Candidate: look: Look; changes: tuple[tuple[str, str], ...]; taste: float = nan; rarity: float = nan; cost: float = 0.0; score: float = nan
def candidates(hero: Look, legal: list[tuple[str, str]], tabu: set[Look]) -> list[Candidate]   # every legal single-slot change + stay (changes=()); tabu looks removed (stay never)
def score(taste, rarity, cost, beta, lam, taste_sd) -> float          # taste + beta·taste_sd·rarity − lam·cost
def choose(cands: list[Candidate], temperature: float, rng: np.random.Generator) -> Candidate   # softmax(score/temperature)
async def plan_day(hero: Look, *, legal_fn, brain_taste, brain_rarity, cost_fn, tabu, cfg, rng, max_steps) -> tuple[Look, list[Candidate], list[list[Candidate]]]
    # up to max_steps greedy steps; each step scores every candidate; stops when stay wins; returns final look, the chosen chain, and every step's candidates
```
Equip cost is 0 (a constant `EQUIP_COST_BRIX = 0.0` in decide.py).

## `lfg_fly/body/reconcile.py` (spec §4.2)

```python
def classify(equip_status: dict) -> str     # done | failed | UNKNOWN per the §4.2 table
async def resolve_unknown(ledger: Ledger, record: Record, fetch_metadata: Callable[[str], Awaitable[dict]], now) -> str
    # waits until record.submitted_at + 10 min has passed (caller sleeps; this function just refuses earlier with NotYet),
    # reads the hero's current URI from the ledger (never LFG's index), fetches metadata, compares attributes:
    # intended look → done; original → failed; else UNKNOWN
def attributes_to_look(attributes: list[dict]) -> Look
```

## `lfg_fly/body/loop.py` (spec §4.1) and `lfg_fly/body/cli_move.py`

```python
async def move(cfg: FlyConfig, brain: FlyBrain, *, dry_run: bool, today: date | None = None, ledger=None, client=None, signer=None) -> Record
```
1. Refuse unless `cfg.enabled` or `dry_run`. Chain identity check once. 2. Sign in
fresh (agent provider). 3. Reconcile every non-terminal record first (§4.2);
an UNKNOWN that cannot be resolved stops the loop with an outbox alert. 4. Read
`/api/nfts`, `/api/economy` (cross-check `z_order` against the checkpoint's
pinned order; refuse on mismatch), `/api/rarity/supply`. 5. Pick the hero
(the record's, else the first mutable non-blank male character; `FLY_HERO`
overrides). 6. Candidates + decide (temperature, date seed; 30-day tabu). 7.
Write the record (`pending`) BEFORE submitting. 8. `POST /api/equip` with all
changes; poll `/api/equip/{id}`; classify; update the record. 9. Post (outbox).
`dry_run` stops after 7 with `state="dry_run"` and never calls equip. Logout in
`finally`. Single-flight lock: `FLY_DATA_DIR/<network>/move.lock` (fcntl).

## `lfg_fly/body/setup.py` and `cli_setup.py` (spec §5.3, §4.3)

`fly setup keygen` (RegularKey pair → wallet.json on testnet; prints the public
address for Xaman on mainnet), `faucet` (testnet: create the master wallet via
the faucet, fund it, store master_seed), `regular-key` (testnet: SetRegularKey
signed by the master seed, once), `trustline`, `closet` (POST /api/closet →
accept → until active), `mint --count N` (mint one at a time or bulk; sign the
Payment under `mint_payment`; accept each delivery under `accept_offer`;
respects the spend cap), `harvest` (every mutable non-hero character except
the hero; wait for each), `status` (prints wallet, balance, characters,
closet). Every signature goes through `Signer.sign_and_submit`.

## `lfg_fly/voice/` (spec §3.2 naming, §4.5)

```python
# naming.py
def nearest_name(look: Look, critic_records: list[dict], threshold: float = 0.6) -> str | None
    # slot-weighted overlap with the critic's judged looks; the name is the winner's idea; None below threshold
# card.py
def before_after(before: Image.Image, after: Image.Image, day: int, changes: list[dict]) -> Image.Image   # 1200×675
# post.py
def compose(record: Record, name: str | None, day: int) -> str    # the §4.5 text; "untitled" when name is None
def publish(cfg, text, card: Image.Image, record) -> dict         # no X credentials → outbox.write("post", …) with the PNG beside it
```

## `lfg_fly/teacher/train.py` (spec §3.3, §3.4) and `fly train`

```python
@dataclass
class LabelSet: looks: list[Look]; a_idx; b_idx; family; near; y; w (strength); test (family split ≥ 20%); n_dropped: int
def load_labels(jsonl: Path, cat: Catalog, test_frac=0.2, seed=0) -> LabelSet     # skips flipped pairs
def train(ctx: G.Context, labels: LabelSet, setting: G.Setting, out_dir: Path, controls: bool = True) -> dict
    # fly features (features_for), weighted BT (probe.evaluate with w), controls: one_hot+pixels LR, MLP, rewired, sign_shuffled,
    # eyes-only / nose-only best Phase 0 settings for those codes (from data/probe/stage_b.jsonl), paired diffs, gate (ci_lo > 0.55)
def write_report(results: dict, qc: dict, out_md: Path) -> None                 # REPORT.md
def write_checkpoint(...)                                                       # checkpoints/fly-v1
```
`probe.evaluate`/`fit_bt`/`evaluate_mlp` gain an optional `w` (per-pair
weights, default None = 1); existing tests must still pass unchanged.
