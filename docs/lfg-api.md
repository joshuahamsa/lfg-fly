# LFG public HTTP API, as the fly uses it

Derived from `Team-Hamsa/LFG` at `ef743fb9` (agent users A/B/C merged: #602,
#603, #610). Line numbers refer to `lfg_service/app.py` at that commit unless
another file is named. This is the contract `lfg_fly/body/client.py` codes to;
when LFG changes, this file and the client change together.

## 0. Conventions

- **Base URL:** the service listens on `WEBAPP_PORT` (staging `8177`, testnet;
  prod `8176`, mainnet).
- **Auth header:** `Authorization: Bearer <session_token>` (`require_auth`,
  1000–1025). Missing/bad/expired/revoked → `401 {"error":"unauthorized"}`.
- **RegularKey tokens** (token carries `key:"regular"`, `signer`) are
  re-checked against the ledger at most once per 60 s (`_regular_key_refusal`,
  967): key removed or rotated → `401 {"code":"key_revoked"}` and the token is
  revoked; ledger unreadable with no cached answer → `503 {"code":"key_unverified"}`.
- **`require_wallet`** (1028): for a web/agent session the wallet is the token's
  `id`. No wallet → `400 {"error":"no wallet registered"}`.
- **Client-signed dispatch:** `require_auth` runs the handler inside
  `signing_context.use(provider, wallet)`. Any transaction the session causes
  whose `Account` equals the session wallet becomes a sign request
  (`lfg-wc://wc-<hex>`) instead of a Xaman payload
  (`xumm_ops.should_use_client_signing`, `CLIENT_SIGNED_PROVIDERS =
  {"walletconnect","agent"}`). A transaction built without `Account` is handed
  to Xaman; the Closet Market bid and ask-buy refuse client-signed sessions
  with `409 wallet_unsupported`.
- **Memos:** an agent session's user-signed transactions carry
  `platform=agent`; backend-signed ones it starts (equip modify, claim payout)
  also `agent` (`memos.backend_platform()`).
- **Session TTL:** 6 h (`SESSION_TTL`, 124). No refresh; sign in again.
- **Token format:** `base64url(JSON{id,name,platform,provider,exp,jti[,key,signer]}) + "." + hex HMAC-SHA256`.

## 1. Agent sign-in

### `POST /api/web/signin` → `handle_web_signin_start` (10015)
- No auth. Body `{"provider":"agent"}`.
- Gates, in order: `AGENT_SIGNIN_ENABLED` off → `503 {"code":"agent_disabled"}`
  (before the rate limiter); >5 starts / 60 s / IP → `429 {"code":"rate_limited"}`.
- Creates a `sign_requests` row: `purpose="signin"`, `nonce=token_hex(32)`,
  TTL 300 s, `created_ledger` = current validated ledger, `provider="agent"`.
- **200:** `{sign_id: "wc-"+32hex, nonce: 64 hex, source_tag: int (2606160021),
  memos: [...], expires_at: float, provider: "agent"}`.
- `memos` is the exact `Memos` array for the proof (use verbatim):
  `initiator=user`, `platform=agent`, `action=signin` (each with
  `MemoFormat=text/plain`), then `lfg/nonce=<nonce>` (no MemoFormat). Hex is
  lowercase.

### `POST /api/web/signin/proof` → `handle_web_signin_proof` (10282)
- No auth (the single-use `sign_id` is the secret). Body
  `{"sign_id", "tx_json"}`. Rate limit 5 / 60 s / IP.
- `tx_json` is a well-formed, **never submitted** Payment
  (`lfg_core/signing/proof.py` `build_proof_tx` / `verify_proof`):

  | field | requirement |
  |---|---|
  | `TransactionType` | `"Payment"` |
  | `Account` | the wallet; not the destination |
  | `Destination` | `"rrrrrrrrrrrrrrrrrNAMEtxvNvQ"` |
  | `Amount` | the string `"1"` (`DeliverMax` may equal it) |
  | `Fee` | digit string, `0 < Fee <= 10000` |
  | `Sequence` | int > 0, **not checked against the ledger** |
  | `LastLedgerSequence` | required, int > 0, `<= created_ledger + 900` |
  | `SourceTag` | `== source_tag` from the start response |
  | `Memos` | decode exactly to `{initiator:user, platform:agent, action:signin, "lfg/nonce": nonce}` |
  | `SigningPubKey` / `TxnSignature` | hex; signature verified locally over `encode_for_signing(tx minus TxnSignature)` |
  | `Flags` | optional; only `0` or `0x80000000` |
  | `NetworkID` | omit |

  No other keys (closed allowlist). Sign at the binary-codec level; no ledger
  autofill needed.
- **Key rule** (`_check_signing_key`, one validated `account_info`):
  master key passes unless `lsfDisableMaster`; a RegularKey must equal the
  account's validated `RegularKey`; lookup failure → `503
  {"code":"regular_key_unverified"}`; an unfunded account counts as
  master-enabled. A proof signed right after `SetRegularKey` fails until it
  validates.
- Row errors: unknown/other purpose 404; expired `410 proof_expired`; consumed
  `409 proof_replayed`; any `ProofError` `400 {"code":"bad_proof"}`.
- **200:** `{"state":"signed","wallet","session_token","user":{"id","username"}}`.
  The token carries `platform:"web"`, `provider:"agent"`, and `key:"regular"`
  + `signer` when a RegularKey signed.

### `POST /api/logout` → `handle_logout` (1859)
Bearer; exempt from the RegularKey recheck; no body. `{"ok":true}`.

### `GET /api/me` (1842)
Bearer. `{"id","username","wallet"}`; for agent sessions `id == wallet`.

## 2. Reads

### `GET /api/nfts` → `handle_nfts` (9225), `require_wallet`
`{"nfts":[NFT...], "swappable_traits":[8 slots], "swap_fee":{"pay_with","amount","per_nft"}|null, "swap_matrix":{...}}`.
Each NFT (`swap_meta.normalize_nft`): `nft_id, name, number, season, image,
video, burn_count, gender (body class), attributes: [{trait_type, value}]`
padded to all 9 slots in `Background, Back, Body, Clothing, Mouth, Eyebrows,
Eyes, Head, Accessory` with `"None"` for empty, `blank: bool, mutable: bool
(lsfMutable), uri_hex`. `swap_fee` is top-level (null if the quote timed out).
Error `502 failed to load wallet NFTs`.

### `GET /api/economy` → `handle_economy` (11863), `require_wallet`
Gate `ECONOMY_ENABLED` (else `403 economy_disabled`). Response
(`webapp/economy_api.read_economy_state`):
```
{"characters":[{nft_id, edition, body, mutable, image_url, video_url, attributes:[{trait_type,value}], blank}],
 "closet":{"assets":[{slot, value, count}], "token":{"status":"none"|"pending_accept"|"active","nft_id"}},
 "trait_order":[9 slots],
 "z_order":{"layers":{"<layer>":int}, "z_overrides":[{trait_type, value, z}]},
 "slots":[8 non-body slots],
 "trait_tokens":[{nft_id, slot, value}]}
```
There is no "hero" field; the fly picks its hero from `characters`.

### `GET /api/rarity/supply` (2824), no auth, cached 60 s
`{"network","as_of","n_live","counts":{"<trait_type>":{"<value>":int}}}`.

### `GET /api/rarity?body=` (2763), no auth
`{"body","slots":{"<Category>":[{value, odds_pct, share_pct, live_count, enabled}]},"stale_slots","generated_at"}`.

### `GET /api/layer?body=&trait=&value=[&thumb=1]` (11819), no auth
Image on 200; **empty 404 with `Content-Type: image/png`** when it doesn't
resolve; `400 {"error":"bad layer params"}` on bad params. `thumb=1` → the 512
px preview.

## 3. Equip (Builder; no user signature)

### `POST /api/equip` → `_economy_post("equip")` (12027), `require_wallet`
Body `{"nft_id", "changes":[{"slot","value"}, ...]}` (≤8, no duplicate slots,
non-empty). Checks (`economy_api.start_equip` 395, `economy_flow.run_equip`
1303): owned and indexed; Closet holds each `(slot, value)` with count ≥ 1,
debiting through the batch in order (units reserved by a Closet Market ask are
excluded: `400 ... is listed in your Closet market`); `can_equip`: not burned,
**not blank** (`409 blank_character`), **mutable** (`400 cannot equip:
character is not mutable`), slot non-body, asset present; `resolve_layer` body
affinity for `value != "None"` (`400 '<v>' does not fit a <body> body`).
Response `{"id","kind":"equip","state":"running","error":null,"displaced":[],"resolution":null,"platform"}`.
Other errors: `409 an economy action is already in progress` / `wait for your
running harvests`, `400 missing or invalid field: 'nft_id'`, `502 could not
start the action`.

### `GET /api/equip/{id}` (12219), `require_auth`
`{"id","kind":"equip","state":"running"|"done"|"failed","error","displaced":[{slot,value}],"resolution":"committed"|"reverted"|"uncertain"|null,"platform"}`.
`uncertain` = outcome unknown (modify or Closet sync indeterminate, or the
revert failed); `null` with `failed` = generic crash, treat as reverted.
Terminal sessions are kept 1 h.

## 4. Harvest

### `POST /api/harvest` (12036), `require_wallet`
Body `{"nft_id"}`. Needs an **active Closet** (`400 Create and claim your
Closet first.`); owned; `can_harvest`: not burned, not blank, mutable or
burnable. Mutable path: `NFTokenModify` to blank in place, every slot plus
Body goes to the Closet, no signature. Legacy path: burn, remint as mutable
blank, `accept = "lfg-wc://..."` (an `NFTokenAcceptOffer`), `new_nft_id` set.
Response `{"id","kind":"harvest","state":"running","error","moved_assets":[],"accept","accept_push","new_nft_id","platform"}`.
Batch: `POST /api/harvest/batch {"nft_ids":[...≤20]}` →
`{"results":[{nft_id, session_id, state, error}]}`.

### `GET /api/harvest/{id}` (12220)
`{"id","kind","state","error","moved_assets":[[slot,value],...],"accept","accept_push","new_nft_id","platform"}`.

## 5. Closet

### `POST /api/closet` (7285), `require_wallet`, no body
Promotes `pending_accept` → `active` if the ledger shows ownership; otherwise
mints the Closet if needed and offers it. Response
`{"status":"pending_accept"|"active","nft_id","accept":"lfg-wc://..."|null,"accept_push"}`.
Sign the accept (§8), then call again until `active`. Errors:
`503 {"error":"closet_mint_transient","retryable":true}`; `502 could not
create or retrieve Closet` (indeterminate; do not retry blindly). No
`GET /api/closet`; contents come from `/api/economy`.

Closet Market (listing only; agents refused on bid and ask-buy):
`GET /api/closet/book`, `/keys`, `/orders/mine`; `POST /api/closet/ask`,
`DELETE /api/closet/ask/{id}`, `POST /api/closet/ask/{id}/buy` (409 for
agents), `POST /api/closet/bid` (409 for agents), `GET/DELETE
/api/closet/bid/{id}`, `POST /api/closet/bid/{id}/fill`, `GET /api/closet/fill/{id}`.

## 6. BRIX

- `GET /api/brix` (1987): `{wallet, claimable, unlisted_last_epoch, accrued_total, claimed_total, open_claim:{claim_id,state,tx_hash}|null, last_epoch, ...}`.
- `POST /api/brix/claim` (2397), no body, backend-signed payout:
  `{"claim_id","state":"confirmed"|"failed"|"submitted","amount","tx_hash"}`;
  errors `503 claims_disabled` (staging today: no `BRIX_DISTRIBUTOR_SEED`),
  `409 trustline_required`, `503 claim_unavailable`, `400 nothing_to_claim`,
  `409 claim_in_flight`, `502 claim_unconfirmed`.
- `GET /api/brix/claim/{claim_id}` (2565): `{"claim_id","state":"pending"|"submitted"|"confirmed"|"failed","amount","tx_hash"}`.
- `POST /api/brix/trustline` (2119): `{"state":"already_set"}` or
  `{"state":"pending","uuid":"wc-...","xumm_url":"lfg-wc://wc-...","qr_png":null,...}`.
  The tx is `TrustSet {Account, Flags:131072, LimitAmount:{currency:BRIX_CURRENCY_HEX, issuer:BRIX_ISSUER, value:"1000000000"}}`.
- `GET /api/brix/trustline/{uuid}` (2199): `pending|opened`, `validating`,
  `signed {tx_hash}`, `rejected {code}`, `expired`.

## 7. Mint

### `POST /api/mint` (7545), `require_wallet`
Optional `{"ref"}`. Pricing (`mint_flow.MintSession.prepare_payment`): LFGO
line with balance → `pay_with:"LFGO"`; otherwise `pay_with:"XRP"`,
`pay_amount = MINT_PRICE_XRP` ("10"), a Payment of `xrp_to_drops(pay_amount)`
to `config.SIGNING_ACCOUNT` (the bot wallet); `SPONSORED` → no payment.
Response (`MintSession.to_dict`): `id, platform, state, error, reason,
sponsored, sponsorship_reason, pay_with, pay_amount, payment_link
("lfg-wc://wc-..."), payment_push, qr_scanned, accept_scanned, accept_signed,
nft_number, nft_id, image_url, video_url, traits, body_type, accept_qr_url,
accept_deeplink, accept_push`. The sign request's txjson is `{Payment,
Account, Destination, Amount, SourceTag, Memos(action=mint)}`. Payment is
detected on-ledger within 300 s (`PAYMENT_TIMEOUT_SECONDS`).
Errors: `409 mint already in progress`, `409 collection_full`, `409
wallet_unfunded|wallet_blocks_nft_offers|wallet_reserve_short`, `409
insufficient_xrp {needed_drops, spendable_drops}`, `503 rate_limited`.
States: `awaiting_payment → generating → minting → creating_offer →
offer_ready` (terminal), plus `done`, `failed`, `payment_timeout`, `cancelled`.

### `GET /api/mint/{id}` (9561)
At `offer_ready`, `accept_deeplink = "lfg-wc://wc-..."`: an
`NFTokenAcceptOffer {Account, NFTokenSellOffer}` to sign. Also
`GET /api/mint/active`, `POST /api/mint/{id}/regenerate`, `POST /api/mint/{id}/cancel`,
`GET /api/sessions/active`.

### Bulk: `POST /api/mint/bulk {"quantity"}` (7837)
Quantity clamped to `min(qty, BULK_MINT_MAX=10, headroom)`. Response
(`BulkMintJob.to_dict`): `id, state, requested_qty, quantity, pay_with,
pay_amount (total), payment_link, units:[{index, state:"pending"|"minted"|"offered"|"failed", nft_number, nft_id, image_url, offer_id, error, traits, body_type}], minted, offered`.
`GET /api/mint/bulk/{id}`; `POST /api/mint/bulk/{id}/units/{index}/accept` →
`{"qr":null,"link":"lfg-wc://wc-...","push":null}`.

## 8. Sign requests (client-signed)

Ids are `wc-` + 32 hex; `lfg-wc://<id>`. Transaction rows live 900 s. States:
`pending, signed, rejected, failed, mismatch, expired, cancelled, consumed`.

### `GET /api/sign/{id}` (10587), `require_wallet`
`{"id","state","txjson":{...},"expires_at","txid"}`. `txjson` already carries
`Account` (the wallet), `SourceTag` and `Memos`. Foreign/unknown → 404.

### `POST /api/sign/{id}/result` (10884), `require_wallet`
Body exactly one of `{"hash":"<64 hex>"}` (you signed **and submitted**),
`{"rejected":true}`, `{"error":"..."}`. The server never accepts a blob: it
fetches the hash with `tx` and checks (`_verify_request_tx`, 10498):
validated; `Account == row.wallet` and same `TransactionType`; every stored
field survives (`_semantic_match`), with no extra fields beyond the autofill
keys (`Fee, Sequence, LastLedgerSequence, Flags, TicketSequence, AccountTxnID,
NetworkID`) and ledger extras; `Flags` equal once `tfFullyCanonicalSig` is
masked; hash unclaimed; close time ≥ `created_at − 120`; `tesSUCCESS`.
Responses: `200 {"state":"signed","txid"}`; `202 {"state":"pending","code":"tx_not_found"}`
(retry until `expires_at`); `503 ledger_unavailable` (retry); `409 tx_mismatch`;
`409 tx_failed`; `410 tx_not_found` / `410 expired`; `409 already_resolved`
(a re-post of the same hash returns 200); `400 bad_request`; `404/403 not_your_request`.

**Client recipe:** GET the request → autofill only `Fee`, `Sequence`,
`LastLedgerSequence` → sign, submit, wait for validation → POST `{"hash"}`,
retrying on 202/503.

## 9. Pending offers

- `GET /api/offers/pending` (8342): sell offers from the bot wallet locked to
  you: `{"offers":[{offer_index, nft_id, kind:"character"|"trait"|"closet"|null, nft_number, image, amount, price_label, ...}]}`.
- `POST /api/offers/accept {"offer_index"}` (8383) → `{"qr":null,"link":"lfg-wc://wc-...","push":null}`;
  errors `410 offer_gone`, `409 trustline_required|offer_trustline_required|insufficient_brix`, `502 payload_failed`.

## 10. Health and config

- `GET /api/health` (11314): `{"ok":true,"active_sessions","detail":{mint,swap,economy,market},"oldest_session_age"}`.
- `GET /api/config` (11260): `{client_id, dev_mode, economy_enabled, market_enabled, ..., bulk_mint_max, walletconnect}`.

## Config values (`lfg_core/config.py`)

| name | value / notes |
|---|---|
| `AGENT_SIGNIN_ENABLED` (:747) | `env_enabled`; **unset on staging as of 2026-09-25** |
| `XRPL_NETWORK` (:59) | default `mainnet`; staging `testnet` |
| `SIGNING_ACCOUNT` (:102) | the bot wallet / mint destination; derived from `SEED` when unset |
| testnet issuers (:74–85) | `SWAP_ISSUER` / `BRIX_ISSUER` = the SEED address on testnet; mainnet `rLfgoMintj3KBcs4s2XKtquvDwEte2kYfJ` / `rLfgoBriX5ZaMP32mtc7RUZJcjnisKh2Px` |
| `MINT_PRICE_XRP` (:223) | `"10"` |
| `BULK_MINT_MAX` (:233) | 10 |
| `PROOF_DESTINATION` / `PROOF_AMOUNT` | `rrrrrrrrrrrrrrrrrNAMEtxvNvQ` / `"1"` |
| `MAX_PROOF_FEE_DROPS` / `PROOF_LLS_WINDOW` / `SIGNIN_TTL` | 10000 / 900 / 300 |
| `SESSION_TTL` (app.py:124) | 21600 s |
| `SOURCE_TAG` (:539) | 2606160021 |
| `PAYMENT_TIMEOUT_SECONDS` (:418) | 300 |

Memos (`lfg_core/memos.py build_memos_json`): a list of
`{"Memo":{"MemoType":hex(key),"MemoData":hex(value),"MemoFormat":hex("text/plain")}}`
in the order `initiator`, `platform`, `action`[, `campaign`]; UTF-8, lowercase
hex; closed enums (platform includes `agent`; actions include `signin`,
`mint`, `accept-offer`, `modify`, `harvest`, `equip`, `trustset`, `brix-claim`).

Testnet JSON-RPC defaults (no override on staging): `https://s.altnet.rippletest.net:51234/`,
fallback `https://testnet.xrpl-labs.com/`; WebSocket
`wss://s.altnet.rippletest.net:51233`; Clio `wss://clio.altnet.rippletest.net:51233`.
The fly submits through the same testnet endpoints so its hashes are visible
to the server's `tx` lookup.

Known doc drift in LFG's CLAUDE.md: the real routes are `POST /api/offers/accept`
and `POST /api/mint` (not `/api/pending-offers/accept`, `/api/mint/start`).
