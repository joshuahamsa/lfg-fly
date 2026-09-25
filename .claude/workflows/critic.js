export const meta = {
  name: 'critic',
  description: "The fly's critic (spec §3.2): one agent per batch of rendered look pairs; forced-choice verdicts with strength and idea names",
  whenToUse: 'After `fly critic render --round rN`; pass the batch list as args (see README in docs/plans/2026-09-25-fly-v1-build.md)',
  phases: [
    { title: 'Judge', detail: 'one agent per batch of cards' },
    { title: 'Retry', detail: 'batches whose verdicts did not cover every item' },
  ],
}

// args: { round: 'r1', dir: '<round directory>', batches: [[id, n], ...] }
//   <dir>/batches/<id>.json is the batch ({items: [{item_id, image}]});
//   <dir>/results/<id>.json is where the agent writes its verdicts.

const RUBRIC = `The rubric (this is the fly's taste; judge only what you see):
1. Appeal: colour harmony and contrast, a readable silhouette, nothing muddy or cluttered, and a background that supports the character.
2. Idea: the combination reads as a character concept you could name ("Retired Space Pirate").
3. Surprise: an unexpected combination that works beats a predictable matched set. Predictable matched sets are fine but not exciting; a clash that doesn't work loses.

Forced choice: every pair has a winner, "A" or "B", never a tie. Strength: 1 = a close call, 2 = clearly better, 3 = no contest.
idea_A and idea_B: at most 5 words naming the concept each look reads as (or "no clear idea"). reason: at most 15 words on why the winner wins.
You are told nothing about rarity, price, or which look anyone chose, and you must not guess at it. Do not favour the left or the right side: some pairs are shown again with the sides swapped, precisely to measure that. Near pairs differ in only one or two slots; the difference is the point, so look at both sides carefully.`

const VERDICT = {
  type: 'object',
  properties: {
    item_id: { type: 'string' },
    winner: { type: 'string', enum: ['A', 'B'] },
    strength: { type: 'integer', minimum: 1, maximum: 3 },
    idea_A: { type: 'string' },
    idea_B: { type: 'string' },
    reason: { type: 'string' },
  },
  required: ['item_id', 'winner', 'strength', 'idea_A', 'idea_B', 'reason'],
}

function schemaFor(n) {
  return {
    type: 'object',
    properties: {
      verdicts: { type: 'array', items: VERDICT, minItems: n, maxItems: n },
      written: { type: 'boolean' },
    },
    required: ['verdicts', 'written'],
  }
}

function prompt(b, note) {
  return `You are the critic that teaches the fly its taste (lfg-fly, spec §3.2). You judge ${b.n} pairs of LFG character looks. Each pair is one PNG card, 1024x512: look A on the left, look B on the right, labelled.

Steps, in this order:
1. Read the batch file ${b.file}. It is JSON: {"items": [{"item_id": ..., "image": <absolute path>}]}.
2. For EVERY item, Read its image with the Read tool and judge it right after seeing it. Read at most 4 images per turn. Never skip an image, never judge from a file name (names carry no information about the look), never extrapolate from earlier pairs: each verdict must come from looking at that card.
3. Write ${b.out} as JSON: {"verdicts": [ {item_id, winner, strength, idea_A, idea_B, reason}, ... ]} with exactly one entry per item, in the batch's order. Use the Write tool.
4. Return the same verdicts as your structured output, with "written": true.

${RUBRIC}
${note ? `\nNOTE: ${note}\n` : ''}`
}

function check(res, b) {
  if (!res || !Array.isArray(res.verdicts)) return 'no verdicts'
  if (res.verdicts.length !== b.n) return `expected ${b.n} verdicts, got ${res.verdicts.length}`
  const ids = new Set(res.verdicts.map(v => v.item_id))
  if (ids.size !== b.n) return 'duplicate item_id'
  if (!res.written) return 'results file not written'
  return null
}

const BATCHES = args.batches.map(([id, n]) => ({
  id, n, file: `${args.dir}/batches/${id}.json`, out: `${args.dir}/results/${id}.json`,
}))
phase('Judge')
log(`${BATCHES.length} batches for round ${args.round}`)
const results = await pipeline(
  BATCHES,
  b => agent(prompt(b), { label: `judge:${b.id}`, phase: 'Judge', schema: schemaFor(b.n) })
    .then(res => ({ b, problem: check(res, b) })),
  async r => {
    if (!r.problem) return { id: r.b.id, ok: true }
    log(`${r.b.id}: ${r.problem}; retrying`)
    const res = await agent(prompt(r.b, `A previous attempt failed: ${r.problem}. Cover every item exactly once and write the file.`),
      { label: `retry:${r.b.id}`, phase: 'Retry', schema: schemaFor(r.b.n) })
    const problem = check(res, r.b)
    return { id: r.b.id, ok: !problem, problem }
  },
)
const done = results.filter(Boolean)
const ok = done.filter(r => r.ok).map(r => r.id)
const failed = done.filter(r => !r.ok).map(r => `${r.id}: ${r.problem}`)
const skipped = BATCHES.length - done.length
log(`${ok.length} batches judged, ${failed.length} failed, ${skipped} skipped`)
return { round: args.round, ok: ok.length, failed, skipped }
