const test = require("node:test");
const assert = require("node:assert/strict");
const { createEvaluationMemory, playerStatus } = require("./view-state.js");

const position = (extra = {}) => ({ record_id: "game-a", cursor: 0, depth: 0, outcome: null,
  analysis_settings: { engine: "az", model: "model.pt", simulations: 20, leaf_batch_size: 4 },
  analysis: null, ...extra });
const result = (node, red) => ({ node, red_win_rate: red, black_win_rate: 1 - red });

test("keeps fixed red/black rates between moves, then replaces them with the current result", () => {
  const memory = createEvaluationMemory();
  assert.equal(memory.update(position({ analysis: result(0, .7) })).red, .7);
  const pending = position({ cursor: 1, depth: 1, analysis_pending: { node: 1 } });
  assert.deepEqual(memory.update(pending), { red: .7, black: .30000000000000004, depth: 0, node: 0, stale: true });
  assert.equal(pending.analysis, null); // Retaining the display never creates current advice.
  const next = memory.update({ ...pending, analysis: result(1, .35) });
  assert.equal(next.red, .35);
  assert.equal(next.black, .65);
  assert.equal(next.stale, false);
});

test("old evaluation remains explicitly stale on navigation or failure; mismatched results cannot replace it", () => {
  const memory = createEvaluationMemory();
  memory.update(position({ cursor: 8, depth: 6, analysis: result(8, .6) }));
  const failed = memory.update(position({ cursor: 2, depth: 2, analysis_error: "failed", analysis: result(99, .9) }));
  assert.equal(failed.red, .6);
  assert.equal(failed.depth, 6);
  assert.equal(failed.stale, true);
});

test("new records, changed analysis configuration and terminal results cannot leak previous percentages", () => {
  const base = position();
  for (const changed of [position({ record_id: "game-b" }),
    position({ analysis_settings: { ...base.analysis_settings, model: "other.pt" } }),
    position({ analysis_settings: { ...base.analysis_settings, simulations: 40 } }),
    position({ analysis_settings: { ...base.analysis_settings, engine: "ris" } })]) {
    const memory = createEvaluationMemory();
    memory.update(position({ analysis: result(0, .7) }));
    assert.equal(memory.update(changed).red, null);
  }
  const memory = createEvaluationMemory();
  memory.update(position({ analysis: result(0, .7) }));
  assert.equal(memory.update(position({ outcome: -1 })).red, 0);
  assert.equal(memory.update(position({ cursor: 0, reviewing: true })).red, null);
});

test("thinking and waiting labels always name the actual side; pausing never blames one player", () => {
  const state = { side: 0, outcome: null, reviewing: false, paused: false, pending: { kind: "move" } };
  assert.deepEqual(playerStatus(state, 0), { active: true, thinking: true, label: "黑方思考中" });
  assert.deepEqual(playerStatus(state, 1), { active: false, thinking: false, label: "红方待行" });
  assert.equal(playerStatus({ ...state, paused: true }, 0).label, "黑方待行");
  assert.equal(playerStatus({ ...state, paused: true }, 0).thinking, false);
  assert.equal(playerStatus({ ...state, side: 1 }, 1).label, "红方思考中");
});
