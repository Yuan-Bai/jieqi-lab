const test = require("node:test");
const assert = require("node:assert/strict");
const prefs = require("./preferences.js");

test("persists independent play, analysis and display settings without any game data", () => {
  const data = new Map();
  const storage = { getItem: key => data.get(key) ?? null, setItem: (key, value) => data.set(key, value) };
  const value = { version: 1,
    play: { mode: "human", human_side: 0, engine: "az", model: "data/model.pt", simulations: 1234, leaf_batch_size: 17 },
    analysis: { engine: "az", model: "data/analysis.pt", simulations: 77, leaf_batch_size: 9, auto_hint: true, auto_evaluate: true },
    display: { analysis: false, record: true, coordinates: false },
    board: ["private"], belief: { private: true }, captured: ["private"] };
  assert.equal(prefs.write(storage, value), true);
  const restored = prefs.read(storage);
  assert.deepEqual(restored.play, value.play);
  assert.deepEqual(restored.analysis, value.analysis);
  assert.equal(restored.display.analysis, false);
  assert.equal(restored.display.coordinates, false);
  assert.equal(restored.display.legalMoves, true);
  assert.equal("captured" in restored.display, false); // Captures cannot be hidden.
  assert.doesNotMatch(data.get(prefs.KEY), /private|belief|board|captured/);
});

test("malformed or blocked storage leaves the application usable", () => {
  for (const serialized of ["{broken", "null", "[]", '{"version":99}', '{"version":1,"play":null}']) {
    const result = prefs.read({ getItem: () => serialized });
    assert.deepEqual(result.display, prefs.DISPLAY);
    assert.equal(result.play, null);
  }
  const blocked = { getItem: () => { throw new Error("blocked"); }, setItem: () => { throw new Error("blocked"); } };
  assert.equal(prefs.read(blocked).play, null);
  assert.equal(prefs.write(blocked, { version: 1 }), false);
});

test("invalid persisted types, bounds and unknown display fields cannot drive the UI", () => {
  const result = prefs.clean({ version: 1,
    play: { mode: "unknown", human_side: true, engine: "network", model: {}, simulations: "9999", leaf_batch_size: -2 },
    analysis: { engine: "ris", simulations: 100001, leaf_batch_size: 0, auto_evaluate: true, auto_hint: "yes" },
    display: { analysis: "false", coordinates: false, captured: false, arbitrary: false } });
  assert.equal(result.play.simulations, 64);
  assert.equal(result.play.leaf_batch_size, 64);
  assert.equal(result.play.human_side, 1);
  assert.equal(result.play.mode, "human");
  assert.equal(result.analysis.simulations, 200);
  assert.equal(result.analysis.auto_evaluate, false);
  assert.equal(result.analysis.auto_hint, false);
  assert.equal(result.display.analysis, true);
  assert.equal(result.display.coordinates, false);
  assert.equal("arbitrary" in result.display, false);
});

test("volume supports mute to full volume, remembers it, and migrates earlier preferences", () => {
  for (const volume of [0, 35, 80, 100]) {
    let data;
    const storage = { setItem: (_, value) => { data = value; }, getItem: () => data };
    prefs.write(storage, { version: 1, volume });
    assert.equal(prefs.read(storage).volume, volume);
  }
  for (const volume of [undefined, -1, 101, "80", null, NaN]) {
    assert.equal(prefs.clean({ version: 1, volume }).volume, 80);
  }
});
