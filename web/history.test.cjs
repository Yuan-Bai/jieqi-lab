const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

function historyView() {
  // Run the application's actual renderer; only its DOM boundary is substituted.
  const source = fs.readFileSync(path.join(__dirname, "app.js"), "utf8");
  const renderer = source.slice(source.indexOf("function renderHistory()"), source.indexOf("function renderPosition()"));
  const nodes = new Map();
  const list = { writes: 0, html: "", querySelectorAll: () => [], querySelector: () => null,
    get innerHTML() { return this.html; },
    set innerHTML(value) { this.html = value; this.writes++; } };
  nodes.set("#move-list", list);
  const context = vm.createContext({
    $: selector => {
      if (!nodes.has(selector)) nodes.set(selector, {});
      return nodes.get(selector);
    },
    escapeText: String,
    act: () => {},
  });
  vm.runInContext(`let lastHistoryKey = "";\n${renderer}`, context);
  return { context, list };
}

function record(label) {
  return { cursor: 1, depth: 1, reviewing: true, outcome: null,
    history: [{ id: 1, parent: 0, depth: 1, side: 1, label, from: 58, to: 49 }] };
}

test("replaces history when another record has the same cursor and node count", () => {
  const { context, list } = historyView();
  context.game = record("暗兵 g3 → g4 · 兵");
  context.renderHistory();
  assert.match(list.innerHTML, /暗兵/);
  context.game = record("暗马 b0 → c2 · 马");
  context.renderHistory();
  assert.match(list.innerHTML, /暗马/);
  assert.doesNotMatch(list.innerHTML, /暗兵/);
});

test("unchanged polled history keeps its DOM while navigation refreshes selection", () => {
  const { context, list } = historyView();
  context.game = record("兵 g3 → g4");
  context.renderHistory();
  context.game = record("兵 g3 → g4");
  context.renderHistory();
  assert.equal(list.writes, 1);
  assert.match(list.innerHTML, /aria-current="step"/);
  context.game.cursor = 0;
  context.game.depth = 0;
  context.renderHistory();
  assert.equal(list.writes, 2);
  assert.doesNotMatch(list.innerHTML, /aria-current="step"/);
});
