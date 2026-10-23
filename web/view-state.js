/* Presentation state only: an old evaluation is never treated as current advice. */
(function (root) {
  "use strict";
  function playerStatus(state, side) {
    const color = side ? "红方" : "黑方";
    const active = state.side === side;
    let label = `${color}待行`;
    if (state.outcome !== null) label = state.outcome === 0 ? "和棋" : (state.outcome === 1) === (side === 1) ? `${color}获胜` : `${color}告负`;
    else if (state.reviewing) label = active ? `${color}行棋 · 复盘` : `${color}待行`;
    else if (active && !state.paused) label = state.pending?.kind === "move" ? `${color}思考中` : `${color}行棋`;
    return { active, label, thinking: active && !state.paused && !state.reviewing && state.pending?.kind === "move" };
  }
  function createEvaluationMemory() {
    let key = "", previous = null;
    const rate = value => Number.isFinite(value) && value >= 0 && value <= 1;
    return { update(state) {
      const cfg = state.analysis_settings;
      const nextKey = JSON.stringify([state.record_id, cfg.engine, cfg.model, cfg.simulations, cfg.leaf_batch_size]);
      if (key !== nextKey) { key = nextKey; previous = null; }
      if (state.outcome !== null) {
        previous = null;
        const red = state.outcome === 1 ? 1 : state.outcome === -1 ? 0 : .5;
        return { red, black: 1 - red, stale: false, terminal: true };
      }
      const result = state.analysis;
      const current = cfg.engine === "az" && result?.node === state.cursor && rate(result.red_win_rate) && rate(result.black_win_rate);
      if (current) previous = { red: result.red_win_rate, black: result.black_win_rate, depth: state.depth, node: state.cursor };
      return previous ? { ...previous, stale: !current } : { red: null, black: null, stale: false };
    } };
  }
  const api = Object.freeze({ playerStatus, createEvaluationMemory });
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.JieqiView = api;
})(globalThis);
