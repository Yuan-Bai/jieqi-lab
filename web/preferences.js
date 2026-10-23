/* Browser preferences contain settings only; game records and private state stay out. */
(function (root) {
  "use strict";
  const KEY = "jieqi-preferences-v1";
  const DISPLAY = Object.freeze({ analysis: true, record: true, engineDetails: false,
                                  legalMoves: true, lastMove: true, coordinates: true });
  const object = value => value && typeof value === "object" && !Array.isArray(value);
  const integer = (value, fallback, max) => Number.isInteger(value) && value >= 1 && value <= max ? value : fallback;
  function search(value, defaults) {
    return { engine: ["ris", "az"].includes(value.engine) ? value.engine : defaults.engine,
             model: typeof value.model === "string" && value.model.length <= 1024 ? value.model : "",
             simulations: integer(value.simulations, defaults.simulations, 100000),
             leaf_batch_size: integer(value.leaf_batch_size, 64, 4096) };
  }
  function clean(value) {
    const result = { version: 1, play: null, analysis: null, volume: 80, display: { ...DISPLAY } };
    if (!object(value) || value.version !== 1) return result;
    if (Number.isInteger(value.volume) && value.volume >= 0 && value.volume <= 100) result.volume = value.volume;
    if (object(value.play)) {
      result.play = { ...search(value.play, { engine: "ris", simulations: 64 }),
        mode: ["human", "local", "aiai"].includes(value.play.mode) ? value.play.mode : "human",
        human_side: value.play.human_side === 0 ? 0 : 1 };
    }
    if (object(value.analysis)) {
      result.analysis = { ...search(value.analysis, { engine: "ris", simulations: 200 }),
        auto_hint: value.analysis.auto_hint === true,
        auto_evaluate: value.analysis.engine === "az" && value.analysis.auto_evaluate === true };
    }
    if (object(value.display)) for (const key of Object.keys(DISPLAY)) {
      if (typeof value.display[key] === "boolean") result.display[key] = value.display[key];
    }
    return result;
  }
  function read(storage) {
    try { return clean(JSON.parse(storage.getItem(KEY))); }
    catch (_) { return clean(null); }
  }
  function write(storage, value) {
    try { storage.setItem(KEY, JSON.stringify(clean(value))); return true; }
    catch (_) { return false; }
  }
  const api = Object.freeze({ KEY, DISPLAY, clean, read, write });
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.JieqiPreferences = api;
})(globalThis);
