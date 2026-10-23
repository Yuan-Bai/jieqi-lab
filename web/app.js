"use strict";

const $ = (selector) => document.querySelector(selector);
const icons = {
  board:'<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 3v18M15 3v18M3 9h18M3 15h18"/>',
  book:'<path d="M12 6c-3-2-6-2-9-1v15c3-1 6-1 9 1 3-2 6-2 9-1V5c-3-1-6-1-9 1Zm0 0v15"/>',
  compass:'<circle cx="12" cy="12" r="9"/><path d="m16 8-2 6-6 2 2-6 6-2Z"/>',
  volume:'<path d="M11 4 6 8H3v8h3l5 4V4Zm4 4a6 6 0 0 1 0 8m3-11a10 10 0 0 1 0 14"/>',
  mute:'<path d="M11 4 6 8H3v8h3l5 4V4Zm5 5 6 6m0-6-6 6"/>',
  sparkles:'<path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Zm7-2v4m-2-2h4"/>',
  users:'<circle cx="9" cy="8" r="3"/><path d="M3 21v-3a6 6 0 0 1 12 0v3m0-16a3 3 0 0 1 0 6m3 3a5 5 0 0 1 3 4v3"/>',
  eye:'<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/>',
  plus:'<path d="M12 5v14M5 12h14"/>',
  'arrow-right':'<path d="M4 12h16m-6-6 6 6-6 6"/>',
  'arrow-up-right':'<path d="M6 18 18 6M6 6h12v12"/>',
  shield:'<path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3Zm-4 9 3 3 5-6"/>',
  undo:'<path d="M4 10h10a6 6 0 0 1 0 12M4 10l5-5m-5 5 5 5"/>',
  bulb:'<path d="M9 18h6m-5 3h4M8 14a6 6 0 1 1 8 0l-1 2H9l-1-2ZM12 1v1M2 10h1m18 0h1M4 3l2 2m12 0 2-2"/>',
  flip:'<path d="M3 8h16l-4-4m4 4-4 4M21 16H5l4 4m-4-4 4-4"/>',
  pause:'<path d="M8 5v14M16 5v14"/>',
  play:'<path d="m8 4 12 8-12 8V4Z"/>',
  flag:'<path d="M4 22V3c5-4 10 4 16 0v11c-6 4-11-4-16 0"/>',
  download:'<path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5"/>',
  first:'<path d="M5 5v14m13-14-8 7 8 7"/>',
  last:'<path d="M19 5v14M6 5l8 7-8 7"/>',
  'chevron-left':'<path d="m15 5-7 7 7 7"/>',
  'chevron-right':'<path d="m9 5 7 7-7 7"/>',
  layers:'<path d="m12 3 10 5-10 5L2 8l10-5ZM2 12l10 5 10-5M2 16l10 5 10-5"/>',
  lock:'<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3m-4 5v2"/>',
  cpu:'<rect x="5" y="5" width="14" height="14" rx="2"/><rect x="9" y="9" width="6" height="6" rx="1"/><path d="M9 2v3m6-3v3M9 19v3m6-3v3M2 9h3m-3 6h3m14-6h3m-3 6h3"/>',
  close:'<path d="m6 6 12 12M6 18 18 6"/>',
  branch:'<circle cx="6" cy="5" r="2"/><circle cx="18" cy="5" r="2"/><circle cx="6" cy="19" r="2"/><path d="M6 7v10m0-4c8 0 12-1 12-6"/>',
  settings:'<path d="m9 3 1-2h4l1 2 2 1 2-.2 2 3-1 2v3l1 2-2 3-2-.2-2 1-1 2h-4l-1-2-2-1-2 .2-2-3 1-2V9L3 7l2-3L7 4l2-1Z"/><circle cx="12" cy="10" r="3"/>',
  sliders:'<path d="M4 7h16M4 17h16"/><circle cx="8" cy="7" r="3" fill="currentColor" stroke="none"/><circle cx="16" cy="17" r="3" fill="currentColor" stroke="none"/>',
  check:'<path d="m5 12 4 4L19 6"/>',
};
function icon(name) { return `<svg viewBox="0 0 24 24" aria-hidden="true">${icons[name] || icons.board}</svg>`; }
function paintIcons(root = document) { root.querySelectorAll("i[data-icon]").forEach((node) => { node.innerHTML = icon(node.dataset.icon); }); }
paintIcons();

let game = null;
let selected = null;
let flipped = false;
let acting = false;
let online = false;
let initializing = true;
let restoringPreferences = false;
let availableModels = null;
let pollTimer;
let pollRequest = null;
let lastBoardKey = "";
let lastHistoryKey = "";
let lastCaptureKey = "";
let toastTimer;
let audioContext;
let masterGain;
let soundEnabled = true;
let browserStorage;
try { browserStorage = window.localStorage; } catch (_) { /* Storage can be restricted. */ }
let preferences = JieqiPreferences.read(browserStorage);
const evaluationMemory = JieqiView.createEvaluationMemory();
try { soundEnabled = localStorage.getItem("jieqi-sound") !== "off"; } catch (_) { /* Storage may be disabled. */ }
const names = { 0: ["", "将", "车", "马", "象", "士", "炮", "卒"], 1: ["", "帅", "车", "马", "相", "仕", "炮", "兵"] };
const sideOf = (piece) => (piece >> 3) & 1;
const darkOf = (piece) => Boolean(piece & 16);
const typeOf = (piece) => piece & 7;
const escapeText = (text) => String(text).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
const squareName = (square) => String.fromCharCode(97 + square % 9) + (9 - Math.floor(square / 9));

function toast(message) {
  clearTimeout(toastTimer);
  $("#toast").textContent = message;
  $("#toast").hidden = false;
  toastTimer = setTimeout(() => { $("#toast").hidden = true; }, 3600);
}

function savePreferences() {
  const saved = JieqiPreferences.write(browserStorage, preferences);
  $("#preferences-status").innerHTML = saved ? `<i>${icon("check")}</i>设置已自动保存` : "浏览器不允许保存，设置仅本次有效";
  return saved;
}
function rememberAppliedSettings(action) {
  if (initializing || restoringPreferences) return;
  if (action === "new" || action === "settings") {
    preferences.play = { mode: game.mode, human_side: game.human_side, engine: game.engine,
      model: game.model, simulations: game.simulations, leaf_batch_size: game.leaf_batch_size };
  }
  if (action === "analysis-settings") preferences.analysis = { ...game.analysis_settings };
  if (["new", "settings", "analysis-settings"].includes(action)) savePreferences();
}
function applyDisplayPreferences() {
  const display = preferences.display;
  $("#analysis-panel").hidden = !display.analysis;
  $("#record-panel").hidden = !display.record;
  $("#engine-status").hidden = !display.engineDetails && !game?.error;
  document.body.classList.toggle("hide-coordinates", !display.coordinates);
  $(".game-layout").classList.toggle("record-hidden", !display.record);
  document.querySelectorAll("[data-view]").forEach(input => { input.checked = display[input.dataset.view]; });
  lastBoardKey = "";
  renderBoard();
}
function fillPlaySettings() {
  document.querySelector(`input[name="mode"][value="${game.mode}"]`).checked = true;
  document.querySelector(`input[name="side"][value="${game.human_side}"]`).checked = true;
  $("#engine-select").value = game.engine;
  $("#model-select").value = game.model || availableModels?.[0] || "";
  $("#simulations-input").value = game.simulations;
  $("#leaf-batch-input").value = game.leaf_batch_size;
  updateSettings();
}
async function restoreSavedPreferences(state) {
  if (state.preferences_ready !== false || restoringPreferences || !(preferences.play || preferences.analysis)) return;
  restoringPreferences = true;
  updateControls();
  try {
    const compatible = settings => {
      const result = { ...settings };
      if (result.engine === "az" && availableModels && !availableModels.includes(result.model)) {
        result.engine = "ris";
        result.model = "";
        if ("auto_evaluate" in result) result.auto_evaluate = false;
        toast("已记住的模型暂不可用，使用 RIS；其他参数已恢复");
      }
      return result;
    };
    if (preferences.play) await act("new", compatible(preferences.play));
    if (preferences.analysis) await act("analysis-settings", compatible(preferences.analysis));
    fillPlaySettings();
    fillAnalysisSettings();
  } finally {
    restoringPreferences = false;
    updateControls();
  }
}

// Short, original Web Audio sounds: a damped wooden knock and soft chimes.
// They are generated locally and need no downloads or third-party assets.
function unlockAudio() {
  try {
    if (!audioContext) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      masterGain = audioContext.createGain();
      masterGain.connect(audioContext.destination);
      updateAudioGain();
    }
    if (audioContext.state === "suspended") audioContext.resume().catch(() => {});
  } catch (_) { /* Audio is optional; game controls always remain usable. */ }
}
document.addEventListener("pointerdown", unlockAudio, { passive: true });
document.addEventListener("keydown", unlockAudio, { passive: true });
function tone(freq, start, duration, volume, type = "sine") {
  const oscillator = audioContext.createOscillator();
  const gain = audioContext.createGain();
  oscillator.type = type;
  oscillator.frequency.setValueAtTime(freq, start);
  gain.gain.setValueAtTime(0, start);
  gain.gain.linearRampToValueAtTime(volume, start + 0.004);
  gain.gain.exponentialRampToValueAtTime(0.0001, start + duration);
  oscillator.connect(gain).connect(masterGain);
  oscillator.start(start);
  oscillator.stop(start + duration + 0.02);
}
function sound(kind) {
  if (!soundEnabled || !preferences.volume || !audioContext || audioContext.state !== "running") return;
  const now = audioContext.currentTime;
  if (["win", "lose", "draw", "check"].includes(kind)) {
    const notes = { win: [523.25, 659.25, 783.99, 1046.5], lose: [392, 349.23, 293.66], draw: [392, 523.25, 392], check: [440, 587.33] }[kind];
    notes.forEach((freq, i) => tone(freq, now + i * 0.12, 0.45, 0.075));
    return;
  }
  const capture = kind === "capture";
  const duration = kind === "select" ? 0.045 : capture ? 0.16 : 0.1;
  tone(kind === "select" ? 650 : capture ? 180 : 280, now, duration, kind === "select" ? 0.045 : 0.13);
  tone(capture ? 530 : 850, now, duration * 0.5, 0.035);
  if (kind !== "select") {
    const buffer = audioContext.createBuffer(1, Math.ceil(audioContext.sampleRate * 0.045), audioContext.sampleRate);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < channel.length; i++) channel[i] = (Math.random() * 2 - 1) * Math.exp(-i / (channel.length * 0.12));
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    const filter = audioContext.createBiquadFilter();
    filter.type = "lowpass";
    filter.frequency.value = capture ? 1700 : 2300;
    const gain = audioContext.createGain();
    gain.gain.value = capture ? 0.18 : 0.1;
    source.connect(filter).connect(gain).connect(masterGain);
    source.start(now);
  }
}
function updateAudioGain() {
  if (masterGain) masterGain.gain.setTargetAtTime(soundEnabled ? preferences.volume / 100 * 2.8 : 0, audioContext.currentTime, .02);
}
function updateSoundButton() {
  const audible = soundEnabled && preferences.volume > 0;
  const button = $("#sound-button");
  button.innerHTML = `<i>${icon(audible ? "volume" : "mute")}</i>`;
  button.title = button.ariaLabel = `音效与音量 · ${audible ? preferences.volume + "%" : "已静音"}`;
  $("#volume-input").value = preferences.volume;
  $("#volume-value").textContent = `${preferences.volume}%`;
  $("#sound-toggle").textContent = audible ? "静音" : "开启音效";
  $("#sound-toggle").setAttribute("aria-pressed", String(audible));
  updateAudioGain();
}
updateSoundButton();

async function jsonRequest(url, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 20000);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal, cache: "no-store" });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "请求失败，请稍后重试");
    return result;
  } finally { clearTimeout(timeout); }
}
function setConnection(value) {
  online = value;
  $("#connection").classList.toggle("offline", !value);
  $("#connection").innerHTML = `<b></b><span>${value ? "本地引擎在线" : "连接已断开"}</span>`;
}
function schedulePoll() {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(poll, game?.pending || game?.analysis_pending ? 450 : game?.mode === "aiai" && !game.paused ? 650 : 3500);
}
async function poll() {
  if (initializing || acting || document.hidden) { schedulePoll(); return; }
  if (pollRequest) return;
  pollRequest = jsonRequest("/api/state");
  try {
    const next = await pollRequest;
    setConnection(true);
    acceptState(next);
    await restoreSavedPreferences(next);
  } catch (_) {
    setConnection(false);
    $("#board-message").innerHTML = '<span class="status-dot"></span>连接中断，正在重试。请确认本地服务仍在运行。';
    $("#board-message").classList.add("is-error");
    updateControls();
  } finally {
    pollRequest = null;
    schedulePoll();
  }
}
async function act(action, data = {}) {
  if (acting) return;
  acting = true;
  clearTimeout(pollTimer);
  updateControls();
  try {
    if (pollRequest) await pollRequest.catch(() => {});
    const next = await jsonRequest(`/api/${action}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...data, revision: game?.revision }) });
    setConnection(true);
    selected = null;
    if (action === "new") {
      flipped = next.mode === "human" && next.human_side === 0;
      drawBoardLines();
    }
    acceptState(next);
    rememberAppliedSettings(action);
    return true;
  } catch (error) {
    toast(error.name === "AbortError" ? "连接超时，请确认本地服务仍在运行" : error.message);
    // Reconcile stale revisions or a response lost after a successful write.
    try { acceptState(await jsonRequest("/api/state")); } catch (_) { setConnection(false); }
    return false;
  } finally {
    acting = false;
    updateControls();
    schedulePoll();
  }
}
function acceptState(next) {
  const old = game;
  game = next;
  if (!old) {
    flipped = game.mode === "human" && game.human_side === 0;
    drawBoardLines();
    fillPlaySettings();
    fillAnalysisSettings();
    updateSettings();
  }
  const moved = old && game.cursor !== old.cursor && !game.reviewing && game.depth > old.depth;
  if (old && (old.cursor !== next.cursor || old.board.join(",") !== next.board.join(","))) selected = null;
  if (moved) {
    if (game.outcome !== null) sound(game.outcome === 0 ? "draw" : game.mode === "human" && ((game.outcome === 1) !== (game.human_side === 1)) ? "lose" : "win");
    else if (game.in_check) sound("check");
    else sound(old.board[game.last_move[1]] ? "capture" : "move");
  }
  if (old && old.outcome === null && game.outcome !== null && !game.reviewing && !moved) sound("lose");
  renderBoard(moved);
  renderPlayers();
  renderHistory();
  renderPosition();
  renderCaptured();
  renderAnalysis();
  updateControls();
  renderStatus();
  if (game.error && game.error !== old?.error) toast(game.error);
  if (game.analysis_error && game.analysis_error !== old?.analysis_error) toast(game.analysis_error);
}

function drawBoardLines() {
  const lines = [];
  for (let row = 0; row < 10; row++) lines.push(`<path d="M50 ${row * 100 + 50}H850"/>`);
  for (let col = 0; col < 9; col++) {
    const x = col * 100 + 50;
    lines.push(col === 0 || col === 8 ? `<path d="M${x} 50V950"/>` : `<path d="M${x} 50V450M${x} 550V950"/>`);
  }
  lines.push('<path d="M350 50 550 250M550 50 350 250M350 750 550 950M550 750 350 950"/>');
  const marks = [[2, 1], [2, 7], [7, 1], [7, 7], ...[0, 2, 4, 6, 8].flatMap(col => [[3, col], [6, col]])];
  for (const [row, col] of marks) {
    const x = 50 + col * 100, y = 50 + row * 100;
    for (const dx of [-1, 1]) {
      if ((col === 0 && dx === -1) || (col === 8 && dx === 1)) continue;
      for (const dy of [-1, 1]) lines.push(`<path d="M${x + dx * 23} ${y + dy * 8}H${x + dx * 8}V${y + dy * 23}"/>`);
    }
  }
  const labels = [];
  for (let i = 0; i < 9; i++) labels.push(`<text x="${50 + 100 * i}" y="23">${flipped ? String.fromCharCode(105 - i) : String.fromCharCode(97 + i)}</text><text x="${50 + 100 * i}" y="989">${flipped ? String.fromCharCode(105 - i) : String.fromCharCode(97 + i)}</text>`);
  for (let i = 0; i < 10; i++) labels.push(`<text x="15" y="${56 + 100 * i}">${flipped ? i : 9 - i}</text><text x="885" y="${56 + 100 * i}">${flipped ? i : 9 - i}</text>`);
  $("#board-lines").innerHTML = `<g fill="none" stroke="currentColor" stroke-width="1.8" opacity=".69">${lines.join("")}<rect x="42" y="42" width="816" height="916" stroke-width="1"/></g><g class="coordinate-labels" fill="currentColor" opacity=".64" text-anchor="middle" font-size="15" font-family="Georgia,serif">${labels.join("")}</g>`;
}
drawBoardLines();
const squares = [];
for (let viewIndex = 0; viewIndex < 90; viewIndex++) {
  const button = document.createElement("button");
  button.className = "square";
  button.type = "button";
  button.addEventListener("click", () => clickSquare(flipped ? 89 - viewIndex : viewIndex));
  $("#board-cells").append(button);
  squares.push(button);
}
function canMove() { return game && online && !initializing && !restoringPreferences && !acting && !game.pending && !game.reviewing && !game.paused && !game.ai_turn && game.outcome === null; }
function renderBoard(animate = false) {
  if (!game) return;
  const key = JSON.stringify([game.board, game.last_move, game.hint, game.in_check, selected, flipped, game.reviewing, game.ai_turn, game.paused, !!game.pending, preferences.display]);
  if (key === lastBoardKey && !animate) return;
  lastBoardKey = key;
  const legal = game.legal_moves.filter(move => move[0] === selected).map(move => move[1]);
  for (let v = 0; v < 90; v++) {
    const i = flipped ? 89 - v : v;
    const piece = game.board[i];
    const cell = squares[v];
    const classes = ["square"];
    if (piece && sideOf(piece) === game.side && !game.ai_turn && !game.reviewing) classes.push("selectable");
    if (i === selected) classes.push("selected");
    if (preferences.display.legalMoves && legal.includes(i)) classes.push("target", ...(piece ? ["capture-target"] : []));
    if (preferences.display.lastMove && game.last_move?.[0] === i) classes.push("last-from");
    if (preferences.display.lastMove && game.last_move?.[1] === i) classes.push("last-to");
    if (game.hint?.[0] === i) classes.push("hinted");
    if (game.hint?.[1] === i) classes.push("hint-target");
    if (piece && game.in_check && typeOf(piece) === 1 && sideOf(piece) === game.side) classes.push("check-square");
    cell.className = classes.join(" ");
    const description = piece ? `${sideOf(piece) ? "红" : "黑"}方${darkOf(piece) ? "暗" : ""}${names[sideOf(piece)][typeOf(piece)]}` : "空位";
    cell.ariaLabel = `${squareName(i)} ${description}${legal.includes(i) ? "，可落子" : ""}${game.hint?.[0] === i ? "，AI 推荐起点" : ""}${game.hint?.[1] === i ? "，AI 推荐落点" : ""}`;
    cell.setAttribute("aria-pressed", String(selected === i));
    cell.dataset.square = String(i);
    const pieceKey = `${piece}-${animate && game.last_move?.[1] === i}`;
    if (cell.dataset.pieceKey !== pieceKey) {
      cell.dataset.pieceKey = pieceKey;
      cell.innerHTML = piece ? `<span class="piece ${sideOf(piece) ? "red-piece" : "black-piece"} ${darkOf(piece) ? "dark-piece" : "bright-piece"} ${animate && game.last_move?.[1] === i ? "landed" : ""}" aria-hidden="true">${names[sideOf(piece)][typeOf(piece)]}${darkOf(piece) ? '<span class="dark-mark">暗</span>' : ""}</span>` : "";
    }
  }
  drawHintArrow();
}
function drawHintArrow() {
  if (!game?.hint) { $("#hint-arrow").innerHTML = ""; return; }
  const points = game.hint.map(square => {
    const view = flipped ? 89 - square : square;
    return { x: 50 + view % 9 * 100, y: 50 + Math.floor(view / 9) * 100 };
  });
  const [from, to] = points;
  const dx = to.x - from.x, dy = to.y - from.y, length = Math.hypot(dx, dy);
  const x1 = from.x + dx / length * 40, y1 = from.y + dy / length * 40;
  const x2 = to.x - dx / length * 28, y2 = to.y - dy / length * 28;
  $("#hint-arrow").innerHTML = `<defs><marker id="ai-arrowhead" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse"><path d="M0 1 9 5 0 9Z" fill="#2676ac"/></marker></defs><path d="M${x1} ${y1}L${x2} ${y2}" stroke="#e2f4ff" stroke-width="11" opacity=".7"/><path d="M${x1} ${y1}L${x2} ${y2}" stroke="#2676ac" stroke-width="6" stroke-linecap="round" marker-end="url(#ai-arrowhead)" opacity=".95"/>`;
}
async function clickSquare(square) {
  if (!canMove()) {
    if (game?.reviewing) toast("正在复盘。选择「从此继续」可从当前局面落子。");
    return;
  }
  if (selected !== null && game.legal_moves.some(move => move[0] === selected && move[1] === square)) {
    await act("move", { from: selected, to: square });
    return;
  }
  const piece = game.board[square];
  if (piece && sideOf(piece) === game.side) {
    selected = selected === square ? null : square;
    sound("select");
  } else {
    if (selected !== null && square !== selected) toast("这个位置不能落子，请选择绿色标记的位置。");
    selected = null;
  }
  renderBoard();
}

function playerName(side) {
  if (game.mode === "local") return side ? "红方棋手" : "黑方棋手";
  if (game.mode === "human" && side === game.human_side) return "我方棋手";
  return game.engine === "az" ? "AlphaZero" : "RIS-ISMCTS";
}
function playerCard(side) {
  const { active, label, thinking } = JieqiView.playerStatus(game, side);
  const ai = game.mode === "aiai" || game.mode === "human" && side !== game.human_side;
  const darkCount = game.dark_remaining[side];
  return `<div class="player-avatar ${side ? "red-avatar" : ""}">${side ? "帅" : "将"}</div><div class="player-info"><strong>${playerName(side)}<span>${ai ? "AI" : game.mode === "human" ? "YOU" : "PLAYER"}</span></strong><small>${side ? "红方 · 先行" : "黑方 · 后行"}<span> &nbsp; / &nbsp; 暗子 ${darkCount}</span></small></div><span class="turn-badge ${!active ? "idle" : ""} ${thinking ? "thinking" : ""}">${active && game.outcome === null ? "<b></b>" : ""}${label}</span>`;
}
function renderPlayers() {
  for (const [id, side] of [["#top-player", flipped ? 1 : 0], ["#bottom-player", flipped ? 0 : 1]]) {
    $(id).innerHTML = playerCard(side);
    $(id).classList.toggle("is-active", game.side === side);
    $(id).classList.toggle("red-player", side === 1);
    $(id).setAttribute("aria-label", `${side ? "红方" : "黑方"}棋手，${JieqiView.playerStatus(game, side).label}`);
  }
  $("#match-mode-label").textContent = { human: "人机对弈", local: "好友切磋", aiai: "观棋研习" }[game.mode];
  $("#match-side-label").textContent = game.reviewing ? "棋谱复盘" : game.paused ? `对局已暂停 · 待${game.side ? "红" : "黑"}方` : game.mode === "human" ? `我执${game.human_side ? "红" : "黑"}` : `第 ${game.depth} 手`;
}
function renderHistory() {
  $("#move-count").textContent = game.history.length;
  $("#step-label").textContent = `第 ${game.depth} 手`;
  $("#record-state").textContent = game.reviewing ? "复盘模式" : game.outcome !== null ? "对局结束" : "实时记录";
  const key = JSON.stringify([game.cursor, game.history]);
  if (key === lastHistoryKey) return;
  lastHistoryKey = key;
  const list = $("#move-list");
  if (!game.history.length) {
    list.innerHTML = '<div class="empty-record"><h3>等待第一着</h3><p>落子后，点击棋谱即可回看。</p></div>';
    return;
  }
  const childCounts = {};
  game.history.forEach(node => { childCounts[node.parent] = (childCounts[node.parent] || 0) + 1; });
  list.innerHTML = game.history.map(node => `<button class="move-item ${node.id === game.cursor ? "current" : ""}" data-node="${node.id}" ${node.id === game.cursor ? 'aria-current="step"' : ""} aria-label="第 ${node.depth} 手，${node.side ? "红" : "黑"}方，${escapeText(node.label)}"><span class="move-number">${String(node.depth).padStart(2, "0")}</span><b class="side-dot ${node.side ? "red" : "black"}"></b><span class="move-text">${escapeText(node.label)}</span>${childCounts[node.parent] > 1 ? `<span class="branch-note" title="父节点 ${node.parent} 的变例">支 ${node.id}</span>` : ""}</button>`).join("");
  list.querySelectorAll("[data-node]").forEach(button => button.addEventListener("click", () => act("navigate", { cursor: Number(button.dataset.node) })));
  const current = list.querySelector(".current");
  if (current) list.scrollTop = current.offsetTop - list.offsetTop - list.clientHeight / 2 + current.clientHeight / 2;
}
function renderPosition() {
  const counts = [0, 0], dark = [0, 0];
  game.board.forEach(piece => { if (piece) { counts[sideOf(piece)]++; if (darkOf(piece)) dark[sideOf(piece)]++; } });
  $("#position-stats").innerHTML = [1, 0].map(side => `<div class="position-stat"><b class="side-dot ${side ? "red" : "black"}"></b><span>${side ? "红方" : "黑方"}存子</span><b>${counts[side]} <small>枚</small></b></div><div class="position-stat"><span>${side ? "红方" : "黑方"}明子 / 暗子</span><b>${counts[side] - dark[side]} / ${dark[side]}</b></div>`).join("");
  $("#draw-count").textContent = `${game.no_capture_plies} / 80`;
  $("#draw-progress").value = game.no_capture_plies;
}
function renderCaptured() {
  const key = JSON.stringify([game.record_id, game.captured, game.depth]);
  if (key === lastCaptureKey) return;
  lastCaptureKey = key;
  let total = 0;
  const events = [1, 0].flatMap(side => game.captured[side].map(piece => ({ ...piece, actor: side })));
  const latest = events.filter(piece => Number.isInteger(piece.depth)).sort((a, b) => b.depth - a.depth)[0];
  const identity = piece => `${piece.side ? "红" : "黑"}方${piece.type === null ? "暗子（身份未知）" : `${piece.dark ? "暗" : ""}${names[piece.side][piece.type]}`}`;
  for (const side of [1, 0]) {
    const items = game.captured[side];
    total += items.length;
    $(side ? "#red-capture-count" : "#black-capture-count").innerHTML = `${items.length} <small>枚</small>`;
    $(side ? "#red-captures" : "#black-captures").innerHTML = items.length ? items.map(piece => {
      const name = piece.type === null ? "暗" : names[piece.side][piece.type];
      const knownMove = Number.isInteger(piece.node);
      const description = `${knownMove ? `第 ${piece.depth} 手，${side ? "红" : "黑"}方吃掉${identity(piece)}，${squareName(piece.source)} → ${squareName(piece.target)}；点击回看` : `导入局面已有：${identity(piece)}`}`;
      const tag = knownMove ? "button" : "span";
      return `<${tag} class="capture-item ${knownMove && piece.node === latest?.node ? "newest" : ""}" ${knownMove ? `data-capture-node="${piece.node}" type="button"` : ""} title="${description}" aria-label="${description}"><span class="captured-token ${piece.side ? "red-token" : ""} ${piece.type === null ? "unknown-token" : ""}">${name}</span><small>${knownMove ? `${piece.depth}手` : "已有"}</small></${tag}>`;
    }).join("") : '<span class="capture-empty">尚未吃子</span>';
  }
  document.querySelectorAll("[data-capture-node]").forEach(button => {
    button.disabled = acting || !online;
    button.addEventListener("click", () => act("navigate", { cursor: Number(button.dataset.captureNode) }));
  });
  $("#latest-capture").hidden = !latest;
  const detail = latest ? `<span>${latest.depth === game.depth ? "本手吃子" : "最近吃子"} · 第 ${latest.depth} 手 · ${latest.actor ? "红" : "黑"}方吃掉</span><strong>${identity(latest)}</strong>` : "";
  if ($("#latest-capture").innerHTML !== detail) $("#latest-capture").innerHTML = detail;
  $("#capture-total").textContent = `共 ${total} 枚`;
}
function fillAnalysisSettings() {
  const cfg = game.analysis_settings;
  $("#analysis-engine-select").value = cfg.engine;
  $("#analysis-model-select").value = cfg.model;
  $("#analysis-simulations-input").value = cfg.simulations;
  $("#analysis-leaf-input").value = cfg.leaf_batch_size;
  $("#auto-hint").checked = cfg.auto_hint;
  $("#auto-evaluate").checked = cfg.auto_evaluate;
  updateAnalysisFields();
}
function updateAnalysisFields() {
  const az = $("#analysis-engine-select").value === "az";
  $("#analysis-model-field").hidden = !az;
  $("#analysis-leaf-field").hidden = !az;
  const simulations = Number($("#analysis-simulations-input").value);
  const leaf = Number($("#analysis-leaf-input").value);
  $("#analysis-settings-note").textContent = az ? `实际分析叶批：${Math.min(leaf, simulations) || "—"}。保存后只影响提示与评估。` : "RIS 提供推荐走法；模型评估需要 AlphaZero。";
}
function renderAnalysis() {
  const result = game.analysis;
  const pending = game.analysis_pending;
  const cfg = game.analysis_settings;
  $("#analysis-engine-tag").textContent = cfg.engine === "az" ? "AlphaZero" : "RIS-ISMCTS";
  const evaluation = evaluationMemory.update(game);
  const { red, black, stale } = evaluation;
  $("#red-evaluation").textContent = red === null ? "—" : `${(red * 100).toFixed(1)}%`;
  $("#black-evaluation").textContent = black === null ? "—" : `${(black * 100).toFixed(1)}%`;
  $("#evaluation-bar").classList.toggle("has-value", red !== null);
  $("#evaluation-bar").classList.toggle("is-stale", stale);
  $("#evaluation-fill").style.width = `${red === null ? 0 : red * 100}%`;
  $("#evaluation-bar").ariaLabel = red === null ? "暂无局势评估" : `${stale ? `保留第 ${evaluation.depth} 手评估，` : ""}红方 ${(red * 100).toFixed(1)}%，黑方 ${(black * 100).toFixed(1)}%`;
  $("#evaluation-note").classList.toggle("is-stale", stale);
  $("#evaluation-note").textContent = game.outcome !== null ? "终局结果" : stale ? `${pending ? "正在更新" : game.analysis_error ? "更新失败" : "当前局面待评估"} · 保留第 ${evaluation.depth} 手胜率` : pending ? `分析第 ${game.depth} 手 · ${pending.seconds.toFixed(1)} 秒` : result ? red !== null ? "模型估计 · 和棋折半" : "RIS 提供走法提示，不提供胜率" : cfg.engine === "az" ? "点击评估，查看当前局势" : "胜率评估需在设置中选择 AlphaZero";
  $("#evaluation-note").title = result ? `${result.simulations} 次模拟 · 叶批 ${result.effective_leaf_batch_size ?? "不适用"} · ${result.seconds.toFixed(2)} 秒` : "";
  $("#hint-description").textContent = pending ? "正在搜索推荐着法…" : result?.label ? `${result.side ? "红方" : "黑方"}推荐：${result.label}` : "棋盘下方可获取 AI 提示";
  $("#analysis-hint").hidden = !pending && !result?.label;
  $("#analysis-error").hidden = !game.analysis_error;
  $("#analysis-error").textContent = game.analysis_error || "";
}
function updateControls() {
  const unavailable = !game || !online || acting || initializing || restoringPreferences;
  for (const id of ["new-game", "open-new-game", "import-nav", "first-button", "prev-button", "next-button", "last-button", "export-button", "flip-button"]) $("#" + id).disabled = unavailable;
  document.querySelectorAll("[data-capture-node]").forEach(button => { button.disabled = unavailable; });
  if (!game) return;
  const ended = game.outcome !== null;
  const fullExport = game.export_format === "jieqi-replay";
  $("#export-label").textContent = fullExport ? "完整棋谱" : "公开记录";
  $("#export-button").ariaLabel = fullExport ? "导出完整棋谱，可重新导入" : "导出公开记录，不支持重新导入";
  $("#export-button").title = fullExport ? "保存全部走子与分支，同时记住当前浏览位置和对局最新位置" : "对局未结束，仅导出公开着法供查阅，不能导入恢复；结束对局后可导出完整棋谱";
  $("#undo-button").disabled = unavailable || game.depth === 0 || game.reviewing;
  const hintUnavailable = unavailable || !!game.analysis_pending || ended || game.ai_turn && !game.reviewing;
  $("#hint-button").disabled = hintUnavailable;
  $("#evaluate-button").disabled = unavailable || !!game.analysis_pending || ended || game.analysis_settings.engine !== "az";
  $("#evaluate-button").title = game.analysis_settings.engine === "az" ? "评估当前局面，显示红黑双方模型估计胜率" : "在分析设置中选择 AlphaZero 后可评估";
  for (const id of ["apply-play-settings", "apply-analysis-settings", "auto-hint", "auto-evaluate"]) $("#" + id).disabled = unavailable;
  $("#auto-evaluate").disabled = unavailable || game.analysis_settings.engine !== "az";
  $("#applied-settings").textContent = `当前对弈：${game.simulations} 次模拟${game.effective_leaf_batch_size !== null ? ` · 实际叶批 ${game.effective_leaf_batch_size}` : " · RIS 无叶批"}`;
  $("#resign-button").disabled = unavailable || ended || game.reviewing;
  $("#resign-button").hidden = game.mode !== "human";
  $("#pause-button").hidden = game.mode !== "aiai" || game.reviewing;
  $("#pause-button").disabled = unavailable || ended;
  $("#pause-button").innerHTML = `<i data-icon="${game.paused ? "play" : "pause"}">${icon(game.paused ? "play" : "pause")}</i><span>${game.paused ? "开始观棋" : "暂停观棋"}</span>`;
  $("#first-button").disabled = unavailable || game.cursor === 0;
  $("#prev-button").disabled = unavailable || game.parent === null;
  $("#next-button").disabled = unavailable || !game.children.length;
  $("#last-button").disabled = unavailable || game.cursor === game.head && !game.reviewing;
  $("#review-actions").hidden = !game.reviewing;
  $("#resume-button").disabled = unavailable;
  $("#continue-button").disabled = unavailable || ended;
  $("#retry-button").hidden = !game.error;
  $("#retry-button").disabled = unavailable;
  $("#engine-status").hidden = !preferences.display.engineDetails && !game.error;
  $("#engine-name").textContent = game.mode === "local" ? "本地双人对弈" : game.engine === "az" ? "AlphaZero" : "RIS-ISMCTS";
  const search = game.last_search;
  $("#engine-detail").textContent = game.pending ? `对弈搜索中 · ${game.pending.seconds.toFixed(1)} 秒` : search ? `${search.simulations} 次模拟 · ${search.seconds.toFixed(2)} 秒${search.device && search.device !== "unknown" ? ` · ${search.device.toUpperCase()}` : ""}` : game.mode === "local" ? "原生规则校验 · 同屏切磋" : `${game.simulations} 次模拟 / 步 · ${game.engine === "az" ? `叶批 ${game.effective_leaf_batch_size}` : "无需模型"}`;
  $("#engine-detail").title = search ? `引擎计算 ${search.seconds.toFixed(2)} 秒，完成后等待网页刷新 ${Number(search.result_wait_seconds || 0).toFixed(2)} 秒` : "";
  $("#rules-version").textContent = game.ruleset_version;
}
function renderStatus() {
  let message;
  if (game.error) message = game.error;
  else if (game.outcome !== null) message = `${game.outcome === 0 ? "握手言和" : game.outcome === 1 ? "红方获胜" : "黑方获胜"} · ${game.reason || "对局结束"}。可以回看棋谱，或开启新对局。`;
  else if (game.reviewing) message = `复盘 · 第 ${game.depth} 手 · ← → 浏览棋谱`;
  else if (game.pending) message = game.pending.kind === "hint" ? "正在寻找值得尝试的一着…" : `${game.side ? "红方" : "黑方"}引擎正在思考，已用 ${game.pending.seconds.toFixed(1)} 秒…`;
  else if (game.paused) message = `对局已暂停 · 待${game.side ? "红" : "黑"}方行棋 · 点击「开始观棋」继续`;
  else if (game.in_check) message = `${game.side ? "红方" : "黑方"}被将军，请先应将。`;
  else if (game.hint) message = `AI 推荐 ${squareName(game.hint[0])} → ${squareName(game.hint[1])} · 蓝色箭头标记`;
  else message = `${game.side ? "红方" : "黑方"}行棋 · 选子后点击落点`;
  $("#board-message").title = message;
  $("#board-message").classList.toggle("is-error", !!game.error);
  $("#board-message").innerHTML = `<span class="status-dot"></span>${escapeText(message)}`;
}

function showModal(content) {
  $("#modal-content").innerHTML = content;
  paintIcons($("#modal-content"));
  if (!$("#modal").open) $("#modal").showModal();
}
function confirmAction({ title, description, confirm = "确认", action }) {
  showModal(`<div class="modal-eyebrow">JIEQI LAB</div><h2 class="modal-title">${title}</h2><p class="modal-description">${description}</p><div class="modal-actions"><button class="button secondary" id="cancel-action">再想想</button><button class="button primary" id="confirm-action">${confirm}</button></div>`);
  $("#cancel-action").addEventListener("click", () => $("#modal").close());
  $("#confirm-action").addEventListener("click", async () => {
    $("#modal").close();
    await action();
  });
}
function showRules() {
  showModal('<div class="modal-eyebrow">A GUIDE TO JIEQI</div><h2 class="modal-title">先藏锋，再见招。</h2><p class="modal-description">揭棋，亦称暗中国象棋。在熟悉的棋盘上，每枚暗子都藏着新的可能。</p><ol class="rules-list"><li><strong>看初始身份，走出第一步</strong>除将帅外，所有棋子起初都是暗子。大字是该位置的初始身份，按此身份走子，落子后自动翻开真实身份。</li><li><strong>揭明之后，各显其能</strong>明子按真实身份行棋。明仕、明相可以离开九宫、跨越河界；暗仕仍限九宫，暗象仍不过河。马腿、象眼、炮架等规则由原生引擎校验。</li><li><strong>暗中有数，吃子有别</strong>被吃暗子的真实身份只告知吃方。人机模式下，对手吃掉的暗子显示为「暗」。同屏双人和观棋使用公共视角。</li><li><strong>方寸之间，亦有章法</strong>将死、困毙判负；连续 6 回合后的长将、长捉受禁着约束。公开局面第三次重复，或连续 80 手无吃子，按当前项目规则判和。</li><li><strong>一步一回望</strong>点击棋谱或按 ← → 复盘。「返回对局」回到最新局面；「从此继续」可以创建变例。重新走到原分支会保留原来的揭示结果。</li></ol><div class="modal-actions"><button class="button primary" id="rules-done">落子，开始探索<i data-icon="arrow-right"></i></button></div>');
  $("#rules-done").addEventListener("click", () => $("#modal").close());
}
$("#modal-close").addEventListener("click", () => $("#modal").close());
$("#modal").addEventListener("click", (event) => {
  if (event.target === $("#modal")) {
    const rect = $("#modal").getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) $("#modal").close();
  }
});
$("#rules-nav").addEventListener("click", showRules);
function setSettingsTab(analysis) {
  $("#play-settings-tab").setAttribute("aria-selected", String(!analysis));
  $("#analysis-settings-tab").setAttribute("aria-selected", String(analysis));
  $("#play-settings-page").hidden = analysis;
  $("#analysis-settings-page").hidden = !analysis;
}
function openSettings(analysis = false) {
  $("#view-options").hidden = true;
  $("#view-options-button").setAttribute("aria-expanded", "false");
  $("#sound-options").hidden = true;
  $("#sound-button").setAttribute("aria-expanded", "false");
  setSettingsTab(analysis);
  if (!$("#settings-dialog").open) $("#settings-dialog").showModal();
}
$("#open-settings").addEventListener("click", () => openSettings());
$("#open-new-game").addEventListener("click", () => openSettings());
$("#open-analysis-settings").addEventListener("click", () => openSettings(true));
$("#settings-close").addEventListener("click", () => $("#settings-dialog").close());
$("#play-settings-tab").addEventListener("click", () => setSettingsTab(false));
$("#analysis-settings-tab").addEventListener("click", () => setSettingsTab(true));
for (const id of ["play-settings-tab", "analysis-settings-tab"]) {
  $("#" + id).addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const analysis = id === "play-settings-tab";
    setSettingsTab(analysis);
    $(analysis ? "#analysis-settings-tab" : "#play-settings-tab").focus();
  });
}
$("#settings-dialog").addEventListener("click", event => {
  if (event.target !== $("#settings-dialog")) return;
  const rect = event.target.getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) event.target.close();
});
$("#view-options-button").addEventListener("click", () => {
  $("#sound-options").hidden = true;
  $("#sound-button").setAttribute("aria-expanded", "false");
  const open = $("#view-options").hidden;
  $("#view-options").hidden = !open;
  $("#view-options-button").setAttribute("aria-expanded", String(open));
  if (open) $("#view-options input").focus();
});
document.addEventListener("click", event => {
  if (!event.target.closest(".view-menu-wrapper")) {
    $("#view-options").hidden = true;
    $("#view-options-button").setAttribute("aria-expanded", "false");
  }
  if (!event.target.closest(".sound-menu-wrapper")) {
    $("#sound-options").hidden = true;
    $("#sound-button").setAttribute("aria-expanded", "false");
  }
});
document.querySelectorAll("[data-view]").forEach(input => input.addEventListener("change", () => {
  preferences.display[input.dataset.view] = input.checked;
  applyDisplayPreferences();
  savePreferences();
}));
applyDisplayPreferences();
$("#sound-button").addEventListener("click", () => {
  const open = $("#sound-options").hidden;
  $("#sound-options").hidden = !open;
  $("#sound-button").setAttribute("aria-expanded", String(open));
  $("#view-options").hidden = true;
  $("#view-options-button").setAttribute("aria-expanded", "false");
  if (open) $("#volume-input").focus();
});
$("#volume-input").addEventListener("input", event => {
  preferences.volume = Number(event.target.value);
  if (preferences.volume > 0 && !soundEnabled) {
    soundEnabled = true;
    try { localStorage.setItem("jieqi-sound", "on"); } catch (_) { /* Optional preference. */ }
  }
  updateSoundButton();
  savePreferences();
});
$("#volume-input").addEventListener("change", () => sound("move"));
$("#sound-preview").addEventListener("click", () => sound("move"));
$("#sound-toggle").addEventListener("click", () => {
  soundEnabled = !(soundEnabled && preferences.volume > 0);
  if (soundEnabled && preferences.volume === 0) { preferences.volume = 80; savePreferences(); }
  try { localStorage.setItem("jieqi-sound", soundEnabled ? "on" : "off"); } catch (_) { /* Optional preference. */ }
  updateSoundButton();
  if (soundEnabled) sound("select");
  toast(soundEnabled ? "音效已开启" : "音效已关闭");
});
function updateSettings() {
  const mode = document.querySelector('input[name="mode"]:checked').value;
  $("#side-field").hidden = mode !== "human";
  $("#engine-fields").hidden = mode === "local";
  $("#model-field").hidden = $("#engine-select").value !== "az";
  $("#leaf-field").hidden = $("#engine-select").value !== "az";
  $("#play-leaf-note").hidden = $("#engine-select").value !== "az";
  $("#play-leaf-note").textContent = `实际对弈叶批：${Math.min(Number($("#leaf-batch-input").value), Number($("#simulations-input").value)) || "—"}，不超过模拟次数。`;
}
document.querySelectorAll('input[name="mode"]').forEach(input => input.addEventListener("change", updateSettings));
$("#engine-select").addEventListener("change", updateSettings);
for (const id of ["simulations-input", "leaf-batch-input"]) $("#" + id).addEventListener("input", updateSettings);
function readSearchSettings(analysis = false) {
  const engine = $(analysis ? "#analysis-engine-select" : "#engine-select").value;
  const model = $(analysis ? "#analysis-model-select" : "#model-select").value;
  const simulationsInput = $(analysis ? "#analysis-simulations-input" : "#simulations-input");
  const leafInput = $(analysis ? "#analysis-leaf-input" : "#leaf-batch-input");
  for (const input of [simulationsInput, leafInput]) {
    if (!input.checkValidity() || !Number.isInteger(input.valueAsNumber)) { input.reportValidity(); toast("请填写范围内的正整数"); return null; }
  }
  if (engine === "az" && !model) { toast("没有可用模型，请选择 RIS 提示或准备 AlphaZero 权重。"); return null; }
  return { engine, model, simulations: simulationsInput.valueAsNumber, leaf_batch_size: leafInput.valueAsNumber };
}
$("#apply-play-settings").addEventListener("click", async () => {
  const settings = readSearchSettings();
  if (settings && await act("settings", settings)) toast("已应用并记住对弈参数");
});
$("#analysis-engine-select").addEventListener("change", updateAnalysisFields);
for (const id of ["analysis-simulations-input", "analysis-leaf-input"]) $("#" + id).addEventListener("input", updateAnalysisFields);
$("#apply-analysis-settings").addEventListener("click", async () => {
  const settings = readSearchSettings(true);
  if (!settings) return;
  settings.auto_hint = $("#auto-hint").checked;
  settings.auto_evaluate = settings.engine === "az" && $("#auto-evaluate").checked;
  if (await act("analysis-settings", settings)) { fillAnalysisSettings(); toast("已应用并记住分析参数"); }
});
for (const id of ["auto-hint", "auto-evaluate"]) {
  $("#" + id).addEventListener("change", async () => {
    const settings = { ...game.analysis_settings, auto_hint: $("#auto-hint").checked, auto_evaluate: $("#auto-evaluate").checked };
    await act("analysis-settings", settings);
    fillAnalysisSettings();
  });
}
$("#new-game").addEventListener("click", () => {
  const mode = document.querySelector('input[name="mode"]:checked').value;
  const search = mode === "local" ? { engine: "ris", simulations: game?.simulations || 64, leaf_batch_size: game?.leaf_batch_size || 64 } : readSearchSettings();
  if (!search) return;
  const settings = { ...search, mode, human_side: Number(document.querySelector('input[name="side"]:checked').value) };
  const start = async () => { if (await act("new", settings)) { $("#settings-dialog").close(); toast(mode === "aiai" ? "新局已准备好，点击「开始观棋」" : "新局已开始，参数已记住"); } };
  if (game?.history.length) confirmAction({ title: "另起一局，重新探索？", description: "当前棋谱将从本次会话移出。你可以先导出公开棋谱；已终局对局可导出完整复盘文件。", confirm: "开启新局", action: start });
  else start();
});
$("#undo-button").addEventListener("click", async () => { if (await act("undo")) { sound("select"); toast("已退回上一轮，原走法保留在棋谱分支中"); } });
$("#hint-button").addEventListener("click", () => act("hint"));
$("#evaluate-button").addEventListener("click", () => act("evaluate"));
$("#flip-button").addEventListener("click", () => { flipped = !flipped; selected = null; drawBoardLines(); renderBoard(); renderPlayers(); sound("select"); });
$("#pause-button").addEventListener("click", () => act("pause"));
$("#retry-button").addEventListener("click", () => act("resume"));
$("#resign-button").addEventListener("click", () => confirmAction({ title: "这一局，就到这里？", description: "认输后将结束本局，你仍可回看棋谱、导出记录，或开启新的对局。", confirm: "确认认输", action: () => act("resign") }));
$("#resume-button").addEventListener("click", () => act("resume"));
$("#continue-button").addEventListener("click", () => act("continue"));
$("#first-button").addEventListener("click", () => act("navigate", { cursor: 0 }));
function previous() { if (game?.parent !== null && game) act("navigate", { cursor: game.parent }); }
function next() {
  if (!game?.children.length) return;
  if (game.children.length === 1) { act("navigate", { cursor: game.children[0] }); return; }
  const choices = game.children.map(id => game.history.find(node => node.id === id));
  showModal(`<div class="modal-eyebrow">EXPLORE A VARIATION</div><h2 class="modal-title">下一着，走哪一条路？</h2><p class="modal-description">这里有 ${choices.length} 条变例，请选择一条继续查看。</p><div class="branch-choices">${choices.map(node => `<button class="button secondary" data-branch="${node.id}">${escapeText(node.label)}<i data-icon="arrow-right"></i></button>`).join("")}</div>`);
  document.querySelectorAll("[data-branch]").forEach(button => button.addEventListener("click", () => { $("#modal").close(); act("navigate", { cursor: Number(button.dataset.branch) }); }));
}
$("#prev-button").addEventListener("click", previous);
$("#next-button").addEventListener("click", next);
$("#last-button").addEventListener("click", () => act("navigate", { cursor: game.head }));
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !$("#sound-options").hidden) {
    $("#sound-options").hidden = true;
    $("#sound-button").setAttribute("aria-expanded", "false");
    $("#sound-button").focus();
    return;
  }
  if (event.key === "Escape" && !$("#view-options").hidden) {
    $("#view-options").hidden = true;
    $("#view-options-button").setAttribute("aria-expanded", "false");
    $("#view-options-button").focus();
    return;
  }
  if (!game || $("#modal").open || $("#settings-dialog").open || /INPUT|SELECT|TEXTAREA/.test(event.target.tagName) || event.altKey || event.ctrlKey || event.metaKey) return;
  if (event.key === "ArrowLeft") { event.preventDefault(); previous(); }
  if (event.key === "ArrowRight") { event.preventDefault(); next(); }
  if (event.key === "Escape") { selected = null; renderBoard(); }
});
function switchTab(position) {
  $("#record-tab").classList.toggle("selected", !position);
  $("#position-tab").classList.toggle("selected", position);
  $("#record-tab").setAttribute("aria-selected", String(!position));
  $("#position-tab").setAttribute("aria-selected", String(position));
  $("#record-content").hidden = position;
  $("#position-content").hidden = !position;
}
$("#record-tab").addEventListener("click", () => switchTab(false));
$("#position-tab").addEventListener("click", () => switchTab(true));
for (const id of ["record-tab", "position-tab"]) {
  $("#" + id).addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    event.stopPropagation();
    const position = id === "record-tab";
    switchTab(position);
    $(position ? "#position-tab" : "#record-tab").focus();
  });
}
async function exportGame() {
  try {
    const record = await jsonRequest("/api/export");
    const blob = new Blob([JSON.stringify(record, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `jieqi-${record.format === "jieqi-replay" ? "replay" : "public"}-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
    toast(record.format === "jieqi-replay" ? "完整棋谱已导出，包含全部走子与分支，可重新导入" : "公开记录已导出，不支持重新导入。对局结束后可导出完整棋谱。");
  } catch (error) { toast(error.message); }
}
$("#export-button").addEventListener("click", exportGame);
$("#import-nav").addEventListener("click", () => $("#import-file").click());
$("#import-file").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  event.target.value = "";
  if (!file) return;
  if (file.size > 2_000_000) { toast("棋谱文件不能超过 2 MB"); return; }
  try {
    const record = JSON.parse(await file.text());
    if (record.format === "jieqi-web-public") { toast("这是未终局的公开记录，仅供查阅。请导入完整的 jieqi-replay 棋谱。"); return; }
    const load = async () => { if (await act("import", { record })) { switchTab(false); toast("棋谱已导入，点击历史走子开始复盘"); } };
    if (game?.history.length) confirmAction({ title: "载入一段新的棋局", description: `即将导入「${escapeText(file.name)}」。当前棋谱将从会话移出，请先导出需要保留的记录。`, confirm: "导入棋谱", action: load });
    else await load();
  } catch (_) { toast("无法读取棋谱，请选择有效的 JSON 对局文件"); }
});
document.addEventListener("visibilitychange", () => { if (!document.hidden) { clearTimeout(pollTimer); poll(); } });

async function init() {
  updateControls();
  try {
    const config = await jsonRequest("/api/config");
    availableModels = config.models;
    const modelOptions = config.models.length ? config.models.map(path => `<option value="${escapeText(path)}">${escapeText(path.replace("data/", ""))}</option>`).join("") : '<option value="">未检测到模型</option>';
    $("#model-select").innerHTML = modelOptions;
    $("#analysis-model-select").innerHTML = modelOptions;
    $("#model-note").textContent = config.models.length ? "仅读取已有权重，不启动训练。" : "无需等待训练，可先使用纯搜索引擎。";
  } catch (_) { $("#model-note").textContent = "暂时无法读取本机模型列表。"; }
  try {
    const state = await jsonRequest("/api/state");
    setConnection(true);
    acceptState(state);
    // Do not reset an existing match, even if it is still at its initial node.
    await restoreSavedPreferences(state);
    fillPlaySettings();
    fillAnalysisSettings();
    updateSettings();
  } catch (_) {
    setConnection(false);
    $("#board-message").textContent = "无法连接服务，正在重试…";
  } finally {
    initializing = false;
    updateControls();
    schedulePoll();
  }
}
init();
