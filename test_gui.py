"""真实 Tk 控件上的回放/走子/保存/分析代次验证，无需加载大模型。"""
import json
import random
import time
import tkinter as tk
from dataclasses import asdict
from types import SimpleNamespace

import pytest

import gui
from game_record import GameRecord


@pytest.fixture(scope='module')
def tk_master():
    # 一个 Tcl 解释器，多份独立窗口；避免 Windows Tk 反复初始化解释器。
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        if 'no display name' in str(exc) or 'couldn\'t connect to display' in str(exc):
            pytest.skip(f'Tk 显示环境不可用：{exc}')
        raise
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def app(monkeypatch, tk_master):
    monkeypatch.setattr(gui.SoundManager, '_ensure_files', lambda self: None)
    monkeypatch.setattr(gui.messagebox, 'showinfo', lambda *a, **k: None)
    monkeypatch.setattr(gui.messagebox, 'askyesnocancel', lambda *a, **k: False)
    monkeypatch.setattr(gui.messagebox, 'showerror', lambda *a, **k: pytest.fail(str(a)))
    root = tk.Toplevel(tk_master)
    root.withdraw()
    app = gui.JieQiGUI(root)
    app.sound_var.set(False)
    errors = []
    tk_master.report_callback_exception = lambda *args: errors.append(args)
    yield app
    app._dirty = False
    for timer in tk_master.tk.call('after', 'info'):
        root.after_cancel(timer)
    app._analysis_after = None
    app._close()
    assert not errors, errors


def sample_game():
    g = GameRecord(metadata={'mode': 'human', 'human_side': 1, 'god_view': False})
    first = g.add_move((58, 49), record={'reveal': 7})
    g.add_move((19, 82), record={'reveal': 6, 'capture': 7})
    g.cursor = 0
    g.add_move((54, 45), record={'reveal': 2})
    g.cursor = 0
    return g, first


def click(app, index):
    x, y = app._point(index)
    app.on_click(SimpleNamespace(x=x, y=y))


def test_next_at_fork_requires_explicit_number_choice(app):
    g, first = sample_game()
    app._activate_replay(g)
    app._replay_step(1)
    assert g.cursor == 0
    assert app._branch_visible
    assert app.canvas.find_withtag('branch_1')
    assert app.canvas.find_withtag('branch_2')
    assert app.next_btn.cget('text') == '选择分支 →'
    app._choose_branch(1)
    assert g.cursor == first.id
    assert not app._branch_visible
    app._replay_step(1)
    assert g.cursor == first.children[0]
    app._replay_step(-1)
    assert g.cursor == first.id


def test_board_creates_branch_and_does_not_mutate_other_positions(app):
    g, _ = sample_game()
    app._activate_replay(g)
    old = g.to_dict()['nodes']
    click(app, 56)
    click(app, 47)
    assert len(g.nodes[0].children) == 3
    assert g.to_dict()['nodes'][:len(old)] == old
    assert app.state is not g.current.state
    assert app._dirty
    app.root.update()
    assert app.record_tree.selection() == (f'n{g.cursor}',)


def test_saving_mid_replay_keeps_entire_tree(app, monkeypatch, tmp_path):
    g, first = sample_game()
    app._activate_replay(g)
    app._replay_goto(first.id)
    path = tmp_path / 'replay.json'
    monkeypatch.setattr(gui.filedialog, 'asksaveasfilename', lambda **kwargs: str(path))
    assert app.save_game_file()
    restored = GameRecord.from_dict(json.loads(path.read_text(encoding='utf-8')))
    assert len(restored.nodes) == len(g.nodes)
    assert restored.cursor == first.id
    assert len(restored.nodes[0].children) == 2
    assert app.save_btn.cget('state') == 'normal'


def test_failed_load_leaves_current_tree_intact(app, monkeypatch, tmp_path):
    g, _ = sample_game()
    app._activate_replay(g)
    errors = []
    monkeypatch.setattr(gui.messagebox, 'showerror', lambda *args: errors.append(args))
    path = tmp_path / 'broken.json'
    path.write_text('{"format":"wrong"}', encoding='utf-8')
    assert not app._load_replay_file(str(path))
    assert app.record is g
    assert errors


def test_gui_loads_old_rules_without_rewriting_file_and_can_save_branch(app, monkeypatch, tmp_path):
    old = {'format': 'jieqi-replay', 'version': 1, 'ruleset_version': 3,
           'moves': [{'from': 58, 'to': 49, 'reveal': 7, 'capture': 0}],
           'end': {'outcome': None, 'reason': ''}}
    source = tmp_path / 'old.json'
    source.write_text(json.dumps(old), encoding='utf-8')
    original = source.read_bytes()
    assert app._load_replay_file(str(source))
    assert '原规则 v3' in app.header_var.get()
    assert f'v{gui.jq.RULESET_VERSION} 重新校验' in app.header_var.get()
    assert app.record.current.record['reveal'] == 7
    app._replay_goto(0)
    click(app, 54)
    click(app, 45)
    destination = tmp_path / 'migrated.json'
    monkeypatch.setattr(gui.filedialog, 'asksaveasfilename', lambda **kwargs: str(destination))
    assert app.save_game_file()
    restored = GameRecord.from_dict(json.loads(destination.read_text(encoding='utf-8')))
    assert len(restored.nodes[0].children) == 2
    assert restored.metadata['rules_migrations'][0]['source_ruleset_version'] == 3
    assert restored.metadata['rules_migrations'][0]['original_end'] == old['end']
    assert source.read_bytes() == original
    assert app._load_replay_file(str(destination))
    assert '原规则 v3' in app.header_var.get()


def test_old_rules_invalid_move_load_preserves_current_record(app, monkeypatch, tmp_path):
    game, _ = sample_game()
    app._activate_replay(game)
    before = game.to_dict()
    old = game.to_dict()
    old['ruleset_version'] = 3
    old['initial']['state']['rules']['ruleset_version'] = 3
    old['nodes'][0]['move']['to'] = 48
    source = tmp_path / 'old-invalid.json'
    source.write_text(json.dumps(old), encoding='utf-8')
    errors = []
    monkeypatch.setattr(gui.messagebox, 'showerror', lambda *args: errors.append(args))
    assert not app._load_replay_file(str(source))
    assert app.record is game
    assert game.to_dict() == before
    assert '节点 1' in errors[0][1]


def test_old_rules_root_tracking_reset_is_visible(app):
    old = GameRecord().to_dict()
    old['ruleset_version'] = 4
    old['initial']['state']['rules']['ruleset_version'] = 4
    old['initial']['state']['consecutive_checks'] = [0, 2]
    assert app._enter_replay(old)
    assert '起点将/捉历史重新计数' in app.header_var.get()
    assert app.state.consecutive_checks == [0, 0]


def pump(app, predicate):
    until = time.monotonic() + 3
    while time.monotonic() < until:
        app.root.update()
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError('GUI 后台结果未返回')


def test_replay_analysis_cache_and_same_depth_branch_invalidates(app, monkeypatch):
    g, first = sample_game()
    app._activate_replay(g)
    app._replay_goto(first.id)
    calls = []

    class Agent:
        def analyse(self, state, belief, history, sims):
            calls.append(tuple(state.board))
            return state.legal_moves()[0], .6

    monkeypatch.setattr(app._agents, 'get', lambda *args: Agent())
    app.analysis_settings = {'agent': 'az', 'sims': 2}
    app.hint_on = app.winrate_on = True
    app._maybe_start_analysis()
    token, key = app._analysis_token, app._analysis_key
    pump(app, lambda: not app.analysis_thinking)
    assert app.hint_move is not None
    assert '20.0%' in app.eval_var.get()  # 黑方行动者 value=.6 => 红方20%
    app._replay_goto(3)
    assert app.hint_move is None
    assert app._analysis_token != token
    app._poll_analysis(token, key)  # 相同步数/相同行动方的旧结果也必须失效
    assert app.hint_move is None
    app._maybe_start_analysis()
    pump(app, lambda: not app.analysis_thinking)
    app._replay_goto(first.id)
    app._maybe_start_analysis()
    assert len(calls) == 2  # 回到已有节点命中缓存


def test_gui_analysis_keeps_private_h7h0_capture_out_of_red_observation(app, monkeypatch):
    game = GameRecord(metadata={'mode': 'aiai', 'god_view': True})
    first = game.add_move((82, 65), record={'reveal': gui.jq.TYPE_PAWN})
    rook = game.add_move((25, 88), record={'reveal': gui.jq.TYPE_ADVISOR, 'capture': gui.jq.TYPE_ROOK})
    game.cursor = first.id
    pawn = game.add_move((25, 88), record={'reveal': gui.jq.TYPE_ADVISOR, 'capture': gui.jq.TYPE_PAWN})
    assert rook.state.pool != pawn.state.pool
    assert rook.belief.pool_counts[gui.jq.BLACK] != pawn.belief.pool_counts[gui.jq.BLACK]
    snapshots = []

    class Agent:
        def analyse(self, state, belief, history, sims):
            info = belief.info_state(state, state.side)
            snapshots.append((asdict(info.info), tuple(history), sims))
            return state.legal_moves()[0], .2

    monkeypatch.setattr(app._agents, 'get', lambda *args: Agent())
    app._activate_replay(game)
    app.analysis_settings = {'agent': 'az', 'sims': 800}
    app.hint_on = False
    app.winrate_on = True
    for node in (rook, pawn):
        app._replay_goto(node.id)
        app._maybe_start_analysis()
        pump(app, lambda: not app.analysis_thinking)
        assert '60.0%' in app.eval_var.get()  # 红方行动的根价值，不读上帝视角捕获标签。
    assert len(snapshots) == 2
    assert snapshots[0] == snapshots[1]
    assert snapshots[0][0]['side'] == gui.jq.RED
    assert snapshots[0][0]['unknown_removals'][gui.jq.RED] == 1
    assert snapshots[0][1:] == (((82, 65), (25, 88)), 800)


def test_cancelled_delayed_ai_cannot_move_in_replay(app):
    app.human_sims_var.set('1')
    app.start_human_game(gui.jq.BLACK)
    old = app._ai_token
    app.enter_replay_from_game()
    app._start_ai_search({'agent': 'ris', 'sims': 1}, old)
    assert len(app.record.nodes) == 1
    assert not app.ai_thinking


def test_position_editor_applies_dark_setup(app):
    app.open_position_editor()
    editor = next(w for w in app.root.winfo_children() if isinstance(w, gui.PositionEditor))
    editor._empty()
    editor.type_var.set('帅/将')
    editor.color_var.set('红方')
    editor._place(SimpleNamespace(x=4.5 * 48, y=9.5 * 48))
    editor.color_var.set('黑方')
    editor._place(SimpleNamespace(x=4.5 * 48, y=.5 * 48))
    editor.color_var.set('红方')
    editor.type_var.set('兵/卒')
    editor.dark_var.set(True)
    editor._place(SimpleNamespace(x=4.5 * 48, y=6.5 * 48))
    editor.pool_vars[gui.jq.RED, gui.jq.TYPE_ROOK].set('1')
    editor._apply()
    assert app.replay_mode
    assert app.game_kind == gui.MODE_STUDY
    assert app.state.pool[gui.jq.RED][gui.jq.TYPE_ROOK] == 1
    assert app.state.board[58] == gui.jq.make_piece(gui.jq.TYPE_PAWN, gui.jq.RED, 1)
    assert app._dirty


def test_continue_human_game_keeps_original_tree(app):
    g, _ = sample_game()
    app._activate_replay(g)
    app.continue_game()
    dialog = next(w for w in app.root.winfo_children() if isinstance(w, tk.Toplevel))
    start = next(w for w in dialog.winfo_children() if isinstance(w, tk.Button) and w.cget('text') == '开始对弈')
    start.invoke()
    assert app.record is g
    assert not app.replay_mode
    assert app.human_side == gui.jq.RED
    assert not app.god_view
    click(app, 56)
    click(app, 47)
    assert len(g.nodes[0].children) == 3
    assert app.ai_thinking


def test_ai_winrate_option_exposes_model_picker(app):
    app.mode_var.set(gui.MODE_AIAI)
    app.red_agent_var.set(gui.AGENT_LABELS[gui.AGENT_RIS])
    app.black_agent_var.set(gui.AGENT_LABELS[gui.AGENT_RIS])
    app.ai_winrate_var.set(True)
    app._refresh_az_visibility()
    assert app.ai_az_frame.winfo_manager() == 'pack'


def test_small_screen_keeps_controls_accessible_by_scroll(app, monkeypatch):
    monkeypatch.setattr(app.root, 'winfo_screenheight', lambda: 768)
    monkeypatch.setattr(app.root, 'winfo_screenwidth', lambda: 900)
    g, _ = sample_game()
    app._activate_replay(g)
    assert app.game_vscroll.winfo_manager() == 'grid'
    assert app.game_hscroll.winfo_manager() == 'grid'
    assert int(app.game_viewport.cget('height')) <= 668


def test_search_worker_does_not_capture_tk_objects(app, monkeypatch):
    import inspect
    g, _ = sample_game()
    app._activate_replay(g)
    app.analysis_settings = {'agent': 'ris', 'sims': 1}
    app.hint_on = True
    jobs = []
    monkeypatch.setattr(app._analysis_runner, 'submit', lambda work: jobs.append(work) or 1)
    app._maybe_start_analysis()
    assert jobs
    for value in inspect.getclosurevars(jobs[0]).nonlocals.values():
        assert not isinstance(value, (gui.JieQiGUI, tk.Misc))
        assert getattr(value, '__self__', None) is not app


def test_az_budget_change_reuses_loaded_model(monkeypatch):
    made = []
    monkeypatch.setattr(gui, 'AzAgent', lambda *args: made.append(args) or SimpleNamespace(config=SimpleNamespace()))
    pool = gui.AgentPool()
    rng = object()
    one = pool.get({'agent': 'az', 'model_path': 'unused.pt', 'device': 'cpu', 'sims': 100}, 'play', rng)
    two = pool.get({'agent': 'az', 'model_path': 'unused.pt', 'device': 'cpu', 'sims': 800}, 'play', rng)
    assert one is two
    assert len(made) == 1
    assert two.config.mcts_simulations == 800
    assert two.config.leaf_batch_size == 64


def test_leaf_batch_changes_reuse_model_and_isolate_analysis_lane(monkeypatch):
    made = []
    monkeypatch.setattr(gui, 'AzAgent', lambda *args: made.append(args) or SimpleNamespace(config=SimpleNamespace()))
    pool = gui.AgentPool()
    settings = {'agent': 'az', 'model_path': 'unused.pt', 'device': 'cpu', 'sims': 800, 'leaf_batch_size': 64}
    play = pool.get(settings, 'play', object())
    analysis = pool.get(dict(settings, leaf_batch_size=32), 'analysis', object())
    assert pool.get(dict(settings, leaf_batch_size=16), 'play', object()) is play
    assert play.config.leaf_batch_size == 16
    assert analysis.config.leaf_batch_size == 32
    assert len(made) == 2
    assert pool.settings_key(settings) != pool.settings_key(dict(settings, leaf_batch_size=16))
    assert pool.get(settings, 'play', object()) is play
    assert play.config.leaf_batch_size == 64


def dialog_controls(dialog):
    for child in dialog.winfo_children():
        yield child
        yield from dialog_controls(child)


def leaf_combo(dialog):
    return next(w for w in dialog_controls(dialog) if isinstance(w, gui.ttk.Combobox)
                and tuple(w.cget('values')) == gui.LEAF_BATCH_CHOICES)


def test_analysis_dialog_changes_batch_and_invalidates_old_value(app, monkeypatch, tmp_path):
    model = tmp_path / 'weights.pt'
    model.write_bytes(b'GUI validation fixture only')
    game, _ = sample_game()
    app._activate_replay(game)
    app.analysis_settings = {'agent': 'az', 'model_path': str(model), 'device': 'cpu',
                             'sims': 800, 'leaf_batch_size': 64}
    app.winrate_on = True
    old_key, old_token = app._current_analysis_key(), app._analysis_token
    app._analysis_cache[old_key] = (None, .5, 1.)
    scheduled = []
    monkeypatch.setattr(app, '_schedule_analysis', lambda: scheduled.append(True))
    app._analysis_settings_dialog()
    dialog = next(w for w in app.root.winfo_children() if isinstance(w, tk.Toplevel))
    leaf_combo(dialog).set('16')
    next(w for w in dialog.winfo_children() if isinstance(w, tk.Button) and w.cget('text') == '应用').invoke()
    assert app.analysis_settings['leaf_batch_size'] == 16
    assert app.analysis_settings['sims'] == 800
    assert app.analysis_leaf_batch_size == 16
    assert app._analysis_token != old_token
    assert app._current_analysis_key() != old_key
    assert app._current_analysis_key() not in app._analysis_cache
    assert '评估中' in app.eval_var.get()
    assert scheduled == [True]
    assert len(game.nodes) == 4


@pytest.mark.parametrize('invalid', ['0', '-1', '1.5', 'abc', '100001'])
def test_invalid_analysis_batch_leaves_settings_and_search_intact(app, monkeypatch, invalid):
    app.analysis_settings = {'agent': 'ris', 'sims': 800}
    app.analysis_leaf_batch_size = 64
    errors = []
    monkeypatch.setattr(gui.messagebox, 'showerror', lambda *args, **kwargs: errors.append(args))
    before, token = dict(app.analysis_settings), app._analysis_token
    app._analysis_settings_dialog()
    dialog = next(w for w in app.root.winfo_children() if isinstance(w, tk.Toplevel))
    # 留空模型；此次验证只检查参数原子应用。
    combos = [w for w in dialog_controls(dialog) if isinstance(w, gui.ttk.Combobox)]
    combos[0].set('')
    leaf_combo(dialog).set(invalid)
    next(w for w in dialog.winfo_children() if isinstance(w, tk.Button) and w.cget('text') == '应用').invoke()
    assert errors and '叶批大小' in errors[0][1]
    assert app.analysis_settings == before
    assert app.analysis_leaf_batch_size == 64
    assert app._analysis_token == token


def test_play_batch_update_discards_old_scheduled_search_without_changing_position(app, monkeypatch):
    app.start_human_game(gui.jq.RED)
    first = app.record.add_move((82, 65), record={'reveal': gui.jq.TYPE_PAWN})
    app._display_node(first.id)
    old_settings = {'agent': 'az', 'sims': 800, 'model_path': 'unused.pt', 'leaf_batch_size': 64}
    app.side_settings[gui.jq.BLACK] = old_settings
    app._maybe_ai_move()
    token, before = app._ai_token, app.record.to_dict()
    restart = []
    monkeypatch.setattr(app, '_maybe_ai_move', lambda: restart.append(dict(app.side_settings[gui.jq.BLACK])))
    app._set_play_leaf_batches({gui.jq.BLACK: 16})
    assert old_settings['leaf_batch_size'] == 64
    assert app.side_settings[gui.jq.BLACK]['leaf_batch_size'] == 16
    assert app.human_leaf_batch_var.get() == '16'
    assert app._ai_token != token
    assert restart[0]['leaf_batch_size'] == 16
    app._start_ai_search(old_settings, token)
    app._poll_ai_search(token)
    assert app.record.to_dict() == before


def test_play_batch_validation_is_atomic_for_both_sides(app):
    app.start_human_game(gui.jq.RED)
    app.game_kind = gui.MODE_AIAI
    app.side_settings = {side: {'agent': 'az', 'sims': 800, 'leaf_batch_size': 64}
                         for side in (gui.jq.RED, gui.jq.BLACK)}
    before, token = {side: dict(v) for side, v in app.side_settings.items()}, app._ai_token
    with pytest.raises(ValueError, match='叶批大小'):
        app._set_play_leaf_batches({gui.jq.RED: 16, gui.jq.BLACK: 0})
    assert app.side_settings == before
    assert app._ai_token == token


def test_play_batch_dialog_updates_both_sides_and_preserves_pause(app, monkeypatch, tmp_path):
    model = tmp_path / 'weights.pt'
    model.write_bytes(b'GUI validation fixture only')
    app.ai_model_box.var.set(str(model))
    app.red_agent_var.set(gui.AGENT_LABELS[gui.AGENT_AZ])
    app.black_agent_var.set(gui.AGENT_LABELS[gui.AGENT_AZ])
    maybe_move = app._maybe_ai_move
    monkeypatch.setattr(app, '_maybe_ai_move', lambda: None)
    app.start_ai_game()
    app.paused = True
    monkeypatch.setattr(app, '_maybe_ai_move', maybe_move)
    before = app.record.to_dict()
    app._play_batch_settings_dialog()
    dialog = next(w for w in app.root.winfo_children() if isinstance(w, tk.Toplevel))
    combos = [w for w in dialog_controls(dialog) if isinstance(w, gui.ttk.Combobox)]
    assert len(combos) == 2
    combos[0].set('16')
    combos[1].set('32')
    next(w for w in dialog.winfo_children() if isinstance(w, tk.Button) and w.cget('text') == '应用').invoke()
    assert app.side_settings[gui.jq.RED]['leaf_batch_size'] == 16
    assert app.side_settings[gui.jq.BLACK]['leaf_batch_size'] == 32
    assert app.paused and not app.ai_thinking
    assert app._ai_runner._thread is None
    assert app.record.to_dict() == before


def test_ai_start_takes_separate_leaf_batches_and_ris_ignores_field(app, monkeypatch, tmp_path):
    model = tmp_path / 'weights.pt'
    model.write_bytes(b'GUI validation fixture only')
    app.ai_model_box.var.set(str(model))
    app.red_agent_var.set(gui.AGENT_LABELS[gui.AGENT_AZ])
    app.black_agent_var.set(gui.AGENT_LABELS[gui.AGENT_AZ])
    app.red_leaf_batch_var.set('16')
    app.black_leaf_batch_var.set('32')
    monkeypatch.setattr(app, '_maybe_ai_move', lambda: None)
    app.start_ai_game()
    assert app.side_settings[gui.jq.RED]['leaf_batch_size'] == 16
    assert app.side_settings[gui.jq.BLACK]['leaf_batch_size'] == 32
    assert app.play_batch_btn.cget('state') == 'normal'
    app.human_agent_var.set(gui.AGENT_LABELS[gui.AGENT_RIS])
    app.human_leaf_batch_var.set('invalid unused AZ field')
    app.start_human_game(gui.jq.RED)
    assert 'leaf_batch_size' not in app.side_settings[gui.jq.BLACK]
    assert app.play_batch_btn.cget('state') == 'disabled'


def test_agent_pool_updates_actual_search_leaf_batches_without_loading_again(monkeypatch):
    import torch
    import azero as az
    from gui_tasks import SearchBelief, SearchRandom

    loads = []

    class RecordingNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batches = []

        def forward(self, x):
            self.batches.append(len(x))
            return torch.zeros((len(x), az.MOVE_SIZE)), torch.zeros(len(x))

    monkeypatch.setattr(az, 'PolicyValueNet', RecordingNet)
    monkeypatch.setattr(az, 'load_model_if_exists', lambda *args: loads.append(True) or True)
    pool, rng, game = gui.AgentPool(), SearchRandom(), GameRecord()
    settings = {'agent': 'az', 'model_path': 'unused.pt', 'device': 'cpu', 'sims': 80}
    agents = []
    for batch in (16, 32, 64):
        agent = pool.get(dict(settings, leaf_batch_size=batch), 'analysis', rng)
        agent.config.leaf_dedup = False  # 验证实际叶前向批量，不让去重缩小批次。
        agent.net.batches.clear()
        rng.seed(3)
        move, value = agent.analyse(game.current.state, SearchBelief(game.current.belief, rng), [], 80)
        assert move in game.current.state.legal_moves()
        assert -1 <= value <= 1
        assert agent.net.batches[0] == 1  # 根单次评估不属于叶批量。
        assert max(agent.net.batches[1:]) == batch
        assert sum(agent.net.batches[1:]) == 80
        agents.append(agent)
    assert agents[0] is agents[1] is agents[2]
    assert loads == [True]


def linear_game(plies=60):
    game = GameRecord()
    rng = random.Random(7)
    for _ in range(plies):
        game.add_move(rng.choice(game.current.state.legal_moves()), rng=rng)
    return game


def test_long_history_is_vertical_and_click_still_selects_position(app):
    game = linear_game()
    app._activate_replay(game)
    assert app.record_tree.get_children('') == tuple(f'n{i}' for i in range(61))
    assert all(app.record_tree.parent(f'n{i}') == '' for i in game.nodes)
    app.record_tree.selection_set('n40')
    app._tree_selected(None)
    assert game.cursor == 40
    assert app.state.board == game.nodes[40].state.board


def test_new_fork_reparents_existing_line_without_staircase(app):
    game = linear_game(8)
    original = game.to_dict()['nodes']
    app._activate_replay(game)
    game.cursor = 2
    existing_move = game.nodes[3].record
    alternative = next(move for move in game.current.state.legal_moves()
                       if move != (existing_move['from'], existing_move['to']))
    branch = game.add_move(alternative, rng=random.Random(5))
    followup = game.add_move(game.current.state.legal_moves()[0], rng=random.Random(6))
    app._replay_goto(followup.id)
    assert app.record_tree.get_children('') == ('n0', 'n1', 'n2')
    expected = tuple(f'n{i}' for i in range(3, 9)) + (f'n{branch.id}', f'n{followup.id}')
    assert app.record_tree.get_children('n2') == expected
    assert all(app.record_tree.parent(iid) == 'n2' for iid in expected)
    assert game.to_dict()['nodes'][:8] == original
    app.record_tree.selection_set('n5')
    app._tree_selected(None)
    assert game.cursor == 5
    app._replay_step(-1)
    assert game.cursor == 4  # 回放仍按真实父节点返回，不按展示父级跳转。
