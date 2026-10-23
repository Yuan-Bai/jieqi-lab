"""棋谱持久化、分叉和摆棋的信息/规则完整性回归。"""
import copy
import json
import random

import pytest

import ismcts as im
import jieqi as jq
from game_record import GameRecord, RULE_FIELDS, pack_belief, pack_state, validate_position


def json_roundtrip(game):
    return GameRecord.from_dict(json.loads(json.dumps(game.to_dict())))


def test_branches_preserve_reveals_and_private_beliefs():
    g = GameRecord()
    first = g.add_move((58, 49), record={"reveal": 7, "capture": 0})
    capture = g.add_move((19, 82), record={"reveal": 6, "capture": 7})
    original = pack_state(capture.state), pack_belief(capture.belief)
    g.cursor = first.id
    alternative = next(m for m in g.current.state.legal_moves() if m != (19, 82))
    branch = g.add_move(alternative, rng=random.Random(3))
    assert first.children == [capture.id, branch.id]
    assert (pack_state(capture.state), pack_belief(capture.belief)) == original
    assert capture.belief.unknown_removals[jq.RED][jq.RED] == 1
    restored = json_roundtrip(g)
    assert restored.cursor == branch.id
    for node_id, node in g.nodes.items():
        other = restored.nodes[node_id]
        assert pack_state(other.state) == pack_state(node.state)
        assert pack_belief(other.belief) == pack_belief(node.belief)
        assert other.captured == node.captured
        assert other.children == node.children


def test_same_move_reuses_recorded_reveal_without_changing_tree():
    g = GameRecord()
    child = g.add_move((58, 49), record={"reveal": 2, "capture": 0})
    g.cursor = 0
    assert g.add_move((58, 49), rng=random.Random(10)) is child
    assert len(g.nodes) == 2
    assert child.record['reveal'] == 2


def test_explicit_alternative_reveal_is_preserved():
    g = GameRecord()
    g.add_move((58, 49), record={"reveal": 2, "capture": 0})
    g.cursor = 0
    g.add_move((58, 49), record={"reveal": 7, "capture": 0})
    assert len(g.nodes[0].children) == 2
    restored = json_roundtrip(g)
    assert [restored.nodes[n].record['reveal'] for n in restored.nodes[0].children] == [2, 7]


def test_b0_new_variations_sample_independently_and_existing_moves_keep_results():
    game = GameRecord()
    first = game.add_move((82, 65), rng=random.Random(7))
    game.cursor = 0
    second = game.add_move((82, 63), rng=random.Random(0))
    assert first.record['reveal'] == jq.TYPE_ELEPHANT
    assert second.record['reveal'] == jq.TYPE_PAWN
    restored = json_roundtrip(game)
    restored.cursor = 0
    rng = random.Random(3)
    before = rng.getstate()
    existing = restored.add_move((82, 65), rng=rng)
    assert existing.record == first.record
    assert rng.getstate() == before
    assert len(restored.nodes) == 3


def test_v1_upgrade_and_corrupt_reveal_rejected():
    old = {"format": "jieqi-replay", "version": 1, "ruleset_version": jq.RULESET_VERSION,
           "moves": [{"from": 58, "to": 49, "reveal": 7, "capture": 0}]}
    game = GameRecord.from_dict(old)
    assert game.to_dict()['version'] == 2
    assert json_roundtrip(game).current.record['reveal'] == 7
    del old['moves'][0]['reveal']
    with pytest.raises(ValueError, match="缺失揭示"):
        GameRecord.from_dict(old)


@pytest.mark.parametrize("captured", [
    None, [], {"0": None}, {"0": [[]]}, {"0": [[1, True]]},
    {"0": [[1, True, 7, 0]]}, {"0": [[2, False, 7]]}, {"0": [[True, False, 7]]},
    {"0": [[1, 1, 7]]}, {"0": [[1, False, 0]]}, {"0": [[1, False, 8]]},
    {"0": [[1, False, True]]}, {"0": [[1, True, jq.TYPE_KING]]},
])
def test_invalid_initial_capture_records_are_rejected(captured):
    data = GameRecord().to_dict()
    data["initial"]["captured"] = captured
    with pytest.raises(ValueError, match="吃子"):
        GameRecord.from_dict(data)


def test_previous_ruleset_replay_is_rejected():
    old = GameRecord().to_dict()
    old["ruleset_version"] = jq.RULESET_VERSION - 1
    with pytest.raises(ValueError, match="规则版本"):
        GameRecord.from_dict(old)


def previous_rules_record(game, source=3):
    data = game.to_dict()
    data['ruleset_version'] = source
    data['initial']['state']['rules']['ruleset_version'] = source
    return data


@pytest.mark.parametrize('source', [3, 4])
def test_old_v1_migration_preserves_reveals_source_and_input(source):
    old = {'format': 'jieqi-replay', 'version': 1, 'ruleset_version': source,
           'moves': [{'from': 58, 'to': 49, 'reveal': 7, 'capture': 0},
                     {'from': 19, 'to': 82, 'reveal': 6, 'capture': 7}],
           'end': {'outcome': None, 'reason': 'legacy'}}
    before = copy.deepcopy(old)
    migrated = GameRecord.from_dict(old, migrate_rules=True)
    reference = GameRecord.from_dict(dict(old, ruleset_version=jq.RULESET_VERSION))
    assert pack_state(migrated.current.state) == pack_state(reference.current.state)
    assert pack_belief(migrated.current.belief) == pack_belief(reference.current.belief)
    assert migrated.current.captured == reference.current.captured
    provenance = migrated.metadata['rules_migrations']
    assert provenance == [{'source_ruleset_version': source, 'target_ruleset_version': jq.RULESET_VERSION,
                           'original_end': old['end'], 'initial_rule_history_reset': False}]
    assert migrated.to_dict()['end'] != old['end']  # 新结果重新计算，旧结果仅保留为来源。
    assert json_roundtrip(migrated).metadata['rules_migrations'] == provenance
    assert old == before


@pytest.mark.parametrize('source', [3, 4])
def test_old_v2_migration_and_new_branch_roundtrip(source):
    game = GameRecord()
    first = game.add_move((58, 49), record={'reveal': 7})
    game.add_move((19, 82), record={'reveal': 6, 'capture': 7})
    game.cursor = first.id
    game.add_move((27, 36), record={'reveal': 7})
    old = previous_rules_record(game, source)
    before = copy.deepcopy(old)
    restored = GameRecord.from_dict(old, migrate_rules=True)
    assert restored.cursor == game.cursor
    for node_id, node in game.nodes.items():
        other = restored.nodes[node_id]
        assert pack_state(other.state) == pack_state(node.state)
        assert pack_belief(other.belief) == pack_belief(node.belief)
        assert other.captured == node.captured
        assert other.children == node.children
    restored.cursor = first.id
    restored.add_move((29, 38), record={'reveal': 2})
    saved = json_roundtrip(restored)
    assert len(saved.nodes[first.id].children) == 3
    assert len(saved.nodes) == 5
    assert saved.to_dict()['ruleset_version'] == jq.RULESET_VERSION
    assert saved.metadata['rules_migrations'] == restored.metadata['rules_migrations']
    assert old == before


def test_old_v2_start_tracking_reset_keeps_repetition_capture_and_private_state():
    game = GameRecord()
    game.add_move((58, 49), record={'reveal': 7})
    game.add_move((19, 82), record={'reveal': 6, 'capture': 7})
    node = game.current
    old = previous_rules_record(GameRecord(node.state, node.belief, node.captured))
    initial = old['initial']['state']
    initial['no_capture_plies'] = 17
    initial['consecutive_checks'][jq.RED] = 2
    initial['check_anchor'][jq.RED] = 49
    initial['positions'][0][2] = 2
    restored = GameRecord.from_dict(old, migrate_rules=True)
    defaults = jq.GameState.empty(with_pools=False)
    for field in RULE_FIELDS:
        assert getattr(restored.current.state, field) == getattr(defaults, field)
    assert restored.current.state.no_capture_plies == 17
    assert pack_state(restored.current.state)['positions'] == initial['positions']
    assert pack_belief(restored.current.belief) == old['initial']['belief']
    assert restored.current.captured == node.captured
    migration = restored.metadata['rules_migrations'][0]
    assert migration['initial_rule_history_reset']
    assert migration['original_rule_history']['consecutive_checks'] == [0, 2]
    assert json_roundtrip(restored).metadata['rules_migrations'] == restored.metadata['rules_migrations']


def test_migration_rebuilds_chase_identity_with_current_rules():
    # v4 曾把逃开的车误认成另一枚同类型车，迁移后不能沿用旧的连续捉计数。
    state = jq.GameState.empty(with_pools=False)
    for row, col, kind, side in [(9, 4, 1, jq.RED), (0, 3, 1, jq.BLACK),
                                 (1, 5, 2, jq.RED), (2, 4, 2, jq.BLACK),
                                 (4, 5, 2, jq.BLACK), (7, 4, 7, jq.RED)]:
        state.place_piece(row, col, kind, side, 0)
    state.reset_history()
    old = previous_rules_record(GameRecord(state), source=4)
    old['nodes'] = [{'id': i, 'parent': i - 1,
                     'move': {'from': src, 'to': dst, 'reveal': 0, 'capture': 0}}
                    for i, (src, dst) in enumerate([(14, 13), (22, 21), (13, 14)], 1)]
    old['cursor'] = 3
    migrated = GameRecord.from_dict(old, migrate_rules=True)
    assert migrated.nodes[2].state.chase_target[jq.RED] == 21
    assert migrated.current.state.chase_streak[jq.RED] == 1
    assert migrated.current.state.chase_target[jq.RED] == 41


@pytest.mark.parametrize('source', [None, 2, '3', jq.RULESET_VERSION + 1])
def test_unknown_or_future_rules_cannot_be_migrated(source):
    old = previous_rules_record(GameRecord(), source)
    before = copy.deepcopy(old)
    with pytest.raises(ValueError, match='不支持.*迁移'):
        GameRecord.from_dict(old, migrate_rules=True)
    assert old == before


@pytest.mark.parametrize('mutation, message', [
    (lambda d: d['initial']['state']['rules'].update(forbidden_moves=False), '初始规则配置'),
    (lambda d: d['initial']['state'].update(check_anchor=[-100, -1]), 'check_anchor'),
    (lambda d: d['initial']['state'].update(no_capture_plies=80), '节点 1.*终局'),
    (lambda d: d['nodes'][0]['move'].update(to=48), '节点 1.*不合法'),
    (lambda d: d['nodes'][0]['move'].pop('reveal'), '节点 1.*缺失揭示'),
])
def test_migration_keeps_current_rule_and_record_validation(mutation, message):
    game = GameRecord()
    game.add_move((58, 49), record={'reveal': 7})
    old = previous_rules_record(game)
    mutation(old)
    before = copy.deepcopy(old)
    with pytest.raises(ValueError, match=message):
        GameRecord.from_dict(old, migrate_rules=True)
    assert old == before


@pytest.mark.parametrize('history', [None, [None], [{}],
                                   [{'source_ruleset_version': 3, 'target_ruleset_version': 99}]])
def test_malformed_migration_metadata_is_rejected(history):
    data = GameRecord().to_dict()
    data['rules_migrations'] = history
    with pytest.raises(ValueError, match='规则迁移记录无效'):
        GameRecord.from_dict(data)


def test_custom_dark_position_and_complete_initial_history():
    s = jq.GameState.empty(with_pools=False)
    s.place_piece(0, 4, jq.TYPE_KING, jq.BLACK, 0)
    s.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
    s.place_piece(6, 4, jq.TYPE_PAWN, jq.RED, 1)
    s.pool[jq.RED][jq.TYPE_ROOK] = 1
    s.pool_total[jq.RED] = 1
    s.side = jq.BLACK
    s.reset_history()
    validate_position(s)
    s.no_capture_plies = 17
    s.consecutive_checks[jq.RED] = 2
    s.check_anchor[jq.RED] = 58
    s.position_counts[(jq.RED, tuple(s.board))] = 2
    g = GameRecord(s, im.init_belief_for_game(s))
    restored = json_roundtrip(g)
    assert pack_state(restored.current.state) == pack_state(s)
    restored.add_move((4, 3))
    restored.add_move((58, 49), record={"reveal": jq.TYPE_ROOK})
    assert restored.current.state.board[49] == jq.make_piece(jq.TYPE_ROOK, jq.RED, 0)


def test_initial_private_belief_survives_reroot():
    g = GameRecord()
    g.add_move((58, 49), record={"reveal": 7})
    g.add_move((19, 82), record={"reveal": 6, "capture": 7})
    node = g.current
    reroot = GameRecord(node.state, node.belief, node.captured)
    restored = json_roundtrip(reroot)
    assert pack_belief(restored.current.belief) == pack_belief(node.belief)
    assert restored.current.captured == node.captured


@pytest.mark.parametrize('mutation', [
    lambda d: d.update(version=88),
    lambda d: d.update(cursor=100),
    lambda d: d['nodes'][0].update(parent=1),
    lambda d: d['nodes'][0].update(id=2),
    lambda d: d['nodes'][0]['move'].update(to=100),
    lambda d: d['initial']['state']['pool'][0].__setitem__(2, 0),
    lambda d: d['initial']['belief']['pool_total'][0].__setitem__(0, 99),
])
def test_invalid_v2_records_rejected_without_mutation(mutation):
    g = GameRecord()
    g.add_move((58, 49), record={"reveal": 7})
    before = g.to_dict()
    invalid = copy.deepcopy(before)
    mutation(invalid)
    with pytest.raises((ValueError, KeyError, TypeError)):
        GameRecord.from_dict(invalid)
    assert g.to_dict() == before


def test_setup_pool_total_and_material_validation():
    state = jq.GameState.initial()
    state.set_piece(0, 0)
    with pytest.raises(ValueError, match="暗池总数"):
        validate_position(state)
    state = jq.GameState.initial()
    state.set_piece(40, jq.make_piece(jq.TYPE_KING, jq.RED, 1))
    with pytest.raises(ValueError, match="将帅不能"):
        validate_position(state)
