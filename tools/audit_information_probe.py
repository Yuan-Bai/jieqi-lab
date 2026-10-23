"""Read-only, model-free probes for the 2026-09-30 information-state review.

Run: python tools/audit_information_probe.py
Updated 2026-10-01: capture probes check the corrected observation boundaries;
rule-encoding probes check distinct inputs; chance probes verify full world sampling
even when the initial branch cache is incomplete.
No game/model/training files are changed.
"""

import json
import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import azero as az
import ismcts as im
import jieqi as jq


def emit(name, **values):
    print(json.dumps({"probe": name, **values}, ensure_ascii=False))


def capture_state(dark_mover):
    state = jq.GameState.empty(rng=random.Random(7), with_pools=False)
    state.place_piece(0, 4, jq.TYPE_KING, jq.BLACK, 0)
    state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
    state.place_piece(5, 4, jq.TYPE_PAWN, jq.RED, 0)
    state.place_piece(5, 0, jq.TYPE_ROOK, jq.RED, int(dark_mover))
    state.place_piece(3, 0, jq.TYPE_PAWN, jq.BLACK, 1)
    state.place_piece(3, 8, jq.TYPE_PAWN, jq.BLACK, 1)
    state.place_piece(6, 8, jq.TYPE_PAWN, jq.RED, 1)
    state.pool[jq.BLACK][jq.TYPE_ROOK] = 1
    state.pool[jq.BLACK][jq.TYPE_KNIGHT] = 1
    state.pool_total[jq.BLACK] = 2
    state.pool[jq.RED][jq.TYPE_PAWN] = 1
    if dark_mover:
        state.pool[jq.RED][jq.TYPE_ROOK] = 1
    state.pool_total[jq.RED] = 1 + int(dark_mover)
    state.reset_history()
    move = (jq.index(5, 0), jq.index(3, 0))
    assert move in state.legal_moves()
    return state, move


def private_capture_probe():
    state, move = capture_state(True)
    belief = im.init_belief_for_game(state)
    info = belief.info_state(state, jq.RED)
    spec = az.chance_spec_for_move(state, info, move)
    root = az.PUCTNode(1.0, jq.RED)
    root.expand([move], torch.ones(az.MOVE_SIZE), {move: spec}, 0, random.Random(9))
    layer = next(
        child for child in root.children[move].children.values()
        if child.reveal_moved == jq.TYPE_PAWN
    )
    outcomes = list(layer.children.values())
    actor_inputs, root_inputs, actor_keys, root_keys = [], [], [], []
    for node in outcomes:
        world = state.clone()
        view = belief.clone()
        undo = world.apply_move(
            move, reveal_choice=jq.TYPE_PAWN,
            capture_reveal_choice=node.reveal_captured,
        )
        view.observe_move(undo)
        actor_inputs.append(az.encode_state_numpy(view.info_state(world, jq.BLACK), [move]))
        root_inputs.append(az.encode_state_numpy(view.info_state(world, jq.RED), [move]))
        public = ((move, node.reveal_moved),)
        actor_keys.append(im.information_set_key(view.info_state(world, jq.BLACK), public, ()))
        root_keys.append(im.information_set_key(view.info_state(world, jq.RED), public, ((0, node.reveal_captured),)))
    assert len(outcomes) == 2
    assert np.array_equal(actor_inputs[0], actor_inputs[1])
    assert not np.array_equal(root_inputs[0], root_inputs[1])
    assert outcomes[0] is not outcomes[1]
    assert actor_keys[0] == actor_keys[1]
    assert root_keys[0] != root_keys[1]
    emit(
        "private_capture",
        next_actor="black", correct_black_inputs_equal=True,
        capturer_inputs_equal=False, environment_outcomes=2, victim_information_sets=1,
        capturer_information_sets=2,
        meaning="distinct environment outcomes resolve to the same victim information-set key",
    )


def bright_capture_probe():
    state, move = capture_state(False)
    spec = az.chance_spec_for_move(state, im.init_belief_for_game(state).info_state(state, jq.RED), move)
    assert spec is not None and spec["moved_counts"] is None and spec["capture_counts"] is not None
    emit("bright_capture_dark", moved_random=False, captured_random=True,
         meaning="private capture chance represented without a public mover reveal")


def lazy_chance_probe():
    state = jq.GameState.initial(rng=random.Random(3))
    move = (jq.index(6, 4), jq.index(5, 4))
    counts = [0] * 8
    counts[jq.TYPE_ROOK] = 9
    counts[jq.TYPE_PAWN] = 1
    state.pool[jq.RED] = counts
    state.pool_total[jq.RED] = 10
    chance = az.PUCTNode(1.0, jq.BLACK, move=move, is_chance=True)
    only = az.PUCTNode(1.0, jq.BLACK)
    only.reveal_moved = jq.TYPE_PAWN
    chance.children[0] = only
    mcts = az.NeuralMCTS(None, az.AZConfig(chance_nodes=True, chance_k=1), torch.device("cpu"))
    actual = [mcts._navigate_chance(chance, state, random.Random(i))[0].reveal_moved for i in range(100)]
    assert set(actual) == {jq.TYPE_ROOK, jq.TYPE_PAWN}
    pawn_frequency = actual.count(jq.TYPE_PAWN) / len(actual)
    assert 0.02 < pawn_frequency < 0.2
    weights = [5, 3, 2]
    old_expected = [0.0, 0.0, 0.0]
    expected = [0.0, 0.0, 0.0]
    types = (jq.TYPE_ROOK, jq.TYPE_KNIGHT, jq.TYPE_PAWN)
    state.pool[jq.RED] = [0] * 8
    for t, weight in zip(types, weights):
        state.pool[jq.RED][t] = weight

    class Seat:
        def __init__(self, seat):
            self.seat = seat

        def randrange(self, total):
            assert total == 10
            return self.seat

    # Enumerate the exact weighted-without-replacement construction for k=2,
    # followed by the old conditional sampler vs the corrected complete-world sampler.
    for first, second in itertools.permutations(range(3), 2):
        pair_probability = weights[first] / 10 * weights[second] / (10 - weights[first])
        for chosen in (first, second):
            old_expected[chosen] += pair_probability * weights[chosen] / (weights[first] + weights[second])
        partial = az.PUCTNode(1.0, jq.BLACK, move=move, is_chance=True)
        for index, chosen in enumerate((first, second)):
            branch = az.PUCTNode(weights[chosen] / 10, jq.BLACK)
            branch.reveal_moved = types[chosen]
            partial.children[index] = branch
        for seat in range(10):
            outcome, _ = mcts._navigate_chance(partial, state, Seat(seat))
            expected[types.index(outcome.reveal_moved)] += pair_probability / 10
    assert not np.allclose(old_expected, [0.5, 0.3, 0.2])
    assert np.allclose(expected, [0.5, 0.3, 0.2])
    emit("lazy_chance", original_pawn_probability=0.1, observed_pawn_probability=pawn_frequency,
         k2_original_distribution=[0.5, 0.3, 0.2],
         old_k2_unconditional_distribution=[round(value, 6) for value in old_expected],
         k2_unconditional_distribution=[round(value, 6) for value in expected],
         meaning="sample complete world pools first, then create/reuse outcome branches")


def encoding_probe():
    state = jq.GameState.empty(rng=random.Random(3), with_pools=False)
    state.place_piece(0, 4, jq.TYPE_KING, jq.BLACK, 0)
    state.place_piece(9, 4, jq.TYPE_KING, jq.RED, 0)
    state.place_piece(5, 4, jq.TYPE_PAWN, jq.RED, 0)
    state.place_piece(2, 3, jq.TYPE_ROOK, jq.RED, 0)
    state.place_piece(2, 5, jq.TYPE_ROOK, jq.RED, 0)
    state.reset_history()
    belief = im.init_belief_for_game(state)
    first = belief.info_state(state, jq.RED)
    second = belief.info_state(state, jq.RED)
    first.consecutive_checks[jq.RED] = second.consecutive_checks[jq.RED] = 6
    first.check_anchor[jq.RED] = jq.index(2, 3)
    second.check_anchor[jq.RED] = jq.index(2, 5)
    anchor_equal = np.array_equal(az.encode_state_numpy(first, []), az.encode_state_numpy(second, []))
    check_move = (jq.index(2, 3), jq.index(2, 4))
    anchor_legality = [check_move in info.determinize(random.Random(3)).legal_moves() for info in (first, second)]
    assert anchor_legality == [False, True]
    state = jq.GameState.initial(rng=random.Random(3))
    belief = im.init_belief_for_game(state)
    first = belief.info_state(state, jq.RED)
    second = belief.info_state(state, jq.RED)
    other_board = state.board.copy()
    other_board[jq.index(6, 4)] = 0
    other_board[jq.index(5, 4)] = jq.make_piece(jq.TYPE_PAWN, jq.RED, 0)
    other_key = (jq.BLACK, tuple(other_board))
    first.position_counts[other_key] = 1
    second.position_counts[other_key] = 2
    repetition_equal = np.array_equal(az.encode_state_numpy(first, []), az.encode_state_numpy(second, []))
    repeat_outcomes = []
    for info in (first, second):
        world = info.determinize(random.Random(3))
        world.apply_move((jq.index(6, 4), jq.index(5, 4)), reveal_choice=jq.TYPE_PAWN)
        repeat_outcomes.append(world.game_outcome()[0])
    assert repeat_outcomes == [None, jq.RESULT_DRAW]
    assert not anchor_equal and not repetition_equal
    emit("encoding", different_check_anchors_same_tensor=anchor_equal,
         check_move_legality=anchor_legality,
         different_other_position_counts_same_tensor=repetition_equal,
         same_next_move_outcomes=repeat_outcomes,
         meaning="rule histories producing different legality/outcomes now have distinct network inputs")


if __name__ == "__main__":
    private_capture_probe()
    bright_capture_probe()
    lazy_chance_probe()
    encoding_probe()
