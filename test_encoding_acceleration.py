"""Bitwise regression against the scalar encoder and original training assembly."""
import copy
import random
import unittest

import numpy as np
import torch
import torch.nn.functional as F

import azero as az
import ismcts as im
import jieqi as jq


def reference_encode(info, history):
    planes = np.zeros((az.STATE_CHANNELS, jq.BOARD_H, jq.BOARD_W), dtype=np.float32)
    for idx, piece in enumerate(info.board):
        if piece:
            plane = (14 if jq.is_dark(piece) else 0) + jq.piece_color(piece) * 7 + jq.piece_type(piece) - 1
            planes[plane, jq.ROWS[idx], jq.COLS[idx]] = 1.0
    planes[28] = 1.0 if info.side == jq.RED else 0.0
    for side in (jq.BLACK, jq.RED):
        for i, piece_type in enumerate(jq.POOL_TYPES):
            denom = jq.POOL_INIT[piece_type]
            if denom:
                planes[29 + side * len(jq.POOL_TYPES) + i] = info.pool_known[side][piece_type] / denom
        planes[41 + side] = info.pool_total_known[side] / jq.POOL_TOTAL
        planes[43 + side] = info.unknown_removals[side] / jq.POOL_TOTAL
    planes[45] = info.no_capture_plies / jq.NO_CAPTURE_LIMIT_TIANTIAN_PLIES
    for i, side in enumerate((jq.RED, jq.BLACK)):
        planes[46 + i] = info.consecutive_checks[side] / (jq.LONG_CHECK_ROUNDS + 1)
        planes[65 + i] = info.chase_streak[side] / jq.LONG_CHASE_ROUNDS
        for channel, idx in ((67 + i, info.chase_target[side]), (69 + i, info.chase_anchor[side]),
                             (71 + i, info.check_anchor[side])):
            if idx >= 0:
                planes[channel, jq.ROWS[idx], jq.COLS[idx]] = 1.0
        planes[73 + i] = info.chase_target_code[side] / 31.0
    for i, (source, target) in enumerate(history[-az.HISTORY_PLIES:]):
        planes[48 + 2 * i, jq.ROWS[source], jq.COLS[source]] = 1.0
        planes[49 + 2 * i, jq.ROWS[target], jq.COLS[target]] = 1.0
    current = (info.side, tuple(info.board))
    planes[64] = (info.position_counts.get(current, 1) if info.position_counts else 1) / jq.REPETITION_LIMIT
    if info.position_counts is not None and info.repetition_history_complete:
        occupied = sum(piece != 0 for piece in info.board)
        dark = sum(jq.is_dark(piece) for piece in info.board if piece)
        entries = [(int(side), bytes(board), int(count)) for (side, board), count in info.position_counts.items()
                   if sum(piece != 0 for piece in board) <= occupied
                   and sum(jq.is_dark(piece) for piece in board if piece) <= dark]
        if current not in info.position_counts:
            entries.append((info.side, bytes(info.board), 1))
        if len(entries) > az.REPETITION_HISTORY_SIZE:
            raise ValueError("repetition history overflow")
        planes[75] = 1.0
        for i, (side, board, count) in enumerate(entries):
            planes[76 + 2 * i] = np.frombuffer(board, dtype=np.uint8).reshape(10, 9) / 31.0
            planes[77 + 2 * i] = (2 * count + side) / (2 * jq.REPETITION_LIMIT + 1)
    return planes


def fixtures():
    rng = random.Random(20261005)
    codes = [0] + [jq.make_piece(kind, side, dark) for kind in range(1, 8)
                   for side in (jq.BLACK, jq.RED) for dark in (0, 1)]
    pairs = []
    for i in range(40):
        info = im.InformationState(
            board=[rng.choice(codes) if rng.random() < 0.5 else 0 for _ in range(jq.BOARD_SIZE)],
            side=i % 2, pool_known=[[rng.randrange(6) for _ in range(8)] for _ in range(2)],
            pool_total_known=[rng.randrange(16), rng.randrange(16)],
            unknown_removals=[rng.randrange(16), rng.randrange(16)], no_capture_plies=rng.randrange(81),
            consecutive_checks=[rng.randrange(9), rng.randrange(9)],
            check_anchor=[rng.randrange(-1, 90), rng.randrange(-1, 90)],
            chase_streak=[rng.randrange(9), rng.randrange(9)],
            chase_target=[rng.randrange(-1, 90), rng.randrange(-1, 90)],
            chase_target_code=[rng.choice(codes), rng.choice(codes)],
            chase_anchor=[rng.randrange(-1, 90), rng.randrange(-1, 90)],
        )
        info.position_counts = {}
        for j in range(i * 2):
            board = info.board.copy()
            for _ in range(5):
                board[rng.randrange(90)] = rng.choice(codes)
            info.position_counts[(j % 2, tuple(board))] = rng.randrange(1, 9)
        if i % 3:
            info.position_counts[(info.side, tuple(info.board))] = 1 + i % 4
        if i % 7 == 0:
            info.repetition_history_complete = False
        if i % 11 == 0:
            info.position_counts = None
        history = [(rng.randrange(90), rng.randrange(90)) for _ in range(i % 18)]
        pairs.append((info, history))
    return pairs


def reference_batch(batch):
    triples = [(torch.from_numpy(reference_encode(az._compact_information_state(s), s.history)),
                az.dense_policy_from_sparse(s.policy_idx, s.policy_prob),
                torch.tensor(s.value, dtype=torch.float32)) for s in batch]
    return tuple(torch.stack(items) for items in zip(*triples))


def reference_train_step(net, optimizer, batch, device, grad_clip):
    states, policies, targets = [x.to(device) for x in reference_batch(batch)]
    logits, values = net(states)
    per_sample_policy = -(policies * F.log_softmax(logits, dim=1)).sum(dim=1)
    weights = torch.tensor([float(getattr(s, "policy_weight", 1.0)) for s in batch],
                           dtype=torch.float32, device=device)
    if float(weights.sum()) > 0:
        policy_loss = (per_sample_policy * weights).sum() / weights.sum()
    else:
        policy_loss = torch.zeros((), dtype=torch.float32, device=device)
    value_loss = F.mse_loss(values, targets)
    loss = policy_loss + value_loss
    optimizer.zero_grad()
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
    optimizer.step()
    return tuple(float(x.item()) for x in (loss, policy_loss, value_loss))


def sample_fixtures():
    samples = []
    for i, (info, history) in enumerate(fixtures()):
        policy = [0.0] * az.MOVE_SIZE
        policy[0], policy[77] = 0.3, 0.7
        sample = az.finalize_samples([az.raw_sample_from_info(info, history, policy)], jq.RESULT_DRAW)[0]
        sample.value = (-1.0, -0.0, 1.0)[i % 3]
        sample.policy_weight = (0.0, 0.25, 1.0)[i % 3]
        # Preserve zip truncation, negative indexing and last-write-wins behavior.
        sample.policy_idx.extend([77, -1, 10])
        sample.policy_prob.extend([0.123456789, -0.0])
        if i % 13 == 0:
            del sample.repetition_history
            del sample.policy_weight
        samples.extend((sample, az.mirror_sample(sample), az.flip_vertical_sample(sample)))
    return samples


class TestEncodingAcceleration(unittest.TestCase):
    def setUp(self):
        self.torch_rng = torch.get_rng_state().clone()

    def tearDown(self):
        torch.set_rng_state(self.torch_rng)

    def assert_bits_equal(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertTrue(torch.equal(actual.view(torch.int32), expected.view(torch.int32)))

    def test_scalar_and_batch_encoding_match_original_bits_without_mutation(self):
        pairs = fixtures()
        before = copy.deepcopy(pairs)
        expected = np.stack([reference_encode(*pair) for pair in pairs])
        for pair, row in zip(pairs, expected):
            self.assertEqual(az.encode_state_numpy(*pair).tobytes(), row.tobytes())
        actual = az._encode_batch_numpy(*zip(*pairs))
        self.assertEqual(actual.tobytes(), expected.tobytes())
        self.assertTrue(actual.flags.c_contiguous)
        self.assertEqual(pairs, before)

    def test_lookup_preserves_float64_division_rounding(self):
        expected = (np.arange(256, dtype=np.uint8) / 31.0).astype(np.float32)
        self.assertEqual(az._REPETITION_PIECE_VALUES.tobytes(), expected.tobytes())
        self.assertFalse(az._REPETITION_PIECE_VALUES.flags.writeable)

    def test_full_history_capacity_and_absent_current_key(self):
        info, history = fixtures()[1]
        info.board = [0] * jq.BOARD_SIZE
        info.board[89] = jq.make_piece(jq.TYPE_ROOK, jq.RED, 0)
        info.position_counts = {}
        for idx in range(az.REPETITION_HISTORY_SIZE - 1):
            board = [0] * jq.BOARD_SIZE
            board[idx] = info.board[89]
            info.position_counts[(idx % 2, tuple(board))] = 1 + idx
        self.assertEqual(az.encode_state_numpy(info, history).tobytes(), reference_encode(info, history).tobytes())
        self.assertTrue(np.all(az.encode_state_numpy(info, history)[-1] != 0))

    def test_compact_batch_matches_original_bits_and_legacy_samples(self):
        samples = sample_fixtures()
        # Rebuild copies through pickle-free slots copying, including missing fields.
        before = copy.deepcopy(samples)
        rng = torch.get_rng_state().clone()
        for actual, expected in zip(az.compact_batch_to_tensors(samples), reference_batch(samples)):
            self.assert_bits_equal(actual, expected)
            self.assertTrue(actual.is_contiguous())
        for sample, saved in zip(samples, before):
            for field in az.CompactSample.__slots__:
                self.assertEqual(getattr(sample, field, None), getattr(saved, field, None))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_batches_do_not_share_storage_or_mutate_samples(self):
        samples = sample_fixtures()[:3]
        first = az.compact_batch_to_tensors(samples)
        second = az.compact_batch_to_tensors(samples)
        expected = reference_batch(samples)
        for a, b, saved in zip(first, second, expected):
            a.fill_(99.0)
            self.assert_bits_equal(b, saved)
        for a, b in zip(az.compact_batch_to_tensors(samples), expected):
            self.assert_bits_equal(a, b)

    def test_empty_batch_rejected(self):
        with self.assertRaises(ValueError):
            az.compact_batch_to_tensors([])
        with self.assertRaises(ValueError):
            az._encode_batch_numpy([], [])

    def test_optimizer_updates_match_original_bits(self):
        torch.manual_seed(11)
        first = az.PolicyValueNet(channels=8, blocks=1)
        second = copy.deepcopy(first)
        opt_first = torch.optim.AdamW(first.parameters(), lr=5e-5, weight_decay=1e-4)
        opt_second = torch.optim.AdamW(second.parameters(), lr=5e-5, weight_decay=1e-4)
        samples = sample_fixtures()
        for step in range(3):
            batch = samples[step * 7:step * 7 + 7]
            if step == 1:
                for sample in batch:
                    sample.policy_weight = 0.0
            expected = reference_train_step(first, opt_first, batch, torch.device("cpu"), 1.0)
            actual = az.train_step(second, opt_second, batch, torch.device("cpu"), 1.0)
            self.assertEqual(actual, expected)
            for key, value in first.state_dict().items():
                self.assertTrue(torch.equal(second.state_dict()[key], value), key)
            for a, b in zip(first.parameters(), second.parameters()):
                if a.grad is None:
                    self.assertIsNone(b.grad)
                else:
                    self.assertTrue(torch.equal(a.grad, b.grad))
            for key, state in opt_first.state_dict()["state"].items():
                for name, value in state.items():
                    self.assertTrue(torch.equal(opt_second.state_dict()["state"][key][name], value))


if __name__ == "__main__":
    unittest.main()
