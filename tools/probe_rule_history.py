"""只读验证规则历史编码、旧检查点兼容和一次内存中的训练步骤。

python tools/probe_rule_history.py --data data/selfplay/某个v2文件.pkl.gz
不保存模型或训练数据；--out 可另存验证摘要。
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import azero as az
import ismcts as im
import jieqi as jq


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="data/models/latest.pt")
    parser.add_argument("--data", default="")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    records, training_samples, blobs, retained_sizes = [], [], [], []
    rng = random.Random(20261001)
    for _ in range(args.games):
        state = jq.GameState.initial(rng=rng)
        belief = im.init_belief_for_game(state)
        history = []
        for _ in range(160):
            if state.game_outcome()[0] is not None:
                break
            info = belief.info_state(state, state.side)
            moves = state.legal_moves()
            policy = [0.0] * az.MOVE_SIZE
            policy[az.move_to_index(moves[0])] = 1.0
            sample = az.finalize_samples([az.raw_sample_from_info(info, history, policy)], jq.RESULT_DRAW)[0]
            tensor = az.compact_to_tensors(sample)[0]
            assert torch.equal(tensor, az.encode_state(info, history, info.side))
            for transform in (az.mirror_sample, az.flip_vertical_sample):
                assert transform(transform(sample)) == sample
            if len(records) < 64:
                records.append(tensor)
                training_samples.append(sample)
            blobs.append(len(sample.repetition_history))
            retained_sizes.append(len(az.unpack_repetition_history(sample.repetition_history)))
            move = rng.choice(moves)
            undo = state.apply_move(move)
            belief.observe_move(undo)
            history.append(move)
    assert records, "need at least one nonterminal state"
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    old_channels = checkpoint["conv.weight"].shape[1]
    old = az.PolicyValueNet().to(device)
    old.conv = torch.nn.Conv2d(old_channels, old.conv.out_channels, 3, padding=1, bias=False).to(device)
    old.load_state_dict(checkpoint)
    new = az.PolicyValueNet().to(device)
    padded = az.load_state_dict_padded(new, checkpoint)
    old.eval()
    new.eval()
    x = torch.stack(records).to(device)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def inference_ms(net, inputs):
        with torch.no_grad():
            for _ in range(5):
                net(inputs)
            synchronize()
            start = time.perf_counter()
            for _ in range(20):
                net(inputs)
            synchronize()
        return (time.perf_counter() - start) * 1000 / 20

    with torch.no_grad():
        expected, actual = old(x[:, :old_channels]), new(x)
        differences = [float((a - b).abs().max()) for a, b in zip(actual, expected)]
        policy_difference = float((actual[0].softmax(dim=1) - expected[0].softmax(dim=1)).abs().max())
        # 通道数变化会选择不同的 cuDNN/TF32 内核。另用完整 FP32 核验补零数学等价。
        allow_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
        expected, actual = old(x[:, :old_channels]), new(x)
        fp32_differences = [float((a - b).abs().max()) for a, b in zip(actual, expected)]
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
        torch.backends.cudnn.allow_tf32 = allow_tf32
    timing = {"old_ms": inference_ms(old, x[:, :old_channels]), "new_ms": inference_ms(new, x)}
    new.train()
    optimizer = torch.optim.SGD(new.parameters(), lr=1e-5)
    loss, _, _ = az.train_step(new, optimizer, training_samples[:8], device, 5.0)
    assert torch.isfinite(torch.tensor(loss))
    assert all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None)
    added_gradient = float(new.conv.weight.grad[:, old_channels:].abs().sum())
    if padded:
        assert added_gradient > 0
    result = dict(device=str(device), channels=az.STATE_CHANNELS, checkpoint_channels=old_channels,
                  padded=padded, positions=len(blobs), max_relevant_history=max(retained_sizes),
                  mean_history_bytes=round(sum(blobs) / len(blobs), 1), max_history_bytes=max(blobs),
                  forward_max_abs_diff=differences, inference_batch=len(records), inference_ms=timing,
                  default_policy_probability_max_diff=policy_difference, fp32_forward_max_abs_diff=fp32_differences,
                  finite_training_loss=loss, added_input_gradient_l1=added_gradient,
                  model_or_training_files_written=False)
    if args.data:
        samples, _ = az.load_compact_payload(args.data)
        result["existing_data_samples"] = len(samples)
        result["existing_data_missing_history"] = sum(s.repetition_history is None for s in samples)
        for sample in samples[:16]:
            for augmented in az.augment_samples([sample], mirror=True, flip=True):
                assert az.compact_to_tensors(augmented)[0].shape[0] == az.STATE_CHANNELS
    print(json.dumps(result, ensure_ascii=False))
    if args.out:
        path = Path(args.out)
        with path.open("x", encoding="utf-8") as file:  # 不覆盖既有结果。
            json.dump(result, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
