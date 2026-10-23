"""短时训练资源/学习信号探针；所有产物另存，不推进主线训练状态。

python tools/probe_training_budget.py --data data/selfplay/某个文件.pkl.gz
旧数据仅用于原 batch 的资源测试；小规模新自博弈按整局划分训练/留出集。
输出不是棋力评估，不代表完整 4M 缓冲或 8 worker 的内存峰值。
"""
import argparse
import gc
import hashlib
import json
import math
import random
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import azero as az

try:
    import psutil  # 可选：缺失时仍报告 CUDA 分配/保留峰值。
except ImportError:
    psutil = None


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


class Resources:
    """采样当前进程 RSS；CUDA 数字含模型/梯度/优化器，不含驱动上下文。"""
    def __init__(self, device):
        self.device = device
        self.stop = threading.Event()
        self.rss_peak = 0
        self.ram_available_min = None

    def poll(self):
        process = psutil.Process() if psutil else None
        while not self.stop.is_set():
            if process:
                self.rss_peak = max(self.rss_peak, process.memory_info().rss)
                available = psutil.virtual_memory().available
                self.ram_available_min = min(self.ram_available_min or available, available)
            self.stop.wait(0.05)

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self.poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.stop.set()
        self.thread.join()
        self.result = dict(seconds=time.perf_counter() - self.started,
                           process_rss_peak_bytes=self.rss_peak or None,
                           system_ram_available_min_bytes=self.ram_available_min)
        if self.device.type == "cuda":
            self.result.update(cuda_allocated_peak_bytes=torch.cuda.max_memory_allocated(self.device),
                               cuda_reserved_peak_bytes=torch.cuda.max_memory_reserved(self.device))


def evaluate_targets(net, samples, device):
    """固定样本的 eval 模式损失；CE 包含目标自身熵，KL 才是策略拟合差距。"""
    net.eval()
    ce = entropy = mse = count = 0
    predicted, target_values = [], []
    with torch.no_grad():
        for start in range(0, len(samples), 64):
            triples = [az.compact_to_tensors(s) for s in samples[start:start + 64]]
            states, policies, targets = [torch.stack(items).to(device) for items in zip(*triples)]
            logits, values = net(states)
            ce += float(-(policies * F.log_softmax(logits, dim=1)).sum())
            entropy += float(-(policies * policies.clamp_min(1e-30).log()).sum())
            mse += float(((values - targets) ** 2).sum())
            count += len(triples)
            predicted.extend(values.tolist())
            target_values.extend(targets.tolist())
    mean = sum(target_values) / count
    constant_mse = sum((v - mean) ** 2 for v in target_values) / count
    return dict(positions=count, policy_ce=ce / count, target_entropy=entropy / count,
                policy_kl=(ce - entropy) / count, value_mse=mse / count,
                value_target_mean=mean, value_constant_baseline_mse=constant_mse)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="data/models/latest.pt")
    parser.add_argument("--data", required=True)
    parser.add_argument("--state", default="data/models/train_state.json")
    parser.add_argument("--out", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--profile-steps", type=int, default=20)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--pilot-steps", type=int, default=100)
    parser.add_argument("--pilot-batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    if min(args.batch, args.profile_steps, args.sims, args.pilot_steps, args.pilot_batch) <= 0 or args.games < 2:
        parser.error("步数、batch、sims 必须为正，games 至少 2")
    root = Path(__file__).resolve().parents[1]
    output = (Path(args.out) if args.out else root / "data" / "pilots" / datetime.now().strftime("training_%Y%m%d_%H%M%S"))
    output = output.resolve()
    if not output.is_relative_to(root):
        parser.error("输出目录必须在本项目内")
    output.mkdir(parents=True, exist_ok=False)
    hashes_before = {str(Path(p).resolve()): digest(p) for p in (args.model, args.data, args.state)}
    parent_state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    # 当前主线 cosine 的下一轮学习率；不修改主线 scheduler/计数。
    iteration = parent_state["iteration"]
    lr = 1e-5 + (5e-5 - 1e-5) * (1 + math.cos(math.pi * iteration / 2000)) / 2
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)

    def new_net():
        net = az.PolicyValueNet().to(device)
        az.load_state_dict_padded(net, checkpoint)
        return net

    manifest = dict(arguments=vars(args), source_hashes=hashes_before, parent_state=parent_state,
                    learning_rate=lr, learning_rate_assumption="用户 5e-5 / cosine / 2000轮 / min 1e-5",
                    ruleset_version=az.jq.RULESET_VERSION, data_version=az.DATA_VERSION,
                    state_channels=az.STATE_CHANNELS, checkpoint_channels=checkpoint["conv.weight"].shape[1],
                    torch_version=torch.__version__, device=str(device),
                    device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                    limitation="短时单进程小缓冲；未测完整4M/8worker/800+3200搜索或棋力；优化器从新建AdamW开始")
    write_json(output / "manifest.json", manifest)
    print(f"输出目录: {output}", flush=True)
    net = new_net()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    net.train()
    with Resources(device) as resource:
        samples, meta = az.load_compact_payload(args.data)
        augmented = az.augment_samples(samples, mirror=True, flip=True)
    data_resource = resource.result
    if len(augmented) < args.batch:
        raise ValueError("旧数据不足原 batch；不通过重复样本伪造资源测试")
    with Resources(device) as resource:
        for _ in range(3):
            az.train_step(net, optimizer, rng.sample(augmented, args.batch), device, 1.0)
    warmup_resource = resource.result
    with Resources(device) as resource:
        losses = [az.train_step(net, optimizer, rng.sample(augmented, args.batch), device, 1.0)
                  for _ in range(args.profile_steps)]
    profile = dict(data_loading=data_resource, warmup=warmup_resource, training=resource.result,
                   raw_positions=len(samples), augmented_positions=len(augmented), batch=args.batch,
                   steps=args.profile_steps, mean_training_loss=sum(x[0] for x in losses) / len(losses),
                   network_parameters=sum(p.numel() for p in net.parameters()),
                   trunk_residual_parameters=sum(p.numel() for p in net.residual.parameters()),
                   policy_fc_parameters=sum(p.numel() for p in net.policy_fc.parameters()))
    profile["training"]["milliseconds_per_step"] = 1000 * profile["training"]["seconds"] / args.profile_steps
    write_json(output / "resource_profile.json", profile)
    print("原 batch 资源测试: " + json.dumps(profile["training"]), flush=True)
    # 丢弃资源测试时的权重/梯度；新数据由原始父模型生成。
    del augmented, samples, net, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    net = new_net()
    config = az.AZConfig(mcts_simulations=args.sims, self_play_games=args.games,
                         self_play_workers=1, chance_nodes=True, chance_k=0)
    games = []
    with Resources(device) as resource:
        for index in range(args.games):
            result = az.self_play_game(net, config, device, random.Random(args.seed + index))
            if result.outcome is None or not result.samples:
                raise RuntimeError("探针不把未结束对局标成和棋")
            games.append(result)
            print(f"新数据: game={index + 1}/{args.games} moves={result.move_count} outcome={result.outcome}", flush=True)
    generation = resource.result
    # 先按整局分开，再增强；留出集不参加优化、不混入其对称样本。
    heldout_games = max(1, args.games // 4)
    train_raw = [s for g in games[:-heldout_games] for s in g.samples]
    heldout_raw = [s for g in games[-heldout_games:] for s in g.samples]
    train_augmented = az.augment_samples(train_raw, mirror=True, flip=True)
    batch = min(args.pilot_batch, len(train_augmented))
    train_fixed = random.Random(args.seed).sample(train_raw, min(256, len(train_raw)))
    heldout_fixed = random.Random(args.seed).sample(heldout_raw, min(256, len(heldout_raw)))
    curve = [dict(step=0, train=evaluate_targets(net, train_fixed, device),
                  heldout=evaluate_targets(net, heldout_fixed, device))]
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    with Resources(device) as resource:
        for step in range(1, args.pilot_steps + 1):
            net.train()
            losses = az.train_step(net, optimizer, rng.sample(train_augmented, batch), device, 1.0)
            if not all(math.isfinite(x) for x in losses):
                raise RuntimeError("训练损失非有限值")
            if step % 25 == 0 or step == args.pilot_steps:
                curve.append(dict(step=step, train=evaluate_targets(net, train_fixed, device),
                                  heldout=evaluate_targets(net, heldout_fixed, device)))
                print("短训: " + json.dumps(curve[-1]), flush=True)
    added_gradient = float(net.conv.weight.grad[:, checkpoint["conv.weight"].shape[1]:].abs().sum())
    if not all(torch.isfinite(p).all() for p in net.parameters()):
        raise RuntimeError("模型参数非有限值")
    with (output / "pilot.pt").open("xb") as file:
        torch.save(net.state_dict(), file)
    # 新样本用现有序列化函数另存，便于复查，不合入主线。
    az.save_compact_samples([s for g in games for s in g.samples], str(output / "fresh.pkl.gz"),
                           dict(diagnostic=True, ruleset_version=az.jq.RULESET_VERSION,
                                game_lengths=[len(g.samples) for g in games], seed=args.seed))
    unchanged = {p: digest(p) == value for p, value in hashes_before.items()}
    pilot = dict(selfplay_config=asdict(config), selfplay=generation, training=resource.result,
                 game_lengths=[g.move_count for g in games], outcomes=[g.outcome for g in games],
                 train_games=args.games - heldout_games, heldout_games=heldout_games,
                 train_positions=len(train_raw), heldout_positions=len(heldout_raw),
                 train_augmented_positions=len(train_augmented), batch=batch,
                 learning_curve=curve, added_input_gradient_l1=added_gradient,
                 finite_parameters=True, source_files_unchanged=unchanged,
                 strength_evaluated=False, production_resume_checkpoint=False)
    write_json(output / "pilot_result.json", pilot)
    if not all(unchanged.values()):
        raise RuntimeError("检测到源文件被外部修改，不能宣称主线产物不变")
    print(f"完成；主线模型、数据、状态哈希保持一致。诊断模型: {output / 'pilot.pt'}", flush=True)


if __name__ == "__main__":
    main()
