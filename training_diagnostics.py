"""Fixed whole-game diagnostics, independent of training RNG and replay augmentation."""
from __future__ import annotations

import math
from pathlib import Path, PurePosixPath
from random import Random

import torch
import torch.nn.functional as F

import run_records as rr


VERSION = 1
SPLIT_SEED = 20261002


def select_games(strong_flags: list[bool], count: int) -> tuple[list[int], list[int]]:
    """Draw once, stratified by search budget. Keep a disjoint training reference."""
    total = len(strong_flags)
    if count < 1 or total < 2 * count:
        raise ValueError("固定留出需要 self_play_games >= 2 * diagnostic_games >= 2")
    rng = Random(SPLIT_SEED)  # Never use global Python/self-play/torch RNG.
    strong = [i for i, flag in enumerate(strong_flags) if flag]
    normal = [i for i, flag in enumerate(strong_flags) if not flag]
    rng.shuffle(strong)
    rng.shuffle(normal)
    n_strong = min(len(strong), max(count - len(normal), round(count * len(strong) / total)))
    heldout = sorted(strong[:n_strong] + normal[:count - n_strong])
    remaining = strong[n_strong:] + normal[count - n_strong:]
    rng.shuffle(remaining)
    # Proportional training reference too, when the remaining stratum permits it.
    ref_strong = [i for i in remaining if strong_flags[i]]
    ref_normal = [i for i in remaining if not strong_flags[i]]
    n_ref_strong = min(len(ref_strong), max(count - len(ref_normal), n_strong))
    reference = sorted(ref_strong[:n_ref_strong] + ref_normal[:count - n_ref_strong])
    return heldout, reference


def dataset_path(data_dir, relative: str) -> Path:
    parts = PurePosixPath(relative)
    if (not isinstance(relative, str) or "\\" in relative or ":" in relative or
            parts.is_absolute() or len(parts.parts) != 4 or parts.parts[0:2] != ("diagnostics", "sets") or
            any(part in (".", "..") for part in parts.parts)):
        raise ValueError("无效诊断数据路径")
    rr.run_dir(".", parts.parts[2])  # Validate the attempt UUID.
    if parts.name not in ("heldout.pkl.gz", "training_reference.pkl.gz"):
        raise ValueError("无效诊断数据文件名")
    return Path(data_dir) / relative


def create_fixed_sets(results, strong_flags, count, attempt, data_dir, save_samples, ruleset):
    heldout, reference = select_games(strong_flags, count)
    if any(not results[i].samples or results[i].outcome is None for i in heldout + reference):
        raise ValueError("固定诊断集只能来自有样本且自然结束的整局")
    folder = Path(data_dir) / "diagnostics" / "sets" / attempt["attempt_id"]
    folder.mkdir(parents=True, exist_ok=False)  # Never replace an earlier failed attempt's files.
    dataset = dict(version=VERSION, requested_games=count, split_seed=SPLIT_SEED,
                   source=dict(attempt), created_iteration=attempt["iteration"], sets={})
    samples_by_role = {}
    for role, indices in (("heldout", heldout), ("training_reference", reference)):
        samples = [sample for i in indices for sample in results[i].samples]
        meta = dict(role=role, ruleset_version=ruleset, provenance=dict(attempt), game_indices=indices,
                    game_lengths=[len(results[i].samples) for i in indices],
                    strong_flags=[strong_flags[i] for i in indices],
                    outcomes=[results[i].outcome for i in indices])
        path = folder / (role + ".pkl.gz")
        save_samples(samples, str(path), meta)
        dataset["sets"][role] = dict(path=path.relative_to(data_dir).as_posix(),
                                     sha256=rr.file_record(path)["sha256"], positions=len(samples),
                                     games=len(indices), game_indices=indices,
                                     strong_games=sum(meta["strong_flags"]))
        samples_by_role[role] = samples
    return dict(version=VERSION, dataset=dataset, latest=None), samples_by_role


def validate_state(state, completed_iterations, model_sha256=None):
    if state is None:
        return
    dataset = state["dataset"]
    if state["version"] != VERSION or dataset["version"] != VERSION:
        raise ValueError("诊断版本不兼容")
    count = dataset["requested_games"]
    if type(count) is not int or count < 1 or dataset["created_iteration"] >= completed_iterations:
        raise ValueError("诊断集创建轮次/局数与检查点不一致")
    source = dataset["source"]
    if source["iteration"] != dataset["created_iteration"]:
        raise ValueError("诊断集来源轮次不一致")
    rr.run_dir(".", source["run_id"])
    rr.run_dir(".", source["attempt_id"])
    if set(dataset["sets"]) != {"heldout", "training_reference"}:
        raise ValueError("诊断集角色不一致")
    groups = []
    for role, record in dataset["sets"].items():
        path = dataset_path(".", record["path"])
        indices = record["game_indices"]
        if (path.parent.name != source["attempt_id"] or path.name != role + ".pkl.gz" or
                record["games"] != count or len(indices) != count or len(set(indices)) != count or
                any(type(i) is not int or i < 0 for i in indices) or record["positions"] < count):
            raise ValueError("诊断集整局清单不一致")
        groups.append(set(indices))
    if groups[0] & groups[1]:
        raise ValueError("留出与训练参照对局重叠")
    report = state.get("latest")
    if report is None:
        raise ValueError("已提交固定诊断缺少报告")
    if (report["version"] != VERSION or report["completed_iterations"] != completed_iterations or
            report["dataset_sha256"] != rr.record_hash(dataset) or
            report["provenance"]["iteration"] + 1 != completed_iterations or
            (model_sha256 is not None and report["evaluated_model_sha256"] != model_sha256)):
        raise ValueError("诊断报告与检查点不一致")


def load_fixed_sets(state, data_dir, load_samples, ruleset):
    samples_by_role = {}
    dataset = state["dataset"]
    for role, record in dataset["sets"].items():
        path = dataset_path(data_dir, record["path"])
        if rr.file_record(path)["sha256"] != record["sha256"]:
            raise ValueError("固定诊断数据内容改变: " + str(path))
        samples, meta = load_samples(str(path))
        lengths = meta.get("game_lengths", [])
        if (meta.get("ruleset_version") != ruleset or meta.get("role") != role or
                meta.get("provenance") != dataset["source"] or
                meta.get("game_indices") != record["game_indices"] or
                len(samples) != record["positions"] or len(lengths) != record["games"] or
                any(type(n) is not int or n < 1 for n in lengths) or sum(lengths) != len(samples)):
            raise ValueError("固定诊断数据来源/局边界不一致")
        samples_by_role[role] = samples
    return samples_by_role


def evaluate_targets(net, samples, device, encode, batch_size=128):
    """Position-weighted eval-mode metrics; PCR policy_weight=0 only contributes value."""
    if not samples or batch_size < 1:
        raise ValueError("诊断样本和 batch_size 必须为正")
    modes = [(module, module.training) for module in net.modules()]
    ce = entropy = policy_weight = mse = pred_sum = target_sum = target_sq = bias = 0.0
    bins = [dict(count=0, predicted=0.0, target=0.0) for _ in range(10)]
    policy_positions = 0
    try:
        net.eval()
        with torch.inference_mode():
            for start in range(0, len(samples), batch_size):
                batch = samples[start:start + batch_size]
                triples = [encode(sample) for sample in batch]
                states, policies, targets = [torch.stack(items).to(device) for items in zip(*triples)]
                logits, values = net(states)
                log_probs = F.log_softmax(logits, dim=1).double()
                policies, targets, values = policies.double(), targets.double(), values.double()
                weights = torch.tensor([float(getattr(s, "policy_weight", 1.0)) for s in batch],
                                       device=device, dtype=torch.float64)
                if not torch.isfinite(weights).all() or (weights < 0).any():
                    raise ValueError("无效策略诊断权重")
                ce += float((-(policies * log_probs).sum(1) * weights).sum())
                entropy += float((-(policies * policies.clamp_min(1e-30).log()).sum(1) * weights).sum())
                policy_weight += float(weights.sum())
                policy_positions += int((weights > 0).sum())
                mse += float(((values - targets) ** 2).sum())
                pred_sum += float(values.sum())
                target_sum += float(targets.sum())
                target_sq += float((targets ** 2).sum())
                bias += float((values - targets).sum())
                indices = (((values + 1) * 5).floor().long()).clamp(0, 9)
                for i in range(10):
                    mask = indices == i
                    bins[i]["count"] += int(mask.sum())
                    bins[i]["predicted"] += float(values[mask].sum())
                    bins[i]["target"] += float(targets[mask].sum())
    finally:
        for module, training in modes:
            module.training = training
    count = len(samples)
    calibration = []
    calibration_mse = 0.0
    for i, item in enumerate(bins):
        n = item["count"]
        predicted = item["predicted"] / n if n else None
        target = item["target"] / n if n else None
        if n:
            calibration_mse += n * (predicted - target) ** 2 / count
        calibration.append(dict(lower=-1 + i * .2, upper=-1 + (i + 1) * .2,
                                count=n, prediction_mean=predicted, target_mean=target))
    metrics = dict(positions=count, policy_positions=policy_positions, policy_weight=policy_weight,
                   policy_ce=ce / policy_weight if policy_weight else None,
                   target_entropy=entropy / policy_weight if policy_weight else None,
                   policy_kl=max(0.0, (ce - entropy) / policy_weight) if policy_weight else None,
                   value_mse=mse / count, value_prediction_mean=pred_sum / count,
                   value_target_mean=target_sum / count, value_bias=bias / count,
                   value_zero_baseline_mse=target_sq / count,
                   value_constant_baseline_mse=max(0.0, target_sq / count - (target_sum / count) ** 2),
                   calibration_mse=calibration_mse, calibration=calibration)
    if any(isinstance(v, float) and not math.isfinite(v) for v in metrics.values()):
        raise ValueError("诊断指标非有限值，拒绝提交")
    return metrics


def evaluate_sets(net, samples_by_role, device, encode, batch_size):
    return {role: evaluate_targets(net, samples, device, encode, batch_size)
            for role, samples in samples_by_role.items()}


def make_report(state, attempt, model_sha256, before, after, steps, batch_size,
                buffer_size, added_samples):
    def gap(metric):
        heldout = after["heldout"][metric]
        reference = after["training_reference"][metric]
        return heldout - reference if heldout is not None and reference is not None else None
    return dict(version=VERSION, provenance=dict(attempt), completed_iterations=attempt["iteration"] + 1,
                evaluated_model_sha256=model_sha256, dataset_sha256=rr.record_hash(state["dataset"]),
                before=before, after=after,
                heldout_minus_reference=dict(policy_kl=gap("policy_kl"), value_mse=gap("value_mse")),
                training=dict(steps=steps, batch_size=batch_size, buffer_size=buffer_size,
                              added_augmented_samples=added_samples, update_draws=steps * batch_size,
                              expected_draws_per_buffer_sample=steps * batch_size / buffer_size if buffer_size else 0.0,
                              steady_state_expected_lifetime_draws=steps * batch_size / added_samples if added_samples else None),
                strength_evaluated=False)


def write_report(data_dir, report):
    attempt_id = report["provenance"]["attempt_id"]
    rr.run_dir(".", attempt_id)
    path = Path(data_dir) / "diagnostics" / "reports" / (attempt_id + ".json")
    rr.write_record(path, report)  # Can recover the embedded report after a committed write failure.
    return path
