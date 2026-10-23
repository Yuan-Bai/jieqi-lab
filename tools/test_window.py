"""滑动窗口（replay buffer 新鲜度）离线验证脚本。

验证假设：把训练数据窗口从"最近 ~140 轮"缩小到"最近 ~10 轮"，是否能
让价值头更好地跟上当前策略（在**未来**自博弈数据上测得更高的 value R²）。

做法（不跑自博弈，直接用已有 data/selfplay 数据，成本低）：
  - 取锚点 T（默认 479，用 checkpoint 对应时刻）。
  - 对每个窗口 W：用 [T-W+1, T] 的样本（含镜像）作为 buffer，
    从同一 checkpoint 出发训练固定步数（配置与 azero.train_step 一致）。
  - 在**窗口外、时间更晚**的 [T+1, T+F] 数据上评估 value 校准与 policy 拟合。
  - 同时打印该窗口的复用次数（steps*batch/每轮新增）与废弃陈旧度。

用法：
  python tools/test_window.py --model data/models/best.pt --anchor 479 \
      --windows 10,20,45,143 --steps 800 --batch 512 \
      --future 480:534 --future-cap 30000 --device cuda

注意：若主训练正在运行，本脚本会与其争抢 GPU（实测 ~2x 变慢）。
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import azero  # noqa: E402

ITER_RE = re.compile(r"iter_\d+_\d+_(\d{4})\.pkl\.gz$")


def list_iters(selfplay_dir: str) -> list[tuple[int, str]]:
    out = []
    for path in glob.glob(os.path.join(selfplay_dir, "*.pkl.gz")):
        m = ITER_RE.search(os.path.basename(path))
        if m:
            out.append((int(m.group(1)), path))
    out.sort()
    return out


def build_window(files: list[tuple[int, str]], lo: int, hi: int, mirror: bool):
    buf = []
    for it, path in files:
        if lo <= it <= hi:
            samples, _ = azero.load_compact_payload(path)
            buf.extend(samples)
            if mirror:
                buf.extend(azero.augment_mirror(samples))
    return buf


def load_future(files: list[tuple[int, str]], lo: int, hi: int, cap: int):
    samples = []
    for it, path in files:
        if lo <= it <= hi:
            s, _ = azero.load_compact_payload(path)
            samples.extend(s)
    if cap and len(samples) > cap:
        idx = np.random.RandomState(7).choice(len(samples), cap, replace=False)
        samples = [samples[i] for i in idx]
    return samples


@torch.no_grad()
def evaluate(net, samples, device, batch=512, cap=None):
    net.eval()
    if cap and len(samples) > cap:
        idx = np.random.RandomState(11).choice(len(samples), cap, replace=False)
        samples = [samples[i] for i in idx]
    mse_sum = 0.0
    ce_sum = 0.0
    top1 = 0
    n = 0
    p_vals, t_vals = [], []
    for i in range(0, len(samples), batch):
        chunk = samples[i : i + batch]
        tensors = [azero.compact_to_tensors(s) for s in chunk]
        states = torch.stack([t[0] for t in tensors]).to(device)
        pol = torch.stack([t[1] for t in tensors]).to(device)
        val = torch.tensor([s.value for s in chunk], dtype=torch.float32, device=device)
        logits, values = net(states)
        probs = torch.softmax(logits, dim=1)
        ce = -(pol * F.log_softmax(logits, dim=1)).sum(1)
        ce_sum += float(ce.sum())
        top1 += int((probs.argmax(1) == pol.argmax(1)).sum())
        mse_sum += float(((values - val) ** 2).sum())
        p_vals.append(values.detach().cpu())
        t_vals.append(val.detach().cpu())
        n += len(chunk)
    p = torch.cat(p_vals).numpy()
    t = torch.cat(t_vals).numpy()
    var = float(((t - t.mean()) ** 2).mean())
    mse = mse_sum / n
    corr = float(np.corrcoef(p, t)[0, 1]) if p.std() > 0 else float("nan")
    return {
        "n": n,
        "mse": mse,
        "var": var,
        "r2": 1 - mse / var if var > 0 else float("nan"),
        "corr": corr,
        "pstd": float(p.std()),
        "tstd": float(t.std()),
        "draw": float((t == 0).mean()),
        "ce": ce_sum / n,
        "top1": top1 / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join("data", "models", "best.pt"))
    ap.add_argument("--anchor", type=int, default=479)
    ap.add_argument("--windows", default="10,20,45,143")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--future", default="480:534")
    ap.add_argument("--future-cap", type=int, default=30000)
    ap.add_argument("--selfplay-dir", default=os.path.join("data", "selfplay"))
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    windows = [int(x) for x in args.windows.split(",") if x.strip()]
    f_lo, f_hi = (int(x) for x in args.future.split(":"))
    T = args.anchor
    mirror = not args.no_mirror

    files = list_iters(args.selfplay_dir)
    print(f"selfplay iters available: {files[0][0]}..{files[-1][0]}  device={device}")
    print(f"anchor T={T}  windows={windows}  steps={args.steps}  future=[{f_lo},{f_hi}] cap={args.future_cap}")

    future = load_future(files, f_lo, f_hi, args.future_cap)
    print(f"future eval set: n={len(future)}  draw%={np.mean([s.value == 0 for s in future])*100:.1f}")

    start_state = torch.load(args.model, map_location=device, weights_only=True)

    def fresh_net():
        net = azero.PolicyValueNet().to(device)
        azero.load_state_dict_padded(net, start_state)
        return net

    hdr = f"{'W':>4} {'bufN':>8} {'pass':>5} {'reuse':>6} | {'fvR2':>6} {'fvcorr':>6} {'fvMSE':>6} {'fvps':>5} | {'fCE':>6} {'fT1':>5} | {'wR2':>6} {'wMSE':>6}"
    print("\n" + hdr)
    print("-" * len(hdr))

    base = fresh_net()
    m0 = evaluate(base, future, device)
    print(f"{'base':>4} {'-':>8} {'-':>5} {'-':>6} | {m0['r2']:>6.3f} {m0['corr']:>6.3f} {m0['mse']:>6.3f} {m0['pstd']:>5.2f} | {m0['ce']:>6.3f} {m0['top1']*100:>5.1f} |", flush=True)

    for W in windows:
        lo = T - W + 1
        t0 = time.time()
        buf = build_window(files, lo, T, mirror)
        if len(buf) < args.batch:
            print(f"{W:>4}  buffer too small ({len(buf)})")
            continue
        load_s = time.time() - t0

        net = fresh_net()
        net.train()
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        rng = random.Random(1234)
        t1 = time.time()
        for _ in range(args.steps):
            batch = rng.sample(buf, args.batch)
            azero.train_step(net, opt, batch, device, args.grad_clip)
        train_s = time.time() - t1

        fm = evaluate(net, future, device)
        wm = evaluate(net, buf, device, cap=30000)  # in-window fit (sampled)
        passes = args.steps * args.batch / len(buf)
        new_per_iter = len(buf) / W
        reuse = args.steps * args.batch / new_per_iter
        print(
            f"{W:>4} {len(buf):>8} {passes:>5.2f} {reuse:>6.1f} | "
            f"{fm['r2']:>6.3f} {fm['corr']:>6.3f} {fm['mse']:>6.3f} {fm['pstd']:>5.2f} | "
            f"{fm['ce']:>6.3f} {fm['top1']*100:>5.1f} | {wm['r2']:>6.3f} {wm['mse']:>6.3f}"
            f"   [load {load_s:.0f}s train {train_s:.0f}s]",
            flush=True,
        )

    print("\n列说明: W=窗口轮数, bufN=窗口样本数(含镜像), pass=每步训练对buffer的passes,")
    print("        reuse=单样本终身复用次数, fv*/*=未来集指标, w*=窗口内拟合, fCE/fT1=未来policy CE/top1")
    print("判读: 若小 W 的 fvR2 明显更高 -> 支持缩小窗口；若随 W 单调升高 -> 数据量更重要。")


if __name__ == "__main__":
    main()
