"""Tiny CUDA/NVTX compatibility target; no game, checkpoint, RNG or optimizer."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import multiprocessing as mp
import os
from pathlib import Path
import threading

CAPTURE_RANGE = "jieqi_native_window"
TORCH = None
INPUTS = None


def write_event(directory, event, **data):
    with (directory / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event=event, pid=os.getpid(), thread_id=threading.get_native_id(), **data)) + "\n")


def init_cuda():
    global TORCH, INPUTS
    import torch
    TORCH = torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    INPUTS = (torch.ones((128, 128), device="cuda"), torch.ones((1, 4, 32, 32), device="cuda"),
              torch.ones((4, 4, 3, 3), device="cuda"))
    tiny_work(False)


@contextmanager
def nvtx(name):
    TORCH.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        TORCH.cuda.nvtx.range_pop()


def tiny_work(labelled):
    matrix, image, kernel = INPUTS
    for _ in range(8):
        if labelled:
            with nvtx("jieqi/compat/forward"):
                product = matrix @ matrix
                convolution = TORCH.nn.functional.conv2d(image, kernel)
        else:
            product = matrix @ matrix
            convolution = TORCH.nn.functional.conv2d(image, kernel)
    # Synthetic test only: validate actual GPU work and a synchronous readback.
    with nvtx("jieqi/compat/readback"):
        product_cpu, convolution_cpu = product.cpu(), convolution.cpu()
    if not TORCH.equal(product_cpu, TORCH.full_like(product_cpu, 128.0)):
        raise AssertionError("matrix result changed")
    if not TORCH.equal(convolution_cpu, TORCH.full_like(convolution_cpu, 36.0)):
        raise AssertionError("convolution result changed")
    return dict(pid=os.getpid(), thread_id=threading.get_native_id(), gpu=TORCH.cuda.get_device_name(),
                torch=str(TORCH.__version__), cuda=TORCH.version.cuda, dtype=str(matrix.dtype), exact=True,
                tf32_matmul=TORCH.backends.cuda.matmul.allow_tf32, tf32_cudnn=TORCH.backends.cudnn.allow_tf32)


def worker(connection):
    init_cuda()
    connection.send(dict(event="warm", pid=os.getpid()))
    if connection.recv() != "run":
        raise RuntimeError("unexpected probe command")
    connection.send(dict(event="result", data=tiny_work(True)))
    if connection.recv() != "exit":
        raise RuntimeError("unexpected probe shutdown")
    connection.close()


def main():
    global TORCH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("marker", "single", "spawn"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    pilots = Path(__file__).resolve().parents[1] / "data/pilots"
    directory = args.output_dir.resolve()
    if directory.exists() or pilots.resolve() not in directory.parents:
        parser.error("require a new directory under this workspace data/pilots")
    directory.mkdir()
    write_event(directory, "start", mode=args.mode)
    if args.mode == "marker":
        (directory / "python_started.marker").touch(exist_ok=False)
        rows = []
    elif args.mode == "single":
        init_cuda()
        write_event(directory, "warm")
        with nvtx(CAPTURE_RANGE):
            rows = [tiny_work(True)]
        write_event(directory, "capture_window_closed")
    else:
        # Two explicit handshakes guarantee both warmed children execute inside the parent trigger.
        import torch
        TORCH = torch
        context = mp.get_context("spawn")
        pairs = [context.Pipe() for _ in range(2)]
        processes = [context.Process(target=worker, args=(child,)) for _, child in pairs]
        for process, (_, child) in zip(processes, pairs):
            process.start()
            child.close()
        for parent, _ in pairs:
            warm = parent.recv()
            if warm["event"] != "warm":
                raise RuntimeError("unexpected warmup result")
            write_event(directory, "warm", worker_pid=warm["pid"])
        with nvtx(CAPTURE_RANGE):
            for parent, _ in pairs:
                parent.send("run")
            results = [parent.recv() for parent, _ in pairs]
            if any(result["event"] != "result" for result in results):
                raise RuntimeError("unexpected worker result")
            rows = [result["data"] for result in results]
        write_event(directory, "capture_window_closed")
        for parent, _ in pairs:
            parent.send("exit")
            parent.close()
        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError("worker did not exit normally")
        write_event(directory, "workers_closed")
    result = dict(kind="native_runtime_probe", mode=args.mode, pid=os.getpid(), rows=rows,
                  models_loaded=False, training_started=False, note="Synthetic compatibility only; not a speed benchmark")
    with (directory / "result.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    write_event(directory, "complete")


if __name__ == "__main__":
    main()
