"""Immutable run/attempt records. Metadata must not consume training RNG."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import jieqi as jq


RECORD_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parent


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def record_hash(value: dict) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def file_record(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return dict(path=str(path), sha256=digest, bytes=path.stat().st_size)


def model_hash(net) -> str | None:
    """Hash actual tensors (including BN buffers), independent of torch.save framing."""
    if net is None:
        return None
    digest = hashlib.sha256()
    for name, tensor in sorted(net.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(json_bytes(dict(name=name, dtype=str(tensor.dtype), shape=list(tensor.shape))))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _search_runtime() -> dict:
    # Avoid importing azero here: it imports this module before defining its nodes.
    for name in ("__main__", "__mp_main__", "azero"):
        module = sys.modules.get(name)
        node = getattr(module, "PUCTNode", None)
        if (module is not None and hasattr(module, "SEARCH_BACKEND_NATIVE") and node is not None
                and node.__module__ == module.__name__):
            return dict(search_backend=module.SEARCH_BACKEND,
                        search_backend_error=module.SEARCH_BACKEND_ERROR,
                        search_backend_native=module.SEARCH_BACKEND_NATIVE)
    return dict(search_backend=None, search_backend_error=None, search_backend_native=None)


def code_record(root: Path = PROJECT_ROOT) -> dict:
    # Include untracked runtime sources: HEAD alone cannot identify a dirty checkout.
    paths = [p for p in root.glob("*.py") if not p.name.startswith("test_")]
    paths += list((root / "tools").rglob("*.py")) + list((root / "tools").glob("*.json"))
    paths += list((root / "native").glob("*.pyx"))
    paths += list(root.glob("_jieqi_rules*.pyd")) + list(root.glob("_jieqi_rules*.so"))
    paths += list(root.glob("_jieqi_search*.pyd")) + list(root.glob("_jieqi_search*.so"))
    paths += list(root.glob("_jieqi_legal*.pyd")) + list(root.glob("_jieqi_legal*.so"))
    paths += [p for p in (root / "requirements.txt", root / "pyproject.toml") if p.exists()]
    hashes = {p.relative_to(root).as_posix(): file_record(p)["sha256"] for p in sorted(paths)}
    for kind, loaded in (("rules", jq.RULES_BACKEND_NATIVE),
                         ("legal", jq.LEGAL_BACKEND_NATIVE),
                         ("search", _search_runtime()["search_backend_native"])):
        if loaded is None:
            continue
        binary = Path(loaded["path"])
        actual = file_record(binary)
        if actual["sha256"] != loaded["sha256"]:
            raise RuntimeError(f"loaded native {kind} file changed; restart Python before recording a run")
        try:
            key = binary.relative_to(root.resolve()).as_posix()
        except ValueError:
            key = binary.as_posix()
        hashes[key] = actual["sha256"]
    git = dict(head=None, dirty=None, status=None)
    try:
        def git_output(*args):
            return subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                  text=True, encoding="utf-8", errors="replace", check=True,
                                  timeout=10).stdout.strip()
        git["head"] = git_output("rev-parse", "HEAD")
        git["status"] = git_output("status", "--porcelain", "--untracked-files=normal")
        git["dirty"] = bool(git["status"])
    except (OSError, subprocess.SubprocessError):
        git["available"] = False
    return dict(sha256=record_hash(hashes), files=hashes, git=git)


def new_id() -> str:
    return uuid.uuid4().hex  # OS entropy; does not advance Python/NumPy/torch RNG.


def run_dir(data_dir: str | Path, run_id: str) -> Path:
    if not isinstance(run_id, str) or re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
        raise ValueError("invalid run/attempt ID")
    return Path(data_dir) / "runs" / run_id


def write_record(path: Path, value: dict) -> None:
    """Atomic and exclusive. Existing identical records are idempotent, never replaced."""
    payload = json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".record-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(f"immutable run record differs: {path}")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def create_run(data_dir: str | Path, kind: str, config: dict, network: dict | None,
               versions: dict, parent: dict, device, start_state: dict | None = None,
               replay_files: list[dict] | None = None) -> dict:
    manifest = dict(record_version=RECORD_VERSION, run_id=new_id(), kind=kind,
                    started_at=datetime.now(timezone.utc).isoformat(), config=config,
                    network=network, versions=versions, code=code_record(), parent=parent,
                    start_state=start_state, initial_replay_files=replay_files or [],
                    runtime=dict(python=platform.python_version(), torch=str(torch.__version__),
                                 numpy=str(np.__version__), device=str(device),
                                 cuda_build=torch.version.cuda, rules_backend=jq.RULES_BACKEND,
                                 rules_backend_error=jq.RULES_BACKEND_ERROR,
                                 rules_backend_native=jq.RULES_BACKEND_NATIVE,
                                 legal_backend=jq.LEGAL_BACKEND, legal_backend_error=jq.LEGAL_BACKEND_ERROR,
                                 legal_backend_native=jq.LEGAL_BACKEND_NATIVE, **_search_runtime()))
    if str(device).startswith("cuda") and torch.cuda.is_available():
        manifest["runtime"]["gpu"] = torch.cuda.get_device_name(device)
    # Freeze caller-owned lists/dicts; a later replay append must not change this run's identity.
    manifest = json.loads(json_bytes(manifest))
    write_record(run_dir(data_dir, manifest["run_id"]) / "manifest.json", manifest)
    return manifest


def source_ref(manifest: dict) -> dict:
    return dict(run_id=manifest["run_id"], manifest_sha256=record_hash(manifest))


def start_attempt(data_dir: str | Path, manifest: dict, iteration: int,
                  generating_model_sha256: str, parent_checkpoint: dict) -> dict:
    attempt = dict(**source_ref(manifest), attempt_id=new_id(), iteration=iteration,
                   generating_model_sha256=generating_model_sha256,
                   parent_checkpoint=parent_checkpoint)
    write_record(run_dir(data_dir, manifest["run_id"]) / "attempts" /
                 (attempt["attempt_id"] + ".json"), attempt)
    return attempt


def validate_lineage(lineage: dict) -> None:
    manifest = lineage["manifest"]
    run_dir(".", manifest["run_id"])
    if manifest["record_version"] != RECORD_VERSION:
        raise ValueError("incompatible run record version")
    if lineage["manifest_sha256"] != record_hash(manifest):
        raise ValueError("run manifest hash mismatch")
    attempt = lineage.get("attempt")
    if attempt is not None:
        run_dir(".", attempt["attempt_id"])
        if any(attempt[key] != value for key, value in source_ref(manifest).items()):
            raise ValueError("attempt/manifest mismatch")
        if attempt["iteration"] + 1 != lineage["completed_iterations"]:
            raise ValueError("attempt/completed iteration mismatch")


def validate_training_lineage(lineage: dict, config: dict, network: dict, versions: dict,
                              completed_iterations: int, replay_files: list[dict]) -> None:
    validate_lineage(lineage)
    manifest = lineage["manifest"]
    for key, expected in dict(kind="training", config=config, network=network, versions=versions).items():
        if manifest[key] != expected:
            raise ValueError(f"checkpoint/run {key} mismatch")
    if lineage["completed_iterations"] != completed_iterations:
        raise ValueError("checkpoint/run iteration mismatch")
    if lineage.get("data_file") is not None and lineage["data_file"] not in replay_files:
        raise ValueError("checkpoint/run data file mismatch")


def commit_record(data_dir: str | Path, lineage: dict, checkpoint: dict) -> None:
    """Called only after atomic checkpoint commit; also repairs a missing marker on resume."""
    validate_lineage(lineage)
    manifest, attempt = lineage["manifest"], lineage.get("attempt")
    folder = run_dir(data_dir, manifest["run_id"])
    write_record(folder / "manifest.json", manifest)
    if attempt is not None:
        write_record(folder / "attempts" / (attempt["attempt_id"] + ".json"), attempt)
    name = attempt["attempt_id"] if attempt else "bootstrap"
    marker = dict(**source_ref(manifest), attempt_id=attempt["attempt_id"] if attempt else None,
                  status="committed", completed_iterations=lineage["completed_iterations"],
                  model_sha256=lineage["model_sha256"], data_file=lineage.get("data_file"),
                  generating_model_sha256=attempt["generating_model_sha256"] if attempt else None,
                  checkpoint_sha256=checkpoint["sha256"])
    write_record(folder / "commits" / (name + ".json"), marker)


def attempt_status(data_dir: str | Path, source: dict, checkpoint_lineage: dict | None = None) -> str:
    """Missing marker is unconfirmed, not proof that a round never committed."""
    folder = run_dir(data_dir, source["run_id"])
    run_dir(".", source["attempt_id"])
    marker_path = folder / "commits" / (source["attempt_id"] + ".json")
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if any(marker[key] != source[key] for key in ("run_id", "attempt_id", "manifest_sha256")):
            raise ValueError("commit/source mismatch")
        if "generating_model_sha256" in source and marker["generating_model_sha256"] != source["generating_model_sha256"]:
            raise ValueError("commit/generating model mismatch")
        if "evaluated_model_sha256" in source and marker["model_sha256"] != source["evaluated_model_sha256"]:
            raise ValueError("commit/evaluated model mismatch")
        return "committed"
    if checkpoint_lineage is not None:
        validate_lineage(checkpoint_lineage)
        attempt = checkpoint_lineage.get("attempt")
        if attempt and all(attempt[key] == source[key] for key in ("run_id", "attempt_id", "manifest_sha256")):
            return "committed"
    return "unconfirmed"
