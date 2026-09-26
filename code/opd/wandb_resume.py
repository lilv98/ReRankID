"""Fail-closed W&B history alignment for resumable 8B OPD training.

verl writes a checkpoint before it logs the corresponding training row.  A
normal interruption after a saved checkpoint can therefore leave W&B ahead of
the local recovery point.  W&B's ordinary ``resume=allow`` continues from that
remote tail and silently mixes metrics from two model trajectories.

This module audits the exact remote prefix and, on resume, asks W&B to rewind
the existing run to the locally committed checkpoint step.  If the checkpoint
row itself never reached W&B, recovery stops because that metric row cannot be
reconstructed exactly from the checkpoint.
"""

from __future__ import annotations

import argparse
from collections import Counter
import os
import shlex
import time

from opd import contract


WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "anonymous")
WANDB_PROJECT = "ReRankID"
RESUME_POLICY = "checkpoint-rewind-v1"
RESUME_POLICY_ENV = "RERANKID_OPD_WANDB_RESUME_POLICY"
HISTORY_KEYS = ("_step", "training/global_step")


class WandbResumeError(RuntimeError):
    """Raised when local checkpoint and remote W&B history cannot align."""


def _strict_step(value, label):
    if not isinstance(value, int) or isinstance(value, bool):
        raise WandbResumeError(f"{label} must be an integer, found {value!r}")
    if value < 1:
        raise WandbResumeError(f"{label} must be positive, found {value}")
    return value


def read_local_marker(run_dir, *, total_steps, save_freq):
    """Read a scheduled, non-final local recovery marker, if one exists."""

    marker_path = os.path.join(
        os.path.abspath(run_dir), "latest_checkpointed_iteration.txt"
    )
    if not os.path.exists(marker_path):
        return None
    try:
        with open(marker_path, encoding="utf-8") as handle:
            raw = handle.read().strip()
        marker = int(raw)
    except (OSError, ValueError) as exc:
        raise WandbResumeError(
            f"cannot read local checkpoint marker {marker_path}"
        ) from exc
    if marker <= 0 or marker >= total_steps or marker % save_freq:
        raise WandbResumeError(
            f"local checkpoint marker {marker} is not a resumable "
            f"{save_freq}-step checkpoint below {total_steps}"
        )
    return marker


def _history_pairs(rows):
    pairs = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise WandbResumeError(f"W&B history row {index} is not a mapping")
        remote_step = _strict_step(row.get("_step"), f"W&B row {index} _step")
        training_step = row.get("training/global_step")
        if training_step is not None:
            training_step = _strict_step(
                training_step,
                f"W&B row {index} training/global_step",
            )
        pairs.append((remote_step, training_step))
    return pairs


def validate_remote_prefix(rows, marker):
    """Require exactly one aligned training row for every step through marker."""

    marker = _strict_step(marker, "local checkpoint marker")
    prefix = []
    for remote_step, training_step in _history_pairs(rows):
        if remote_step > marker:
            continue
        if training_step is None:
            raise WandbResumeError(
                f"W&B step {remote_step} has no training/global_step"
            )
        if training_step != remote_step:
            raise WandbResumeError(
                f"W&B step {remote_step} carries training/global_step={training_step}"
            )
        prefix.append(remote_step)
    counts = Counter(prefix)
    duplicates = sorted(step for step, count in counts.items() if count != 1)
    missing = sorted(set(range(1, marker + 1)) - set(prefix))
    unexpected = sorted(set(prefix) - set(range(1, marker + 1)))
    if len(prefix) != marker or duplicates or missing or unexpected:
        raise WandbResumeError(
            "W&B history does not contain one exact row per committed "
            f"step through {marker}: rows={len(prefix)}, "
            f"duplicates={duplicates}, missing={missing}, "
            f"unexpected={unexpected}"
        )
    return marker


def validate_final_history(rows, total_steps):
    """Require one aligned W&B row for every formal training step."""

    total_steps = _strict_step(total_steps, "formal total steps")
    pairs = _history_pairs(rows)
    for remote_step, training_step in pairs:
        if training_step is None:
            raise WandbResumeError(
                f"W&B step {remote_step} has no training/global_step"
            )
        if training_step != remote_step:
            raise WandbResumeError(
                f"W&B step {remote_step} carries training/global_step={training_step}"
            )
    steps = [remote_step for remote_step, _ in pairs]
    counts = Counter(steps)
    duplicates = sorted(step for step, count in counts.items() if count != 1)
    expected = set(range(1, total_steps + 1))
    missing = sorted(expected - set(steps))
    unexpected = sorted(set(steps) - expected)
    if len(steps) != total_steps or duplicates or missing or unexpected:
        raise WandbResumeError(
            "final W&B history is not exactly one row per formal step: "
            f"rows={len(steps)}, duplicates={duplicates}, "
            f"missing={missing}, unexpected={unexpected}"
        )
    return total_steps


def _is_missing_run(error):
    """Recognize only W&B's explicit missing-run response, not network errors."""

    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        message = str(current)
        if message.startswith("Could not find run ") or "HTTP 404" in message:
            return True
        response = getattr(current, "response", None)
        if getattr(response, "status_code", None) == 404:
            return True
        pending.extend(
            (
                getattr(current, "exc", None),
                getattr(current, "__cause__", None),
                getattr(current, "__context__", None),
            )
        )
    return False


def fetch_remote_history(api, *, entity, project, run_id):
    """Fetch complete uncached scalar history, returning ``None`` if absent."""

    path = f"{entity}/{project}/{run_id}"
    try:
        run = api.run(path)
    except Exception as exc:
        if _is_missing_run(exc):
            return None
        raise WandbResumeError(f"cannot query W&B run {path}: {exc}") from exc
    identity = {
        "entity": getattr(run, "entity", None),
        "project": getattr(run, "project", None),
        "id": getattr(run, "id", None),
    }
    expected = {"entity": entity, "project": project, "id": run_id}
    if identity != expected:
        raise WandbResumeError(
            f"W&B returned run identity {identity!r}, expected {expected!r}"
        )
    try:
        return list(
            run.scan_history(
                keys=list(HISTORY_KEYS),
                page_size=500,
                use_cache=False,
            )
        )
    except Exception as exc:
        raise WandbResumeError(f"cannot scan W&B history for {path}: {exc}") from exc


def prepare_resume(api, *, entity, project, run_id, marker):
    """Return a fresh or rewind plan after auditing remote history."""

    rows = fetch_remote_history(api, entity=entity, project=project, run_id=run_id)
    if marker is None:
        if rows:
            raise WandbResumeError(
                "W&B already has training history but the local run has no "
                "recoverable checkpoint; refusing to mix trajectories"
            )
        return {"mode": "fresh", "marker": None, "remote_rows": 0}
    if rows is None:
        raise WandbResumeError(
            f"local checkpoint {marker} exists but W&B run is absent"
        )
    validate_remote_prefix(rows, marker)
    return {
        "mode": "rewind",
        "marker": marker,
        "remote_rows": len(rows),
    }


def shell_exports(plan, *, entity, project, run_id):
    """Render the audited plan as shell assignments for the launcher."""

    values = {
        "WANDB_ENTITY": entity,
        "WANDB_PROJECT": project,
        "WANDB_RUN_ID": run_id,
    }
    lines = [f"export {key}={shlex.quote(value)}" for key, value in values.items()]
    if plan["mode"] == "fresh":
        lines.extend(("unset WANDB_RESUME_FROM", "export WANDB_RESUME=allow"))
    elif plan["mode"] == "rewind":
        marker = _strict_step(plan.get("marker"), "rewind marker")
        lines.extend(
            (
                "unset WANDB_RESUME",
                "export WANDB_RESUME_FROM=" + shlex.quote(f"{run_id}?_step={marker}"),
            )
        )
    else:
        raise WandbResumeError(f"unknown W&B resume plan mode {plan.get('mode')!r}")
    return "\n".join(lines)


def _retry(operation, *, attempts, delay_seconds):
    if attempts < 1:
        raise WandbResumeError("W&B audit attempts must be positive")
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except WandbResumeError as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(delay_seconds)
    raise last_error


def _api():
    try:
        import wandb
    except ImportError as exc:
        raise WandbResumeError("wandb is not installed") from exc
    if wandb.__version__ != "0.28.0":
        raise WandbResumeError(
            "checkpoint rewind requires the audited wandb 0.28.0 client, "
            f"found {wandb.__version__}"
        )
    return wandb.Api(timeout=30)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_remote_arguments(command, *, attempts):
        command.add_argument("--run-id", required=True)
        command.add_argument("--entity", default=WANDB_ENTITY)
        command.add_argument("--project", default=WANDB_PROJECT)
        command.add_argument(
            "--total-steps", type=int, default=contract.TOTAL_TRAINING_STEPS
        )
        command.add_argument("--attempts", type=int, default=attempts)
        command.add_argument("--delay-seconds", type=float, default=5.0)

    prepare = subparsers.add_parser("prepare")
    add_remote_arguments(prepare, attempts=6)
    prepare.add_argument("--run-dir", required=True)
    prepare.add_argument("--save-freq", type=int, default=contract.SAVE_FREQ)

    verify = subparsers.add_parser("verify-final")
    add_remote_arguments(verify, attempts=12)

    args = parser.parse_args()
    if args.entity != WANDB_ENTITY or args.project != WANDB_PROJECT:
        raise WandbResumeError(
            "8B formal OPD must use the frozen W&B entity and project"
        )
    api = _api()
    if args.command == "prepare":
        marker = read_local_marker(
            args.run_dir,
            total_steps=args.total_steps,
            save_freq=args.save_freq,
        )

        def operation():
            return prepare_resume(
                api,
                entity=args.entity,
                project=args.project,
                run_id=args.run_id,
                marker=marker,
            )

        # A fresh absent run is immediately valid.  Retry resume checks because
        # a just-finished W&B process can be briefly behind the local marker.
        attempts = args.attempts if marker is not None else 1
        plan = _retry(
            operation,
            attempts=attempts,
            delay_seconds=args.delay_seconds,
        )
        print(
            shell_exports(
                plan,
                entity=args.entity,
                project=args.project,
                run_id=args.run_id,
            )
        )
        return

    def operation():
        rows = fetch_remote_history(
            api,
            entity=args.entity,
            project=args.project,
            run_id=args.run_id,
        )
        if rows is None:
            raise WandbResumeError("final W&B run is absent")
        validate_final_history(rows, args.total_steps)
        return len(rows)

    rows = _retry(
        operation,
        attempts=args.attempts,
        delay_seconds=args.delay_seconds,
    )
    print(
        f"W&B final history verified: {rows} unique aligned rows "
        f"in {args.entity}/{args.project}/{args.run_id}"
    )


if __name__ == "__main__":
    main()
