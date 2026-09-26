"""Fail-closed online export and retention for verl OPD checkpoints.

The archive is deliberately independent of verl's in-memory
``previous_saved_paths`` bookkeeping because that list is empty after a
resume.  The filesystem and immutable compact-adapter provenance are the
source of truth.

The transaction order is:

1. verl completes its raw checkpoint and commits the latest-step marker;
2. every compact adapter through that marker is verified or synchronously
   exported;
3. only after successful publication are raw checkpoints pruned to the
   newest two; and
4. reconstructed bookkeeping is atomically written.

If export or verification fails, pruning is skipped and the exception is
allowed to fail training.

verl may remove its oldest raw immediately before writing a new one.  A
process crash during that save can therefore leave the committed marker raw
as the sole recovery point plus a newer partial directory.  This module never
adopts or deletes that partial directory.  Resume remains fail-closed until
an operator quarantines it, after which the sole marker raw is accepted.
Once a new save succeeds, the archive again requires both the new marker raw
and one previous recoverable raw.
"""

from contextlib import contextmanager
import fcntl
import json
import os
import re
import shutil
import tempfile

from opd.export_lora_checkpoint import (
    ExportError,
    _manifest_context,
    _validate_scheduled_step,
    export_checkpoint,
    inspect_source_checkpoint,
    read_checkpoint_marker,
    sha256,
    verify_compact_adapter,
)


ARCHIVE_ENV = "RERANKID_OPD_CHECKPOINT_ARCHIVE"
EXPORT_DIR_ENV = "RERANKID_OPD_EXPORT_DIR"
MERGER_TMP_ENV = "RERANKID_OPD_MERGER_TMPDIR"
RAW_KEEP_ENV = "RERANKID_OPD_RAW_KEEP"
STATE_SCHEMA_VERSION = 1
STATE_NAME = "checkpoint_archive.json"
LOCK_NAME = ".checkpoint_archive.lock"
RAW_KEEP = 2
RAW_PATTERN = re.compile(r"global_step_(\d+)$")
COMPACT_PATTERN = re.compile(r"step-(\d{6})$")
LIVE_RUNTIME_SOURCES = (
    "code/opd/checkpoint_archive.py",
    "code/opd/export_lora_checkpoint.py",
    "code/opd/verl_checkpoint_hook.py",
    "code/opd/wandb_resume.py",
    "code/opd/verl_overlay/sitecustomize.py",
)


class ArchiveError(RuntimeError):
    """A fail-closed online checkpoint archive error."""


def _config_get(config, key, default=None):
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _atomic_json(path, payload):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(
            directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def verify_live_runtime_sources(manifest, repo_root=None):
    """Re-hash online-export code against the immutable run audit."""

    if repo_root is None:
        repo_root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.realpath(__file__))
            )
        )
    try:
        sources = manifest["audit"]["runtime_sources"]
    except (KeyError, TypeError) as exc:
        raise ArchiveError(
            "run manifest has no audited runtime sources"
        ) from exc
    result = {}
    for relative in LIVE_RUNTIME_SOURCES:
        record = sources.get(relative)
        expected = record.get("sha256") if isinstance(record, dict) else None
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected
            )
        ):
            raise ArchiveError(
                f"run manifest has no valid hash for {relative}"
            )
        path = os.path.join(repo_root, relative)
        if not os.path.isfile(path) or os.path.islink(path):
            raise ArchiveError(
                f"audited online-export source is missing or linked: {path}"
            )
        actual = sha256(path)
        if actual != expected:
            raise ArchiveError(
                f"live online-export source drifted for {relative}: "
                f"{actual} != {expected}"
            )
        result[relative] = actual
    return result


class CheckpointArchive:
    """Reconcile compact adapters and retain exactly two recoverable raws."""

    def __init__(
        self,
        run_dir,
        export_dir,
        merger_temp_root=None,
        raw_keep=RAW_KEEP,
    ):
        self.run_dir = os.path.abspath(run_dir)
        self.export_dir = os.path.abspath(export_dir)
        self.merger_temp_root = (
            os.path.abspath(merger_temp_root)
            if merger_temp_root
            else None
        )
        if raw_keep != RAW_KEEP:
            raise ArchiveError(
                f"online OPD raw retention must be exactly {RAW_KEEP}, "
                f"got {raw_keep}"
            )
        self.raw_keep = raw_keep
        self.manifest_path = os.path.join(
            self.run_dir, "run_manifest.json"
        )
        self.state_path = os.path.join(self.run_dir, STATE_NAME)
        self.lock_path = os.path.join(self.run_dir, LOCK_NAME)

    @classmethod
    def from_trainer(cls, trainer):
        """Build and validate the archive configuration from RayPPOTrainer."""

        if os.environ.get(ARCHIVE_ENV) != "1":
            raise ArchiveError(
                f"online checkpoint hook requires {ARCHIVE_ENV}=1"
            )
        trainer_config = trainer.config.trainer
        run_dir = str(trainer_config.default_local_dir)
        if not os.path.isabs(run_dir):
            run_dir = os.path.join(os.getcwd(), run_dir)
        default_hdfs = _config_get(
            trainer_config, "default_hdfs_dir", None
        )
        if default_hdfs not in (None, ""):
            raise ArchiveError(
                "online compact export supports local checkpoints only"
            )
        max_actor = _config_get(
            trainer_config, "max_actor_ckpt_to_keep", None
        )
        if max_actor != RAW_KEEP:
            raise ArchiveError(
                "trainer.max_actor_ckpt_to_keep must be 2 so a failed "
                "export leaves one previous recoverable raw checkpoint"
            )
        if _config_get(
            trainer_config, "del_local_ckpt_after_load", False
        ):
            raise ArchiveError(
                "trainer.del_local_ckpt_after_load must be false"
            )
        actor = trainer.config.actor_rollout_ref.actor
        checkpoint = _config_get(actor, "checkpoint", {})
        if _config_get(checkpoint, "async_save", False):
            raise ArchiveError(
                "online compact export requires synchronous verl saves"
            )
        raw_keep_text = os.environ.get(RAW_KEEP_ENV, str(RAW_KEEP))
        try:
            raw_keep = int(raw_keep_text)
        except ValueError as exc:
            raise ArchiveError(
                f"{RAW_KEEP_ENV} must be an integer"
            ) from exc
        export_dir = os.environ.get(
            EXPORT_DIR_ENV, f"{os.path.abspath(run_dir)}_adapters"
        )
        merger_temp_root = os.environ.get(MERGER_TMP_ENV) or None
        return cls(
            run_dir,
            export_dir,
            merger_temp_root=merger_temp_root,
            raw_keep=raw_keep,
        )

    @contextmanager
    def _locked(self):
        os.makedirs(self.run_dir, exist_ok=True)
        with open(self.lock_path, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _load_state(self, manifest_context):
        if not os.path.exists(self.state_path):
            return None
        try:
            with open(self.state_path, encoding="utf-8") as handle:
                state = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ArchiveError(
                f"cannot recover checkpoint archive state: {exc}"
            ) from exc
        expected = {
            "schema_version": STATE_SCHEMA_VERSION,
            "raw_keep": self.raw_keep,
            "run_manifest_sha256": manifest_context["sha256"],
            "manifest_payload_sha256": manifest_context["payload_sha256"],
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ArchiveError(
                    f"checkpoint archive state {key}={state.get(key)!r}, "
                    f"expected {value!r}"
                )
        marker = state.get("latest_marker")
        if (
            not isinstance(marker, int)
            or isinstance(marker, bool)
            or marker <= 0
        ):
            raise ArchiveError(
                "checkpoint archive state has an invalid latest marker"
            )
        compacts = state.get("compact_adapters")
        raw_steps = state.get("recoverable_raw_steps")
        if not isinstance(compacts, dict) or not isinstance(raw_steps, list):
            raise ArchiveError(
                "checkpoint archive state has invalid bookkeeping"
            )
        return state

    def _raw_directories(self):
        result = {}
        if not os.path.isdir(self.run_dir):
            return result
        for name in os.listdir(self.run_dir):
            match = RAW_PATTERN.fullmatch(name)
            if match:
                result[int(match.group(1))] = os.path.join(
                    self.run_dir, name
                )
        return result

    def _raw_status(self):
        result = {}
        for step, path in sorted(self._raw_directories().items()):
            try:
                inspect_source_checkpoint(
                    self.run_dir, step, hash_model=False
                )
            except ExportError as exc:
                result[step] = {
                    "path": path,
                    "complete": False,
                    "error": str(exc),
                }
            else:
                result[step] = {
                    "path": path,
                    "complete": True,
                    "error": None,
                }
        return result

    def _compact_directories(self):
        result = {}
        if not os.path.isdir(self.export_dir):
            return result
        for name in os.listdir(self.export_dir):
            if name.startswith("."):
                continue
            match = COMPACT_PATTERN.fullmatch(name)
            if match is None:
                raise ArchiveError(
                    f"unexpected compact-adapter entry: "
                    f"{os.path.join(self.export_dir, name)}"
                )
            step = int(match.group(1))
            result[step] = os.path.join(self.export_dir, name)
        return result

    def _verify_compacts(self, manifest_context, marker, state):
        records = {}
        for step, path in sorted(self._compact_directories().items()):
            if step > marker:
                raise ArchiveError(
                    f"compact adapter step {step} is newer than marker "
                    f"{marker}"
                )
            try:
                records[step] = verify_compact_adapter(
                    path, step, self.manifest_path
                )
            except ExportError as exc:
                raise ArchiveError(
                    f"invalid old compact adapter at step {step}: {exc}"
                ) from exc
        if state is not None:
            if state["latest_marker"] > marker:
                raise ArchiveError(
                    "checkpoint archive state is newer than verl marker"
                )
            for step_text, old_record in state[
                "compact_adapters"
            ].items():
                try:
                    step = int(step_text)
                except (TypeError, ValueError) as exc:
                    raise ArchiveError(
                        "checkpoint archive state has a non-integer "
                        "compact step"
                    ) from exc
                record = records.get(step)
                if record is None:
                    raise ArchiveError(
                        f"previously archived compact step {step} is missing"
                    )
                for key in (
                    "adapter_model_sha256",
                    "adapter_config_sha256",
                    "provenance_sha256",
                ):
                    if old_record.get(key) != record[key]:
                        raise ArchiveError(
                            f"compact step {step} {key} changed since the "
                            "last archive transaction"
                        )
        return records

    @staticmethod
    def _expected_steps(marker, manifest_context):
        _validate_scheduled_step(marker, manifest_context)
        save_freq = manifest_context["save_freq"]
        return list(range(save_freq, marker + 1, save_freq))

    def _export_missing(
        self,
        manifest_context,
        expected_steps,
        raw_status,
        compact_records,
    ):
        for step in expected_steps:
            if step in compact_records:
                continue
            raw = raw_status.get(step)
            if not raw or not raw["complete"]:
                detail = raw["error"] if raw else "raw directory is absent"
                raise ArchiveError(
                    f"checkpoint step {step} has neither a valid compact "
                    f"adapter nor an exportable raw: {detail}"
                )
            try:
                verify_live_runtime_sources(
                    manifest_context["payload"]
                )
                export_checkpoint(
                    step,
                    self.run_dir,
                    self.export_dir,
                    merger_temp_root=self.merger_temp_root,
                    if_exists="verify",
                )
                compact_records[step] = verify_compact_adapter(
                    os.path.join(
                        self.export_dir, f"step-{step:06d}"
                    ),
                    step,
                    self.manifest_path,
                )
            except (ExportError, OSError) as exc:
                raise ArchiveError(
                    f"compact export failed at step {step}: {exc}"
                ) from exc

    def _prune_raws(
        self,
        marker,
        manifest_context,
        raw_status,
        compact_records,
        require_previous,
    ):
        newer = [step for step in raw_status if step > marker]
        if newer:
            raise ArchiveError(
                "raw checkpoint directories are newer than the committed "
                f"marker {marker}: {newer}"
            )
        complete = sorted(
            step
            for step, status in raw_status.items()
            if status["complete"] and step <= marker
        )
        if marker not in complete:
            raise ArchiveError(
                f"latest marker step {marker} is not a complete raw "
                "checkpoint"
            )
        first_step = self._expected_steps(marker, manifest_context)[0]
        if require_previous and marker > first_step:
            previous = [step for step in complete if step < marker]
            if not previous:
                raise ArchiveError(
                    "latest raw checkpoint has no previous recoverable raw"
                )
        keep = set(complete[-self.raw_keep :])
        for step, status in sorted(raw_status.items()):
            if step in keep:
                continue
            if step not in compact_records:
                raise ArchiveError(
                    f"refusing to delete raw step {step} without a "
                    "verified compact adapter"
                )
            shutil.rmtree(status["path"])
        remaining = self._raw_status()
        complete_remaining = sorted(
            step
            for step, status in remaining.items()
            if status["complete"]
        )
        if len(complete_remaining) > self.raw_keep:
            raise ArchiveError(
                "raw checkpoint retention exceeds the two-checkpoint limit"
            )
        if marker not in complete_remaining:
            raise ArchiveError("raw retention removed the latest checkpoint")
        if (
            require_previous
            and marker > first_step
            and len(complete_remaining) < 2
        ):
            raise ArchiveError(
                "raw retention did not preserve a previous checkpoint"
            )
        return complete_remaining

    def _write_state(
        self, manifest_context, marker, compact_records, raw_steps
    ):
        payload = {
            "schema_version": STATE_SCHEMA_VERSION,
            "raw_keep": self.raw_keep,
            "latest_marker": marker,
            "run_manifest_sha256": manifest_context["sha256"],
            "manifest_payload_sha256": manifest_context["payload_sha256"],
            "compact_adapters": {
                str(step): {
                    "adapter_model_sha256": record[
                        "adapter_model_sha256"
                    ],
                    "adapter_config_sha256": record[
                        "adapter_config_sha256"
                    ],
                    "provenance_sha256": record["provenance_sha256"],
                }
                for step, record in sorted(compact_records.items())
            },
            "recoverable_raw_steps": list(raw_steps),
        }
        _atomic_json(self.state_path, payload)
        return payload

    def _reconcile_locked(self, marker, require_previous):
        manifest_context = _manifest_context(self.run_dir)
        _validate_scheduled_step(marker, manifest_context)
        state = self._load_state(manifest_context)
        compact_records = self._verify_compacts(
            manifest_context, marker, state
        )
        expected_steps = self._expected_steps(marker, manifest_context)
        raw_status = self._raw_status()
        newer = [step for step in raw_status if step > marker]
        if newer:
            raise ArchiveError(
                "raw checkpoint directories are newer than the committed "
                f"marker {marker}: {newer}"
            )
        self._export_missing(
            manifest_context,
            expected_steps,
            raw_status,
            compact_records,
        )
        if set(compact_records) != set(expected_steps):
            raise ArchiveError(
                "compact-adapter set does not match committed checkpoints"
            )
        raw_status = self._raw_status()
        raw_steps = self._prune_raws(
            marker,
            manifest_context,
            raw_status,
            compact_records,
            require_previous=require_previous,
        )
        return self._write_state(
            manifest_context, marker, compact_records, raw_steps
        )

    def reconcile_resume(self):
        """Restore bookkeeping and validate adapters before loading a resume."""

        with self._locked():
            _manifest_context(self.run_dir)
            marker = read_checkpoint_marker(self.run_dir, required=False)
            if marker is None:
                if (
                    os.path.exists(self.state_path)
                    or self._raw_directories()
                    or self._compact_directories()
                ):
                    raise ArchiveError(
                        "checkpoint evidence exists without a committed "
                        "latest marker"
                    )
                return None
            return self._reconcile_locked(
                marker, require_previous=False
            )

    def before_save(self, step):
        """Validate the previous recovery point before verl starts a save."""

        with self._locked():
            manifest_context = _manifest_context(self.run_dir)
            _validate_scheduled_step(step, manifest_context)
            marker = read_checkpoint_marker(self.run_dir, required=False)
            if marker is None:
                if (
                    os.path.exists(self.state_path)
                    or self._raw_directories()
                    or self._compact_directories()
                ):
                    raise ArchiveError(
                        "checkpoint evidence exists without a committed "
                        "latest marker"
                    )
                return
            if step <= marker:
                raise ArchiveError(
                    f"refusing checkpoint step {step} after marker {marker}"
                )
            self._reconcile_locked(marker, require_previous=False)
            previous = inspect_source_checkpoint(
                self.run_dir, marker, hash_model=False
            )
            if not previous:
                raise ArchiveError(
                    f"previous raw checkpoint {marker} is not recoverable"
                )

    def after_save(self, step):
        """Export only after verl has committed the matching latest marker."""

        with self._locked():
            marker = read_checkpoint_marker(self.run_dir)
            if marker != step:
                raise ArchiveError(
                    f"verl checkpoint marker {marker} does not match "
                    f"completed save step {step}"
                )
            return self._reconcile_locked(
                marker, require_previous=True
            )
