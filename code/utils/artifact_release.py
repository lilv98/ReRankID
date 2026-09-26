"""Shared guards for publishing immutable formal W&B artifacts."""

import os
import subprocess


def _git(root, *args):
    return subprocess.check_output(
        ["git", *args],
        cwd=root,
        stderr=subprocess.PIPE,
        text=True,
    ).strip()


def git_snapshot(root):
    """Return the local commit and dirty state without requiring an upstream."""
    commit = _git(root, "rev-parse", "HEAD")
    dirty = bool(
        _git(root, "status", "--porcelain", "--untracked-files=all")
    )
    return {"commit": commit, "dirty": dirty}


def require_clean_pushed_release(root, expected_commit=None):
    """Require a clean branch whose HEAD exactly matches its remote upstream."""
    snapshot = git_snapshot(root)
    if snapshot["dirty"]:
        raise SystemExit(
            "formal artifact publication requires a clean Git worktree"
        )
    if expected_commit is not None and snapshot["commit"] != expected_commit:
        raise SystemExit(
            "Git HEAD changed after artifact validation: "
            f"{snapshot['commit']} != {expected_commit}"
        )
    try:
        branch = _git(root, "symbolic-ref", "--quiet", "--short", "HEAD")
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            "formal artifact publication requires a named Git branch"
        ) from error
    try:
        upstream = _git(
            root, "rev-parse", "--abbrev-ref", "--symbolic-full-name",
            "@{upstream}",
        )
        remote = _git(root, "config", "--get", f"branch.{branch}.remote")
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"formal artifact publication requires a remote upstream for {branch}"
        ) from error
    if not remote or remote == ".":
        raise SystemExit(
            f"formal artifact publication requires a remote upstream for {branch}"
        )
    try:
        _git(root, "fetch", "--prune", remote)
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"formal artifact publication could not refresh remote {remote}"
        ) from error
    divergence = _git(
        root, "rev-list", "--left-right", "--count", "HEAD...@{upstream}"
    ).split()
    if divergence != ["0", "0"]:
        raise SystemExit(
            "formal artifact publication requires pushed HEAD; "
            f"branch={branch}, upstream={upstream}, "
            f"HEAD...upstream={divergence}"
        )
    return {
        "commit": snapshot["commit"],
        "branch": branch,
        "upstream": upstream,
    }


def require_exact_regular_files(directory, names, label):
    """Return an exact allowlist of regular, non-symlink files."""
    if not os.path.isdir(directory) or os.path.islink(directory):
        raise ValueError(f"{label} is not a regular directory: {directory}")
    expected = set(names)
    with os.scandir(directory) as scan:
        entries = {entry.name: entry for entry in scan}
    missing = sorted(expected - set(entries))
    unexpected = sorted(set(entries) - expected)
    invalid = sorted(
        name
        for name in expected & set(entries)
        if entries[name].is_symlink()
        or not entries[name].is_file(follow_symlinks=False)
    )
    if missing or unexpected or invalid:
        raise ValueError(
            f"{label} must contain exactly regular files {sorted(expected)}; "
            f"missing={missing}, unexpected={unexpected}, invalid={invalid}"
        )
    return {name: os.path.join(directory, name) for name in sorted(expected)}


def require_regular_file(path, label):
    """Reject missing files, special files, and symlink indirection."""
    if os.path.islink(path) or not os.path.isfile(path):
        raise ValueError(f"{label} is not a regular non-symlink file: {path}")
    return path


def log_immutable_artifact(run, artifact):
    """Log, remove W&B's automatic aliases, and verify server state."""
    logged = run.log_artifact(artifact, aliases=[])
    committed = logged.wait() or logged
    committed.aliases = []
    committed.save()
    verified = run._public_api().artifact(committed.qualified_name)
    aliases = sorted(verified.aliases)
    if aliases:
        raise RuntimeError(
            "W&B artifact still has mutable aliases after cleanup: "
            f"{verified.qualified_name} aliases={aliases}"
        )
    identity = {
        "qualified_name": verified.qualified_name,
        "version": verified.version,
        "digest": verified.digest,
    }
    if not all(identity.values()):
        raise RuntimeError(
            "W&B did not return a complete immutable artifact identity"
        )
    return identity
