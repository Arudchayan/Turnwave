"""Guard the native runner helper with a fake checkout/toolchain only."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "build_cuda_server.sh"
EXPECTED_REVISION = "ec9d1fdd9cc18643c5e161b65b8053f6f6f34a4b"


def _win_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if not drive or not resolved.as_posix().startswith(resolved.drive):
        raise RuntimeError(f"Cannot map test path into WSL: {path}")
    return f"/mnt/{drive}/{resolved.as_posix()[len(resolved.drive) + 1:]}"


def _shell_command(path: Path, args: list[str], env: dict[str, str]):
    if os.name != "nt":
        return ["bash", str(path), *args], env

    wsl = shutil.which("wsl.exe")
    if not wsl:
        pytest.skip("WSL is unavailable for the fake-shell contract test")
    probe = subprocess.run([wsl, "-d", "Ubuntu", "--", "true"], capture_output=True)
    if probe.returncode:
        pytest.skip("Ubuntu WSL is unavailable for the fake-shell contract test")

    # Test files live on a local Windows drive and are accessed via its WSL mount.
    linux_env = {
        key: (_win_to_wsl(Path(value)) if key in {
            "LOCALVOICE_RUNNER_SOURCE", "LOCALVOICE_RUNNER_BUILD_DIR",
            "FAKE_TOPLEVEL", "FAKE_BUILD_DIR", "FAKE_CMAKE_LOG", "FAKE_GIT_LOG", "HOME",
        } else value)
        for key, value in env.items()
    }
    fake_bin = _win_to_wsl(Path(env["PATH"].split(os.pathsep)[0]))
    linux_env["PATH"] = f"{fake_bin}:/usr/bin:/bin"
    wsl_args = ["env", "-i", *[f"{key}={value}" for key, value in linux_env.items()]]
    command = [wsl, "-d", "Ubuntu", "--", *wsl_args, "bash", _win_to_wsl(path), *args]
    # Scripts created on NTFS may not have the executable bit under DrvFs.
    fake_dir = Path(env["PATH"].split(os.pathsep)[0])
    scripts = [fake_dir / name for name in ("git", "cmake", "nvcc", "c++") if (fake_dir / name).exists()]
    subprocess.run([wsl, "-d", "Ubuntu", "--", "chmod", "+x",
                    *[_win_to_wsl(script) for script in scripts]], check=True, capture_output=True)
    return command, None


def _fake_tools(tmp_path: Path, *, revision=EXPECTED_REVISION, patch_state="applied",
                dirty=" M existing-local-change.cpp", include_cmake=True, include_nvcc=True):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    source = tmp_path / "source checkout with spaces"
    source.mkdir()
    build = tmp_path / "build output"
    build.mkdir()
    git_log = tmp_path / "git-calls.log"
    cmake_log = tmp_path / "cmake-calls.log"
    (fake_bin / "git").write_bytes(r'''#!/bin/sh
set -eu
printf '%s\n' "$*" >> "$FAKE_GIT_LOG"
if [ "$1" = "-C" ]; then shift 2; fi
case "$1" in
  rev-parse)
    if [ "$2" = "--show-toplevel" ]; then printf '%s\n' "$FAKE_TOPLEVEL";
    else printf '%s\n' "$FAKE_REVISION"; fi ;;
  status) printf '%s' "$FAKE_STATUS" ;;
  apply)
    if [ "$2" = "--reverse" ]; then
      case "$FAKE_PATCH_STATE" in
        applied) exit 0 ;;
        partial) case "$4" in *liquid-turn-reset.patch) exit 0 ;; *) exit 1 ;; esac ;;
        *) exit 1 ;;
      esac
    fi
    case "$FAKE_PATCH_STATE" in
      base) exit 0 ;;
      partial) case "$3" in *liquid-turn-reset.patch) exit 1 ;; *) exit 0 ;; esac ;;
      *) exit 1 ;;
    esac ;;
  *) echo "unexpected git operation: $*" >&2; exit 91 ;;
esac
'''.encode("utf-8"))
    if include_nvcc:
        (fake_bin / "nvcc").write_bytes(b"#!/bin/sh\nexit 0\n")
    (fake_bin / "c++").write_bytes(b"#!/bin/sh\nexit 0\n")
    if include_cmake:
        (fake_bin / "cmake").write_bytes(r'''#!/bin/sh
set -eu
printf '%s\n' "$*" >> "$FAKE_CMAKE_LOG"
if [ "$1" = "--version" ]; then echo 'cmake version 3.30.5'; exit 0; fi
if [ "$1" = "--build" ]; then
  mkdir -p "$FAKE_BUILD_DIR/bin"
  printf '#!/bin/sh\nexit 0\n' > "$FAKE_BUILD_DIR/bin/llama-liquid-audio-server"
  printf '#!/bin/sh\nexit 0\n' > "$FAKE_BUILD_DIR/bin/llama-liquid-audio-cli"
  chmod +x "$FAKE_BUILD_DIR/bin/llama-liquid-audio-server" "$FAKE_BUILD_DIR/bin/llama-liquid-audio-cli"
fi
'''.encode("utf-8"))
    env = {
        "PATH": str(fake_bin) + os.pathsep + os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
        "LOCALVOICE_RUNNER_SOURCE": str(source),
        "LOCALVOICE_RUNNER_BUILD_DIR": str(build),
        "LOCALVOICE_BUILD_JOBS": "2",
        "FAKE_TOPLEVEL": str(source),
        "FAKE_BUILD_DIR": str(build),
        "FAKE_CMAKE_LOG": str(cmake_log),
        "FAKE_GIT_LOG": str(git_log),
        "FAKE_REVISION": revision,
        "FAKE_PATCH_STATE": patch_state,
        "FAKE_STATUS": dirty,
    }
    return env, source, build, git_log, cmake_log, fake_bin


def _run(tmp_path: Path, env: dict[str, str]):
    command, native_env = _shell_command(SCRIPT, [], env)
    if os.name == "nt":
        return subprocess.run(command, text=True, capture_output=True, timeout=30)
    return subprocess.run(command, text=True, capture_output=True, timeout=30, env=native_env)


def test_native_build_script_syntax_and_no_moving_or_destructive_git_commands(tmp_path):
    text = SCRIPT.read_text(encoding="utf-8")
    assert EXPECTED_REVISION in text
    assert "pull/18641/head" not in text
    for forbidden in ("git clone", "git fetch", "git checkout", "git reset", "git clean", "curl ", "wget "):
        assert forbidden not in text
    if os.name == "nt":
        wsl = shutil.which("wsl.exe")
        if not wsl:
            pytest.skip("WSL is unavailable for Bash syntax validation")
        probe = subprocess.run([wsl, "-d", "Ubuntu", "--", "true"], capture_output=True)
        if probe.returncode:
            pytest.skip("Ubuntu WSL is unavailable for Bash syntax validation")
        result = subprocess.run([wsl, "-d", "Ubuntu", "--", "bash", "-n", _win_to_wsl(SCRIPT)],
                                text=True, capture_output=True, timeout=30)
    else:
        result = subprocess.run(["bash", "-n", str(SCRIPT)], text=True, capture_output=True,
                                timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("patch_state", ["base", "applied"])
def test_pinned_checkout_builds_with_clean_or_fully_patched_state(tmp_path, patch_state):
    dirty = " M existing-local-change.cpp" if patch_state == "applied" else ""
    env, source, build, git_log, cmake_log, _ = _fake_tools(
        tmp_path, patch_state=patch_state, dirty=dirty)
    marker = source / "tracked-local-change.cpp"
    if patch_state == "applied":
        marker.write_text("preserve this local change", encoding="utf-8")
    result = _run(tmp_path, env)
    assert result.returncode == 0, result.stderr
    if patch_state == "applied":
        assert "all 5 reviewed Turnwave patches already applied" in result.stdout
        assert "existing local changes" in result.stderr
        assert marker.read_text(encoding="utf-8") == "preserve this local change"
    else:
        assert "without Turnwave patches" in result.stdout
        assert "existing local changes" not in result.stderr
    assert (build / "bin" / "llama-liquid-audio-server").is_file()
    assert (build / "bin" / "llama-liquid-audio-cli").is_file()
    calls = git_log.read_text(encoding="utf-8")
    assert "rev-parse --verify HEAD^{commit}" in calls
    operations = [line.split("source checkout with spaces ", 1)[-1].split()[0]
                  for line in calls.splitlines()]
    assert not set(operations) & {"fetch", "checkout", "reset", "clean", "clone"}
    cmake_calls = cmake_log.read_text(encoding="utf-8")
    assert "-DGGML_CUDA=ON" in cmake_calls
    assert "--target llama-liquid-audio-server llama-liquid-audio-cli" in cmake_calls
    assert source.is_dir()


def test_wrong_revision_fails_closed_before_cmake(tmp_path):
    env, _, _, _, cmake_log, _ = _fake_tools(tmp_path, revision="ec9d1fdd9" + "0" * 31)
    result = _run(tmp_path, env)
    assert result.returncode != 0
    assert "expected exact commit" in result.stderr
    assert not cmake_log.exists()


def test_partial_reviewed_patch_set_fails_without_building(tmp_path):
    env, _, _, _, cmake_log, _ = _fake_tools(tmp_path, patch_state="partial")
    result = _run(tmp_path, env)
    assert result.returncode != 0
    assert "only 1/5 reviewed patches are applied" in result.stderr
    assert not cmake_log.exists()


def test_missing_cuda_toolkit_reports_prerequisite_without_download(tmp_path):
    env, _, _, _, cmake_log, _ = _fake_tools(tmp_path, include_nvcc=False)
    result = _run(tmp_path, env)
    assert result.returncode != 0
    assert "CUDA Toolkit nvcc is required but was not found" in result.stderr
    assert not cmake_log.exists()
