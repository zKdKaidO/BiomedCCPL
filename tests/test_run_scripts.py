"""The run scripts pass VGAP.MODE through and keep each variant's outputs apart."""

import os
import stat
import subprocess

import pytest

from conftest import REPO_ROOT

SEP = "\x1f"


def run_script(tmp_path, script, args, **env):
    """Run a script with a stand-in `python` that records its arguments."""
    log = tmp_path / "calls.log"
    fake = tmp_path / "bin" / "python"
    fake.parent.mkdir()
    fake.write_text(
        "#!/bin/bash\n"
        f"{{ printf '%s{SEP}' \"$@\"; printf '\\n'; }} >> \"$FAKE_PY_LOG\"\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    result = subprocess.run(
        ["bash", os.path.join("scripts", script), *args],
        cwd=REPO_ROOT,
        env={**os.environ, **env, "FAKE_PY_LOG": str(log),
             "PATH": f"{fake.parent}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )
    calls = []
    if log.exists():
        calls = [line.split(SEP)[:-1] for line in log.read_text().splitlines()]
    return result, calls


def option(call, name):
    return call[call.index(name) + 1]


def test_few_shot_defaults_to_original_model(tmp_path):
    result, calls = run_script(tmp_path, "run_few_shot.sh", ["/data", "BTMRI", "16"])
    assert result.returncode == 0, result.stderr
    train_calls = [c for c in calls if c[0] == "train.py"]
    assert len(train_calls) == 3
    for seed, call in enumerate(train_calls, start=1):
        assert option(call, "--output-dir") == f"output/few_shot/text/BTMRI/16shot/seed{seed}"
        assert option(call, "TRAINER.BIOMEDCCPL.VGAP.MODE") == "text"
    assert calls[-1] == ["tools/parse_test_res.py", "output/few_shot/text/BTMRI/16shot"]


def test_few_shot_mode_tag_and_overrides(tmp_path):
    result, calls = run_script(
        tmp_path, "run_few_shot.sh",
        ["/data", "BTMRI", "4", "1", "TRAINER.BIOMEDCCPL.VGAP.IMAGE_LAYERS", "[3, 7, 11]"],
        VGAP_MODE="image", RUN_TAG="r3",
    )
    assert result.returncode == 0, result.stderr
    call = calls[0]
    assert option(call, "--output-dir") == "output/few_shot/r3/BTMRI/4shot/seed1"
    assert option(call, "TRAINER.BIOMEDCCPL.VGAP.MODE") == "image"
    assert call[-2:] == ["TRAINER.BIOMEDCCPL.VGAP.IMAGE_LAYERS", "[3, 7, 11]"]


def test_base_to_novel_passes_mode_to_training_and_evaluation(tmp_path):
    result, calls = run_script(
        tmp_path, "run_base_to_novel.sh",
        ["/data", "BTMRI", "0", "TRAINER.BIOMEDCCPL.VGAP.READOUT_LR_MULT", "10.0"],
        VGAP_MODE="image",
    )
    assert result.returncode == 0, result.stderr
    train_calls = [c for c in calls if c[0] == "train.py"]
    assert len(train_calls) == 6  # base training + novel evaluation, 3 seeds
    for call in train_calls:
        assert option(call, "TRAINER.BIOMEDCCPL.VGAP.MODE") == "image"
        assert call[-2:] == ["TRAINER.BIOMEDCCPL.VGAP.READOUT_LR_MULT", "10.0"]
    base, novel = train_calls[0], train_calls[1]
    assert option(base, "--output-dir") == "output/base_to_novel/image/BTMRI/base/seed1"
    assert option(novel, "--model-dir") == option(base, "--output-dir")
    assert option(novel, "--output-dir") == "output/base_to_novel/image/BTMRI/novel/seed1"


@pytest.mark.parametrize("args", [["/data", "BTMRI", "3"], ["/data", "Nope", "16"]])
def test_few_shot_rejects_bad_arguments(tmp_path, args):
    result, calls = run_script(tmp_path, "run_few_shot.sh", args)
    assert result.returncode == 2
    assert calls == []
