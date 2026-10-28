"""Shared fixtures for the ADR-0001 tests.

The tests run on CPU and do not need Dassl or the datasets. If Dassl is not
installed, the handful of names that trainers/biomedccpl.py imports from it are
replaced by stand-ins, which is enough to build and run CustomCLIP.
"""

import importlib.util
import os
import subprocess
import sys
import types

import pytest
import torch
import yaml
from yacs.config import CfgNode as CN

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# The commit the ADR was written against: the unmodified BiomedCCPL model code.
ORIGINAL_COMMIT = "3257176"
BIOMEDCLIP = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CLASSNAMES = ["glioma", "meningioma", "no tumor", "pituitary"]  # BTMRI, C = 4


def _install_dassl_stub():
    class _Registry:
        def register(self, obj=None):
            return obj if obj is not None else (lambda o: o)

    def _unavailable(*args, **kwargs):
        raise RuntimeError("Dassl is not installed; this test environment uses a stub")

    modules = {
        "dassl": {},
        "dassl.engine": {"TRAINER_REGISTRY": _Registry(), "TrainerX": object},
        "dassl.utils": {"load_checkpoint": _unavailable},
        "dassl.optim": {
            "build_optimizer": _unavailable,
            "build_lr_scheduler": _unavailable,
        },
        "dassl.metrics": {"compute_accuracy": _unavailable},
    }
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        sys.modules[name] = module


if importlib.util.find_spec("dassl") is None:
    _install_dassl_stub()

torch.set_num_threads(4)


def make_cfg(dataset="BTMRI", shots=16, **vgap):
    """Build the trainer config from a shipped YAML without Dassl."""
    from trainers.config import add_biomedccpl_cfg

    cfg = CN()
    cfg.TRAINER = CN()
    cfg.INPUT = CN()
    cfg.INPUT.SIZE = (224, 224)
    cfg.DATASET = CN()
    cfg.DATASET.NAME = dataset
    add_biomedccpl_cfg(cfg)

    path = os.path.join(
        REPO_ROOT, "configs", "trainers", "BiomedCCPL", "few_shot", str(shots),
        f"{dataset}.yaml",
    )
    with open(path) as f:
        trainer_cfg = yaml.safe_load(f)["TRAINER"]["BIOMEDCCPL"]
    cfg.TRAINER.BIOMEDCCPL.merge_from_other_cfg(CN(trainer_cfg))
    for key, value in vgap.items():
        cfg.TRAINER.BIOMEDCCPL.VGAP[key] = value
    return cfg


def load_original_module():
    """Load trainers/biomedccpl.py as it was at ORIGINAL_COMMIT."""
    try:
        source = subprocess.check_output(
            ["git", "show", f"{ORIGINAL_COMMIT}:trainers/biomedccpl.py"],
            cwd=REPO_ROOT,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"cannot read the original model code from git: {exc}")
    # The trainer class would clash with the current one in Dassl's registry.
    source = source.replace("@TRAINER_REGISTRY.register()\n", "")
    module = types.ModuleType("biomedccpl_original")
    exec(compile(source, "biomedccpl_original.py", "exec"), module.__dict__)
    return module


@pytest.fixture(scope="session")
def biomedclip():
    from open_clip import create_model_from_pretrained

    try:
        model, _ = create_model_from_pretrained(BIOMEDCLIP)
    except Exception as exc:  # no network and no cached weights
        pytest.skip(f"BiomedCLIP weights are not available: {exc}")
    return model.float()
