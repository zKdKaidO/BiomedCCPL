"""Cost benchmark for the VGAP variants (ADR-0001 section 7.3).

Synthetic inputs only; no dataset or Dassl needed, and the timed optimizer
steps are not a training run (random images, nothing is saved). For each
VGAP.MODE and batch size B it reports:
  - training step time, median over --steps after --warmup steps;
  - peak memory (CUDA: max_memory_allocated; MPS: driver memory after the
    steps, an approximation; CPU: not measured);
  - text-encoder sequences per training step;
  - test-time throughput in images/s over --test-images images, including
    the text-feature computation.

$ python tools/benchmark_cost.py --device cuda --modes text image --batch-sizes 1 8 32

Results go to results/cost/<dataset>_<device>.json unless --out is set.
"""

import argparse
import json
import os
import os.path as osp
import statistics
import sys
import time

import torch
import yaml
from yacs.config import CfgNode as CN

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from trainers.config import VGAP_MODES, add_biomedccpl_cfg  # noqa: E402

BIOMEDCLIP = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
BTMRI_CLASSNAMES = ["glioma", "meningioma", "no tumor", "pituitary"]


def build_cfg(dataset="BTMRI", shots=16, mode="text", image_layers=(11,)):
    """The trainer config of a shipped few-shot YAML, without Dassl."""
    cfg = CN()
    cfg.TRAINER = CN()
    cfg.INPUT = CN()
    cfg.INPUT.SIZE = (224, 224)
    cfg.DATASET = CN()
    cfg.DATASET.NAME = dataset
    add_biomedccpl_cfg(cfg)
    path = osp.join(
        REPO_ROOT, "configs", "trainers", "BiomedCCPL", "few_shot", str(shots),
        f"{dataset}.yaml",
    )
    with open(path) as f:
        full = yaml.safe_load(f)
    cfg.TRAINER.BIOMEDCCPL.merge_from_other_cfg(CN(full["TRAINER"]["BIOMEDCCPL"]))
    cfg.TRAINER.BIOMEDCCPL.VGAP.MODE = mode
    cfg.TRAINER.BIOMEDCCPL.VGAP.IMAGE_LAYERS = list(image_layers)
    return cfg, full["OPTIM"]["LR"]


def build_model(biomedclip, cfg, classnames, device):
    from trainers.biomedccpl import TRAINABLE_PARAM_NAMES, CustomCLIP

    model = CustomCLIP(cfg, classnames, biomedclip)
    for name, param in model.named_parameters():
        param.requires_grad = any(t in name for t in TRAINABLE_PARAM_NAMES)
    return model.to(device)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _is_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc)


def benchmark(model, batch_size, device, lr=0.0025, steps=200, warmup=20,
              test_images=512, seed=0):
    """Time one configuration; returns a JSON-ready dict."""
    generator = torch.Generator().manual_seed(seed)
    image = torch.randn(batch_size, 3, 224, 224, generator=generator).to(device)
    label = (torch.arange(batch_size) % model.n_cls).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)

    sequences = []
    first_text_layer = model.biomedclip_model.text.transformer.encoder.layer[0]
    handle = first_text_layer.register_forward_hook(
        lambda m, inp, out: sequences.append(inp[0].shape[0])
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        model.train()
        times = []
        for step in range(warmup + steps):
            sequences.clear()
            _sync(device)
            start = time.perf_counter()
            total_loss = model(image, label)[1]
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            _sync(device)
            if step >= warmup:
                times.append(time.perf_counter() - start)
        sequences_per_step = list(sequences)

        model.eval()
        model._text_cache = None  # include the text features in the test cost
        n_done = 0
        _sync(device)
        start = time.perf_counter()
        with torch.no_grad():
            while n_done < test_images:
                n = min(batch_size, test_images - n_done)
                model(image[:n])
                n_done += n
        _sync(device)
        test_seconds = time.perf_counter() - start
    finally:
        handle.remove()

    if device.type == "cuda":
        memory = torch.cuda.max_memory_allocated(device)
    elif device.type == "mps":
        memory = torch.mps.driver_allocated_memory()
    else:
        memory = None
    return {
        "batch_size": batch_size,
        "step_time_median_s": statistics.median(times) if times else None,
        "memory_bytes": memory,
        "text_sequences_per_step": sequences_per_step,
        "test_images_per_s": n_done / test_seconds if n_done else None,
        "oom": False,
    }


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--modes", nargs="+", default=["text", "image"],
                        choices=VGAP_MODES)
    parser.add_argument("--image-layers", nargs="+", type=int, default=[11])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--test-images", type=int, default=512)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    from open_clip import create_model_from_pretrained
    from trainers.biomedccpl import count_trainable_params

    device = torch.device(args.device)
    biomedclip, _ = create_model_from_pretrained(BIOMEDCLIP)
    biomedclip = biomedclip.float()

    results = []
    for mode in args.modes:
        cfg, lr = build_cfg(mode=mode, image_layers=args.image_layers)
        torch.manual_seed(0)
        model = build_model(biomedclip, cfg, BTMRI_CLASSNAMES, device)
        n_params = sum(count_trainable_params(model).values())
        for batch_size in args.batch_sizes:
            row = {"mode": mode, "trainable_params": n_params}
            try:
                row.update(benchmark(
                    model, batch_size, device, lr=lr, steps=args.steps,
                    warmup=args.warmup, test_images=args.test_images,
                ))
            except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                if not _is_oom(exc):
                    raise
                row.update(batch_size=batch_size, oom=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            results.append(row)
            print(json.dumps(row))
        del model

    record = {
        "dataset": "BTMRI",
        "n_classes": len(BTMRI_CLASSNAMES),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "torch": torch.__version__,
        "steps": args.steps,
        "warmup": args.warmup,
        "test_images": args.test_images,
        "results": results,
    }
    out = args.out or osp.join(
        REPO_ROOT, "results", "cost", f"BTMRI_{device.type}.json"
    )
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Saved cost benchmark to {out}")


if __name__ == "__main__":
    main()
