"""Phase 0 diagnostics on a trained text-side VGAP model (ADR-0001 section 5).

Inference only; nothing is trained. For every test image the model is run
four times through VGAPProbe:
  (a) full:             the model as trained
  (b) text_only_alpha:  the last VGAP layer's readout removed, alpha * t_c kept
  (c) text_only:        the same layer uses t_c unscaled
  swap:                 every VGAP layer's readout replaced by its class mean
and reports their accuracies, the relative logit change of (b) and (c)
against (a), and the class-specificity of each cross-attention module (mean
pairwise Jensen-Shannon divergence between the classes' attention over the
prototypes). Takes the same arguments as an evaluation run of train.py:

$ python tools/phase0_diagnostics.py \\
    --root /path/to/data \\
    --config-file configs/trainers/BiomedCCPL/few_shot/16/BTMRI.yaml \\
    --model-dir output/few_shot/text/BTMRI/16shot/seed1 \\
    --load-epoch 50 --seed 1 \\
    DATASET.NAME BTMRI DATASET.NUM_SHOTS 16

Results go to results/phase0/<dataset>_k<K>_seed<s>.json (with a _base or _new
suffix for base-to-novel) unless --out is set.
"""

import json
import os
import os.path as osp
import sys

import torch

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from trainers.biomedccpl import VGAPProbe  # noqa: E402

VARIANTS = ("full", "text_only_alpha", "text_only", "swap")


def pairwise_js_divergence(probs):
    """Mean Jensen-Shannon divergence (nats) over class pairs.

    probs: (..., ncls, K) distributions; returns (...,). 0 means every class
    attends to the prototypes in the same way; the maximum is log 2.
    """
    probs = probs.clamp_min(1e-12)
    p, q = probs.unsqueeze(-2), probs.unsqueeze(-3)  # (..., C, 1, K), (..., 1, C, K)
    m = 0.5 * (p + q)
    js = 0.5 * (p * (p / m).log()).sum(-1) + 0.5 * (q * (q / m).log()).sum(-1)
    n_cls = probs.shape[-2]
    upper = torch.triu(torch.ones(n_cls, n_cls, dtype=torch.bool), diagonal=1)
    return js[..., upper].mean(-1)


def _probes(crosslayers):
    last = max(crosslayers)
    return {
        "full": VGAPProbe(non_causal=True),
        "text_only_alpha": VGAPProbe("text_only_alpha", layers=[last]),
        "text_only": VGAPProbe("text_only", layers=[last]),
        "swap": VGAPProbe("class_mean", layers=crosslayers),
    }


@torch.no_grad()
def run_phase0(model, batches):
    """batches: iterable of (image, label); returns a JSON-ready dict."""
    if model.vgap_mode not in ("text", "text_agnostic"):
        raise ValueError(
            f"Phase 0 needs a text-side VGAP model, not {model.vgap_mode!r}"
        )
    model.eval()
    probes = _probes(model.crosslayers)
    correct = {name: 0 for name in VARIANTS}
    rel_change = {name: 0.0 for name in ("text_only_alpha", "text_only")}
    js_sum, n_images = {}, 0

    try:
        for image, label in batches:
            logits = {}
            for name, probe in probes.items():
                probe.attn = {}
                model.vgap_probe = probe
                logits[name] = model(image)
            for name in VARIANTS:
                correct[name] += (logits[name].argmax(-1) == label).sum().item()
            full = logits["full"]
            for name in rel_change:
                change = (logits[name] - full).abs() / full.abs().clamp_min(1e-12)
                rel_change[name] += change.mean(-1).sum().item()
            for (branch, layer), attn in probes["full"].attn.items():
                key = f"{branch}_L{layer}"
                js = pairwise_js_divergence(attn).sum().item()
                js_sum[key] = js_sum.get(key, 0.0) + js
            n_images += image.shape[0]
    finally:
        model.vgap_probe = None

    accuracy = {name: 100.0 * correct[name] / n_images for name in VARIANTS}
    return {
        "vgap_mode": model.vgap_mode,
        "alpha": model.alpha,
        "crosslayers": list(model.crosslayers),
        "n_images": n_images,
        "accuracy": accuracy,
        "swap_drop": accuracy["full"] - accuracy["swap"],
        "mean_rel_logit_change": {k: v / n_images for k, v in rel_change.items()},
        "js_divergence": {k: v / n_images for k, v in sorted(js_sum.items())},
    }


def main():
    import train

    parser = train.build_parser()
    parser.add_argument("--out", default="", help="output JSON path")
    args = parser.parse_args()
    trainer = train.prepare_trainer(args)
    trainer.load_model(args.model_dir, epoch=args.load_epoch)

    batches = (trainer.parse_batch_test(batch) for batch in trainer.test_loader)
    results = run_phase0(trainer.model, batches)

    cfg = trainer.cfg
    results.update(
        dataset=cfg.DATASET.NAME,
        shots=cfg.DATASET.NUM_SHOTS,
        seed=cfg.SEED,
        subsample_classes=cfg.DATASET.SUBSAMPLE_CLASSES,
        model_dir=args.model_dir,
        load_epoch=args.load_epoch,
    )
    name = f"{cfg.DATASET.NAME}_k{cfg.DATASET.NUM_SHOTS}_seed{cfg.SEED}"
    if cfg.DATASET.SUBSAMPLE_CLASSES != "all":  # base-to-novel: base / new
        name += f"_{cfg.DATASET.SUBSAMPLE_CLASSES}"
    out = args.out or osp.join(REPO_ROOT, "results", "phase0", f"{name}.json")
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Saved Phase 0 diagnostics to {out}")


if __name__ == "__main__":
    main()
