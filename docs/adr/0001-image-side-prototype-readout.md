# ADR-0001: Move the VGAP prototype readout from the text branch to the image branch

**Status:** Proposed (experiment, not yet run)
**Date:** 2026-10-27
**Branch:** `exp/image-side-readout`
**Scope:** BiomedCCPL on BiomedCLIP (ViT-B/16 image encoder, PubMedBERT text encoder), few-shot medical image classification

---

## 1. Context

### 1.1 What BiomedCCPL does today

1. The frozen ViT turns an image $x$ into 196 patch tokens (768-d) plus a [CLS] token at every layer.
2. At layers 3, 7, 11, VGAP compresses the 196 patches into $N = 14$ prototypes with a per-layer prototype MLP:
   $\mathbf{A} = \text{softmax}_{\text{dim}=1}(\text{MLP}(\mathcal{V}))$, $\mathbf{P} = \mathbf{A}^\top \mathcal{V}$.
3. For each class $c$, the text-branch [CLS] state $t_c$ queries the prototypes via cross-attention, and the answer is mixed back into the text state:
   $t_c \leftarrow \alpha\, t_c + (1-\alpha)\,\text{CrossAttn}(t_c, \mathbf{P}_l(x))$.
   This is done separately for the causal and the non-causal (SCD) text branches, with separate cross-attention modules (6 modules total).
4. Logits: $s_c(x) = \tau \cdot \langle \text{norm}(F_I(x)),\ \text{norm}(F_{T,c}(x)) \rangle$.

### 1.2 The problem

The last VGAP injection (layer 11) runs **after** the last text-transformer layer, so the class embedding is approximately

$$
F_{T,c}(x) \approx \text{proj}\big(\alpha\, t_c + (1-\alpha)\,\text{CrossAttn}(t_c,\ \mathbf{P}_{11}(x))\big)
$$

With the typical $\alpha \le 0.4$ (some configs use $\alpha = 0.1$), most of the "text" embedding is a learned summary **of the same image it is compared against**. Ignoring normalization, the logit splits into two terms:

$$
s_c(x) \propto \underbrace{\alpha\,\langle F_I,\ \text{proj}(t_c)\rangle}_{\text{image vs. class text}} \;+\; \underbrace{(1-\alpha)\,\langle F_I,\ \text{proj}(\text{CrossAttn}(t_c, \mathbf{P}_{11}(x)))\rangle}_{\text{image vs. a class-specific summary of itself}}
$$

So the trainable "image readout" lives on the text side of the comparison. Two consequences:

- **Cost.** Text features depend on the image, so the text encoder must run $B \times C$ times per step (×2 for causal + non-causal), instead of $C$ times as in CoOp. This is likely why the configs use batch size 1 (our guess; the paper does not say). At test time, text features cannot be cached either: every test image needs $C \times 2$ text forwards.
- **Interpretability.** We do not know whether VGAP's gain (+~10 pts at 16-shot, Table 2 of the paper) comes from (a) having a learned readout of the prototypes at all, or (b) each class reading the image in its own way (class-conditioned readout). The paper never separates these.

### 1.3 Why moving the readout is not a trivial rewrite

Note a structural point the coder should understand: if the text-side readout at layer 11 were made class-agnostic (same output $r(x)$ for every class), then the second logit term becomes $\langle F_I, \text{proj}(r(x))\rangle$, which is (up to normalization) **the same for every class** and cannot change the argmax. In other words, on the text side, the final-layer readout only helps *because* it is class-conditioned. On the image side, a class-agnostic readout does change the ranking, because it changes $F_I$ and therefore $\langle F_I', \text{proj}(t_c)\rangle$ differs per class. The image-side design is therefore a genuinely different model, not an equivalent refactor. That is what makes the comparison informative.

---

## 2. Hypothesis and predictions

### 2.1 Hypothesis (H1)

Most of VGAP's gain comes from **having a learned, prototype-based readout of the image**, not from **class-conditioned readout**. A single class-agnostic readout placed on the image side, with fixed (image-independent) text prompts and SCD still on, recovers most of BiomedCCPL's accuracy.

### 2.2 Null / alternative (H0)

Class-conditioned readout is the key ingredient: each class querying the prototypes with its own text state is what produces the gain. The image-side readout will land close to the "no VGAP" baseline.

### 2.3 Pre-registered predictions

Decide these thresholds **before** running anything, and do not change them after seeing results.

| ID | Prediction under H1 | Measured by |
|---|---|---|
| P1 | Image-side readout (R2) is within **1.0 pt** of original BiomedCCPL (R0) in mean accuracy, averaged over datasets × K, and no single dataset/K cell is worse by more than **2.0 pts** | Few-shot top-1 accuracy, 3 seeds |
| P2 | R2 beats the no-VGAP baseline (R1) by at least **half** of the R0 − R1 gap | Same |
| P3 | In base-to-novel, R2's **novel** accuracy and HM are ≥ R0's (the readout is class-agnostic, so nothing in it is fit to base-class queries) | Base / novel / HM |
| P4 | Text-encoder forward passes per training step drop from $2BC$ to $2C$; test-time text features are computed once and cached | Call counter, wall-clock |
| P5 (Phase 0) | In the trained original model, the cross-attention distributions over the 14 prototypes are **similar across classes**, and replacing each class's readout with the class-average readout at inference costs **< 1 pt** | Diagnostics in §5 |

### 2.4 How to read the outcomes

| Outcome | Interpretation | What to write |
|---|---|---|
| R2 ≈ R0 (P1 holds) | Learned readout is what matters; class conditioning is unnecessary | "Same accuracy at a fraction of the cost": contribution is a cheaper method + an analysis of VGAP |
| R1 < R2 < R0, clearly | Both ingredients contribute | Report the split: share of gain from readout vs. from class conditioning |
| R2 ≈ R1 | Class-conditioned readout is the key ingredient (H0) | Direct evidence the paper lacks; supports the original design |
| R2 > R0 | Class-conditioned readout overfits in few-shot | Strong result; check it holds across seeds before claiming |
| P5 says classes attend almost identically, but R2 ≪ R0 | Something other than conditioning differs (capacity, multi-layer injection, placement). Run R3 / R4 to locate it | |

---

## 3. Decision

Implement, on branch `exp/image-side-readout`, a variant of BiomedCCPL in which:

1. **Text side:** VGAP injections are removed from both the causal and the non-causal text branches. The learnable prompts (whatever the repo uses: shallow or deep context tokens) stay exactly as they are. SCD losses ($\mathcal{L}_{NEM}$, $\mathcal{L}_{Ort}$, and any others) stay on and are computed on the now image-independent text features.
2. **Image side:** one prototype readout module is added after the frozen ViT:

$$
F_I'(x) = \text{norm}\Big(F_I(x) + \beta\, \mathbf{W}_o\,\text{CrossAttn}\big(q = v_{cls}(x),\ K{=}V{=}\mathbf{P}_{11}(x)\big)\Big), \qquad \beta = \tanh(g),\ g_0 = 0
$$

3. **Logits:** $s_c(x) = \tau \cdot \langle F_I'(x), \text{norm}(F_{T,c}) \rangle$ for both the causal and the non-causal text features, used in the losses exactly as the original uses $F_I$ against each branch.

The original code path must remain available behind a config flag so R0 and R2 run from the same commit.

---

## 4. Options considered

### Option A: Keep VGAP on the text side (status quo, R0)

| Dimension | Assessment |
|---|---|
| Complexity | Existing code |
| Training cost | Text encoder runs $2BC$ times/step; effectively forces batch size 1 |
| Inference cost | $2C$ text forwards per test image, no caching |
| Trainable params | ~14.8M (3 prototype MLPs ≈ 0.6M + 6 cross-attention modules ≈ 14.2M) |
| Overfitting risk (1–16 shots) | High |

**Pros:** Published numbers; class-conditioned readout.
**Cons:** Expensive; unclear which ingredient drives the gain.

### Option B: Image-side readout, layer 11 only (R2, main proposal)

| Dimension | Assessment |
|---|---|
| Complexity | Low: one new module, removal of text-side injection |
| Training cost | Text encoder runs $2C$ times/step regardless of $B$ |
| Inference cost | Text features computed once per class set and cached |
| Trainable params | Prototype MLP (~0.2M) + 1 cross-attention (~2.4M) + $\mathbf{W}_o$ (~0.4M) + gate ≈ **3M** (verify with the counter in §6.6) |
| Overfitting risk | Lower (≈5× fewer params) |

**Pros:** Cheap; isolates the "learned readout" ingredient; safe init (identical to baseline at step 0).
**Cons:** Loses multi-scale injection (layers 3, 7) and class conditioning at the same time, so a failure alone would not tell us which one mattered. R3 and R4 exist to separate this.

### Option C: Image-side readout over multi-scale prototypes (R3)

Same as B, but K/V are the 42 prototypes from layers 3, 7, 11 concatenated, each with a learned layer embedding. Restores multi-scale information without restoring class conditioning.

### Option D: Class-agnostic readout kept on the text side (R4, diagnostic only)

Original architecture, but the cross-attention query is a **single shared learnable vector** (768-d, one per injection layer and branch) instead of $t_c$. Same cost as R0. Holds placement and multi-layer injection fixed, removes only class conditioning. Not a candidate method; it exists to attribute the result. (Per §1.3, at layer 11 this readout cannot change rankings; the effect can only come through layers 3 and 7.)

---

## 5. Phase 0: diagnostics on a trained original model (do first, no new training)

Run on R0 checkpoints (BTMRI, K = 16, 3 seeds is enough). Expected effort: a few hours.

1. **Logit decomposition.** For each test image and class, compute both terms of the logit split in §1.2 (before normalization; also report the normalized version by computing cosine with only the $t_c$ part vs. full $F_{T,c}$). Report, per layer-11 configuration $\alpha$:
   - mean |term 2| / (|term 1| + |term 2|);
   - accuracy using term 1 only vs. full logits.
2. **Class-specificity of the readout.** For each of the 6 cross-attention modules, take the attention weights over the 14 prototypes (average over heads), shape [C, 14] per image. Report the mean pairwise Jensen–Shannon divergence between classes, averaged over test images. Near 0 means every class reads the same prototypes.
3. **Class-average swap test.** At inference only, replace each class's cross-attention output by the mean over classes (per image, per layer, per branch), keep everything else. Report the accuracy drop. This is the cheapest direct test of P5.

Save numbers to `results/phase0/<dataset>_k<K>_seed<s>.json`. If (3) costs > 3 pts, expect H0; still run Phase 1, since that is the informative outcome.

---

## 6. Implementation instructions

> The repo structure is not reproduced here. Items marked **VERIFY** must be checked in the code before implementing. Do not guess them.

### 6.1 Locate the relevant code (VERIFY)

Search the repo for:
- the prototype MLP and `softmax(dim=1)` (VGAP prototype learner);
- the cross-attention modules and where $\alpha$ is applied (likely a config key such as `ALPHA`);
- the loop that runs the vision and text transformer blocks in lockstep and injects VGAP at layers 3, 7, 11;
- where `image_features` and `text_features` are normalized and multiplied by `logit_scale`;
- where the SCD losses consume causal and non-causal text features;
- whether "layer 11" is 0-indexed (the last block of ViT-B/16) or 1-indexed. Use **exactly the same layer index and the same pre-/post-LayerNorm hidden state** that the original layer-11 VGAP uses.

### 6.2 Config flags

Add (adapt the namespace to the repo's config system; CoOp-family code typically uses Dassl/yacs `cfg.TRAINER.<NAME>`):

```yaml
VGAP:
  MODE: "text"            # "text" (R0, original) | "image" (R2/R3) | "none" (R1) | "text_agnostic" (R4)
  IMAGE_LAYERS: [11]      # R2: [11]; R3: [3, 7, 11]
  READOUT_HEADS: 8        # VERIFY: use the same head count as the original cross-attention
  READOUT_DROPOUT: 0.0
  GATE_INIT: 0.0          # g_0; beta = tanh(g)
  READOUT_LR_MULT: 1.0    # LR multiplier for readout params (see 6.7)
```

`MODE: "text"` must reproduce the original behaviour bit-for-bit (same seeds → same numbers as before your change). Check this before anything else.

### 6.3 Decouple the encoders

When `MODE != "text"`, do not run the encoders in lockstep:

1. Run the frozen ViT once per batch, keeping the hidden states of the layers in `IMAGE_LAYERS` plus the final image embedding $F_I$ (projected, **not yet normalized**). Wrap it in `torch.no_grad()`; it receives no gradient.
2. Run the text encoder once on the $C$ prompts for the causal branch and once for the non-causal branch. Output shapes `[C, D]` each (D = 512, VERIFY the projection dim). No dependency on the image.

### 6.4 The readout module

```python
class ImageSideReadout(nn.Module):
    """Class-agnostic prototype readout on the image side (ADR-0001)."""

    def __init__(self, width=768, embed_dim=512, n_proto=14, heads=8,
                 layers=(11,), dropout=0.0, gate_init=0.0):
        super().__init__()
        self.layers = list(layers)
        # One prototype learner per layer. Reuse the ORIGINAL prototype-MLP class
        # and its init so prototypes are computed exactly as in VGAP.
        self.proto = nn.ModuleDict({str(l): OriginalPrototypeMLP(width, n_proto)  # VERIFY class name
                                    for l in self.layers})
        # Layer embedding only needed when mixing several layers (R3).
        self.layer_emb = (nn.Parameter(torch.zeros(len(self.layers), 1, width))
                          if len(self.layers) > 1 else None)
        self.ln_q = nn.LayerNorm(width)
        self.ln_kv = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.W_o = nn.Linear(width, embed_dim, bias=False)
        self.g = nn.Parameter(torch.tensor(float(gate_init)))   # beta = tanh(g)

    def forward(self, hidden, cls_tok, f_img):
        # hidden:  dict {layer: [B, 196, width]}  patch tokens (no CLS) at each layer
        # cls_tok: [B, width]  ViT [CLS] hidden state at the deepest layer in self.layers
        # f_img:   [B, embed_dim]  projected image feature, NOT normalized
        protos = []
        for i, l in enumerate(self.layers):
            P = self.proto[str(l)](hidden[l])            # [B, 14, width]
            if self.layer_emb is not None:
                P = P + self.layer_emb[i]
            protos.append(P)
        P = torch.cat(protos, dim=1)                     # [B, 14*len(layers), width]

        q = self.ln_q(cls_tok).unsqueeze(1)              # [B, 1, width]
        kv = self.ln_kv(P)
        r, attn_w = self.attn(q, kv, kv, need_weights=True, average_attn_weights=True)
        r = self.W_o(r.squeeze(1))                       # [B, embed_dim]

        beta = torch.tanh(self.g)
        f_new = f_img + beta * r
        return F.normalize(f_new, dim=-1), {"beta": beta.detach(),
                                            "attn": attn_w.detach(),          # [B, 1, 14*L]
                                            "r_norm_ratio": (r.norm(dim=-1) / f_img.norm(dim=-1)).mean().detach()}
```

Notes:
- **Init:** keep PyTorch default init for `attn` and `W_o`; only the gate is zero. At step 0, $\beta = 0$, so $F_I' = \text{norm}(F_I)$ and the model equals R1 exactly. $g$ gets a gradient at step 0; `W_o`/`attn` start receiving gradient once $\beta \ne 0$ (from step 1). This is the Flamingo-style tanh gate.
- **Query:** `cls_tok` is the ViT [CLS] hidden state at the same depth and same LN stage as the patch tokens fed to the prototype MLP. Do not use the projected 512-d feature as the query.
- **Do not** apply $\alpha$ here; $\alpha$ belongs to the text-side design. $\beta$ replaces it.
- R3 uses `layers=(3, 7, 11)`; the query stays the layer-11 [CLS].

### 6.5 Forward pass and losses

```python
# MODE == "image"
with torch.no_grad():
    hidden, cls_tok, f_img = vit_forward_keep(x, layers=cfg.VGAP.IMAGE_LAYERS)
f_img_n, logs = readout(hidden, cls_tok, f_img)          # [B, D], normalized

t_causal = F.normalize(text_encoder(prompts_causal), dim=-1)      # [C, D]
t_noncausal = F.normalize(text_encoder(prompts_noncausal), dim=-1)

logits_causal = logit_scale * f_img_n @ t_causal.t()           # [B, C]
logits_noncausal = logit_scale * f_img_n @ t_noncausal.t()
# Feed these into the SAME loss code the original uses (CE + SCD terms).
# Any SCD term that used per-image text features of shape [B, C, D] now receives [C, D];
# broadcast it, or rewrite the term so it is mathematically identical when B == 1. VERIFY each term.
```

- `MODE == "none"` (R1): same, but skip the readout and use `F.normalize(f_img)`.
- `MODE == "text_agnostic"` (R4): original lockstep code, but in every VGAP cross-attention replace the query $t_c$ with a learnable vector `q_l` of shape `[1, 768]` (one per injection layer and branch, init `normal(std=0.02)`), broadcast to all classes. Keep $\alpha$ and everything else as in R0.

### 6.6 Inference

For R1/R2/R3, compute `t_causal` and `t_noncausal` **once** per evaluation (and once per class set in base-to-novel), then run images in batches. Use whichever text branch(es) the original uses at test time (VERIFY: causal only, or a combination).

### 6.7 Training protocol

- Same optimizer, LR schedule, epochs, augmentations, $\tau$, prompt init, and SCD weights as the original config for each dataset.
- **Batch size 1 for all accuracy runs**, so R0 and R2 differ only in architecture. Larger batches are evaluated only in the cost benchmark (§7.3).
- Tuning budget, applied identically to R2 and R3, chosen on the few-shot **validation** split only, never on test: `READOUT_LR_MULT ∈ {1, 10}`, `READOUT_HEADS ∈ {original, 1}`. Report the chosen values. R0 uses the published config, no extra tuning.
- If $\beta$ stays below 0.05 for all of training on every dataset, the readout is unused; note it and retry with `READOUT_LR_MULT = 10` before concluding anything.

### 6.8 Tests (must pass before any experiment)

1. `MODE: "text"` reproduces pre-change outputs exactly (same seed, compare logits on 10 images, `atol=1e-6`).
2. At init, `MODE: "image"` logits equal `MODE: "none"` logits (`atol=1e-6`).
3. With a counter or forward hook on the text encoder: per training step, `MODE: "image"` calls it exactly 2 times (causal + non-causal) for batch sizes 1, 4 and 16; `MODE: "text"` calls it $2B$ times (or with $2BC$ sequences, depending on how batching is implemented).
4. No gradient reaches ViT or text-encoder weights (`requires_grad == False`, and `.grad is None` after a step).
5. After one optimizer step, `readout.g.grad` was nonzero; after two steps, `readout.W_o.weight.grad` is nonzero.
6. Shapes: prototypes `[B, 14·L, 768]`, readout output `[B, D]`.
7. A parameter counter prints trainable params per top-level module for each MODE and writes them to the run log.

### 6.9 Logging (per epoch, per run)

`beta`, `r_norm_ratio`, entropy of the readout attention over prototypes (mean over images), train/val accuracy, and the trainable-parameter count. Save the readout attention for 20 fixed test images per dataset for later visualization (prototype maps × patch grid).

---

## 7. Evaluation plan

### 7.1 Runs

| Run | Description | Purpose |
|---|---|---|
| R0 | Original BiomedCCPL (`MODE: text`) | Reference. Must match paper numbers within ~1 pt before trusting anything else |
| R1 | Fixed prompts + SCD, no VGAP (`MODE: none`) | Lower bound; the "no readout" point |
| R2 | Image-side readout, layer 11 (`MODE: image`, `[11]`) | **Main hypothesis test** |
| R3 | Image-side readout, layers 3/7/11 | Separates "multi-scale" from "class conditioning" |
| R4 | Class-agnostic text-side VGAP | Separates "placement" from "class conditioning"; diagnostic only |

R3 and R4 are needed only if R2 falls short of R0 (outcome rows 2–3 in §2.4). Run R0, R1, R2 first.

### 7.2 Accuracy grid

- Datasets: **BTMRI** (MRI), **DermaMNIST** (dermatoscopy), **KneeXray** (X-ray), to cover three modalities.
- Few-shot: K ∈ {4, 16} (add K = 1 if time allows; that is where the parameter-count difference should matter most).
- Base-to-novel: the paper's split for each dataset; report base, novel, HM.
- Seeds: 3 (same seeds and same few-shot samples for every run).
- Metrics: top-1 accuracy (for comparison with the paper) **and macro-F1** (DermaMNIST in particular is heavily imbalanced, so accuracy alone can hide a readout that ignores minority classes). Apply the P1/P2 thresholds to accuracy; report macro-F1 alongside and flag any case where the two disagree in direction.
- Report mean ± std. For R2 vs R0, also report the **paired per-seed difference** (same seed = same shots), mean and std, since the seed-to-seed variance of few-shot sampling is usually larger than the effect.

Total: 3 datasets × (2 K + 1 B2N) × 3 seeds = 27 runs per variant.

### 7.3 Cost benchmark (separate from accuracy runs)

On one GPU, BTMRI (C = 4), synthetic batches of B ∈ {1, 8, 32}, for R0 and R2:
- training step time (median over 200 steps, after 20 warm-up steps);
- peak GPU memory (`torch.cuda.max_memory_allocated`);
- text-encoder calls/sequences per step;
- test-time throughput (images/s) over the full test set, including text-feature computation.

Expected: at B = 1, training step cost is **similar** (both run the text encoder $2C$ times); the gain appears at B > 1 and at inference. Write this down explicitly in the paper; do not claim a training speedup at batch size 1. Report whether R0 at B = 32 fits in memory at all.

Optional, only if R2 ≈ R0: retrain R2 at a larger batch (e.g. B = 16 at K = 16 with linear LR scaling) to show the cost advantage is usable. Note that at K = 4 with 4 classes the whole training set is only 16 images, so large batches are not meaningful there.

### 7.4 What gets reported

1. Table: accuracy (mean ± std) for R0, R1, R2 (+R3, R4 if run) across the grid, plus paired R2 − R0 differences.
2. Table: trainable params, step time, memory, inference throughput.
3. Phase 0 diagnostics (logit-term share, cross-class JS divergence, swap-test drop).
4. Final $\beta$ and readout attention maps on the 20 fixed images.
5. A one-paragraph verdict against the predictions in §2.3, stated as hold / fail, using the pre-registered thresholds.

---

## 8. Consequences

**Becomes easier**
- Larger batches and faster inference (text features cacheable).
- ~5× fewer trainable parameters, which should help in the 1–16-shot regime.
- Cleaner analysis: the image-dependent part of the model sits in one module.

**Becomes harder / lost**
- No class-specific view of the image. Classes that differ only in a small region may suffer, since one readout must serve all classes.
- No multi-scale injection into the text encoder (partly restored by R3).
- The SCD story changes: text features are now image-independent, so the causal/non-causal separation applies only to prompts. Check whether any SCD term loses meaning (e.g. if a term compared per-image text features). Document this in the paper.

**Threats to validity**
- **Capacity confound:** R2 has ~3M vs ~14.8M trainable params. If R2 loses, it could be capacity rather than conditioning. R4 (same capacity as R0, no conditioning) addresses this.
- **Reproduction gap:** if R0 does not reproduce the paper, compare only against our R0, and say so.
- **Seed variance:** with 3 seeds, differences under ~1 pt are not reliable. Use the paired differences and do not over-claim.
- **Related work:** attention pooling is standard and Multi-Modal Adapter is close to R2. Frame the contribution as an analysis of why VGAP works plus a cheaper equivalent, not as a new mechanism.

---

## 9. Action items

1. [ ] Locate code points in §6.1 and fill in every **VERIFY** item in this ADR.
2. [ ] Add config flags; confirm `MODE: text` is bit-identical to the original (test 6.8.1).
3. [ ] Reproduce R0 on BTMRI K = 16, 3 seeds; compare to the paper.
4. [ ] Run Phase 0 diagnostics (§5) on those checkpoints.
5. [ ] Implement `ImageSideReadout`, decoupled forward, `MODE: none` / `image`; pass all tests in §6.8.
6. [ ] Run R0, R1, R2 on the full grid (§7.2).
7. [ ] Run the cost benchmark (§7.3).
8. [ ] Depending on the outcome (§2.4), run R3 and/or R4.
9. [ ] Write the verdict against P1–P5; update this ADR's status to Accepted or Rejected with a link to the results.
