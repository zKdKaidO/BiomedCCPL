"""Save the image-side readout attention of a trained checkpoint (ADR-0001 6.9).

Training runs in VGAP.MODE "image" already write readout_attention.pt to their
output directory; this re-creates it from a checkpoint. Takes the same
arguments as an evaluation run of train.py, for example:

$ python tools/save_readout_attention.py \\
    --root /path/to/data \\
    --output-dir output/attention/BTMRI/seed1 \\
    --config-file configs/trainers/BiomedCCPL/few_shot/16/BTMRI.yaml \\
    --model-dir output/few_shot/image/BTMRI/16shot/seed1 \\
    --load-epoch 50 \\
    DATASET.NAME BTMRI DATASET.NUM_SHOTS 16 TRAINER.BIOMEDCCPL.VGAP.MODE image
"""

import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import train  # noqa: E402


def main():
    parser = train.build_parser()
    parser.add_argument("--n-images", type=int, default=20)
    args = parser.parse_args()
    trainer = train.prepare_trainer(args)
    if trainer.model.vgap_mode != "image":
        sys.exit("Readout attention exists only in TRAINER.BIOMEDCCPL.VGAP.MODE image")
    trainer.load_model(args.model_dir, epoch=args.load_epoch)
    os.makedirs(trainer.output_dir, exist_ok=True)
    trainer.save_readout_attention(n_images=args.n_images)


if __name__ == "__main__":
    main()
