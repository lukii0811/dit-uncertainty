"""Validate a Mentor checkpoint against the provisional TREAD-SiT model."""

import argparse
import json

from checkpoint import build_model_from_checkpoint


def main(args):
    _, report = build_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        state_key=args.state_key,
        resolution=args.resolution,
        num_classes=args.num_classes,
        encoder_depth=args.encoder_depth,
        start_layer_idx=args.start_layer_idx,
        end_layer_idx=args.end_layer_idx,
        qk_norm=args.qk_norm,
        fused_attn=args.fused_attn,
        ignored_state_prefixes=("projectors.",),
        allow_unsafe_pickle=args.allow_unsafe_checkpoint_load,
    )
    print(json.dumps(report, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--state-key", default="auto")
    parser.add_argument("--model", default="SiT-B/2")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--start-layer-idx", type=int, default=2)
    parser.add_argument("--end-layer-idx", type=int, default=8)
    parser.add_argument(
        "--qk-norm", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fused-attn", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--allow-unsafe-checkpoint-load", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
