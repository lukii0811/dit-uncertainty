"""Generate a class-balanced seed-class manifest for paired SiT experiments."""

import argparse
import json
import random
import re
from pathlib import Path


def parse_class_spec(value):
    class_ids = []
    range_pattern = re.compile(r"^(\d+)-(\d+)$")
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        match = range_pattern.match(item)
        if match:
            start, end = map(int, match.groups())
            if end < start:
                raise ValueError(f"Invalid descending class range: {item}")
            class_ids.extend(range(start, end + 1))
        else:
            class_ids.append(int(item))
    if not class_ids:
        raise ValueError("No class IDs were provided")
    return class_ids


def load_class_file(path):
    content = Path(path).read_text(encoding="utf-8")
    return parse_class_spec(",".join(content.split()))


def validate_classes(class_ids, dataset_num_classes):
    if len(class_ids) != len(set(class_ids)):
        raise ValueError("The selected class list contains duplicate IDs")
    invalid = [
        value
        for value in class_ids
        if not 0 <= value < dataset_num_classes
    ]
    if invalid:
        raise ValueError(
            f"Class IDs must be in [0, {dataset_num_classes - 1}]; "
            f"invalid values: {invalid[:10]}"
        )
    return list(class_ids)


def select_classes(args):
    if args.classes is not None:
        class_ids = parse_class_spec(args.classes)
        selection_method = "explicit"
    elif args.class_file is not None:
        class_ids = load_class_file(args.class_file)
        selection_method = "file"
    else:
        count = args.num_selected_classes
        if not 1 <= count <= args.dataset_num_classes:
            raise ValueError(
                "num_selected_classes must be between 1 and dataset_num_classes"
            )
        if args.class_selection == "first":
            class_ids = list(range(count))
        else:
            rng = random.Random(args.class_selection_seed)
            class_ids = sorted(
                rng.sample(range(args.dataset_num_classes), count)
            )
        selection_method = args.class_selection
    return (
        validate_classes(class_ids, args.dataset_num_classes),
        selection_method,
    )


def generate_pairs(
    class_ids,
    samples_per_class,
    noise_master_seed,
    seed_min,
    seed_max,
    noise_seed_mode,
):
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")
    if noise_seed_mode not in {
        "independent-per-class",
        "shared-across-classes",
    }:
        raise ValueError(f"Unknown noise seed mode: {noise_seed_mode}")

    required_unique_seeds = (
        len(class_ids) * samples_per_class
        if noise_seed_mode == "independent-per-class"
        else samples_per_class
    )
    if (
        seed_max < seed_min
        or seed_max - seed_min + 1 < required_unique_seeds
    ):
        raise ValueError("The seed range does not contain enough unique seeds")
    rng = random.Random(noise_master_seed)
    noise_seeds = rng.sample(
        range(seed_min, seed_max + 1),
        required_unique_seeds,
    )
    pairs = []
    for class_position, class_idx in enumerate(class_ids):
        for noise_index in range(samples_per_class):
            if noise_seed_mode == "shared-across-classes":
                seed = noise_seeds[noise_index]
            else:
                seed = noise_seeds[
                    class_position * samples_per_class + noise_index
                ]
            pairs.append(
                {
                    "sample_id": len(pairs),
                    "seed": seed,
                    "class_idx": class_idx,
                    "class_position": class_position,
                    "noise_index_within_class": noise_index,
                }
            )
    return pairs, noise_seeds


def main(args):
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output} exists; pass --overwrite to replace it"
        )
    class_ids, selection_method = select_classes(args)
    pairs, generated_noise_seeds = generate_pairs(
        class_ids=class_ids,
        samples_per_class=args.samples_per_class,
        noise_master_seed=args.noise_master_seed,
        seed_min=args.seed_min,
        seed_max=args.seed_max,
        noise_seed_mode=args.noise_seed_mode,
    )
    payload = {
        "format": "tread-class-balanced-seed-pairs-v3",
        "dataset_num_classes": args.dataset_num_classes,
        "num_selected_classes": len(class_ids),
        "selected_classes": class_ids,
        "class_selection": selection_method,
        "class_selection_seed": (
            args.class_selection_seed
            if selection_method == "random"
            else None
        ),
        "samples_per_class": args.samples_per_class,
        "count": len(pairs),
        "noise_master_seed": args.noise_master_seed,
        "noise_seed_range": [args.seed_min, args.seed_max],
        "noise_seed_mode": args.noise_seed_mode,
        "num_unique_noise_seeds": len(generated_noise_seeds),
        "generated_noise_seeds": generated_noise_seeds,
        "pairs": pairs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Wrote {len(pairs)} pairs = {len(class_ids)} classes x "
        f"{args.samples_per_class} noises/class to {output} "
        f"({args.noise_seed_mode}; {len(generated_noise_seeds)} unique "
        "noise seeds)"
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="./seed/seed_pairs_32x8_shared.json")
    parser.add_argument("--dataset-num-classes", type=int, default=1000)
    parser.add_argument("--samples-per-class", type=int, default=8)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--classes", help="IDs/ranges, e.g. 0,7,42,100-112", default="0-31")
    group.add_argument("--class-file")
    group.add_argument("--num-selected-classes", type=int)
    parser.add_argument(
        "--class-selection", choices=["random", "first"], default="first"
    )
    parser.add_argument("--class-selection-seed", type=int, default=20260717)
    parser.add_argument("--noise-master-seed", type=int, default=20260718)
    parser.add_argument(
        "--noise-seed-mode",
        choices=["independent-per-class", "shared-across-classes"],
        default="independent-per-class",
        help=(
            "independent-per-class: each class receives distinct noise seeds; "
            "shared-across-classes: every class receives the same ordered "
            "set of --samples-per-class distinct noise seeds"
        ),
    )
    parser.add_argument("--seed-min", type=int, default=0)
    parser.add_argument("--seed-max", type=int, default=2**31 - 1)
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
