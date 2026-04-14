#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


DEFAULT_DATASETS = ["gsm8k", "mt_bench", "alpaca"]
DEFAULT_MODES = ["ar", "naive-eagle", "policy"]


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def question_count(root: Path, dataset: str) -> int:
    question_file = root / "eagle" / "data" / dataset / "question.jsonl"
    if not question_file.exists():
        raise FileNotFoundError(f"Missing question file: {question_file}")
    return count_lines(question_file)


def baseline_output(dataset: str) -> Path:
    return Path(dataset) / "llama3p18b-baserebuttal-temperature-0.0.jsonl"


def naive_eagle_output(dataset: str) -> Path:
    return Path(dataset) / "llama3p18brebuttal-temperature0.0depth-8token-60choices-1.jsonl"


def policy_output(dataset: str, depth_policy: Path, size_policy: Path) -> Path:
    depth_dir = depth_policy.parent.name
    size_dir = size_policy.parent.name
    depth_stem = depth_policy.stem
    size_stem = size_policy.stem
    name = (
        "llama3p18brebuttal-temperature0.0depth-8token-60choices-1"
        f"{size_dir}{size_stem}token"
        f"{depth_dir}{depth_stem}depth.jsonl"
    )
    return Path(dataset) / name


def is_complete(root: Path, dataset: str, output_file: Path) -> bool:
    expected = question_count(root, dataset)
    actual = count_lines(root / output_file)
    return expected > 0 and actual >= expected


def build_command(
    mode: str,
    dataset: str,
    output_file: Path,
    base_model_path: str,
    ea_model_path: str,
    depth_policy: Path,
    size_policy: Path,
) -> List[str]:
    if mode == "ar":
        return [
            sys.executable,
            "-m",
            "eagle.evaluation.gen_baseline_answer_llama3chat",
            "--bench-name",
            dataset,
            "--base-model-path",
            base_model_path,
            "--ea-model-path",
            ea_model_path,
            "--temperature",
            "0",
            "--num-choices",
            "1",
            "--answer-file",
            str(output_file),
        ]

    base_cmd = [
        sys.executable,
        "-m",
        "eagle.evaluation.gen_ea_answer_llama3chat",
        "--bench-name",
        dataset,
        "--base-model-path",
        base_model_path,
        "--ea-model-path",
        ea_model_path,
        "--depth",
        "8",
        "--temperature",
        "0",
        "--num-choices",
        "1",
        "--total-token",
        "60",
        "--answer-file",
        str(output_file),
    ]

    if mode == "naive-eagle":
        return base_cmd

    if mode == "policy":
        return base_cmd + [
            "--use_dyn_depth",
            "--use_dyn_token",
            "--token_model",
            str(size_policy),
            "--depth_model",
            str(depth_policy),
        ]

    raise ValueError(f"Unsupported mode: {mode}")


def resolve_output(
    mode: str,
    dataset: str,
    depth_policy: Path,
    size_policy: Path,
) -> Path:
    if mode == "ar":
        return baseline_output(dataset)
    if mode == "naive-eagle":
        return naive_eagle_output(dataset)
    if mode == "policy":
        return policy_output(dataset, depth_policy, size_policy)
    raise ValueError(f"Unsupported mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AR / Naive EAGLE / Policy EAGLE with resume checks.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to run. Default: gsm8k mt_bench alpaca",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=DEFAULT_MODES,
        choices=DEFAULT_MODES,
        help="Modes to run. Choices: ar naive-eagle policy",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=os.environ.get("EVAL_GPU", "0"),
        help="CUDA_VISIBLE_DEVICES value",
    )
    parser.add_argument(
        "--base-model-path",
        default=os.environ.get(
            "BASE_MODEL_PATH",
            "/mnt/hdd/hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
        ),
    )
    parser.add_argument(
        "--ea-model-path",
        default=os.environ.get("EA_MODEL_PATH", "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"),
    )                                                    
    parser.add_argument(
        "--depth-policy",
        default="checkpoints/depth/ppo_speculative_decoder_controller_v1_single_action.zip",
        help="Depth policy checkpoint for policy mode",
    )
    parser.add_argument(
        "--size-policy",
        default="checkpoints/size/ppo_speculative_decoder_controller_step_100000.zip",
        help="Size policy checkpoint for policy mode",
    )
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip dataset/mode if output already has full number of lines",
    )
    parser.add_argument(
        "--rerun-partial",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete and rerun partial output files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    depth_policy = (root / args.depth_policy).resolve()
    size_policy = (root / args.size_policy).resolve()

    if "policy" in args.modes:
        if not depth_policy.exists():
            raise FileNotFoundError(f"Missing depth policy: {depth_policy}")
        if not size_policy.exists():
            raise FileNotFoundError(f"Missing size policy: {size_policy}")

    for mode in args.modes:
        for dataset in args.datasets:
            output_file = resolve_output(mode, dataset, depth_policy, size_policy)
            output_path = root / output_file

            complete = is_complete(root, dataset, output_file)
            if args.skip_completed and complete:
                print(f"[skip-complete] mode={mode} dataset={dataset} file={output_file}")
                continue

            if output_path.exists() and not complete and args.rerun_partial:
                if args.dry_run:
                    print(f"[rerun-partial] would delete {output_file}")
                else:
                    print(f"[rerun-partial] deleting {output_file}")
                    output_path.unlink()

            cmd = build_command(
                mode=mode,
                dataset=dataset,
                output_file=output_file,
                base_model_path=args.base_model_path,
                ea_model_path=args.ea_model_path,
                depth_policy=depth_policy,
                size_policy=size_policy,
            )

            print(f"[run] mode={mode} dataset={dataset} gpu={args.gpu}")
            print("      " + " ".join(cmd))

            if args.dry_run:
                continue

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            subprocess.run(cmd, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
