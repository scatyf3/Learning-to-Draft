#!/usr/bin/env bash

set -euo pipefail

Depth_Policy=${1:-}
Size_Policy=${2:-}

PYTHON_BIN=${PYTHON_BIN:-/home/external/miniconda3/envs/ea_env311/bin/python}
EVAL_GPU=${EVAL_GPU:-0}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-/mnt/hdd/hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}
EA_MODEL_PATH=${EA_MODEL_PATH:-yuhuili/EAGLE3-LLaMA3.1-Instruct-8B}
DATASETS_STR=${DATASETS:-alpaca mt_bench gsm8k}
MODES_STR=${MODES:-ar naive-eagle policy}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}

read -r -a DATASETS <<< "$DATASETS_STR"
read -r -a MODES <<< "$MODES_STR"

question_count() {
    local dataset=$1
    wc -l < "eagle/data/${dataset}/question.jsonl"
}

jsonl_count() {
    local file=$1
    if [ ! -f "$file" ]; then
        echo 0
        return
    fi
    wc -l < "$file"
}

baseline_output() {
    local dataset=$1
    echo "${dataset}/llama3p18b-baserebuttal-temperature-0.0.jsonl"
}

naive_eagle_output() {
    local dataset=$1
    echo "${dataset}/llama3p18brebuttal-temperature0.0depth-8token-60choices-1.jsonl"
}

policy_output() {
    local dataset=$1
    local depth_base size_base
    depth_base=$(basename "$Depth_Policy")
    size_base=$(basename "$Size_Policy")
    echo "${dataset}/llama3p18brebuttal-temperature0.0depth-8token-60choices-1$(basename "$(dirname "$Size_Policy")")${size_base%.zip}token$(basename "$(dirname "$Depth_Policy")")${depth_base%.zip}depth.jsonl"
}

is_complete() {
    local dataset=$1
    local output_file=$2
    local expected actual
    expected=$(question_count "$dataset")
    actual=$(jsonl_count "$output_file")
    [ "$actual" -ge "$expected" ]
}

prepare_output() {
    local dataset=$1
    local output_file=$2
    if [ -f "$output_file" ] && ! is_complete "$dataset" "$output_file"; then
        echo "  -> 删除不完整结果并重跑: ${output_file}"
        rm -f "$output_file"
    fi
}

run_ar() {
    local dataset=$1
    CUDA_VISIBLE_DEVICES=${EVAL_GPU} "${PYTHON_BIN}" -m eagle.evaluation.gen_baseline_answer_llama3chat \
        --bench-name "$dataset" \
        --base-model-path "${BASE_MODEL_PATH}" \
        --ea-model-path "${EA_MODEL_PATH}" \
        --temperature 0 \
        --num-choices 1
}

run_naive_eagle() {
    local dataset=$1
    CUDA_VISIBLE_DEVICES=${EVAL_GPU} "${PYTHON_BIN}" -m eagle.evaluation.gen_ea_answer_llama3chat \
        --bench-name "$dataset" \
        --base-model-path "${BASE_MODEL_PATH}" \
        --ea-model-path "${EA_MODEL_PATH}" \
        --depth 8 \
        --temperature 0 \
        --num-choices 1 \
        --total-token 60
}

run_policy() {
    local dataset=$1
    if [ -z "$Depth_Policy" ] || [ -z "$Size_Policy" ]; then
        echo "policy mode requires: $0 <depth_policy_zip> <size_policy_zip>"
        exit 1
    fi
    CUDA_VISIBLE_DEVICES=${EVAL_GPU} "${PYTHON_BIN}" -m eagle.evaluation.gen_ea_answer_llama3chat \
        --bench-name "$dataset" \
        --base-model-path "${BASE_MODEL_PATH}" \
        --ea-model-path "${EA_MODEL_PATH}" \
        --depth 8 \
        --temperature 0 \
        --num-choices 1 \
        --total-token 60 \
        --use_dyn_depth \
        --use_dyn_token \
        --token_model "$Size_Policy" \
        --depth_model "$Depth_Policy"
}

for mode in "${MODES[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        case "$mode" in
            ar)
                output_file=$(baseline_output "$dataset")
                ;;
            naive-eagle)
                output_file=$(naive_eagle_output "$dataset")
                ;;
            policy)
                output_file=$(policy_output "$dataset")
                ;;
            *)
                echo "Unknown mode: $mode"
                exit 1
                ;;
        esac

        if [ "$SKIP_COMPLETED" = "1" ] && is_complete "$dataset" "$output_file"; then
            echo "  -> 跳过已完成: mode=${mode} dataset=${dataset} file=${output_file}"
            continue
        fi

        prepare_output "$dataset" "$output_file"

        echo "  -> 开始评测: mode=${mode} dataset=${dataset} gpu=${EVAL_GPU}"
        case "$mode" in
            ar)
                run_ar "$dataset"
                ;;
            naive-eagle)
                run_naive_eagle "$dataset"
                ;;
            policy)
                run_policy "$dataset"
                ;;
        esac
    done
done
