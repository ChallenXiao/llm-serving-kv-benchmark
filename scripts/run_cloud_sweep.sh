#!/usr/bin/env bash
set -euo pipefail

ENGINE=${ENGINE:-"vllm"}
BASE_URL=${BASE_URL:-"http://127.0.0.1:8000/v1"}
MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}

RUN_ID=${RUN_ID:-$(date +"%Y%m%d_%H%M%S")}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-"experiments/${ENGINE}_${RUN_ID}"}

RAW_DIR="${EXPERIMENT_DIR}/raw"
PROCESSED_DIR="${EXPERIMENT_DIR}/processed"
PLOT_DIR="${EXPERIMENT_DIR}/plots"
LOG_DIR="${EXPERIMENT_DIR}/logs"

mkdir -p "${RAW_DIR}" "${PROCESSED_DIR}" "${PLOT_DIR}" "${LOG_DIR}"

SHORT_CONCURRENCIES=${SHORT_CONCURRENCIES:-"1 4 8 16 32 64"}
LONG_CONCURRENCIES=${LONG_CONCURRENCIES:-"1 2 4 8 16"}
SHARED_CONCURRENCIES=${SHARED_CONCURRENCIES:-"1 4 8 16 32"}

NUM_REQUESTS_SHORT=${NUM_REQUESTS_SHORT:-128}
NUM_REQUESTS_LONG=${NUM_REQUESTS_LONG:-64}
NUM_REQUESTS_SHARED=${NUM_REQUESTS_SHARED:-128}

SHORT_INPUT_TOKENS=${SHORT_INPUT_TOKENS:-256}
LONG_INPUT_TOKENS=${LONG_INPUT_TOKENS:-2048}
SHARED_PREFIX_TOKENS=${SHARED_PREFIX_TOKENS:-2048}

SHORT_MAX_TOKENS=${SHORT_MAX_TOKENS:-256}
LONG_MAX_TOKENS=${LONG_MAX_TOKENS:-256}
SHARED_MAX_TOKENS=${SHARED_MAX_TOKENS:-256}

TIMEOUT_S=${TIMEOUT_S:-600}

echo "========== Cloud Sweep Config =========="
echo "ENGINE=${ENGINE}"
echo "BASE_URL=${BASE_URL}"
echo "MODEL=${MODEL}"
echo "EXPERIMENT_DIR=${EXPERIMENT_DIR}"
echo "SHORT_CONCURRENCIES=${SHORT_CONCURRENCIES}"
echo "LONG_CONCURRENCIES=${LONG_CONCURRENCIES}"
echo "SHARED_CONCURRENCIES=${SHARED_CONCURRENCIES}"

cat > "${EXPERIMENT_DIR}/run_config.txt" <<CONFIG
ENGINE=${ENGINE}
BASE_URL=${BASE_URL}
MODEL=${MODEL}
RUN_ID=${RUN_ID}
EXPERIMENT_DIR=${EXPERIMENT_DIR}
SHORT_CONCURRENCIES=${SHORT_CONCURRENCIES}
LONG_CONCURRENCIES=${LONG_CONCURRENCIES}
SHARED_CONCURRENCIES=${SHARED_CONCURRENCIES}
NUM_REQUESTS_SHORT=${NUM_REQUESTS_SHORT}
NUM_REQUESTS_LONG=${NUM_REQUESTS_LONG}
NUM_REQUESTS_SHARED=${NUM_REQUESTS_SHARED}
SHORT_INPUT_TOKENS=${SHORT_INPUT_TOKENS}
LONG_INPUT_TOKENS=${LONG_INPUT_TOKENS}
SHARED_PREFIX_TOKENS=${SHARED_PREFIX_TOKENS}
SHORT_MAX_TOKENS=${SHORT_MAX_TOKENS}
LONG_MAX_TOKENS=${LONG_MAX_TOKENS}
SHARED_MAX_TOKENS=${SHARED_MAX_TOKENS}
TIMEOUT_S=${TIMEOUT_S}
CONFIG

echo
echo "Checking server models endpoint..."
curl -s "${BASE_URL}/models" | tee "${LOG_DIR}/models_endpoint.json" || true
echo

run_case () {
  local workload=$1
  local concurrency=$2
  local num_requests=$3
  local input_tokens=$4
  local shared_prefix_tokens=$5
  local max_tokens=$6

  local output_file="${RAW_DIR}/${ENGINE}_${workload}_c${concurrency}.jsonl"
  local log_file="${LOG_DIR}/${ENGINE}_${workload}_c${concurrency}.log"

  echo
  echo "========== Running ${ENGINE} ${workload} c=${concurrency} =========="

  python -m benchmark.run_benchmark \
    --engine "${ENGINE}" \
    --base-url "${BASE_URL}" \
    --model "${MODEL}" \
    --workload "${workload}" \
    --concurrency "${concurrency}" \
    --num-requests "${num_requests}" \
    --input-tokens "${input_tokens}" \
    --shared-prefix-tokens "${shared_prefix_tokens}" \
    --max-tokens "${max_tokens}" \
    --timeout-s "${TIMEOUT_S}" \
    --output "${output_file}" | tee "${log_file}"
}

echo
echo "========== Short Prompt Sweep =========="
for c in ${SHORT_CONCURRENCIES}; do
  run_case "short" "${c}" "${NUM_REQUESTS_SHORT}" "${SHORT_INPUT_TOKENS}" 1024 "${SHORT_MAX_TOKENS}"
done

echo
echo "========== Long Prompt Sweep =========="
for c in ${LONG_CONCURRENCIES}; do
  run_case "long" "${c}" "${NUM_REQUESTS_LONG}" "${LONG_INPUT_TOKENS}" 1024 "${LONG_MAX_TOKENS}"
done

echo
echo "========== Shared Prefix Sweep =========="
for c in ${SHARED_CONCURRENCIES}; do
  run_case "shared_prefix" "${c}" "${NUM_REQUESTS_SHARED}" 128 "${SHARED_PREFIX_TOKENS}" "${SHARED_MAX_TOKENS}"
done

echo
echo "========== Generating Summary and Plots =========="
python -m benchmark.plot_results \
  --raw-dir "${RAW_DIR}" \
  --summary-output "${PROCESSED_DIR}/summary.csv" \
  --plot-dir "${PLOT_DIR}" | tee "${LOG_DIR}/plot_results.log"

echo
echo "========== Packaging Experiment =========="
tar -czf "${EXPERIMENT_DIR}.tar.gz" "${EXPERIMENT_DIR}"

echo
echo "Done."
echo "Experiment directory: ${EXPERIMENT_DIR}"
echo "Archive: ${EXPERIMENT_DIR}.tar.gz"
echo
echo "Important files:"
echo "- ${PROCESSED_DIR}/summary.csv"
echo "- ${PLOT_DIR}/request_throughput_vs_concurrency.png"
echo "- ${PLOT_DIR}/output_throughput_vs_concurrency.png"
echo "- ${PLOT_DIR}/p95_latency_vs_concurrency.png"
echo "- ${PLOT_DIR}/p95_ttft_vs_concurrency.png"
