#!/bin/bash

# Define common arguments for all scripts
PRED_DIR="Benchmarking_QA_res"
OUTPUT_DIR="Benchmarking_QA_score"
API_KEY="sk-HDFRLuIBlMC7bOl38fuKm2OKY7a1noiB3437PI0T6D3OiQdr"
NUM_TASKS="5"

# Run the "correctness" evaluation script
python quantitative_evaluation/evaluate_benchmark_1_correctness.py \
  --pred_path "${PRED_DIR}/res_generic_cd.json" \
  --output_dir "${OUTPUT_DIR}/correctness_eval_cd" \
  --output_json "${OUTPUT_DIR}/correctness_cd_results.json" \
  --api_key $API_KEY \
  --num_tasks $NUM_TASKS

# Run the "detailed orientation" evaluation script
python quantitative_evaluation/evaluate_benchmark_2_detailed_orientation.py \
  --pred_path "${PRED_DIR}/res_generic_cd.json" \
  --output_dir "${OUTPUT_DIR}/detailed_eval_cd" \
  --output_json "${OUTPUT_DIR}/detailed_orientation_cd_results.json" \
  --api_key $API_KEY \
  --num_tasks $NUM_TASKS

# Run the "contextual understanding" evaluation script
python quantitative_evaluation/evaluate_benchmark_3_context.py \
  --pred_path "${PRED_DIR}/res_generic_cd.json" \
  --output_dir "${OUTPUT_DIR}/context_eval_cd" \
  --output_json "${OUTPUT_DIR}/contextual_understanding_cd_results.json" \
  --api_key $API_KEY \
  --num_tasks $NUM_TASKS

# Run the "temporal understanding" evaluation script
python quantitative_evaluation/evaluate_benchmark_4_temporal.py \
  --pred_path "${PRED_DIR}/res_temporal_cd.json" \
  --output_dir "${OUTPUT_DIR}/temporal_eval_cd" \
  --output_json "${OUTPUT_DIR}/temporal_cd_results.json" \
  --api_key $API_KEY \
  --num_tasks $NUM_TASKS

# Run the "consistency" evaluation script
python quantitative_evaluation/evaluate_benchmark_5_consistency.py \
  --pred_path "${PRED_DIR}/res_consistency_cd.json" \
  --output_dir "${OUTPUT_DIR}/consistency_eval_cd" \
  --output_json "${OUTPUT_DIR}/consistency_cd_results.json" \
  --api_key $API_KEY \
  --num_tasks $NUM_TASKS


echo "All evaluations completed!"
