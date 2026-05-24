# vLLM setup [VM]

## Qwen3.5-4B

```bash

# q35_4b_base-conv
vllm serve Qwen/Qwen3.5-4B \
  --served-model-name qwen3.5-4b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --language-model-only \
  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}'

# q35_4b_prefix-cache
vllm serve Qwen/Qwen3.5-4B \
  --served-model-name qwen3.5-4b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --language-model-only \
  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-prefix-caching

# q35_4b_mtp-2
vllm serve Qwen/Qwen3.5-4B \
  --served-model-name qwen3.5-4b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --language-model-only \
  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'

# q35_4b_dflash-8
vllm serve Qwen/Qwen3.5-4B \
  --served-model-name qwen3.5-4b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 80000 \
  --language-model-only \
  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{"method": "dflash", "model": "z-lab/Qwen3.5-4B-DFlash", "num_speculative_tokens": 8}'

# q35_4b_dflash-2
vllm serve Qwen/Qwen3.5-4B \
  --served-model-name qwen3.5-4b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 80000 \
  --language-model-only \
  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{"method": "dflash", "model": "z-lab/Qwen3.5-4B-DFlash", "num_speculative_tokens": 2}'

# q35_4b_ngram
vllm serve Qwen/Qwen3.5-4B \
  --served-model-name qwen3.5-4b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 80000 \
  --language-model-only \
  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 2, "prompt_lookup_min": 2, "prompt_lookup_max": 2}'

# NOT WORKING - incompatible heads (draft model Qwen3.5-0.8B)
vllm serve Qwen/Qwen3.5-4B \
  --served-model-name qwen3.5-4b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 80000 \
  --language-model-only \
  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{"method": "draft_model", "ngram": "Qwen/Qwen3.5-0.8B", "num_speculative_tokens": 2}'

```

## Qwen3-4B

```bash

# q3_4b_base
vllm serve Qwen/Qwen3-4B \
  --served-model-name qwen3-4b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 35000 \
  --language-model-only \
  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}'

# q3_4b_prefix-cache
vllm serve Qwen/Qwen3-4B \
  --served-model-name qwen3-4b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 35000 \
  --language-model-only \
  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-prefix-caching

# q3_4b_eagle3-2
vllm serve Qwen/Qwen3-4B \
  --served-model-name qwen3-4b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 35000 \
  --language-model-only \
  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{"method": "eagle3","model": "AngelSlim/Qwen3-4B_eagle3","num_speculative_tokens": 2,"draft_tensor_parallel_size": 1}'

# q3_4b_dflash-2
vllm serve Qwen/Qwen3-4B \
  --served-model-name qwen3-4b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 35000 \
  --language-model-only \
  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{"method": "dflash", "model": "z-lab/Qwen3-4B-DFlash-b16", "num_speculative_tokens": 2}'

# q3_4b_spec-draft-0.6b
vllm serve Qwen/Qwen3-4B \
  --served-model-name qwen3-4b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 35000 \
  --language-model-only \
  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{"method": "draft_model", "model": "Qwen/Qwen3-0.6B", "num_speculative_tokens": 2}'

# q3_4b_ngram
vllm serve Qwen/Qwen3-4B \
  --served-model-name qwen3-4b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 35000 \
  --language-model-only \
  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 2, "prompt_lookup_min": 2, "prompt_lookup_max": 2}'

```


# execution [local]

## trigger (parametrized template)
```bash
export EXPERIMENT_PREFIX=q35_4b_base-conv
export NUM_CONVERSATIONS=64
export MAX_CONVERSATIONS=64
export MAX_TURNS_PER_CONV=16
export MAX_SESSIONS=32
export MAX_TOKENS=1024

export TECHNIQUE=${EXPERIMENT_PREFIX}-num-conv-${NUM_CONVERSATIONS}-max-turns-${MAX_TURNS_PER_CONV}-max-sessions-${MAX_SESSIONS}-max-tokens-${MAX_TOKENS}
```

```bash
python sharegpt_turn_bench.py \
  --technique ${TECHNIQUE} \
  --mode static \
  --num-conversations ${NUM_CONVERSATIONS} \
  --max-conversations ${MAX_CONVERSATIONS} \
  --max-turns-per-conv ${MAX_TURNS_PER_CONV} \
  --max-sessions ${MAX_SESSIONS} \
  --max-tokens ${MAX_TOKENS}
```

```bash
python scripts/export_experiments.py \        
  --log-path logs/gateway/gateway_metrics_2026-05-19.jsonl \
  --technique ${TECHNIQUE} \
  --output-csv data/experiments_${TECHNIQUE}.csv
```

