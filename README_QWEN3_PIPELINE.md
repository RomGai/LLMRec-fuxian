# Simplified LLMRec Pipeline (Qwen3-8B)

This implementation rewrites the LLMRec workflow using conventional packages (`argparse`, `csv`, `json`, `numpy`, `torch`, `transformers`) and keeps the same 3-stage augmentation logic:

1. **u-i edge augmentation** (favorite/dislike from candidates)
2. **user profile augmentation**
3. **item attribute augmentation**
4. **BPR training with augmented signals**
5. **Ranking evaluation with 1 target + 1000 negatives** and metrics: HR/NDCG at @10/@20/@40

## Prompt consistency
The script keeps the original LLMRec prompt structure and key constraints (output format, no reasoning, candidate indices, profile fields, `director::country::language`).

## Qwen3-8B configuration
Uses:
- `AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")`
- `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype="auto", device_map="auto")`
- `apply_chat_template(..., enable_thinking=False)` (Think mode off)

## Run sequence
```bash
bash run_llmrec_new_data_qwen3.sh
```

or directly:
```bash
python llmrec_qwen3_pipeline.py --data_dir LLMRec-new-data --dataset Baby_Products --stage all --output_dir outputs_qwen3_pipeline
```
