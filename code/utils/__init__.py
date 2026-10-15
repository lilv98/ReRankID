# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared library. Entry scripts put code/ on sys.path and import `from utils import config`.

  config             repository paths, pinned base models, API keys, GPU count
  prompts            item rendering (SID, caption, both) and the listwise prompts
  metrics            output parsing, NDCG/Recall/MRR, aggregation, paired bootstrap, Kendall tau
  data_io            benchmark parquet to row dicts, nested slates, JSON and stats output
  raw_data           raw RecIF paths, engagement masks and the negative pool
  pools              training pool grids and the shared 6k post-training subset
  llm                chat client for text-reading LLMs (OpenRouter or Vertex AI)
  sid_engine         vLLM two-stage listwise engine with LoRA hot-swap
  model_config       RoPE config fix for merged models (also a CLI)
  export_lora        verl FSDP checkpoint to PEFT LoRA adapter (CLI, OPD and RLVR)
  select_checkpoint  dev-set checkpoint selection by mean NDCG@5 (CLI)
  import_hook        patch a module on its first import (sitecustomize, OPD verl patches)
"""
