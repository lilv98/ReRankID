"""Shared, reusable pieces of the ReRankID pipeline. Import as `from utils import config` etc.
(entry scripts put `code/` on sys.path first — see the bootstrap header any runnable script starts
with). Modules:

  config      repo-rooted paths (self-contained tree), API keys, GPU count, W&B project
  prompts     item rendering (SID / caption) + the listwise prompt pieces + the leak-word filter
  metrics     NDCG / MRR / Recall, baselines, parse_topk, strict-parse aggregation
  data_io     bench/pool parquet -> row dicts, json dump
  raw_data    raw OpenOneRec parquet paths + negative-pool / mask / profile primitives
  llm         unified frontier-LLM chat client (cu128: OpenRouter; cu130: Vertex AI)
  sid_engine  the vLLM two-stage listwise generation engine (eval scoring + RFT sampling)
"""
