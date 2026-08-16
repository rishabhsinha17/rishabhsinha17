### 👋 I'm Rishabh.

Currently, I'm a software engineer at Google, working on LLM post-training and serving — QLoRA/SFT post-training on TPUs, and latency work on retrieval and serving paths (sub-25 ms p99 at 100+ QPS). I'm also finishing a part-time M.S. (MIDS) at UC Berkeley (Dec 2026); before that, a B.S. in Computer Science & Mathematics at the University of Maryland ('25).

My latest focus is the open-source LLM inference stack — KV-cache management, layout, and quantization; prefill–decode disaggregation; deterministic kernels; and GPU transfer engines:

- [kvcached](https://github.com/ovg-project/kvcached) virtualizes GPU KV-cache for vLLM/SGLang. Three merged PRs: per-token-head FP8 KV-cache quantization wired through `cache_dtype_str` forwarding ([#433](https://github.com/ovg-project/kvcached/pull/433)), allocation rollback in `KVCacheManager.alloc()` so a mid-allocation failure can't leave partially-allocated pages behind ([#430](https://github.com/ovg-project/kvcached/pull/430)), and the sleep-manager test suite brought under pytest-asyncio for CI ([#432](https://github.com/ovg-project/kvcached/pull/432)).
- [LMCache](https://github.com/LMCache/LMCache) is a KV-cache layer for LLM serving. A declarative `KVLayoutDescriptor` that states each engine's physical KV-cache layout as data, with a bijection to all 16 `EngineKVFormat` members pinned by an exhaustive round-trip test — step 1 of the standardized KV-layout RFC ([#4592](https://github.com/LMCache/LMCache/pull/4592)).
- [SGLang](https://github.com/sgl-project/sglang) is a high-performance LLM serving framework. Arch-aware persistent matmul configs fixing deterministic-mode `OutOfResources` shared-memory failures on sm_89 ([#34486](https://github.com/sgl-project/sglang/pull/34486)); truth-table wire tests for `is_dummy` handling across the Mooncake and NIXL prefill–decode disaggregation backends ([#34977](https://github.com/sgl-project/sglang/pull/34977)).
- [NIXL](https://github.com/ai-dynamo/nixl) is NVIDIA's inference transfer library. Use-after-free fix in the POSIX backend when a transfer is released while still in progress ([#2077](https://github.com/ai-dynamo/nixl/pull/2077)).
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) is a kernel library for LLM serving. JIT builds now warn when CUDA toolkit < 12.8 silently drops compute-capability flags ([#4429](https://github.com/flashinfer-ai/flashinfer/pull/4429)).
- [LLM Compressor](https://github.com/vllm-project/llm-compressor) applies compression algorithms to LLMs for optimized deployment. Rejection of over-length untruncated calibration samples when `max_seq_length` is unset, preventing silent calibration skew ([#3012](https://github.com/vllm-project/llm-compressor/pull/3012)).

Some systems I've built end to end:

- [low-latency-llm-inference-server](https://github.com/rishabhsinha17/low-latency-llm-inference-server) serves Llama-3-8B at 120 tokens/s on an A10G via vLLM, behind a batched gRPC/Protobuf gateway that cut p99 latency 40% under 32-request concurrency, with Prometheus + DCGM observability.
- [slurm-vision-rag-platform](https://github.com/rishabhsinha17/slurm-vision-rag-platform) fine-tunes LLaVA-1.5-7B on a 4× A100 Slurm partition (gradient checkpointing, under 80 GB/GPU) and serves a RAG API over 1M CLIP vectors at 50 QPS with 250 ms p99.
- [rl-hyperparam-tuner](https://github.com/rishabhsinha17/rl-hyperparam-tuner) trains a PPO agent to auto-tune ResNet-18 learning rates, with a JAX `pmap` harness benchmarking a 910 img/s multi-GPU inference ceiling.

I previously worked on agentic systems @aws — a Bedrock AgentCore/LangGraph multi-agent orchestrator (40% lower response latency via semantic caching) and deterministic agent eval harnesses with trace replay and guarded tool execution.

Publications:

- Under review: [Latent-State Auditing for Tool-Using Language Agents](https://github.com/rishabhsinha17/latent-state-auditing) hooks security-critical decision points in agent eval traces, captures residual-stream activations, and trains cheap probes that flag latent risk before tool execution (AUROC 0.85–0.88 on AgentDojo). Solo author.
- ICLR 2027 submission: [SuperRed: AI Red-Teaming & Security Benchmarking](https://superred.simonsure.com/) (Sure, Munshi, Sinha, Wang, Song) — plus two merged PRs hardening the benchmark harness: per-task time-cap enforcement across three task-lifecycle paths ([#46](https://github.com/RoldSI/superred/pull/46)) and endpoint/credential-alias stripping ([#45](https://github.com/RoldSI/superred/pull/45)).
- NeurIPS 2026 submission: A Nonzero Final LR Floor Improves Transfer of Agent-Discovered Schedule Tweaks. Solo author.

Reach me: [rishabhsinha.dev](https://rishabhsinha.dev) · [LinkedIn](https://www.linkedin.com/in/rishabhsinha17/) · [rsinha17@terpmail.umd.edu](mailto:rsinha17@terpmail.umd.edu)
