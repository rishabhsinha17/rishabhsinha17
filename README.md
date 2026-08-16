# Hi, I'm Rishabh 👋

Software engineer at **Google**, working on LLM post-training and serving — QLoRA/SFT post-training on TPUs, and latency work on retrieval and serving paths (sub-25 ms p99 at 100+ QPS). Previously built multi-agent orchestration and eval harnesses on Bedrock at **AWS**. Part-time MIDS at **UC Berkeley** (Dec 2026); B.S. in Computer Science & Mathematics, University of Maryland '25.

My open-source work is in LLM inference internals: KV-cache management, layout, and quantization; prefill-decode disaggregation; deterministic kernels; and GPU transfer engines.

## 🔧 Open source

**[kvcached](https://github.com/ovg-project/kvcached)** — GPU KV-cache virtualization for vLLM/SGLang. Three merged PRs:
- Per-token-head **FP8 KV-cache quantization** in the vLLM integration, wired through `cache_dtype_str` forwarding ([#433](https://github.com/ovg-project/kvcached/pull/433))
- Allocation rollback in `KVCacheManager.alloc()` — a mid-allocation `alloc_page()` failure can no longer leave partially-allocated pages behind ([#430](https://github.com/ovg-project/kvcached/pull/430))
- Sleep-manager test suite brought under pytest via `pytest-asyncio` so it runs in CI ([#432](https://github.com/ovg-project/kvcached/pull/432))

**[LMCache](https://github.com/LMCache/LMCache)** — KV-cache layer for LLM serving
- Declarative `KVLayoutDescriptor` that states each engine's physical KV-cache layout as data, with a bijection to all 16 `EngineKVFormat` members pinned by an exhaustive round-trip test — step 1 of the standardized KV-layout RFC ([#4592](https://github.com/LMCache/LMCache/pull/4592))

**[SGLang](https://github.com/sgl-project/sglang)**
- Arch-aware persistent matmul configs fixing deterministic-mode `OutOfResources` shared-memory failures on sm_89 (Ada) ([#34486](https://github.com/sgl-project/sglang/pull/34486))
- Truth-table wire tests for `is_dummy` handling across the Mooncake and NIXL prefill-decode disaggregation backends ([#34977](https://github.com/sgl-project/sglang/pull/34977))

**[NIXL](https://github.com/ai-dynamo/nixl)** — use-after-free fix in the POSIX backend when a transfer is released while still in progress ([#2077](https://github.com/ai-dynamo/nixl/pull/2077))

**[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** — JIT builds now warn when CUDA toolkit < 12.8 silently drops compute-capability flags ([#4429](https://github.com/flashinfer-ai/flashinfer/pull/4429))

**[LLM Compressor](https://github.com/vllm-project/llm-compressor)** — rejection of over-length untruncated calibration samples when `max_seq_length` is unset, preventing silent calibration skew ([#3012](https://github.com/vllm-project/llm-compressor/pull/3012))

## 📄 Research

- **[Latent-State Auditing for Tool-Using Language Agents](https://github.com/rishabhsinha17/latent-state-auditing)** — solo author, under review. Hooks security-critical decision points in agent eval traces (AgentDojo/InjecAgent-style), captures residual-stream activations, and trains cheap probes that flag latent risk before tool execution — AUROC **0.85–0.88** on AgentDojo.
- **[SuperRed: AI Red-Teaming & Security Benchmarking](https://superred.simonsure.com/)** — Sure, Munshi, Sinha, Wang, Song. ICLR 2027 submission. Two merged harness-hardening PRs: fixes for three task-lifecycle paths that defeated or misreported the per-task time cap ([#46](https://github.com/RoldSI/superred/pull/46)), and endpoint/credential-alias stripping ([#45](https://github.com/RoldSI/superred/pull/45)). ([code](https://github.com/RoldSI/superred))
- **A Nonzero Final LR Floor Improves Transfer of Agent-Discovered Schedule Tweaks** — solo author. NeurIPS 2026 submission.

## 🛠 Projects

- **[low-latency-llm-inference-server](https://github.com/rishabhsinha17/low-latency-llm-inference-server)** — Llama-3-8B at **120 tokens/s** on an A10G via vLLM, behind a batched gRPC/Protobuf gateway that cut p99 latency **40%** under 32-request concurrency vs. REST. Prometheus + DCGM observability.
- **[slurm-vision-rag-platform](https://github.com/rishabhsinha17/slurm-vision-rag-platform)** — LLaVA-1.5-7B fine-tuned on a 4× A100 Slurm partition with gradient checkpointing to stay under 80 GB/GPU; RAG API over **1M** CLIP vectors in Pinecone serving **50 QPS** at **250 ms** p99.
- **[rl-hyperparam-tuner](https://github.com/rishabhsinha17/rl-hyperparam-tuner)** — PPO agent (Stable-Baselines3) that auto-tunes ResNet-18 learning rates; JAX `pmap` harness benchmarking a **910 img/s** multi-GPU inference ceiling.

## 🧰 Stack

**Languages:** Python · C++ · Go · Rust · Java · TypeScript · SQL

**ML & systems:** PyTorch · JAX · vLLM · SGLang · CUDA · TPUs · Kubernetes · Ray · gRPC · Slurm · Prometheus · OTel · AWS

## 🌐 Links

[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://linkedin.com/in/rishabhsinha17)
[![Website](https://img.shields.io/badge/rishabhsinha.dev-222222?logo=safari&logoColor=white)](https://rishabhsinha.dev)
[![Email](https://img.shields.io/badge/Email-D14836?logo=gmail&logoColor=white)](mailto:rsinha17@terpmail.umd.edu)

![GitHub stats](https://github-readme-stats.vercel.app/api?username=rishabhsinha17&hide_border=true&include_all_commits=true&count_private=true)
