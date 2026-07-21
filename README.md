> README.md written by Claude under direction from njbrake. There may be mistakes.
> 
# laguna-otari-bridge

Run [`poolside/Laguna-S-2.1`](https://huggingface.co/poolside/Laguna-S-2.1) —
a 118B / 8B-active MoE — locally on Apple Silicon with llama.cpp + Metal, and
expose it as an authenticated OpenAI-compatible endpoint.

Measured on an M4 Max Mac Studio (128GB): **~40 tok/s** generation, ~470 tok/s
prompt processing, at 128K context.

## Findings

Most of the value here is the things that are not documented elsewhere, and
that cost time to discover.

### DFlash speculative decoding gets 0% draft acceptance here

The model card recommends the DFlash drafter. Enabling it made generation
several times *slower*:

| Config | gen tok/s |
|---|---|
| spec decoding **off** | **39.5** |
| spec decoding on, n-max=4 | 15.9 |
| spec decoding on, n-max=15 | 6.9 |

The cause is visible in the server log:

```
draft acceptance = 0.00000 (0 accepted / 8865 generated), mean len = 1.00
```

**Not one** drafted token was ever accepted. Every drafted token is wasted
compute, and the waste is proportional to `n-max` — which is exactly the
slowdown pattern above.

**This is a failure, not a tuning curve.** A 0% acceptance rate means the
drafter is not working, so do not read the table as "speculative decoding is
bad for this model." Correctly configured, it may well help; these numbers
say nothing about its potential.

Root cause is **not established**. What is known:

- The flags match the model card exactly, and the draft GGUF's metadata is
  correct (`dflash.decoder_arch = laguna`, `block_size = 16`).
- DFlash is EAGLE-style: the drafter consumes **hidden states extracted from
  the target model's internal layers** (`llama_set_embeddings_layer_inp`). If
  that extraction path is broken or unimplemented on the Metal backend, the
  drafter would receive garbage and accept nothing — consistent with what is
  observed, but **unverified**.
- Startup logs an initialization failure that is annotated as benign:
  `dflash requires ctx_other to be set (this warning is normal during memory
  fitting)`. Whether it is genuinely benign here is untested.

The obvious next experiment is comparing draft acceptance on the CPU backend
against Metal. If CPU accepts and Metal does not, it is a backend gap.

### Runtime options

- **Upstream llama.cpp does not support Laguna-S.** PR
  [#25165](https://github.com/ggml-org/llama.cpp/pull/25165) covers only XS.2
  and M.1, and is still open. Use
  [poolside's fork](https://github.com/poolsideai/llama.cpp), branch `laguna`,
  which added Laguna-S.2 support on 2026-07-20.
- **MLX**: poolside publishes an official MLX export,
  [`Laguna-S-2.1-NVFP4-mlx`](https://huggingface.co/poolside/Laguna-S-2.1-NVFP4-mlx)
  (4-bit, 71.9GB). This repo has **not** benchmarked it against the llama.cpp
  path; it may be the better option on Apple Silicon. Note that reports of
  `mlx-lm` failing to recognise the architecture refer to *community quants of
  Laguna-XS.2*, which is a different model from official S 2.1.

### A latent Metal f16 overflow can cause empty output

Metal's `MUL_MAT_ID` casts its activation input to f16. Laguna produces large
activations in later layers, and anything above the f16 max of 65504 overflows
to NaN — the model then returns **empty output**.

The fix is not merged anywhere: upstream PR
[#25442](https://github.com/ggml-org/llama.cpp/pull/25442) is open, and an
earlier attempt (#25389) was closed on a contributor-guideline technicality
rather than on technical grounds.

**It did not reproduce for S 2.1 here** (verified with short and 8.2K-token
prompts), and the reports were against XS 2.1 — but treat it as latent rather
than fixed. `metal-moe-f16-overflow.patch` in this repo applies cleanly if
empty output ever appears.

### Reasoning tokens count against `max_tokens`

Laguna is a reasoning model and returns its reasoning in a separate
`reasoning_content` field. Those tokens consume `max_tokens`, so a low limit
yields an empty `content` — which looks identical to the Metal bug above.
Give clients generous headroom.

### The server ignores the requested model id

`llama-server` serves whatever single model it loaded, regardless of the
`model` field in the request. A stale client config gets silently answered by
the wrong model and looks perfectly healthy. The Caddy gate in
[`otari-bridge/`](otari-bridge/) rejects a mismatched id rather than letting
that happen.

## Requirements

- Apple Silicon Mac. 128GB unified memory for the Q4_K_M (75GB) quant — Q8_0
  is 128GB and F16 is 235GB, both out of reach.
- ~80GB free disk.
- cmake, and a Tailscale account if you want the public endpoint.

## Setup

```bash
# 1. build poolside's fork with Metal
git clone --branch laguna https://github.com/poolsideai/llama.cpp
cd llama.cpp
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 14
cd ..

# 2. fetch weights (75GB) -- note: pass filenames positionally, NOT via
#    --include, which the CLI silently ignores when filenames are given
pip install huggingface_hub
hf download poolside/Laguna-S-2.1-GGUF laguna-s-2.1-Q4_K_M.gguf --local-dir ./models

# 3. serve
llama.cpp/build/bin/llama-server \
  --model models/laguna-s-2.1-Q4_K_M.gguf \
  --alias laguna-s-2.1 \
  --host 127.0.0.1 --port 8000 \
  --ctx-size 131072 \
  --n-gpu-layers 999 \
  --flash-attn on \
  --jinja
```

`--jinja` is required — Laguna ships a Jinja chat template and tool-call
parser. Tool calling works and returns proper `finish_reason: tool_calls`.

### Context and memory

75GB of weights plus KV cache. Because 36 of the 48 layers are sliding-window
(512 tokens), KV scales with only the 12 global-attention layers — roughly 7GB
at 128K and 13GB at 256K, far cheaper than the layer count suggests.

macOS caps GPU-wired memory at ~75% of RAM by default, which 128K fits under.
For the full 256K:

```bash
sudo sysctl iogpu.wired_limit_mb=114688   # 112GB; resets on reboot
```

## Exposing it

[`otari-bridge/`](otari-bridge/) puts it behind a bearer-token gate on a public
HTTPS URL, for Otari or any other OpenAI-compatible client:

```
Internet ──TLS:443──▶ Tailscale Funnel ──▶ Caddy :9000 (auth) ──▶ 127.0.0.1:8000 (llama-server)
```

See [`otari-bridge/README.md`](otari-bridge/README.md).

## License

Scripts in this repo: MIT (see [LICENSE](LICENSE)). The model itself is
OpenMDW-1.1, set by poolside.
