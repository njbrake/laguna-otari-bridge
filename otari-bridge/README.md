# Laguna-S 2.1 ↔ Otari bridge

Runs `poolside/Laguna-S-2.1` locally on the Mac Studio via llama.cpp + Metal,
and exposes it to Otari as an OpenAI-compatible endpoint.

```
Internet ──TLS:443──▶ Tailscale Funnel ──▶ Caddy :9000 (bearer check) ──▶ 127.0.0.1:8000 (llama-server, Metal)
```

Same shape as the ds4 bridge, and it **reuses the same bearer token**, so
Otari's `api_key` does not change — only the model id does.

## The model

118B total / 8B active MoE, 48 layers (12 global attention + 36 sliding-window
at 512 tokens), 256K context, OpenMDW-1.1 license.

Only `Q4_K_M` (75GB) fits in 128GB of unified memory. Q8_0 is 128GB and F16 is
235GB — both are out on this machine.

| File | Size | Used |
|---|---|---|
| `laguna-s-2.1-Q4_K_M.gguf` | 75.2 GB | main weights |
| `laguna-s-2.1-DFlash-BF16.gguf` | 2.2 GB | speculative-decoding drafter |

## Why this llama.cpp and not MLX / stock llama.cpp

- **MLX is not an option.** `mlx-lm` does not recognise the Laguna
  architecture at all.
- **Upstream llama.cpp does not support Laguna-S.** PR #25165 covers only
  XS.2 and M.1, and is still open.
- So we build **poolside's fork**, branch `laguna`, which added
  `laguna : add Laguna-S.2 (chat template, autoparser test, 48-layer type)`
  on 2026-07-20. Built here with `-DGGML_METAL=ON`.

```bash
git clone --branch laguna https://github.com/poolsideai/llama.cpp
cd llama.cpp && cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 14
```

## Measured performance (M4 Max, 128GB)

| Config | gen tok/s |
|---|---|
| **spec decoding off (default)** | **39.5 local / 49.3 via funnel** |
| spec decoding on, n-max=4 | 15.9 |
| spec decoding on, n-max=15 | 6.9 |

Prompt processing ~470 tok/s. Model load ~40s. RSS 82.6GB at 128K context,
with swap essentially untouched.

### Why speculative decoding is off despite the model card recommending it

Because the drafter never lands a single token:

```
draft acceptance = 0.00000 (0 accepted / 8865 generated), mean len = 1.00
```

Every drafted token is wasted work, proportional to `n-max` — hence the
39.5 → 15.9 → 6.9 progression. This is a broken drafter, not evidence that
speculative decoding is inherently unsuited to this model; see the root-cause
notes in the [top-level README](../README.md).

Re-test after a fork update with `SPEC=1 ./run-bridge.sh` (or `SPEC=1 SPEC_N=8`).

## ⚠️ Known Metal bug — did NOT reproduce here, but read this if you get empty output

Metal's `MUL_MAT_ID` casts its activation input to f16. Laguna produces large
activations in later layers, and anything above the f16 max of 65504 overflows
to NaN — the model then returns **empty output**. It is a real, reproduced bug
on Apple Silicon.

The fix is *not merged anywhere*: upstream PR #25442 is open, and an earlier
attempt (#25389) was closed on a contributor-guideline technicality rather than
on technical grounds. Poolside's fork does not carry it either.

**It did not reproduce on this setup** — verified with both a short prompt and
an 8.2K-token prompt, which returned correct non-empty output. The reports were
against Laguna XS 2.1; S 2.1 may simply not push activations past the f16 limit
in the same way. Treat it as latent rather than fixed: it could still surface on
a much longer or unusual prompt, and empty output is the symptom to watch for.

If generation comes back empty, apply the saved patch:

```bash
cd llama.cpp && git apply ../metal-moe-f16-overflow.patch
cmake --build build --config Release -j 14
```

It rescales each column by its L2 norm before the down-projection and undoes
the scaling after — a per-column mathematical identity, so it costs ~1 ULP of
accuracy plus a little speed.

## Memory

75GB of weights plus KV cache. Because 36 of the 48 layers are sliding-window
(512 tokens), KV scales with only the 12 global layers — roughly 7GB at 128K
and 13GB at 256K, far cheaper than the layer count suggests.

macOS caps GPU-wired memory at about 75% of RAM (~96GB here) by default. 128K
context fits under that. For the full 256K, raise it:

```bash
sudo sysctl iogpu.wired_limit_mb=114688   # 112GB; resets on reboot
```

## Run

```bash
# stop the ds4 bridge first -- same ports, same funnel
./run-bridge.sh          # defaults to 128K context
CTX=262144 ./run-bridge.sh   # full 256K (raise wired limit first)
```

## Verify

```bash
URL=https://<your-node>.<your-tailnet>.ts.net
TOKEN=$(cat .token)

curl -s $URL/v1/models                                      # -> 401
curl -s $URL/v1/models -H "Authorization: Bearer $TOKEN"    # -> 200 JSON
```

## Point Otari at it

`api_base` and `api_key` are unchanged from the ds4 setup; only the model id
differs:

```yaml
providers:
  openai:
    api_base: "https://<your-node>.<your-tailnet>.ts.net/v1"
    api_key: "<same token as before>"
```

Call it with model id `openai:laguna-s-2.1`.
