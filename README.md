# Qwen3.8-Flash-Next with MTP: 10 → 16.5 t/s on 16 GB VRAM + 64 GB RAM

A measured, copy-pasteable setup for running **Qwen3.8-Flash-Next** at **131k context**
(q8_0 KV) on a card like an **RTX 4080 16 GB**, using the **AtomicChat AD-4.27bpw Q4_K_M**
quant and the **Unsloth shared MTP head**. Numbers are from real agent traffic on:

| part | spec |
|---|---|
| GPU | RTX 4080 16 GB |
| CPU | i5-13600K, hyperthreading off, `taskset 0-11` |
| RAM | 64 GB |
| Disk | NVMe, ~95 GB for the model shards |

If you are on the Unsloth IQ4_XS shards and seeing ~10 t/s, this whole setup replaces that.
Expect roughly **50-55 GB resident RAM** while serving: the model is memory-mapped and its
experts stream from system RAM, so decode speed is RAM-bandwidth-bound. That is also why
64 GB is the floor here.

## Step 1 - Build

Plain upstream master is not enough: it accepts `--spec-type draft-mtp` but cannot load
this model's MTP head yet. The needed code is upstream PR
[#28243](https://github.com/ggml-org/llama.cpp/pull/28243) plus one small fix. I carry
both as a branch on my fork: 11 cherry-picks of the PR, then one commit declaring
`nextn.hc_head_norm` as `{ n_embd, hc }` + `TENSOR_ALLOW_RESHAPE` (without it the load
asserts in `graph_mtp` on recent masters).

```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama-mtp
cd ~/llama-mtp
git remote add dtm https://github.com/dtm-beep/llama.cpp
git fetch dtm pr-mtp-fix
git checkout -b qwen38-mtp dtm/pr-mtp-fix
mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . -j -- llama-server
```

The checkout is a fixed, consistent tree (base `fb27a525d`), no matter how far master has
moved since. If PR #28243 has merged by the time you read this, plain master is enough,
skip the fetch.

## Step 2 - Download models

```bash
pip install -U "huggingface_hub[cli]"

# target, ~95 GB across 33 shards. This Q4_K_M quant is faster AND better than
# the IQ4_XS shards: KLD 0.0842, 89.5% same-top-1 vs BF16
hf download AtomicChat/Qwen3.8-Flash-Next-GGUF \
  --include "Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64*" \
  --local-dir ~/models/qwen38

# MTP draft head, ~1.9 GB. Get the SHARED variant: it borrows the target's
# embeddings and lm head instead of carrying its own, saving ~1.4 GB VRAM
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  "MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf" \
  --local-dir ~/models/qwen38
```

## Step 3 - Run

```bash
LLAMA_ATTN_ROT_DISABLE=1 GGML_CUDA_NO_PINNED=1 taskset -c 0-11 \
~/llama-mtp/build/bin/llama-server \
  -m ~/models/qwen38/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf \
  -md ~/models/qwen38/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf \
  --alias qwen38 \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --spec-draft-cpu-moe \
  --load-mode none --lazy-mode on \
  --fit on \
  --fit-target 100 \
  -ctk q8_0 -ctv q8_0 \
  -c 131072 \
  -b 2048 -ub 2048 \
  -t 6 -tb 12 \
  --cache-ram 2048 --ctx-checkpoints 0 \
  --host 127.0.0.1 --port 8080 \
  --jinja \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --repeat-penalty 1.0
```

This block is the config currently serving my daily driver, with every value verified by
A/B this month (see the notes after it). Two deliberate variations:

- **`-ub` 2048 vs 1024**: 2048 is for agent/compaction-style harnesses that re-send large
  prompts (measured here: a fresh 60k prompt in 161 s instead of ~234, cost ~9% decode
  t/s, every token). Pure chat traffic: use 1024 and gain those 9% back.
- **`--cache-ram`**: 2048 assumes your box can hold model + cache in RAM. It must exceed
  your session's prompt state (~19 KB/token at q8_0/q8_0) or caching silently does
  nothing. Tight-RAM boxes: `--cache-ram 0` and accept the full reprompt per turn.

Sampling flags are the model card's recommended defaults, keep them unless you know
why you are changing them. If you host a coding agent behind this server, harness-side
flags like `--tools all`, a reasoning format and chat-template kwargs go on top; they
change the API surface, not the speed.

The two env vars and taskset are kept as-is because every number on this page was
measured with them set (attention rotation disabled as an arch workaround, pinned
memory off under RAM pressure, taskset to the physical cores).

What the non-obvious flags do:

- `--spec-draft-cpu-moe` is **the trick**. The draft's experts are cold (at most 2 of its
  10 are touched per step), so they live in RAM and the target's hot-path experts keep the
  GPU. Putting the draft fully on GPU is measurably slower: 15.4 vs 16.5 t/s in my A/B.
- `--load-mode none --lazy-mode on`: lazy mmap loading, kills the slow cold-load ramp.
- `--fit on --fit-target 100`: `--fit` packs weights onto the GPU, and `--fit-target`
  is the VRAM **margin in MiB to leave free** afterwards (default 1024). Lower margin =
  more weights packed on GPU. Keep is 100 (floor-tested, see below; the older A/B ladder
  best was 1100 at 16.5 t/s and every margin from 100 to 1300 measures the same decode).
  On this box 800, 900, 1100 and 1300 all load
  cleanly and hold `VmSwap` flat; 2800 loads but measured slower, and 128 crashed when
  the draft still lived fully on GPU. Load OOM: raise one step (1300). Floor test with
  `--spec-draft-cpu-moe`: `--fit-target 100` loads fine (14.2 GB used, draft and all) and
  measures the same decode as any higher margin (16.4-17.5 t/s probes, within noise of
  the 1100 record). Below 100, the margin still has to cover the draft's GPU share,
  CUDA graphs and runtime workspace, so do not go lower without testing.
  Also note: a commented-out `#--fit-target` line does not mean 0, it means the default
  1024. Only an explicit `--fit-target 0` is 0.
- `-ctk q8_0 -ctv q8_0`: near-lossless KV at 131k (about 3.2 GB, this lives in the
  budget). Do not try `-ctv q4_0` to save VRAM: the server warns that no FlashAttention
  vector kernel exists for q8_0-q4_0 and converts K/V to f16 on every decode step.
- `--cache-ram 2048`: important on this hybrid-memory arch. The server's
  host prompt cache exports roughly 230 MB of recurrent state **per request, regardless of
  prompt size**, and grows to the 8 GiB default cap. Untended, that froze a 64 GB box
  (upstream PR [#24785](https://github.com/ggml-org/llama.cpp/pull/24785), recurrent state
  shrink, still open). The keep config puts the cap *below* the damage line instead:
  `--cache-ram 2048` holds ~2 sessions of 60-70k tokens, bounds the worst case at 2 GiB,
  and turned 150 s per-turn reprompts into ~12 s (cache hits reprocess only the new tail:
  4,915 of 68,241 tokens, logged). The cap must EXCEED your session's prompt state or it
  silently caches nothing: 40k tokens of q8_0/q8_0 state is ~782 MB, and `--cache-ram 512`
  logged `prompt state size 782 MiB exceeds cache size limit, skipping` on every request.
  The cap is host RAM, VRAM is unrelated.
- `--ctx-checkpoints 0`: context checkpoints are full snapshots of a slot's context, the
  machinery behind mid-session context rollback ("time travel" edits/branches). On this
  hybrid arch a checkpoint is another complete recurrent-state copy through the same
  export path described above, and neither chat nor agent clients use server-side rollback
  (compaction happens client-side). So: a feature nobody here calls, priced at another
  state copy each. Zero it.
- `-t` is physical cores, `-tb` is the full thread count of the pool. Mine is a 6-core
  part with HT off behind `taskset 0-11`. Do not give it every core.
- `--parallel`: the A/B ran single-stream. In production the default 4 slots works fine
  with the bounded 2 GiB cache, one conversation at a time still gets the whole GPU.

## Step 4 - Verify it worked

Generate something and watch the server log:

```
draft acceptance = 0.55 (mean len 2.1)   -> healthy, 55% of guesses kept
graphs reused = 4872                     -> climbing, zero recapture = healthy
```

- Acceptance is a property of your **request content**, not the config. Cold fresh
  reasoning sits around 0.5-0.57, follow-ups that reuse a long prefix reach 0.7-0.9.
- On my box: `tg ≈ acceptance × 30 + noise`, measured across several runs (14.4 t/s at
  0.48, 16.5 at 0.55, 17.5 at 0.60). Compare your tg against that line before blaming
  the build: if acceptance × 30 ≈ your tg, the config is fine, the content is what you got.
- Acceptance under ~0.30 for your workload: MTP is not paying off here, drop the `-md`
  line and `--spec-type` flags.
- Judge prompt-processing only from the server's own `prompt eval` lines on full real
  prompts. Incremental chunks and synthetic repeated-sentence probes overstate by 50%+ in
  my measurements.

## Results (131k ctx, q8_0/q8_0, real agent traffic)

| setup | prompt t/s | token t/s |
|---|---|---|
| Unsloth IQ4_XS, baseline | ~185 | ~10 |
| + AtomicChat AD-4.27bpw Q4_K_M, master build | 296 | 15.2 |
| + MTP with `--spec-draft-cpu-moe`, fit 1100 | **305** | **16.5** |

Warm follow-up tasks reach 19-24 t/s when acceptance is high. VRAM 14.6/16 GB, RAM ~50 GB.
Decode is flat with context length on this arch (the QSA indexer reads a fixed budget of
KV rows per step), so these numbers hold at 131k.

Different card? The knobs that matter: `--fit-target` (bigger card, lower or drop it) and
the quant choice. The MTP head is quant-agnostic, any Qwen3.8-Flash-Next GGUF of this arch
works with the branch.

## Credits and license

- Model code is llama.cpp, MIT license, by the ggml authors. The MTP support comes from
  their PR #28243 (Ryan Monsurate), this repo's branch only carries the cherry-picks plus
  the one-line fix on top. Not affiliated with ggml-org, and this is not a contribution.
- Target quant: AtomicChat (`AtomicChat/Qwen3.8-Flash-Next-GGUF`).
- MTP heads: Unsloth (`unsloth/Qwen3.8-Flash-Next-GGUF`).
- Guide and fix: MIT, do what you want with it. If #28243 lands upstream, this whole page
  collapses into "use master".

Issue reports welcome as issues here. Please include your GPU, RAM, the `draft acceptance`
line, and `VmSwap` after 20 turns.
