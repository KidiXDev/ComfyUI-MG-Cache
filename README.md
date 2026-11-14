# ComfyUI-MG-Cache

Training-free [MG-Cache](https://arxiv.org/abs/2611.11365) (multi-granularity caching) for DiT models in ComfyUI, written for the native MOSS Image model.

Two caches run together:

- **Timestep cache** – a whole model call is skipped while the accumulated output drift since the last computed step stays within `alpha`.
- **Layer cache** – on computed steps, a transformer block is reused while its accumulated input drift (times its importance) stays below `beta`.

## Nodes

Both are under `advanced/model`. Put them between the model loader and the sampler.

### MG-Cache

| Input | Meaning |
| --- | --- |
| `alpha` | Timestep budget. Higher skips more model calls. `0` turns the timestep cache off. |
| `beta` | Layer budget. Higher reuses more blocks. `0` turns the layer cache off. |
| `calibration` | `none`, or a file made by MG-Cache Calibrate. |
| `forecast` | On a skipped call, extrapolate the output from the last two computed ones instead of repeating the last one. |

Without a calibration the output drift is estimated online from the latent change, and all blocks have importance 1. Use `alpha` around `0.05` to `0.1` and leave `beta` at `0`; see the measurements below. The console prints how many model calls and blocks were skipped after each run.

## Measured

RTX 4050 Laptop 6 GB, `mossImage_sft_3301904`, 832x1280, 25 steps, euler/simple, CFG 3, same seed, no calibration. Seconds are the whole prompt. PSNR is against the output without the node.

| Settings | Seconds | Speedup | PSNR | Skipped calls | Reused blocks |
| --- | --- | --- | --- | --- | --- |
| Without the node | 48.9 | 1.00x | - | - | - |
| `alpha 0.1`, `beta 0` | 31.4 | 1.56x | 29.3 dB | 19/50 | - |
| `alpha 0.1`, `beta 0`, `forecast` | 31.8 | 1.54x | 28.1 dB | 19/50 | - |
| `alpha 0`, `beta 0.02` | 75.7 | 0.65x | 18.6 dB | 0/50 | 420/2100 |
| `alpha 0`, `beta 0.05` | 57.6 | 0.85x | 14.0 dB | 0/50 | 1008/2100 |
| `alpha 0`, `beta 0.1` | 48.0 | 1.02x | 13.4 dB | 0/50 | 1428/2100 |
| `alpha 0.1`, `beta 0.3` | 21.2 | 2.31x | 12.8 dB, image destroyed | 28/50 | - |

- **The timestep cache works.** `alpha 0.1` keeps the same picture with small detail changes.
- **The uncalibrated layer cache does not work on MOSS Image.** Every tested `beta` degrades the image badly, and below about two thirds of blocks reused it is slower than no cache, because each computed block copies its residual to system RAM. It was not tested with a layer-importance calibration.
- **`forecast` is inconclusive.** Its mean pixel error is lower than plain reuse (3.9 against 5.4 of 255) but its PSNR is also lower.

### MG-Cache Calibrate

Records the paper's offline data. Connect it instead of MG-Cache, sample once with your usual sampler, steps, shift and CFG, and it writes `calibrations/<name>.json`. Refresh ComfyUI, then pick the file in MG-Cache.

- `importance_stride 0` records the timestep drift schedule and the per-channel output regression. Costs one normal generation.
- `importance_stride N` also profiles layer importance every N steps. Each profiled step costs one extra model call per block (42 for MOSS Image), so use a large stride.

The drift schedule and importance are looked up by sigma, so they tolerate a different step count. The regression is only applied when the run uses the same sigma schedule as the calibration.

## Differences from the paper

- The cache unit is a whole block (its residual), not separate attention and MLP outputs.
- Layer importance is normalized to a mean of 1, so `beta` has the same scale with or without calibration.
- The regression is chained step to step.
- Calibration uses the one prompt you sample, not a 50 prompt batch.

## Memory

The layer cache keeps one residual per block per cond batch, roughly 2.3 GB for MOSS Image at 1024x1024 with CFG. It lives in system RAM (in VRAM only with `--gpu-only`), so the node adds well under 100 MB of VRAM.

With `beta 0` the node does not touch the blocks and keeps only a few latent-sized tensors.

## Layout

| Path | Contents |
| --- | --- |
| `src/nodes.py` | The two ComfyUI nodes |
| `src/cache.py` | Timestep and layer cache |
| `src/calibration.py` | Calibration recording, saving and loading |
| `src/patching.py` | Attaching wrappers and block replacements to a model |
| `src/metrics.py` | Drift and regression math |
| `tests/` | Self-check on a tiny MOSS Image model |

## Development

Run the test with ComfyUI's Python:

```bash
python tests/test_mg_cache.py
```

Lint and format:

```bash
ruff check .
```

```bash
ruff format .
```
