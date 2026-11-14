import os
import sys
import tempfile

import torch

NODE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path[:0] = [os.path.dirname(os.path.dirname(NODE_DIR)), NODE_DIR]

import comfy.model_patcher  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.patcher_extension  # noqa: E402
from comfy.ldm.moss_image.model import MOSSImage  # noqa: E402

from src import calibration as calibration_module  # noqa: E402
from src.cache import BranchState, MGCache  # noqa: E402
from src.calibration import MGCalibration, load_calibration  # noqa: E402
from src.patching import attach  # noqa: E402

SIGMAS = torch.linspace(1.0, 0.0, 21)
LAYERS = 4


def build_patcher():
    torch.manual_seed(0)
    diffusion_model = MOSSImage(
        dim=64, ffn_dim=128, num_heads=4, num_layers=LAYERS, text_dim=32, latent_channels=8, operations=comfy.ops.disable_weight_init
    )
    for name, parameter in diffusion_model.named_parameters():
        if "norm" in name:
            torch.nn.init.ones_(parameter)
        else:
            torch.nn.init.normal_(parameter, std=0.1)
    wrapper = torch.nn.Module()
    wrapper.diffusion_model = diffusion_model
    return comfy.model_patcher.ModelPatcher(wrapper, torch.device("cpu"), torch.device("cpu"))


def sample(patcher):
    torch.manual_seed(1)
    latent = torch.randn(1, 8, 6, 6)
    context = torch.randn(1, 5, 32)
    options = {**patcher.model_options["transformer_options"], "wrappers": patcher.wrappers}

    def loop():
        x = latent
        for sigma, next_sigma in zip(SIGMAS[:-1], SIGMAS[1:], strict=True):
            step_options = {**options, "sigmas": sigma.reshape(1), "cond_or_uncond": [0]}
            velocity = patcher.model.diffusion_model(x, sigma.reshape(1), context, transformer_options=step_options)
            assert torch.isfinite(velocity).all()
            x = x + (next_sigma - sigma) * velocity
        return x

    wrappers = comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, options)
    return comfy.patcher_extension.WrapperExecutor.new_executor(loop, wrappers).execute()


def test_zero_budgets_are_lossless():
    patcher = build_patcher()
    reference = sample(patcher)
    cached = sample(attach(patcher, MGCache(0.0, 0.0, None)))
    assert torch.equal(reference, cached)


def test_budgets_skip_steps_and_blocks():
    patcher = build_patcher()
    reference = sample(patcher)
    cache = MGCache(0.5, 0.5, None)
    cached = sample(attach(patcher, cache))
    assert cache.counts["steps"] == 20
    assert cache.counts["skipped_steps"] > 0
    assert cache.counts["reused_blocks"] > 0
    assert not cache.states
    assert torch.isfinite(cached).all()
    assert not torch.equal(reference, cached)


def test_forecast_extrapolates_linearly():
    state = BranchState()
    first, second = torch.full((1, 2, 2, 2), 3.0), torch.full((1, 2, 2, 2), 2.0)
    state.history = [(1.0, first), (0.8, second)]
    state.output = second
    assert torch.allclose(MGCache(1.0, 0.0, None, True).reuse_output(state, 0.6), torch.full((1, 2, 2, 2), 1.0))
    assert torch.equal(MGCache(1.0, 0.0, None, False).reuse_output(state, 0.6), second)
    patcher = build_patcher()
    assert torch.equal(sample(patcher), sample(attach(patcher, MGCache(0.0, 0.0, None, True))))


def test_calibration_round_trip():
    with tempfile.TemporaryDirectory() as directory:
        calibration_module.CALIBRATION_DIR = directory
        patcher = build_patcher()
        reference = sample(patcher)
        calibrated = sample(attach(patcher, MGCalibration(calibration_module.calibration_path("tiny"), 5)))
        assert torch.equal(reference, calibrated)
        calibration = load_calibration("tiny.json")
        assert len(calibration["sigmas"]) == 20
        assert calibration["drift"] == sorted(calibration["drift"])
        assert all(len(row) == LAYERS for row in calibration["phi"])
        assert calibration["k"][0] is None and calibration["k"][1].shape == (8,)
        lossless = sample(attach(patcher, MGCache(-1.0, 0.0, calibration)))
        assert torch.equal(reference, lossless)
        cached = sample(attach(patcher, MGCache(calibration["drift"][-1] / 4, 0.3, calibration)))
        assert torch.isfinite(cached).all()
        assert not torch.equal(reference, cached)
        for unsafe_name in ("../tiny.json", "missing.json"):
            try:
                load_calibration(unsafe_name)
            except ValueError:
                continue
            raise AssertionError(f"accepted {unsafe_name}")
        try:
            calibration_module.calibration_path("../escape")
        except ValueError:
            return
        raise AssertionError("path traversal accepted")


if __name__ == "__main__":
    test_zero_budgets_are_lossless()
    test_budgets_skip_steps_and_blocks()
    test_forecast_extrapolates_linearly()
    test_calibration_round_trip()
    sys.stdout.write("ok\n")
