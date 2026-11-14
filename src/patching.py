import comfy.patcher_extension
from torch import nn

WRAPPER_KEY = "mg_cache"
BLOCK_TYPES = ("double_block", "single_block")


def branch_key(transformer_options):
    return tuple(transformer_options.get("cond_or_uncond", ()))


def current_sigma(transformer_options):
    return transformer_options["sigmas"].max().item()


def block_depth(model):
    block_lists = (child for child in model.model.diffusion_model.children() if isinstance(child, nn.ModuleList))
    return max((len(blocks) for blocks in block_lists), default=0)


def attach(model, handler, replace_blocks=True):
    model = model.clone()
    model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, WRAPPER_KEY, handler.outer_sample)
    model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, handler.diffusion_model)
    if replace_blocks:
        for block_type in BLOCK_TYPES:
            for index in range(block_depth(model)):
                model.set_model_patch_replace(handler.block, "dit", block_type, index)
    return model
