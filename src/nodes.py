from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from .cache import MGCache
from .calibration import NO_CALIBRATION, MGCalibration, calibration_path, list_calibrations, load_calibration
from .patching import attach

CATEGORY = "advanced/model"


class MGCacheNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MGCache",
            display_name="MG-Cache",
            category=CATEGORY,
            description="Multi-granularity caching for DiT models: skips whole model calls and reuses stable transformer blocks.",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "alpha",
                    default=0.05,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    tooltip="Timestep budget. Model calls are skipped while the accumulated output drift stays below it. 0 disables timestep caching.",
                ),
                io.Float.Input(
                    "beta",
                    default=0.0,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    tooltip="Layer budget. A block is reused while its accumulated input drift stays below it. 0 disables layer caching. Without a calibration it degrades MOSS Image at any tested value.",
                ),
                io.Combo.Input(
                    "calibration",
                    options=[NO_CALIBRATION] + list_calibrations(),
                    default=NO_CALIBRATION,
                    tooltip="File written by MG-Cache Calibrate. Without one the drift is estimated online and every block has equal importance.",
                ),
                io.Boolean.Input(
                    "forecast",
                    default=False,
                    tooltip="On a skipped model call, extrapolate the output from the last two computed ones instead of repeating the last one.",
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, alpha, beta, calibration, forecast) -> io.NodeOutput:
        return io.NodeOutput(attach(model, MGCache(alpha, beta, load_calibration(calibration), forecast), beta > 0))


class MGCacheCalibrateNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MGCacheCalibrate",
            display_name="MG-Cache Calibrate",
            category=CATEGORY,
            description="Sample once through this model to record the MG-Cache timestep schedule, output regression and layer importance.",
            inputs=[
                io.Model.Input("model"),
                io.String.Input("name", default="moss_image"),
                io.Int.Input(
                    "importance_stride",
                    default=0,
                    min=0,
                    max=1000,
                    tooltip="Profile layer importance every N steps; each profiled step costs one extra model call per block. 0 skips it.",
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, name, importance_stride) -> io.NodeOutput:
        return io.NodeOutput(attach(model, MGCalibration(calibration_path(name), importance_stride), importance_stride > 0))


class MGCacheExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MGCacheNode, MGCacheCalibrateNode]


async def comfy_entrypoint() -> MGCacheExtension:
    return MGCacheExtension()
