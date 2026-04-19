import os
import sys
import torch
import click
from typing import Callable, cast

# Register ourselves as the module `flute` so internal bare references
# like `flute.tune.TuneMetaData` in the vendored wrapper files resolve
# to this package without touching every file. Use setdefault so any
# pre-existing external flute-kernel install wins — avoids surprising
# users who already depend on the upstream package.
sys.modules.setdefault("flute", sys.modules[__name__])

# Vendored FLUTE uses torch JIT-compile via turboquant_vllm.flute_build.
# First import triggers a compile (~5-15 min); cached thereafter.
try:
    from . import _C  # type: ignore[attr-defined]  # noqa: F401
except ImportError:
    from turboquant_vllm import flute_build as _flute_build

    _C = _flute_build.build()

from . import ops  # noqa: E402

__version__ = "0.4.2"


qgemm = cast(
    Callable[
        [
            torch.Tensor,  # inputs
            torch.Tensor,  # weight
            torch.Tensor,  # scales
            torch.Tensor,  # tables
            torch.Tensor,  # tables2
            torch.Tensor,  # workspace
            int,           # num_bits
            int,           # group_size
            int,           # template_id
            int,           # num_sms
        ],
        torch.Tensor,
    ],
    torch.ops.turboquant_flute.qgemm_raw_simple,
)


qgemm_hadamard = cast(
    Callable[
        [
            torch.Tensor,  # inputs
            torch.Tensor,  # weight
            torch.Tensor,  # scales
            torch.Tensor,  # tables
            torch.Tensor,  # tables2
            torch.Tensor,  # workspace
            int,           # num_bits
            int,           # group_size
            int,           # hadamard_size
            int,           # template_id
            int,           # num_sms
        ],
        torch.Tensor,
    ],
    torch.ops.turboquant_flute.qgemm_raw_simple_hadamard,
)


# Load the template configs
if os.environ.get("FLUTE_ABLATIONS", "0") == "1":
    click.secho(f"[FLUTE]: Abalations enabled", fg="yellow")
    TEMPLATE_CONFIGS_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data/qgemm_kernel_raw_generated_configs.ablations.pth")
else:
    TEMPLATE_CONFIGS_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data/qgemm_kernel_raw_generated_configs.pth")

if os.path.exists(TEMPLATE_CONFIGS_PATH):
    TEMPLATE_CONFIGS = torch.load(TEMPLATE_CONFIGS_PATH, weights_only=True)
    click.secho(f"[FLUTE]: Template configs loaded from {TEMPLATE_CONFIGS_PATH}", fg="green")
else:
    TEMPLATE_CONFIGS = None
    click.secho(f"[FLUTE]: Template configs not found at {TEMPLATE_CONFIGS_PATH}", fg="red")
