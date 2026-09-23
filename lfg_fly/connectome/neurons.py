"""Named neuron populations of a built Graph."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lfg_fly.connectome.build import Graph

PHOTORECEPTOR_TYPES = ("R1-R6", "R7p", "R7y", "R7d", "R7_unclear",
                       "R8p", "R8y", "R8d", "R8_unclear", "R7R8_unclear")
# R1-R6 broad luminance; R8y green; R8p blue; R7* short-wavelength (blue stands in for UV).
CHANNEL_BY_TYPE = {
    "R1-R6": "lum", "R7p": "B", "R7y": "B", "R7d": "B", "R7_unclear": "B",
    "R8p": "B", "R8y": "G", "R8d": "lum", "R8_unclear": "lum", "R7R8_unclear": "lum",
}
READOUT_SUPERCLASSES = ("descending_neuron", "vnc_motor", "cb_motor")
BRAIN_SENSORY = ("ol_sensory", "cb_sensory")
VNC_SENSORY = ("vnc_sensory",)


@dataclass(frozen=True)
class Populations:
    readout: np.ndarray
    photoreceptor: np.ndarray
    photoreceptor_type: np.ndarray
    orn: np.ndarray
    orn_glomerulus: np.ndarray
    sensory_brain: np.ndarray
    sensory_vnc: np.ndarray
    sensory_any: np.ndarray


def populations(g: Graph) -> Populations:
    sc, t = g.superclass, g.cell_type
    pr = np.flatnonzero(np.isin(t, PHOTORECEPTOR_TYPES))
    orn = np.flatnonzero(np.char.startswith(t, "ORN_"))
    return Populations(
        readout=np.flatnonzero(np.isin(sc, READOUT_SUPERCLASSES)),
        photoreceptor=pr,
        photoreceptor_type=t[pr],
        orn=orn,
        orn_glomerulus=np.char.replace(t[orn], "ORN_", "", count=1),
        sensory_brain=np.flatnonzero(np.isin(sc, BRAIN_SENSORY)),
        sensory_vnc=np.flatnonzero(np.isin(sc, VNC_SENSORY)),
        # Exact set, not a substring match: only the three superclasses an input
        # code can actually drive (ol_sensory/cb_sensory/vnc_sensory). Related
        # superclasses like sensory_ascending/sensory_descending and any *_tbc
        # variant are NOT sensory here -- they still get the tonic bias.
        sensory_any=np.isin(sc, BRAIN_SENSORY + VNC_SENSORY),
    )
