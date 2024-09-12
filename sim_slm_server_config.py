import Pyro4
Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config.PICKLE_PROTOCOL_VERSION = 2
from microscope.device_server import device

from microscope.simulators import SimulatedSpatialLightModulator
from microscope.slm.sim_slm import SIM_SLM


slm = SimulatedSpatialLightModulator()

DEVICES = [
    device(
        SIM_SLM,
        host="127.0.0.1",
        port=8001,
        conf={
            "slm": slm,
            "sim_diffraction_angle": 0.48,
            "sim_modulation_factors": {488: 185, 561: 180, 647: 180},
            "pixel_pitch": 15.0,
        },
    )
]
