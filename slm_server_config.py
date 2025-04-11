from microscope.device_server import device
from microscope.simulators import SimulatedFilterWheel
from microscope.simulators import SimulatedCamera
from microscope.simulators import SimulatedSpatialLightModulator

DEVICES = [
    device(
        SimulatedFilterWheel,
        host="127.0.0.1",
        port=8001,
        conf={"positions": 6},
    ),
    device(
        SimulatedCamera,
        host="127.0.0.1",
        port=8002,
    ),
    device(
        SimulatedSpatialLightModulator,
        host="127.0.0.1",
        port=8003,
        conf={"shape": (512, 512), "default_wavelength_nm": 561},
    ),
]
