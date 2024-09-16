import Pyro4

Pyro4.config.COMPRESSION = True
Pyro4.config.SERIALIZERS_ACCEPTED = "pickle"
Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.PICKLE_PROTOCOL_VERSION = 2

from microscope.device_server import device
from microscope.slm.meadowlark import SLM_512
from microscope.slm.sim_slm import SIM_SLM

# slm = SLM_512(
#     header_definitions_path=r"C:\Users\omxt\PycharmProjects\microscope\microscope\slm\Blink_SDK_C_wrapper_defs.h",
#     blink_sdk_dll_path=r"C:\Users\omxt\PycharmProjects\microscope\microscope\slm\Blink_SDK_C.dll",
#     bit_depth=8,
#     luts={
#         473: 'slm4039_at473_P8.lut',
#         488: 'slm4039_at488_P8.lut',
#         532: 'slm4039_at532_P8.lut',
#         561: 'slm4039_at561_P8.lut',
#         635: 'slm4039_at635_P8.lut',
#         642: 'slm4039_at642_P8.lut',
#     },
#     phase_calibration_files_path=r"C:\Users\omxt\PycharmProjects\bnsdevice\LUT_files",
#     default_wavelength=488,
#     default_static_lut_file="SLM_lut.txt",
#     is_nematic_type=True,
#     ram_write_enable=True,
#     use_gpu=True,
#     use_odp=False
# )
slm_kwargs = {
    "header_definitions_path": r"C:\Users\omxt\PycharmProjects\microscope\microscope\slm\Blink_SDK_C_wrapper_defs.h",
    "blink_sdk_dll_path": r"C:\Users\omxt\PycharmProjects\microscope\microscope\slm\Blink_SDK_C.dll",
    "bit_depth": 8,
    "luts": {
        473: 'slm4039_at473_P8.lut',
        488: 'slm4039_at488_P8.lut',
        532: 'slm4039_at532_P8.lut',
        561: 'slm4039_at561_P8.lut',
        635: 'slm4039_at635_P8.lut',
        642: 'slm4039_at642_P8.lut',
    },
    "phase_calibration_files_path": r"C:\Users\omxt\PycharmProjects\bnsdevice\LUT_files",
    "default_wavelength": 488,
    "default_static_lut_file": "SLM_lut.txt",
    "is_nematic_type": True,
    "ram_write_enable": True,
    "use_gpu": True,
    "use_odp": False,
    "trigger_timeout_ms": 100000
}

DEVICES = [
    device(
        SIM_SLM,
        host="10.6.19.23",
        port=8000,
        conf={
            "slm": SLM_512,
            "slm_kwargs": slm_kwargs,
            "sim_diffraction_angle": 0.48,
            "sim_modulation_factors": {488: 190, 561: 190, 642: 180},
            "pixel_pitch": 15.0,
        }
    ),
]
