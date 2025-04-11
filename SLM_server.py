from microscope.devices import device
from microscope.slm.meadowlark import SLM_512
from pathlib import Path

host = "localhost"
port = 8000

DEVICES = [
    device(
        SLM_512,
        host=host,
        port=port,
        conf={
            # "blink_sdk_dll_path": Path("C:\\Users\omxt\PycharmProjects\microscope\microscope\slm\Blink_SDK_C.dll"),
            "blink_sdk_dll_path": Path(
                "C:\\Program Files\Meadowlark Optics\Blink OverDrive Plus\SDK\Blink_C_wrapper"
            ),
            "luts": {
                488: "slm4039_at488_P8.lut",
                561: "slm4039_at561_P8.lut",
                642: "slm4039_at642_P8.lut",
            },
            "default_wavelength": 561,
            "default_static_lut_file": "SLM_lut.txt",

            "phase_calibration_files_path": Path("C:\\Users\omxt\PycharmProjects\microscope\microscope\slm\LUT_files"),
            # "bit_depth": 8,
            "max_transients": 10,
            "external_trigger": True,
        }
    )
]
