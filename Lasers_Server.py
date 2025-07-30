"""Config file for devicebase.

Import device classes, then define entries in DEVICES as:
 devices(CLASS, HOST, PORT, other_args)
"""
## Function to create record for each device.
from microscope.devices import device
## Import device modules/classes here.
from microscope.filterwheels.ell_slider import ThorlabsELLSlider
from microscope.lights.deepstar import DeepstarLaser
from microscope.lights.obis import ObisLaser

host = "localhost"

DEVICES = [
           # device(DeepstarLaser, host, 9011, conf={'com': 'COM10', 'baud': 9600, 'timeout': 0.5}),  # Deepstar 488
           device(ObisLaser, host, 9012, conf={'com': 'COM11', 'baud': 115200, 'timeout': 2.0}),  # Obis 561
           device(ObisLaser, host, 9013, conf={'com': 'COM12', 'baud': 115200, 'timeout': 2.0}),  # Obis 642
           device(ThorlabsELLSlider, host, 9014, conf={'com': 'COM13'})
]
