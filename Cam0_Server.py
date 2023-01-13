"""Configuration file for deviceserver.
"""
from microscope.devices import device

# Import required device classes
from microscope.cameras.andorsdk3 import AndorSDK3


# host is the IP address (or hostname) from where the device will be
# accessible.  If everything is on the same computer, then host will
# be '127.0.0.1'.  If devices are to be available on the network,
# then it will be the IP address on that network.
host = '10.6.19.30'

#
# def construct_camera_0() -> typing.Dict[str, device]:
#     camera = AndorSDK3()
#     camera.set_setting("trigger_mode", "External Exposure")
#     return {"camera_0": camera}
#
#
DEVICES = [
    device(AndorSDK3, host, 8001, uid="VSC-01604")  # {'transform': (0, 1, 1)}),  # timeout=1, buffer_length=, index=0,
    # device(TestCamera, host, 8005, otherargs=1,),
    # device(TestCamera, host, 8006, otherargs=1,),
    ]
# from microscope.device_server import device
# from microscope.simulators import SimulatedCamera, SimulatedFilterWheel
#
# def construct_camera():
#     camera = SimulatedCamera()
#     camera.set_setting("display image number", False)
#     return {'name': camera}
#
# def construct_device():
#     d = SimulatedFilterWheel(positions=6)
#     return {"device": d}
#
# DEVICES = [
#     device(construct_camera, host="127.0.0.1", port=8000),
#     device(construct_device, host="127.0.0.1", port=8001),
    # device(SimulatedCamera, host="127.0.0.1", port=8001),
# ]
#