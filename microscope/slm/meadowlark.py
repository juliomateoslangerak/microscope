#!/usr/bin/env python3
import logging
import threading
from abc import ABC, abstractmethod

## Copyright (C) 2024 Julio Mateos Langerak <julio.mateos-langerak@igh.cnrs.fr>
##
## This file is part of Microscope.
##
## Microscope is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## Microscope is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with Microscope.  If not, see <http://www.gnu.org/licenses/>.

import microscope.abc
from microscope import DeviceError, IncompatibleStateError, InitialiseError, DisabledDeviceError, UnsupportedFeatureError, LibraryLoadError
from microscope import TriggerType, TriggerMode

import os.path
from cffi import FFI
import numpy as np

HEADER_DEFINITIONS_PATH = (
    "C:\\Users\\omxt\\PycharmProjects\\bnsdevice\\Blink_SDK_C_wrapper_defs.h"
)
BLINK_SDK_DLL_PATH = "Blink_SDK_C.dll"

PHASE_CALIBRATION_FILES_PATH = (
    "C:\\Users\\omxt\\PycharmProjects\\bnsdevice\\LUT_files"
)
DEFAULT_STATIC_LUT_FILE = "SLM_lut.txt"
LUTS = {
    473: "slm4039_at473_P8.lut",
    532: "slm4039_at532_P8.lut",
    635: "slm4039_at635_P8.lut",
}
DEFAULT_LUT_FILE = "slm4039_at473_P8.lut"

BIT_DEPTH = 8
# Overdrive  SLMs  the  true  frames  parameter  should  be  set  to  5.
# For non-overdrive  operation  true frames should be set to 3
TRUE_FRAMES = 3

CLASS_NAME = "BNSDevice_ODP"


# DECORATORS
# decorator definition for methods that require an SLM to be initialized
def requires_slm(func):
    def wrapper(self, *args, **kwargs):
        if not self._constructed_okay:
            raise DisabledDeviceError("SLM is not initialized.")
        else:
            return func(self, *args, **kwargs)

    return wrapper


def float_to_8_bit(array):
    """Converts a float array values 0.0 to 1.0 to 8-bit. Values outside that range are clipped"""
    array = np.clip(array, 0.0, 1.0)
    # TODO: apply some logic to use the most efficient portion of the range in the SLM
    return np.round(array * 255).astype("uint8")


def transform_16_to_8_bit(array, fitting=None):
    if array.dtype == "uint16":
        coef = np.array([np.iinfo("uint8").max / np.iinfo("uint16").max])
        new_array = np.multiply(array, coef).astype("uint8")
        if fitting is None:
            return new_array
        elif fitting == "low":
            return new_array - new_array.min()
        elif fitting == "up":
            return new_array + (np.iinfo("uint8").max - new_array.max())
        elif fitting == "mid":
            return new_array + (
                127
                - ((new_array.max() - new_array.min()) // 2)
                + new_array.min()
            )

    elif array.dtype == "uint8":
        return array
    else:
        raise ValueError("The datatype is neither uint8 or uint16")


class MeadowlarkSpatialLightModulator(microscope.abc.SpatialLightModulator, ABC):
    """Meadowlark Spatial Light Modulator.

    This microscope device is for controlling Meadowlark Optics' Spatial Light
    Modulators (formerly Boulder Non-Linear).
    Important note: The header defs is a text file containing the headers from
    the Blink_SDK.h file with some modifications. Namely, all #include and #ifdef have been
    removed.

    :param header_definitions_path: Absolute path to the header definitions from the SDK.
    :param blink_sdk_dll_path: name of, or absolute path to, the SID4_SDK.dll file
    :param luts_path: Absolute path to the LUT files.
    :param default_wavelength:
    :param phase_calibration_files_path:
    :param bit_depth:
    :param slm_resolution:
    :param is_nematic_type:
    :param ram_write_enable:
    :param use_gpu:
    :param max_transients:
    """

    def __init__(
        self,
        header_definitions_path: str,
        blink_sdk_dll_path: str,
        bit_depth: int,
        luts: dict,
        phase_calibration_files_path: str,
        default_wavelength: str = None,
        default_static_lut_file: str = None,
        is_nematic_type: bool = True,
        ram_write_enable: bool = True,
        use_gpu: bool = True,
        trigger_timeout_ms: int = 0,
        max_transients: int = 10,
        **kwargs,
    ) -> None:
        if not is_nematic_type:
            raise NotImplementedError(
                "Only nematic SLMs are supported at the moment"
            )

        super().__init__(**kwargs)

        self._slm_handle = None

        self._transient_patterns = []

        # Get the SDK library
        try:
            with open(header_definitions_path, "r") as header_definitions_path:
                cdef_from_file = header_definitions_path.read()
        except FileNotFoundError as e:
            raise e from e
        except IOError as e:
            raise e(f'Unable to open "{header_definitions_path}"') from e
        finally:
            if cdef_from_file == "" or cdef_from_file is None:
                logging.error(f'File "{header_definitions_path}" is empty')
                exit(3)

        # Create here the interface to the SDK
        self._ffi = FFI()
        try:
            self._ffi.cdef(cdef_from_file, override=True)
            self._blink_sdk = self._ffi.dlopen(blink_sdk_dll_path)
        except Exception as e:
            raise LibraryLoadError(e) from e

        # Basic SLM parameters required for initialization
        # The board makes reference to the board id in case there are more than one SLM.
        # Currently only one is supported
        self._board = self._ffi.cast("int", 1)
        self._bit_depth = self._ffi.cast("unsigned int", bit_depth)
        self._num_boards_found = self._ffi.new("unsigned int *", 0)
        self._constructed_okay = self._ffi.new("int *", True)
        self._is_nematic_type = self._ffi.cast("int", is_nematic_type)
        self._RAM_write_enable = self._ffi.cast("int", ram_write_enable)
        self._use_GPU = self._ffi.cast("int", use_gpu)
        self._max_transients = self._ffi.cast("int", max_transients)
        self._ramp_delay = self._ffi.cast("unsigned int", 10)
        self._pre_ramp_slope = self._ffi.cast("unsigned int", 14)
        self._post_ramp_slope = self._ffi.cast("unsigned int", 24)

        # Variables to be redefined in the subclasses
        self._image_size = self._ffi.NULL

        # LUTs
        self._luts = {k: v.encode() for k, v in luts.items()}
        if default_wavelength is None:
            self._default_lut_file = list(self._luts.values())[0]
        else:
            self._default_lut_file = self._luts[default_wavelength]
        self._default_static_lut_file = default_static_lut_file.encode()
        # TODO: apply some logic to get a real path, without escaping characters
        self._phase_calibration_files_path = phase_calibration_files_path.encode()

        if self._default_static_lut_file is None:
            self._default_static_regional_lut_file = self._ffi.NULL
        else:
            self._default_static_regional_lut_file = self._ffi.new(
                "char[]",
                os.path.join(
                    self._phase_calibration_files_path, self._default_static_lut_file
                ),
            )
            
        # Trigger parameters

        self._initialize()

        # Verify construction of resources
        if int(self._blink_sdk.Is_slm_transient_constructed(self._slm_handle)):
            raise InitialiseError(
                "Overdrive  frame calculation  engine  was not properly  constructed"
            )

        # Load the default LUT
        self._load_lut(self._default_lut_file)

        # Two threads for running the patterns and a boolean to control it.
        # One thread is for running the patterns in the hardware and the other
        # is for running the patterns in the software.
        self._pattern_running = False
        self._hw_pattern_running_thread = threading.Thread(target=self._hw_run_pattern)
        self._sw_pattern_running_thread = threading.Thread(target=self._sw_run_pattern)

        # Boolean to control triggers use
        self._wait_for_trigger = self._ffi.cast("int", 0)
        # self._external_pulse = self._ffi.cast("int", 1)
        # # TODO: this is presumably the same thing as external_pulse
        self._output_pulse_image_flip = self._ffi.cast("int", 1)
        self._trigger_timeout_ms = self._ffi.cast(
            "unsigned int", trigger_timeout_ms
        )

        self.add_setting(
            name="trigger_timeout_ms",
            dtype="int",
            get_func=lambda: int(self._trigger_timeout_ms),
            set_func=self._set_trigger_timeout_ms,
            values=lambda: (0, 2**16),
            readonly=lambda: False,
        )

        self.add_setting(
            name="max_transients",
            dtype="int",
            get_func=lambda: int(self._max_transients),
            set_func=self._set_max_transients,
            values=lambda: (0, 2**8),
            readonly=lambda: False,
        )

        self.add_setting(
            name="version_info",
            dtype="str",
            get_func=lambda: self._get_version_info(),
            set_func=None,
            values=lambda: "Returns the version information of the SLM",
        )

        self.add_setting(
            name="output_pulse_image_flip",
            dtype="bool",
            get_func=lambda: bool(self._output_pulse_image_flip),
            set_func=self._set_output_pulse_image_flip,
            values=lambda: "This setting is controlling weather an output trigger is sent after the image is written.",
            readonly=lambda: False,
        )

    def _initialize(self):
        # Need to unload and reload the DLL here.
        # Otherwise, the DLL can open an error window about having already
        # initialized another DLL, which we won't see on a remote machine.

        # Initialize the library, looking for nematic SLMs.
        try:
            self._slm_handle = self._blink_sdk.Create_SDK(
                self._bit_depth,
                self._num_boards_found,
                self._constructed_okay,
                self._is_nematic_type,
                self._RAM_write_enable,
                self._use_GPU,
                self._max_transients,
                self._default_static_regional_lut_file,
            )
            if self._num_boards_found[0] == 0:
                raise InitialiseError("No SLM device found.")
            elif self._num_boards_found[0] > 1:
                raise InitialiseError(
                    "More than one SLM device found. This module can only handle one device."
                )
            elif self._constructed_okay[0] == -1:
                raise InitialiseError("SLM constructor did not succeed.")

        except Exception as e:
            raise InitialiseError("Could not Initialize") from e
    
    @property
    def trigger_mode(self) -> microscope.TriggerMode:
        return TriggerMode.ONCE
    
    @property
    def trigger_type(self) -> microscope.TriggerType:
        if bool(self._wait_for_trigger):
            return TriggerType.FALLING_EDGE
        else:
            return TriggerType.SOFTWARE
        
    def set_trigger(
        self, ttype: microscope.TriggerType, tmode: microscope.TriggerMode
    ) -> None:
        if self._pattern_running:
            raise IncompatibleStateError(
                "Cannot set trigger while a sequence of patterns is running."
                "Stop the sequence before setting the trigger type by disabling the device."
            )
        if tmode != TriggerMode.ONCE:
            raise UnsupportedFeatureError(
                "Only TriggerMode.ONCE is supported by this SLM"
            )

        if ttype == TriggerType.SOFTWARE:
            self._wait_for_trigger = self._ffi.cast("int", 0)
        else:
            self._wait_for_trigger = self._ffi.cast("int", 1)

    @requires_slm
    def _get_shape(self):
        return self._shape

    def _do_shutdown(self) -> None:
        self._blink_sdk.Delete_SDK(self._slm_handle)
        self._constructed_okay[0] = 0

    @requires_slm
    def _load_lut(self, filename):
        lut_file = self._ffi.new(
            "char[]",
            os.path.join(self._phase_calibration_files_path, filename)
        )
        _r = self._blink_sdk.Load_LUT_file(
            self._slm_handle, self._board, lut_file
        )
        if int(_r):
            raise DeviceError(self._get_last_error())

    @requires_slm
    def _load_linear_lut(self):
        _r = self._blink_sdk.Load_linear_LUT(self._slm_handle, self._board)
        if int(_r):
            raise DeviceError(self._get_last_error())

    def _load_wavelength_lut(self, wavelength):
        """Loads the LUT to the SLM that fits the best for a specified wavelength"""
        # Load the default LUT
        lut_wavelengths = self._luts.keys()
        nearest = min(lut_wavelengths, key=lambda x: abs(x - wavelength))
        lut_file = os.path.join(
            self._phase_calibration_files_path, self._luts[nearest]
        )
        self._load_lut(lut_file)

    @abstractmethod
    def _write_pattern(self, pattern):
        """This function should be implemented in the subclasses"""
        raise NotImplemented()

    @requires_slm
    def _do_apply_pattern(self, pattern, wavelength=None):
        if self._pattern_running:
            raise Exception("Sequence is running. Cannot write single patterns")

        if wavelength is not None:
            self._load_wavelength_lut(wavelength)

        pattern = float_to_8_bit(pattern)

        self._write_pattern(pattern)

    @abstractmethod
    def _hw_run_pattern(self):
        """This function is running on a separate thread to write the patterns.
        This function should be implemented in the subclasses"""
        raise NotImplemented()

    @abstractmethod
    def _sw_run_pattern(self):
        """This function is running on a separate thread to write the patterns.
        This function should be implemented in the subclasses"""
        raise NotImplemented()

    def _set_trigger_timeout_ms(self, timeout_ms):
        self._trigger_timeout_ms = self._ffi.cast(
            "unsigned int", timeout_ms
        )

    @requires_slm
    def _set_true_frames(self, true_frames):
        self._true_frames = true_frames
        self._blink_sdk.Set_true_frames(self._slm_handle, self._true_frames)

    @requires_slm
    def _get_last_error(self):
        return self._ffi.string(
            self._blink_sdk.Get_last_error_message(self._slm_handle)
        )

    def _set_max_transients(self, max_transients):
        self._max_transients = self._ffi.cast("int", max_transients)

    def _set_output_pulse_image_flip(self, output_flip):
        if output_flip:
            self._output_pulse_image_flip = self._ffi.cast("int", 1)
        else:
            self._output_pulse_image_flip = self._ffi.cast("int", 0)

    def _get_version_info(self):
        return self._ffi.string(self._blink_sdk.Get_version_info(self._slm_handle)).decode()


class SLM_512(MeadowlarkSpatialLightModulator):
    """Meadowlark Spatial Light Modulator with 512x512 resolution."""
    def __init__(self, use_odp: bool = False, **kwargs):
        super().__init__(**kwargs)
        # OverDrive Plus parameters
        self._use_odp = use_odp
        self._true_frames = self._ffi.cast("int", 0)

        try:
            self._shape = (
                int(self._blink_sdk.Get_image_width(self._slm_handle, self._board)),
                int(self._blink_sdk.Get_image_height(self._slm_handle, self._board)),
            )
        except AttributeError:
            # Some SDK versions do not have the Get_image_width/height methods
            # TODO: Verify if modern SDKs have or it is only exclusive to 1024x1024 versions
            self._shape = (512, 512)
        if self._shape != (512, 512):
            raise InitialiseError("The shape of the SLM is not 512x512. You may"
                                  "have initialized the wrong SLM device or used"
                                  "the wrong device class.")

        if self._use_odp:
            self._set_true_frames(5)
        else:
            self._set_true_frames(3)

        self._image_size = self._ffi.cast("unsigned int", 512)
        self._power_state = self._ffi.cast("int", 0)

        self.add_setting(
            name="use_odp",
            dtype="bool",
            get_func=lambda: self._use_odp,
            set_func=lambda x: self._set_odp(x),
            values=lambda: "This setting is controlling the use of ODP. Disable device before setting. True or False",
            readonly=lambda: self.enabled or not self._blink_sdk.Is_slm_transient_constructed(self._slm_handle),
        )

    def _set_odp(self, use_odp):
        # We have to change here the true_frames when we enable/disable DOP
        raise NotImplemented()

    def _do_enable(self):
        if bool(self._power_state):
            print("already enabled")
            # Already enabled
            return True

        # and if hardware we run from the in a separate thread
        if self._transient_patterns:
            self._start_sequence()
        self._power_state = self._ffi.cast("int", 1)
        self._blink_sdk.SLM_power(self._slm_handle, self._power_state)
        return True

    def _do_disable(self):
        if not bool(self._power_state):
            return True
        self._stop_sequence()
        self._power_state = self._ffi.cast("int", 0)
        self._blink_sdk.SLM_power(self._slm_handle, self._power_state)
        return True

    def _write_pattern(self, image):
        _r = self._blink_sdk.Write_image(
            self._slm_handle,
            self._board,
            self._ffi.from_buffer(image),
            self._image_size,
            self._wait_for_trigger,
            self._output_pulse_image_flip,
            self._trigger_timeout_ms,
        )
        if int(_r):
            raise Exception(self._get_last_error())

    @requires_slm
    def _queue_patterns(self) -> None:
        # Verify that the calculation engine is properly loaded
        if self._blink_sdk.Is_slm_transient_constructed(self._slm_handle) < 0:
            raise Exception(
                "SLM transient calculation engine not properly constructed"
            )

        self._transient_patterns = []
        current_wavelength = None

        for pattern, wavelength in zip(self._patterns, self._wavelengths):
            if wavelength != current_wavelength:
                self._load_wavelength_lut(wavelength)
                current_wavelength = wavelength
            pattern = float_to_8_bit(pattern)
            transients = self._compute_transients(pattern)
            self._transient_patterns.append(transients)

    @requires_slm
    def _compute_transients(self, pattern):
        byte_count = self._ffi.new("unsigned int*", 0)
        self._blink_sdk.Calculate_transient_frames(
            self._slm_handle, self._ffi.from_buffer(pattern), byte_count
        )
        transients = self._ffi.new("unsigned char[]", byte_count[0])
        self._blink_sdk.Retrieve_transient_frames(self._slm_handle, transients)
        return transients

    @requires_slm
    def _start_sequence(self):
        """Sequence wil restart if already running"""
        logging.debug("Starting sequence of patterns")
        if self._pattern_running:
            logging.debug("Sequence already running. Restarting it.")
            self._stop_sequence()
            self._pattern_running = True
        if bool(self._wait_for_trigger):
            self._hw_pattern_running_thread.start()
        else:
            self._sw_pattern_running_thread.start()

    def _hw_run_pattern(self):
        while self._pattern_running:
            if self._transient_patterns:
                for i, transients in enumerate(self._transient_patterns):
                    if self._pattern_running:
                        self._pattern_idx = i
                        _r = self._blink_sdk.Write_transient_frames(
                            self._slm_handle,
                            self._board,
                            transients,
                            self._wait_for_trigger,
                            self._output_pulse_image_flip,
                            self._trigger_timeout_ms,
                        )
                        logging.debug(f"applied pattern index: {self._pattern_idx}")
                        if int(_r):
                            logging.error(self._get_last_error())
                            self._pattern_running = False
                            return
                    else:
                        return

    def _sw_run_pattern(self):
        raise NotImplemented()

    @requires_slm
    def _stop_sequence(self):
        self._pattern_running = False
        if bool(self._wait_for_trigger):
            self._blink_sdk.Stop_sequence(self._slm_handle)
            if self._hw_pattern_running_thread.is_alive():
                self._hw_pattern_running_thread.join()
        else:
            if self._sw_pattern_running_thread.is_alive():
                self._sw_pattern_running_thread.join()
        logging.debug("sequence stopped")


class SLM_1024(MeadowlarkSpatialLightModulator):
    """Meadowlark Spatial Light Modulator with 1024x1024 resolution."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._shape = (
            int(self._blink_sdk.Get_image_width(self._slm_handle, self._board)),
            int(self._blink_sdk.Get_image_height(self._slm_handle, self._board)),
        )
        if self._shape != (1024, 1024):
            raise InitialiseError("The shape of the SLM is not 1024x1024. You may"
                                  "have initialized the wrong SLM device or used"
                                  "the wrong device class.")

        self._image_size = self._ffi.cast("unsigned int",
                                          self._shape[0] * self._shape[1])
        self._flip_immediate = self._ffi.cast("int", 0)

        self.add_setting(
            # TODO: Verify if only 1024 version
            name="SLM_temperature",
            dtype="float",
            get_func=self._get_temperature(),
            set_func=None,
            values=lambda: "This setting is read only. It returns the temperature of the SLM",
        )

        self.add_setting(
            name="ramp_delay",
            dtype="int",
            get_func=lambda: int(self._ramp_delay),
            set_func=self._set_ramp_delay,
            values=lambda: (1, 32),
        )

        self.add_setting(
            name="pre_ramp_slope",
            dtype="int",
            get_func=lambda: int(self._pre_ramp_slope),
            set_func=self._set_pre_ramp_slope,
            values=lambda: (1, 32),
        )

        self.add_setting(
            name="post_ramp_slope",
            dtype="int",
            get_func=lambda: int(self._post_ramp_slope),
            set_func=self._set_post_ramp_slope,
            values=lambda: (1, 32),
        )

    def _get_temperature(self):
        # Presumably this is a 1024 only method.
        # TODO: verify this point
        return float(self._blink_sdk.Read_SLM_temperature(self._slm_handle, self._board))

    def _write_pattern(self, image):
        _r = self._blink_sdk.Write_image(
            self._slm_handle,
            self._board,
            self._ffi.from_buffer(image),
            self._image_size,
            self._wait_for_trigger,
            self._flip_immediate,
            self._output_pulse_image_flip,
            self._trigger_timeout_ms,
        )
        if int(_r):
            raise Exception(self._get_last_error())

    # TODO: verify if these ramp parameters are exclusive to 1024 versions of the SLM. Modify add settings accordingly.
    def _set_ramp_delay(self, ramp_delay):
        prev = int(self._ramp_delay)
        self._ramp_delay = self._ffi.cast("unsigned int", ramp_delay)
        _r = self._blink_sdk.SetRampDelay(self._slm_handle, self._board, self._ramp_delay)
        if int(_r):
            self._ramp_delay = self._ffi.cast("unsigned int", prev)
            raise Exception(self._get_last_error())
        else:
            return None

    def _set_pre_ramp_slope(self, pre_ramp_slope):
        prev = int(self._pre_ramp_slope)
        self._pre_ramp_slope = self._ffi.cast("unsigned int", pre_ramp_slope)
        _r = self._blink_sdk.SetPreRampSlope(self._slm_handle, self._board, self._pre_ramp_slope)
        if int(_r):
            self._pre_ramp_slope = self._ffi.cast("unsigned int", prev)
            raise Exception(self._get_last_error())
        else:
            return None

    def _set_post_ramp_slope(self, post_ramp_slope):
        prev = int(self._post_ramp_slope)
        self._post_ramp_slope = self._ffi.cast("unsigned int", post_ramp_slope)
        _r = self._blink_sdk.SetPostRampSlope(self._slm_handle, self._board, self._post_ramp_slope)
        if int(_r):
            self._post_ramp_slope = self._ffi.cast("unsigned int", prev)
            raise Exception(self._get_last_error())
        else:
            return None

# class OldInterface():
#     """
#     Functions to implement:
#     In experiment:
#         slmdev.connection.get_sim_diffraction_angle
#     In Device BoulderSLM:
#         self.connection.get_sim_diffraction_angle()
#         self.connection.set_sim_diffraction_angle(newTheta)
#         self.connection.get_is_enabled()
#         self.connection.get_sim_sequence()
#         self.connection.run()
#         self.connection.stop()
#         self.connection.get_sequence_index()
#         self.asproxy.set_sim_sequence(sequence)
#     """
#     def __init__(self, use_odp=False):
#         # Logging
#         loggerName = '.'.join([__name__, self.__class__.__name__])
#         self.logger = logging.getLogger(loggerName)
#         ## SLM geometry
#         # Physical pitch in microns
#         self.pixel_pitch = 15.0
#         # SLM size in pixels
#         self.pixels = (512, 512)
#         # Might as well evaluate image indices only once
#         x_range = np.arange(self.pixels[0])
#         y_range = np.arange(self.pixels[1])
#         self.kk, self.ll = np.meshgrid(x_range, y_range)
#         ## Image sequence
#         self.sequence = []
#         self.sequence_parameters = []
#         ## SIM parameters
#         self.use_ODP = use_odp
#         self.sim_phase_offset = 0
#         self.sim_angle_offset = TWO_PI / 5.
#         self.sim_num_phases = 5
#         self.sim_num_angles = 3
#         self.sim_diffraction_angle = 0.46  # degrees was 0.5 then .25
#         ## Look-up tables and calibration data
#         # Paths
#         self._LUTFolder = "LUT_files"
#         self._calibrationFolder = "Phase_Calibration_Files"
#         # Mapped by wavelength
#         self.luts = {}
#         self.calibs = {}
#         ## Connect to the hardware.
#         self.hardware = BNSDevice()
#         ## Initialize the hardware.
#         self.hardware.initialize()
#
#     def get_sequence(self):
#         return self.sequence
#
#     def get_sim_sequence(self):
#         return self.sequence_parameters
#
#     def set_sim_sequence(self, angle_phase_wavelength):
#         """ Generate a SIM sequence from a list of parameters.
#
#         angle_phase_wavelength is a list where each element is a tuple of the
#         form (angle_number, phase_number, wavelength).
#         """
#         logging.info(f'calling set_sim_sequence with: {angle_phase_wavelength}')
#         num_phases = 0
#         num_angles = 0
#         wavelengths = []
#         for (angle, phase, wavelength) in angle_phase_wavelength:
#             num_phases = max(num_phases, phase + 1)
#             num_angles = max(num_angles, angle + 1)
#             wavelengths.append(wavelength)
#
#         phases = [self.sim_phase_offset + n * TWO_PI / num_phases
#                   for n in range(num_phases)]
#         angles = [self.sim_angle_offset + n * TWO_PI / num_angles
#                   for n in range(num_angles)]
#
#         ## Calculate line pitches for each wavelength, once.
#         # d  = m * wavelength / np.sin theta
#         # 1/1000 since wavelength in nm, pixel pitch in microns.
#         pitches = {w: w / (1000. * np.sin(self.sim_diffraction_angle * TWO_PI / 360.))
#                    for w in set(wavelengths)}
#
#         patterns = []
#         for (angle, phase, wavelength) in angle_phase_wavelength:
#             # retardation for equal powers in 0 and combined +/-1 orders
#             modulation = 65535 * self.get_modulation_factor(wavelength) / 360.0
#             pp = pitches[wavelength] / self.pixel_pitch
#             th = angles[angle]
#             ph = phases[phase]
#             # Create a stripe 16-bit pattern
#             pattern = numpy.ushort(
#                 np.rint(
#                     (0.5 * modulation) + (0.5 * modulation) * np.cos(
#                         ph + TWO_PI * (np.cos(th) * self.kk + np.sin(th) * self.ll)
#                         / pp)
#                     ))
#             patterns.append(pattern)
#         self.set_sequence(wavelengths, patterns)
#
#         self.sequence_parameters = angle_phase_wavelength
#
#     def dump_sequence(self):
#         from matplotlib import pyplot as plt
#         for n, im in enumerate(self.sequence):
#             fn = ''.join(['-', str(n), '.jpeg'])
#             implot = plt.imshow(im)
#             implot.set_cmap('gray')
#             plt.savefig(fn)
#         return np.amin(self.sequence), np.amax(self.sequence)
#
#     def get_lut(self, wavelength):
#         """ Returns the LUT closest to wavelength. """
#         lut_wavelengths = self.luts.keys()
#         nearest = min(lut_wavelengths, key=lambda x: abs(x - wavelength))
#         return self.luts[nearest]
#
#     def load_sequence(self):
#         """ Loads images to the device. """
#         if not self.sequence:
#             raise MicroscopeError(
#                 'No data to load to SLM --- generate sequence then load.')
#         else:
#             self.hardware.load_sequence(self.sequence)
#         return None
#
#     def set_test_sequence(self, wavelength=488):
#         """ Generate a series of test images. """
#         from PIL import Image, ImageDraw, ImageFont
#         sequence = []
#         labels = range(15)
#         imsize = self.pixels
#         font = ImageFont.truetype('arial.ttf', imsize[0]//2)
#         for c in labels:
#             image = Image.new('L', imsize)
#             draw = ImageDraw.Draw(image)
#             draw.setink(255)
#             draw.text((128, 0), str(c), font=font)
#             pattern16 = numpy.array(image.getdata(),
#                                     dtype=numpy.ushort).reshape(imsize)
#             pattern16 *= (65535 * 123.9 / 360) / pattern16.max()
#             # pattern = lut[pattern16 / 4]
#             pattern = pattern16
#             # Append to the sequence.
#             if self.use_ODP:
#                 sequence.append((pattern, wavelength))
#             else:
#                 sequence.append(pattern)
#         self.sequence_parameters = map(lambda x: (x, 0, 0), labels)
#         self.sequence = sequence
#         self.load_sequence()
#
#     def get_shape(self):
#         """ Return the device shape in pixels. """
#         return self.pixels
#
#     def set_sequence(self, wavelengths, patterns):
#         """ Generate sequence from given wavelengths and patterns.
#
#         Accepts:
#           single wavelength, N patterns;
#           N wavelengths, N patterns
#
#         Patterns should be arrays of 16-bit unsigned integers; they will be
#         reshaped and rescaled to the device size and bit depth.
#         SLM shape can be queried with get_shape().
#         """
#         if type(wavelengths) in [list, tuple]:
#             assert len(wavelengths) == len(patterns), \
#                 "len(wavelengths) != len(patterns)."
#         else:
#             wavelengths = len(patterns) * [wavelengths]
#         # Generate the sequence.
#         self.sequence = []
#         for p, w in zip(patterns, wavelengths):
#             # # Cast and reshape provided pattern.
#             # pattern = numpy.array(p, dtype=numpy.ushort).reshape(self.pixels)
#             self.sequence.append((p, w))
#         self.load_sequence()
#
#     def run(self):
#         """ Power on and make device respond to triggers. """
#         self.hardware.power = True
#         self.hardware._start_sequence()
#         return None
#
#     def stop(self):
#         """ Power off and stop device responding to triggers. """
#         self.hardware._stop_sequence()
#         self.hardware.power = 0
#         return None
#
#     def get_temperature(self):
#         return self.hardware.temperature
#
#     def get_is_enabled(self):
#         return int(self.hardware.power)
#
#     def get_power(self):
#         return int(self.hardware.power)
#
#     def get_sequence_index(self):
#         index = self.hardware.curr_seq_image
#         # Index is actually that of the image that will be displayed
#         # on the next trigger.
#         return index - 1 if index > 0 else len(self.sequence) - 1
#
#     def get_sim_diffraction_angle(self):
#         return self.sim_diffraction_angle
#
#     def set_sim_diffraction_angle(self, angle):
#         self.sim_diffraction_angle = float(angle)
#
#     def get_modulation_factor(self, wavelength):
#         if self.use_ODP:
#             return ODP_MODULATION_FACTORS[wavelength]
#         else:
#             return MODULATION_FACTORS[wavelength]
#
#     def set_modulation_factor(self, factor, wavelength):
#         if 0 >= factor >= 360:
#             raise ValueError('The modulation factor must have a value between 0 and 360')
#         if self.use_ODP:
#             ODP_MODULATION_FACTORS[wavelength] = factor
#         else:
#             MODULATION_FACTORS[wavelength] = factor
#
#     def single_frame(self, index):
#         self.hardware._stop_sequence()
#         self.hardware.write_image(image=self.sequence[index][0], wavelength=self.sequence[index][1])
#
#     def run_modulation_calibration_seq(self, wavelength, mod_start=80, mod_stop=220, step=10):
#         current_mod_factor = self.get_modulation_factor(wavelength)
#         for test_mod_factor in range(mod_start, mod_stop, step):
#             seq = [(0, 0, wavelength), (0, 0, wavelength)]
#             self.set_modulation_factor(test_mod_factor, wavelength)
#             self.set_sim_sequence(seq)
#             self.run()
#             if input(f'Current modulation factor is: {test_mod_factor}. '
#                      f'Trigger a few times the SLM to start to measure or type any key and enter to stop sequence'):
#                 self.stop()
#                 break
#             self.stop()
#         self.set_modulation_factor(current_mod_factor, wavelength)
#
#     def run_calibration(self,
#                         camera,
#                         camera_roi=(512, 512, 1024, 1024),
#                         exp_time=0.01,
#                         pattern="stripes",
#                         bit_depth=8,
#                         step_size=1,
#                         pattern_size=4,
#                         scale="upper",
#                         use_odp=False,
#                         apply_linear=True,
#                         wavelength=None):
#
#         image_buffer = Queue()
#         camera.set_exposure_time(exp_time)
#         camera.set_trigger(TriggerType.SOFTWARE, TriggerMode.ONCE)
#         camera.set_roi(camera_roi)
#         camera.enable()
#
#         # For the moment do the tests in non ODP mode
#         self.hardware.use_odp = use_odp
#
#         # Load a linear LUT
#         if apply_linear:
#             self.hardware._load_linear_lut()
#         else:
#             self.hardware._load_wavelength_lut(wavelength)
#
#         scale_max = np.iinfo(np.dtype(f"uint{bit_depth}")).max
#         scale_min = np.iinfo(np.dtype(f"uint{bit_depth}")).min
#
#         if pattern == "stripes":
#             pattern_generator = stripe_pattern
#         elif pattern == "checkerboard":
#             pattern_generator = checkerboard_pattern
#         else:
#             raise MicroscopeError("unknown calibration pattern")
#
#         for p in range(scale_min, scale_max, step_size):
#             if scale == "upper":
#                 min_val = p * step_size
#                 pattern_image = pattern_generator(min_val, scale_max,
#                                                   stripes_width=pattern_size,
#                                                   bit_depth=bit_depth,
#                                                   pattern_shape=self.pixels)
#             if scale == "lower":
#                 max_val = scale_max - (p * step_size)
#                 pattern_image = pattern_generator(scale_min, max_val,
#                                                   stripes_width=pattern_size,
#                                                   bit_depth=bit_depth,
#                                                   pattern_shape=self.pixels)
#             if scale == "mid":
#                 max_val = scale_max - (p * step_size)//2
#                 min_val = (p * step_size)//2
#                 pattern_image = pattern_generator(min_val, max_val,
#                                                   stripes_width=pattern_size,
#                                                   bit_depth=bit_depth,
#                                                   pattern_shape=self.pixels)
#
#             self.hardware.write_image(pattern_image)
#             print(f"Acquiring phase {p} out of {scale_max}")
#             time.sleep(.2)
#             image_buffer.put(camera.trigger_and_wait()[0])
#             # time.sleep(exp_time + .2)
#
#         with TiffWriter(f'cal_{time.time()}_ODP-{use_odp}_linearLUT-{apply_linear}_wavelength-{wavelength}.ome-tif') as tif:
#             while not image_buffer.empty():
#                 tif.write(image_buffer.get(),
#                           metadata={'axes': 'YX'})
#
#         print("Calibration done.")
