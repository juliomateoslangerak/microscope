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
from microscope import (
    DeviceError,
    IncompatibleStateError,
    InitialiseError,
    DisabledDeviceError,
    UnsupportedFeatureError,
    LibraryLoadError,
)
from microscope import TriggerType, TriggerMode

from pathlib import Path
import numpy as np
import ctypes

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


class MeadowlarkSpatialLightModulator(
    microscope.abc.SpatialLightModulator, ABC
):
    """Meadowlark Spatial Light Modulator.

    This microscope device is for controlling Meadowlark Optics' Spatial Light
    Modulators (formerly Boulder Non-Linear).
    Important note: The header defs is a text file containing the headers from
    the Blink_SDK.h file with some modifications. Namely, all #include and #ifdef have been
    removed.

    :param header_definitions_path: Absolute path to the header definitions from the SDK.
    :param blink_sdk_dll_path: name of, or absolute path to, the SID4_SDK.dll file
    :param luts: dictionary with the lut files per wavelength.
    :param default_wavelength:
    :param phase_calibration_files_path:
    :param bit_depth:
    :param shape:
    :param is_nematic_type:
    :param ram_write_enable:
    :param use_gpu:
    :param max_transients:
    """

    def __init__(
        self,
        blink_sdk_dll_path: Path,
        # bit_depth: int,
        luts: dict,
        phase_calibration_files_path: Path,
        default_static_lut_file: str,
        external_trigger: bool,
        is_nematic_type: bool = True,
        ram_write_enable: bool = True,
        use_gpu: bool = True,
        # TODO: retry what a timeout=0 does
        trigger_timeout_ms: int = 1000000,
        max_transients: int = 20,
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
            ctypes.cdll.LoadLibrary(blink_sdk_dll_path)
            self._blink_sdk = ctypes.CDLL(blink_sdk_dll_path.name)
        except FileNotFoundError as e:
            raise e from e
        except IOError as e:
            raise e(f'Unable to open "{blink_sdk_dll_path}"') from e
        except Exception as e:
            raise LibraryLoadError(e) from e

        # Basic SLM parameters required for initialization
        # The board makes reference to the board id in case there are more than one SLM.
        # Currently only one is supported
        self._board = ctypes.c_uint(1)
        # We set the bit depth to 12 bits per pixel before getting the value from the SLM
        self._bit_depth = ctypes.c_uint(12)
        self._num_boards_found = ctypes.c_uint(0)
        self._constructed_okay = ctypes.c_uint(-1)
        self._is_nematic_type = ctypes.c_bool(is_nematic_type)
        self._RAM_write_enable = ctypes.c_bool(ram_write_enable)
        self._use_GPU = ctypes.c_bool(use_gpu)
        self._max_transients = ctypes.c_uint(max_transients)
        self._ramp_delay = ctypes.c_uint(10)
        self._pre_ramp_slope = ctypes.c_uint(14)
        self._post_ramp_slope = ctypes.c_uint(24)
        self._image_size = None

        # LUTs
        self._luts = {k: v.encode() for k, v in luts.items()}
        if self._default_wavelength is None:
            self._default_lut_file = list(self._luts.values())[0]
        else:
            self._default_lut_file = self._luts[self._default_wavelength]
        self._default_static_lut_file = default_static_lut_file.encode()
        self._phase_calibration_files_path = str(phase_calibration_files_path).encode()

        # We need to initialize the SLM here to get the shape and bit depth
        self._power_state = ctypes.c_uint(0)

        self._initialize()

        # Verify construction of resources
        if self._blink_sdk.Is_slm_transient_constructed(self._slm_handle):
            raise InitialiseError(
                "Overdrive  frame calculation  engine  was not properly  constructed"
            )

        # Load the default LUT
        self._load_lut(self._default_lut_file)

        # Two threads for running the patterns and a boolean to control it.
        # One thread is for running the patterns in the hardware and the other
        # is for running the patterns in the software.
        self._hw_queue_running_thread = threading.Thread(
            target=self._hw_run_queue
        )
        self._sw_queue_running_thread = threading.Thread(
            target=self._sw_run_queue
        )

        # Boolean to control triggers use
        self._wait_for_trigger = ctypes.c_bool(external_trigger)
        self._trigger_timeout_ms = ctypes.c_uint(trigger_timeout_ms)
        # Both pulse options can be false, but only one can be true.
        # You either generate a pulse when the new image begins loading to the SLM
        # or every 1.184 ms on SLM refresh boundaries,
        # or if both are false no output pulse is generated.
        self._flip_immediate = ctypes.c_int(0)  # only supported in the 1024 version
        self._output_pulse_image_flip = ctypes.c_int(0)  # only supported in the 1024 version
        self._output_pulse_image_refresh = ctypes.c_uint(0)  #only supported on 1920x1152, FW rev 1.8.

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
            values=lambda: "This setting is controlling weather an output trigger is sent when the image is written.",
            readonly=lambda: False,
        )

        self.add_setting(
            name="output_pulse_image_refresh",
            dtype="bool",
            get_func=lambda: bool(self._output_pulse_image_refresh),
            set_func=self._set_output_pulse_image_flip,
            values=lambda: "This setting is controlling weather an output trigger is sent every 1.184 ms on SLM refresh boundaries.",
            readonly=lambda: False,
        )

    def _initialize(self):
        # Need to unload and reload the DLL here.
        # Otherwise, the DLL can open an error window about having already
        # initialized another DLL, which we won't see on a remote machine.

        # if there is no static LUT file, only defined in some SLMs, we set it to 0
        try:
            self._default_static_regional_lut_file
        except NameError:
            self._default_static_regional_lut_file = 0

        # Initialize the library, looking for nematic SLMs.
        try:
            self._slm_handle = self._blink_sdk.Create_SDK(
                self._bit_depth,
                ctypes.byref(self._num_boards_found),
                ctypes.byref(self._constructed_okay),
                self._is_nematic_type,
                self._RAM_write_enable,
                self._use_GPU,
                self._max_transients,
                self._default_static_regional_lut_file,
            )
            if self._num_boards_found.value == 0:
                raise InitialiseError(f"No SLM device found: {self._get_last_error()}")
            elif self._num_boards_found[0] > 1:
                raise InitialiseError(
                    f"More than one SLM device found. This module can only handle one device."
                    f"{self._get_last_error()}"
                )
            elif self._constructed_okay[0] <= 0:
                raise InitialiseError(
                    f"SLM constructor did not succeed."
                    f"{self._get_last_error()}"
                )

        except Exception as e:
            raise InitialiseError("Error during Initialization") from e

        self._shape = (
            int(
                self._blink_sdk.Get_image_width(
                    self._slm_handle, self._board
                )
            ),
            int(
                self._blink_sdk.Get_image_height(
                    self._slm_handle, self._board
                )
            ),
        )

        self._bit_depth = self._blink_sdk.Get_image_depth(
            self._slm_handle, self._board
        )

        self._image_size = self._shape[0] * self._shape[1] * self._bit_depth // 8

    @property
    def trigger_mode(self) -> microscope.TriggerMode:
        return TriggerMode.ONCE

    @property
    def trigger_type(self) -> microscope.TriggerType:
        if self._wait_for_trigger:
            return TriggerType.FALLING_EDGE
        else:
            return TriggerType.SOFTWARE

    def set_trigger(
        self, ttype: microscope.TriggerType, tmode: microscope.TriggerMode
    ) -> None:
        if self._queue_running:
            raise IncompatibleStateError(
                "Cannot set trigger while a sequence of patterns is running."
                "Stop the sequence before setting the trigger type by disabling the device."
            )
        if tmode != TriggerMode.ONCE:
            raise UnsupportedFeatureError(
                "Only TriggerMode.ONCE is supported by this SLM"
            )

        if ttype == TriggerType.SOFTWARE:
            self._wait_for_trigger = ctypes.c_bool(False)
        else:
            self._wait_for_trigger = ctypes.c_bool(True)

    @requires_slm
    def _get_shape(self):
        return self._shape

    def _do_enable(self):
        self._power_state.value = 1
        self._blink_sdk.SLM_power(self._slm_handle, self._power_state)
        return True

    def _do_disable(self):
        self._power_state.value = 0
        self._blink_sdk.SLM_power(self._slm_handle, self._power_state)
        return True

    def _do_shutdown(self) -> None:
        self._blink_sdk.Delete_SDK(self._slm_handle)
        self._constructed_okay.value = -1

    @requires_slm
    def _load_lut(self, filename):
        lut_file = self._phase_calibration_files_path + b"\\" + filename
        _r = self._blink_sdk.Load_LUT_file(
            self._slm_handle, self._board, lut_file
        )
        if int(_r):
            raise DeviceError(self._get_last_error())

    @requires_slm
    def _load_linear_lut(self):
        # TODO: verify if this is works
        _r = self._blink_sdk.Load_linear_LUT(self._slm_handle, self._board)
        if int(_r):
            raise DeviceError(self._get_last_error())

    def _load_wavelength_lut(self, wavelength):
        """Loads the LUT to the SLM that fits the best for a specified wavelength"""
        nearest = min(self._luts.keys(), key=lambda x: abs(x - wavelength))
        self._load_lut(self._luts[nearest])

    def _write_pattern(self, pattern):
        _r = self._blink_sdk.Write_image(
            self._slm_handle,
            self._board,
            pattern.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            self._image_size,
            self._wait_for_trigger,
            self._flip_immediate,
            self._output_pulse_image_flip,
            self._output_pulse_image_refresh,
            self._trigger_timeout_ms,
        )
        if int(_r):
            raise DeviceError(self._get_last_error())

    @requires_slm
    def _do_apply_pattern(self, pattern, wavelength):
        self._load_wavelength_lut(wavelength)
        self._write_pattern(pattern)

    @abstractmethod
    def _hw_run_queue(self, start_idx: int):
        """This function is running on a separate thread to write the patterns.
        This function should be implemented in the subclasses.

        Args:
            start_idx (int): The index of the pattern to start the sequence.
        """
        raise NotImplemented()

    @abstractmethod
    def _sw_run_queue(self, start_idx: int):
        """This function is running on a separate thread to write the patterns.
        This function should be implemented in the subclasses.

        Args:
            start_idx (int): The index of the pattern to start the sequence.
        """
        raise NotImplemented()

    def _set_trigger_timeout_ms(self, timeout_ms):
        self._trigger_timeout_ms.value = timeout_ms

    @requires_slm
    def _get_last_error(self):
        """Returns the last error message from the SLM"""
        self._blink_sdk.Get_last_error_message.restype = ctypes.c_char_p
        return self._blink_sdk.Get_last_error_message(self._slm_handle)

    def _set_max_transients(self, max_transients):
        self._max_transients.value = max_transients

    def _set_output_pulse_image_flip(self, output_flip):
        self._output_pulse_image_flip.value = output_flip

    def _get_version_info(self):
        self._blink_sdk.Get_version_info.restype = ctypes.c_char_p
        return self._blink_sdk.Get_version_info(self._slm_handle)


class SLM_512(MeadowlarkSpatialLightModulator):
    """Meadowlark Spatial Light Modulator with 512x512 resolution."""

    def __init__(self, use_odp: bool = False, **kwargs):
        # OverDrive Plus parameters
        self._use_odp = ctypes.c_bool(use_odp)
        self._true_frames = ctypes.c_int(3)
        self._external_pulse = ctypes.c_bool(False)
        if self._use_odp:
            if self._default_static_lut_file is None:
                raise InitialiseError("A static LUT file is required for ODP")
            self._set_true_frames(5)
            self._default_static_regional_lut_file = (
                self._phase_calibration_files_path.join(self._default_static_lut_file)
            )
        else:
            self._set_true_frames(3)
            self._default_static_regional_lut_file = 0

        super().__init__(**kwargs)

        self.add_setting(
            name="use_odp",
            dtype="bool",
            get_func=lambda: bool(self._use_odp),
            set_func=lambda x: self._set_odp(x),
            values=lambda: "This setting is controlling the use of ODP. Disable device before setting. True or False",
            readonly=lambda: self.enabled
            or not self._blink_sdk.Is_slm_transient_constructed(
                self._slm_handle
            ),
        )

        self.add_setting(
            name="true_frames",
            dtype="int",
            get_func=lambda: int(self._true_frames.value),
            set_func=self._set_true_frames,
            values=lambda: (3, 5),
            readonly=lambda: self.enabled,
        )

        self.add_setting(
            name="external_pulse",
            dtype="bool",
            get_func=lambda: bool(self._external_pulse),
            set_func=self._set_external_pulse,
            values=lambda: "This setting is controlling the use of external pulse.",
            readonly=lambda: self.enabled,
        )

    def _write_pattern(self, pattern):
        # The 512 SLM has two different calls for loading the image depending on the use of the ODP
        if self._use_odp:
            _r = self._blink_sdk.Write_overdrive_image(
                self._slm_handle,
                self._board,
                pattern.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                self._wait_for_trigger,
                self._flip_immediate,
                self._external_pulse,
                self._trigger_timeout_ms,
            )
        else:
            _r = self._blink_sdk.Write_image(
                self._slm_handle,
                self._board,
                pattern.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                self._image_size,
                self._wait_for_trigger,
                self._flip_immediate,
                self._output_pulse_image_flip,
                self._output_pulse_image_refresh,
                self._trigger_timeout_ms,
            )
        if int(_r):
            raise DeviceError(self._get_last_error())


    @requires_slm
    def _queue_patterns(self) -> None:
        # Verify that the calculation engine is properly loaded
        if self._blink_sdk.Is_slm_transient_constructed(self._slm_handle) < 0:
            raise DeviceError(
                "SLM transient calculation engine not properly constructed"
            )

        self._transient_patterns = []
        current_wavelength = None

        for pattern, wavelength in zip(self._patterns, self._wavelengths):
            if wavelength != current_wavelength:
                self._load_wavelength_lut(wavelength)
                current_wavelength = wavelength
            transients = self._compute_transients(pattern)
            self._transient_patterns.append(transients)
        print(f"queued {len(self._transient_patterns)}")

    def _transform_dtype(self, pattern: np.ndarray) -> np.ndarray:
        """Converts a float array values 0.0 to 1.0 to 8-bit. Values outside that range are clipped"""
        pattern = np.clip(pattern, 0.0, 1.0)
        pattern = np.round(pattern * 255).astype("uint8")
        return pattern + (np.iinfo("uint8").max - pattern.max())

    @requires_slm
    def _compute_transients(self, pattern):
        byte_count = ctypes.c_uint(0)
        self._blink_sdk.Calculate_transient_frames(
            self._slm_handle,
            pattern.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            ctypes.byref(byte_count)
        )
        transients = (ctypes.c_ubyte * byte_count.value)()
        self._blink_sdk.Retrieve_transient_frames(self._slm_handle, transients)
        return transients

    @requires_slm
    def _run_queue(self, start_idx):
        """Sequence wil restart if already running"""
        self._queue_running = True
        if bool(self._wait_for_trigger):
            self._hw_queue_running_thread.start()
        else:
            self._sw_queue_running_thread.start()

    def _hw_run_queue(self, start_idx):
        first_iteration = True
        while self._queue_running:
            for i, transients in enumerate(self._transient_patterns):
                if first_iteration and i < start_idx:
                    continue
                first_iteration = False
                if self._queue_running:
                    self._pattern_idx = i
                    # print(f"waiting for trigger {i}")
                    _r = self._blink_sdk.Write_transient_frames(
                        self._slm_handle,
                        self._board,
                        transients,
                        self._wait_for_trigger,
                        self._flip_immediate,
                        self._external_pulse,
                        self._trigger_timeout_ms,
                    )
                    if int(_r):
                        logging.info(self._get_last_error())
                        self._queue_running = False
                        return
                else:
                    return

    def _sw_run_queue(self, start_idx):
        raise NotImplemented()

    def _do_trigger(self) -> None:
        # TODO: implement this function for instances without HW triggers
        raise NotImplementedError()

    @requires_slm
    def _stop_queue(self):
        self._queue_running = False
        if self._wait_for_trigger:
            self._blink_sdk.Stop_sequence(self._slm_handle)
            if self._hw_queue_running_thread.is_alive():
                self._hw_queue_running_thread.join()
            self._pattern_idx = None
        elif self._sw_queue_running_thread.is_alive():
            self._sw_queue_running_thread.join()

    def _set_odp(self, use_odp):
        # We have to change here the true_frames when we enable/disable DOP
        raise NotImplemented()

    @requires_slm
    def _set_true_frames(self, true_frames):
        self._true_frames.value = true_frames
        self._blink_sdk.Set_true_frames(self._slm_handle, self._true_frames)


class SLM_1024(MeadowlarkSpatialLightModulator):
    """Meadowlark Spatial Light Modulator with 1024x1024 resolution."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self._shape != (1024, 1024):
            raise InitialiseError(
                "The shape of the SLM is not 1024x1024. You may"
                "have initialized the wrong SLM device or used"
                "the wrong device class."
            )

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
        return float(
            self._blink_sdk.Read_SLM_temperature(self._slm_handle, self._board)
        )

    # TODO: verify if these ramp parameters are exclusive to 1024 versions of the SLM. Modify add settings accordingly.
    def _set_ramp_delay(self, ramp_delay):
        prev = self._ramp_delay.value
        self._ramp_delay.value = ramp_delay
        _r = self._blink_sdk.SetRampDelay(
            self._slm_handle, self._board, self._ramp_delay
        )
        if int(_r):
            self._ramp_delay.value = prev
            raise DeviceError(self._get_last_error())
        else:
            return None

    def _set_pre_ramp_slope(self, pre_ramp_slope):
        prev = self._pre_ramp_slope.value
        self._pre_ramp_slope.value = pre_ramp_slope
        _r = self._blink_sdk.SetPreRampSlope(
            self._slm_handle, self._board, self._pre_ramp_slope
        )
        if int(_r):
            self._pre_ramp_slope.value = prev
            raise DeviceError(self._get_last_error())
        else:
            return None

    def _set_post_ramp_slope(self, post_ramp_slope):
        prev = self._post_ramp_slope.value
        self._post_ramp_slope.value = post_ramp_slope
        _r = self._blink_sdk.SetPostRampSlope(
            self._slm_handle, self._board, self._post_ramp_slope
        )
        if int(_r):
            self._post_ramp_slope.value = prev
            raise DeviceError(self._get_last_error())
        else:
            return None
