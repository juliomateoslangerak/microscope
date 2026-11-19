#!/usr/bin/env python3

## Copyright (C) 2025 Julio Mateos Langerak <julio.mateos-langerak@igh.cnrs.fr>
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

"""A controller for the enderscope, a microscope based on a modified 3D printer.
https://royalsocietypublishing.org/doi/10.1098/rsta.2023.0214
"""

import logging
from typing import Dict, List

import serial

import microscope
import microscope._utils
import microscope.abc

_logger = logging.getLogger(__name__)

G_CODES = {
    'move': "G0",
    'absolute': "G90",
    'relative': "G91",
    'homing': "G28",
    'finish': "M400",
    'current_position': "M114",
}

class _EnderscopeStageAxis(microscope.abc.StageAxis):
    def __init__(self, stage_conn, axis) -> None:
        super().__init__()
        self.connection = stage_conn
        self._axis = axis
        # TODO: get min and max using g-code
        self.min_limit = 0.0
        self.max_limit = 100000.0

    def _write(self, command: bytes, wait: bool = True) -> int:
        res = self.connection.write(command + b"\n")
        if wait:
            self._wait_until_idle()

        return res

    def move_by(self, delta: float) -> None:
        self.connection._write(G_CODES['relative'].encode())
        self._move(delta)
        self.connection._write(G_CODES['absolute'].encode())

    def move_to(self, pos: float) -> None:
        self._move(pos)

    def _move(self, value):
        self.connection._write(f"{G_CODES['move']} {self._axis}{value}".encode())

    @property
    def position(self) -> float:
        if self.connection.is_busy():
            _logger.warning("querying stage axis position but device is busy")
            self.connection.wait_until_idle()
        return float(self.connection.get_absolute_position(self._axis))

    @property
    def limits(self) -> microscope.AxisLimits:
        return microscope.AxisLimits(
            lower=self.min_limit, upper=self.max_limit
        )

    def _wait_until_idle(self) -> None:
        self._write(G_CODES["finish"].encode())


class EnderscopeStage(microscope.abc.Stage, microscope.abc.SerialDeviceMixin):
    def __init__(self, port: str, baudrate:int = 115200, timeout:int = 1, **kwargs) -> None:
        self.connection = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout,
            bytesize=serial.EIGHTBITS,
            stopbits=serial.STOPBITS_ONE,
            parity=serial.PARITY_NONE,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )
        self.axes = {
            a: _EnderscopeStageAxis(self.connection, a)
            for a in ["x", "y", "z"]
        }

        super().__init__(**kwargs)

        self.homed = False

    def _write(self, command: bytes, wait: bool = True) -> int:
        res = self.connection.write(command + b"\n")
        if wait:
            self._wait_until_idle()

        return res

    def _get_setting(self, command, axis, dtype, value):
        pass

    def _set_setting(self, value, command, axis):
        pass

    def _add_settings(self, settings) -> None:
        for axis, axis_settings in settings.items():
            for setting_name, setting_params in axis_settings.items():
                if setting_params["value"].replace("-", "", 1).isdigit():
                    value = int(setting_params["value"])
                    dtype = "int"
                elif (
                    setting_params["value"]
                    .replace(".", "", 1)
                    .replace("-", "", 1)
                    .isdigit()
                ):
                    value = float(setting_params["value"])
                    dtype = "float"
                else:
                    value = setting_params["value"]
                    dtype = "str"  # It might be a enum but we dont know

                if setting_params["command"] is not None:
                    read_only = False
                else:
                    read_only = True

                self.add_setting(
                    name=f"{setting_name} {axis}",
                    dtype=dtype,
                    get_func=lambda command=setting_params[
                        "command"
                    ], axis=axis, dtype=dtype, value=value: self._get_setting(
                        command, axis, dtype, value
                    ),
                    set_func=lambda value, command=setting_params[
                        "command"
                    ], axis=axis: self._set_setting(value, command, axis),
                    values=lambda: f"Units are {setting_params['units']}",
                    # readonly=read_only,
                )

    def _do_shutdown(self) -> None:
        pass

    def _do_enable(self) -> bool:
        self._write(G_CODES["homing"].encode())
        self.homed = True
        return True

    def _wait_until_idle(self) -> None:
        self._write(G_CODES["finish"].encode())

    def may_move_on_enable(self) -> bool:
        return not self.homed

    def move_by(self, delta) -> None:
        """Move specified axes by the specified distance."""
        self._write(G_CODES["relative"].encode(), wait=False)
        self._write(
            f"{G_CODES['move']} X{delta['x']} Y{delta['y']} Z{delta['z']}".encode()
        )
        self._write(G_CODES["absolute"].encode(), wait=False)

    def move_to(self, position) -> None:
        """Move specified axes by the specified distance."""
        self._write(
            f"{G_CODES['move']} X{position['x']} Y{position['y']} Z{position['z']}".encode()
        )
