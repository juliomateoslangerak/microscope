#!/usr/bin/env python3

## Copyright (C) 2022 Julio Mateos Langerak <julio.mateos-langerak@igh.cnrs.fr>
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

import io
import string
import threading
import logging

import serial

import microscope
import microscope.abc

_logger = logging.getLogger(__name__)

MODEL_TO_NR_POSITIONS = {"06": 2,
                         "09": 4,
                         "12": 6}
ERROR_CODES = {0: "OK, no error",
               1: "Communication time out",
               2: "Mechanical time out",
               3: "Command error or not supported",
               4: "Value out of range",
               5: "Module isolated",
               6: "Module out of isolation",
               7: "Initializing error",
               8: "Thermal error",
               9: "Busy",
               10: "Sensor Error (May appear during self test. If code persists there is an error)",
               11: "Motor Error (May appear during self test. If code persists there is an error)",
               12: "Out of Range (e.g. stage has been instructed to move beyond its travel range)",
               13: "Over Current error"
               }
PULSES_PER_MM = 2048


class ThorlabsELLSlider(microscope.abc.FilterWheel, microscope.abc.SerialDeviceMixin):
    """Implements interface for Thorlabs ELL Multi-Position Sliders with Resonant Piezoelectric Motors.

    """

    def __init__(self, com, baud=9600, timeout=2.0, address=0, **kwargs):
        """Create ThorlabsELLSlider

        :param com: COM port
        :param baud: baud rate
        :param timeout: serial timeout
        :param address: the address of the device in teh daisy chain
        """
        self.address = str(address).encode()
        self.connection = serial.Serial(
            port=com,
            baudrate=baud,
            timeout=timeout,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
        )
        self._comms_lock = threading.RLock()

        # Getting information about hte device
        self._write(self.address + b"in")
        info = self._readline().decode()

        self.travel = int(info[21:25], 16)
        self.pulses_per_unit = int(info[25:33], 16)

        _logger.info(f"Connected to slider: s/n{info[5:13]}")
        position_count = MODEL_TO_NR_POSITIONS[info[3:5].deccode()]

        # Slider has to be initialized
        self.initialize()
        super().__init__(positions=position_count, **kwargs)

    def initialize(self) -> None:
        _logger.info("Initializing slider")
        self._write(self.address + b"gs")
        reply = self._readline()
        if reply != b"0GS00":
            raise microscope.InitialiseError(
                f"Failed to initialize: {ERROR_CODES[int(reply[3:].decode())]}"
            )
        else:
            self._write(self.address + b"ho")

    def _do_shutdown(self) -> None:
        pass

    def _do_set_position(self, new_position: int) -> None:
        pulses = new_position * self.travel * self.pulses_per_unit
        pulses = hex(pulses)[2:].zfill(8).encode()

        self._write(self.address + b"ma" + pulses)


    def _do_get_position(self):
        # Thorlabs positions start at 1, hence the -1
        try:
            return int(self._send_command("pos?")) - 1
        except TypeError:
            raise microscope.DeviceError(
                "Unable to get position of %s", self.__class__.__name__
            )

    def _readline(self):
        """Custom _readline to overcome limitations of the serial implementation."""
        result = []
        with self._lock:
            while not result or result[-1] not in ("\n", ""):
                char = self.connection.read()
                # Do not allow lines to be empty.
                if result or (char not in string.whitespace):
                    result.append(char)
        return "".join(result)

    def _send_command(self, command):
        """Send a command and return any result."""
        with self._lock:
            self.connection.write(command + self.eol)
            response = "dummy"
            while command not in response and ">" not in response:
                # Read until we receive the command echo.
                response = self._readline().strip()
            if command.endswith("?"):
                # Last response was the command. Next is result.
                return self._readline().strip()
        return None
