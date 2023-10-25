#!/usr/bin/env python3

## Copyright (C) 2023 Julio Mateos Langerak <julio.mateos-langerak@igh.cnrs.fr>
##
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

"""A module to control a stage using a Raspberry Pi and a stepper driver like the
Adafruit DC & Stepper Motor Bonnet (Product ID: 4280) or the
Motor HAT for Raspberry Pi - Mini Kit (Product ID: 2348).

Follow this tutorial to connect steppers
(https://learn.adafruit.com/adafruit-dc-and-stepper-motor-hat-for-raspberry-pi/downloads).

Additionally you will have to connect 1 or 2 end-stops for each axis.
"""

# import time
import typing
import logging

import microscope.abc

from adafruit_motorkit import MotorKit
from adafruit_motor import stepper
import board
import digitalio

_logger = logging.getLogger(__name__)

PIN_TO_GPIO = {
    4: board.D4,
    5: board.D5,
    6: board.D6,
    12: board.D12,
    13: board.D13,
    16: board.D16,
    17: board.D17,
    18: board.D18,
    19: board.D19,
    20: board.D20,
    21: board.D21,
    22: board.D22,
    23: board.D23,
    24: board.D24,
    25: board.D25,
    27: board.D27,
}


class _RPiStageAxis(microscope.abc.StageAxis):
    def __init__(self,
                 axis: str,
                 stepper: stepper.StepperMotor,
                 um_per_step: float,
                 direction: str,
                 lower_limit: float,
                 lower_endstop_pin: int,
                 lower_endstop_active: bool,
                 upper_limit: float,
                 upper_endstop_pin: int = None,
                 upper_endstop_active: bool = None,
                 ) -> None:
        """Initialize the stage axis.

        Args:
            axis: The axis name.
            um_per_step: The number of micrometers per motor step.
            direction: The direction of the axis, either "forward" or "reverse".
            lower_limit: The lower limit of the axis in micrometers.
            lower_endstop_pin: The pin number of the minimum endstop.
            lower_endstop_active: Whether the minimum endstop is active low or high.
            upper_limit: The upper limit of the axis in micrometers.
            upper_endstop_pin: The pin number of the maximum endstop.
            upper_endstop_active: Whether the maximum endstop is active low or high.
        """
        super().__init__()
        self._axis = axis
        self._stepper = stepper
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        if direction == "forward":
            self.um_per_step = um_per_step
        elif direction == "reverse":
            self.um_per_step = -um_per_step
        else:
            raise microscope.DeviceError(
                f"Invalid direction {direction} for axis {axis}."
                f"Must be either 'forward' or 'reverse'."
            )

        self.lower_endstop_pin = digitalio.DigitalInOut(PIN_TO_GPIO[lower_endstop_pin])
        self.lower_endstop_pin.direction = digitalio.Direction.INPUT
        if lower_endstop_active:
            self.lower_endstop_pin.pull = digitalio.Pull.DOWN
        else:
            self.lower_endstop_pin.pull = digitalio.Pull.UP

        self.upper_endstop_pin = digitalio.DigitalInOut(PIN_TO_GPIO[upper_endstop_pin])
        self.upper_endstop_pin.direction = digitalio.Direction.INPUT
        if upper_endstop_active:
            self.upper_endstop_pin.pull = digitalio.Pull.DOWN
        else:
            self.upper_endstop_pin.pull = digitalio.Pull.UP

        self._position = None

    def move_by(self, delta: float) -> None:
        delta_steps = int(delta / self.um_per_step)
        new_position = self._position + delta
        if not self.lower_limit < new_position < self.upper_limit:
            raise microscope.IncompatibleStateError(
                f"Cannot move axis {self._axis} by {delta} um, "
                f"new position {new_position} is out of limits "
            )

        if delta_steps > 0:
            for _ in range(delta_steps):
                self._stepper.onestep(direction=stepper.FORWARD)
        elif delta_steps < 0:
            for _ in range(-delta_steps):
                self._stepper.onestep(direction=stepper.BACKWARD)

        self._position = new_position

    def move_to(self, pos: float) -> None:
        delta = pos - self._position
        self.move_by(delta)

    @property
    def position(self) -> float:
        return self._position

    @property
    def limits(self) -> microscope.AxisLimits:
        return microscope.AxisLimits(lower=self.lower_limit, upper=self.upper_limit)

    def home(self) -> None:
        # TODO: check low or high endstop
        if self.um_per_step > 0:
            while self.lower_endstop_pin.value:
                self._stepper.onestep(direction=stepper.BACKWARD)
        else:
            while self.lower_endstop_pin.value:
                self._stepper.onestep(direction=stepper.FORWARD)
        self._position = 0
        self._stepper._current_microstep = 0
        self.move_to(self.lower_limit)

    def set_speed(self, speed: int) -> None:
        pass

    def find_limits(self) -> None:
        self.home()
        if self.um_per_step > 0:
            while self.upper_endstop_pin.value:
                self._stepper.onestep(direction=stepper.FORWARD)
        else:
            while self.upper_endstop_pin.value:
                self._stepper.onestep(direction=stepper.BACKWARD)
        self._position = self.um_per_step * self._stepper._current_microstep / 16
        if self._position > self.upper_limit:
            self.move_to(self.upper_limit)
        else:
            _logger.error(f"The defined upper limit of axis {self._axis} is beyond physical limits"
                          f"of the stage. Please check the axis length and the direction."
                          f"Modifying the upper limit to {self._position} um instead.")
            self.upper_limit = self._position
            

class RPiStage(microscope.abc.Stage):
    """Adafruit 4280 or 2348 stepper driver."""

    def __init__(self,
                 um_per_step: typing.Tuple[float],
                 direction: typing.Tuple[str],
                 lower_limit: typing.Tuple[float],
                 lower_endstop_pin: typing.Tuple[int],
                 lower_endstop_active: typing.Tuple[bool],
                 upper_limit: typing.Tuple[float],
                 upper_endstop_pin: typing.Tuple[int],
                 upper_endstop_active: typing.Tuple[bool],
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._conn = MotorKit()
        self._axes = {}
        for a in range(2):
            self._axes[str(a)] = _RPiStageAxis(
                axis=str(a),
                stepper=self._conn.__getattribute__(f"stepper{a + 1}"),
                um_per_step=um_per_step[a],
                direction=direction[a],
                lower_limit=lower_limit[a],
                lower_endstop_pin=lower_endstop_pin[a],
                lower_endstop_active=lower_endstop_active[a],
                upper_limit=upper_limit[a],
                upper_endstop_pin=upper_endstop_pin[a],
                upper_endstop_active=upper_endstop_active[a],
            )

        self.homed = False

    def _do_shutdown(self) -> None:
        for axis in self._axes.values():
            axis._stepper.release()

    def _do_enable(self) -> bool:
        if not self.homed:
            for axis in self._axes.values():
                axis.home()
            self.homed = True
        return True

    def may_move_on_enable(self) -> bool:
        return not self.homed

    @property
    def axes(self) -> typing.Mapping[str, microscope.abc.StageAxis]:
        return self._axes

    def move_by(self, delta: typing.Mapping[str, float]) -> None:
        """Move specified axes by the specified distance."""
        for axis_name, axis_delta in delta.items():
            self._axes[axis_name].move_by(axis_delta)

    def move_to(self, position: typing.Mapping[str, float]) -> None:
        """Move specified axes by the specified distance."""
        for axis_name, axis_position in position.items():
            self._axes[axis_name].move_to(axis_position)