#!/usr/bin/env python3

## Copyright (C) 2024 CNRS Julio Mateos Langerak <julio.mateos-langerak@igh.cnrs.fr>
## Copyright 2014-2015 Mick Phillips (mick.phillips at gmail dot com)
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

"""Wrapper of a microscope SLM hardware device and implements all the functionality to do 3D-SIM.
This module is based on Mick Phillips: https://github.com/mickp/bnsdevice"""
from typing import Dict

import numpy as np
import logging

import microscope.abc
from microscope import TriggerType, TriggerMode
# from bns_device_odp import BNSDevice_ODP as BNSDevice

TWO_PI = 2. * np.pi

class SIM_SLM(microscope.abc.Device):
    """This wrapper microscope device takes a microscopeSLM device and adds all the functionality to do 3D-SIM.
    """
    def __init__(
            self,
            slm: microscope.abc.SpatialLightModulator,
            slm_kwargs: Dict,
            sim_diffraction_angle: float = None,
            sim_modulation_factors: Dict[int, int] = None,
            pixel_pitch: float = 15.0,
    ):
        super().__init__()
        self._slm = slm(**slm_kwargs)
        self._sim_diffraction_angle = sim_diffraction_angle
        self._sim_modulation_factors = sim_modulation_factors
        self._pixel_pitch = pixel_pitch  # microns  # TODO: This should go into the SLM
        self._sim_phase_offset = 0.
        self._sim_angle_offset = TWO_PI / 5.
        self._sim_num_phases = 5
        self._sim_num_angles = 3

        self.kk, self.ll = np.meshgrid(np.arange(self._slm.get_shape()[0]), np.arange(self._slm.get_shape()[1]))
        self._sequence_parameters = []

        self.add_setting(
            name='sim_diffraction_angle',
            dtype="float",
            get_func=lambda: self._sim_diffraction_angle,
            set_func=self._set_sim_diffraction_angle,
            values=lambda: (0.0, 2.0),
            readonly=self.get_is_enabled,
        )

        self.add_setting(
            name='sim_phase_offset',
            dtype="float",
            get_func=lambda: self._sim_phase_offset,
            set_func=lambda x: self._set_sim_phase_offset(x),
            values=lambda: (0.0, TWO_PI),
            readonly=self.get_is_enabled,
        )

        self.add_setting(
            name='sim_num_phases',
            dtype="int",
            get_func=lambda: self._sim_num_phases,
            set_func=lambda x: self._set_sim_num_phases(x),
            values=lambda: (0, 255),
            readonly=self.get_is_enabled,
        )

        self.add_setting(
            name='sim_num_angles',
            dtype="int",
            get_func=lambda: self._sim_num_angles,
            set_func=lambda x: self._set_sim_num_angles(x),
            values=lambda: (0, 255),
            readonly=self.get_is_enabled,
        )

    def _set_sim_diffraction_angle(self, angle):
        self._sim_diffraction_angle = angle

    def _set_sim_phase_offset(self, offset):
        self._sim_phase_offset = offset

    def _set_sim_num_phases(self, num_phases):
        self._sim_num_phases = num_phases

    def _set_sim_num_angles(self, num_angles):
        self._sim_num_angles = num_angles

    def set_sim_modulation_factors(self, factors: dict):
        self._sim_modulation_factors = {int(w): int(f) for w, f in factors.items()}

    def get_sim_modulation_factors(self):
        return self._sim_modulation_factors

    def get_sim_sequence(self):
        return self._sequence_parameters

    def get_sequence_index(self):
        return self._slm._pattern_idx

    def get_patterns(self):
        return self._slm._patterns

    def run(self):
        self._slm.run_queue()

    def stop(self):
        self._slm.stop_queue()

    def set_sim_sequence(self, angle_phase_wavelength):
        """Generate a SIM sequence from a list of parameters.
        angle_phase_wavelength is a list where each element is a tuple of the
        form (angle_number, phase_number, wavelength).
        """
        logging.info(f'calling set_sim_sequence with: {angle_phase_wavelength}')
        num_phases = 0
        num_angles = 0
        wavelengths = []
        for (angle, phase, wavelength) in angle_phase_wavelength:
            num_phases = max(num_phases, phase + 1)
            num_angles = max(num_angles, angle + 1)
            if wavelength not in wavelengths:
                wavelengths.append(wavelength)

        phases = [self._sim_phase_offset + n * TWO_PI / num_phases
                  for n in range(num_phases)]
        angles = [self._sim_angle_offset + n * TWO_PI / num_angles
                  for n in range(num_angles)]

        # Calculate line pitches for each wavelength, once.
        # d = m * wavelength / np.sin theta
        # 1/1000 since wavelength in nm, pixel pitch in microns.
        pitches = {w: w / (1000. * np.sin(self._sim_diffraction_angle * TWO_PI / 360.))
                   for w in wavelengths}

        patterns = np.zeros((len(angle_phase_wavelength), *self._slm.get_shape()), dtype=np.float32)
        wavelength_seq = []
        for i, (angle, phase, wavelength) in enumerate(angle_phase_wavelength):
            # retardation for equal powers in 0 and combined +/-1 orders
            modulation = self._sim_modulation_factors[wavelength] / 360.0

            pp = pitches[wavelength] / self._pixel_pitch
            th = angles[angle]
            ph = phases[phase]
            # Create a stripe float pattern
            patterns[i] = ((0.5 * modulation) + (0.5 * modulation) * np.cos(
                        ph + TWO_PI * (np.cos(th) * self.kk + np.sin(th) * self.ll)
                        / pp)).astype(np.float32)
            # Lose two LSBs and pass through the LUT for given wavelength.
            wavelength_seq.append(wavelength)

        self._sequence_parameters = angle_phase_wavelength
        self._slm.queue_patterns(patterns, wavelength_seq)

    def _do_enable(self):
        return self._slm.enable()

    def _do_disable(self):
        return self._slm.disable()

    def _do_shutdown(self) -> None:
        self.stop()
        self._slm.shutdown()

