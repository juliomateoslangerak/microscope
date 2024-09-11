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
    ):
        super().__init__()
        self._slm = slm
        self._sim_diffraction_angle = None
        self._sim_modulation_factor = None
        self._pixel_pitch = 15.0  # microns  # TODO: This should go into the SLM
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
            name='sim_modulation_factor',
            dtype="int",
            get_func=lambda: self._sim_modulation_factor,
            set_func=lambda x: self._set_sim_modulation_factor(x),
            values=lambda: (0, 360),
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

    def _set_sim_modulation_factor(self, factor):
        self._sim_modulation_factor = factor

    def _set_sim_phase_offset(self, offset):
        self._sim_phase_offset = offset

    def _set_sim_num_phases(self, num_phases):
        self._sim_num_phases = num_phases

    def _set_sim_num_angles(self, num_angles):
        self._sim_num_angles = num_angles

    def get_sim_sequence(self):
        return self._sequence_parameters

    def get_sequence_index(self):
        return self._slm._pattern_idx

    def run(self):
        self._slm.enable()

    def stop(self):
        self._slm.disable()

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

        ## Calculate line pitches for each wavelength, once.
        # d  = m * wavelength / np.sin theta
        # 1/1000 since wavelength in nm, pixel pitch in microns.
        pitches = {w: w / (1000. * np.sin(self._sim_diffraction_angle * TWO_PI / 360.))
                   for w in wavelengths}
        # ## Figure out the LUTs we need for each wavelength, once.
        # if not self.use_ODP:
        #     luts = {w: self.get_lut(w) for w in set(wavelengths)}

        # retardation for equal powers in 0 and combined +/-1 orders
        modulation = self._sim_modulation_factor / 360.0

        patterns = np.zeros((len(angle_phase_wavelength), *self._slm.get_shape()), dtype=np.float64)
        wavelength_seq = []
        for i, (angle, phase, wavelength) in enumerate(angle_phase_wavelength):
            pp = pitches[wavelength] / self._pixel_pitch
            th = angles[angle]
            ph = phases[phase]
            # Create a stripe float pattern
            patterns[i] = (0.5 * modulation) + (0.5 * modulation) * np.cos(
                        ph + TWO_PI * (np.cos(th) * self.kk + np.sin(th) * self.ll)
                        / pp)
            # Lose two LSBs and pass through the LUT for given wavelength.
            wavelength_seq.append(wavelength)

        self._sequence_parameters = angle_phase_wavelength
        self._slm.queue_patterns(patterns, wavelength_seq)

    def _do_shutdown(self) -> None:
        self.stop()
        self._slm.shutdown()






##################
"""
SLM SERVICE
from bns_device_odp import BNSDevice_ODP as BNSDevice

CONFIG_NAME = 'slm'
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
LOG_DATE_FORMAT = '%m-%d %H:%M'
TWO_PI = 2. * np.pi

#Pyro4.config.SERIALIZERS_ACCEPTED.remove('serpent')
Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')
#Pyro4.config.SERIALIZER='pickle'
Pyro4.config.REQUIRE_EXPOSE = False

logging.basicConfig(level=logging.INFO,
                    format=LOG_FORMAT,
                    datefmt=LOG_DATE_FORMAT,
                    filename='slmservice.log',
                    filemode='w')


def stripe_pattern(min_val, max_val, stripes_width=4, bit_depth=8, pattern_shape=(512, 512)):
    pattern = np.zeros(shape=pattern_shape, dtype=np.dtype(f"uint{bit_depth}"))
    pattern.fill(max_val)
    for s_nr in range(pattern_shape[0] // stripes_width):
        if (s_nr % 2) == 0:
            pattern[(s_nr * stripes_width): ((s_nr + 1) * stripes_width), :] = min_val

    return pattern


def sine_pattern(min_val, max_val, stripes_width=4, bit_depth=8, pattern_shape=(512, 512)):
    x = np.linspace(0, (np.pi/stripes_width)*pattern_shape[0], pattern_shape[0])
    norm_sine = (((np.sin(x) + 1) / (1 + 1)) * (max_val - min_val)) + min_val
    pattern = np.stack([norm_sine.astype(f"uint{bit_depth}") for _ in range(pattern_shape[1])], axis=1)

    return pattern


def checkerboard_pattern(min_val, max_val, squares_width=4, bit_depth=8, pattern_shape=(512, 512)):
    tile = np.zeros((squares_width * 2, squares_width * 2), dtype=np.dtype(f"uint{bit_depth}"))
    tile.fill(max_val)
    tile[:squares_width, :squares_width] = min_val
    tile[-squares_width:, -squares_width:] = min_val
    pattern = np.tile(tile, tuple(x // (squares_width * 2) for x in pattern_shape))

    if pattern.shape != pattern_shape:
        raise ValueError('The pattern shape must be a multiple of the square_width.')

    return pattern


@Pyro4.expose
class SpatialLightModulator(object):
    def __init__(self, use_odp=False):
        # Logging
        loggerName = '.'.join([__name__, self.__class__.__name__])
        self.logger = logging.getLogger(loggerName)
        ## SLM geometry
        # Physical pitch in microns
        self.pixel_pitch = 15.0
        # SLM size in pixels
        self.pixels = (512, 512)
        # Might as well evaluate image indices only once
        x_range = np.arange(self.pixels[0])
        y_range = np.arange(self.pixels[1])
        self.kk, self.ll = np.meshgrid(x_range, y_range)
        ## Image sequence
        self.sequence = []
        self.sequence_parameters = []
        ## SIM parameters
        self.use_ODP = use_odp
        self.sim_phase_offset = 0
        self.sim_angle_offset = TWO_PI / 5.
        self.sim_num_phases = 5
        self.sim_num_angles = 3
        self.sim_diffraction_angle = 0.45  # degrees was 0.5 then .25
        self.modulation_factor = 190
        ## Look-up tables and calibration data
        # Paths
        self._LUTFolder = "LUT_files"
        self._calibrationFolder = "Phase_Calibration_Files"
        # Mapped by wavelength
        self.luts = {}
        self.calibs = {}
        # Load calib. data and LUT files
        self.load_calibration_data()
        ## Instantiate the hardware.
        self.hardware = BNSDevice(use_odp=self.use_ODP)
        ## Initialize the hardware.
        self.hardware.initialize()

    def get_sequence(self):
        return self.sequence

    def get_sim_sequence(self):
        return self.sequence_parameters

    def set_sim_sequence(self, angle_phase_wavelength):
        " Generate a SIM sequence from a list of parameters.

        angle_phase_wavelength is a list where each element is a tuple of the
        form (angle_number, phase_number, wavelength).
        "
        logging.info(f'calling set_sim_sequence with: {angle_phase_wavelength}')
        num_phases = 0
        num_angles = 0
        wavelengths = []
        for (angle, phase, wavelength) in angle_phase_wavelength:
            num_phases = max(num_phases, phase + 1)
            num_angles = max(num_angles, angle + 1)
            if wavelength not in wavelengths:
                wavelengths.append(wavelength)

        phases = [self.sim_phase_offset + n * TWO_PI / num_phases
                  for n in range(num_phases)]
        angles = [self.sim_angle_offset + n * TWO_PI / num_angles
                  for n in range(num_angles)]

        ## Calculate line pitches for each wavelength, once.
        # d  = m * wavelength / np.sin theta
        # 1/1000 since wavelength in nm, pixel pitch in microns.
        pitches = {w: w / (1000. * np.sin(self.sim_diffraction_angle * TWO_PI / 360.))
                   for w in wavelengths}
        ## Figure out the LUTs we need for each wavelength, once.
        if not self.use_ODP:
            luts = {w: self.get_lut(w) for w in set(wavelengths)}

        # retardation for equal powers in 0 and combined +/-1 orders
        modulation = 65535 * self.modulation_factor / 360.0

        sequence = []
        for (angle, phase, wavelength) in angle_phase_wavelength:
            pp = pitches[wavelength] / self.pixel_pitch
            th = angles[angle]
            ph = phases[phase]
            # Create a stripe 16-bit pattern
            pattern16 = numpy.ushort(
                np.rint(
                    (0.5 * modulation) + (0.5 * modulation) * np.cos(
                        ph + TWO_PI * (np.cos(th) * self.kk + np.sin(th) * self.ll)
                        / pp)
                    ))              
            # Lose two LSBs and pass through the LUT for given wavelength.
            # Append to the sequence. If use_odp we must load together with the images the wavelength as computing
            # the transients needs that info
            sequence.append((pattern16, wavelength))
            # else:
                # pattern = luts[wavelength][pattern16 / 4]
                # sequence.append(pattern)
        self.sequence_parameters = angle_phase_wavelength
        self.sequence = sequence
        self.load_sequence()

    def dump_sequence(self):
        from matplotlib import pyplot as plt
        for n, im in enumerate(self.sequence):
            fn = ''.join(['-', str(n), '.jpeg'])
            implot = plt.imshow(im)
            implot.set_cmap('gray')
            plt.savefig(fn)
        return np.amin(self.sequence), np.amax(self.sequence)

    def get_lut(self, wavelength):
        " Returns the LUT closest to wavelength. "
        lut_wavelengths = self.luts.keys()
        nearest = min(lut_wavelengths, key=lambda x: abs(x - wavelength))
        return self.luts[nearest]

    def load_calibration_data(self):
        " Loads any calibration data found below module path. "
        # module path
        modpath = os.path.dirname(__file__)
        # filename format
        pattern = r'(slm)?(?P<serial>[0-9]+)[_](at(?P<wavelength>[0-9]+)_P)'

        ## Find calibration files
        path = os.path.join(modpath, self._calibrationFolder)
        if os.path.exists(path):
            files = os.listdir(path)
            matches = [re.match(pattern, f) for f in files]
        else:
            files = []
            matches = []

        ## Load any calibration files
        self.logger.info('Loading calibration files:')
        for f, match in zip(files, matches):
            if not match:
                # Not a calibration file.
                self.logger.warning('\tignoring %s' % f)
                continue
            try:
                im = Image.open(os.path.join(path, f))
            except IOError:
                self.logger.error('\tcould not open %s' % f)
                continue
            except Exception as e:
                raise e

            if im.size == self.pixels:
                try:
                    calib_data = numpy.array(im)
                except Exception as e:
                    # Not a calibration file.
                    continue

            wavelength = int(match.groupdict()['wavelength'])
            self.calibs[wavelength] = calib_data
            # TODO: use the flatness calibration somewhere.
            self.logger.info("\tloaded data from %s." % f)

        ## Find lookup table files.        
        path = os.path.join(modpath, self._LUTFolder)
        if os.path.exists(path):
            files = os.listdir(path)
            matches = [re.match(pattern, f) for f in files]
        else:
            files = []
            matches = []

        self.logger.info('Loading LUT files:')
        for f, match in zip(files, matches):
            if not match:
                # This is not a LUT file.
                self.logger.warning('\tignoring %s' % f)
                continue

            try:
                # Load the second column of the LUT into an ndarray.
                lut_data = numpy.loadtxt(os.path.join(path, f), 
                                         usecols=(1,),
                                         dtype=numpy.ushort)
            except IOError:
                self.logger.error('\tcould not open %s' % f)
                continue
            except Exception as e:
                raise e from e

            wavelength = int(match.groupdict()['wavelength'])
            self.luts[wavelength] = lut_data
            self.logger.info("\tloaded data from %s" % f)

        return None

    def load_sequence(self):
        " Loads images to the device. "
        if not self.sequence:
            raise Exception(
                'No data to load to SLM --- generate sequence then load.')
        else:
            self.hardware.load_sequence(self.sequence)
        return None

    def set_test_sequence(self, wavelength=488):
        " Generate a series of test images. "
        from PIL import Image, ImageDraw, ImageFont
        sequence = []
        labels = range(15)
        # lut = self.get_lut(550)
        imsize = self.pixels
        font = ImageFont.truetype('arial.ttf', imsize[0]//2)
        for c in labels:
            image = Image.new('L', imsize)
            draw = ImageDraw.Draw(image)
            draw.setink(255)
            draw.text((128, 0), str(c), font=font)
            pattern16 = numpy.array(image.getdata(), 
                                    dtype=numpy.ushort).reshape(imsize)
            pattern16 *= (65535 * 123.9 / 360) / pattern16.max()
            # pattern = lut[pattern16 / 4]
            pattern = pattern16
            # Append to the sequence.
            sequence.append((pattern, wavelength))
        self.sequence_parameters = map(lambda x: (x, 0, 0), labels)
        self.sequence = sequence
        self.load_sequence()

    def get_shape(self):
        " Return the device shape in pixels. "
        return self.pixels

    def set_custom_sequence(self, wavelengths, patterns):
        " Generate sequence from given wavelengths and patterns.

        Accepts:
          single wavelength, N patterns;
          N wavelengths, N patterns

        Patterns should be arrays of 16-bit unsigned integers; they will be
        reshaped to the device size, which can be queried with get_shape().
        "
        print('calling set_custom_sequence with: ', wavelengths, patterns)
        if type(wavelengths) in [list, tuple]:
            assert len(wavelengths) == len(patterns), \
                "len(wavelengths) != len(patterns)."
        else:
            wavelengths = len(patterns) * [wavelengths]
        # Determine LUT once for each wavelength.
        luts = {w: self.get_lut(w) for w in set(wavelengths)}
        # Generate the sequence.
        self.sequence = []
        for (w, p) in zip(wavelengths, patterns):
            # Cast and reshape provided pattern.
            pattern16 = numpy.array(p, dtype=numpy.ushort).reshape(self.pixels)
            # Lose two LSBs and pass through the LUT for given wavelength.
            pattern = luts[w][pattern16 / 4]
            # Append to the sequence.
            self.sequence.append(pattern)
        # Load sequence to the hardware.
        self.load_sequence()

    def run(self):
        " Power on and make device respond to triggers. "
        self.hardware.power = True
        self.hardware.start_sequence()
        return None

    def stop(self):
        " Power off and stop device responding to triggers. "
        self.hardware.stop_sequence()
        self.hardware.power = 0
        return None

    def get_temperature(self):
        return self.hardware.temperature

    def get_is_enabled(self):
        return int(self.hardware.power)

    def get_power(self):
        return int(self.hardware.power)

    def get_sequence_index(self):
        index = self.hardware.curr_seq_image
        # Index is actually that of the image that will be displayed
        # on the next trigger.
        return index - 1 if index > 0 else len(self.sequence) - 1

    def get_sim_diffraction_angle(self):
        return self.sim_diffraction_angle

    def set_sim_diffraction_angle(self, angle):
        self.sim_diffraction_angle = float(angle)

    def get_modulation_factor(self):
        return self.modulation_factor

    def set_modulation_factor(self, factor):
        if 0 <= factor <= 360:
            self.modulation_factor = factor
        else:
            raise ValueError('The modulation factor must have a value between 0 and 360')

    def single_frame(self, index):
        self.hardware.stop_sequence()
        self.hardware.write_image(image=self.sequence[index][0], wavelength=self.sequence[index][1])

    def run_modulation_calibration_seq(self, wavelength, mod_start=80, mod_stop=220, step=10):
        current_mod_factor = self.modulation_factor
        for test_mod_factor in range(mod_start, mod_stop, step):
            seq = [(0, 0, wavelength), (0, 0, wavelength)]
            self.set_modulation_factor(test_mod_factor)
            self.set_sim_sequence(seq)
            self.run()
            if input(f'Current modulation factor is: {test_mod_factor}. '
                     f'Trigger a few times the SLM to start to measure or type any key and enter to stop sequence'):
                self.stop()
                break
            self.stop()
        self.set_modulation_factor(current_mod_factor)

    def run_calibration(self,
                        camera=None,
                        camera_roi=(512, 512, 1024, 1024),
                        exp_time=0.01,
                        pattern="stripes",
                        bit_depth=8,
                        step_size=1,
                        pattern_size=4,
                        scale="upper",
                        apply_linear=True,
                        wavelength=None):

        if camera is not None:
            image_buffer = Queue()
            camera.set_exposure_time(exp_time)
            camera.set_trigger(TriggerType.SOFTWARE, TriggerMode.ONCE)
            camera.set_roi(camera_roi)
            camera.enable()

        # Load a linear LUT
        if apply_linear:
            self.hardware.load_linear_lut()
        else:
            self.hardware.load_wavelength_lut(wavelength)

        scale_max = np.iinfo(np.dtype(f"uint{bit_depth}")).max
        scale_min = np.iinfo(np.dtype(f"uint{bit_depth}")).min

        if pattern == "stripes":
            pattern_generator = stripe_pattern
        elif pattern == "checkerboard":
            pattern_generator = checkerboard_pattern
        elif pattern == "sine":
            pattern_generator = sine_pattern
        else:
            raise Exception("unknown calibration pattern")

        for p in range(scale_min, scale_max + 1, step_size):
            if scale == "upper":
                min_val = p * step_size
                pattern_image = pattern_generator(min_val, scale_max,
                                                  stripes_width=pattern_size,
                                                  bit_depth=bit_depth,
                                                  pattern_shape=self.pixels)
            if scale == "lower":
                max_val = scale_max - (p * step_size)
                pattern_image = pattern_generator(scale_min, max_val,
                                                  stripes_width=pattern_size,
                                                  bit_depth=bit_depth,
                                                  pattern_shape=self.pixels)
            if scale == "mid":
                max_val = scale_max - (p * step_size)//2
                min_val = (p * step_size)//2
                pattern_image = pattern_generator(min_val, max_val,
                                                  stripes_width=pattern_size,
                                                  bit_depth=bit_depth,
                                                  pattern_shape=self.pixels)

            self.hardware.write_image(pattern_image)
            print(f"Acquiring phase {p} out of {scale_max}")
            time.sleep(.2)
            if camera is not None:
                image_buffer.put(camera.trigger_and_wait()[0])
                time.sleep(exp_time + .2)

        if camera is not None:
            with TiffWriter(f'cal_{time.time()}_linearLUT-{apply_linear}_wavelength-{wavelength}_pattern-{pattern}.ome-tif') as tif:
                while not image_buffer.empty():
                    tif.write(image_buffer.get(),
                              metadata={'axes': 'YX'})

        print("Calibration done.")


class Server(object):
    def __init__(self):
        self.server = None
        self.daemon_thread = None
        self.config = None
        self.run_flag = True

    def __del__(self):
        self.run_flag = False

    def run(self):
        import readconfig
        config = readconfig.config

        host = config.get(CONFIG_NAME, 'ipAddress')
        port = config.getint(CONFIG_NAME, 'port')

        self.server = SpatialLightModulator()

        daemon = Pyro4.Daemon(port=port, host=host)

        # Start the daemon in a new thread.
        self.daemon_thread = threading.Thread(
            target=Pyro4.Daemon.serveSimple,
            args = ({self.server: 'pyroSLM'},),
            kwargs = {'daemon': daemon, 'ns': False}
            )
        self.daemon_thread.start()

        # Wait until run_flag is set to False.
        while self.run_flag:
            sleep(1)

        # Do any cleanup.
        daemon.shutdown()

        self.daemon_thread.join()

    def stop(self):
        self.run_flag = False


def main():
    server = Server()
    server_thread = threading.Thread(target = server.run)
    server_thread.start()
    try:
        while True:
            sleep(1)
    except (KeyboardInterrupt, SystemExit):
        server.stop()
        server_thread.join()


if __name__ == '__main__':
    main()
    
SLM_ODP DEVICE

    @requires_slm
    def curr_seq_image(self):
        return self._sequence_index

    @property
    @requires_slm
    def power(self):
        return self.power_state

    @power.setter
    @requires_slm
    def power(self, value):
        temp_power = self.power_state
        self.power_state = value
        try:
            self.blink_sdk.SLM_power(self.slm_handle, self.power_state)
        except Exception as e:
            self.power_state = temp_power
            raise Exception(self.get_last_error()) from e

    @property
    @requires_slm
    def temperature(self):  # TODO: this is not implemented
        return 20

    # METHODS
    ## Don't call this unless an SLM was initialised:  if you do, the next call
    # can open a dialog box from some other library down the chain.
    @requires_slm
    def cleanup(self):
        try:
            self.blink_sdk.Delete_SDK(self.slm_handle)
        except Exception as e:
            raise Exception from e
        self.constructed_okay[0] = 0
        self.haveSLM = False

    def initialize(self):
        ## Need to unload and reload the DLL here.
        # Otherwise, the DLL can open an error window about having already
        # initialized another DLL, which we won't see on a remote machine.

        # Initialize the library, looking for nematic SLMs.
        try:
            self.slm_handle = self.blink_sdk.Create_SDK(self.bit_depth,
                                                        self.num_boards_found,
                                                        self.constructed_okay,
                                                        self.is_nematic_type,
                                                        self.RAM_write_enable,
                                                        self.use_GPU,
                                                        self.max_transients,
                                                        self.default_static_regional_lut_file)
            if self.num_boards_found[0] == 0:
                raise Exception("No SLM device found.")
            elif self.num_boards_found[0] > 1:
                raise Exception("More than one SLM device found. This module can only handle one device.")
            else:
                if self.constructed_okay[0] == -1:
                    raise Exception("SLM constructor did not succeed.")
                else:
                    self.haveSLM = True
        except Exception as e:
            raise Exception('Could not Initialize') from e

        # Turn off external trigger
        self.wait_for_trigger = 0

        # This is required after initialization
        self.set_true_frames(self.true_frames)

        # Verify contruction of ressounces
        if int(self.blink_sdk.Is_slm_transient_constructed(self.slm_handle)):
            raise Exception('Overdrive  frame calculation  engine  was not properly  constructed')

        # Load the default LUT
        self.load_lut(self.default_lut_file)

        # Load a white image
        self.write_image(self.cal_image)

    @requires_slm
    def load_lut(self, filename):
        if filename is None:
            return

        # We assume that the SLM has been initialized and that the linear LUT is already loaded on HW
        filename = os.path.join(self.phase_calibration_files, filename)
        if type(filename) != bytes:
            filename = filename.encode()

        lut_file = self.ffi.new('char[]', filename)
        self._r = self.blink_sdk.Load_LUT_file(self.slm_handle,
                                               self.board,
                                               lut_file)
        if int(self._r):
            raise Exception(self.get_last_error())

    @requires_slm
    def load_linear_lut(self):
        self._r = self.blink_sdk.Load_linear_LUT(self.slm_handle,
                                                 self.board)
        if int(self._r):
            raise Exception(self.get_last_error())

    def load_wavelength_lut(self, wavelength):
        "Loads the LUT to the SLM that fits the best for a specified wavelength"
        # Load the default LUT
        lut_wavelengths = self.luts.keys()
        nearest = min(lut_wavelengths, key=lambda x: abs(x - wavelength))
        lut_file = os.path.join(self.phase_calibration_files, self.luts[nearest])
        self.load_lut(lut_file)
        return None

    def write_cal(self, type, calImage):
        "A pass through for old SDK compatibility"
        return self.write_image(calImage)

    @requires_slm
    def write_image(self, image, wavelength=None, external_trigger=False):
        # This function loads an image to the SLM
        if self._sequence_running:
            raise Exception('Sequence is running. Cannot write single image')

        self.wait_for_trigger = external_trigger

        if wavelength:
            self.load_wavelength_lut(wavelength)

        image = transform_16_to_8_bit(image)

        self._r = self.blink_sdk.Write_image(self.slm_handle,
                                             self.board,
                                             self.ffi.from_buffer(image),
                                             self.slm_resolution,
                                             self.wait_for_trigger,
                                             self.external_pulse,
                                             self.trigger_timeout_ms)
        if int(self._r):
            raise Exception(self.get_last_error())

    @requires_slm
    def compute_transients(self, image):
        print('Computing Transients')
        byte_count = self.ffi.new('unsigned int*', 0)
        self.blink_sdk.Calculate_transient_frames(self.slm_handle,
                                                  self.ffi.from_buffer(image),
                                                  byte_count)
        transients = self.ffi.new('unsigned char[]', byte_count[0])
        self.blink_sdk.Retrieve_transient_frames(self.slm_handle,
                                                 transients)
        return transients

    @requires_slm
    def load_sequence(self, image_wavelength_list):
        if len(image_wavelength_list) < 2:
            raise Exception("load_sequence expects a list of two or more " \
                            "images - it was passed %s images." % len(image_wavelength_list))
        # We pre-compute here the transient images
        # Verify that the calculation engine is properly loaded
        if self.blink_sdk.Is_slm_transient_constructed(self.slm_handle) < 0:
            raise Exception('SLM transient calculation engine not properly constructed')
        # Empty the list of transient images
        self.transient_images = []
        current_wavelength = None
        for image, wavelength in image_wavelength_list:
            if wavelength != current_wavelength:
                self.load_wavelength_lut(wavelength)
                current_wavelength = wavelength
            if type(image) is np.ndarray:
                image = transform_16_to_8_bit(image, fitting='up')
                transients = self.compute_transients(image)
                self.transient_images.append(transients)
            else:
                raise Exception('Sequence of images is not in the right format')

    @requires_slm
    def start_sequence(self, external_trigger=True):
        "Sequence wil restart if already running"
        print('Starting sequence')
        self.wait_for_trigger = external_trigger
        if self._sequence_running:
            self.stop_sequence
        self._sequence_running = True
        self.t = threading.Thread(target=self._run_sequence)
        self.t.start()

    def _run_sequence(self):
        while self._sequence_running:
            if self.transient_images:
                for i, transients in enumerate(self.transient_images):
                    if self._sequence_running:
                        self._sequence_index = i
                        self._r = self.blink_sdk.Write_transient_frames(self.slm_handle,
                                                                        self.board,
                                                                        transients,
                                                                        self.wait_for_trigger,
                                                                        self.external_pulse,
                                                                        self.trigger_timeout_ms)
                        # print(self._sequence_index)
                        if int(self._r):
                            print(self.get_last_error())
                            self._sequence_running = False
                            return
                    else:
                        return

    @requires_slm
    def stop_sequence(self):
        self._sequence_running = False
        self.blink_sdk.Stop_sequence(self.slm_handle)
        self.t.join()
        print('sequence stopped')

    def set_external_trigger_timeout(self, timeout_ms):
        self.trigger_timeout_ms = timeout_ms

    @requires_slm
    def set_sequencing_framerate(self, frame_rate):
        raise NotImplemented('Not implemented in Blink_sdk')

    @requires_slm
    def set_true_frames(self, true_frames):
        self.true_frames = true_frames
        self.blink_sdk.Set_true_frames(self.slm_handle,
                                       self.true_frames)

    @requires_slm
    def get_last_error(self):
        return self.ffi.string(self.blink_sdk.Get_last_error_message(self.slm_handle))
        
"""