"""
high level support for doing this and that.
"""


from collections import namedtuple
from tqdm import tqdm
import numpy
from .base import Waveform
from .prior import Priors
from .source import Source
from .detector import Detector


class WaveformDataset(Waveform):
    """Contains all the tools related to a GW waveform.
    conversion: str, BBH/BNS, optional (used for Prior/Source)
    """
    def __init__(self, sampling_frequency, duration, conversion=None):
        super().__init__(sampling_frequency,
                         duration)
        self.conversion = conversion
        self.waveform_arguments = None
        self.prior = None
        self.source = None
        self.dets = {}
        self.parameters = None

        self._mass_parameters = ['mass_1', 'mass_2', 'mass_ratio', 'chirp_mass']
        self._internal_parameters = self._mass_parameters + ['a_1', 'a_2',
                                                             'tilt_1', 'tilt_2', 'phi_12', 'phi_jl',
                                                             'luminosity_distance', 'theta_jn', 'phase']
        self._external_parameters = ['geocent_time', 'ra', 'dec', 'psi']

    def load_prior_source_detector(self, filename=None, base='bilby', dets=None, waveform_arguments=None):
        """
        base: str, bilby/pycbc, used for Source
        dets: list, eg: ['H1', 'L1'], used for Detector
        filename: str, optional, used for Prior
        waveform_arguments: dict, optional, used for Source
            A dictionary of fixed keyword arguments to pass to either
            `frequency_domain_source_model` or `time_domain_source_model`.
            Note: the arguments of frequency_domain_source_model (except the first,
            which is the frequencies at which to compute the strain) will be added to
            the WaveformGenerator object and initialised to `None`.
            - waveform_approximant  ('IMRPhenomPv2' for BBH, 'IMRPhenomPv2_NRTidal' for BNS)
                Details: https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group___l_a_l_sim_inspiral__h.html
            - reference_frequency   (50 Hz)
            - minimum_frequency     (20 Hz)
            - maximum_frequency     (sampling_frequency/2)
            - catch_waveform_errors (False)
            - pn_spin_order         (-1)
            - pn_tidal_order        (-1)
            - pn_phase_order        (-1)
            - pn_amplitude_order    (0)
            - start_time            (0)
            - mode_array:           (None)
                Activate a specific mode array and evaluate the model using those
                modes only.  e.g. waveform_arguments =
                dict(waveform_approximant='IMRPhenomHM', mode_array=[[2,2],[2,-2])
                returns the 22 and 2-2 modes only of IMRPhenomHM.  You can only
                specify modes that are included in that particular model.  e.g.
                waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
                mode_array=[[2,2],[2,-2],[5,5],[5,-5]]) is not allowed because the
                55 modes are not included in this model.  Be aware that some models
                only take positive modes and return the positive and the negative
                mode together, while others need to call both.  e.g.
                waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
                mode_array=[[2,2],[4,-4]]) returns the 22 and 2-2 of IMRPhenomHM.
                However, waveform_arguments =
                dict(waveform_approximant='IMRPhenomXHM', mode_array=[[2,2],[4,-4]])
                returns the 22 and 4-4 of IMRPhenomXHM.
        """
        self.prior = Priors(filename, self.conversion)
        self.update_waveform()  # Create self.parameters
        self._check_parameters_for_generate_waveform(self.parameters)
        self.source = Source(base=base,
                             conversion=self.conversion,
                             sampling_frequency=self.sampling_frequency,
                             duration=self.duration)
        for det in dets:
            self.dets[det] = Detector(det,
                                      sampling_frequency=self.sampling_frequency,
                                      duration=self.duration)
        self.waveform_arguments = waveform_arguments
        self.start_time = 0

    def generate_waveform(self, parameters):
        """Generate a waveform generator
        """
        return self.source.waveform_generator(parameters, **self.waveform_arguments)

    # Update
    def update_waveform(self):
        """Sample a parameter of waveform from prior sets
        """
        self.parameters = self.prior.sample(1)

    def update_noise(self):
        """Update the self.dets noise generator

        Extra:
        Both `self.frequency_array` and `self.time_array` works.
        """
        for det in self.dets.values():
            det.update(start_time=self.start_time)

    def update_detector_response(self, start_time, geocent_time=None):
        """Resample external parameters or Create/Update/resample external parameters for geocent_time

        geocent_time: tuple, optional, (minimum, maximum)
            The GPS arrival time of the signal data
        start_time: float (default: 0)
            The GPS start-time of the data / Starting time of the time array

        Extra:
        Both `self.frequency_array` and `self.time_array` updates.

        TODO:
        Need to refine this method.
        """
        self.start_time = start_time
        self.update_noise()  # Update det.strain_data.start_time
        if self._is_parameters_for_detector_response(self.parameters) and (geocent_time is None):
            self.parameters.update(self.prior.sample_subset(self._external_parameters, 1))
        elif geocent_time is not None:
            self.prior.append_cosine_prior('dec', -1.5707963267948966, 1.5707963267948966,
                                           latex_label='$\\mathrm{DEC}$', unit=None, boundary=None)
            self.prior.append_uniform_prior('geocent_time', geocent_time[0], geocent_time[1],
                                            latex_label='$t_c$', unit='$s$')
            self.prior.append_uniform_prior('ra', 0, 6.283185307179586,
                                            latex_label='$\\mathrm{RA}$', unit=None, boundary='periodic')
            self.prior.append_uniform_prior('psi', 0, 3.141592653589793,
                                            latex_label='$\\psi$', unit=None, boundary='periodic')
            self.parameters.update(self.prior.sample_subset(self._external_parameters, 1))

    @property
    def start_time(self):
        """start_time: float (default: 0)
        The GPS start-time of the data / Starting time of the time array

        Extra:
        Works when `self.update_noise`
        updates when `self.update_detector_response()` with modified `self.start_time`
        """
        self.frequency_array = self.dets[list(self.dets.keys())[0]].frequency_array
        self.time_array = self.dets[list(self.dets.keys())[0]].time_array
        return self._start_time

    @start_time.setter
    def start_time(self, start_time):
        self._start_time = start_time

    # Noise
    @property
    def frequency_colored_noise(self):
        """Generate a whitened noise w.r.t self.dets in frequency domain
        """
        strain = numpy.empty(shape=(len(self.dets.keys()), self.num_f), dtype=numpy.complex128)
        for i, (_, det) in enumerate(self.dets.items()):
            strain[i, :] = det.frequency_domain_strain
        return strain

    @property
    def time_colored_noise(self):
        """Generate a colored noise w.r.t self.dets in time domain
        """
        strain = numpy.empty(shape=(len(self.dets.keys()), self.num_t), dtype=numpy.float64)
        for i, (_, det) in enumerate(self.dets.items()):
            strain[i, :] = det.time_domain_strain
        return strain

    @property
    def frequency_whitened_noise(self):
        """Generate a whitened noise w.r.t self.dets in frequency domain
        """
        strain = numpy.empty(shape=(len(self.dets.keys()), self.num_f), dtype=numpy.complex128)
        for i, (_, det) in enumerate(self.dets.items()):
            strain[i, :] = det.frequency_domain_whitened_strain
        return strain

    @property
    def time_whitened_noise(self):
        """Generate a whitened noise w.r.t self.dets in time domain
        """
        strain = numpy.empty(shape=(len(self.dets.keys()), self.num_t), dtype=numpy.float64)
        for i, (_, det) in enumerate(self.dets.items()):
            strain[i] = det.time_domain_whitened_strain
        return strain

    # Waveform polarizations
    @property
    def frequency_waveform_polarizations(self):
        """Generate a waveforem polarization w.r.t self.parameters in frequency domain
        """
        return self.generate_waveform(self.parameters).frequency_domain_strain()

    @property
    def time_waveform_polarizations(self):
        """Generate a waveforem polarization w.r.t self.parameters in time domain
        """
        return self.generate_waveform(self.parameters).time_domain_strain()

    # Waveform response
    @property
    def frequency_waveform_response(self):
        """Generate a waveforem response w.r.t self.dets and self.parameters in frequency domain
        """
        strain = numpy.empty(shape=(len(self.dets.keys()), self.num_f), dtype=numpy.complex128)
        for i, (_, det) in enumerate(self.dets.items()):
            strain[i, :] = det.get_detector_response(self.frequency_waveform_polarizations,
                                                     dict((k, self.parameters[k][0])
                                                          for k in self._external_parameters))
        return strain

    @property
    def time_waveform_response(self):
        """Generate a waveforem response w.r.t self.dets and self.parameters in time domain
        """
        strain = numpy.empty(shape=(len(self.dets.keys()), self.num_t), dtype=numpy.float64)
        for i, (_, det) in enumerate(self.dets.items()):
            strain[i, :] = det.frequency_to_time_domain(self.frequency_waveform_response[i])[0]
        return strain

    # Noise block (unwhittened / whittened)
    def frequency_colored_noise_block(self, num):
        """Generate a data block with colored detector noises in frequency domain
        num: int
            Number of data
        """
        block = numpy.empty(shape=(num, len(self.dets.keys()), self.num_f), dtype=numpy.complex128)
        for i in tqdm(range(num)):
            self.update_noise()
            block[i] = self.frequency_colored_noise
        return block

    def time_colored_noise_block(self, num):
        """Generate a data block with colored detector noises in time domain
        num: int
            Number of data
        """
        block = numpy.empty(shape=(num, len(self.dets.keys()), self.num_t), dtype=numpy.float64)
        for i in tqdm(range(num)):
            self.update_noise()
            block[i] = self.time_colored_noise
        return block

    def frequency_whitened_noise_block(self, num):
        """Generate a data block with whitened detector noises in frequency domain
        num: int
            Number of data
        """
        block = numpy.empty(shape=(num, len(self.dets.keys()), self.num_f), dtype=numpy.complex128)
        for i in tqdm(range(num)):
            self.update_noise()
            block[i] = self.frequency_whitened_noise
        return block

    def time_whitened_noise_block(self, num):
        """Generate a data block with whitened detector noises in time domain
        num: int
            Number of data
        """
        block = numpy.empty(shape=(num, len(self.dets.keys()), self.num_t), dtype=numpy.float64)
        for i in tqdm(range(num)):
            self.update_noise()
            block[i] = self.time_whitened_noise
        return block

    # Waveform block
    def frequency_waveform_response_block(self, num, start_time, geocent_time=None, target_optimal_snr_tuple=None):
        """Generate a data block with detector responses of GW waveform in frequency domain
        num: int
            Number of data
        geocent_time: tuple, optional, (minimum, maximum)
            The GPS arrival time of the signal data
        start_time: float (default: 0)
            The GPS start-time of the data / Starting time of the time array
        target_optimal_snr_tuple： tuple, optional, (target_detector_index, target_optimal_snr)
        """
        Record = namedtuple('Record', 'block noise optimal_snr matched_filter_snr')
        block = Record(numpy.empty(shape=(num, len(self.dets.keys()), self.num_f), dtype=numpy.complex128),
                       self.frequency_colored_noise_block(num),
                       numpy.empty(shape=(num, len(self.dets.keys())), dtype=numpy.float64),
                       numpy.empty(shape=(num, len(self.dets.keys())), dtype=numpy.complex64))
        meta_data = {}
        alpha = 1
        # noise_block = self.frequency_colored_noise_block(num)
        # optimal_snr = numpy.empty(shape=(num, len(self.dets.keys())), dtype=numpy.float64)
        # matched_filter_snr = numpy.empty(shape=(num, len(self.dets.keys())), dtype=numpy.complex64)
        # block = numpy.empty(shape=(num, len(self.dets.keys()), self.num_f), dtype=numpy.complex128)
        for i in tqdm(range(num)):
            self.update_waveform()
            self.update_detector_response(start_time=start_time, geocent_time=geocent_time)

            if isinstance(target_optimal_snr_tuple, tuple):

                optimal_snr_cache = self.dets[list(self.dets.keys())[target_optimal_snr_tuple[0]]].optimal_snr(
                    self.frequency_waveform_response[target_optimal_snr_tuple[0]]
                )
                alpha = target_optimal_snr_tuple[1] / optimal_snr_cache
                block.optimal_snr[i] = numpy.concatenate(
                    [numpy.asarray([target_optimal_snr_tuple[1]])
                     if j == target_optimal_snr_tuple[0] else
                     numpy.asarray([det.optimal_snr(self.frequency_waveform_response[j] * alpha)])
                     for j, (_, det) in enumerate(self.dets.items())
                     ],
                )
            elif target_optimal_snr_tuple is None:
                block.optimal_snr[i] = numpy.concatenate(
                    [numpy.asarray([det.optimal_snr(
                     self.frequency_waveform_response[j] * alpha)])
                     for j, (_, det) in enumerate(self.dets.items())
                     ],
                )

            block.block[i] = self.frequency_waveform_response * alpha

            block.matched_filter_snr[i] = numpy.concatenate(
                [numpy.asarray([det.matched_filter_snr(
                 self.frequency_waveform_response[j] * alpha,
                 self.frequency_waveform_response[j] * alpha + block.noise[i, j])])
                 for j, (_, det) in enumerate(self.dets.items())
                 ],
            )

            for key, value in self.parameters.items():
                if key in meta_data:
                    meta_data[key][i] = value
                else:
                    meta_data[key] = numpy.empty([num, ])
                    meta_data[key][i] = value
        meta_data['matched_filter_snr'] = block.matched_filter_snr
        meta_data['optimal_snr'] = block.optimal_snr
        return block.block, meta_data, block.noise

    def time_waveform_response_block(self, num, start_time, geocent_time=None, target_optimal_snr_tuple=None):
        """Generate a data block with detector responses of GW waveform in time domain
        num: int
            Number of data
        geocent_time: tuple, optional, (minimum, maximum)
            The GPS arrival time of the signal data
        start_time: float (default: 0)
            The GPS start-time of the data / Starting time of the time array
        target_optimal_snr_tuple： tuple, optional, (target_detector_index, target_optimal_snr)

        #TODO too-many-locals
        """
        Record = namedtuple('Record', 'block noise noise_freq optimal_snr matched_filter_snr meta_data')
        block = Record(numpy.empty(shape=(num, len(self.dets.keys()), self.num_t), dtype=numpy.float64),
                       numpy.empty(shape=(num, len(self.dets.keys()), self.num_t), dtype=numpy.float64),
                       self.frequency_colored_noise_block(num),
                       numpy.empty(shape=(num, len(self.dets.keys())), dtype=numpy.float64),
                       numpy.empty(shape=(num, len(self.dets.keys())), dtype=numpy.complex64),
                       {},)
        # meta_data = {}
        alpha = 1
        # noise_freq_block = self.frequency_colored_noise_block(num)
        # noise_block = numpy.empty(shape=(num, len(self.dets.keys()), self.num_t), dtype=numpy.float64)
        # optimal_snr = numpy.empty(shape=(num, len(self.dets.keys())), dtype=numpy.float64)
        # matched_filter_snr = numpy.empty(shape=(num, len(self.dets.keys())), dtype=numpy.complex64)
        # block = numpy.empty(shape=(num, len(self.dets.keys()), self.num_t), dtype=numpy.float64)
        for i in tqdm(range(num)):
            self.update_waveform()
            self.update_detector_response(start_time=start_time, geocent_time=geocent_time)

            if isinstance(target_optimal_snr_tuple, tuple):

                optimal_snr_cache = self.dets[list(self.dets.keys())[target_optimal_snr_tuple[0]]].optimal_snr(
                    self.frequency_waveform_response[target_optimal_snr_tuple[0]]
                )
                alpha = target_optimal_snr_tuple[1] / optimal_snr_cache
                block.optimal_snr[i] = numpy.concatenate(
                    [numpy.asarray([target_optimal_snr_tuple[1]])
                     if j == target_optimal_snr_tuple[0] else
                     numpy.asarray([det.optimal_snr(self.frequency_waveform_response[j] * alpha)])
                     for j, (_, det) in enumerate(self.dets.items())
                     ],
                )
            elif target_optimal_snr_tuple is None:
                block.optimal_snr[i] = numpy.concatenate(
                    [numpy.asarray([det.optimal_snr(
                     self.frequency_waveform_response[j] * alpha)])
                     for j, (_, det) in enumerate(self.dets.items())
                     ],
                )

            block.block[i] = self.time_waveform_response * alpha

            block.matched_filter_snr[i] = numpy.concatenate(
                [numpy.asarray([det.matched_filter_snr(
                 self.frequency_waveform_response[j] * alpha,
                 self.frequency_waveform_response[j] * alpha + block.noise_freq[i, j])])
                 for j, (_, det) in enumerate(self.dets.items())
                 ],
            )

            for key, value in self.parameters.items():
                if key in block.meta_data:
                    block.meta_data[key][i] = value
                else:
                    block.meta_data[key] = numpy.empty([num, ])
                    block.meta_data[key][i] = value

            for j, (_, det) in enumerate(self.dets.items()):
                block.noise[i, j], _ = det.frequency_to_time_domain(block.noise_freq[i, j])
        block.meta_data['matched_filter_snr'] = block.matched_filter_snr
        block.meta_data['optimal_snr'] = block.optimal_snr
        return block.block, block.meta_data, block.noise

    # Check
    def _check_parameters_for_generate_waveform(self, param):
        """ Is the parameters (param) can work when generate waveform?
        """
        assert len(set(self._mass_parameters) & set(param)) == 2
        assert len((set(self._internal_parameters)-set(self._mass_parameters)) & set(param)) == 9

    def _is_parameters_for_detector_response(self, param):
        """ Is the parameters (param) can work when calculating detector response?
        """
        return bool(len(set(self._external_parameters) & set(param)) == 4)
