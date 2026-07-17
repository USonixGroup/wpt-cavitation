import numpy as np
from scipy.integrate import ode

class vsbd:  # class for dynamics of a phospholipid-coated gas bubble in a viscoelastic medium
    def __init__(
        self,
        R_init,
        Rdot_init,
        frequency,
        # pac,
        time, 
        deltat,
        medium_parameters={
            # "A": 304.0e6,
            # "B": 303.9e6,
            # "nTaitEq": 7,
            # "sound_speed": 1482,
            "rho_m": 1000,  # medium density, in kg/m^3
            # "sigma": 72.5e-3,  # surface tension of water, in N/m
            "mu": 2e-3,  # viscosity of medium, in Pa.s
            # "vapour_pressure": 3.2718e3,
            # "gamma": 1.4,  # polytropic exponent of the gas inside the bubble
            "p0": 1e5,  # atp, in Pa
            "G": 0,  # [0:10:200] elastic modulus of the medium, in Pa
        },
        nd_params={
            "sigma": 72e-3,  # initial surface tension of the bubble, in N/m
            "kappa_s": 6e-9,  # dilational viscosity of the bubble shell, in kg/s
            "gamma": 1.07,  # polytropic exponent of the gas inside the bubble
        }
    ):
        self.R0 = R_init  # initial radius of the bubble
        self.rdot0 = Rdot_init  # initial velocity of the bubble wall
        self.frequency = frequency  # excitation frequency
        self.time = time  # time array
        self.deltat = deltat  # time step
        self.nd_params = nd_params
        self.nd_params["r0"] = R_init
        self.medium_parameters = medium_parameters  # medium parameters

        self.R = np.ndarray([])  # radius of the bubble as a function of time
        self.Rdot = np.ndarray([])  # velocity of the bubble wall as a function of time


    def add_pac_wave(
        self,
        frequency,
        pac,
        time_delay,
        pulse_width,
        time_initial,
        phase_shift,
        no_harmonics,
    ):
        """
        Add an additional sine wave pulse to the input pressure for the model.

        Parameters
        ----------
        frequency: float
            Frequency of the sine wave (Hz).
        pac: float
            Amplitude of the sine wave (Pa).
        time_delay: float
            Time delay of the pulse (s).
        pulse_width: float
            Width of the pulse (s).
        time_initial: float
            Initial time of the sine wave (seconds).
        phase_shift: float
            Phase shift to apply to the sine wave (radians).
        no_harmonics: int
            Number of harmonics to include in the pulse (integer).
        """
        pac_wave = np.zeros(self.time.shape)
        time = self.time
        omega = 2 * np.pi * frequency
        for i in range(1, no_harmonics + 1):
            pac_tmp = np.sin(i * omega * (time - time_initial) + phase_shift) / i
            pac_wave = pac_wave + pac_tmp
        pac_wave = (
            pac_wave * pac * np.exp(-(((time - time_delay) / pulse_width) ** 2))
        )
        if not hasattr(self, "pac_wave"):
            self.pac_wave = np.zeros(self.time.shape)
        self.pac_wave += pac_wave

    def add_tanh_wave(
        self,
        frequency,
        pac,
        time_delay,
        n_cycles,
        n_cycles_taper,
        phase_shift
    ):
        """
        Add a tanh-shaped pulse to the input pressure for the model. The pulse is 
        defined by a carrier sine wave at the specified frequency, modulated by a 
        tanh envelope that smoothly turns on and off over the specified number of cycles.
        Parameters
        ----------
        frequency: float
            Frequency of the sine wave (Hz).
        pac: float
            Amplitude of the sine wave (Pa).
        time_delay: float
            Time delay of the pulse (s).
        n_cycles: int
            Number of cycles of the sine wave to include in the pulse.
        n_cycles_taper: int
            Number of cycles over which to apply the tanh taper at the start and end of 
            the pulse.
        phase_shift: float
            Phase shift to apply to the sine wave (radians).
        """
        time = self.time
        omega = 2 * np.pi * frequency

        # Total duration and taper duration
        T_total = n_cycles / frequency
        T_taper = n_cycles_taper / frequency

        # Define start and end times (centered at time_delay)
        t_start = time_delay - T_total / 2
        t_end = time_delay + T_total / 2

        # Width parameter controls how sharp the tanh transition is
        width = T_taper / 5.0  # heuristic choice

        # Tanh envelope (smooth on/off)
        rise = 0.5 * (1 + np.tanh((time - t_start) / width))
        fall = 0.5 * (1 + np.tanh((t_end - time) / width))
        envelope = rise * fall

        # Carrier wave with phase shift
        carrier = np.sin(omega * (time - time_delay) + phase_shift)

        # Apply envelope and amplitude
        pac_wave = pac * carrier * envelope

        # Accumulate
        if not hasattr(self, "pac_wave"):
            self.pac_wave = np.zeros_like(time)

        self.pac_wave += pac_wave

    def solver(self):
        x0 = np.array([self.R0, self.rdot0])
        if not hasattr(self, "pac_wave"):
            self.pac_wave = np.zeros(self.time.shape)

        solver = ode(self.KVmodel).set_integrator(
            "vode", method="bdf", atol=1e-6, rtol=1e-6
        )
        solver.set_initial_value(x0, self.time[0]).set_f_params(
            self.pac_wave, 
            self.nd_params, 
            self.medium_parameters,
            self.deltat
        )

        x = np.zeros((len(self.time), len(x0)))
        x[0] = x0

        for i in range(1, len(self.time)):
            solver.integrate(self.time[i])
            x[i] = solver.y

        self.R = x[:, 0]
        self.Rdot = x[:, 1]


    @staticmethod
    def KVmodel(t, x, u, nd_params, medium_params, deltat):
        "Kelvin-Voigt model for a viscoelastic medium with a spherically symmetric phospholipid-coated gas bubble."
        r, rdot = x
        P_ac = u[min(int(t / deltat), len(u) - 1)]  # acoustic pressure at time t

        sigma = nd_params["sigma"]
        kappa_s = nd_params["kappa_s"]
        r_0 = nd_params["r0"]
        gamma = nd_params["gamma"]

        rho_m = medium_params["rho_m"]
        mu = medium_params["mu"]
        G = medium_params["G"]
        P_0 = medium_params["p0"]

        dx1 = rdot
        dx2 = (
            1/r * (
                1 / rho_m * (
                    (P_0 + 2 * sigma / r_0) * (r_0 / r) ** (3 * gamma)
                    - P_0 - P_ac
                    - 2 * sigma / r - 4 * kappa_s * rdot / r
                    - 4 * mu * rdot / r - 4 / 3 * G * (r ** 3 - r_0 ** 3) / (r ** 3)
                )
                - 3 / 2 * rdot ** 2
            )
        )

        return np.array([dx1, dx2])
    
    @staticmethod
    def KVmodel_array(x, u, nd_params, medium_params):
        "Kelvin-Voigt model for a viscoelastic medium with a spherically symmetric phospholipid-coated gas bubble. This version is for array inputs."
        r = x[0]
        rdot = x[1]
        P_ac = u  # acoustic pressure array

        sigma = nd_params["sigma"]
        kappa_s = nd_params["kappa_s"]
        r_0 = nd_params["r0"]
        gamma = nd_params["gamma"]

        rho_m = medium_params["rho_m"]
        mu = medium_params["mu"]
        G = medium_params["G"]
        P_0 = medium_params["p0"]

        dx1 = rdot
        dx2 = (
            1/r * (
                1 / rho_m * (
                    (P_0 + 2 * sigma / r_0) * (r_0 / r) ** (3 * gamma)
                    - P_0 - P_ac
                    - 2 * sigma / r - 4 * kappa_s * rdot / r
                    - 4 * mu * rdot / r - 4 / 3 * G * (r ** 3 - r_0 ** 3) / (r ** 3)
                )
                - 3 / 2 * rdot ** 2
            )
        )

        return np.array([dx1, dx2])
    
    def calculate_p_radiated(self, r_measure):
        """
        Calculating the radiated pressure from oscillating bubble using the model 
        derived by Vokurka. Refer to Tim Leighton, The Acoustic Bubbles, chapter 3, 
        eqs(3.114, 3.115), page 151.

        Parameters
        ----------
        r_measure: float
            distance r from the bubble centre to calculate the radiated pressure 
            (acoustic emission), in meters.

        Returns
        -------
        solutions as attributes of the bubble dynamics object
        """

        _, self.Rddot = self.KVmodel_array(
            [self.R, self.Rdot],
            self.pac_wave,
            self.nd_params,
            self.medium_parameters
        )

        p_rad = self.medium_parameters["rho_m"] * self.R / r_measure * (
            2 * self.Rdot ** 2 + self.R * self.Rddot
        )

        self.p_radiated = p_rad

    @property
    def npac_wave(self):
        return self.pac_wave