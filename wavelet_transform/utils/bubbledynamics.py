import numpy as np
from scipy.integrate import ode


class BubbleDynamics(object):
    """
    An object created for bubble dynamics calculations. The pulse for nonlinear waves is calculated using the foemulation derived by AYME, CARSTENSE, see here for the paper: https://pubmed.ncbi.nlm.nih.gov/2922882/
    The Gilmore implementation can be checked against the results presented in the same paper, too.

    Parameters
    ----------
    R_init: int
        the initial bubble radius in meters
    Rdot_init: int
        the initial bubble's wall velocity in m/s
    frequency: int
        the fundamental frequency of excitation in Hz.
    pac: array
        the sonication time series in Pa.
    time_delay: float
        defines the centre of the pulse
    pulse_width: int
        the width of the sonication pulse, using a Gaussian envelope to pulse the signal.
    phase_shift: float
        constant phase shift added to all harmonics
    phase_initial: float
        phase shift of each harmonic. The actual phase shift of each harmonic is harmonic_number * phase_initial
    no_harmonics: int
        total number of harmonics to reconstruct the distorted wave
    time: array
        the time array of the sonication signal in seconds.
    deltat: float
        the time step (delta t) of the time array of the sonication signal in seconds.
    r_measure: float
        the distansound_speede from the bubble centre to calculate the radiated pressure using Akulichev equation, in m.
    medium_parameters: dict {'A','B','nTaitEq','sound_speed','density','sigma','mu','vapour_pressure','gamma','p0'}
        a dictionary insound_speedluding the physical properties of the bubble's medium. Default is water at room temperature.
        A, B, nTaitEq: constant in the pressure-density state equation of water (adiabatic condition), the equation is p = A(rho/rho_0)**nTaitEq - B , Church used the B =A-1 relationship
        sound_speed:speed of sound in m/s,
        density:density in kg/m^3
        sigma: equilibrium gas/shell surface tension in nTaitEq/m
        mu: medium viscosity in Pa.s
        vapour_pressure: vapour pressure of the gas in bubble in Pa.
        gamma: gas constant in the equation of state for gas in the bubble (dimensionless). The default is adiabatic water vapour constant.
        p0:medium referensound_speede (static) pressure in Pa.

    Returns
    -------
    bubble dynamics_object: an object containing the results from solving the Gilmore model
    """

    def __init__(
        self,
        R_init,
        Rdot_init,
        frequency,
        pac,
        time_delay,
        pulse_width,
        phase_initial,
        phase_shift,
        no_harmonics,
        time,
        deltat,
        medium_parameters={
            "A": 304.0e6,
            "B": 303.9e6,
            "nTaitEq": 7,
            "sound_speed": 1482,
            "density": 1000,
            "sigma": 72.5e-3,
            "mu": 1e-3,
            "vapour_pressure": 3.2718e3,
            "gamma": 1.4,
            "p0": 1e5,
        },
    ):
        self.R_init = R_init
        self.Rdot_init = Rdot_init
        self.frequency = frequency
        self.pac = pac
        self.time = time
        self.deltat = deltat
        self.medium_parameters = medium_parameters
        self.p_radiated = []
        self.Mach_number_acoustic = []
        self.R = []
        self.Rdot = []
        self.R_ratio_max = []
        self.Mach_number_acoustic_max = []
        self.sound_speed_instant_ratio = []
        self.nomega_res = []
        self.rect_diff_threshold = []
        self.enthalpy_integral = []
        self.time_delay = time_delay
        self.pulse_width = pulse_width
        self.phase_initial = phase_initial
        self.phase_shift = phase_shift
        self.no_harmonics = no_harmonics
        self.resonance_frequency = []

    def initialize(self):
        """
        Set up the problem parameters

        Parameters
        ----------
        None

        Returns
        -------
        non-dimensionalize them and calculate the incident acoustic field
        """

        # Non-dimensionalise the parameters of the Gilmore ODE
        omega = 2 * np.pi * self.frequency
        nsound_speed = self.medium_parameters["sound_speed"] / (self.R_init * omega)
        nrho = (
            self.medium_parameters["density"]
            * (self.R_init * omega) ** 2
            / self.medium_parameters["p0"]
        )
        np0 = self.medium_parameters["p0"] / self.medium_parameters["p0"]
        nsigma = self.medium_parameters["sigma"] / (
            self.medium_parameters["p0"] * self.R_init
        )
        nR_init = self.R_init / self.R_init
        nRdot_init = self.Rdot_init / (self.R_init * omega)
        nmu = self.medium_parameters["mu"] * omega / self.medium_parameters["p0"]
        nomega = omega / omega
        npac = self.pac / self.medium_parameters["p0"]
        tau = omega * self.time
        tau_delay = omega * self.time_delay
        npulse_width = omega * self.pulse_width
        nA = self.medium_parameters["A"] / self.medium_parameters["p0"]
        nB = self.medium_parameters["B"] / self.medium_parameters["p0"]
        nTaitEq = self.medium_parameters["nTaitEq"]
        dtau = omega * self.deltat
        A_RecDiff_coeff = 2 * nsigma / np0
        B_RecDiff_coeff = nomega**2 * nrho / (3 * self.medium_parameters["gamma"] * np0)

        # Calculating the linear (first order) resonance frequency of a bubble with R_init.
        # This equation ignores the viscosity and vapour pressure effects. Taken from Eq. 43, Neppiras 1970, Acoustic Cavitation.
        omega_resonance_sqrd = (
            1 / (self.medium_parameters["density"] * self.R_init**2)
        ) * (
            3
            * self.medium_parameters["gamma"]
            * (
                self.medium_parameters["p0"]
                + 2 * self.medium_parameters["sigma"] / self.R_init
            )
            - 2 * self.medium_parameters["sigma"] / self.R_init
        )
        omega_resonance = np.sqrt(omega_resonance_sqrd)
        nomega_resonance = omega_resonance / omega
        self.resonance_frequency = omega_resonance / 2 / np.pi

        npac_wave = np.zeros(tau.shape)
        for i in range(self.no_harmonics):
            i = (
                i + 1
            )  # to get rid off the error of divided by 0 at i=0 which is the case when no_harmincs = 1
            npac_tmp = (
                np.sin(i * nomega * (tau - self.phase_initial) + self.phase_shift) / i
            )
            npac_wave = npac_wave + npac_tmp
        self.npac_wave = (
            npac_wave * npac * np.exp(-(((tau - tau_delay) / npulse_width) ** 2))
        )

        self.ND_params = {
            "nsound_speed": nsound_speed,
            "nrho": nrho,
            "np0": np0,
            "nsigma": nsigma,
            "nR_init": nR_init,
            "nRdot_init": nRdot_init,
            "nmu": nmu,
            "nomega": nomega,
            "nomega_resonance": nomega_resonance,
            "npac": npac,
            "tau": tau,
            "nA": nA,
            "nB": nB,
            "dtau": dtau,
            "A_RecDiff_coeff": A_RecDiff_coeff,
            "B_RecDiff_coeff": B_RecDiff_coeff,
            "nTaitEq": nTaitEq,
            "tau_delay": tau_delay,
            "npulse_width": npulse_width,
            "no_harmonics": self.no_harmonics,
            "phase_initial": self.phase_initial,
            "phase_shift": self.phase_shift,
        }

    def add_pac_wave(
        self,
        frequency,
        pac,
        time_delay,
        pulse_width,
        phase_initial,
        phase_shift,
        no_harmonics,
    ):
        """
        Add an additional sine wave pulse to the input pressure for the model.

        Parameters
        ----------


        """
        omega = 2 * np.pi * frequency
        tau = omega * self.time
        tau_delay = omega * time_delay
        npulse_width = omega * pulse_width
        npac = pac / self.medium_parameters["p0"]

        npac_wave = np.zeros(tau.shape)
        for i in range(1, no_harmonics + 1):
            npac_tmp = np.sin(i * (tau - phase_initial) + phase_shift) / i
            npac_wave = npac_wave + npac_tmp
        npac_wave = (
            npac_wave * npac * np.exp(-(((tau - tau_delay) / npulse_width) ** 2))
        )
        self.npac_wave += npac_wave

    def solver(self):
        """
        Solving a spherical bubble dynamics and determines the radiated pressure in an unbounded medium using Gilmore Akulichev model.

        Parameters
        ----------
        None

        Returns
        -------
        solutions as attributes of the bubble dynamics object
        """

        self.initialize()

        # defining initial conditions and setting up the ODE model
        initial_vals = np.array(
            [self.ND_params["nR_init"], self.ND_params["nRdot_init"]]
        )
        model_integrator = ode(self.GAmodel).set_integrator(
            "vode", method="BDF", atol=1e-15
        )
        model_integrator.set_initial_value(
            initial_vals, self.ND_params["tau"][0]
        ).set_f_params(
            self.ND_params,
            self.medium_parameters["gamma"],
        )

        # allocating arrays for the solution, the first column is nR(t) and the second column is nRdot(t)
        solution = np.ndarray(shape=(len(self.ND_params["tau"]), 2), order="F")
        enthalpy_integral = np.ndarray(shape=(len(self.ND_params["tau"]),))

        idx_int = 0
        while (
            model_integrator.successful()
            and model_integrator.t < self.ND_params["tau"][-1]
        ):
            solution[idx_int:] = model_integrator.integrate(
                model_integrator.t + self.ND_params["dtau"]
            )
            global enthalpy
            enthalpy_integral[idx_int] = enthalpy
            idx_int = idx_int + 1

        if solution.imag.any():
            raise ValueError("IMAGINARY solution error, check the parameters.....")

        # Calculate rectified diffusion threshold. It is useful to check if inertial cavitation can take place for this set of parameters and R0.
        nomega_res = np.sqrt(
            (1 / self.ND_params["nrho"] / self.ND_params["nR_init"] ** 2)
            * (
                3 * self.medium_parameters["gamma"] * self.ND_params["np0"]
                + 2
                * self.ND_params["nsigma"]
                / self.ND_params["nR_init"]
                * (3 * self.medium_parameters["gamma"] - 1)
            )
        )
        damp_factor = (
            4
            * self.ND_params["nmu"]
            / (self.ND_params["nrho"] * nomega_res * self.ND_params["nR_init"] ** 2)
        )
        rect_diff_threshold = np.sqrt(
            1.5
            * (
                self.ND_params["A_RecDiff_coeff"]
                * self.ND_params["np0"] ** 2
                / (self.ND_params["A_RecDiff_coeff"] + self.ND_params["nR_init"])
            )
            * (
                (1 - self.ND_params["nR_init"] ** 2 * self.ND_params["B_RecDiff_coeff"])
                ** 2
                + self.ND_params["nR_init"] ** 2
                * self.ND_params["B_RecDiff_coeff"]
                * damp_factor
            )
        )

        # Calculate the speed of sound at the bubble wall location in the fluid, accounting for the compressibility of the fluid. See the Gilmore model for detail.
        nsound_speed_instantaneous = np.sqrt(
            self.ND_params["nsound_speed"] ** 2
            + (self.ND_params["nTaitEq"] - 1) * enthalpy_integral
        )
        # Determine maximum value of acoustic Mach number. Used to classify bubble dynamics.
        Mach_number = np.abs(solution[:, 1]) / self.ND_params["nsound_speed"]
        speed_ratio = nsound_speed_instantaneous / self.ND_params["nsound_speed"]

        self.R = solution[:, 0]
        self.Rdot = solution[:, 1]
        self.R_ratio_max = self.R.max()
        self.Mach_number_acoustic = Mach_number
        self.Mach_number_acoustic_max = Mach_number.max()
        self.sound_speed_instant_ratio = speed_ratio
        self.nomega_res = nomega_res
        self.rect_diff_threshold = rect_diff_threshold
        self.enthalpy_integral = enthalpy_integral

    @staticmethod
    def GAmodel(tau, y, ND_params, gamma):
        """
        the Gilmore Akulichev model to be solved in the solver function.

        Parameters
        ----------
        tau: float
            the integration time instant, dimensionless
        y: array, float
            a 2-element array of [R,Rdot] at instant t,
        ND_params: dict
            dictionary of nondimensionalised parameters to be used for building the Gilmore model
        gamma: float
            the constant in the vapour equation of state

        Returns
        -------
        solutions [R,Rdot] after numerical integration at time instant tau
        """

        global enthalpy
        R = y[0]
        Rdot = y[1]
        # Excitation Pressure
        p0 = ND_params["np0"]
        sigma = ND_params["nsigma"]
        mu = ND_params["nmu"]
        rho = ND_params["nrho"]
        A = ND_params["nA"]
        B = ND_params["nB"]
        nT = ND_params["nTaitEq"]

        npac_wave = 0
        npac_wave_dt = 0
        for i in range(ND_params["no_harmonics"]):
            i = (
                i + 1
            )  # to get rid off the error of divided by 0 at i=0 which is the case when no_harmincs = 1
            npac_tmp = (
                np.sin(
                    i * ND_params["nomega"] * (tau - ND_params["phase_initial"])
                    + ND_params["phase_shift"]
                )
                / i
            )
            npac_wave = npac_wave + npac_tmp

            npac_tmp_dt = ND_params["nomega"] * np.cos(
                i * ND_params["nomega"] * (tau - ND_params["phase_initial"])
                + ND_params["phase_shift"]
            )
            npac_wave_dt = npac_wave_dt + npac_tmp_dt

        pulse_window = np.exp(
            -(((tau - ND_params["tau_delay"]) / ND_params["npulse_width"]) ** 2)
        )

        npac_wave = ND_params["npac"] * pulse_window * npac_wave
        nPinf = p0 + npac_wave

        nPinf_dt = (
            ND_params["npac"]
            * pulse_window
            * (
                npac_wave_dt
                + npac_wave
                * (-2 * (tau - ND_params["tau_delay"]) / ND_params["npulse_width"] ** 2)
            )
        )

        npint_bubble = (p0 + 2.0 * sigma / ND_params["nR_init"]) * (
            ND_params["nR_init"] / R
        ) ** (3 * gamma)

        p_bubble = npint_bubble - 2.0 * sigma / R - 4.0 * mu * Rdot / R

        coeff_enthalpy = nT * A ** (1.0 / nT) / (nT - 1.0) / rho

        # Calculate radiated pressure from oscillating bubble.
        enthalpy = coeff_enthalpy * (
            (p_bubble + B) ** (1.0 - 1.0 / nT) - (nPinf + B) ** (1.0 - 1.0 / nT)
        )

        # Speed of sound at the wall of bubble
        nsound_speed_at_R = np.sqrt(
            ND_params["nsound_speed"] ** 2 + (nT - 1) * enthalpy
        )

        # setting up the linear system of ODEs
        X1 = Rdot
        # calculating dH/dt at the bubble's wall
        dnpint_bubble_dR = (-3 * gamma) * npint_bubble
        enthalpy_dt_term1 = (
            (A ** (1.0 / nT) / rho)
            * (X1 / R)
            * (p_bubble + B) ** (-1.0 / nT)
            * (dnpint_bubble_dR + 2 * sigma / R + 4 * mu * X1 / R)
        )
        enthalpy_dt_term2 = (
            (-1 * A ** (1.0 / nT) / rho) * (nPinf + B) ** (-1.0 / nT) * nPinf_dt
        )

        coeff_of_eta_term1 = (
            (A ** (1.0 / nT) / nsound_speed_at_R / rho)
            * (4 * mu / R)
            * (p_bubble + B) ** (-1.0 / nT)
        )

        coeff_eta = -1 * (1 + coeff_of_eta_term1) ** (-1)
        eta_term1 = -1 * enthalpy_dt_term1 / nsound_speed_at_R
        eta_term2 = -1 * enthalpy_dt_term2 / nsound_speed_at_R
        eta_term3 = (
            (3 * nsound_speed_at_R - X1) * X1**2 / (nsound_speed_at_R - X1) / 2 / R
        )
        eta_term4 = (
            -1 * (nsound_speed_at_R + X1) * enthalpy / (nsound_speed_at_R - X1) / R
        )
        X2 = coeff_eta * (eta_term1 + eta_term2 + eta_term3 + eta_term4)

        return [X1, X2]

    def calculate_p_radiated(self, r_measure):
        """
        Calculating the radiated pressure from oscillating bubble using Akulichev model. See Akulichev 1971 chapter (high intensity ultrasound fields book, Rozenberg 1971),
        or the following paper: Bailey, 2011: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3259669/

        Parameters
        ----------
        r_measure: float
            distance r from the bubble centre to calculate the radiated pressure (acoustic emission), in meters.

        Returns
        -------
        solutions as attributes of the bubble dynamics object
        """

        nr_measure = r_measure / self.R_init
        # nG_bubble = self.Rdot * (
        #     self.R + 0.5 * self.Rdot**2
        # )

        nG_bubble = self.R * (self.enthalpy_integral + 0.5 * self.Rdot**2)

        nP_rad = (
            self.ND_params["nA"]
            * (
                2.0 / (self.ND_params["nTaitEq"] + 1.0)
                + (self.ND_params["nTaitEq"] - 1.0)
                / (self.ND_params["nTaitEq"] + 1.0)
                * np.sqrt(
                    1.0
                    + nG_bubble
                    * (self.ND_params["nTaitEq"] + 1.0)
                    / (nr_measure * self.ND_params["nsound_speed"] ** 2)
                )
            )
            ** (2 * self.ND_params["nTaitEq"] / (self.ND_params["nTaitEq"] - 1.0))
            - self.ND_params["nB"]
        )

        self.p_radiated = nP_rad

    def calculate_p_radiated_vokurka(self, r_measure):
        """
        Calculating the radiated pressure from oscillating bubble using the model derived by Vokurka. Refer to Tim Leighton, The Acoustic Bubbles, chapter 3, eqs(3.114, 3.115), page 151.

        Parameters
        ----------
        r_measure: float
            distance r from the bubble centre to calculate the radiated pressure (acoustic emission), in meters.

        Returns
        -------
        solutions as attributes of the bubble dynamics object
        """

        nr_measure = r_measure / self.R_init
        _, self.Rddot = self.GAmodel(
            self.ND_params["tau"],
            [self.R, self.Rdot],
            self.ND_params,
            self.medium_parameters["gamma"],
        )

        nP_rad = (self.ND_params["nrho"] * self.R / nr_measure) * (
            self.R * self.Rddot + 2 * self.Rdot**2
        ) - (self.ND_params["nrho"] * self.Rdot**2 / 2) * (self.R / nr_measure) ** 4

        self.p_radiated_vokurka = nP_rad + self.ND_params["np0"]
