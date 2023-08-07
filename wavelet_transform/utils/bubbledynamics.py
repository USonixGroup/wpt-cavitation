import numpy as np
from scipy.integrate import ode


class BubbleDynamics(object):
    """
    An object created for bubble dynamics calculations

    Parameters
    ----------
    R_init: int
        the initial bubble radius in meters
    Rdot_init: int
        the initial bubble's wall velocity in m/s
    frequensound_speedy: int
        the fundamental frequensound_speedy of excitation in Hz.
    pac: array
        the sonication time series in Pa.
    time: array
        the time array of the sonication signal in seconds.
    tstep: float
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
        frequensound_speedy,
        pac,
        time,
        tstep,
        medium_parameters={
            "A": 304.0e6,
            "B": 303.9e6,
            "nTaitEq": 7,
            "sound_speed": 1450,
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
        self.frequensound_speedy = frequensound_speedy
        self.pac = pac
        self.time = time
        self.tstep = tstep
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

        # Non-dimensionalise the parameters of the Gilmore ODE
        omega = 2 * np.pi * self.frequensound_speedy
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
        nA = self.medium_parameters["A"] / self.medium_parameters["p0"]
        nB = self.medium_parameters["B"] / self.medium_parameters["p0"]
        nTaitEq = self.medium_parameters["nTaitEq"]
        dtau = omega * self.tstep
        A_RecDiff_coeff = 2 * nsigma / np0
        B_RecDiff_coeff = (
            nomega**2 * nrho / (3 * self.medium_parameters["gamma"] * np0)
        )
        self.nondimensional_params = {
            "nsound_speed": nsound_speed,
            "nrho": nrho,
            "np0": np0,
            "nsigma": nsigma,
            "nR_init": nR_init,
            "nRdot_init": nRdot_init,
            "nmu": nmu,
            "nomega": nomega,
            "npac": npac,
            "tau": tau,
            "nA": nA,
            "nB": nB,
            "nTaitEq": nTaitEq,
        }

        # defining initial conditions and setting up the ODE model
        initial_vals = np.array([nR_init, nRdot_init])
        model_integrator = ode(self.GAmodel).set_integrator(
            "vode", method="BDF", atol=1e-16
        )
        model_integrator.set_initial_value(initial_vals, tau[0]).set_f_params(
            self.nondimensional_params,
            self.medium_parameters["gamma"],
        )

        # allocating arrays for the solution, the first column is nR(t) and the second column is nRdot(t)
        solution = np.ndarray(shape=(len(tau), 2), order="F")
        enthalpy_integral = np.ndarray(shape=(len(tau),))

        idx_int = 0
        while model_integrator.successful() and model_integrator.t < tau[-1]:
            solution[idx_int:] = model_integrator.integrate(model_integrator.t + dtau)
            global nenthalpy_term
            enthalpy_integral[idx_int] = nenthalpy_term
            idx_int = idx_int + 1

        if solution.imag.any():
            raise ValueError("IMAGINARY solution error, check the parameters.....")

        # Calculate rectified diffusion threshold. It is useful to check if inertial cavitation can take place for this set of parameters and R0.
        nomega_res = np.sqrt(
            (1 / nrho / nR_init**2)
            * (
                3 * self.medium_parameters["gamma"] * np0
                + 2 * nsigma / nR_init * (3 * self.medium_parameters["gamma"] - 1)
            )
        )
        damp_factor = 4 * nmu / (nrho * nomega_res * nR_init**2)
        rect_diff_threshold = np.sqrt(
            1.5
            * (A_RecDiff_coeff * np0**2 / (A_RecDiff_coeff + nR_init))
            * (
                (1 - nR_init**2 * B_RecDiff_coeff) ** 2
                + nR_init**2 * B_RecDiff_coeff * damp_factor
            )
        )

        # Calculate the speed of sound at the bubble wall location in the fluid, accounting for the compressibility of the fluid. See the Gilmore model for detail.
        nsound_speed_instantaneous = np.sqrt(
            nsound_speed**2 + (nTaitEq - 1) * enthalpy_integral
        )
        # Determine maximum value of acoustic Mach number. Used to classify bubble dynamics.
        Mach_number = np.abs(solution[:, 1]) / nsound_speed
        speed_ratio = nsound_speed_instantaneous / nsound_speed

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
    def GAmodel(tau, y, nondimensional_params, gamma):
        """
        the Gilmore Akulichev model to be solved in the solver function.

        Parameters
        ----------
        tau: float
            the integration time instant, dimensionless
        y: array, float
            a 2-element array of [R,Rdot] at instant t,
        nondimensional_params: dict
            dictionary of nondimensionalised parameters to be used for building the Gilmore model
        gamma: float
            the constant in the vapour equation of state

        Returns
        -------
        solutions [R,Rdot] after numerical integration at time instant tau
        """

        global nenthalpy_term
        R = y[0]
        Rdot = y[1]
        # Excitation Pressure
        npac_wave = nondimensional_params["npac"] * np.sin(
            nondimensional_params["nomega"] * tau
        )
        nPinf = nondimensional_params["np0"] + npac_wave

        npint_bubble = (
            nondimensional_params["np0"]
            + 2.0 * nondimensional_params["nsigma"] / nondimensional_params["nR_init"]
        ) * (nondimensional_params["nR_init"] / R) ** (3 * gamma)
        p_bubble = (
            npint_bubble
            - 2.0 * nondimensional_params["nsigma"] / R
            - 4.0 * nondimensional_params["nmu"] * Rdot / R
        )
        coeff = (
            nondimensional_params["nTaitEq"]
            * nondimensional_params["nA"] ** (1.0 / nondimensional_params["nTaitEq"])
            / (nondimensional_params["nTaitEq"] - 1.0)
            / nondimensional_params["nrho"]
        )

        # Calculate radiated pressure from oscillating bubble.
        nenthalpy_term = coeff * (
            (p_bubble + nondimensional_params["nB"])
            ** (1.0 - 1.0 / nondimensional_params["nTaitEq"])
            - (nPinf + nondimensional_params["nB"])
            ** (1.0 - 1.0 / nondimensional_params["nTaitEq"])
        )

        # Speed of sound at the wall of bubble
        nsound_speed_at_R = np.sqrt(
            nondimensional_params["nsound_speed"] ** 2
            + (nondimensional_params["nTaitEq"] - 1) * nenthalpy_term
        )
        # calculating dH/dR at the bubble's wall
        dnpint_bubble_dR = -3 * gamma / R * npint_bubble
        dp_bubble_dR = (
            dnpint_bubble_dR
            + 2.0 * nondimensional_params["nsigma"] / R**2
            + 4.0 * nondimensional_params["nmu"] * Rdot / R**2
        )
        dnenthalpy_term_dR = (
            (
                nondimensional_params["nA"] ** (1.0 / nondimensional_params["nTaitEq"])
                / nondimensional_params["nrho"]
            )
            * (p_bubble + nondimensional_params["nB"])
            ** (-1.0 / nondimensional_params["nTaitEq"])
            * dp_bubble_dR
        )

        X1 = Rdot
        X2 = (1.0 / R) * (
            nenthalpy_term * (nsound_speed_at_R + Rdot) / (nsound_speed_at_R - Rdot)
            + dnenthalpy_term_dR * Rdot * R / nsound_speed_at_R
            - 0.5
            * Rdot**2
            * (3.0 * nsound_speed_at_R - Rdot)
            / (nsound_speed_at_R - Rdot)
        )

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
            self.nondimensional_params["nA"]
            * (
                2.0 / (self.nondimensional_params["nTaitEq"] + 1.0)
                + (self.nondimensional_params["nTaitEq"] - 1.0)
                / (self.nondimensional_params["nTaitEq"] + 1.0)
                * np.sqrt(
                    1.0
                    + nG_bubble
                    * (self.nondimensional_params["nTaitEq"] + 1.0)
                    / (nr_measure * self.nondimensional_params["nsound_speed"] ** 2)
                )
            )
            ** (
                2
                * self.nondimensional_params["nTaitEq"]
                / (self.nondimensional_params["nTaitEq"] - 1.0)
            )
            - self.nondimensional_params["nB"]
        )
        self.p_radiated = nP_rad
