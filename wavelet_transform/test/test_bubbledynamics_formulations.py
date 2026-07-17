import unittest
import numpy as np

from wavelet_transform.utils.statistical_features import kurtosis
from wavelet_transform.utils.bubbledynamics import BubbleDynamics as BD


def make_time(n=10, frequency=1e6):
    # short time vector for fast unit tests
    t = np.linspace(0.0, 1.0 / frequency, n)
    dt = t[1] - t[0]
    return t, dt


class TestBubbleDynamicsFormulations(unittest.TestCase):
    def test_old_bubbledynamics_calculate_p_radiated_and_vokurka(self):
        # The older bubbledynamics module: run the full solver via subprocess
        # and verify it completes without crashing. Keep grid short so test is fast.
        pac = 1e6  # the amplitude of the sine wave in Pa
        f0_frequency = 1.1* 1e6  # insonation frequency (Hz)
        period = 1 / f0_frequency # the period of the acoustic wave at the fundamental frequency 
        no_cycle = 10  # no. of cycles of the sine wave
        fs = f0_frequency * 1e3  # sampling rate to generate the time vector

        time_delay = 0#4 * period  # time delay, t1
        pulse_width = 10 * no_cycle * period  # pulse width, W = 1 / Q
        phase_initial = 0#-0.325 * period  # tau
        phase_shift = 0#np.pi / 4  # phi
        no_harmonics = 1   # k


        deltat = 1.0 / fs
        tmax = no_cycle * period
        time = np.arange(
            0.0, tmax, deltat
        )  # time vector for computing the sine wave, in seconds


        # R_init = 0.523e-6  # initial bubble radius in meters
        R_init = 30e-6  # initial bubble radius in meters
        Rdot_init = 0  # initial bubble's wall velocity in m/s

        r_measure = 1e-4 # distance from the centre of bubble which radiated pressure (acoustic emission from bubble) is calculated over time

        # Create and initialize an instance locally to ensure initialization works
        bd = BD(
            R_init=R_init,
            Rdot_init = Rdot_init,
            frequency = f0_frequency,
            pac = pac,
            time_delay = time_delay,
            pulse_width = pulse_width,
            phase_initial = phase_initial,
            phase_shift = phase_shift,
            no_harmonics = no_harmonics,
            time = time,
            deltat = deltat,
        )
        bd.initialize()
        # Run solver to populate R, Rdot and enthalpy_integral
        try:
            bd.solver()
        except Exception as e:
            self.fail(f"BubbleDynamics.solver() raised an exception: {e}")

        # Akulichev (calculate_p_radiated uses R, Rdot and enthalpy from solver)
        bd.calculate_p_radiated(r_measure)
        p_akul = np.asarray(bd.p_radiated)
        self.assertFalse(np.isnan(p_akul).any(), msg="Akulichev returned NaNs")

        # Vokurka (uses GAmodel on bd.R, bd.Rdot to compute Rddot)
        bd.calculate_p_radiated_vokurka(r_measure)
        p_vok = np.asarray(bd.p_radiated_vokurka)
        self.assertFalse(np.isnan(p_vok).any(), msg="Vokurka returned NaNs")

        aku_k = kurtosis(p_akul)
        vok_k = kurtosis(p_vok)

        # Threshold chosen to detect numerical blow-ups (very conservative)
        KURTOSIS_THRESHOLD = 100
        self.assertTrue(np.isfinite(aku_k), msg=f"Akulichev kurtosis not finite: {aku_k}")
        self.assertTrue(np.isfinite(vok_k), msg=f"Vokurka kurtosis not finite: {vok_k}")
        self.assertLess(aku_k, KURTOSIS_THRESHOLD, msg=f"Akulichev kurtosis too large: {aku_k}")
        self.assertLess(vok_k, KURTOSIS_THRESHOLD, msg=f"Vokurka kurtosis too large: {vok_k}")

        # basic check that ND grid was created correctly for the locally created bd
        n = time.shape[0]
        self.assertEqual(bd.ND_params["tau"].shape[0], n)

