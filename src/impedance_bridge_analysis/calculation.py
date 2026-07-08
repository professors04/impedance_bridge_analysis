
import scipy
import xarray as xr
import numpy as np

from .load_data import get_parameter_snapshot_value


#calculations based on: https://pubs.aip.org/aip/rsi/article/90/8/084706/360200/Integrated-impedance-bridge-for-absolute


class bridge_calculation():

    def __init__(self, R_ref):
        self.R_ref = R_ref

    def calculate_impedance(self, data: xr.Dataset) -> complex:
        """
        Calculate the impedance of the device under test (DUT) based on the reference resistor and measured voltages. at bridge point!

        Args:
            data (xr.Dataset): The input dataset from a phase and lockin ref amplitude sweep containing the balance point

        returns:
            complex: The calculated impedance of the DUT.    
        
        """

        phase_dif = data["lockin_phase_diff"].values
        lockin_amp_ref = data["lockin_amplitude_ref"].values
        
        v_out_r = data["v_out_r"].values

        i_min, j_min = np.unravel_index(
                np.argmin(v_out_r),
                v_out_r.shape
            )

        x_min = phase_dif[j_min]
        y_min = lockin_amp_ref[i_min]

        delta_phi = x_min
        V_amp_ref = y_min
        V_amp_dut = get_parameter_snapshot_value(data, "lockin_amplitude_dut")

        Z_dut = -self.R_ref * (V_amp_dut / V_amp_ref) * np.exp(1j * np.deg2rad(delta_phi))

        return Z_dut


    def calculate_capacitance(self, data: xr.Dataset) -> float:
        """
        # expects data of a lockin_frequency sweep

        Args:
            data (xr.Dataset): The input dataset from a frequency sweep at balance point voltages.

        Returns:
            float: The calculated capacitance value.
        """

        f = data["lockin_frequency"].values
        v_out_r = data["v_out_r"].values

        #find output voltage minimum
        i_min = np.argmin(v_out_r)
        f_min = f[i_min]

        #print(f"minimum output voltage frequency: {f_min} Hz")

        V_amp_ref = get_parameter_snapshot_value(data, "lockin_amplitude_ref")
        V_amp_dut = get_parameter_snapshot_value(data, "lockin_amplitude_dut")

        C = 1 / (2*np.pi*f_min * self.R_ref) * V_amp_ref/V_amp_dut  # capacity in F

        return C

    def calculate_RC_from_dut_Impedance(self, Z_dut: complex, f: float) -> tuple:
        """
        Calculate the resistance and capacitance of the device under test (DUT) from its impedance.
        The model here is a prallel R and C !

        Args:
            Z_dut (complex): The impedance of the DUT.
            f (float): The frequency at which the impedance was measured. (can also be a numpy array, but must be same shape as Z_dut)

        Returns:
            tuple: A tuple containing the resistance (R) and capacitance (C) of the DUT.
        """

        Z_dut = np.asarray(Z_dut)
        omega = 2 * np.pi * f

        a = np.real(Z_dut)
        b = np.imag(Z_dut)

        R = a * (1 + (b/a)**2)
        C = - b/(a**2 * omega * (1 + (b/a)**2))

        return R, C
