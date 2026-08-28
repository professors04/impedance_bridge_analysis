
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

from impedance_bridge_analysis.load_data import get_parameter_snapshot_value
from impedance_bridge_analysis.calculation import bridge_calculation

from matplotlib.colors import LogNorm

import numpy as np
import xarray as xr

from scipy.optimize import curve_fit

import os



cwd = os.getcwd()


def size_scale(factor: float, base_size: tuple):
    """Scale a base size by a given factor."""
    return (base_size[0] * factor, base_size[1] * factor)

def linear(x: float, P0: tuple, P1: tuple):
    m = (P1[1] - P0[1]) / (P1[0] - P0[0])
    b = P0[1] - m * P0[0]
    return m * x + b


def varactor_curve(V, C0, Vr, gamma):
    """Model for a varactor diode's capacitance-voltage relationship."""
    return C0 / ((1 + V/Vr) ** gamma)


def prefix(input: float) -> str:
    """Determine the SI prefix for a given value."""

    value = 1/input

    if value == 1e9:
        return "G"
    elif value == 1e6:
        return "M"
    elif value == 1e3:
        return "k"
    elif value == 1:
        return ""
    elif value == 1e-3:
        return "m"
    elif value == 1e-6:
        return "µ"
    elif value == 1e-9:
        return "n"
    else:
        return f"10^{int(np.log10(1/value))} "  # For other values, return in scientific notation



class Plotter:
    def __init__(self, theme: str):
        self.theme = theme
        self._set_style()

    def _set_style(self):
        """Set matplotlib style depending on selected format."""
        
        if self.theme == "thesis":
            fontpath_normal= r"C:\Users\s.schreibing\Documents\Latin-Modern-Sans\lmsans10-regular.otf"
            fm.fontManager.addfont(fontpath_normal)
            prop = fm.FontProperties(fname=fontpath_normal)
            latin_modern_name = prop.get_name()

            fontpath_bold = r"C:\Users\s.schreibing\Documents\Latin-Modern-Sans\lmsans10-bold.otf"
            fm.fontManager.addfont(fontpath_bold)
            prop_bold = fm.FontProperties(fname=fontpath_bold)
            latin_modern_name_bold = prop_bold.get_name()

            fontpath_italic = r"C:\Users\s.schreibing\Documents\Latin-Modern-Sans\lmsans10-oblique.otf"
            fm.fontManager.addfont(fontpath_italic)
            prop_italic = fm.FontProperties(fname=fontpath_italic)
            latin_modern_name_italic = prop_italic.get_name()

            mpl.rcParams.update({
                # Schriftarten & Layout
                "mathtext.fontset": "custom",
                "mathtext.rm": latin_modern_name,
                "mathtext.it": f"{latin_modern_name_italic}:italic",
                "mathtext.bf": f"{latin_modern_name_bold}:bold",
                "text.usetex": False,               
                "font.size": 11,
                "axes.labelsize": 11,
                "axes.titlesize": 11,
                "legend.fontsize": 10,
                "legend.title_fontsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,

                # Linien
                "lines.linewidth": 1.5,
                "lines.markersize": 4,

                # PDF
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            })

        elif self.theme == "presentation":
            mpl.rcParams.update({
                "font.size": 13,
                "axes.titlesize": 18,
                "axes.labelsize": 16,
                "legend.fontsize": 13,
                "lines.linewidth": 2.5,
                "axes.linewidth": 1.5,
                
                "xtick.major.size": 6,
                "ytick.major.size": 6,
                "xtick.major.width": 1.2,
                "ytick.major.width": 1.2,

                "savefig.dpi": 300,
                "text.usetex": False,
            })
        
        elif self.theme == "poster":
            #plt.rcParams['figure.figsize'] = (6.7*1.25,5*1.25)

            plt.rcParams.update({'font.size': 18})
            RWTHblue = (0,103/255,166/255)
            RWTHred = (161/255,16/255,53/255)
            RWTHgreen = (87/255,171/255,39/255)
            RWTHlightblue = (119/255,158/255,201/255)
            plt.rcParams.update({'font.size': 18})
            plt.rcParams['figure.constrained_layout.use'] = True


            mpl.rcParams['axes.linewidth'] = 2
            mpl.rcParams['xtick.major.size'] = 8
            mpl.rcParams['xtick.major.width'] = 2
            mpl.rcParams['xtick.minor.size'] = 8
            mpl.rcParams['xtick.minor.width'] = 2
            mpl.rcParams['ytick.major.size'] = 8
            mpl.rcParams['ytick.major.width'] = 2
            mpl.rcParams['ytick.minor.size'] = 8
            mpl.rcParams['ytick.minor.width'] = 2

        else:
            raise ValueError(f"Unknown style: {self.style}. Choose 'thesis', 'presentation' or 'poster'.")



    ##########################


    def lockin_ref_phase_sweep(self, data: xr.Dataset, balance_point: bool = False, log_scale: bool = False, y_factor = 1e3, size_factor: float = 1, save_path: str = None) -> None:
        
        phase_dif = data["lockin_phase_diff"].values
        lockin_amp_ref = data["lockin_amplitude_ref"].values *y_factor # convert to xV
        
        v_out_p = data["v_out_p"].values
        v_out_r = data["v_out_r"].values *y_factor # convert to xV

        si_prefix = prefix(y_factor)



        size = size_scale(size_factor, base_size=(16/2.54,7/2.54))
        fig, ax = plt.subplots(figsize=size, nrows=1, ncols=2, sharex=True, sharey=True, constrained_layout=True)

        if log_scale:
            c_r = ax[0].pcolormesh(phase_dif, lockin_amp_ref, v_out_r, norm=LogNorm(), shading='auto', cmap='viridis')
        else:
            c_r = ax[0].pcolormesh(phase_dif, lockin_amp_ref, v_out_r, shading='auto', cmap='viridis')
        c_p = ax[1].pcolormesh(phase_dif, lockin_amp_ref, v_out_p, shading='auto', cmap='viridis')

        fig.colorbar(c_r, ax=ax[0], label=r"$v_\text{out amp}$" + f" [{si_prefix}V]")
        fig.colorbar(c_p, ax=ax[1], label=r"$v_\text{out phase}$ [deg]")

        for i in [0,1]:
            ax[i].set_xlabel(r"$\Delta\varphi$ [deg]")
        ax[0].set_ylabel(r"$v_\text{ref amp}$" + f" [{si_prefix}V]")

        if balance_point:
            # red x for the minimum of v_out_r
            i_min, j_min = np.unravel_index(
                np.argmin(v_out_r),
                v_out_r.shape
            )

            x_min = phase_dif[j_min]
            y_min = lockin_amp_ref[i_min]

            ax[0].plot(x_min, y_min, "rx", markersize=12, markeredgewidth=3, label="balance point\n" + f"({x_min:.1f} deg, {y_min:.1f} {si_prefix}V)")

            ax[0].legend(loc="upper right")

            print(f"min v_out_r: {np.min(v_out_r)*1e3:.2f} uV")


        if save_path is not None:
            plt.savefig(cwd + save_path, dpi=300)

        plt.show()

    
    def lockin_frequency_sweep(self, data: xr.Dataset, balance_point: bool = False, size_factor: float = 1, save_path: str = None) -> None:
        
        f = data["lockin_frequency"].values /1e3 # convert to kHz
        
        v_out_p = data["v_out_p"].values
        v_out_r = data["v_out_r"].values *1e3 # convert to mV


        size = size_scale(size_factor, base_size=(16/2.54,7/2.54))
        fig, ax = plt.subplots(figsize=size, nrows=1, ncols=2, sharex=True, constrained_layout=True)

        ax[0].plot(f, v_out_r)
        ax[1].plot(f, v_out_p)

        for i in [0,1]:
            ax[i].grid()
            ax[i].set_xlabel(r"$f_\text{lockin}$ [kHz]")

        ax[0].set_ylabel(r"$v_\text{out amp}$ [mV]")
        ax[1].set_ylabel(r"$v_\text{out phase}$ [deg]")

        if balance_point:
            # red x for the minimum of v_out_r
            i_min = np.argmin(v_out_r)

            x_min = f[i_min]
            y_min = v_out_r[i_min]

            ax[0].plot(x_min, y_min, "rx", markersize=12, markeredgewidth=3, label="balance point\n" + f"({x_min:.2f} kHz, {y_min*1e3:.3f} uV)")

            ax[0].legend(loc="upper right")

        if save_path is not None:
            plt.savefig(cwd + save_path, dpi=300)
        
        plt.show()

    
    def lockin_ref_dut_bias_sweep(self, data: xr.Dataset, size_factor: float = 1, save_path: str = None, R_ref=27e3, fit: bool = False) -> None:
        
        dut_bias = data["V_dut"].values
        lockin_amp_ref = data["lockin_amplitude_ref"].values

        f_lockin = get_parameter_snapshot_value(data, "lockin_frequency")
        lockin_amp_dut = get_parameter_snapshot_value(data, "lockin_amplitude_dut")

        T = get_parameter_snapshot_value(data, "T4k")
        C = []

        for v in dut_bias:
            v_out_r = data["v_out_r"].sel(V_dut=v).values

            #find lockin_amplitude_ref where v_out_r is minimum
            i_min = np.argmin(v_out_r)
            amp_ref_min = lockin_amp_ref[i_min]

            omega = 2*np.pi*f_lockin
            C.append( 1 / (omega * R_ref) * amp_ref_min/lockin_amp_dut)
        

        size = size_scale(size_factor, base_size=(8/2.54,7/2.54))
        fig, ax = plt.subplots(figsize=size, nrows=1, ncols=1, constrained_layout=True)

        ax.scatter(dut_bias, np.array(C)*1e12, color='blue', label='Data points')

        if fit:
            popt, pcov = curve_fit(varactor_curve, dut_bias, np.array(C)*1e12, p0=[200, 15, 0.7])

            x = np.linspace(np.min(dut_bias), np.max(dut_bias), 100)
            y_fit = varactor_curve(x, *popt)

            ax.plot(x, y_fit, color='red', label = (
                rf"Fit: $C(V) = \frac{{{popt[0]:.2f}}}"
                rf"{{\left(1 + \frac{{V}}{{{popt[1]:.2f}~[V]}}\right)^{{{popt[2]:.2f}}}}}$ [pF]"
            ))

        ax.legend(title=f"T = {T:.2f} K", fontsize=16, alignment="left")
        ax.set_xlabel(r"$V_\text{bias}$ [V]")
        ax.set_ylabel(r"$C$ [pF]")

        ax.grid()

        if save_path is not None:
            plt.savefig(cwd + save_path, dpi=300)
        
        plt.show()


    def lockin_ref_phase_dut_bias_3d_sweep(self, data_list: list[xr.Dataset], size_factor: float = 1, save_path: str = None, R_ref=27e3, fit: bool = False, add_sim: str = None) -> None:
        
        calc = bridge_calculation(R_ref=R_ref)

        V_bias = []
        R_list = []
        C_list = []
        
        for data in data_list:
            dut_bias = get_parameter_snapshot_value(data, "V_dut")
            T = get_parameter_snapshot_value(data, "T4k")

            V_bias.append(dut_bias)
            f_lockin = get_parameter_snapshot_value(data, "lockin_frequency")

            Z_dut = calc.calculate_impedance(data)
            R, C = calc.calculate_RC_from_dut_Impedance(Z_dut, f_lockin)

            R_list.append(R)
            C_list.append(C)
        

        size = size_scale(size_factor, base_size=(11/2.54,5/2.54))
        fig, ax = plt.subplots(figsize=size, nrows=1, ncols=2, sharex=True, constrained_layout=True)

        ax[0].scatter(V_bias, np.array(C_list)*1e12, label = "Data points")
        ax[1].scatter(V_bias, np.array(R_list)/1e6)

        if fit:
            popt, pcov = curve_fit(varactor_curve, np.array(V_bias), np.array(C_list)*1e12, p0=[200, 15, 0.7])

            x = np.linspace(np.min(V_bias), np.max(V_bias), 100)
            y_fit = varactor_curve(x, *popt)

            ax[0].plot(x, y_fit, color='red', label = (
                rf"Fit: $C(V) = \frac{{{popt[0]:.2f}}}"
                rf"{{\left(1 + \frac{{V}}{{{popt[1]:.2f}}}\right)^{{{popt[2]:.2f}}}}}$ [pF]"
            ))

        if add_sim is not None:
            data = np.loadtxt(add_sim)

            V_bias = data[:, 0]
            C_tot = data[:, 1]

            ax[0].plot(V_bias, C_tot, "o--",color='green', alpha=0.6,  label = "Simulation")

        ax[0].legend(fontsize=8)

        for i in [0,1]:
            ax[i].set_xlabel(r"$V_\text{bias}$ [V]")
            ax[i].grid()

        ax[0].set_ylabel(r"$C$ [pF]")
        ax[1].set_ylabel(r"$R$ [$M\Omega$]")
        
        #ax[0].set_title("DUT Impedance vs DUT Bias Voltage @ " + f"{f_lockin/1e3:.2f} kHz" + f" @ {T:.2f} K", loc="center", fontsize=12)

        if save_path is not None:
            plt.savefig(cwd + save_path, dpi=300)
        
        plt.show()

    
    def lockin_ref_gate_sweep(self, data: xr.Dataset, gate_label: str, log_scale: bool = False, y_factor = 1e3, size_factor: float = 1, save_path: str = None) -> None:
        
        gate_voltage = data[gate_label].values
        lockin_amp_ref = data["lockin_amplitude_ref"].values *y_factor # convert to xV
        
        v_out_p = data["v_out_p"].transpose("lockin_amplitude_ref", gate_label)
        v_out_r = data["v_out_r"].transpose("lockin_amplitude_ref", gate_label) *y_factor # convert to xV

        si_prefix = prefix(y_factor)



        size = size_scale(size_factor, base_size=(16/2.54,7/2.54))
        fig, ax = plt.subplots(figsize=size, nrows=1, ncols=2, sharex=True, sharey=True, constrained_layout=True)

        if log_scale:
            c_r = ax[0].pcolormesh(gate_voltage, lockin_amp_ref, v_out_r, norm=LogNorm(), shading='auto', cmap='viridis')
        else:
            c_r = ax[0].pcolormesh(gate_voltage, lockin_amp_ref, v_out_r, shading='auto', cmap='viridis')
        c_p = ax[1].pcolormesh(gate_voltage, lockin_amp_ref, v_out_p, shading='auto', cmap='viridis')

        fig.colorbar(c_r, ax=ax[0], label=r"$v_\text{out amp}$" + f" [{si_prefix}V]")
        fig.colorbar(c_p, ax=ax[1], label=r"$v_\text{out phase}$ [deg]")

        for i in [0,1]:
            ax[i].set_xlabel(f"{gate_label} [V]")
        ax[0].set_ylabel(r"$v_\text{ref amp}$" + f" [{si_prefix}V]")


        if save_path is not None:
            plt.savefig(cwd + save_path, dpi=300)

        plt.show()