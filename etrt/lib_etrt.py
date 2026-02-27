from IPython.display import display, Math
import numpy as np
from scipy.special import exp1
import scipy
from discretize import CylindricalMesh
from simpeg import maps
from simpeg.electromagnetics.static import resistivity as dc
from simpeg import SolverLU as Solver
from simpeg import utils
from simpeg.electromagnetics.static.utils import (
    apparent_resistivity_from_voltage, geometric_factor
)
import torch
from torch.special import bessel_y0, bessel_j0, bessel_j1, bessel_y1
from typing import Union, Tuple

def print_matrix(array, pretext=''):
    if array.dtype == 'int64':
        fstring = ' %d '
    else:
        fstring = ' %.3f '

    data = ''
    for line in array:
        if not hasattr(line, '__len__'):
            data += fstring%line + r' \\'
            continue
        if len(line) == 1:
            data += fstring%line + r'& \\\n'
            continue
        for element in line:
            data += fstring%element + r'&'
        data = data[:-1] + r'\\' + '\n'
        data = data[:-1] + r'\\' + '\n'
    display(Math(pretext+'\\begin{bmatrix} \n%s\\end{bmatrix}'%data))

def _ILS(r, t, q, k, crho):
    a = k / (1e6*crho)
    u = r ** 2 / 4 / a / t
    dt = q / 4 / np.pi / k * exp1(u)
    return np.squeeze(dt)

def ILS(
    r: Union[float, np.ndarray],
    t: Union[float, np.ndarray],
    q: float,
    k: float,
    c: float,
    getJ: bool = False,
    eps: float = 1e-8
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute the temperature change in a borehole using the infinite
    line-source model.

    The solution is given by:

        ΔT(r, t) = (q / 4πk) × E₁(u)

    where:
        u = r² / (4αt)
        α = k / c = thermal diffusivity [m²/s]
        E₁ = exponential integral function

    :param r:      Distance to the borehole [m], shape: (nr,)
    :param t:      Time since injection start [s], shape: (nt,)
    :param q:      Injection rate per unit length [W/m]
    :param k:      Thermal conductivity [W/m/K]
    :param c:      Volumetric heat capacity [MJ/m³/K]
    :param getJ:   If True, also compute sensitivities w.r.t k, c, q
    :param eps:    Relative perturbation for finite differences

    :return:
        ΔT :       Temperature change [K], shape: (nt, nr) or squeezed if scalar
        J  :       (if getJ=True) Jacobian w.r.t k, c, q, shape (nt*nr, 3)
    """
    r = np.asarray(r).reshape(1, -1)
    t = np.asarray(t).reshape(-1, 1)
    dt = _ILS(r, t, q, k, c)
    if getJ:
        J = np.zeros((len(t)*len(r), 3), dtype=np.double)
        J[:, 0] = ((_ILS(r, t, q, k * (1 + eps), c) - dt) / (eps * k)).ravel()
        J[:, 1] = ((_ILS(r, t, q, k, c * (1 + eps)) - dt) / (eps * c)).ravel()
        J[:, 2] = ((_ILS(r, t, q * (1 + eps), k, c) - dt) / (eps * q)).ravel()
        return dt, J
    else:
        return dt

def dt2sigma(
    dt: Union[float, np.ndarray],
    m: float,
    sigma25: float,
    sigma0: float = None
) -> Union[float, np.ndarray]:
    """
    Compute the electrical conductivity from a temperature change.

    The relationship is given by:

        σ(T) = σ₀ + m × σ₂₅ × ΔT

    where:
        σ(T)  = electrical conductivity at temperature T [S/m]
        σ₀    = reference conductivity [S/m] (defaults to σ₂₅)
        σ₂₅   = electrical conductivity at 25°C [S/m]
        m     = temperature coefficient of conductivity [1/K]
        ΔT    = temperature change from reference [K]

    :param dt:      Temperature change [K], shape: (n,)
    :param m:       Temperature coefficient of conductivity [1/K]
    :param sigma25: Electrical conductivity at 25°C [S/m]
    :param sigma0:  Reference conductivity [S/m], defaults to sigma25

    :return: Electrical conductivity [S/m], shape: (n,)
    """
    if sigma0 is None:
        sigma0 = sigma25
    return m*dt*sigma25 + sigma0

def ert_setup(
    zrec: np.ndarray,
    mesh_selection: Union[str, tuple] = "fast"
):
    """
    Initialize the ERT modeling for pole-pole borehole measurements with
    an axi-symmetric cylindrical grid. Electrodes are placed at r=0.

    :param zrec:           Position of the M electrode w.r.t A at z=0 [m],
                           shape: (nrec,)
    :param mesh_selection: Selection of the mesh quality. Either "fast" or
                           "accurate", or a tuple (hr, hz) for custom grid

    :return:
        mesh       : SimPEG Cylindrical mesh
        simulation : SimPEG simulation object
        survey     : SimPEG survey object
    """

    if isinstance(mesh_selection, (list, tuple)):
        hr, hz = mesh_selection
    elif mesh_selection == "accurate":
        # Create a 2D cylindrical mesh
        hr = [(0.00001, 40, 1.01),  (0.0001, 80, 1.01), (0.001, 160, 1.01),
              (0.01, 180, 1.01), (0.04, 80, 1.12)]
        hz = [ (0.02, 80, -1.12), (0.01, 180, -1.01), (0.001, 80, -1.01),
               (0.0001, 40, -1.01), (0.00001, 40, -1.01), (0.00001, 40, 1.01),
               (0.0001, 40, 1.01), (0.001, 80, 1.01),  (0.01, 180, 1.01),
               (0.02, 80, 1.12)]
    elif mesh_selection == "fast":
        # Faster, for testing
        hr = [(0.0001, 200, 1.05), (1.2, 40, 1.12)]
        hz = [(1.2, 40, -1.12), (0.0001, 200, -1.05), (0.0001, 200, 1.05),
              (1.2, 40, 1.12)]
    else:
        raise ValueError("mesh_selection must be 'fast', 'accurate' "
                         "or a tuple of (hr, hz)")
    mesh = CylindricalMesh([hr, 1, hz], x0="00C")

    # Create the survey
    xyz_rxM = utils.ndgrid(np.r_[0.0], np.r_[0.0], zrec)
    receivers = dc.receivers.Pole(xyz_rxM)
    sources = dc.sources.Pole([receivers], np.array([0.0, 0.0, 0.0]))
    survey = dc.Survey([sources], survey_geometry="borehole")
    map = maps.IdentityMap(mesh)
    simulation = dc.simulation.Simulation3DCellCentered(
        mesh, survey=survey, sigmaMap=map, solver=Solver, bc_type='Dirichlet')

    return mesh, simulation, survey

def simulate_ert(
    k: float,
    crho: float,
    q: float,
    m: float,
    sigma25: float,
    sigma_water: Union[float, np.ndarray],
    rbh: float,
    times: np.ndarray,
    mesh,
    survey,
    simulation,
    getJ: bool = False,
    thermal_model: str = 'ILS'
) -> Union[np.ndarray,Tuple[np.ndarray, np.ndarray]]:
    """
    Simulate the ERT response during a TRT for a set of times after the start
    of a constant rate injection in a borehole. The temperature change is
    computed with the infinite line-source or cylindrical-source model.

    :param k:             Thermal conductivity of the rock [W/m/K]
    :param crho:          Volumetric heat capacity of the rock [MJ/m³/K]
    :param q:             Injection rate per unit length [W/m]
    :param m:             Temperature-conductivity coefficient [1/K]
    :param sigma25:       Electrical conductivity at 25°C [S/m]
    :param sigma_water:   Electrical conductivity of borehole water [S/m],
                          shape: (nt,) or scalar
    :param rbh:           Borehole radius [m]
    :param times:         Times after injection start [h], shape: (nt,)
    :param mesh:          SimPEG cylindrical mesh
    :param survey:        SimPEG survey object
    :param simulation:    SimPEG simulation object
    :param getJ:          If True, compute sensitivities w.r.t k, c, q, m
    :param thermal_model: Thermal model to use: 'ILS' or 'ICS'

    :return:
        rhoas : (nt, nrec) Apparent resistivity at each time and receiver
        J     : (nt*nrec, 4) Sensitivities w.r.t k, c, q, m (if getJ=True)
    """
    r = mesh.cell_centers_x
    rhoas = np.zeros((len(times), survey.nD))

    if sigma_water is not None:
        if np.isscalar(sigma_water):
            sigma_water = sigma_water * np.ones(len(times))
        elif len(sigma_water) != len(times):
            raise ValueError("sigma_water must be a scalar or have the same "
                             "length as times")
    if thermal_model == 'ICS':
        thermal_fun = lambda r, t, q, k, c: ICS(r, t, q, k, c, rbh).ravel()
    elif thermal_model == 'ILS':
        thermal_fun = lambda r, t, q, k, c: ILS(r, t, q, k, c)
    else:
        raise ValueError("thermal_model must be 'ILS' or 'ICS'")

    if getJ:
        J = np.zeros((len(times)*survey.nD, 4), dtype=np.double)
        eps=1e-6
        m0 = np.r_[k, crho, q, m]
        K = geometric_factor(survey, space_type="whole-space")

    sigma0 = np.ones(len(r)) * sigma25

    for ii, t in enumerate(times):
        dt = thermal_fun(r, t *3600, q, k, crho)

        sigma = dt2sigma(dt, m, sigma25)

        if sigma_water is not None:
            sigma[r<rbh] = sigma_water[ii]
        else:
            sigma[r<rbh] = sigma[r>=rbh][0]
        sigma3D = np.tile(sigma, mesh.vnC[-1])
        f = simulation.fields(sigma3D)
        dpred = simulation.dpred(sigma3D, f=f)
        rhoas[ii, :] = apparent_resistivity_from_voltage(survey, dpred,
                                                         space_type="whole-space")
        if getJ:
            Jsim = simulation.getJ(sigma3D, f=f)
            for el in range(4):
                mp = m0.copy()
                mp[el] *= (1 + eps)
                if el < 3:
                    Jdt = ((thermal_fun(r, t * 3600, mp[2], mp[0], mp[1]) - dt)
                           / (eps * m0[el]))
                    Jsigma = m * sigma25 * Jdt
                else:
                    Jsigma = sigma25 * dt
                Jsigma3D = np.tile(Jsigma, mesh.vnC[-1])
                Jdpred = Jsim @ Jsigma3D
                Jrhoa = np.sign(dpred) * (1.0 / (K + 1e-10)) * Jdpred
                J[ii*survey.nD:(ii+1)*survey.nD, el] = Jrhoa
    if getJ:
        return rhoas, J
    else:
        return rhoas


def simulate_etrt(
    k: float,
    crho: float,
    q: float,
    m: float,
    sigma25: float,
    rbh: float,
    times_ert: np.ndarray,
    times_trt: np.ndarray,
    mesh,
    survey,
    simulation,
    getJ: bool = False,
    type: str = 'diff',
    thermal_model: str = 'ICS',
    sigma_water: Union[float, np.ndarray] = None,
    param_q: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Simulate the combined ERT and TRT response during a constant rate
    injection in a borehole. The temperature change is computed with the
    infinite line-source or cylindrical-source model. The injection rate q
    is included in the output to propagate uncertainties.

    :param k:             Thermal conductivity of the rock [W/m/K]
    :param crho:          Volumetric heat capacity of the rock [MJ/m³/K]
    :param q:             Injection rate per unit length [W/m]
    :param m:             Temperature-conductivity coefficient [1/K]
    :param sigma25:       Electrical conductivity at 25°C [S/m]
    :param rbh:           Borehole radius [m]
    :param times_ert:     ERT measurement times after injection start [h],
                          shape: (nt_ert,)
    :param times_trt:     TRT measurement times after injection start [h],
                          shape: (nt_trt,)
    :param mesh:          SimPEG cylindrical mesh
    :param survey:        SimPEG survey object
    :param simulation:    SimPEG simulation object
    :param getJ:          If True, compute the Jacobian w.r.t k, c, m
    :param thermal_model: Thermal model to use: 'ILS' or 'ICS'
    :param sigma_water:   Electrical conductivity of borehole water [S/m],
                          If None, uses conductivity at borehole edge
    :param param_q:      If True, include q in the output parameters and in the data vector

    :return:
        d: (nrec*nt_ert + nt_trt + 1,) Combined ERT and TRT response with q
        J: (nrec*nt_ert + nt_trt + 1, 3) Jacobian of k, c, m (if getJ=True )
    """

    if type == 'diff':
        if times_ert[0] != 0:
            times_ert = np.hstack([0.0, times_ert])

    outERT = simulate_ert(k, crho, q, m, sigma25, sigma_water, rbh,
                          times_ert, mesh, survey, simulation, getJ=getJ,
                          thermal_model=thermal_model)

    if type == 'diff':
        if getJ:
            rhoas, J_ert = outERT
            rhoas_diff =  rhoas.reshape(len(times_ert), survey.nD)[1:] - rhoas.reshape(len(times_ert), survey.nD)[0]
            J_ert = J_ert.reshape(len(times_ert), survey.nD, -1)
            J_ert_drhoa = J_ert[1:, :, :] - J_ert[0, :, :][np.newaxis, :, :]
            J_ert_drhoa = J_ert_drhoa.reshape(len(times_ert) - 1, survey.nD, -1).transpose(0, 1, 2).reshape((len(times_ert) - 1) * survey.nD, 4)
            outERT = rhoas_diff.ravel(), J_ert_drhoa
        else:
            rhoas = outERT
            rhoas_diff = rhoas.reshape(len(times_ert),survey.nD)[1:] - rhoas.reshape(len(times_ert), survey.nD)[0]
            outERT = rhoas_diff.ravel()

    if thermal_model == 'ILS':
        outTRT = ILS(rbh, times_trt * 3600, q, k, crho, getJ=getJ)
    elif thermal_model == 'ICS':
        outTRT = ICS(rbh, times_trt * 3600, q, k, crho, rbh, getJ=getJ)

    if param_q:
        if getJ:
            rhoas, J_ert = outERT
            dt, J_trt = outTRT
            J_trt = np.hstack([J_trt, np.zeros((J_trt.shape[0], 1))])
            J_q = np.zeros((1, 4), dtype=np.double)
            J_q[0, 2] = 1.0
            J = np.vstack([J_ert, J_trt, J_q])
            return np.r_[rhoas.ravel(), dt.ravel(), q], J
        else:
            return np.r_[outERT.ravel(), outTRT.ravel(), q]
    else:
        if getJ:
            rhoas, J_ert = outERT
            dt, J_trt = outTRT
            J_trt = np.hstack([J_trt, np.zeros((J_trt.shape[0], 1))])
            J = np.vstack([J_ert, J_trt])
            J = np.delete(J, 2, axis=1)
            return np.r_[rhoas.ravel(), dt.ravel()], J
        else:
            return np.r_[outERT.ravel(), outTRT.ravel()]


def jacobian_ert_fd(
    k: float,
    crho: float,
    q: float,
    m: float,
    sigma25: float,
    sigma_water: Union[float, np.ndarray],
    rbh: float,
    times: np.ndarray,
    mesh,
    survey,
    simulation,
    type: str = 'abs',
    eps: float = 1e-6
) -> np.ndarray:
    """
    Compute the sensitivities of the ERT response w.r.t the parameters
    k, c, q, m using finite differences.

    :param k:          Thermal conductivity of the rock [W/m/K]
    :param crho:       Volumetric heat capacity of the rock [MJ/m³/K]
    :param q:          Injection rate per unit length [W/m]
    :param m:          Temperature-conductivity coefficient [1/K]
    :param sigma25:    Electrical conductivity at 25°C [S/m]
    :param times:      Times after injection start [h], shape: (nt,)
    :param mesh:       SimPEG cylindrical mesh
    :param survey:     SimPEG survey object
    :param simulation: SimPEG simulation object
    :param type:       Absolute of difference apparent resistivity
    :param eps:        Relative perturbation for finite differences

    :return:
        J : (ntimes*nrec, 4) Sensitivities w.r.t k, c, q, m
    """
    J = np.zeros((len(times)*survey.nD, 4), dtype=np.double)

    sim0 = simulate_ert(k, crho, q, m, sigma25, sigma_water, rbh,
                          times, mesh, survey, simulation).ravel()

    if type == 'abs':
        J[:, 0] = (simulate_ert(k * (1 + eps), crho, q, m, sigma25, sigma_water, rbh,
                              times, mesh, survey, simulation).ravel()
                   -sim0) / (eps * k)

        J[:, 1] = (simulate_ert(k, crho * (1 + eps), q, m,
                                sigma25, sigma_water, rbh,
                                times, mesh, survey, simulation).ravel()
                   -sim0) / (eps * crho)
        J[:, 2] = (simulate_ert(k, crho, q * (1 + eps), m,
                                sigma25, sigma_water, rbh,
                                times, mesh, survey, simulation).ravel()
                   - sim0) / (eps * q)
        J[:, 3] = (simulate_ert(k, crho, q, m * (1 + eps),
                                sigma25, sigma_water, rbh,
                                times, mesh, survey, simulation).ravel()
                   - sim0) / (eps * m)

    elif type == 'diff':
        sim0 = sim0.reshape(len(times), survey.nD)
        dsim0 = (sim0[1:, :] - sim0[0, :]).ravel()

        Jparam = simulate_ert(k * (1 + eps), crho, q, m,sigma25, sigma_water, rbh,
                           times, mesh, survey, simulation).reshape(len(times), survey.nD)
        J[:, 0] = ((Jparam[1:, :] - Jparam[0, :]).ravel() - dsim0) / (eps * k)

        Jparam = simulate_ert(k , crho * (1 + eps), q, m, sigma25, sigma_water, rbh,
                           times, mesh, survey, simulation).reshape(len(times), survey.nD)
        J[:, 1] = ((Jparam[1:,:] - Jparam[0,:]).ravel() - dsim0) / (eps * crho)

        Jparam = simulate_ert(k, crho, q * (1 + eps), m, sigma25, sigma_water, rbh,
                           times, mesh, survey, simulation).reshape(len(times), survey.nD)
        J[:, 3] = ((Jparam[1:,:] - Jparam[0,:]).ravel() - dsim0) / (eps * q)

        Jparam = simulate_ert(k , crho, q, m * (1 + eps), sigma25, sigma_water, rbh,
                           times, mesh, survey, simulation).reshape(len(times), survey.nD)
        J[:, 4] = ((Jparam[1:,:] - Jparam[0,:]).ravel() - dsim0) / (eps * m)

    else:
        raise ValueError("type must be 'abs' or 'diff'")

        return J

def _ICS(r, t, q, k, c, rbh, device, getJ, n=20000):
    # Note: n is usually set to 20000 for good accuracy, not need to change it.

    z = k / (c * 1e6) * t / rbh ** 2  # [nt]

    # Discretization of the integral
    d = torch.tensor([1e-160, 1e2], device=device, dtype=torch.float64)
    d = torch.log10(d) / torch.log10(torch.tensor(5.0, device=device))
    B = 5 ** torch.cat([torch.linspace(d[0], d[1], n - 1, device=device),
                        d[1].unsqueeze(0)])  # [nB]
    dB = torch.cat(
        [B[1:2] - B[0:1], (B[2:] - B[:-2]) / 2, B[-1:] - B[-2:-1]])  # [nB]

    J1B = bessel_j1(B)  # [nB]
    Y1B = bessel_y1(B)  # [nB]
    denom = B ** 2 * (J1B ** 2 + Y1B ** 2)  # [nB]

    nt = t.numel()
    nr = r.numel()
    DT = torch.empty((nt, nr), device=device, dtype=torch.float64)

    for i in range(nr):
        r_i = r[i]
        pB = B * r_i / rbh  # [nB]
        exp_term = (torch.exp(-B[None, :] ** 2 * z[:, None]) - 1)  # [nt, nB]
        num = bessel_j0(pB) * Y1B - bessel_y0(pB) * J1B  # [nB]
        integrand = exp_term * (num / denom * dB)[None, :]  # [nt, nB]
        DT[:, i] = q / (k * torch.pi ** 2) * torch.sum(integrand, dim=1)  # [nt]

    # Singularity at t=0
    id_zero = t == 0
    DT[id_zero, :] = 0
    DT = DT.squeeze()
    if getJ:
        return DT, DT
    else:
        return DT


def ICS(
    r: np.ndarray,
    t: np.ndarray,
    q: float,
    k: float,
    c: float,
    rbh: float,
    device: str = None,
    getJ: bool = False,
    Jtype: str = "fd",
    eps: float = 1e-8
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute the temperature variation ΔT(r, t) at distance r from the center
    of a borehole of radius rbh, after time t since the start of a constant
    rate injection q, in a medium with thermal conductivity k and volumetric
    heat capacity c. The solution is given by the integral:

        ΔT(r, t) = (q / k π²) ∫₀^∞ [exp(-β²z) - 1] ×
                   [J₀(pβ) Y₁(β) - Y₀(pβ) J₁(β)] / [β² (J₁²(β) + Y₁²(β))] dβ

    where:
        z  = (k / c) × (t / rbh²) = Fourier number
        p  = r / rbh
        J₀, J₁ = Bessel functions of the first kind
        Y₀, Y₁ = Bessel functions of the second kind

    :param r:      Distances from borehole center [m], shape: (nr,)
    :param t:      Times since injection start [s], shape: (nt,)
    :param q:      Constant rate injection [W/m]
    :param k:      Thermal conductivity [W/m/K]
    :param c:      Volumetric heat capacity [MJ/m³/K]
    :param rbh:    Borehole radius [m]
    :param device: Device for computation ('cpu' or 'cuda:0'), auto if None
    :param getJ:   If True, also compute sensitivities w.r.t k, c, q
    :param Jtype:  Jacobian computation: 'fd' (finite differences) or 'autodiff'
    :param eps:    Relative perturbation for finite differences

    :return:
        DT (np.ndarray):     Temperature variation [K], shape: (nt, nr)
        J  (np.ndarray, optional): Sensitivities w.r.t k, c, q,
                                   shape: (nt*nr, 3) (if getJ=True)
    """
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    r = torch.as_tensor(r, device=device).view(-1)  # [nr]
    t = torch.as_tensor(t, device=device).view(-1)  # [nt]
    k = torch.as_tensor(k, device=device, dtype=torch.float64)
    c = torch.as_tensor(c, device=device, dtype=torch.float64)
    q = torch.as_tensor(q, device=device, dtype=torch.float64)
    rbh = torch.as_tensor(rbh, device=device, dtype=torch.float64)
    if getJ:
        k = k.requires_grad_(True)
        c = c.requires_grad_(True)
        q = q.requires_grad_(True)

    if getJ:
        if Jtype == "autodiff":
            fun = torch.func.jacfwd(_ICS, argnums=(3, 4, 2), has_aux=True)
            # J: tuple of (dDT/dk, dDT/dc, dDT/dq)
            J, DT = fun(r, t, q, k, c, rbh, device, getJ)
            J = torch.stack(J, dim=-1)  # [nt, nr, 3]
            J = J.reshape(-1, 3)  # [nt*nr, 3]
        elif Jtype == "fd":
            eps = 1e-8
            DT, _ = _ICS(r, t, q, k, c, rbh, device, getJ)
            DT_k, _ = _ICS(r, t, q, k * (1 + eps), c, rbh, device,
                           getJ)  # k + eps
            DT_c, _ = _ICS(r, t, q, k, c * (1 + eps), rbh, device,
                           getJ)  # c + eps
            DT_q, _ = _ICS(r, t, q * (1 + eps), k, c, rbh, device,
                           getJ)  # q + eps
            J = torch.zeros((DT.numel(), 3), device=device, dtype=torch.float64)
            J[:, 0] = ((DT_k - DT) / (k * eps)).ravel()
            J[:, 1] = ((DT_c - DT) / (c * eps)).ravel()
            J[:, 2] = ((DT_q - DT) / (q * eps)).ravel()
        else:
            raise ValueError("Jtype must be 'fd' or 'autodiff'")
        return DT.detach().cpu().numpy(), J.detach().cpu().numpy()
    else:
        DT = _ICS(r, t, q, k, c, rbh, device, getJ)
        return DT.cpu().numpy()

def bayesian_inversion(fun: callable,
                       d: np.ndarray,
                       m0: np.ndarray,
                       Cdi: np.ndarray = None,
                       Cmi: np.ndarray = None,
                       niter: int = 5,
                       step: float = 1,
                       doprint: bool = False,
                       minit: np.ndarray = None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Bayesian linearized inversion using the Gauss-Newton method.

    The update equation at each iteration is:

        m^(i+1) = m^(i) - step × [J^T C_d^(-1) J + C_m^(-1)]^(-1) ×
                          [J^T C_d^(-1) (d_pred - d_obs) + C_m^(-1) (m^(i) - m_0)]

    where:
        m^(i)     = model parameters at iteration i
        m_0       = prior/reference model
        d_obs     = observed data
        d_pred    = predicted data from forward model
        J         = Jacobian matrix (sensitivities)
        C_d^(-1)  = inverse data covariance matrix
        C_m^(-1)  = inverse model covariance matrix
        step      = step length (damping factor)

    The objective function being minimized is:

        Φ(m) = ||C_d^(-1/2) (d_pred - d_obs)||² + ||C_m^(-1/2) (m - m_0)||²

    The posterior covariance matrix is computed as:

        C_m_post = [J^T C_d^(-1) J + C_m^(-1)]^(-1)

    :param d:       Observed data vector, shape: (nd,)
    :param m0:      Prior/reference model parameters, shape: (nm,)
    :param Cmi:     Inverse model covariance matrix, shape: (nm, nm)
    :param Cdi:     Inverse data covariance matrix, shape: (nd, nd)
    :param fun:     Forward modeling function that returns (d_pred, J) where
                    d_pred has shape (nd,) and J has shape (nd, nm)
    :param niter:   Number of iterations
    :param step:    Step length for update (default: 1.0, full step)
    :param doprint: If True, print misfit at each iteration
    :param minit:   Initial model parameters, shape: (nm,). If None, uses m0

    :return:
        m:            Inverted model after niter iterations, shape: (nm,)
        Cm_post:      Posterior covariance matrix, shape: (nm, nm)
        conf_95:      95% confidence intervals (1.96 × sqrt(diag(Cm_post))), shape: (nm,)
        cost_history: Cost function values at each iteration, shape: (niter,)
        cost_data:    Data misfit at each iteration, shape: (niter,)
        cost_model:   Model regularization at each iteration, shape: (niter,)
    """
    if minit is None:
        minit = m0
    if Cdi is None:
        Cdi = np.eye(len(d))
    prior = True
    if Cmi is None:
        Cmi = 0
        prior = False
    m = minit.copy()
    cost_history = np.zeros(niter)
    cost_data = np.zeros(niter)
    cost_model = np.zeros(niter)

    for i in range(niter):
        dmod, J = fun(m)
        r_d = dmod - d

        # Compute cost function:
        cost_data[i] = r_d.T @ Cdi @ r_d

        if prior:
            r_m = m - m0
            cost_model[i] = r_m.T @ Cmi @ r_m
        cost_history[i] = cost_data[i] + cost_model[i]
        Hessian = J.T @ Cdi @ J + Cmi
        r = J.T @ Cdi @ r_d
        if prior:
            r += Cmi @ r_m
        m = m - step * np.linalg.inv(Hessian) @ r
        if doprint:
            print(f"iter {i}: cost = {cost_history[i]:.3e} "
                  f"(data = {cost_data[i]:.3e}, model = {cost_model[i]:.3e}) "
                  f"m = {m}")

    # Compute posterior covariance matrix at final iteration
    dmod, J = fun(m)
    Cm_post = np.linalg.inv(J.T @ Cdi @ J + Cmi)

    # Compute 95% confidence intervals (1.96 × standard deviation)
    conf_95 = 1.96 * np.sqrt(np.diag(Cm_post))

    return m, Cm_post, conf_95, cost_history, cost_data, cost_model


def CTi(
    t: np.ndarray,
    a: float = 7,
    varT: float = 0.013):
    '''
    Covariance matrix for temperature data based on exponential covariogram
    :param t: time points (hours), shape: (nT,)
    :param a: correlation range (hours)
    :param varT: covariance value (degC^2)
    :return:  covariance matrix, shape: (nT, nT)
    '''
    nt = len(t)
    covD = np.zeros((nt, nt))

    for i in range(nt):
        for j in range(nt):
            dt = abs(t[i] - t[j])
            covD[i, j] = np.exp(-dt / a)

    covD *= varT
    return covD


def get_Cdi(
    tR: np.ndarray,
    tT: np.ndarray,
    de: np.ndarray,
    sigma_ert: float = 0.092,
    sigma_trt: float = 0.26,
    varT: float = 0.013,
    a: float = 7,
    covariogram: bool =True,
    sigma_q: bool =None):
    '''
    Get covariance matrix
    :param tR: time points for ERT (hours), shape: (nR,)
    :param tT: time points for TRT (hours), shape: (nT,)
    :param de: electrode separations (m), shape: (ne,)
    :param sigma_ert: measurement error for ERT
    :param sigma_trt: measurement error for TRT
    :param sigma_q: measurement error for injection rate
    :param varT: temperature covariance value (degC^2)
    :param covariogram: use covariogram with correlated residuals for TRT data
    :return: inversion of E-TRT covariance/precision matrix, shape: (ne*nR+nT, ne*nR+nT)
    '''
    if covariogram:
        C_ert = np.eye(len(de) * len(tR)) * sigma_ert ** 2
        C_trt = CTi(tT, a, varT)
        if sigma_q:
            C_q = np.eye(1) * sigma_q ** 2
            Cdi = scipy.linalg.block_diag(C_ert, C_trt, C_q)
            Cdi = np.linalg.inv(Cdi)
        else:
            Cdi = scipy.linalg.block_diag(C_ert, C_trt)
            Cdi = np.linalg.inv(Cdi)
    else:
        if sigma_q:
            Cdi = np.diag(np.concatenate([
                np.ones(len(de) * len(tR)) / sigma_ert ** 2,
                np.ones(len(tT)) / sigma_trt ** 2,
                np.ones(1) / sigma_q ** 2]))
        else:
            Cdi = np.diag(np.concatenate([
                np.ones(len(de) * len(tR)) / sigma_ert ** 2,
                np.ones(len(tT)) / sigma_trt ** 2]))
    return Cdi
