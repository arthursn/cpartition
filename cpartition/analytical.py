import numpy as np
from scipy.special import erfc
from scipy.optimize import newton

__all__ = ['Andrews', 'KM', 'D_fcc', 'parabolic_growth']

K = 273.15


def Andrews(C=0., Mn=0., Ni=0., Cr=0., Mo=0.):
    """
    Calculates the Ms temperature using Andrews equation

    [1] K.W. Andrews, Iron Steel Inst. J. 203 (1965) 721–727.

    Parameters
    ----------
    C, Mn, Ni, Cr, Mo : float
        Weight percentage of the respective elements

    Returns
    -------
    Ms : float
        Ms temperature in Celsius
    """
    return 539. - 423.*C - 30.4*Mn - 17.7*Ni - 12.1*Cr - 7.5*Mo


def KM(T, Ms, beta=0.011):
    """
    Koistinen-Margurger equation

    f_austenite = 1 - f_martensite = exp[-beta(Ms - T)]

    [1] D.P. Koistinen, R.E. Marburger, Acta Metall. 7 (1959) 59–60.

    Parameters
    ----------
    T : float
        Quenching temperature in Celsius or Kelvin
    Ms : float
        Ms temperature in the same unit as T
    beta : KM equation fitting parameter beta

    Returns
    -------
    f_fcc : float
        Untransformed phase fraction of austenite
    """
    return np.exp(-beta*(Ms - T))


def D_fcc(T_C, C=0):
    """
    Diffusion coefficient of austenite as calculated by Ågren's empirical
    formula

    [1] J. Ågren, Scr. Metall. 20 (1986) 1507–1510.

    Parameters
    ----------
    T_C : float
        Temperature in Celsius
    C : float
        Molar fraction of carbon in austenite

    Returns
    -------
    D : float
        Diffusion coefficient
    """
    T = T_C + K
    yC = C/(1. - C)
    D0 = 4.53e5*(1. + yC*(1.-yC)*8339.9/T)  # Pre-exponential term
    D = D0*np.exp(-(1./T - 2.221e-4)*(17767 - yC*26436))  # um^2/s
    return D


def parabolic_growth(T_C, c0, ci_bcc, ci_fcc):
    """
    Analytical solution for diffusion controlled growth of a ferrite plate,
    given by the equation: s = 2*alpha*(t^.5)

    Parameters
    ----------
    T_C : float
        Temperature in Celsius
    c0 : float
        Initial composition
    ci_bcc : float
        Interfacial composition in ferrite
    ci_fcc : float
        Interfacial composition in austenite

    Returns
    -------
    s(t) : function
        s(t) = 2*alpha*(t^.5)
    """
    D = D_fcc(T_C, c0)
    k = (D/np.pi)**.5*(c0 - ci_fcc)/(ci_bcc - ci_fcc)

    def f(x): return k*np.exp(-(D**-.5*x)**2)/erfc(D**-.5*x) - x
    alpha = newton(func=f, x0=1e-6)
    return lambda t: 2.*alpha*(t**.5)
