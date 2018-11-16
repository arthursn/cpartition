import numpy as np


def D(T, xC):
    """
    Composition dependent diffusion coefficient of FCC phase calculated 
    using Agren's equation
    """
    a = 4.53e5
    b = 8339.9/T
    c = 1./T - 2.221e-4
    d = 17767
    e = -26436

    yC = xC/(1. - xC)

    # D0 = 4.53e5*(1. + yC*(1.-yC)*8339.9/T)  # Pre-exponential term
    # D = D0*np.exp(-(1./T - 2.221e-4)*(17767 - yC*26436))  # um^2/s
    # return D
    return a*(1 + b*yC*(1-yC))*np.exp(-c*(d + e*yC))


def dDdx(T, xC):
    a = 4.53e5
    b = 8339.9/T
    c = 1./T - 2.221e-4
    d = 17767
    e = -26436

    yC = xC/(1. - xC)

    dDdy = a*(b*c*e*yC**2 - b*yC*(c*e + 2) + b - c*e)*np.exp(-c*(d + e*yC))

    return dDdy/(1 - xC)**2.


def dDdx_cdiff(T, xC, h=1e-5):
    return (D(T, xC+h) - D(T, xC-h))/(2.*h)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from cpartition import w2x

    fig, ax1 = plt.subplots()

    TC = 375  # T in oC
    T = TC + 273.15  # oC -> K

    wpC = np.linspace(0, 6, 100)  # wt.% C
    xC = w2x(wpC*1e-2)  # wt.% -> at. fraction

    ax1.plot(wpC, D(T, xC), 'k-', label=r'$D$')
    ax1.plot(wpC, dDdx(T, xC)*xC/4, 'r-',
             label=r'$\frac{x}{4} \frac{\partial D}{\partial x}$')

    ax2 = ax1.twinx()
    ax2.plot(wpC, (dDdx(T, xC)*xC/4)/D(T, xC), 'k--')

    ax1.set_title('{:.0f} °C'.format(TC))
    ax1.set_xlabel(u'Carbon (wt.%)')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left')

    plt.show()
