# -*- coding: utf-8 -*-

import sys, os
from itertools import cycle

import numpy as np 

from scipy.sparse import spdiags, csc_matrix
from scipy.sparse.linalg import dsolve
from scipy.optimize import newton, bisect, curve_fit
from scipy.interpolate import UnivariateSpline, interp1d, splrep, splev

from fnmatch import fnmatch

from .tab_datafile import load_table


K = 273.15
R = 8.3144598           # Gas constant in J/(K.mol)
kB = 1.38064852e-23     # Boltzmann constant in J/K
Tr = 25.                # room temperature


def WBs(T_C):
    """
    Calculates the extra energy for growth of bainite and Widmansttaten ferrite
    
    Parameters
    ----------
    T_C : float
        Temperature in Celsius

    References
    ----------
    .. [1] Hillert, M., Hoglund, L. & Agren, J. Role of carbon and alloying
        elements in the formation of bainitic ferrite. Metall. Mater. Trans. A 
        35, 3693--3700 (2004).
    """
    tck = splrep([-200., 300., 450., 700., 800.], [6000., 2329., 1283., 107., 4.])
    return splev(T_C, tck)


class Domain(object):
    # type code for the diffusion coefficient
    tcode = {0: 0, 'comp.local': 0, 'local': 0,
             1: 1, 'comp.avg': 1, 'avg': 1,
             2: 2, 'comp.initial': 2, 'initial': 2, 'init': 2, 'c0': 2,
             3: 3, 'pseudophase': 3, 'carbides': 3}
    # label code for the diffusion coefficient
    lcode = {0: '0 (comp.local)', 
             1: '1 (comp.avg)',
             2: '2 (comp.initial)',
             3: '3 (pseudophase alpha + carbides)'}

    def __init__(self, T_C, dt, z, c, **kwargs):
        self.T_C = T_C
        self.RT = R*self.T
        self.dt = dt
        self.z = z
        self.c = c

        self.n_time = kwargs.get('n_time', None)

        if self.z is None:
            self.z0, self.zn = kwargs.get('z0', 0.), kwargs.get('zn', 1.)  # positions at th 0-th and n-th nodes
            try:
                self.n = len(self.c)
            except:
                self.n = kwargs.get('n', 100)   # number of nodes in the grid
            self.z = np.linspace(self.z0, self.zn, self.n)  # position of each node
        
        if self.c is None:
            self.c0 = kwargs.get('c0', 0)   # value at t=0
            try:
                self.n = len(self.z)
            except:
                self.n = kwargs.get('n', 100)   # number of nodes in the grid
            self.c = np.full(self.n, self.c0) # composition in the instant t
            
        self.initialize_grid()

        self.active = True

    @property
    def T(self):
        """
        Absolute temperature in K
        """
        return self.T_C + K

    def initialize_grid(self, reset=True):
        """
        Initialize grid for the FDM method and arrays containing 
        information about the grid (ci0, cin, cavg, s0, sn)
        ci0 : composition at the 0-th node
        cin : composition at the n-th node
        cavg : average composition in the domain
        s0 : position of the 0-th node
        sn : position of the n-th node

        Parameters
        ----------
        reset : boolean
            If True, clears the arrays ci0, cin, cavg, s0, and sn
        """
        self.z0 = self.z[0]
        self.zn = self.z[-1]
        if len(self.z) != len(self.c):
            raise Exception('Sizes of z and c are not equal')
        else:
            self.n = len(self.z)
        self.dz = (self.z[-1] - self.z[0])/(self.n - 1)  # distance between two nodes
        self.L = np.abs(self.z[-1] - self.z[0])  # length of the grid

        self._c = np.zeros(self.c.shape)  # self._c is used for solving the FDM
        self._c1 = np.zeros(self.c.shape)  # self._c is used for solving the explicit FDM
        self._b = np.zeros(self.c.shape)  # self._b is used for solving the implicit FDM
        
        self.r = np.zeros(self.c.shape)
        self.g = np.zeros(self.c.shape)
        
        if reset is True:
            if self.n_time is None:
                self.ci0 = np.array([])
                self.cin = np.array([])
                self.cavg = np.array([])
                self.s0 = np.array([])
                self.sn = np.array([])
            else:
                self.ci0 = np.zeros(self.n_time)
                self.cin = np.zeros(self.n_time)
                self.cavg = np.zeros(self.n_time)
                self.s0 = np.zeros(self.n_time)
                self.sn = np.zeros(self.n_time)
                self.it = 0
            self.ds = 0.

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def toggle_status(self):
        if self.active:
            self.deactivate()
        else:
            self.activate()

    def FDM_implicit(self, bc0=[-1.5,2.,-.5,0.], bcn=[1.5,-2.,.5,0.], lowerbound=0):
        """
        Calculate one step of the implicit Finite Elements Method for Fick's 
        second law equation.
        
        Parameters
        ----------
        bc0 : array_like
            Parameters for the boundary condition at the 0-th node
        bcn : array_like
            Parameters for the boundary condition at the n-th node
        
        At the 0-th node, the correspondent linear equation is given by 
        bc0[0]*c[0] + bc0[1]*c[1] + bc0[1]*c[2] = bc0[3].
        At the n-th node, the correspondent linear equation is given by 
        bcn[0]*c[-1] + bcn[1]*c[-2] + bcn[2]*c[-3] = bcn[3].

        Dirichlet boundary condition is given by bc0/bcn = (1.,0.,0.,c_i).
        Neumann boundary condition (flux = 0) can be defined either by 
        bc0 (bcn) = [1.5,-2.,.5,0.] or bc0 (bcn) = [1.,-1.,0.,0.]
        """
        self.get_r()
        self.get_g()

        dia1_, dia1 = np.zeros(self.n), np.zeros(self.n)
        dia2_, dia2 = np.zeros(self.n), np.zeros(self.n)

        dia0 = 1. + 2.*self.r
        dia1_[:-1], dia1[1:] = -(self.r[1:] - self.g[1:]), -(self.r[:-1] + self.g[:-1])
        self._b[:] = self.c

        # First row of the matrix. Determines the boundary condition at i=0
        dia0[0], dia1[1], dia2[2] = bc0[0], bc0[1], bc0[2]
        # Determines the boundary condition at i=0
        self._b[0] = bc0[3]

        # Last row of the matrix. Determines the boundary condition at i=n
        dia0[-1], dia1_[-2], dia2_[-3] = bcn[0], bcn[1], bcn[2]
        # Determines the boundary condition at i=n
        self._b[-1] = bcn[3]

        mtx = spdiags([dia2_, dia1_, dia0, dia1, dia2], [-2, -1, 0, 1, 2], self.n, self.n)
        A = csc_matrix(mtx, dtype=np.float64)

        # Solve the linear system A*c = b
        self._c = dsolve.spsolve(A, self._b, use_umfpack=True)
        
        if self._c[0] < lowerbound or self._c[-1] < lowerbound:
            return True

        self.c[:] = self._c

    def FDM_explicit(self, bc0=[1.5,-2.,.5,0.], bcn=[1.5,-2.,.5,0.]):
        """
        Calculate one step of the explicit Finite Elements Method for Fick's
        second law equation.
        
        Parameters
        ----------
        bc0 : array_like
            Parameters for the boundary condition at the 0-th node
        bcn : array_like
            Parameters for the boundary condition at the n-th node
        
        At the 0-th node, the correspondent linear equation is given by 
        bc0[0]*c[0] + bc0[1]*c[1] + bc0[1]*c[2] = bc0[3].
        At the n-th node, the correspondent linear equation is given by
        bcn[0]*c[-1] + bcn[1]*c[-2] + bcn[2]*c[-3] = bcn[3].

        Dirichlet boundary condition is given by bc0/bcn = (1.,0.,0.,c_i).
        Neumann boundary condition (flux = 0) can be defined either by 
        bc0 (bcn) = [1.5,-2.,.5,0.] or bc0 (bcn) = [1.,-1.,0.,0.]
        """
        self._c[:] = self.c
        self._c1.fill(0)

        self.get_r()
        self.get_g()

        self._c1[1:-1] = (self.r[1:-1]-self.g[1:-1])*self._c[:-2] + \
            (1. - 2.*self.r[1:-1])*self._c[1:-1] + \
            (self.r[1:-1]+self.g[1:-1])*self._c[2:]

        self._c1[0] = (bc0[3] - (bc0[1]*self._c1[1] + bc0[2]*self._c1[2]))/bc0[0]
        self._c1[-1] = (bcn[3] - (bcn[1]*self._c1[-2] + bcn[2]*self._c1[-3]))/bcn[0]
        self.c[:] = self._c1

    def update_grid(self, v0=0., vn=0.):
        """
        Update the interface position and interpolate the composition

        Parameters
        ----------
        v0 : float
            Interface velocity at the 0-th node
        vn : float
            Interface velocity at the n-th node
        """
        if v0 != 0 or vn != 0:
            ds0 = v0*self.dt
            dsn = vn*self.dt
            z1 = np.linspace(self.z[0] + ds0, self.z[-1] + dsn, self.n)
            dz1 = (z1[-1] - z1[0])/(self.n - 1)

            if self.dz != 0.:
                self._c = self.c
                k = (z1 - self.z)/self.dz
                self._c[1:-1] = self.c[1:-1] + .5*k[1:-1]*(self.c[2:] - self.c[:-2])
                self.c[:] = self._c

            self.z[:] = z1
            self.L = np.abs(self.z[-1] - self.z[0])
            self.dz = dz1

        if self.n_time is None:
            self.ci0 = np.append(self.ci0, self.c[0])
            self.cin = np.append(self.cin, self.c[-1])
            self.cavg = np.append(self.cavg, self.get_cavg())
            self.s0 = np.append(self.s0, self.z[0])
            self.sn = np.append(self.sn, self.z[-1])
        else:
            self.ci0[self.it] = self.c[0]
            self.cin[self.it] = self.c[-1]
            self.cavg[self.it] = self.get_cavg()
            self.s0[self.it] = self.z[0]
            self.sn[self.it] = self.z[-1]
            self.it += 1

    def get_cavg(self):
        """
        Calculates the average composition in the Domain using the trapezium 
        rule
        """
        return np.trapz(y=self.c)/(self.n - 1)

    def x2w(self, x):
        """
        Convert molar fraction of carbon to weight fraction
        Method only available if the chemical potentials are
        loaded from a Thermo-Calc generated file
        """
        from periodictable import elements

        if isinstance(x, list):
            x = np.array(x)
        # select X(*) like columns 
        xcols = [c for c in self.chempot.columns if fnmatch(c, 'X(*)') and c != 'X(Z)']
        # created DataFrame with selected columns and rename them accordingly
        dfx = self.chempot[xcols].rename(mapper=lambda c: c[2:-1].title(), axis='columns')
        # get molar mass
        M = np.array([elements.symbol(el).mass for el in dfx.columns])
        # interpolate composition
        xcomp = interp1d(dfx['C'].values, dfx.values.T, axis=1)(x)
        if xcomp.ndim == 1:
            sumxcomp = float(np.sum(xcomp.T*M))
        else:
            sumxcomp = np.sum(xcomp.T*M, axis=1)

        return x*elements.symbol('C').mass/sumxcomp

    def prepare_tdata(self):
        # Calculates chemical potential of fictitious element Z
        # This is necessary to calculate the paraequilibrium
        muZ = np.zeros(len(self.chempot.index))
        for c in self.chempot:
            if fnmatch(c, 'X(*)') and c != 'X(C)':
                X, MU = c, 'MU(' + c[2:-1] + ')'
                self.chempot[MU] += self.E
                muZ += self.chempot[X]*self.chempot[MU]

        self.chempot['X(Z)'] = 1 - self.chempot['X(C)'].values
        self.chempot['MU(Z)'] = muZ.values/self.chempot['X(Z)']
        self.chempot['MU(C)'] += self.E

        self.x2mu['C'] = interp1d(self.chempot['X(C)'], self.chempot['MU(C)'],
                        fill_value='extrapolate')  # X(C) to MU(C)
        self.x2mu['Z'] = interp1d(self.chempot['X(C)'], self.chempot['MU(Z)'],
                        fill_value='extrapolate')  # X(C) to MU(Z)
        self.mu2x['C'] = interp1d(self.chempot['MU(C)'], self.chempot['X(C)'],
                        fill_value='extrapolate')  # MU(C) to X(C)

        self.muC2muZ = interp1d(self.chempot['MU(C)'], self.chempot['MU(Z)'],
                        fill_value='extrapolate')


class BCC(Domain):
    """
    FDM simulation domain for a BCC phase

    Parameters
    ----------
    T_C : float
        Temperature in Celsius
    dt : float
        Time step
    z : array_like
        n-length array representing the positions of each node
    c : array_like
        n-length array representing composition in each node
    type_D : integer or string
        Decides how the diffusion coefficient will be calculated at the
        interface. D 
        - {0, 1, 2, comp.local, local, comp.avg, avg, comp.initial, 
        initial, init, c0} : D is indenpendent of carbon content
        - {3, pseudophase, carbides} : D will depend on the ferrite 
        fraction in the alpha + carbides pseudophase. The ferrite 
        fraction is calculated using the lever rule. By default it is
        assumed that the carbon solubility of 0.25 for the carbide and
        0 for the bcc phase. 
        If you wish defining a carbide with a different stoichiometry,
        or a different solubility limit for the bcc phase, you can supply
        the values as kwargs arguments c_carbide and cmax_bcc (see below)
    tdata : string
        Filepath to table generated by Thermo-Calc containing 
        thermodynamical data
    tpar : array_like
        Thermodynamical parameters obtained after fitting of chemical 
        potential curves

    **kwargs :
        c_carbide : float
            Carbon solubility of the carbide
        cmax_bcc : float
            Carbon solubility of the bcc phase
        Vm : float
            Molar volume of the phase
        E : float
            Extra energy to added to the phase
    """
    structure = 'bcc'

    def __init__(self, T_C, dt=0., z=None, c=None, type_D=0,
                 tdata=None, tpar=[111918., -51.44], **kwargs):
        # Instantiate super class Domain
        super(BCC, self).__init__(T_C, dt, z, c, **kwargs)

        self.Vm = kwargs.get('Vm', 7.0923e-6)    # Molar volume of iron (m^3/mol)
        self.E = kwargs.get('E', 0.)   # Extra energy (e.g., to fit WBs theory)

        self.type_D = type_D
        try:
            self.type_D =  self.tcode[self.type_D]
        except:
            raise Exception('Invalid option')

        self.c_carbide = kwargs.get('c_carbide', 0.25)
        self.cmax_bcc = kwargs.get('cmax_bcc', 0)

        self.tdata = tdata
        self.tpar = tpar  # thermodynamical parameters obtained after fitting of chemical potential curves

        self.prepare_chempot()

    def prepare_chempot(self):
        """
        Creates chemical potential functions for C and Fe either from the 
        Themo-Calc generated table or based on the thermodynamical parameters
        supplied in tdata
        """
        self.x2mu, self.mu2x = {}, {}

        if self.tdata:
            try:
                self.chempot = load_table(self.tdata, 'X(C)', ignorephaseregions='*#2')  # Loads thermodynamical data from tdata file
            except:
                raise Exception('Cannot load file "{}".'.format(self.tdata))
            else:
                self.prepare_tdata()

                # I'm assuming that the relation between the chemical potential and the composition
                # is described by the ideal solution approximation:
                # mu = RT log(G*x) = RT (log(x) + log(G)) => log(x) = mu/RT - log(G)
                # It must be used VERY CAREFULLY, once the extrapolated values of muC might significantly
                # deviate from the real values for high carbon contents.            
                def func(mu, logG):
                    return mu/self.RT - logG
                popt, pcov = curve_fit(func, self.chempot['MU(C)'], np.log(self.chempot['X(C)']))
                RTlogG = lambda: self.RT*popt[0]    # RT log(G), where G is the activity coefficient
        else:
            print('Warning! tdata=None does not support mobiles interfaces')
            
            cypar = cycle(['A', 'B', 'C', 'D', 'E', 'F'])
            self.tdata = '; '.join(['{} = {:.3f}'.format(next(cypar), par) for par in self.tpar])

            # Again, ideal solution approximation is used, but considering the supplied tpar parameters
            RTlogG = lambda: self.tpar[0] + self.tpar[1]*self.T + self.E  # RT log(G)

        self.x2mu['C'] = lambda x: self.RT*np.log(x) + RTlogG()  # Inform carbon comp. (x), return chem. pot. (mu)
        self.mu2x['C'] = lambda mu: np.exp((mu - RTlogG())/self.RT)  # Inform mu, return x

    def D(self, C=0):
        """
        Diffusion coefficient of carbon in the BCC phase determined according 
        Agren, 1982

        Parameters
        ----------
        C : float
            Mole fraction of carbon in solid solution in the BCC phase. Agren's 
            equation does not take composition dependence into account for 
            computing D, so changing C has no effect in the returned value
        
        Returns
        -------
        D : float
            Diffusion coefficient in um^2/s
        
        References
        ----------
        .. [1] J. Agren, 'Diffusion in phases with several components and 
            sublattices', J. Phys. Chem. Solids, vol. 43, no. 5, pp. 421--430, 
            Jan. 1982.
        """
        D0 = 0.02e8*np.exp(-10115./self.T)   # Pre-exponential term
        D = D0*np.exp(0.5898*(1. + 2.*np.arctan(1.4985 - 15309./self.T)/np.pi))   # um^2/s

        if isinstance(C, np.ndarray):
            D = np.repeat(D, len(C))

        if self.type_D == 3:
            D *= (self.c_carbide - C)/(self.c_carbide - self.cmax_bcc)

        return D

    def get_r(self):
        """
        r term from the FDM discretization (r = D*dt/dx^2)
        """
        self.r[:] = self.D(self.c)*self.dt/self.dz**2

    def get_g(self):
        """
        The value of D in the bcc phase doesn't take into account the effect of
        the carbon content, so the g term is 0
        """
        if self.type_D == 3:
            D, D1 = np.zeros(self.n), np.zeros(self.n)
            D[:-1], D1[1:] = self.D(self.c[1:]), self.D(self.c[:-1])  # Very tricky relation here!
            self.g[:] = .25*(D - D1)*self.dt/self.dz**2
        else:
            self.g.fill(0)

class FCC(Domain):
    """
    FDM simulation domain for a FCC phase

    Parameters
    ----------
    T_C : float
        Temperature in Celsius
    dt : float
        Time step
    z : array_like
        n-length array representing the positions of each node
    c : array_like
        n-length array representing composition in each node
    type_D : integer or string
        Decides how the diffusion coefficient will be calculated at the 
        interface according to the tcode dictionary:
        - {0, comp.local, local} : the local interfacial carbon content is
        used
        - {1, comp.avg, avg} : the average carbon content on the domains is
        used
        - {2, comp.initial, initial, init, c0} : the initial carbon of the
        domain is used
    tdata : string
        Filepath to table generated by Thermo-Calc containing
        thermodynamical data
    tpar : array_like
        Thermodynamical parameters obtained after fitting of chemical
        potential curves

    **kwargs:
        Vm : float
            Molar volume of the phase
        E : float
            Extra energy to added to the phase
    """
    structure = 'fcc'

    def __init__(self, T_C, dt=0., z=None, c=None, type_D=0,
                 tdata=None, tpar=[35129., -7.639, 169105., -120.4], **kwargs):
        # Instantiate super class Domain
        super(FCC, self).__init__(T_C, dt, z, c, **kwargs)

        self.Vm = kwargs.get('Vm', 7.0923e-6)  # Molar volume of iron (m^3/mol)
        self.E = kwargs.get('E', 0.)  # Extra energy (e.g., WBs theory)

        self.type_D = type_D
        try:
            self.type_D =  self.tcode[self.type_D]
        except:
            raise Exception('Invalid option')

        self.tdata = tdata
        self.tpar = tpar  # thermodynamical parameters obtained after fitting of chemical potential curves

        self.prepare_chempot()

    def prepare_chempot(self):
        """
        Creates chemical potential functions for C and Fe either from the 
        Themo-Calc generated table or based on the thermodynamical parameters
        supplied in tdata
        """
        self.x2mu, self.mu2x = {}, {}

        if self.tdata:
            try:
                self.chempot = load_table(self.tdata, 'X(C)', ignorephaseregions='*#2')
            except:
                raise Exception('Cannot load file "{}".'.format(self.tdata))
            else:
                self.prepare_tdata()
        else:
            print('Warning! tdata=None does not support mobiles interfaces')
            
            if len(self.tpar) > 6:
                raise Exception('tpar length must not be longer than 6')

            inc = 6 - len(self.tpar)
            # Six parameters for quadratic approximation
            self.tpar = np.pad(self.tpar, (0, inc), 'constant')

            cypar = cycle(['A', 'B', 'C', 'D', 'E', 'F'])
            self.tdata = '; '.join(['{} = {:.3f}'.format(next(cypar), par) for par in self.tpar])

            # Quadratic approximation for RT log(G)
            # RT log(G) = A + B*T + (C + D*T)*x + (E + F*T)*x
            RTlogG = lambda x: self.tpar[0] + self.tpar[1]*self.T + \
                            (self.tpar[2] + self.tpar[3]*self.T)*x + \
                            (self.tpar[4] + self.tpar[5]*self.T)*x**2. + self.E

            self.x2mu['C'] = lambda x: self.RT*np.log(x) + RTlogG(x)
            # TO DO: IMPLEMENT mu2x
            # self.mu2x['C'] = lambda mu: np.exp((mu - RTlogG())/self.RT)

    def D(self, C=0):
        """
        Composition dependent diffusion coefficient of FCC phase calculated 
        using Agren's equation

        Parameters
        ----------
        C : float, numpy array
            Molar fraction of carbon in solid solution in the FCC phase.
        
        Returns
        -------
        D : float, numpy array
            Diffusion coefficient in um^2/s

        References
        ----------
        .. [1] J. Agren, 'A revised expression for the diffusivity of carbon in
            binary Fe-C austenite', Scr. Metall., vol. 20, no. 11, pp. 
            1507--1510, Nov. 1986.
        """
        yC = C/(1. - C)
        D0 = 4.53e5*(1. + yC*(1.-yC)*8339.9/self.T) # Pre-exponential term
        D = D0*np.exp(-(1./self.T - 2.221e-4)*(17767 - yC*26436)) # um^2/s
        return D

    def get_r(self):
        """
        r term from the FDM discretization (r = D*dt/dx^2)
        """
        if self.type_D == 0:
            self.r[:] = self.D(C=self.c)*self.dt/self.dz**2
        elif self.type_D == 1:
            self.r[:] = np.repeat(self.D(self.get_cavg())*self.dt/self.dz**2, self.n)
        elif self.type_D == 2:
            self.r[:] = np.repeat(self.D(self.c0)*self.dt/self.dz**2, self.n)

    def get_g(self):
        """
        g term from the FDM discretization
        """
        if self.type_D == 0:
            D, D1 = np.zeros(self.n), np.zeros(self.n)
            D[:-1], D1[1:] = self.D(self.c[1:]), self.D(self.c[:-1])  # Very tricky relation here!
            self.g[:] = .25*(D - D1)*self.dt/self.dz**2
        else:
            self.g.fill(0)


class Interface(object):
    """
    Initialize interface
    
    Parameters
    ----------
    domain1 : FCC or BCC object
        If domain2 is a BCC object, domain1 must be a FCC object and 
        vice-versa
    domain2 : FCC or BCC object
        If domain1 is a BCC object, domain2 must be a FCC object and 
        vice-versa
    type_int : integer or string
        Defines the method of calculation of the interfacial compositions 
        according to the tcode dictionary
        - {0, fixed.balance, fixed.bal, balance, bal, fixed} : for 
        martensite/austenite interfaces. Int. comp. calculated using global
        mass balance
        - {1, fixed.fluxes, fixed.flux, fluxes, flux}: for martensite/
        austenite interfaces. Int. comp. calc. using equality of fluxes
        - {2, mobile.equilibrium, mobile.eq, equilibrium, eq, mobile}: for
        ferrite/austenite interfaces. Local [para]equilibrium. Int. comp. 
        is calculated only once.
        - {3, mobile.mmode, mmode, mixed}: for ferrite/austenite 
        interfaces. Mixed-mode approach. Equality of chemical potentials is
        considered
    
    **kwargs :
        Parameters for the interface mobility equation (M = M0 exp(-Qa/RT))
        M0 : float
            M0
        Qa : float
            Qa, activation energy
    """

    # type code for the boundary conditions at the interface
    tcode = {0: 0, 'fixed.balance': 0, 'fixed.bal': 0, 'balance': 0, 'bal': 0, 'fixed': 0,
             1: 1, 'fixed.fluxes': 1, 'fixed.flux': 1, 'fluxes': 1, 'flux': 1,
             2: 2, 'mobile.equilibrium': 2, 'mobile.eq': 2, 'equilibrium': 2, 'eq': 2, 'mobile': 2, 
             3: 3, 'mobile.mmode': 3, 'mmode': 3, 'mixed': 3}
    # label code for the boundary conditions at the interface
    lcode = {0: '0 (fixed.balance)',
             1: '1 (fixed.fluxes)',
             2: '2 (mobile.equilibrium)',
             3: '3 (mobile.mmode)'}

    def __init__(self, domain1, domain2, type_int=0, **kwargs):
        # p_xxx denotes the position of the domain relative to the interface
        # p_xxx = 0 means that the domain is positioned to the left of the interface
        # p_xxx = 1 means that the domain is positioned to the right of interface
        if domain1.structure == 'bcc':
            self.bcc, self.fcc = domain1, domain2
            self.p_bcc, self.p_fcc = 0, 1
        else:
            self.fcc, self.bcc = domain1, domain2
            self.p_fcc, self.p_bcc = 0, 1

        if self.bcc.T_C != self.fcc.T_C:
            raise Exception('Different temperatures set to each phase')

        self.v = 0.
        self.type_int = self.tcode[type_int]
        try:
            self.type_int = self.tcode[self.type_int]
        except:
            raise Exception('Invalid option')

        self.M0 = kwargs.get('M0', 2.e-4)
        self.Qa = kwargs.get('Qa', 140e3)

        self.initialize()

        self.active = True

    def initialize(self):
        """
        Redefines methods 'CCE' and 'k' of this class depending on the chosen
        options for interface.
        """
        self.update_y()     # updates y values (composition in the nodes at the vicinity of the interface)

        self.RT = R*self.bcc.T   # RT
        self.ci_bcc, self.ci_fcc = self.y_bcc[0], self.y_fcc[0]   # Interfacial compositions

        # self.comp is redefined for each interface type
        if self.type_int == 0:
            self.comp = self.balance_fixed_int
        elif self.type_int == 1:
            self.comp = self.stefan_local_equilibrium
        elif self.type_int == 2:
            g = lambda x: self.fcc.muC2muZ(x) - self.bcc.muC2muZ(x)
            lo = max(min(self.fcc.chempot['MU(C)']), min(self.bcc.chempot['MU(C)']))
            hi = min(max(self.fcc.chempot['MU(C)']), max(self.bcc.chempot['MU(C)']))
            muC_eq = bisect(g, lo, hi, xtol=1e-3)
            self.ci_bcc = self.bcc.mu2x['C'](muC_eq)
            self.ci_fcc = self.fcc.mu2x['C'](muC_eq)
            self.comp = lambda: (self.ci_bcc, self.ci_fcc)
        elif self.type_int == 3:
            self.comp = self.stefan_local_equilibrium

        if self.fcc.type_D == 0:
            self.k = lambda x: (self.fcc.D(C=x)*self.bcc.dz)/(self.bcc.D(C=self.y_bcc[0])*self.fcc.dz)
        elif self.fcc.type_D == 1:
            self.k = lambda x: (self.fcc.D(C=self.fcc.get_cavg())*self.bcc.dz)/(self.bcc.D(C=self.y_bcc[0])*self.fcc.dz)
        elif self.fcc.type_D == 2:
            self.k = lambda x: (self.fcc.D(C=self.fcc.c0)*self.bcc.dz)/(self.bcc.D(C=self.y_bcc[0])*self.fcc.dz)
    
    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def toggle_status(self):
        if self.active:
            self.deactivate()
        else:
            self.activate()

    def update_y(self):
        """
        Creates/updates array y_xxx corresponding to the composition at the 
        extremities of the nodes. It's necessary to update their values after
        each iteration.
        """
        if self.p_bcc == 0:
            self.y_bcc, self.y_fcc = self.bcc.c[:-4:-1], self.fcc.c[0:3]
            self._y_bcc, self._y_fcc = self.bcc.c[0], self.fcc.c[-1]
        else:
            self.y_fcc, self.y_bcc = self.fcc.c[:-4:-1], self.bcc.c[0:3]
            self._y_fcc, self._y_bcc = self.fcc.c[0], self.bcc.c[-1]

    def D_interface(self):
        """
        Calculate the diffusion coefficient of carbon at the interface
        """
        self.update_y() # Important!
        
        D_bcc = self.bcc.D(C=self.y_bcc[0])

        if self.fcc.type_D == 0:
            c_fcc = self.y_fcc[0]
        elif self.fcc.type_D == 1:
            c_fcc = self.fcc.get_cavg()
        elif self.fcc.type_D == 2:
            c_fcc = self.fcc.c0
        D_fcc = self.fcc.D(C=c_fcc)
        
        return D_bcc, D_fcc

    def CCE(self, c_fcc):
        """
        Receives c_fcc as input and returns c_bcc with same chemical potential.
        
        Parameters
        ----------
        c_fcc : float
            Carbon composition in the FCC phase
        """
        mu = self.fcc.x2mu['C'](c_fcc)
        self.ci_bcc = self.bcc.mu2x['C'](mu)
        self.ci_fcc = c_fcc
        return self.ci_bcc
        # A, B, C, D = (35129., -7.639, 169105., -120.4)
        # AA, BB = (111918., -51.44)
        # return c_fcc*np.exp((A - AA + (B - BB)*self.bcc.T + (C + D*self.bcc.T)*c_fcc)/(self.RT))

    def stefan_local_equilibrium(self, **kwargs):
        """
        Computes interfacial compositions assuming local equilibrium for carbon
        (same chemical potential) for the problem of fixed interface

        Parameters
        ----------
        **kwargs :
            poly_deg : integer
                Degree of the polynomial used for calculating the fluxes at the
                interface
            guess : float
                Initial guess for Newton method used to solve the problem
        """
        self.update_y()
        if self.v == 0.:
            if kwargs.get('poly_deg', 2) == 2:
                f = lambda x: (self.CCE(x) - self.y_bcc[1])/self.k(x) + (x - self.y_fcc[1])
            else:
                f = lambda x: (1.5*self.CCE(x) - 2.*self.y_bcc[1] + \
                        .5*self.y_bcc[2])/self.k(x) + \
                        (1.5*x - 2.*self.y_fcc[1] + .5*self.y_fcc[2])
        else:
            # BE EXTRA CAREFUL
            if kwargs.get('poly_deg', 2) == 2:
                f = lambda x: (self.CCE(x) - self.y_bcc[1]) + \
                        (x - self.y_fcc[1])*self.k(x) - \
                        (-1.)**self.p_bcc*self.v*(x - self.CCE(x))*self.bcc.dz/self.bcc.D(C=self.CCE(x))
            else:
                f = lambda x: (1.5*self.CCE(x) - 2.*self.y_bcc[1] + .5*self.y_bcc[2]) + \
                        (1.5*x - 2.*self.y_fcc[1] + .5*self.y_fcc[2])*self.k(x) - \
                        (-1.)**self.p_bcc*self.v*(x - self.CCE(x))*self.bcc.dz/self.bcc.D(C=self.CCE(x))
            
        guess = kwargs.get('guess', self.y_fcc[0])
        try:
            self.ci_fcc = newton(func=f, x0=guess)
        except:
            print('Cannot solve Stefan problem')
            print('y_fcc[0]={}, y_fcc[1]={}, y_bcc[0]={}, v={}'.format(
                self.y_fcc[0], self.y_fcc[1], self.y_bcc[1], self.v))
            raise
        self.ci_bcc = self.CCE(self.ci_fcc)
        return self.ci_bcc, self.ci_fcc

    def balance_fixed_int(self, c0, **kwargs):
        """
        Computes interfacial compositions assuming mass balance of carbon in 
        the whole system

        Parameters
        ----------
        c0 : float
            Initial composition of the domain
        
        **kwargs :
            guess : float
                Initial guess for Newton method used to solve the problem
        """
        self.update_y()
        sum_bcc = np.sum(self.bcc.c[1:-1]) + .5*self._y_bcc
        sum_fcc = np.sum(self.fcc.c[1:-1]) + .5*self._y_fcc
        f = lambda x: (.5*self.CCE(x) + sum_bcc)*self.bcc.dz + (.5*x + sum_fcc)*self.fcc.dz - c0*(self.bcc.L + self.fcc.L)
        guess = kwargs.get('guess', self.y_fcc[0])
        self.ci_fcc = newton(func=f, x0=guess)
        self.ci_bcc = self.CCE(self.ci_fcc)
        return self.ci_bcc, self.ci_fcc

    def chem_driving_force(self, **kwargs):
        """
        Calculates chemical driving force
        
        Parameters
        ----------
        **kwargs
            ci_bcc and ci_fcc : float
                Interfacial compositions (carbon)
        
        Returns
        -------
        F : float
            Chemical driving force in J/mol
        """
        ci_bcc = kwargs.get('ci_bcc', self.ci_bcc)
        ci_fcc = kwargs.get('ci_fcc', self.ci_fcc)
        F = (1. - ci_bcc)*(self.fcc.x2mu['Z'](ci_fcc) - self.bcc.x2mu['Z'](ci_bcc))
        return (-1.)**self.p_bcc*F

    def M(self):
        """
        Calculates the interface mobility
        Returns
        -------
        M : float
            Interface mobility in m^4/(J.s)
        """
        M = self.M0*np.exp(-self.Qa/(self.RT))
        return M

    def flux(self, who=['bcc','fcc'], nnodes=3):
        """
        Calculates flux at the interfaces of both domains. The gradient is 
        evaluated using Lagrangian interpolation of either the 3 (nnodes=3)
        or 2 (nnodes=2) nearest points to the interface. nnodes=3 by default

        Parameters
        ----------
        who : string or array_like
            {bcc, fcc} : Phase(s) to which the fluxes will be calculated
        nnodes : integer
            Degree of the Lagrangian polynomial used to calculate the 
            composition gradient
        """
        D_bcc, D_fcc = self.D_interface()
        J_bcc, J_fcc = 0., 0.
        fluxes = []
        coef = np.array([1.5, -2., .5])
        if nnodes == 2:
            coef = np.array([1., -1., 0.])
        if self.bcc.dz > 0.:
            # The derivative is calculated at the n-th node
            grad = np.dot(coef, self.y_bcc)/self.bcc.dz
            # Clever solution! If the domain is in the position 0, grad doesn't change its sign
            grad *= (-1.)**self.p_bcc
            J_bcc = -D_bcc*grad
        if self.fcc.dz > 0.:
            # The derivative is calculated at the 0-th node
            grad = np.dot(coef, self.y_fcc)/self.fcc.dz
            # If the domain is in the position 1, grad changes its sign
            grad *= (-1.)**self.p_fcc
            J_fcc = -D_fcc*grad
        if 'bcc' in who:
            fluxes.append(J_bcc)
        if 'fcc' in who:
            fluxes.append(J_fcc)
        return fluxes

    def velocity(self, nnodes=3):
        """
        Return interface velocity after solving 'Stefan problem'
        """
        J_bcc, J_fcc = self.flux(nnodes=nnodes)
        self.v = (J_fcc - J_bcc)/(self.y_fcc[0] - self.y_bcc[0])
        return self.v


def merge_domains(*domains):
    """
    Merge domains (FCC and/or BCC objects)
    
    Parameters
    ---------
    *domains : 
        FCC and/or BCC objects to be merged

    Returns
    -------
    zca : tuple
        (z, c, cavg) a tuple containing the arrays for the
        node positions z and the node compositions c, a float for the average
        composition cavg.
    """

    z = [dom.z for dom in domains]
    c = [dom.c for dom in domains]

    L = [dom.L for dom in domains]
    cavg = [dom.get_cavg() for dom in domains]
    cavg = np.average(cavg, weights=L)

    z = np.hstack(z)
    c = np.hstack(c)

    return z, c, cavg