import numpy as np
from six.moves import cPickle as pickle
import matplotlib.colors as colors
from PDE_search import FiniteDiff, TotalFiniteDiff_t
import operator
import functools


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


def scalar_pde_solver(descr, coefs, u0, t, x, bc="periodic", nskip=100, fix_bndry=False, bndry_w=20, bndry_val_l=-0.5, bndry_val_r=0.5):
    """Solver for the first order PDE in form u_t=F(u, u_x, u_xx, ...).
       Assumes scalar complex-valued function u(t,x).
       Builds equation from symbolic description.

       This PDE solver is very basic (explicit Euler method).
       Pros - Provides flexibility of implementation for many types of PDEs.
       Cons - Need extra time steps

       Parameters:
            (complex) coefs -- vector of coefficients multiplying each term in F(.)
            (int) nskip -- number of additional time steps for PDE solver (per dt step)
                           (increases number of steps in vector 't' by a factor 'nskip')
            (complex) u0 -- initial condition for function u(t, x) at t=0

    """
    t = np.linspace(t[0], t[-1], len(t)*nskip)  # Create a refined t-grid
    nt, nx = len(t), len(x)

    u_ev = np.zeros((nt, nx), dtype=complex)
    u = u0

    dt = t[1] - t[0]
    dx = x[1] - x[0]
    for it, t in enumerate(t):
        u_t = np.sum([coefs[i]*get_u_term_from_descr(descr_i, u, x) \
                    for i, descr_i in enumerate(descr)], axis=0)
        u_next = u + dt * u_t
        u_ev[it, :] = u.copy()
        u = u_next
        if fix_bndry:
            u[:bndry_w] = bndry_val_l
            u[-bndry_w:] = bndry_val_r
        if np.isnan(np.sum(u)):
            # Solution exploded, interrupt
            return np.array([np.nan])

    return u_ev[::nskip]


def generalized_euler_solver(descr, coefs, rho0, v0, t, x, bc="periodic", num_integrator_steps=1, fix_vvx_term=True):
    """Solver for Euler hydro system.
       Builds RHS of the Euler equation Dv_t = f(...) from symbolic description.
    """
    t_pde = np.linspace(t[0], t[-1], len(t)*num_integrator_steps)  # Create a refined t-grid
    nt, nx = len(t), len(x)

    rho_ev, v_ev = np.zeros((len(t), nx)), np.zeros((len(t), nx))
    rho, v = rho0, v0

    dt = t_pde[1] - t_pde[0]
    dx = x[1] - x[0]
    for it, t in enumerate(t_pde):
        rhox = FiniteDiff(rho, dx, 1, bc)
        vx = FiniteDiff(v, dx, 1, bc)

        rho_t = -(rhox*v + vx*rho)
        rho_next = rho + dt*rho_t
        # Add RHS terms to Dt(v) = v_t+v*v_x = f (...)
        f = np.sum([coefs[i]*get_euler_term_from_descr(descr_i, rho, v, x) \
                    for i, descr_i in enumerate(descr)], axis=0)
        v_t = f
        if fix_vvx_term:
            v_t -= v*vx

        v_next = v + dt*v_t  # D_t(v) = f(rho, v, ...)

        step = it // num_integrator_steps
        if it % num_integrator_steps == 0:
            rho_ev[step, :] = rho.copy()
            v_ev[step, :] = v.copy()
        rho = rho_next.copy()
        v = v_next.copy()
        if np.isnan(np.sum(rho)):
            # Solution exploded, interrupt
            return np.array([np.nan]), np.array([np.nan])
    return rho_ev, v_ev


def generalized_euler_solver_jrepr(rhot_descr, rhot_coefs, vt_descr, vt_coefs, rho0, j0, t, x, bc="periodic", nskip=1, fix_vvx_term=True):
    """Solver for Euler hydro system in (rho, j) representation
       Builds RHS of the Euler equation j_t = f(...) from symbolic description.
    """
    t_pde = np.linspace(t[0], t[-1], len(t)*nskip)  # Create a refined t-grid
    nt, nx = len(t), len(x)

    rho_ev, v_ev = np.zeros((len(t), nx)), np.zeros((len(t), nx))
    rho, j = rho0, j0

    dt = t_pde[1] - t_pde[0]
    dx = x[1] - x[0]
    for it, t in enumerate(t_pde):
        rhox = FiniteDiff(rho, dx, 1, bc)
        v = j/(rho + 1e-16)
        vx = FiniteDiff(v, dx, 1, bc)
        jx = FiniteDiff(j, dx, 1, bc)
        convect_term = FiniteDiff(rho*v**2, dx, 1, bc)
        g = np.sum([rhot_coefs[i]*get_euler_term_from_descr(descr_i, rho, v, x) \
                    for i, descr_i in enumerate(rhot_descr)], axis=0)
        rho_t = g
        rho_next = rho + dt*rho_t
        # Add RHS terms to Dt(j) = f (rho, j, ...)
        f = np.sum([vt_coefs[i]*get_euler_term_from_descr(descr_i, rho, v, x) \
                    for i, descr_i in enumerate(vt_descr)], axis=0)
        j_t = f
        if fix_vvx_term:
            # \partial_x (rho*v^2)
            j_t -= convect_term

        j_next = j + dt*j_t  # D_t(v) = f(rho, v, ...)

        step = it // nskip
        if it % nskip == 0:
            rho_ev[step, :] = rho.copy()
            v_ev[step, :] = (j/rho).copy()
        rho = rho_next.copy()
        j = j_next.copy()
        # if np.isnan(np.sum(rho)):
        #     # Solution exploded, interrupt
        #     return np.array([np.nan]), np.array([np.nan])

    return rho_ev, v_ev


def get_u_term_from_descr(descr_term, u, x, bc="periodic"):
    """Parse a string representing a given term in PDE.
    For simplicity onsider Euler-like system.
    All terms are combinations of rho, v and their derivatives.

    Notation examples: u_x = \partial_x u
                       u_xx = \partial_xx u
                       u*u_x = u*(\partial_x u)
                       u_x^2 = (\partial_x u)^2
    Supports only * and ^ binary operations
    Returns: np.array(complex) term
    """
    # Split multipliers according to * sign
    multis = descr_term.split('*')
    collection = []
    obs_dict = {'u': u, '': np.ones(len(u))}
    dx = x[1] - x[0]
    for m in multis:
        expr = m.split('^')  # check if have a power sign
        if len(expr) > 1:
            m, pow = expr[0], expr[1]
        elif len(expr) == 1:
            m, pow = expr[0], 1
        else:
            raise ValueError("Found multiple symbols for u, should be one")
        f = m.split('_')
        obs = f[0]
        # assert obs == 'u'
        if len(f) > 1:
            der_d = f[1].count('x')
        if len(f) == 1:
            der_d = 0
        collection.append(FiniteDiff(obs_dict[obs], dx, der_d, bc=bc)**pow)
    term = functools.reduce(operator.mul, collection, 1)
    return term


def get_euler_term_from_descr(descr_term, rho, v, x, bc="periodic"):
    """Parse a string representing a given term in PDE.
    For simplicity onsider Euler-like system.
    All terms are combinations of rho, v and their derivatives.

    Notation: rho_x = partial_x rho
              rho_xx = partial_xx rho
    Supports only multiplication sign, no power
    Returns: term
    """
    # Define dx
    dx = x[1] - x[0]
    # Split multipliers according to * sign
    multis = descr_term.split('*')
    collection = []
    obs_dict = {'rho': rho, 'v': v, 'log(rho)': np.log(rho),
                '1/rho': 1./rho,
                'j': rho*v, 'rhov2': rho*v**2}
    for m in multis:
        # parse ^ symbol
        if '^' in m:
            g, pow = m.split('^')
        else:
            g = m
            pow = 1
        # parse spatial derivatives
        f = g.split('_')
        obs = f[0]
        if len(f) > 1:
            der_d = f[1].count('x')
        if len(f) == 1:
            der_d = 0
        expr = FiniteDiff(obs_dict[obs], dx, der_d, bc=bc)
        expr = expr ** float(pow)
        collection.append(expr)
    term = functools.reduce(operator.mul, collection, 1)
    return term


def extract_v(rho, t, x):
    """Extract velocity from density data using continuity equation"""
    dt, dx = t[1]-t[0], x[1]-x[0]
    rhot = TotalFiniteDiff_t(rho, dt, 1)
    v = np.zeros_like(rho)
    for i in range(rho.shape[1]):
        v[:, i] = - 1./rho[:, i] * dx * np.sum(rhot[:, :i], axis=1)
    return v


# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
