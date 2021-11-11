from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from six.moves import cPickle as pickle
from core import save_dict, load_dict, unitary, spdm


def velocity(rho, L, J1, J2):
    n = 0.5 * (np.roll(np.diag(rho), +1) + np.roll(np.diag(rho), -1)).real
    j = np.zeros(L)
    x = np.arange(L)
    # J1 contribution
    # j[:] = J1*(rho[x, np.roll(x, -1)]-rho[x, np.roll(x, 1)]).imag
    # # # J2 contribution
    # j[:] += J2*(rho[x, np.roll(x, -2)]-rho[x, np.roll(x, 2)]).imag
    j[:-1] = (0.5j * (np.diag(rho, +1) - np.diag(rho, -1))).real
    j[-1] = (0.5j * (rho[-1, 0] - rho[0, -1])).real
    return j / (n + 1e-16)


# 1d system
def plot_free_fermions(J1, J2, L, N, tmax, dt, bc="periodic", U0=0, x0=0, sigma=0.2, ifplot=False, init_state='domain_wall'):
    H = J1 * (np.eye(L, L, 1) + np.eye(L, L, -1))
    H += J2 * (np.eye(L, L, 2) + np.eye(L, L, -2))
    if bc == "periodic":
        H[0, L - 1], H[L - 1, 0] = J1, J1
        H[0, L - 2], H[L - 2, 0] = J2, J2
        H[1, L - 1], H[L - 1, 1] = J2, J2
    nu = N / L
    # Usual potential
    x = np.linspace(-1, 1, L)
    pot = U0 * (np.exp(- ((x-x0) / sigma) ** 2) + np.exp(- ((x+x0) / sigma) ** 2))
    Hw = H + np.diag(pot)
    if init_state == 'gs':
        rho = spdm(Hw, nu, L)
    elif init_state == 'domain_wall':
        rho = domain_wall_state(L)
    else:
        raise NotImplementedError()

    Udt = unitary(H, dt)
    tres = int(tmax / dt)
    n, v = np.zeros([tres, L]), np.zeros([tres, L])
    rho_arr = np.zeros([1, L, L], dtype=np.complex)
    #
    # plt.plot(np.diag(rho).real)
    # plt.show()

    for ti in range(tres):
        n[ti] = np.diag(rho).real
        v[ti] = velocity(rho, L, J1, J2)
        rho = np.dot(Udt, np.dot(rho, Udt.T.conj()))
    rho_arr[0] = rho.copy()
    if ifplot:
        # X,T = np.meshgrid(np.arange(L),np.linspace(0,tmax,tres))
        plt.subplot(1, 3, 1)
        Xm, Tm = np.meshgrid(np.arange(L), np.linspace(0, tmax, tres))
        plt.pcolor(Xm, Tm, n, cmap="bwr")
        plt.title("Free fermions density I")
        # plt.yticks(np.arange(0,tres+1,int(tres/5)),(np.linspace(0,tmax,6)))
        plt.ylabel("time")
        plt.colorbar()
        vF = np.pi * nu
        x = np.arange(L)
        y = (x - int(L / 2)) / vF
        plt.plot(x[y > 0], y[y > 0], c="k", ls="--")
        y = -(x - int(L / 2)) / vF
        plt.plot(x[y > 0], y[y > 0], c="k", ls="--")
        y = -(x - int(3 * L / 2)) / vF
        plt.plot(x[(y > 0) * (y < tmax)], y[(y > 0) * (y < tmax)], c="k", ls="--")
        y = (x + int(L / 2)) / vF
        plt.plot(x[(y > 0) * (y < tmax)], y[(y > 0) * (y < tmax)], c="k", ls="--")
        plt.subplot(1, 3, 2)
        plt.title("Free fermions density II")
        # times = [0, 50, 100]
        # for t in times:
        plt.plot(n[-1])
        plt.subplot(1, 3, 3)
        plt.title("Free fermions velocity")
        plt.pcolor(v, cmap="bwr")
        plt.colorbar()
        plt.show()
    dict_data = {"n": n, "v": v, "x": np.arange(L),
                 "t": np.arange(0, tres*dt, dt), "U": pot, "cdag_c": rho_arr}
    return n, v, dict_data


if __name__ == "__main__":

    L = 1000
    N = 101
    J1, J2 = -0.5, 0.
    tmax = L
    U0 = -0.01
    x0 = .0
    sigma = 0.4
    dt = .5

    bc = 'periodic'
    init_state = 'gs'
    n, v, dict_data = plot_free_fermions(
        J1=J1, J2=J2, L=L, N=N, tmax=tmax, dt=dt, bc=bc, U0=U0, x0=x0, sigma=sigma,
        ifplot=True, init_state=init_state
    )
    fname = f"./data/free_fermions_{init_state}_L={L}_N={N}_J1={J1}_J2={J2}_tmax={tmax}_dt={dt}_U0={U0}_x0={x0}_sigma={sigma}.npy"
    save_dict(dict_data, fname)
    print('Saved results in', fname)
