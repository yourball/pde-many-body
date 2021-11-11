from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from six.moves import cPickle as pickle
import time


def fourier_unitary(L):
    k = 2 * np.pi * np.arange(L) / L
    K, X = np.meshgrid(k, np.arange(L))
    U = np.exp(-1j * K * X) / np.sqrt(L)
    return k, U


def save_dict(di_, filename_):
    with open(filename_, "wb") as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, "rb") as f:
        ret_di = pickle.load(f)
    return ret_di


def unitary(H, dt):
    E, Q = np.linalg.eigh(H)
    return np.dot(Q, np.dot(np.diag(np.exp(-1j * E * dt)), Q.T.conj()))


def spdm(H, nu, L):
    E, Q = np.linalg.eigh(H)
    rho = np.dot(Q, np.dot(np.diag(np.arange(L) < int(nu * L)), Q.T.conj()))
    return rho


def spdm_zero_temp(H, nu, L):
    E, Q = np.linalg.eigh(H)
    rho = np.dot(Q, np.dot(np.diag(np.arange(L) < int(nu * L)), Q.T.conj()))
    return rho


def domain_wall_state(L):
    rho = np.diag(np.arange(L) < L // 2).astype(dtype=np.complex)
    return rho


def velocity(rho, L):
    n = 0.5 * (np.roll(np.diag(rho), 1) + np.roll(np.diag(rho), -1)).real
    j = np.zeros(L)
    j[:-1] = (0.5j * (np.diag(rho, +1) - np.diag(rho, -1))).real
    j[-1] = (0.5j * (rho[-1, 0] - rho[0, -1])).real
    return j / (n + 1e-5)


def init_H(J1, J2, L, N, U, bc="periodic"):
    H = J1 * (np.eye(L, L, 1) + np.eye(L, L, -1))
    H += J2 * (np.eye(L, L, 2) + np.eye(L, L, -2))
    if bc == "periodic":
        H[0, L - 1], H[L - 1, 0] = J1, J1
        H[0, L - 2], H[L - 2, 0] = J2, J2
        H[1, L - 1], H[L - 1, 1] = J2, J2
    return H


def plot_spectrum(H, rho):
    k, U = fourier_unitary(len(H))
    occ = np.zeros(len(k))
    for i in range(len(k)):
        psi = U.T[i]
        occ[i] = np.dot(psi.conj(), np.dot(rho, psi)).real

    E = np.zeros(len(k))
    for i in range(len(k)):
        psi = U.T[i]
        E[i] = np.dot(psi.conj(), np.dot(H, psi)).real

    plt.plot(k, E)
    plt.scatter(k, E, s=50 * occ.real)
    plt.show()


# 1d system
def evolve_free_fermions(rho0, H, tmax, dt, bc, plot=False):
    L = H.shape[0]
    assert rho0.shape[0] == L
    # pot = - 0.1 * np.exp(-6.25 * np.linspace(-1, 1, L) ** 2)

    Udt = unitary(H, dt)
    tres = int(tmax / dt)
    n, v = np.zeros([tres, L]), np.zeros([tres, L])
    # Assign rho at t=0
    rho = rho0.copy()
    for ti in range(tres):
        n[ti] = np.diag(rho).real
        v[ti] = velocity(rho, L)
        rho = np.dot(Udt, np.dot(rho, Udt.T.conj()))
    if plot:
        Xm, Tm = np.meshgrid(np.arange(L), np.linspace(0, tmax, tres))
        plt.pcolormesh(Xm, Tm, n, cmap="bwr", rasterized=True)
        plt.title("Free fermions density")
        plt.ylabel("time")
        plt.colorbar()
        plt.draw()
        plt.pause(3)
        plt.close()

    dict_data = {"n": n, "v": v, "x": np.arange(L), "t": np.arange(0, tmax, dt)}
    return n, v, dict_data
