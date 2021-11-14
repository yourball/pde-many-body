import numpy as np
from numpy import linalg as LA
import scipy.sparse as sparse
from scipy.sparse import csc_matrix
from scipy.sparse import dia_matrix
from scipy import interpolate
import itertools
import operator
import time

"""
Includes fragments of the code from S. Rudy's & N. Kutz paper + my own code
"""


def FiniteDiff(u, dx, d, bc="periodic"):
    """
    Takes dth derivative from 1D array u_i using 2nd order finite difference method (up to d=3)
    Works but with poor accuracy for d > 3

    Input:
    u = data to be differentiated
    dx = Grid spacing.  Assumes uniform spacing
    d = order of the derivative
    bc = boundary condition

    Returns: dth derivative D^n(u)/dx^n.

    # Note that numpy.roll([u], 1) cyclically shifts entries in the array forward
    # numpy.roll([u1, u2, u3, ..], 1) = [uN, u1, u2, u3, ...]
    """

    n = u.size
    ux = np.zeros(n, dtype=np.complex64)
    if d == 0:
        return u

    if d == 1:
        ux = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)

        if bc == "open":
            ux[0] = (-3. / 2 * u[0] + 2 * u[1] - u[2] / 2) / dx
            ux[n - 1] = (3. / 2 * u[n - 1] - 2 * u[n - 2] + u[-3] / 2) / dx
        return ux

    if d == 2:
        ux = (np.roll(u, -1) + np.roll(u, 1) - 2 * u) / dx ** 2

        if bc == "open":
            ux[0] = (2 * u[0] - 5 * u[1] + 4 * u[2] - u[3]) / dx ** 2
            ux[n - 1] = (2 * u[n - 1] - 5 * u[n - 2] + 4 * u[-3] - u[n - 4]) / dx ** 2
        return ux

    if d == 3:
        ux = (
            np.roll(u, -2) / 2 - np.roll(u, -1) + np.roll(u, 1) - np.roll(u, 2) / 2
        ) / dx ** 3

        if bc == "open":
            ux[0] = (
                -2.5 * u[0] + 9 * u[1] - 12 * u[2] + 7 * u[3] - 1.5 * u[4]
            ) / dx ** 3
            ux[1] = (
                -2.5 * u[1] + 9 * u[2] - 12 * u[3] + 7 * u[4] - 1.5 * u[5]
            ) / dx ** 3
            ux[n - 1] = (
                2.5 * u[n - 1]
                - 9 * u[n - 2]
                + 12 * u[-3]
                - 7 * u[n - 4]
                + 1.5 * u[n - 5]
            ) / dx ** 3
            ux[n - 2] = (
                2.5 * u[n - 2]
                - 9 * u[-3]
                + 12 * u[n - 4]
                - 7 * u[n - 5]
                + 1.5 * u[n - 6]
            ) / dx ** 3
        return ux

    if d > 3:
        return FiniteDiff(FiniteDiff(u, dx, 2, bc), dx, d - 2, bc)


def FiniteDiff_t(u, dt, d):
    """
    Takes time derivative using 2nd order finite difference scheme.
    Edges are processed separately by imposing "open boundary" condition
    at t=0 and t=tf.
    This allows to get 2nd order precision vs 1st order precision at the start/end time steps.

    Input:
    u(t_i) = data to be differentiated (1D array)
    dt = time spacing.  Assumes uniform spacing.
    """

    ut = FiniteDiff(u, dt, d, bc="open")
    return ut


def TotalFiniteDiff(u, dx, d, bc="periodic"):
    """Calculate d-th order spatial derivative at all time points"""
    assert len(u.shape) == 2
    m, n = u.shape
    Du = np.zeros((m, n), dtype=u.dtype)
    for i in range(m):
        Du[i, :] = FiniteDiff(u[i, :], dx, d, bc=bc)
    return Du


def TotalFiniteDiff_t(u, dt, d=1, bc=""):
    """Calculate 1st order time derivative at all spatial points"""
    assert len(u.shape) == 2
    m, n = u.shape
    Du = np.zeros((m, n), dtype=u.dtype)
    for i in range(n):
        Du[:, i] = FiniteDiff_t(u[:, i], dt, d)
    return Du


def TotalFourierDiff(u, dx, d):
    """Calculate d-th order spatial derivative at all time points"""
    assert len(u.shape) == 2
    m, n = u.shape
    Du = np.zeros((m, n), dtype=u.dtype)
    for i in range(m):
        Du[i, :] = FourierDiff(u[i, :], dx, d)
    return Du


def FourierDiff(u, dx, d, **kwargs):
    """
    Takes dth derivative in (q) Fourier representation, then converts back to (x) representation

    Input:
    u = data to be differentiated
    dx = x-spacing
    """

    N = len(u)
    q = np.fft.fftfreq(N, dx) * 2 * np.pi
    uhat = np.fft.fft(u)
    ux = np.fft.ifft((1j * q) ** d * uhat)
    return ux


def PolyDiff(u, x, deg=3, diff=1, width=5, bc="open"):

    """
    u = values of some function
    x = x-coordinates where values are known
    deg = degree of polynomial to use
    diff = maximum order derivative we want
    width = width of window to fit to polynomial

    This throws out the data close to the edges since the polynomial derivative only works
    well when we're looking at the middle of the points fit.
    """

    x = x.flatten()
    dx = x[1] - x[0]
    n = len(x)

    # Take the derivatives in the center of the domain
    # Open boundary condition
    if bc == "open":
        du = np.zeros(n - 2 * width)

        for j in range(width, n - width):
            points = np.arange(j - width, j + width)
            # Fit to a Chebyshev polynomial
            # this is the same as any polynomial since we're on a fixed grid but it's better conditioned :)
            poly = np.polynomial.chebyshev.Chebyshev.fit(x[points], u[points], deg)
            # Take derivatives
            du[j - width] = poly.deriv(m=diff)(x[j])

    elif bc == "periodic":
        # Check this code
        du = np.zeros(n)
        for j in range(n):
            # Create extended x vector
            fit_points = np.arange(x[j] - dx * width, x[j] + dx * width, dx)
            u_vals = np.roll(u, j + width)[: 2 * width]
            poly = np.polynomial.chebyshev.Chebyshev.fit(fit_points, u_vals, deg)
            # Take derivatives
            du[j] = poly.deriv(m=diff)(x[j])
    return du


def PolyDiffPoint(u, x, deg=3, diff=1, index=None):

    """
    Same as above but now just looking at a single point

    u = values of some function
    x = x-coordinates where values are known
    deg = degree of polynomial to use
    diff = maximum order derivative we want
    """

    n = len(x)
    if index == None:
        index = (n - 1) / 2

    # Fit to a Chebyshev polynomial
    # better conditioned than normal polynomials
    poly = np.polynomial.chebyshev.Chebyshev.fit(x, u, deg)

    # Take derivatives
    derivatives = []
    for d in range(1, diff + 1):
        derivatives.append(poly.deriv(m=d)(x[index]))

    return derivatives


##################################################################################
##################################################################################
#
# Functions specific to PDE-search
#
##################################################################################
##################################################################################


def build_Theta(
    data,
    derivatives,
    derivatives_description,
    P,
    deriv_max_power=1,
    data_description=None,
):
    """
    builds a matrix with columns representing polynoimials up to degree P of all variables

    This is used when we subsample and take all the derivatives point by point or if there is an
    extra input (Q in the paper) to put in.

    input:
        data: column 0 is U, and columns 1:end are Q
        derivatives: a bunch of derivatives of U and maybe Q, should start with a column of ones
        derivatives_description: description of what derivatives have been passed in
        P: max power of polynomial function of U to be included in Theta

    returns:
        Theta = Theta(U,Q)
        descr = description of what all the columns in Theta are
    """

    n, d = data.shape
    m, d2 = derivatives.shape
    if n != m:
        raise Exception("dimension error")
    if data_description is not None:
        if len(data_description) != d:
            raise Exception("data descrption error")

    # Create a list of all polynomials in d variables up to degree P
    rhs_functions = {}
    f = lambda x, y: np.prod(np.power(list(x), list(y)))
    powers = []
    for p in range(1, P + 1):
        size = d + p - 1
        for indices in itertools.combinations(list(range(size)), d - 1):
            starts = [0] + [index + 1 for index in indices]
            stops = indices + (size,)
            powers.append(tuple(map(operator.sub, stops, starts)))
    for power in powers:
        rhs_functions[power] = [lambda x, y=power: f(x, y), power]

    # First column of Theta is just ones.
    Theta = np.ones((n, 1), dtype=np.complex64)
    descr = [""]

    # Add the derivaitves and their powers into Theta
    for D in range(1, derivatives.shape[1]):
        for p_D in range(1, deriv_max_power + 1):
            Theta = np.hstack([Theta, (derivatives[:, D] ** p_D).reshape(n, 1)])
            if p_D != 1:
                descr.append(derivatives_description[D] + f"^{p_D}")
            else:
                descr.append(derivatives_description[D])

    # Add on derivatives times polynomials
    for D in range(derivatives.shape[1]):
        for k in list(rhs_functions.keys()):
            func = rhs_functions[k][0]
            new_column = np.zeros((n, 1), dtype=np.complex64)
            for i in range(n):
                new_column[i] = func(data[i, :]) * derivatives[i, D]
            Theta = np.hstack([Theta, new_column])
            if data_description is None:
                descr.append(str(rhs_functions[k][1]) + derivatives_description[D])
            else:
                function_description = ""
                for j in range(d):
                    if rhs_functions[k][1][j] != 0:
                        if rhs_functions[k][1][j] == 1:
                            function_description = (
                                function_description + data_description[j]
                            )
                        else:
                            function_description = (
                                function_description
                                + data_description[j]
                                + "^"
                                + str(rhs_functions[k][1][j])
                            )
                descr.append(function_description + derivatives_description[D])

    return Theta, descr


def build_custom_Theta(
    data,
    data_description=[],
    add_constant_term=True,
):
    """
    builds a matrix Theta(U) from a predefined set of terms

    This is used when we subsample and take all the derivatives point by point or if there is an
    extra input to put in.

    input:
        data: column 0 is U
        derivatives_description: description of candidate terms in Theta
        P: max power of polynomial function of U to be included in Theta

    returns:
        Theta = Theta(U,Q)
        descr = description of what all the columns in Theta are
    """
    if len(data) > 0:
        n, m = data.shape

    # Add first column of Theta as ones.
    Theta = np.array([], dtype=np.complex64).reshape((n, 0))
    descr = []
    # Add "u"-part into Theta
    if len(data_description) > 0:
        Theta = np.hstack([Theta, data])
        descr += data_description
    return Theta, descr


def build_linear_system(
    u,
    dt,
    dx,
    D=3,
    P=3,
    time_diff="poly",
    space_diff="poly",
    lam_t=None,
    lam_x=None,
    width_x=None,
    width_t=None,
    deg_x=5,
    deg_t=None,
    sigma=2,
):
    """
    Constructs a large linear system to use in later regression for finding PDE.
    This function works when we are not subsampling the data or adding in any forcing.

    Input:
        Required:
            u = data to be fit to a pde
            dt = temporal grid spacing
            dx = spatial grid spacing
        Optional:
            D = max derivative to include in rhs (default = 3)
            P = max power of u to include in rhs (default = 3)
            time_diff = method for taking time derivative
                        options = 'poly', 'FD', 'FDconv','TV'
                        'poly' (default) = interpolation with polynomial
                        'FD' = standard finite differences
                        'FDconv' = finite differences with convolutional smoothing
                                   before and after along x-axis at each timestep
                        'Tik' = Tikhonov (takes very long time)
            space_diff = same as time_diff with added option, 'Fourier' = differentiation via FFT
            lam_t = penalization for L2 norm of second time derivative
                    only applies if time_diff = 'TV'
                    default = 1.0/(number of timesteps)
            lam_x = penalization for L2 norm of (n+1)st spatial derivative
                    default = 1.0/(number of gridpoints)
            width_x = number of points to use in polynomial interpolation for x derivatives
                      or width of convolutional smoother in x direction if using FDconv
            width_t = number of points to use in polynomial interpolation for t derivatives
            deg_x = degree of polynomial to differentiate x
            deg_t = degree of polynomial to differentiate t
            sigma = standard deviation of gaussian smoother
                    only applies if time_diff = 'FDconv'
                    default = 2
    Output:
        ut = column vector of length u.size
        R = matrix with ((D+1)*(P+1)) of column, each as large as ut
        rhs_description = description of what each column in R is
    """

    n, m = u.shape

    if width_x == None:
        width_x = int(n / 10)
    if width_t == None:
        width_t = int(m / 10)
    if deg_t == None:
        deg_t = deg_x

    # If we're using polynomials to take derviatives, then we toss the data around the edges.
    if time_diff == "poly":
        m2 = m - 2 * width_t
        offset_t = width_t
    else:
        m2 = m
        offset_t = 0
    if space_diff == "poly":
        n2 = n - 2 * width_x
        offset_x = width_x
    else:
        n2 = n
        offset_x = 0

    if lam_t == None:
        lam_t = 1.0 / m
    if lam_x == None:
        lam_x = 1.0 / n

    ########################
    # First take the time derivaitve for the left hand side of the equation
    ########################
    ut = np.zeros((n2, m2), dtype=np.complex64)

    if time_diff == "poly":
        T = np.linspace(0, (m - 1) * dt, m)
        for i in range(n2):
            ut_real = PolyDiff(
                u[i + offset_x, :].real, T, diff=1, width=width_t, deg=deg_t
            )
            ut_imag = PolyDiff(
                u[i + offset_x, :].imag, T, diff=1, width=width_t, deg=deg_t
            )
            ut[i, :] = ut_real + 1j * ut_imag

    else:
        for i in range(n2):
            ut[i, :] = FiniteDiff_t(u[i + offset_x, :], dt, 1)

    ut = np.reshape(ut, (n2 * m2, 1), order="F")

    ########################
    # Now form the rhs one column at a time, and record what each one is
    ########################

    u2 = u[offset_x : n - offset_x, offset_t : m - offset_t]
    Theta = np.zeros((n2 * m2, (D + 1) * (P + 1)), dtype=np.complex64)
    ux = np.zeros((n2, m2), dtype=np.complex64)
    rhs_description = ["" for i in range((D + 1) * (P + 1))]

    if space_diff == "poly":
        Du = {}
        for i in range(m2):
            Du[i] = PolyDiff(
                u[:, i + offset_t],
                np.linspace(0, (n - 1) * dx, n),
                diff=D,
                width=width_x,
                deg=deg_x,
            )
    if space_diff == "Fourier":
        ik = 1j * np.fft.fftfreq(n) * n

    for d in range(D + 1):

        if d > 0:
            for i in range(m2):
                if space_diff == "FD":
                    ux[:, i] = FiniteDiff(u[:, i + offset_t], dx, d)
                elif space_diff == "poly":
                    ux[:, i] = Du[i][:, d - 1]
                elif space_diff == "Fourier":
                    ux[:, i] = np.fft.ifft(ik ** d * np.fft.fft(ux[:, i]))
        else:
            ux = np.ones((n2, m2), dtype=np.complex64)

        for p in range(P + 1):
            Theta[:, d * (P + 1) + p] = np.reshape(
                np.multiply(ux, np.power(u2, p)), (n2 * m2), order="F"
            )

            if p == 1:
                rhs_description[d * (P + 1) + p] = (
                    rhs_description[d * (P + 1) + p] + "u"
                )
            elif p > 1:
                rhs_description[d * (P + 1) + p] = (
                    rhs_description[d * (P + 1) + p] + "u^" + str(p)
                )
            if d > 0:
                rhs_description[d * (P + 1) + p] = (
                    rhs_description[d * (P + 1) + p]
                    + "u_{"
                    + "".join(["x" for _ in range(d)])
                    + "}"
                )

    return ut, Theta, rhs_description


def print_pde(xi, rhs_description, lhs_descr="u_t", verbose=True):
    pde = lhs_descr + " = "
    first = True
    for i in range(len(xi)):
        if xi[i] != 0:
            if not first:
                pde = pde + " + "
            pde = (
                pde
                + "(%05f %+05fi)" % (xi[i].real, xi[i].imag)
                + rhs_description[i]
                + "\n   "
            )
            first = False
    if verbose:
        print(pde)
    return pde


##################################################################################
##################################################################################
#
# Functions for sparse regression.
#
##################################################################################
##################################################################################


def TrainSTRidge(
    R,
    Ut,
    descr,
    lam,
    d_tol,
    maxit=25,
    STR_iters=10,
    l0_penalty=None,
    normalize=2,
    norm_err=2,
    split=0.8,
    print_best_tol=False,
    add_l2_loss=False,
    lhs_descr='ut'
):
    """
    This function trains a predictor using STRidge.

    It runs over different values of tolerance and trains predictors on a training set, then evaluates them
    using a loss function on a holdout set.

    Please note published article has typo.  Loss function used here for model selection evaluates fidelity using 2-norm,
    not squared 2-norm.
    """
    print("TrainSTRidge: start")
    start = time.time()
    train_log = {}
    tol_log = []
    err_log = []
    # Split data into 80% training and 20% test, then search for the best tolderance.
    np.random.seed(0)  # for consistancy
    n, _ = R.shape
    print("Selecting random dataset rows for train")
    train = np.random.choice(n, int(n * split), replace=False)
    print("Use remaining data for validation")
    test = np.setdiff1d(np.arange(n), train)
    TrainR = R[train, :]
    TestR = R[test, :]
    TrainY = Ut[train, :]
    TestY = Ut[test, :]
    D = TrainR.shape[1]

    # Set up the initial tolerance and l0 penalty
    print("Setting up the initial tolerance and l0 penalty ...")
    d_tol = float(d_tol)
    tol = d_tol

    if l0_penalty == None:
        l0_penalty = 0.001 * np.linalg.cond(R)
    print("l0_penalty:", l0_penalty)
    # Get the standard least squares estimator
    xi = np.zeros((D, 1))
    xi_best = np.linalg.lstsq(TrainR, TrainY)[0]
    L_best = np.linalg.norm(
        TestY - TestR.dot(xi_best), 2
    ) + l0_penalty * np.count_nonzero(xi_best)
    if add_l2_loss:
        L_best += lam * np.linalg.norm(xi)

    tol_best = 0

    # Hyperparameter search: l0_penalty
    # Now increase tolerance until test performance decreases
    tol_log = []
    print("Starting STRidge iterations ...")
    for iter in range(maxit):
        # Get a set of coefficients and error
        xi = STRidge(R, Ut, lam, STR_iters, tol, normalize=normalize)
        L_curr = np.linalg.norm(
            TestY - TestR.dot(xi), norm_err
        ) + l0_penalty * np.count_nonzero(xi)

        if add_l2_loss:
            L_curr += lam * np.linalg.norm(xi)
        # Has the accuracy improved?
        if L_curr < L_best:
            L_best = L_curr
            xi_best = xi
            tol_best = tol
            tol = tol + d_tol

        else:
            tol = max([0, tol - 2 * d_tol])
            d_tol = 2 * d_tol / (maxit - iter)
            tol = tol + d_tol

        err_log.append(L_curr)
        tol_log.append(tol)
        if print_best_tol:
            print(
                f"Tol: {tol}, Optimal tolerance: {tol_best}, Obj function best: {L_best}, l0_pen: {l0_penalty}, xi: {xi}"
            )

    train_log["best_xi"] = xi_best
    print('Best PDE found with STRidge:')
    print_pde(xi_best, descr, lhs_descr=lhs_descr)
    end = time.time()
    print("Time elapsed (s):", end - start)
    return xi_best, train_log


def Lasso(X0, Y, lam, xi=np.array([0]), maxit=100, normalize=2):
    """
    Uses accelerated proximal gradient (FISTA) to solve Lasso
    argmin (1/2)*||X*xi-Y||_2^2 + lam||xi||_1
    """

    # Obtain size of X
    n, d = X0.shape
    X = np.zeros((n, d), dtype=np.complex64)
    Y = Y.reshape(n, 1)

    # Create w if none is given
    if xi.size != d:
        xi= np.zeros((d, 1), dtype=np.complex64)
    xi_old = np.zeros((d, 1), dtype=np.complex64)

    # Initialize a few other parameters
    converge = 0
    objective = np.zeros((maxit, 1))

    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d, 1))
        for i in range(0, d):
            Mreg[i] = 1.0 / (np.linalg.norm(X0[:, i], normalize))
            X[:, i] = Mreg[i] * X0[:, i]
    else:
        X = X0

    # Lipschitz constant of gradient of smooth part of loss function
    L = np.linalg.norm(X.T.dot(X), 2)

    # Now loop until converged or max iterations
    for iters in range(0, maxit):

        # Update w
        z = xi+ iters / float(iters + 1) * (xi- xi_old)
        xi_old = xi
        z = z - X.T.dot(X.dot(z) - Y) / L
        for j in range(d):
            xi[j] = np.multiply(np.sign(z[j]), np.max([abs(z[j]) - lam / L, 0]))

        # Could put in some sort of break condition based on convergence here.

    # Now that we have the sparsity pattern, used least squares.
    biginds = np.where(xi!= 0)[0]
    if biginds != []:
        xi[biginds] = np.linalg.lstsq(X[:, biginds], Y)[0]

    # Finally, reverse the regularization so as to be able to use with raw data
    if normalize != 0:
        return np.multiply(Mreg, xi)
    else:
        return w


def ElasticNet(X0, Y, lam1, lam2, xi=np.array([0]), maxit=100, normalize=2):
    """
    Uses accelerated proximal gradient (FISTA) to solve elastic net
    argmin (1/2)*||X*xi-Y||_2^2 + lam_1||xi||_1 + (1/2)*lam_2||xi||_2^2
    """

    # Obtain size of X
    n, d = X0.shape
    X = np.zeros((n, d), dtype=np.complex64)
    Y = Y.reshape(n, 1)

    # Create xi if none is given
    if xi.size != d:
        xi = np.zeros((d, 1), dtype=np.complex64)
    xi_old = np.zeros((d, 1), dtype=np.complex64)

    # Initialize a few other parameters
    converge = 0
    objective = np.zeros((maxit, 1))

    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d, 1))
        for i in range(0, d):
            Mreg[i] = 1.0 / (np.linalg.norm(X0[:, i], normalize))
            X[:, i] = Mreg[i] * X0[:, i]
    else:
        X = X0

    # Lipschitz constant of gradient of smooth part of loss function
    L = np.linalg.norm(X.T.dot(X), 2) + lam2

    # Now loop until converged or max iterations
    for iters in range(0, maxit):

        # Update w
        z = xi+ iters / float(iters + 1) * (xi- xi_old)
        xi_old = xi
        z = z - (lam2 * z + X.T.dot(X.dot(z) - Y)) / L
        for j in range(d):
            xi[j] = np.multiply(np.sign(z[j]), np.max([abs(z[j]) - lam1 / L, 0]))

        # Could put in some sort of break condition based on convergence here.

    # Now that we have the sparsity pattern, used least squares.
    biginds = np.where(xi!= 0)[0]
    if biginds != []:
        xi[biginds] = np.linalg.lstsq(X[:, biginds], Y)[0]

    # Finally, reverse the regularization so as to be able to use with raw data
    if normalize != 0:
        return np.multiply(Mreg, xi)
    else:
        return xi


def STRidge(X0, y, lam, maxit, tol, normalize=2, print_results=False):
    """
    Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse
    approximation to X^{-1}y.  The idea is that this may do better with correlated observables.

    This assumes y is only one column
    """

    n, d = X0.shape
    X = np.zeros((n, d), dtype=np.complex64)
    xi_err = np.zeros(d)
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d, 1))
        for i in range(0, d):
            Mreg[i] = 1.0 / (np.linalg.norm(X0[:, i], normalize))
            X[:, i] = Mreg[i] * X0[:, i]
    else:
        X = X0

    # Get the standard ridge esitmate
    if lam != 0:
        xi_fit = np.linalg.lstsq(X.T.dot(X) + lam * np.eye(d), X.T.dot(y))
        xi, xi_err = xi_fit[0], xi_fit[1]
    else:
        xi_fit = np.linalg.lstsq(X, y)
        xi, xi_err = xi_fit[0], xi_fit[1]
    num_relevant = d
    # print(f'abs(xi): {np.abs(xi)}, tol: {tol}')
    biginds = np.where(np.abs(xi) > tol)[0]

    # Threshold and continue
    for j in range(maxit):

        # Figure out which items to cut out
        smallinds = np.where(np.abs(xi) < tol)[0]
        new_biginds = [i for i in range(d) if i not in smallinds]

        # If nothing changes then stop
        if num_relevant == len(new_biginds):
            break
        else:
            num_relevant = len(new_biginds)

        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0:
                if print_results:
                    print("Tolerance too high - all coefficients set below tolerance")
                return xi
            else:
                break
        biginds = new_biginds

        # Otherwise get a new guess
        xi[smallinds] = 0
        if lam != 0:
            xi_fit = np.linalg.lstsq(
                X[:, biginds].T.dot(X[:, biginds]) + lam * np.eye(len(biginds)),
                X[:, biginds].T.dot(y),
            )
            xi[biginds], _ = xi_fit[0], xi_fit[1]
            # print('w_fit', w_fit)
        else:
            xi_fit = np.linalg.lstsq(X[:, biginds], y)
            xi[biginds], _ = xi_fit[0], xi_fit[1]

    # Now that we have the sparsity pattern, use standard least squares to get w
    if biginds != []:
        xi_fit = np.linalg.lstsq(X[:, biginds], y)
        xi[biginds], _ = xi_fit[0], xi_fit[1]

    if normalize != 0:
        return np.multiply(Mreg, xi)
    else:
        return xi


def BruteForceL0(
    R,
    Ut,
    descr,
    lam_l2=0,
    l0_penalty=None,
    split=0.8,
    verbose=False,
    seed=0,
    add_l2_loss=False,
    lhs_descr='u_t'
):
    """
    This function uses Brute force search over subsets of all possible 2^M combinations of terms to find the best PDE.
    We use
    """
    start = time.time()
    # Split data into 80% training and 20% test, then search for the best tolderance.
    np.random.seed(seed)  # for consistancy
    n, _ = R.shape
    train = np.random.choice(n, int(n * split), replace=False)
    test = np.setdiff1d(np.arange(n), train)
    TrainR = R[train, :]
    TestR = R[test, :]
    TrainY = Ut[train, :]
    TestY = Ut[test, :]
    X = TrainR
    D = TrainR.shape[1]

    train_log = {}
    # assert D <= 15, "Brute force search is supported for <= 15 candidate terms"
    if l0_penalty is None:
        l0_penalty = 0.001 * np.linalg.cond(R)
    if verbose:
        print("Brute Force search with l0_penalty", l0_penalty)
    # Get the standard least squares estimator
    xi_best = np.linalg.lstsq(X, TrainY)[0]
    L_best = np.linalg.norm(
        TestY - TestR.dot(xi_best), 2
    ) + l0_penalty * np.count_nonzero(xi_best)
    if add_l2_loss:
        L_best += lam_l2 * np.linalg.norm(xi_best)

    best_pde = print_pde(xi_best, descr, lhs_descr=lhs_descr, verbose=False)

    for i, indexes in enumerate(itertools.product([False, True], repeat=D)):
        if np.all(1 - np.array(indexes)):  # all indexes == False (no terms in fit)
            continue
        if verbose:
            print(f'iteration {i} from total {2**D}')
        # Accelerated BruteForce: if L0 penalty term is larger than the L_best, then skipping
        if np.count_nonzero(indexes) * l0_penalty > L_best:
            continue
        # iterate over all possible subsets of terms
        xi = np.zeros((D, 1), dtype=np.complex64)
        X = TrainR[:, list(indexes)]
        non_zero_xi = np.count_nonzero(xi)

        if lam_l2 != 0:
            # import pdb; pdb.set_trace()
            xi[list(indexes)] = np.linalg.lstsq(
                X.T.dot(X) + lam_l2 * np.eye(X.shape[1]), X.T.dot(TrainY)
            )[0]
            L_curr = np.linalg.norm(
                TestY - TestR.dot(xi), 2
            ) + l0_penalty * np.count_nonzero(xi)
        else:
            xi[list(indexes)] = np.linalg.lstsq(TrainR[:, list(indexes)], TrainY)[0]
            L_curr = np.linalg.norm(
                TestY - TestR.dot(xi), 2
            ) + l0_penalty * np.count_nonzero(xi)
        if add_l2_loss:
            L_curr += lam_l2 * np.linalg.norm(xi)

        # Has the accuracy improved?
        if L_curr < L_best:
            L_best = L_curr
            xi_best = xi
            best_pde = print_pde(xi_best, descr, lhs_descr=lhs_descr, verbose=False)

        if verbose:
            print("Current PDE")
            print_pde(xi, descr)
            print(
                f"Obj function {L_curr}, Obj function best: {L_best}, l0_penalty: {l0_penalty}, pde: "
            )

    train_log["err_test"] = L_best - l0_penalty * np.count_nonzero(xi_best)
    train_log["err_train"] = np.linalg.norm(TrainY - TrainR.dot(xi_best))
    train_log["best_pde"] = best_pde
    train_log["best_xi"] = xi_best
    print('Best PDE found with BruteForce:')
    print_pde(xi_best, descr, lhs_descr=lhs_descr)
    end = time.time()
    print("Time elapsed (s):", end - start)
    return xi_best, train_log


def BruteForceL0Scan(
    R, Ut, descr, l0_init=1e-8, l0_fin=0.1, lam_l2=1e-5, num_points=10, verbose=False
):
    l0_arr = np.logspace(np.log10(l0_init), np.log10(l0_fin), num=num_points)
    train_data = []
    results = {"l0_arr": l0_arr, "train_data": train_data}
    for l0 in l0_arr:
        print('l0', l0)
        xi_best, train_log = BruteForceL0(R, Ut, descr, l0_penalty=l0, lam_l2=lam_l2, verbose=verbose)
        train_data.append(train_log)
    return results


def STRidgeScan(
    R,
    Ut,
    descr,
    d_tol=10,
    l0_init=1e-8,
    l0_fin=0.1,
    lam_l2=1e-5,
    num_points=10,
    verbose=False,
):
    l0_arr = np.logspace(np.log10(l0_init), np.log10(l0_fin), num=num_points)
    train_data = []
    results = {"l0_arr": l0_arr, "train_data": train_data}
    for l0 in l0_arr:
        xi_best, train_log = TrainSTRidge(R, Ut, lam_l2, d_tol, l0_penalty=l0)
        train_data.append(train_log)
    return results


class SoftmaxActionPolicy(object):
    def __init__(self, theta):
        self.theta = theta

    def act(self):
        """ Sample vector of terms from Bernoulli distr with p=probs
        """
        logits = np.exp(self.theta)
        probs = logits / np.sum(logits)
        num_terms = len(self.theta)
        terms = np.array(
            [np.random.choice(2, p=[1 - probs[i], probs[i]]) for i in range(num_terms)]
        )
        terms = np.where(terms == 1)[0]
        return terms


class CEM:
    def __init__(
        self,
        data_dict,
        descr,
        n_iter=8,
        batch_size=100,
        elite_frac=0.01,
        l0_penalty=1e-2,
        num_rollouts=100,
        l2_penalty=0,
    ):
        self.data_dict = data_dict
        self.descr = descr
        self.num_terms = len(descr)
        thetas_init = np.zeros(self.num_terms)
        self.agent = SoftmaxActionPolicy(thetas_init)
        self.train_params = dict(
            n_iter=n_iter,
            batch_size=batch_size,
            elite_frac=elite_frac,
            l0_penalty=l0_penalty,
            l2_penalty=l2_penalty,
            num_rollouts=num_rollouts,
        )
        self.train_info = {}

    def min_loss(self, theta, l0_penalty, l2_penalty, num_rollouts):
        agent = SoftmaxActionPolicy(theta)
        err_list = []
        terms_list = []

        ut = self.data_dict["ut"]
        X = self.data_dict["X"]
        for j in range(num_rollouts):
            terms = agent.act()
            terms_list.append(terms)
            xi = np.linalg.lstsq(X[:, terms], ut)[0]
            L_curr= (
                np.linalg.norm(np.dot(X[:, terms], xi) - ut)
                + l0_penalty * np.count_nonzero(xi)
                + l2_penalty * np.linalg.norm(xi)
            )
            err_list.append(err)

        best_indx = np.argmin(err_list)
        return err_list[best_indx], terms_list[best_indx]

    def iter_train(
        self,
        num_terms,
        batch_size,
        n_iter,
        elite_frac,
        l0_penalty,
        l2_penalty,
        num_rollouts,
    ):
        """
        Generic implementation of the cross-entropy method for maximizing a black-box function
        elite_frac: in each batch select this fraction of the top-performing samples
        """
        n_elite = int(batch_size * elite_frac)
        th_mean = np.zeros(num_terms)
        th_std = np.ones_like(th_mean)
        for it in range(n_iter):
            print('iter', it)
            ths = np.array(
                [
                    th_mean + dth
                    for dth in th_std[:] * np.random.randn(batch_size, th_mean.size)
                ]
            )
            ys = np.array(
                [
                    self.min_loss(th, l0_penalty, l2_penalty, num_rollouts)[0]
                    for th in ths
                ]
            )
            elite_inds = ys.argsort()[:n_elite]
            elite_ths = ths[elite_inds]
            th_mean = elite_ths.mean(axis=0)
            th_std = elite_ths.std(axis=0)
            print("elite_ths", elite_ths)
            yield {"ys": ys, "theta_mean": th_mean, "y_mean": ys.mean()}

    def train(self):
        np.random.seed(1)
        best_loss = np.inf
        loss_arr, terms_arr, theta_mean = [], [], []
        self.train_info = {
            "loss": loss_arr,
            "terms": terms_arr,
            "theta_mean": theta_mean,
        }
        X = self.data_dict["X"]
        ut = self.data_dict["ut"]
        # Train the agent, and snapshot each stage
        for (i, iterdata) in enumerate(
            self.iter_train(self.num_terms, **self.train_params)
        ):
            print(("Iteration %2i. Episode mean loss: %7.3f" % (i, iterdata["y_mean"])))
            theta_mean.append(iterdata["theta_mean"])
            self.agent = SoftmaxActionPolicy(iterdata["theta_mean"])
            loss, terms = self.min_loss(
                self.agent.theta,
                self.train_params["l0_penalty"],
                self.train_params["l2_penalty"],
                self.train_params["num_rollouts"],
            )
            loss_arr.append(loss)
            terms_arr.append(terms)
            if loss < best_loss:
                best_terms = terms.copy()
                best_loss = loss
            xi = np.zeros((X.shape[1], 1), dtype=np.complex)
            xi[best_terms] = np.linalg.lstsq(X[:, best_terms], ut)[0]
            print("best terms", best_terms)
            print("best loss", best_loss)
            print_pde(xi, self.descr, "u_t")
