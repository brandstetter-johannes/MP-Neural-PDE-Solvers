"""
In this file, we will reproduce table 1 of 
"Message Passing Neural PDE Solvers", see https://arxiv.org/abs/2202.03376.

We are interested in solving the 1D Burgers' equation with forcing and diffusion.

We will use a finite-volume method. The flux will be the "WENO5" flux, though
the "Godunov" flux function is simpler and would work fine as well. We'll use periodic
boundary conditions.

We will use explicit timestepping, in particular SSPRK3 timestepping, see
https://gkeyll.readthedocs.io/en/latest/dev/ssp-rk.html or
https://www.cfm.brown.edu/people/sg/SSPlinear.pdf.

The timestep is given by min(dt_cfl, dt_diff) where dt_cfl and dt_diff
are the timestep due to the CFL condition (see https://en.wikipedia.org/wiki/Courant–Friedrichs–Lewy_condition)
and the diffusion term with explicit timestepping (see https://en.wikipedia.org/wiki/FTCS_scheme).

While implicit timestepping can sometimes be better for diffusive problems, for problems with a 
combination of advection and diffusion, explicit timestepping is often superior. In this case,
explicit timestepping is vastly superior because (a) it allows us to take advantage of optimized
JIT-compiled vector algebra and (b) the allowed timestep is larger. 

We will reproduce E1 and E2 but not E3. We average over 10 initializations.
"""

###############################################
# Hyperparameters that apply to both E1 and E2
###############################################

L = 16
J = 5
amplitude_max = 0.5
omega_max = 0.4
l_min = 1
l_max = 3
num_timesteps_eval = 250
T_final = 4.0
nxs = [40, 50, 100]
nx_exact = 200
n_test_runtime = 10
n_test_table1 = 10

colors = ["red", "green", "blue"]

###############################################

import jax.numpy as jnp
import numpy as onp
from jax import vmap, jit, random
from jax.lax import scan
from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)
from time import time
import matplotlib.pyplot as plt
PI = jnp.pi


def ssp_rk3(a_n, t_n, F, dt):
    """
    This is our explicit Runge-Kutta timestepping scheme.
    """
    a_1 = a_n + dt * F(a_n, t_n)
    a_2 = 0.75 * a_n + 0.25 * (a_1 + dt * F(a_1, t_n + dt))
    a_3 = 1 / 3 * a_n + 2 / 3 * (a_2 + dt * F(a_2, dt + dt / 2))
    return a_3, t_n + dt


def quadrature(f, a, b, n=1):
    """
    This function integrates a scalar function f(x) from x=a to x=b.
    It uses Gaussian Quadrature of order (2n-1), 
    see https://en.wikipedia.org/wiki/Gaussian_quadrature.

    Default is n=1, 1st-order quadrature.
    """
    assert isinstance(n, int) and n <= 8 and n > 0
    w = {
        1: jnp.asarray([2.0]),
        2: jnp.asarray([1.0, 1.0]),
        3: jnp.asarray(
            [
                0.5555555555555555555556,
                0.8888888888888888888889,
                0.555555555555555555556,
            ]
        ),
        4: jnp.asarray(
            [
                0.3478548451374538573731,
                0.6521451548625461426269,
                0.6521451548625461426269,
                0.3478548451374538573731,
            ]
        ),
        5: jnp.asarray(
            [
                0.2369268850561890875143,
                0.4786286704993664680413,
                0.5688888888888888888889,
                0.4786286704993664680413,
                0.2369268850561890875143,
            ]
        ),
        6: jnp.asarray(
            [
                0.1713244923791703450403,
                0.3607615730481386075698,
                0.4679139345726910473899,
                0.4679139345726910473899,
                0.3607615730481386075698,
                0.1713244923791703450403,
            ]
        ),
        7: jnp.asarray(
            [
                0.1294849661688696932706,
                0.2797053914892766679015,
                0.38183005050511894495,
                0.417959183673469387755,
                0.38183005050511894495,
                0.279705391489276667901,
                0.129484966168869693271,
            ]
        ),
        8: jnp.asarray(
            [
                0.1012285362903762591525,
                0.2223810344533744705444,
                0.313706645877887287338,
                0.3626837833783619829652,
                0.3626837833783619829652,
                0.313706645877887287338,
                0.222381034453374470544,
                0.1012285362903762591525,
            ]
        ),
    }[n]

    xi_i = {
        1: jnp.asarray([0.0]),
        2: jnp.asarray([-0.5773502691896257645092, 0.5773502691896257645092]),
        3: jnp.asarray([-0.7745966692414833770359, 0.0, 0.7745966692414833770359]),
        4: jnp.asarray(
            [
                -0.861136311594052575224,
                -0.3399810435848562648027,
                0.3399810435848562648027,
                0.861136311594052575224,
            ]
        ),
        5: jnp.asarray(
            [
                -0.9061798459386639927976,
                -0.5384693101056830910363,
                0.0,
                0.5384693101056830910363,
                0.9061798459386639927976,
            ]
        ),
        6: jnp.asarray(
            [
                -0.9324695142031520278123,
                -0.661209386466264513661,
                -0.2386191860831969086305,
                0.238619186083196908631,
                0.661209386466264513661,
                0.9324695142031520278123,
            ]
        ),
        7: jnp.asarray(
            [
                -0.9491079123427585245262,
                -0.7415311855993944398639,
                -0.4058451513773971669066,
                0.0,
                0.4058451513773971669066,
                0.7415311855993944398639,
                0.9491079123427585245262,
            ]
        ),
        8: jnp.asarray(
            [
                -0.9602898564975362316836,
                -0.7966664774136267395916,
                -0.5255324099163289858177,
                -0.1834346424956498049395,
                0.1834346424956498049395,
                0.5255324099163289858177,
                0.7966664774136267395916,
                0.9602898564975362316836,
            ]
        ),
    }[n]

    x_i = (b + a) / 2 + (b - a) / 2 * xi_i
    wprime = w * (b - a) / 2
    return jnp.sum(wprime * vmap(f)(x_i))


def integrate_function_fv(f, t, nx, n=1):
    """
    Takes a scalar function f(x,t) and outputs
    an integral representation of the function
    using an n-point Gaussian quadrature at time t.
    The integral representation is over each of the nx
    cells and results in a vector of shape (nx,).
    """
    cell_boundaries = jnp.linspace(0, L, nx+1)
    cell_boundaries_L = cell_boundaries[:-1]
    cell_boundaries_R = cell_boundaries[1:]

    quad_vmap = vmap(lambda a, b: quadrature(lambda x: f(x,t), a, b, n=n))
    return quad_vmap(cell_boundaries_L, cell_boundaries_R)


def forcing_func(key):
    """
    This is the forcing function in equation (13) of the paper.
    """
    key1, key2, key3, key4 = random.split(key, 4)
    phases = random.uniform(key1, (J,)) * 2 * PI
    ls = random.randint(key2, (J,), l_min, l_max + 1)
    amplitudes = (random.uniform(key3, (J,)) - 0.5) * 2 * amplitude_max
    omegas = (random.uniform(key4, (J,)) - 0.5) * 2 * omega_max

    def sum_modes(x, t):
        return jnp.sum(amplitudes * jnp.sin(2 * PI * ls / L * x + omegas * t + phases))

    return sum_modes


def _scan(sol, x, rk_F):
    a, t = sol
    a_f, t_f = rk_F(a, t)
    return (a_f, t_f), None


def _scan_output(sol, x, rk_F):
    a, t = sol
    a_f, t_f = rk_F(a, t)
    return (a_f, t_f), a


def f(u):
    """ This is the Burgers' equation flux function. """
    return u**2/2

def _godunov_flux_1D_burgers(a):
    """ 
    Here I use the 'Godunov' flux to compute the flux at the right cell boundary. 
    """
    a = jnp.pad(a, ((0, 1)), "wrap")
    u_left = a[:-1]
    u_right = a[1:]
    zero_out = 0.5 * jnp.abs(jnp.sign(u_left) + jnp.sign(u_right))
    compare = jnp.less(u_left, u_right)
    return compare * zero_out * jnp.minimum(F(u_left), F(u_right)) + (
        1 - compare
    ) * jnp.maximum(f(u_left), f(u_right))



def _weno_flux_1D_burgers(a):
    """ 
    Here I use the 'WENO' flux to compute the flux at the right cell boundary. 
    """
    epsilon = 1e-6
    d0 = 1/10
    d1 = 6/10
    d2 = 3/10
    a_minus2 = jnp.roll(a, 2)
    a_minus1 = jnp.roll(a, 1)
    a_plus1 = jnp.roll(a, -1)
    a_plus2 = jnp.roll(a, -2)
    a_plus3 = jnp.roll(a, -3)

    f_a_minus2 = f(a_minus2)
    f_a_minus1 = f(a_minus1)
    f_a = f(a)
    f_a_plus1 = f(a_plus1)
    f_a_plus2 = f(a_plus2)
    f_a_plus3 = f(a_plus3)

    # Moving to right, a > 0, f_plus
    f0 = (2/6) * f_a_minus2 - (7/6) * f_a_minus1 + (11/6) * f_a
    f1 = (-1/6) * f_a_minus1 + (5/6) * f_a + (2/6) * f_a_plus1
    f2 = (2/6) * f_a + (5/6) * f_a_plus1 + (-1/6) * f_a_plus2
    beta0 = (13/12) * (f_a_minus2 - 2 * f_a_minus1 + f_a)**2 + (1/4) * (f_a_minus2 - 4 * f_a_minus1 + 3 * f_a)**2
    beta1 = (13/12) * (f_a_minus1 - 2 * f_a  + f_a_plus1)**2 + (1/4) * (- f_a_minus1 + f_a_plus1)**2
    beta2 = (13/12) * (f_a - 2 * f_a_plus1  + f_a_plus2)**2 + (1/4) * (3 * f_a - 4 * f_a_plus1  + f_a_plus2)**2
    alpha0 = d0 / (epsilon + beta0)**2
    alpha1 = d1 / (epsilon + beta1)**2
    alpha2 = d2 / (epsilon + beta2)**2
    f_plus = (alpha0 * f0 + alpha1 * f1 + alpha2 * f2) / (alpha0 + alpha1 + alpha2)

    

    # Moving to left, a < 0, f_minus
    f0 = (2/6)  * f_a_plus3 - (7/6) * f_a_plus2 + (11/6) * f_a_plus1
    f1 = (-1/6) * f_a_plus2 + (5/6) * f_a_plus1 + (2/6)  * f_a
    f2 = (2/6)  * f_a_plus1 + (5/6) * f_a       + (-1/6) * f_a_minus1
    beta0 = (13/12) * (f_a_plus3 - 2 * f_a_plus2 + f_a_plus1)**2  + (1/4) * (     f_a_plus3 - 4 * f_a_plus2  + 3 * f_a_plus1)**2
    beta1 = (13/12) * (f_a_plus2 - 2 * f_a_plus1 + f_a)**2        + (1/4) * ( -   f_a_plus2                  +     f_a)**2
    beta2 = (13/12) * (f_a_plus1 - 2 * f_a       + f_a_minus1)**2 + (1/4) * ( 3 * f_a_plus1 - 4 * f_a        +     f_a_minus1)**2
    alpha0 = d0 / (epsilon + beta0)**2
    alpha1 = d1 / (epsilon + beta1)**2
    alpha2 = d2 / (epsilon + beta2)**2
    f_minus = (alpha0 * f0 + alpha1 * f1 + alpha2 * f2) / (alpha0 + alpha1 + alpha2)

    compare = jnp.less(a, a_plus1)
    zero_out = 0.5 * jnp.abs(jnp.sign(a) + jnp.sign(a_plus1))
    return compare * zero_out * jnp.minimum(f_minus, f_plus) + (
        1 - compare
    ) * jnp.maximum(f_minus, f_plus)


def _diffusion_term_1D_burgers(a, dx, nu):
    """
    The approximation of the diffusion term is
    nu * (u_{j+1} - 2 * u_j + u_{j-1}) / (dx)**2
    """
    return nu * (jnp.roll(a, -1) + jnp.roll(a, 1) - 2 * a) / dx


def time_derivative_1D_burgers(
    a, t, dx, flux, nu, forcing_func=None,
):
    """
    We are solving the 1D Burgers' equation du/dt + u^2/2 = F(x,t) + nu * d^2u/dx^2.
    Here we compute the time-derivative at time t if the solution is given by 
    the vector 'a' of shape (nx,).
    """

    if flux == "godunov":
        flux_right = _godunov_flux_1D_burgers(a)
    elif flux == "weno":
        flux_right = _weno_flux_1D_burgers(a)
    else:
        raise Exception

    if nu is not None:
        dif_term = _diffusion_term_1D_burgers(a, dx, nu)
    else:
        dif_term = 0.0

    nx = a.shape[0]
    forcing_term = integrate_function_fv(forcing_func, t, nx)

    flux_left = jnp.roll(flux_right, 1)
    flux_term = (flux_left - flux_right)
    return (flux_term + dif_term + forcing_term) / dx
    
def simulate_1D(
    a0,
    t0,
    dx,
    dt,
    nt,
    nu = 0.0,
    output=False,
    forcing_func=None,
    rk=ssp_rk3,
    flux="godunov"
):
    dadt = lambda a, t: time_derivative_1D_burgers(
        a,
        t,
        dx,
        flux,
        nu,
        forcing_func=forcing_func,
    )

    rk_F = lambda a, t: rk(a, t, dadt, dt)

    if output:
        scanf = jit(lambda sol, x: _scan_output(sol, x, rk_F))
        _, data = scan(scanf, (a0, t0), None, length=nt)
        return data
    else:
        scanf = jit(lambda sol, x: _scan(sol, x, rk_F))
        (a_f, t_f), _ = scan(scanf, (a0, t0), None, length=nt)
        return a_f


def get_dt_nt(Tf, nu, nx, max_u, cfl_safety, diff_safety):
    """
    This function returns the number of timesteps nt and
    the timestep dt for a given integration time Tf, diffusion
    coefficient nu, number of gridpoints nx, maximum value of
    the Burgers' solution, and dimensionless scalar values representing
    the cfl number and diffusion number for explicit timestepping.
    """
    dx = L / nx
    dt_cfl = cfl_safety * dx / max_u
    if nu is not None and nu > 0.0:
        dt_diffusion = diff_safety * dx**2 / nu
        dt = jnp.minimum(dt_cfl, dt_diffusion)
    else:
        dt = dt_cfl

    if nx == nx_exact:
        nt = 320
    else:
        nt = int(((Tf / dt)) // 20 + 1) * 20

    return Tf/nt, nt


def accumulated_mse(traj_1, traj_2):
    """
    traj_1 and traj_2 are both of shape (nt, nx).
    The accumulated error is the sum over the 0th axis 
    and the mean squared error over the 1th axis.
    """
    return jnp.sum(jnp.mean((traj_1-traj_2)**2,axis=1),axis=0)

 
def plot_subfig(a, subfig, color="blue", linewidth=0.5, linestyle="solid", label=None):
    """
    This is a plotting function. a is a vector of shape (nx,)
    """

    nx = a.shape[0]
    dx = L / nx
    xjs = jnp.arange(nx) * L / nx
    xs = xjs[None, :] + jnp.linspace(0.0, dx, 2)[:, None]
    subfig.plot(
        xs.T.reshape(-1),
        jnp.repeat(a, 2),
        color=color,
        linewidth=linewidth,
        label=label,
        linestyle=linestyle,
    )
    return




###################################################################################################################
###################################################################################################################
# Experiment 1 (E1)
print("We are now beginning experiment 1.")
###################################################################################################################
###################################################################################################################
fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, squeeze=True, figsize=(8, 5))

nu = None
flux="weno"
max_u = 3.0
cfl_safety = 1.0
diff_safety = 0.5

#######################
# Test Runtime
#######################

t0 = 0.0
key = random.PRNGKey(4)
f_forcing = forcing_func(key)
f_init = lambda x, t: f_forcing(x, t0)

for i, nx in enumerate(nxs):
    dx = L/nx
    a0 = integrate_function_fv(f_init, t0, nx) / dx


    @partial(jit, static_argnums=(3,))
    def sim(a0, t0, dt, nt):
        return simulate_1D(a0, t0, dx, dt, nt, nu, forcing_func=f_forcing, flux=flux)

    dt, nt = get_dt_nt(T_final, nu, nx, max_u, cfl_safety, diff_safety)
    af = sim(a0, t0, dt, nt).block_until_ready()
    ti = time()
    for _ in range(n_test_runtime):
        af = sim(a0, t0, dt, nt).block_until_ready()
    tf = time()
    print("E1 WENO5: nx = {}, Runtime = {:.5f}".format(nx, (tf-ti)/n_test_runtime))



#######################
# Accumulate Errors
#######################

accumulated_errors = onp.zeros(len(nxs))

@partial(jit, static_argnums=(1,))
def get_trajectory(key, nx):
    f_forcing = forcing_func(key)
    f_init = lambda x, t: f_forcing(x, t0)
    dx = L/nx
    a0 = integrate_function_fv(f_init, t0, nx) / dx
    dt, nt = get_dt_nt(T_final, nu, nx, max_u, cfl_safety, diff_safety)
    return simulate_1D(a0, t0, dx, dt, nt, nu, output=True, forcing_func=f_forcing, flux=flux)


for n in range(n_test_table1):

    key, _ = random.split(key)
    trajectory_exact = get_trajectory(key, nx_exact)

    if n == 0:
        plot_subfig(trajectory_exact[-1], axs, color="black", label="nx={}".format(nx_exact), linewidth=1.2,)

    for i, nx in enumerate(nxs):


        trajectory = get_trajectory(key, nx)

        UPSAMPLE = trajectory_exact.shape[0] // trajectory.shape[0]
        trajectory_exact_ds = jnp.mean(trajectory_exact[::UPSAMPLE].reshape(-1, nx, nx_exact//nx),axis=-1)
        
        accumulated_errors[i] += accumulated_mse(trajectory, trajectory_exact_ds) / n_test_table1 * (250 / trajectory.shape[0])

        if n == 0:
            plot_subfig(trajectory[-1], axs, color=colors[i], label="nx={}".format(nx), linewidth=1.2,)

for i, nx in enumerate(nxs):
    print("E1 WENO5: nx = {}, Accumulated MSE = {:.3f}".format(nx, accumulated_errors[i]))

plt.legend()
plt.show()

###################################################################################################################
###################################################################################################################
# Experiment 2 (E2)
print("We are now beginning experiment 2.")
###################################################################################################################
###################################################################################################################
fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, squeeze=True, figsize=(8, 5))

nu_max = 0.2
flux="weno"
max_u = 3.0
cfl_safety = 1.0
diff_safety = 0.5

#######################
# Test Runtime
#######################

t0 = 0.0
key = random.PRNGKey(4)
key1, key2 = random.split(key)
f_forcing = forcing_func(key1)
f_init = lambda x, t: f_forcing(x, t0)

for i, nx in enumerate(nxs):
    dx = L/nx
    a0 = integrate_function_fv(f_init, t0, nx) / dx

    nu = random.uniform(key2, (1,)) * nu_max

    @partial(jit, static_argnums=(3,))
    def sim(a0, t0, dt, nt):
        return simulate_1D(a0, t0, dx, dt, nt, nu, forcing_func=f_forcing, flux=flux)

    dt, nt = get_dt_nt(T_final, nu_max, nx, max_u, cfl_safety, diff_safety)
    af = sim(a0, t0, dt, nt).block_until_ready()
    ti = time()
    for _ in range(n_test_runtime):
        af = sim(a0, t0, dt, nt).block_until_ready()
    tf = time()
    print("E2 WENO5: nx = {}, Runtime = {:.5f}".format(nx, (tf-ti)/n_test_runtime))





#######################
# Accumulate Errors
#######################


key = random.PRNGKey(5)

accumulated_errors = onp.zeros(len(nxs))

@partial(jit, static_argnums=(1, 3))
def get_trajectory(key, nx, dt, nt):
    key1, key2 = random.split(key)
    f_forcing = forcing_func(key1)
    nu = random.uniform(key2, (1,)) * nu_max
    f_init = lambda x, t: f_forcing(x, t0)
    dx = L/nx
    a0 = integrate_function_fv(f_init, t0, nx) / dx
    return simulate_1D(a0, t0, dx, dt, nt, nu, output=True, forcing_func=f_forcing, flux=flux)



for n in range(n_test_table1):

    key, _ = random.split(key)


    dt, nt = get_dt_nt(T_final, nu_max, nx_exact, max_u, cfl_safety, diff_safety)
    trajectory_exact = get_trajectory(key, nx_exact, dt, nt)

    if n == 0:
        plot_subfig(trajectory_exact[-1], axs, color="black", label="nx={}".format(nx_exact), linewidth=1.2,)

    for i, nx in enumerate(nxs):

        dt, nt = get_dt_nt(T_final, nu_max, nx, max_u, cfl_safety, diff_safety)
        trajectory = get_trajectory(key, nx, dt, nt)

        UPSAMPLE = trajectory_exact.shape[0] // trajectory.shape[0]
        trajectory_exact_ds = jnp.mean(trajectory_exact[::UPSAMPLE].reshape(-1, nx, nx_exact//nx),axis=-1)
        
        accumulated_errors[i] += accumulated_mse(trajectory, trajectory_exact_ds) / n_test_table1 * (250 / trajectory.shape[0])

        if n == 0:
            plot_subfig(trajectory[-1], axs, color=colors[i], label="nx={}".format(nx), linewidth=1.2,)

for i, nx in enumerate(nxs):
    print("E2 WENO5: nx = {}, Accumulated MSE = {:.3f}".format(nx, accumulated_errors[i]))

plt.legend()
plt.show()