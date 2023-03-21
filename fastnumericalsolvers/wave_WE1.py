import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from jax import vmap, jit
from functools import partial
import numpy as onp
import time

from jax import config
config.update("jax_enable_x64", True)

c = 2
c_sq = c**2
Lx = 16
sigma = 1

def ssp_rk3(a_n, F, dt):
    a_1 = a_n + dt * F(a_n)
    a_2 = 0.75 * a_n + 0.25 * (a_1 + dt * F(a_1))
    return 1 / 3 * a_n + 2 / 3 * (a_2 + dt * F(a_2))


def dd_periodic(u, dx):
    dx_R = (dx + jnp.roll(dx, -1))
    F_R = (jnp.roll(u, -1) - u) * (dx * 4) / dx_R**2
    F_L = jnp.roll(F_R, 1)
    return (F_R - F_L) / (dx)

def dd_ghost(u, dx):
    dx_R = (dx[:-1] + dx[1:])
    F = (u[1:] - u[:-1]) * (dx[1:] * 4) / dx_R**2
    return (F[1:] - F[:-1]) / (dx[1:-1])


def dadt_periodic(a, dx):
    u, v = a
    return jnp.asarray([v, c_sq * dd_periodic(u, dx)])

def dadt_ghost(a, dx):
    a = jnp.pad(a, ((0, 0), (1, 1)), mode='constant')
    dx = jnp.pad(dx, (1, 1), mode='edge')
    u, v = a
    return jnp.asarray([v[1:-1], c_sq * dd_ghost(u, dx)])


def f_gaussian(x, t):
    return jnp.exp(- ((x - c*t - Lx/2) ** 2 / (2 * sigma**2)))

def f_gaussian_deriv(x, t):
    return f_gaussian(x, t) * c * (x - c * t - Lx/2) / sigma**2

def _fixed_quad(f, a, b, n=5):
    """
    Single quadrature of a given order.

    Inputs
    f: function which takes a vector of positions of length n
    and returns a (possibly) multivariate output of length (n, p)
    a: beginning of integration
    b: end of integration
    n: order of quadrature. max n is 8.
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
    return jnp.sum(wprime * f(x_i))

def get_edges(dx):
    a = jnp.concatenate([jnp.asarray([0.0]), jnp.cumsum(dx)[:-1]])
    b = jnp.cumsum(dx)
    return a, b

def integrate_f(f, t, dx, quad_func=_fixed_quad, n=8):
    _vmap_fixed_quad = vmap(
        lambda f, a, b: quad_func(f, a, b, n=n), (None, 0, 0), 0
    )

    xL, xR = get_edges(dx)

    to_int_func = lambda x: f(x, t)


    return _vmap_fixed_quad(to_int_func, xL, xR) / dx


def get_dx(nx):
    i = jnp.arange(nx + 1)
    edges = -Lx * jnp.cos(i * jnp.pi / (nx)) / 2
    return edges[1:] - edges[:-1]

def get_ic(nx):
    dx = get_dx(nx)
    n = 1
    u_0 = integrate_f(f_gaussian, 0.0, dx, n = n)
    v_0 = integrate_f(f_gaussian_deriv, 0.0, dx, n=n)
    return jnp.asarray([u_0, v_0])


def get_dt(nx):
    cfl = 0.1
    dx = get_dx(nx)
    return cfl * jnp.max(dx)**2 / c_sq

def plot_fv(a, dx, ax, color="blue", label=None):
    u = a[0]
    xL, xR = get_edges(dx)

    x_plot = jnp.concatenate((xL[None,:], xR[None,:]),axis=0).T.reshape(-1)
    a_plot = jnp.concatenate((u[None,:], u[None,:]),axis=0).T.reshape(-1)

    ax.plot(x_plot, a_plot, color=color, label=label)

def plot_multiple_fv(a_list, dx, fig, ax, colors=["blue", "red", "green", "orange", "black", "grey"], labels=None):
    if labels is None:
        labels = [None] * len(colors)

    for k, a in enumerate(a_list):
        plot_fv(a, dx, ax, color=colors[k], label=labels[k])
    ax.set_ylim([-1.1, 1.1])


@partial(
    jit,
    static_argnums=(
        0, 2
    ),
)
def simulate(nx, dt, nt):
    dx = get_dx(nx)
    a0 = get_ic(nx)

    F = lambda a: dadt_ghost(a, dx)

    def step(a, _):
        a_f = ssp_rk3(a, F, dt)
        return a_f, a

    _, trajectory = jax.lax.scan(step, a0, xs=None, length=nt)
    return trajectory



def evalf_1D(x, u, xL, xR):
    """
    assume x is scalar, u is vector, xL and xR are vectors
    """
    j = jnp.argmax((x >= xL) & (x < xR))
    return u[j]


@partial(
    jit,
    static_argnums=(
        1,
        2,
    ),
)
def convert_FV_representation_u(u, nx_new, nx_old):
    dx_old = get_dx(nx_old)
    dx_new = get_dx(nx_new)
    xL, xR = get_edges(dx_old)

    def f_old(x, t):
        return evalf_1D(x, u, xL, xR)

    vmap_f_old = vmap(f_old, (0, None))

    return integrate_f(vmap_f_old, 0.0, dx_new, n=8)

@partial(
    jit,
    static_argnums=(
        1,
        2,
    ),
)
def convert_FV_representation(a, nx_new, nx_old):
    return vmap(convert_FV_representation_u, (0, None, None))(a, nx_new, nx_old)




#####
# Loss function
#####
def accumulated_mse_loss(a_trajectory, a_exact_trajectory):
    """
    both a and a_exact should be of shape (nt, 2, nx)
    """
    assert a_trajectory.shape[0] == 250
    assert a_trajectory.shape[1] == 2

    u_traj = a_trajectory[...,0]
    u_exact_traj = a_exact_trajectory[...,0]
    return jnp.sum(jnp.mean((u_traj - u_exact_traj)**2, axis=-1))


##################
# HYPERPARAMS
##################

T_final = 8.0
nxs = [20, 40, 50, 100]
nt_resamples = [1, 4, 5, 20]
nts = [250 * nt_resample for nt_resample in nt_resamples]
dts = [T_final / nt for nt in nts]
EXACT_UPSAMPLE = 8


fig, axs = plt.subplots(4, figsize=(10, 7.5), squeeze=True)

#########################
# Plot simulations
#########################

# exact simulation
nx_exact = nxs[-1] * EXACT_UPSAMPLE
dt_exact = dts[-1] / EXACT_UPSAMPLE**2
nt_exact = nts[-1] * EXACT_UPSAMPLE**2

trajectory_exact = simulate(nx_exact, dt_exact, nt_exact)[::nt_resamples[-1] * EXACT_UPSAMPLE**2]
a_exact_final = trajectory_exact[-1]

losses_ps = ['breaks', '194.622', '0.450', '0.004']
losses_mp = ['0.059', '0.042', '0.035', '0.137']

runtimes_ps = ['0.20', '0.25', '0.35', '0.60']
runtimes_mp = ['0.07', '0.09', '0.09', '0.09']



for k, nx in enumerate(nxs):


    dt = dts[k]
    nt = nts[k]
    trajectory = simulate(nx, dt, nt)[::nt_resamples[k]]


    a0 = trajectory[0]
    a_quarter = trajectory[62]
    a_three_quarter = trajectory[187]
    a_final = trajectory[-1]

    trajectory_exact_ds = vmap(convert_FV_representation, (0, None, None))(trajectory_exact, nx, nx_exact)
    a_exact_final_ds = trajectory_exact_ds[-1]

    #########
    # Compute error
    #########

    MSE = accumulated_mse_loss(trajectory, trajectory_exact_ds)
    
    #########
    # Compute runtime
    #########

    average_runtime = 0.0
    N_ave = 10

    _ = simulate(nx, dt, nt)

    for _ in range(N_ave):
        t1 = time.time()
        res = simulate(nx, dt, nt)
        res.block_until_ready()  
        t2 = time.time()
        average_runtime += (t2 - t1) / N_ave

    print("average runtime for nx = {} is {}".format(nx, average_runtime))



    plot_multiple_fv([a0, a_quarter, a_three_quarter, a_final, a_exact_final_ds], get_dx(nx), fig, axs[k], labels=["t=0", "t=2", "t=6", "t=8", "Exact t=8"])
    axs[k].set_title("nx = {}".format(nx))

    fig.suptitle("WE1 (Dirichlet)")


    text_losses_str = '\n'.join((
    'Accumulated MSE:',
    'PS: {}'.format(losses_ps[k]),
    'MP-PDE: {}'.format(losses_mp[k]),
    'FV: {0:.5f}'.format(MSE)))

    text_runtimes_str = '\n'.join((
    'Runtime (s):',
    'PS: {}'.format(runtimes_ps[k]),
    'MP-PDE: {}'.format(runtimes_mp[k]),
    'FV: {0:.5f}'.format(average_runtime)))

    props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)

    axs[k].text(0.01, 0.75, text_losses_str, transform=axs[k].transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

    axs[k].text(0.21, 0.75, text_runtimes_str, transform=axs[k].transAxes, fontsize=12,
        verticalalignment='top', bbox=props)


    print("Accumulated error for nx = {} is {}".format(nx, MSE))


handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
vals = list(by_label.values())
keys = list(by_label.keys())

fig.legend(vals, keys, prop={'size': 10})
fig.tight_layout()


plt.savefig('WE1_reproduction.eps')
plt.savefig('WE1_reproduction.png')

plt.show()








