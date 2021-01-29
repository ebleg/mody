import dill

import numpy as np
from scipy.integrate import solve_ivp, cumtrapz
from scipy.special import erf

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation
import timeit

import parameters as par
import plot

# from lagrange import sym_pars

f_nopars = dill.load(open("func/f_dyn", "rb"))
V_nopars = dill.load(open("func/V_func", "rb"))
T_nopars = dill.load(open("func/T_func", "rb"))
A_nopars = dill.load(open("func/A_func", "rb"))
B_nopars = dill.load(open("func/B_func", "rb"))
C_nopars = dill.load(open("func/C_func", "rb"))

num_pars = [*par.m_links, *par.d_links, *par.m_point, par.b_cart, par.b_joint,
            par.g, par.k, par.l0]


def f_multi_body(F, qdq): return f_nopars(*qdq, F, *num_pars)
def V(qdq): return V_nopars(*qdq, *num_pars)
def T(qdq): return T_nopars(*qdq, *num_pars)
def A_pos(qdq): return A_nopars(*qdq, *num_pars)
def B_pos(qdq): return B_nopars(*qdq, *num_pars)
def C_pos(qdq): return C_nopars(*qdq, *num_pars)


# With electric motor
def f_full(t, x, U, F_brake):
    dx = np.zeros(x.shape)

    # Current change
    dx[0] = 1/par.L_A*(-par.R_A*x[0] - par.Kt*x[4]/par.wheel_radius + U(t))
    # Cart dynamics
    dx[1:] = f_multi_body(par.Kt*x[0]/par.wheel_radius
                          - F_brake(t)*x[4]/par.wheel_radius, x[1:])
    return dx


# ----------------------------------------------------------------------------
#                                 Verification
# ----------------------------------------------------------------------------

t_max = 10
t = np.linspace(0, t_max, 300)
x0 = np.array([0, 0, np.deg2rad(25), np.pi/6, 0, 0, 0])
t0 = timeit.default_timer()
sol = solve_ivp(lambda t, x: f_full(t, x, lambda t: 0, lambda t: 0), 
                (0, t_max), x0, method="RK45", t_eval=t)
print(f"Time elapsed {timeit.default_timer() - t0}")

t_ver = sol.t
y_ver = sol.y

# fig, ax = plt.subplots(4, 1, sharex=True)
# fig.set_size_inches((fig.get_size_inches()[0], fig.get_size_inches()[1]*1.1))
# ax[0].plot(t_ver, y_ver[0, :])
# ax[0].set_title("Motor current")
# ax[0].set_ylabel("$i$ (A)")
# ax[1].plot(t_ver, y_ver[1, :])
# ax[1].set_ylabel("$q_1$ (m)")
# ax[1].set_title("Cart position")
# ax[2].plot(t_ver, np.rad2deg(y_ver[2, :]))
# ax[2].set_title("Pendulum angle")
# ax[2].set_ylabel("$q_2$ ($^\circ$)")
# ax[3].plot(t_ver, np.rad2deg(y_ver[3, :]))
# ax[3].set_title("Link angle")
# ax[3].set_ylabel("$q_2$ ($^\circ$)")
# ax[3].set_xlabel("Time (s)")
# fig.tight_layout(pad=0.3)
# fig.show()
# fig.savefig("media/verification.eps")

# ----------------------------------------------------------------------------
#                             Plot time simulation
# ----------------------------------------------------------------------------


def U(t):
    if t < 2:
        return 12
    else:
        return 0


# F_brake = lambda t: 0
def F_brake(t):
    if t < 2:
        return 0
    else:
        return 1


t_max = 10
t = np.linspace(0, t_max, 600)
x0 = np.array([0, 0, 0, np.pi/6, 0, 0, 0])
t0 = timeit.default_timer()
sol = solve_ivp(lambda t, x: f_full(t, x, U, F_brake), (0, t_max),
                x0, method="RK45", t_eval=t)
print(f"Time elapsed {timeit.default_timer() - t0}")

t_sim = sol.t
y_sim = sol.y

# fig, ax = plt.subplots(5, 1, sharex=True)
# fig.set_size_inches((fig.get_size_inches()[0], fig.get_size_inches()[1]*1.3))
# ax[0].plot(t_sim, y_sim[0, :])
# ax[0].set_title("Motor current")
# ax[0].set_ylabel("$i$ (A)")
# ax[1].plot(t_sim, y_sim[1, :])
# ax[1].set_ylabel("$q_1$ (m)")
# ax[1].set_title("Cart position")
# ax[2].plot(t_sim, y_sim[4, :])
# ax[2].set_ylabel("$\dot{q}_1$ (m)")
# ax[2].set_title("Cart speed")
# ax[3].plot(t_sim, np.rad2deg(y_sim[2, :]))
# ax[3].set_title("Pendulum angle")
# ax[3].set_ylabel("$q_2$ ($^\circ$)")
# ax[4].plot(t_sim, np.rad2deg(y_sim[3, :]))
# ax[4].set_title("Link angle")
# ax[4].set_ylabel("$q_2$ ($^\circ$)")
# ax[4].set_xlabel("Time (s)")
# fig.tight_layout(pad=0.3)
# fig.show()
# fig.savefig("media/time_simulation.eps")


# ---------------------------------------------------------------------------- #
#                                 Plot energies                                #
# ---------------------------------------------------------------------------- #

def plot_energy(states, t, U, F_brake, ax=None):
    V_rn = V(states[1:, :])
    V_rn -= np.min(V_rn)
    T_rn = T(states[1:, :])
    # T_cart = 0.5*par.m_point[0]*states[4,:]**2
    # T_rest = T_rn - T_cart
    E_total = cumtrapz(np.vectorize(U)(t)*states[0, :], t, initial=0)
    E_total += V_rn[0]
    E_in_brake = cumtrapz(np.vectorize(F_brake)(t)
                          * states[4, :]/par.wheel_radius,
                          states[1, :], initial=0)
    E_inductor = cumtrapz(par.L_A*states[0, :], states[0, :], initial=0)
    losses_electric = cumtrapz(states[0, :]**2*par.R_A,
                               t, initial=0)
    losses_mechanical = (E_total - losses_electric - V_rn
                         - T_rn - E_in_brake - E_inductor)

    if ax is None:
        fig, ax = plt.subplots()

    ax.stackplot(t, V_rn, T_rn, E_in_brake,
                 losses_mechanical, losses_electric, E_inductor,
                 labels=("Potential energy", "Kinetic energy", "Brake energy",
                         "Mechanical losses", "Electric losses",
                         "Energy in inductor"))
    ax.plot(t, E_total, color="black")
    ax.set_ylabel("Energy (J)")
    ax.set_xlabel("Time (s)")


def animate_system(t, states, filename=None):
    # Adapted from https://www.moorepants.info/blog/npendulum.html

    fig = plt.figure()
    cart_width = 0.4
    cart_height = 0.4

    ax = plt.axes(xlim=(-1.2, np.max(states[1, :]) + 1.2),
                  ylim=(-0.5, 2))
    ax.axis("equal")
    ax.set_xlim((-1.2, np.max(states[1, :]) + 1.2))
    ax.set_ylim((-0.5, 2))
    time_text = ax.text(0.04, 0.9, '', transform=ax.transAxes)

    # Draw the cart
    rect = Rectangle([states[0, 0] - cart_width/2., -cart_height/2.],
                     cart_width, cart_height, fill=True, color='red',
                     ec='black')
    ax.add_patch(rect)

    ax.invert_yaxis()

    # Empty line for pendulum
    line, = ax.plot([], [], lw=2, marker="o", markersize=6, color="black")
    spring, = ax.plot([], [], lw=2, color="grey")

    def init():
        time_text.set_text("")
        rect.set_xy((0, 0))
        line.set_data([], [])
        return time_text, rect, line

    def animate(i):
        time_text.set_text('time = {:2.2f}'.format(t[i]))
        rect.set_xy((states[1, i] - cart_width/2., -cart_height/2))
        A = A_pos(states[1:, i])
        B = B_pos(states[1:, i])
        C = C_pos(states[1:, i])

        x_data = (states[1, i], A[0], C[0], B[0], states[1, i])
        y_data = (0, A[1], C[1], B[1], 0)
        line.set_data(x_data, y_data)
        spring.set_data((A[0], B[0]), (A[1], B[1]))
        return time_text, rect, line,

    anim = animation.FuncAnimation(fig, animate, frames=len(t), init_func=init,
                                   interval=t[-1] / len(t) * 1000, blit=True,
                                   repeat=False)

    # save the animation if a filename is given
    if filename is not None:
        anim.save(filename, fps=30, codec='libx264')


# fig, ax = plt.subplots(1, 2)
# plot_energy(y_ver, t_ver, lambda t: 0, lambda t: 0, ax[0])
# plot_energy(y_sim, t_sim, U, F_brake, ax[1])
# ax[1].set_ylabel(None)
# ax[0].set_title("Verification case", fontsize="medium")
# ax[1].set_title("Simulation with inputs", fontsize="medium")
# fig.legend(("Total energy", "Potential energy", "Kinetic energy",
#             "Brake energy", "Mechanical losses", "Electrial losses",
#             "Inductor"), ncol=3, loc="lower center", fontsize="small")
# fig.suptitle("Energy losses")
# fig.tight_layout(pad=0.3)
# fig.subplots_adjust(bottom=0.22, top=0.9)
# fig.show()
# fig.savefig("media/energy.eps")

animate_system(t_sim, y_sim, filename="media/simulation.gif")
animate_system(t_ver, y_ver, filename="media/verification.gif")
