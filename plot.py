import numpy as np
from scipy.integrate import cumtrapz
from scipy.special import erf
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation

import parameters as par

plt.style.use("seaborn-bright")
plt.rc("font", family="serif")
# plt.rc("font", serif="STIXGeneral")
plt.rc("xtick", labelsize="x-small")
plt.rc("ytick", labelsize="x-small")
plt.rc("xtick.minor", visible=True)
plt.rc("ytick.minor", visible=True)
plt.rc("axes", grid=True)
plt.rc("axes", xmargin=0)
# plt.rc("text", usetex=True)
plt.rc('axes', axisbelow=True)
plt.rc("figure", autolayout=False)

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#2ecc71", "#3498db",
                                                    "#e74c3c", "#f1c40f",
                                                    "#e67e22", "#9b59b6"])

def plot_states(states, which_states, t):
    titles = ("Motor current", "Cart position", "Pendulum angle",
              "Link angle", "Cart speed", "Pendulum angular velocity",
              "Link angular velocity")
    ylabels = ("$i$ (A)", "$q_1$ (m)",
               "$q_2$ ($^\\circ$)",
               "$q_3$ ($^\\circ$)",
               "$\\dot{q}_1$ (m/s)",
               "$\\dot{q}_2$ ($^\\circ/s$)",
               "$\\dot{q}_3$ ($^\\circ/s$)")

    fig, ax = plt.subplots(len(which_states), 1, sharex=True)
    fig.set_size_inches((fig.get_size_inches()[0],
                         fig.get_size_inches()[1]*1.2*len(which_states)/7))
    k = 0
    for i in which_states:
        ax[k].plot(t, states[i, :])
        # ax[k].set_title(titles[i])
        ax[k].set_ylabel(ylabels[i], rotation="vertical")
        k += 1

    ax[-1].set_xlabel("Time (s)")
    fig.tight_layout(pad=0.3)
    return fig, ax


def plot_energy(states, t, input_voltage, brake_force, V, T, ax=None):
    V_rn = V(states[1:, :])
    V_rn -= np.min(V_rn)
    T_rn = T(states[1:, :])
    # T_cart = 0.5*par.m_point[0]*states[4,:]**2
    # T_rest = T_rn - T_cart
    E_total = cumtrapz(np.vectorize(input_voltage)(t)*states[0, :], t, initial=0)
    E_total += V_rn[0]
    E_in_brake = cumtrapz(np.vectorize(brake_force)(t)
                          *erf(3*states[4, :]), states[1, :], initial=0)
    E_inductor = cumtrapz(par.L_A*states[0, :], states[0, :], initial=0)
    losses_electric = cumtrapz(states[0, :]**2*par.R_A,
                               t, initial=0)
    losses_mechanical = (E_total - losses_electric - V_rn
                         - T_rn - E_in_brake - E_inductor)

    # if ax is None:
    #    fig, ax = plt.subplots()

    ax.stackplot(t, V_rn, T_rn, E_in_brake,
                 losses_mechanical, losses_electric, E_inductor,
                 labels=("Potential energy", "Kinetic energy", "Brake energy",
                         "Mechanical losses", "Electric losses",
                         "Energy in inductor"))
    ax.plot(t, E_total, color="black")
    ax.set_ylabel("Energy (J)")
    ax.set_xlabel("Time (s)")


def animate_system(t, states, A_pos, B_pos, C_pos, filename=None, move_along=False):
    # Adapted from https://www.moorepants.info/blog/npendulum.html

    fig = plt.figure()
    cart_width = 0.2
    cart_height = 0.2

    ax = plt.axes()
    ax.plot([-50, 50], [par.ground_height, par.ground_height], color="black")
    # ax.axis("equal")
    # ax.set_xlim((-1.2, np.max(states[1, :]) + 1.2))
    # ax.set_ylim((-0.5, 1.5))
    time_text = ax.text(0.04, 0.9, '', transform=ax.transAxes)

    # Draw the cart
    rect = Rectangle((states[0, 0] - cart_width/2., -cart_height/2.),
                     cart_width, cart_height, fill=True, color='red',
                     ec='black')
    ax.add_patch(rect)

    ax.invert_yaxis()

    # Empty line for pendulum
    line, = ax.plot([], [], lw=2, marker="o", markersize=10, color="black")
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
        ax.ignore_existing_data_limits = True
        ax.update_datalim(((states[1, i] - 1.2, -1.5), (states[1, i] + 1.2, 1.5)))
        ax.autoscale_view()

        return time_text, rect, line,

    anim = animation.FuncAnimation(fig, animate, frames=len(t), init_func=init,
                                   interval=t[-1] / len(t) * 1000, blit=True,
                                   repeat=False)

    # save the animation if a filename is given
    if filename is not None:
        anim.save(filename, fps=30, codec='libx264')


if __name__ == "__main__":
    t = np.linspace(0, 10, 200)

    fig, ax = plt.subplots()
    ax.plot(t, np.sin(t), color="k")
    ax.set_title("A nice example", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (-)")
    fig.show()

    input("Press <ENTER> to quit")
