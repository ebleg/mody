import dill

import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

import parameters as par
# from lagrange import sym_pars

f_nopars = dill.load(open("f_dyn", "rb"))

num_pars = [*par.m_links, *par.d_links, *par.m_point, par.b_cart, par.b_joint,
            par.g, par.k, par.l0]


def f_multi_body(F, qdq): return f_nopars(qdq, F, *num_pars)


# With electric motor
def f_full(t, x, U, F_brake):
    dx = np.zeros(x.shape)

    # Current change
    dx[0] = 1/par.L_A*(-par.R_A*x[0] - par.Kt*x[4]/par.wheel_radius + U(t))
    # Cart dynamics
    dx[1:] = f_multi_body(par.Kt*x[0]/par.wheel_radius - F_brake(t)*x[4],
                          x[1:])
    return dx

U = lambda t: 10
F_brake = lambda t: 0

t_max = 4
t = np.linspace(0, t_max, 300)
x0 = np.array([0, 0, 0, np.pi/6, 0, 0, 0])
sol = solve_ivp(lambda t, x: f_full(t, x, U, F_brake), (0, 4),
                x0, method="RK45", t_eval=t)
t = sol.t
y = sol.y

fig, ax = plt.subplots(3, 1)
ax[0].plot(t, y[1, :])
ax[0].set_title("Cart position")
ax[1].plot(t, y[2, :])
ax[1].set_title("Pendulum angle")
ax[2].plot(t, y[3, :])
ax[2].set_title("Link angle")
fig.tight_layout()
fig.show()

input("Press <ENTER> to quit")
