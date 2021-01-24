from math import floor

import numpy as np
from scipy.linalg import solve
from scipy.integrate import odeint

import matplotlib.pyplot as plt

from sympy import symbols, asin, sin, simplify
from sympy.physics.mechanics import (
             Particle, Point, ReferenceFrame, RigidBody,
             dynamicsymbols, inertia, kinetic_energy,
             potential_energy, LagrangesMethod)
from sympy.vector import express
from sympy.utilities.lambdify import lambdify

import parameters as par

# Clear terminal output

# --------------------------------------------------
# Lagrangian dynamics
# --------------------------------------------------

# ---------------- Define symbols ------------------
n_coords = 3
q = dynamicsymbols("q:" + str(n_coords))
dq = dynamicsymbols("q:" + str(n_coords), level=1)

m_links = symbols("m_l:2")
d_links = symbols("d_l:2")
m_point = symbols("m_A m_B m_C")
m_cart = symbols("m_cart")

b_cart, b_joint = symbols("b_cart b_joint")

g, t = symbols("g, t")
k, l0 = symbols("k l0")

# Lists to store the objects
particles = []
links = []
points = []
com = []
frames = []

# Reference frames
N = ReferenceFrame("N")  # Inertial reference frame
(origin := Point("O")).set_vel(N, 0)  # Set velocity to zero

# --------------------- Cart -----------------------

# Define center of mass for the cart; set position to q0
(com_cart := origin.locatenew("com_cart", q[0]*N.x)).set_vel(N, dq[0]*N.x)

particles.append(Particle("cart", com_cart, m_cart))  # Define particle

# ------------------- Pendulum ---------------------

pend_frame = N.orientnew("pend", "axis", (q[1], N.z))
pend_frame.set_ang_vel(N, dq[1]*N.z)

# Link 1: upper left link
link1_frame = pend_frame.orientnew("link_1", "axis", (q[2], pend_frame.z))
link1_frame.set_ang_vel(pend_frame, dq[2]*pend_frame.z)

# Link 2: upper right link
link2_frame = pend_frame.orientnew("link_2", "axis", (-q[2], pend_frame.z))
link2_frame.set_ang_vel(pend_frame, -dq[2]*pend_frame.z)

beta = q[2] + asin(d_links[0]/d_links[1]*sin(q[2]))
beta_dot = beta.diff(t).simplify()

link3_frame = link1_frame.orientnew("link_3",
                                    "axis", (-beta, link1_frame.z))
link3_frame.set_ang_vel(link1_frame, -beta_dot*link1_frame.z)

link4_frame = link2_frame.orientnew("link_4",
                                    "axis", (beta, link2_frame.z))
link4_frame.set_ang_vel(link2_frame, beta_dot*link2_frame.z)

frames = (link1_frame, link2_frame, link3_frame, link4_frame)

# Set points
A = com_cart.locatenew("A", link1_frame.y*d_links[0])
B = com_cart.locatenew("B", link2_frame.y*d_links[0])
C = A.locatenew("C", link3_frame.y*d_links[1])

# Define corresponding particles
for i in range(3):
    pnt = [A, B, C][i]
    pnt.set_vel(N, pnt.pos_from(origin).dt(N).simplify())
    part = Particle(pnt.name, pnt, m_point[i])
    part.potential_energy = (-m_point[i]*g
                             * N.y.dot(pnt.pos_from(origin)))
    particles.append(part)

# Rigid bodies for each link
# Define center of mass
com.append(com_cart.locatenew("com1", 0.5*link1_frame.y*d_links[0]))
com.append(com_cart.locatenew("com2", 0.5*link2_frame.y*d_links[0]))
com.append(A.locatenew("com3", 0.5*link3_frame.y*d_links[1]))
com.append(B.locatenew("com4", 0.5*link4_frame.y*d_links[1]))

for i in range(4):
    com[i].set_vel(N, com[i].pos_from(origin).dt(N).simplify())
    j = floor(i/2)  # Index for link mass and length
    links.append(RigidBody(f"link{i}", com[i], frames[i],
                           m_links[j],
                           (inertia(frames[0], 0, 0, m_links[j]/12
                                    * d_links[j]**2), com[i])))
    links[i].potential_energy = (-m_links[j]*g
                                 * N.y.dot(com[i].pos_from(origin)))

T = kinetic_energy(N, *particles, *links)
V = potential_energy(*particles, *links)

# Add the contribution of the spring
V += simplify(0.5*k*(A.pos_from(B).magnitude() - l0)**2)

L = T - V

LM = LagrangesMethod(L, q, frame=N)
LM.form_lagranges_equations()

# Perform simulation
sym_pars = [*m_links, *d_links, *m_point, m_cart, g, k, l0]
num_pars = [*par.m_links, *par.d_links, *par.m_point, par.m_cart,
            par.g, par.k, par.l0]

subs_dict = {sym_pars[i]: num_pars[i] for i in range(len(sym_pars))}

M_num = LM.mass_matrix_full.subs(subs_dict)
F_num = LM.forcing_full.subs(subs_dict)

fun_args = (*q, *dq)
M_func = lambdify([fun_args], M_num)
F_func = lambdify([fun_args], F_num)


def f(x, t):
    return np.array(solve(M_func(x), F_func(x))).T[0]


if __name__ == "__main__":
    t = np.linspace(0, 4, num=300)
    x0 = np.array([0, 0, 1.2*np.pi/6, 0, 0, 0])
    y = odeint(f, x0, t)

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(t, y[:, 0])
    ax[0].set_title("Cart position")
    ax[1].plot(t, y[:, 1])
    ax[1].set_title("Pendulum angle")
    ax[2].plot(t, y[:, 2])
    ax[2].set_title("Link angle")
    fig.tight_layout()
    fig.show()
