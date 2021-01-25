import dill

from math import floor

import numpy as np
from scipy.linalg import solve
from scipy.integrate import odeint

import matplotlib.pyplot as plt

from sympy import symbols, asin, sin, simplify
from sympy.physics.mechanics import (
             Particle, Point, ReferenceFrame, RigidBody,
             dynamicsymbols, inertia, kinetic_energy,
             potential_energy, LagrangesMethod, mechanics_printing)
from sympy.utilities.lambdify import lambdify

import parameters as par

# For shorter expressions
mechanics_printing(pretty_print=False)

# ----------------------------------------------------------------------------
#                                Define symbols
# ----------------------------------------------------------------------------

# Generalized coordinates
n_coords = 3
q = dynamicsymbols("q:" + str(n_coords))
dq = dynamicsymbols("q:" + str(n_coords), level=1)

# Mass and link lengths
m_links = symbols("m_l:2")
d_links = symbols("d_l:2")
m_point = symbols("m_cart m_A m_B m_C")  # for point objects

b_cart, b_joint = symbols("b_cart b_joint")  # Viscous damping

g, t = symbols("g, t")  # General parameters
k, l0 = symbols("k l0")  # Spring parameters
F = symbols("F")  # Input force; both motor and hydraulic brake

# Lists to store the objects
particles = []
links = []
points = []
com = []
frames = []

sym_pars = [*m_links, *d_links, *m_point, b_cart, b_joint, g, k, l0]

# ----------------------------------------------------------------------------
#                               Reference frames
# ----------------------------------------------------------------------------

N = ReferenceFrame("N")  # Inertial reference frame
(origin := Point("O")).set_vel(N, 0)  # Set velocity to zero

pend_frame = N.orientnew("pend", "axis", (q[1], N.z))
pend_frame.set_ang_vel(N, dq[1]*N.z)

# Link 1: upper left link
link1_frame = pend_frame.orientnew("link_1", "axis", (q[2], pend_frame.z))
link1_frame.set_ang_vel(pend_frame, dq[2]*pend_frame.z)

# Link 2: upper right link
link2_frame = pend_frame.orientnew("link_2", "axis", (-q[2], pend_frame.z))
link2_frame.set_ang_vel(pend_frame, -dq[2]*pend_frame.z)

# Compute angle between upper and lower links
beta = q[2] + asin(d_links[0]/d_links[1]*sin(q[2]))
beta_dot = beta.diff(t).simplify()

# Link 3: lower left link
link3_frame = link1_frame.orientnew("link_3", "axis", (-beta, link1_frame.z))
link3_frame.set_ang_vel(link1_frame, -beta_dot*link1_frame.z)

# Link 4: lower right link
link4_frame = link2_frame.orientnew("link_4", "axis", (beta, link2_frame.z))
link4_frame.set_ang_vel(link2_frame, beta_dot*link2_frame.z)

frames = (link1_frame, link2_frame, link3_frame, link4_frame)


# ----------------------------------------------------------------------------
#                                 Point masses
# ----------------------------------------------------------------------------

# Define center of mass for the cart; set position to q0
(com_cart := origin.locatenew("com_cart", q[0]*N.x)).set_vel(N, dq[0]*N.x)

# Set point locations and their velocities
A = com_cart.locatenew("A", link1_frame.y*d_links[0])
A.v2pt_theory(com_cart, N, link1_frame)
B = com_cart.locatenew("B", link2_frame.y*d_links[0])
B.v2pt_theory(com_cart, N, link2_frame)
C = A.locatenew("C", link3_frame.y*d_links[1])
C.v2pt_theory(A, N, link3_frame)

# Assemble points and particles in a list
points = (com_cart, A, B, C)
particles = [Particle(points[i].name, points[i], m_point[i])
             for i in range(4)]


# ----------------------------------------------------------------------------
#                                 Rigid bodies
# ----------------------------------------------------------------------------

# ---------------------------- Define mass centers ---------------------------

com1 = com_cart.locatenew("com1", 0.5*link1_frame.y*d_links[0])  # Link 1
com1.v2pt_theory(com_cart, N, link1_frame)

com2 = com_cart.locatenew("com2", 0.5*link2_frame.y*d_links[0])  # Link 2
com2.v2pt_theory(com_cart, N, link2_frame)

com3 = A.locatenew("com3", 0.5*link3_frame.y*d_links[1])  # Link 3
com3.v2pt_theory(A, N, link3_frame)

com4 = B.locatenew("com4", 0.5*link4_frame.y*d_links[1])  # Link 4
com4.v2pt_theory(B, N, link4_frame)

com = (com1, com2, com3, com4)

# ------------------------------- Rigid bodies -------------------------------

for i in range(4):
    j = floor(i/2)  # Index for link mass and length
    links.append(RigidBody(f"link{i}", com[i], frames[i],
                           m_links[j],
                           (inertia(frames[0], 0, 0, m_links[j]/12
                                    * d_links[j]**2), com[i])))


# ----------------------------------------------------------------------------
#                          Compute energy expressions
# ----------------------------------------------------------------------------

def set_pot_grav_energy(thing):
    # Set the potential gravitational energy for either a Particle or RigidBody
    # based on its height in the inertial frame N.
    try:
        point = thing.point
    except AttributeError:
        point = thing.masscenter

    height = point.pos_from(origin).dot(N.y).simplify()
    thing.potential_energy = -height*thing.mass*g
    return thing


particles = list(map(set_pot_grav_energy, particles))
links = list(map(set_pot_grav_energy, links))


# ----------------------------------------------------------------------------
#                              Equations of motion
# ----------------------------------------------------------------------------

if __name__ == "__main__":  # Do not perform derivation when imported

    simplify_exps = True

    T = kinetic_energy(N, *particles, *links)  # Kinetic energy
    V = potential_energy(*particles, *links)  # Potential energy

    # Add the contribution of the spring
    V += simplify(0.5*k*(A.pos_from(B).magnitude() - l0)**2)

    if simplify_exps:
        T = T.simplify()
        V = V.simplify()

    L = T - V  # Lagrangian

    # -------------------------- Friction torques -----------------------------
    # N.z is used because all z-axis are parallel
    torques = [(pend_frame, -b_joint*dq[1]*N.z),

               (link1_frame, -b_joint*2*dq[2]*N.z),
               (link2_frame, b_joint*2*dq[2]*N.z),

               (link1_frame, -b_joint*(beta_dot)*N.z),
               (link2_frame, b_joint*(beta_dot)*N.z),
               (link3_frame, b_joint*(beta_dot)*N.z),
               (link4_frame, -b_joint*(beta_dot)*N.z),

               (link3_frame, -b_joint*2*(beta_dot - dq[2])*N.z),
               (link4_frame, b_joint*2*(beta_dot - dq[2])*N.z)]

    forces = [(com_cart, -b_cart*dq[0]*N.x),  # Rolling resistance of the cart
              (com_cart, F)]

    LM = LagrangesMethod(L, q, forcelist=torques, frame=N)
    LM.form_lagranges_equations()

    M_symb = LM.mass_matrix_full
    F_symb = LM.forcing_full

    if simplify_exps:
        M_symb = M_symb.simplify()
        F_symb = F_symb.simplify()

    fun_args = [(*q, *dq), F, *sym_pars]
    M_func = lambdify(fun_args, M_symb, modules="scipy")
    F_func = lambdify(fun_args, F_symb, modules="scipy")

    def f(qdq, *sym_pars):
        return np.array(solve(M_func(qdq, *sym_pars),
                              F_func(qdq, *sym_pars))).T[0]

    dill.settings['recurse'] = True
    dill.dump(f, open("f_dyn", "wb"))
