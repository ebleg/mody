# Mechanical parameters
m_links = [0.5, 0.375]
m_point = [5., 2, 2, 2]

d_links = [0.5, 0.375]

g = 9.81

# Spring parameters
k = 200.
l0 = 0.5

# Viscous friction
b_joint = 0.4
b_cart = 0.02

# No mechanical losses
# b_joint = 0
# b_cart = 0

# Electric motor
R_A = 2.5  # Armature resistance
L_A = 50e-3  # Armature inductance e-6
Kt = 6.7e-1  # Back EMF constant e-4

# Misc
wheel_radius = 0.05
ball_radius = 0.05
ground_height = 0.15

# Hybrid simulation
restitution_coeff = 0.85
baud = 100  # Symbol rate of sensor
sensor_threshold = 10  # Angular velocity alarm threshold
