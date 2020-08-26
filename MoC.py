# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 21:41:28 2018

@author: Gabriel
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

"""Method of characteristics for a minimum length nozzle."""
t_start = time.perf_counter() # Start timer
Mdes = 3  # Design Mach number
g = 1.2  # Ratio of specific heats
Nc = 100  # Number of characteristic lines
d = 1  # Small initial turning angle of flow, in degrees

Np = 2
for i in np.arange(1, Nc):
    Np = Np + i + 2  # Number of points for Nc characteristic lines
print('Calculating %i points' % Np)

Mmax = 7
# List of Mach numbers between 1 and Mmax
Ms = np.linspace(1, Mmax, 100*(Mmax-1)+1)
# List of Prandtl-Meyer angles corresponding to each element of Ms
nus = np.degrees(((g+1)/(g-1))**0.5*np.arctan(((g-1)/(g+1)*(Ms**2-1))**0.5)
                 - np.arctan((Ms**2-1)**0.5))


def NUofM(M):
    """Takes a mach number M between 1 and 4 to two decimal places, returns the
    corresponding Prandtl-Meyer angle NU in degrees."""
    nu = 0
    for i in np.arange(100*(Mmax-1)+1):
        if Ms[i] == M:
            nu = nus[i]
            break
    return nu


def MofNU(nu):
    """Takes a Prandtl-Meyer angle NU between 0 and 88.4 degrees, and returns
    the nearest corresponding Mach number M from list Ms.
    """
    if nu > 0 and nu <= NUofM(Mmax):
        for i, n in enumerate(nus):
            if nu <= n and i > 0:
                Mhigh = Ms[i]
                NUhigh = n
                Mlow = Ms[i - 1]
                NUlow = nus[i - 1]
                break
        if nu-NUlow > NUhigh - nu:
            return Mhigh
        else:
            return Mlow
    else:
        raise ValueError('ERROR: NU out of range. Must be between 0 and 88.4.')

MofNU = njit(MofNU)
NUofM = njit(NUofM)

A = np.zeros((Np+Nc, 10))
Rp = np.ones(Np+Nc)  # R+, Riemann invariant of C+
Rm = np.ones(Np+Nc)  # R-, Riemann invariant of C-
theta = np.ones(Np+Nc)  # Flow turning angle
nu = np.ones(Np+Nc)  # Prandtl-Meyer angle
M = np.ones(Np+Nc)  # Mach number
mu = np.ones(Np+Nc)  # Mach angle
alpha_p = np.ones(Np+Nc)  # Angle of C+
alpha_m = np.ones(Np+Nc)  # Angle of C-
x = np.ones(Np+Nc)  # x-coordinate
y = np.ones(Np+Nc)  # y-coordinate

# Array of points along each characteristic line.
print('%i characteristic lines' % Nc)
C = np.zeros((Nc, Nc+1))
C[0] = np.arange(1, Nc+2)
C[:, 0] = C[0, :-1]
for i in np.arange(1, Nc):
    C[i, i:] = C[i-1, i:] + Nc+1-i
    C[i:, i] = C[i, i:-1]
C = C.astype(int)

# Array of points along centerline.
CL = np.zeros((2, Nc))
print('%i points on centerline' % len(CL[0]))
for i in np.arange(Nc):
    CL[0, i] = C[i, i]  # Points along the centerline
for i in np.arange(1, Nc):
    CL[1, i] = C[i-1, i]  # Previous points along C-
CL[1, 0] = -(Nc-1)
CL = CL.astype(int)

Rp[:Nc] = 0  # R+ = 0 at starting point
thetamax = NUofM(Mdes)/2  # Maximum turning angle = nu(Mdes)/2
theta[:Nc] = np.linspace(d, thetamax, Nc)  # Equally spaced turning angles
nu[:Nc] = theta[:Nc]  # nu = theta near starting point
Rm[:Nc] = (nu+theta)[:Nc]  # R+ = nu + theta near starting point

print('R+ and R-')
for i in np.arange(1, Nc):
    p = CL[0, i-1] + Nc - 1  # Index of points along centerline
    q = CL[0, i] + Nc - 1  # Index of previous points along C-
    Rp[p:q] = Rm[i-1]
    Rm[p:q-1] = Rm[i-1:Nc]
    Rm[q-1] = None
Rm[q] = Rm[Nc-1]
Rm[-1] = None
Rp[q:] = Rm[Nc-1]

theta[Nc:] = (Rm-Rp)[Nc:]/2
nu[Nc:] = (Rm+Rp)[Nc:]/2

print('theta, nu and M')
for i in np.arange(Nc+Np):
    if Rm[i] > 0:
        M[i] = MofNU(nu[i])
    else:
        theta[i] = theta[i-1]
        nu[i] = nu[i-1]
        M[i] = MofNU(nu[i])

mu = np.degrees(np.arcsin(1/M))  # Mach angle of M

print('alpha_m for centerline')
j = CL[0, :] + Nc - 1  # Index of points along centerline
k = CL[1, :] + Nc - 1  # Index of previous points along C-
# Angle of C- crossing centerline
alpha_m[j] = 0.5*((theta[k]-mu[k])+(theta[j]-mu[j]))

# Wallpoints
W = np.zeros((3, Nc))
print('%i wall points' % len(W[0]))
W[0] = C[:, -1]  # Points along the wall
W[2] = W[0] - 1  # Previous point along C+
W[1, 1:] = W[0, :-1]  # Previous points along wall
W = W.astype(int)

print('alpha_m and alpha_p for wall points')
p = W[0, :] + Nc - 1  # Index of wallpoint
q = W[1, :] + Nc - 1  # Index of previous wallpoints
r = W[2, :] + Nc - 1  # Index of previous points along C+
# Angle between wallpoint and its previous wallpoint
alpha_m[p] = 0.5*(theta[q] + theta[p])
# Angle of C+ leading to wallpoint
alpha_p[p] = theta[r] + mu[r]

# Every interior point, and the previous points along C+ and C-
IN = np.zeros((3, np.sum(np.arange(Nc))))
print('%i interior points' % len(IN[0]))
a = np.array([])
for i in np.arange(0, Nc-1):
    a = np.append(a, C[:-1, 1:-1][i, i:])
c = np.arange(-(Nc-2), 1)
for i in np.arange(0, Nc-2):
    c = np.append(c, C[:-2, 2:-1][i, i:])
IN = np.array((a, a-1, c))
IN = IN.astype(int)


# Angle of C+ and C- leading to each interior point.
print('alpha_m and alpha_p for interior points')
p = IN[0, :] + Nc - 1
a = IN[1, :] + Nc - 1
b = IN[2, :] + Nc - 1
alpha_p[p] = 0.5*((theta+mu)[a] + (theta+mu)[p])
alpha_m[p] = 0.5*((theta-mu)[b] + (theta-mu)[p])

# Every point and the previous point on C+ and C-
a = np.array([])  # Every point
a = np.append(a, CL[0])
a = np.append(a, W[0])
a = np.append(a, IN[0])

b = np.array([])  # previous point on C+
b = np.append(b, CL[1])
b = np.append(b, W[2])
b = np.append(b, IN[1])

c = np.array([])  # previous point on C-
c = np.append(c, CL[1])
c = np.append(c, W[1])
c = np.append(c, IN[2])

D = np.array((a, b, c))
P = np.zeros((3, Np))

print('argsort running')
sort = np.argsort(D[0])  # Indices for putting points in correct order
print('ordering points')
P = D[:, sort]  # Correct order of points and their previous C+ and C- points
P = P.astype(int)

tam = np.tan(np.radians(alpha_m))  # tan(alpha_m)
tap = np.tan(np.radians(alpha_p))  # tan(alpha_p)

print('Calculating x-y coordinates of all points')
x[:Nc] = 0  # x-coordinate of starting point
y[:Nc] = 1  # y-coordinate of starting point
p = P[0, :] + Nc - 1  # Index of each point
a = P[1, :] + Nc - 1  # Index of previous point on C+
b = P[2, :] + Nc - 1  # Index of previous point on C-
for i in np.arange(Np):
    if theta[p[i]] == 0 and Rm[p[i]] > 0:  # Centerline coordinates
        x[p[i]] = x[b[i]] - y[b[i]]/tam[p[i]]
        y[p[i]] = 0
    else:  # All other coordinates
        x[p[i]] = (x[b[i]]*tam[p[i]] - x[a[i]]*tap[p[i]] + y[a[i]]
                   - y[b[i]])/((tam-tap)[p[i]])
        y[p[i]] = y[a[i]] + (x[p[i]] - x[a[i]])*tap[p[i]]

# Method of characteristics table
A[:, 0] = np.around(Rp, 2)
A[:, 1] = np.around(Rm, 2)
A[:, 2] = np.around(theta, 2)
A[:, 3] = np.around(nu, 2)
A[:, 4] = M
A[:, 5] = np.around(mu, 2)
A[:, 6] = np.around(theta + mu, 2)
A[:, 7] = np.around(theta - mu, 2)
A[:, 8] = np.around(x, 2)
A[:, 9] = np.around(y, 2)

# Arrays of wall coordinates
print('Array of wall coordinates')
xw = np.zeros(Nc+1)
yw = np.zeros(Nc+1)
j = W[0, :].astype(int) + Nc - 1
xw[1:] = x[j]
yw[1:] = y[j]
xw[0] = 0  # Wall starts at x = 0
yw[0] = 1  # Wall starts at y = 1

# Array of (x,y) coordinates of points along each characteristic line.
print('Array of characteristic line coordinates')
CX = np.zeros((Nc, Nc+2))
CY = np.zeros((Nc, Nc+2))
p = C[:, :] + Nc - 1  # Index of each point along each characteristic
CX[:, 1:] = x[p]  # x-coordinate of each point along each characteristic
CY[:, 1:] = y[p]  # y-coordinate of each point along each characteristic
CX[:, 0] = 0  # All characteristics begin at x = 0
CY[:, 0] = 1  # All characteristics begin at y = 1

# plt.figure(figsize=(10, 10))
# plt.plot(xw, yw, 'k')  # Plot wall
# for i in np.arange(Nc):  # Plot characteristic lines
#     plt.plot(CX[i], CY[i], 'b', linewidth=0.1)

# plt.xlabel('x position')
# plt.ylabel('y position')
# plt.axis('scaled')
# plt.title('Nozzle geometry plot')
# plt.show()
print('A/A* = %f' % yw[-1])
t_end = time.perf_counter()  # Stop timer
print(f'Complete in {round(t_end-t_start, 2)} seconds')
