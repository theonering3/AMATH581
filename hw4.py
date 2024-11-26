import numpy as np
from scipy.sparse import diags, kron, identity
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)

# Grid size and step size
n = 8  # Number of grid points in each direction
dx = 20/8
dy = dx  # Assuming square grid cells

# Laplacian in x and y directions for a single dimension
main_diag = -2 * np.ones(n)
side_diag = np.ones(n)

# Create 1D finite-difference Laplacian (D2) for x and y directions
D2_1D = diags([main_diag, side_diag, side_diag], [0, -1, 1], shape=(n, n))

# Add periodic boundary conditions for 1D Laplacian (optional if not periodic)
D2_1D = D2_1D.toarray()
D2_1D[0, -1] = 1
D2_1D[-1, 0] = 1
#D2_1D = diags(D2_1D)

# Full 2D Laplacian (64x64) using Kronecker products
A = (kron(identity(n), D2_1D) + kron(D2_1D, identity(n)))
A=A/(dx**2)


# First derivative in x direction
B_1D = diags([-1, 1], [-1, 1], shape=(n, n))
B_1D = B_1D.toarray()
B_1D[0, -1] = -1   # Periodic boundary condition (optional)
B_1D[-1, 0] = 1


# First derivative in y direction
C_1D = diags([-1, 1], [-1, 1], shape=(n, n))
C_1D = C_1D.toarray()
C_1D[0, -1] = -1  # Periodic boundary condition (optional)
C_1D[-1, 0] = 1
C_1D=C_1D


# Expand to 2D using Kronecker products
B = kron(B_1D,identity(n))
C = kron(identity(n),C_1D)

# Convert to dense arrays for visualization, if desired
A = A.toarray()
B = B.toarray()
C = C.toarray()
B=B/(2*dx)
C=C/(2*dy)
A1=A
A2=B
A3=C
plt.spy(A3)

print(A3)

