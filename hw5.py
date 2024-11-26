import time
import numpy as np
from scipy.linalg import lu,solve_triangular
from scipy.sparse import diags, kron, identity
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve, bicgstab, gmres, splu
from scipy.integrate import solve_ivp
from scipy.fft import fft2, ifft2, fftfreq
from matplotlib.animation import FuncAnimation


L = 10  # Domain size
n = 64  # Grid size
x1 = np.linspace(-L, L, n)  # x grid
y1 = np.linspace(-L, L, n)  # y grid
X1, Y1 = np.meshgrid(x1, y1)  # mesh grid for plotting
dx = dy = 20/64
nu = 0.001  # viscosity
Lx, Ly = 20, 20
nx, ny = 64, 64
N = nx * ny
tspan = np.arange(0, 4.5, 0.5)  # time span
omega0 = np.exp(-X1 ** 2 - Y1 ** 2 / 20).flatten()


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
A[0,0]=2
A=A/(dx**2)
B=B/(2*dx)
C=C/(2*dy)

P, L, U = lu(A)

x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid(x, y)

w =  np.exp(-X ** 2 - Y ** 2 / 20)
wt0 = w.reshape(N)# Initialize as complex

# Define spectral k values
kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2

# Define the ODE system
def spc_rhs(t, wt2, nx, ny, N, KX, KY, K, nu):
    #wtc = wt2[:N] + 1j * wt2[N:]
    w = wt2.reshape((nx,ny))
    wt=fft2(w)
    psit = -wt / K
    psi=np.real(ifft2(psit)).reshape(N)
    rhs=nu*np.dot(A,wt2)-np.dot(B,psi)*np.dot(C,wt2)+np.dot(C,psi)*np.dot(B,wt2)
    return rhs

starttime=time.time()
solfft = solve_ivp(spc_rhs, [0,4], wt0, t_eval=tspan, args=(nx, ny, N, KX, KY, K, nu),method='RK45')
endtime=time.time()
elapsed_time=endtime-starttime
print(f"Elapsed time: {elapsed_time:.2f} seconds for FFT")
A1=solfft.y.reshape(4096,9)
print(A1)

def solve_stream_function(omega_flat, A, method='direct'):
    #omega_flat = omega_flat.flatten()  # 保证omega是展平的一维数组
    if method == 'direct':
        return np.linalg.solve(A, omega_flat)
    elif method == 'LU':

        Pb = np.dot(P, omega_flat)
        y = solve_triangular(L, Pb, lower=True)
        x = solve_triangular(U, y)
        return x
    elif method == 'BICGSTAB':
        return bicgstab(A, omega_flat, rtol=1e-5)[0]
    elif method == 'GMRES':
        return gmres(A, omega_flat, rtol=1e-4, maxiter=1000)[0]
    else:
        pass


# 涡量方程的右端项
def rhs(t, wt2,  nu, dx, dy, method):
    #w = wt2.reshape((n, n))
    psi = solve_stream_function(wt2, A, method)

    rhs_omega = nu * np.dot(A,wt2.flatten()) -  np.dot(B, psi.flatten()) * np.dot(C,wt2.flatten()) +  np.dot(C, psi.flatten()) * np.dot(B, wt2.flatten())

    return rhs_omega

# Function to solve using various methods
def solve_for_method(method):
    starttime=time.time()
    sol = solve_ivp(
        rhs,
        [0,4],  # time span
        wt0,
        args=(nu, dx, dy, method),
        method='RK45',
        t_eval=tspan
    )
    endtime=time.time()
    elapsed_time=endtime-starttime
    print(f"Elapsed time: {elapsed_time:.2f} seconds for {method}")
    return sol


# Plotting the final solution for each method
def plot_solution_for_solver(omega, X, Y, solver_name, t):
    omega_reshaped = omega.reshape((n, n))
    plt.contourf(X, Y, omega_reshaped, levels=100)
    plt.colorbar()
    plt.title(f'Vorticity at t = {t} using {solver_name}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.draw()
    plt.pause(1)



# Running animation for each method
for method in ['direct', 'LU','BICGSTAB','GMRES']:
    print(f"Animating vorticity solution using {method}...")
    sol = solve_for_method(method)

    #animate_solution(method,sol,n,L)

    # Plot the final time step solution for each method
    #plt.figure(figsize=(5, 4))
    if method=='direct':
        A2=sol.y.reshape(4096,9)
    elif method=='LU':
        A3=sol.y.reshape(4096,9)
