import numpy as np
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define parameters
Lx, Ly = 20, 20  # Domain size
nf = 64  # Number of grid points
beta = 1  # Parameter beta
D1, D2 = 0.1, 0.1  # Diffusion coefficients
m = 1  # Number of spirals
tspan = (0, 4)
teval = np.arange(0, 4.5, 0.5)
num_steps = len(teval)

# Define spatial domain
xf = np.linspace(-Lx / 2, Lx / 2, nf+1)
yf = np.linspace(-Ly / 2, Ly / 2, nf+1)
xf = xf[:nf]
yf = yf[:nf]
Xf, Yf = np.meshgrid(xf, yf)

# Initialize u and v (spiral initial conditions)
u = np.tanh(np.sqrt(Xf ** 2 + Yf ** 2)) * np.cos(m * np.angle(Xf + 1j * Yf) - np.sqrt(Xf ** 2 + Yf ** 2))
v = np.tanh(np.sqrt(Xf ** 2 + Yf ** 2)) * np.sin(m * np.angle(Xf + 1j * Yf) - np.sqrt(Xf ** 2 + Yf ** 2))

# Define spectral k values
kx = 2 * np.pi / Lx * np.fft.fftfreq(nf, d=1 / nf)
ky = 2 * np.pi / Ly * np.fft.fftfreq(nf, d=1 / nf)
KX, KY = np.meshgrid(kx, ky)
Laplacianf = -(KX ** 2 + KY ** 2)

# Fourier transform of initial conditions
u_hat = fft2(u)
v_hat = fft2(v)


# Define the right-hand side in Fourier domain
def rhs_fourier(t, w_hat):
    u_hat = w_hat[:nf * nf].reshape((nf, nf))  # Split into u and v Fourier modes
    v_hat = w_hat[nf * nf:].reshape((nf, nf))

    # Transform back to physical space
    u = np.real(ifft2(u_hat))
    v = np.real(ifft2(v_hat))

    # Compute A^2 in physical space
    A2 = u ** 2 + v ** 2

    # Compute lambda and omega
    lambda_A2 = 1-A2
    omega_A2 = -beta * A2

    # Compute nonlinear terms in physical space
    NL_u = lambda_A2 * u - omega_A2 * v
    NL_v = omega_A2 * u + lambda_A2 * v

    # Transform nonlinear terms back to Fourier space
    NL_u_hat = fft2(NL_u)
    NL_v_hat = fft2(NL_v)

    # Compute RHS in Fourier space
    u_rhs = NL_u_hat + D1 * Laplacianf * u_hat
    v_rhs = NL_v_hat + D2 * Laplacianf * v_hat

    # Stack real and imaginary parts for ODE solver
    return np.hstack([u_rhs.ravel(), v_rhs.ravel()])


# Initial conditions for ODE solver
w_hat0 = np.hstack([u_hat.ravel(), v_hat.ravel()])

# Solve using RK45
solf = solve_ivp(rhs_fourier, tspan, w_hat0, t_eval=teval, method='RK45')

# Extract solution in Fourier domain
A1 = solf.y


n = 30
def cheb(N):
    if N == 0:
        D = 0.0
        x = np.array([1.0])
    else:
        n = np.arange(0, N + 1)
        x = np.cos(np.pi * n / N).reshape(N + 1, 1)
        c = (np.hstack(([2.], np.ones(N - 1), [2.])) * (-1)**n).reshape(N + 1, 1)
        X = np.tile(x, (1, N + 1))
        dX = X - X.T
        D = (np.dot(c, 1.0 / c.T)) / (dX + np.eye(N + 1))
        D -= np.diag(np.sum(D.T, axis=0))

    return D, x.flatten()

# Parameters

D, x_cheb = cheb(n)
D[n,:]=0
D[0,:]=0

#x_cheb = 10 * x_cheb  # Scale x to [-10, 10]
L2 = np.dot(D, D) / ((20 / 2)**2)  # Scale the Laplacian to match the domain

# Create 2D Laplacian
Laplacian = np.kron(L2, np.eye(n + 1)) + np.kron(np.eye(n + 1), L2)

# Rescale y to match x
x, y = x_cheb, x_cheb
X, Y = np.meshgrid(x, y)
X=X*10
Y=Y*10

# Initial conditions
theta = np.angle(X + 1j * Y)  # Ensure 2D input here
u0 = np.tanh(np.sqrt(X**2 + Y**2)) * np.cos(m * theta - np.sqrt(X**2 + Y**2))
v0 = np.tanh(np.sqrt(X**2 + Y**2)) * np.sin(m * theta - np.sqrt(X**2 + Y**2))
U0 = u0.flatten()
V0 = v0.flatten()
y0 = np.concatenate([U0, V0])

# Define the system of equations
def reaction_diffusion(t, y):
    U = y[: (n + 1)**2].reshape((n + 1, n + 1))  # Reshape to 2D
    V = y[(n + 1)**2 :].reshape((n + 1, n + 1))  # Reshape to 2D
    A = U**2 + V**2
    λ = 1 - A
    ω = -beta * A

    # Reaction terms
    U_reaction = λ * U - ω * V
    V_reaction = ω * U + λ * V

    # Diffusion terms
    U_diffusion = D1 * Laplacian.dot(U.flatten()).reshape(U.shape)
    V_diffusion = D2 * Laplacian.dot(V.flatten()).reshape(V.shape)

    # Combine
    dUdt = U_reaction + U_diffusion
    dVdt = V_reaction + V_diffusion

    return np.concatenate([dUdt.flatten(), dVdt.flatten()])

# Solve using RK45
sol = solve_ivp(
    reaction_diffusion, tspan, y0, method="RK45", t_eval=teval
)

# Reshape solution and save
A2 = sol.y

print(A1)
print(A2)