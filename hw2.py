import numpy as np
from scipy.integrate import odeint,solve_ivp
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.sparse.linalg import eigs
from math import pi

K = 1
L = 4  # Range for x
n_modes = 5  # Number of eigenfunctions to find
xspan = np.arange(-L, L+0.1, 0.1)


def eigen_func( x,y, e):
    return [y[1],(K * (x ** 2)-e) * y[0]]


def shooting_method(L, tol=1e-4):

    eigenvalues = []
    eigenfunctions = []

    beta=0.1

    for mode in range(5):
        dbeta = 0.2
        for i in range(1000):

            y0=[1,np.sqrt(L**2-beta)]
            #y = odeint(eigen_func, y0, xspan, args=(beta,))
            sol=solve_ivp(lambda x,y:eigen_func(x,y,beta),[xspan[0],xspan[-1]],y0,t_eval=xspan)
            ys=sol.y.T
            phi=ys[-1, 1]+np.sqrt((K*L**2)-beta)*ys[-1,0]
            if abs(phi) < tol:
                break

            if (-1)**(mode)*phi> 0:
                beta += dbeta
            else:
                beta -= dbeta
                dbeta /= 2

        eigenvalues.append(beta)

        norm = np.trapz(abs(ys[:, 0])**2, xspan)
        eigenfunctions.append(abs(ys[:, 0] / np.sqrt(norm)))
        beta+=0.2

    return np.array(eigenfunctions).T, np.array(eigenvalues)


A1, A2 = shooting_method(L)


print("Eigenfunctions (columns correspond to modes):")
print(A1)
print("Eigenvalues:")
print(A2)


for i in range(n_modes):
    plt.plot(xspan, A1[:, i], label=f'ϕ_{i + 1}(x)')
plt.title('Normalized Eigenfunctions of the Harmonic Oscillator')
plt.xlabel('x')
plt.ylabel('Eigenfunction value')
plt.legend()
plt.grid()
