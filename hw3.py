import numpy as np
from scipy.integrate import odeint,solve_ivp
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.sparse.linalg import eigs
from math import pi

# Define parameters
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



L = 4
k = 1
dx=0.1
xspan = np.arange(-L, L + dx, dx)
x=xspan[1:-1]
N=len(xspan)
n = len(x)
A = np.zeros((n, n))
for i in range(n):
    A[i,i]= -2-(dx**2) * (x[i]**2)
    if i<n-1:
        A[i+1,i]=1
        A[i,i+1]=1
A[0,0]+=4/3
A[0,1]-=1/3
A[-1,-1]+=4/3
A[-1,-2]-=1/3

eval,evec=eigs(-A,k=5,which='SM')
M=np.vstack([4/3*evec[0,:]-1/3*evec[1,:],evec,4/3*evec[-1,:]-1/3*evec[-2,:]])
fevec=np.zeros((N,5))
feval=np.zeros(5)
for i in range(5):
    norm=np.sqrt(np.trapz(M[:,i]**2,xspan))
    fevec[:,i]=np.abs(M[:,i]/norm)
feval=np.sort(eval[:5]/dx**2)
A3=fevec
A4=feval
for i in range(5):
   plt.plot(xspan,A3[:,i])

print(A3)
print(A4)

for i in range(5):
    plt.plot(xspan, A3[:, i], label=f'ϕ_{i + 1}(x)')
plt.title('Normalized Eigenfunctions of the Harmonic Oscillator')
plt.xlabel('x')
plt.ylabel('Eigenfunction value')
plt.legend()
plt.grid()
#plt.show()



L2 = 2
k = 1
dx=0.1
xspan2 = np.arange(-L2, L2 + dx, dx)
n=len(xspan2)
tol=1e-4

def efunc(x,y,e,gamma):
    return [y[1],(gamma*y[0]**2+x**2-e)*y[0]]

peval=np.zeros(2)
neval=np.zeros(2)
pevec=np.zeros((n,2))
nevec=np.zeros((n,2))

for gamma in [0.05,-0.05]:
    beta_start=0.1
    a=1e-6
    for mode in range(2):
        da=0.1
        for i in range(100):
            beta=beta_start
            dbeta=0.2
            for j in range(100):
                y0=[a,np.sqrt(L2**2-beta)*a]
                yt_sol=solve_ivp(lambda x,y:efunc(x,y,beta,gamma),[xspan2[0],xspan2[-1]],y0,t_eval=xspan2)
                y_sol=yt_sol.y.T
                x_sol=yt_sol.t

                phi=y_sol[-1,1]+np.sqrt(L2**2-beta)*y_sol[-1,0]
                if abs(phi)<tol:
                    break
                if(-1)**(mode)*phi>0:
                    beta+=dbeta
                else:
                    beta-=dbeta
                    dbeta/=2

            adjust=np.trapz(y_sol[:,0]**2,x=x_sol)
            if abs(adjust-1)<tol:
                break
            elif adjust <1:
                a+=da
            else:
                a-=da/2
                da/=2
        beta_start=beta+0.2

        if gamma>0:
            peval[mode]=beta

            pevec[:,mode]=np.abs(y_sol[:,0])
        else:
            neval[mode] = beta

            pevec[:, mode] = np.abs(y_sol[:, 0])

A5=pevec
A6=peval
A7=pevec
A8=neval

print(A5)
print(A7)
print(A6,A8)



for i in range(2):
    plt.plot(xspan2, A5[:, i], label=f'ϕ_{i + 1}(x)')
    plt.title('Normalized Eigenfunctions of the Harmonic Oscillator')
    plt.xlabel('x')
    plt.ylabel('Eigenfunction value')
    plt.legend()
    plt.grid()
    plt.show()
for j in range(2):
    plt.plot(xspan2, A7[:, j], label=f'ϕ_{j + 1}(x)')
    plt.title('Normalized Eigenfunctions of the Harmonic Oscillator')
    plt.xlabel('x')
    plt.ylabel('Eigenfunction value')
    plt.legend()
    plt.grid()
    plt.show()


# Given parameters
K = 1
e = 1
gamma = 0  # As per the problem statement
a=1


# Tolerance values
tolerances = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

# Initial conditions
x_span = [-L2, L2]
y0 = [a, np.sqrt(L2**2-e)*a]

# Placeholder to store average step sizes
average_step_sizes_RK45 = []
average_step_sizes_RK23 = []
average_step_sizes_Radau = []
average_step_sizes_BDF = []


# Define the differential equation system based on the provided equation
def hw1_rhs_a(x,y,e):
    return [y[1],(x**2-e)*y[0]]


# Loop through each tolerance level
for tol in tolerances:
    options = {'rtol': tol, 'atol': tol}

    # Solve using RK45
    sol_RK45 = solve_ivp(hw1_rhs_a, x_span, y0, method='RK45', args=(e,), **options)
    average_step_sizes_RK45.append(np.mean(np.diff(sol_RK45.t)))

    # Solve using RK23
    sol_RK23 = solve_ivp(hw1_rhs_a, x_span, y0, method='RK23', args=(e,), **options)
    average_step_sizes_RK23.append(np.mean(np.diff(sol_RK23.t)))

    # Solve using Radau
    sol_Radau = solve_ivp(hw1_rhs_a, x_span, y0, method='Radau', args=(e,), **options)
    average_step_sizes_Radau.append(np.mean(np.diff(sol_Radau.t)))

    # Solve using BDF
    sol_BDF = solve_ivp(hw1_rhs_a, x_span, y0, method='BDF', args=(e,), **options)
    average_step_sizes_BDF.append(np.mean(np.diff(sol_BDF.t)))

# Convert tolerance values and average step sizes to logarithmic scale for fitting
log_tolerances = np.log10(tolerances)
log_avg_step_sizes_RK45 = np.log10(average_step_sizes_RK45)
log_avg_step_sizes_RK23 = np.log10(average_step_sizes_RK23)
log_avg_step_sizes_Radau = np.log10(average_step_sizes_Radau)
log_avg_step_sizes_BDF = np.log10(average_step_sizes_BDF)

# Calculate the slopes using polyfit
slope_RK45, _ = np.polyfit(log_avg_step_sizes_RK45, log_tolerances, 1)
slope_RK23, _ = np.polyfit(log_avg_step_sizes_RK23, log_tolerances, 1)
slope_Radau, _ = np.polyfit(log_avg_step_sizes_Radau, log_tolerances, 1)
slope_BDF, _ = np.polyfit(log_avg_step_sizes_BDF, log_tolerances, 1)

# Save slopes as a 4x1 vector
A9 = np.array([slope_RK45, slope_RK23, slope_Radau, slope_BDF])

# Plotting the log-log scale of average step size vs tolerance
plt.figure(figsize=(10, 6))
plt.plot(log_avg_step_sizes_RK45, log_tolerances, label=f'RK45, Slope = {slope_RK45:.2f}')
plt.plot(log_avg_step_sizes_RK23, log_tolerances, label=f'RK23, Slope = {slope_RK23:.2f}')
plt.plot(log_avg_step_sizes_Radau, log_tolerances, label=f'Radau, Slope = {slope_Radau:.2f}')
plt.plot(log_avg_step_sizes_BDF, log_tolerances, label=f'BDF, Slope = {slope_BDF:.2f}')
plt.xlabel('Log of Average Step Size')
plt.ylabel('Log of Tolerance')
plt.legend()
plt.grid(True)
plt.title('Log-Log Scale: Average Step Size vs Tolerance')
#plt.show()

print(A9)


def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


h=np.array([np.ones_like(xspan),2*xspan,4*xspan**2-2,8*xspan**3-12*xspan,16*xspan**4-48*xspan**2+12])

phi=np.zeros((N,5))
erps_a=np.zeros(5)
erps_b=np.zeros(5)
er_a=np.zeros(5)
er_b=np.zeros(5)

for j in range(5):
    phi[:,j]=np.array((np.exp((-(xspan**2))/2)*(h[j,:]))/(np.sqrt(factorial(j)*(2**j)*(np.sqrt(pi))))).T
for j in range(5):
    erps_a[j]=np.trapz((abs(A1[:,j])-abs(phi[:,j]))**2,xspan)
    erps_b[j]=np.trapz((abs(A3[:,j])-abs(phi[:,j]))**2,xspan)
    er_a[j]=100*(abs(A2[j]-(2*(j+1)-1))/(2*(j+1)-1))
    er_b[j]=100*(abs(A4[j]-(2*(j+1)-1))/(2*(j+1)-1))

A10=erps_a
A12=erps_b
A11=er_a
A13=er_b

print(A10)
print(A12)
print(A11)
print(A13)