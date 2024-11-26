import numpy as np

def f(x):
    return x * np.sin(3 * x) - np.exp(x)

def f_d(x):
    return 3 * x * np.cos(3 * x) + np.sin(3 * x) - np.exp(x)

def nr(x0):
    x = [x0]
    for i in range(100):
        x_next = x[-1] - f(x[-1]) / f_d(x[-1])
        x=np.append(x,x_next)

        if abs(f(x[-2])) <= 1e-6:
            break
    return np.array(x), i + 1


def bisection(l, r):
    mid_p = []
    for i in range(100):
        mid = (l +r) / 2
        mid_p=np.append(mid_p,mid)
        if f(mid) == 0 or abs(f(mid)) <= 1e-6:
            break
        elif f(mid) > 0:
            l = mid
        else:
            r = mid
    return np.array(mid_p), i + 1


A1, i_nr = nr(-1.6)
A2, i_b = bisection(-0.7, -0.4)
A3 = np.array([i_nr, i_b])

A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([1,0])
y = np.array([0,1])
z = np.array([1,2,-1])

A4 = A + B
A5 = (3 * x - 4 * y)
A6 = (np.dot(A, x))
A7 = (np.dot(B, x - y))
A8 = (np.dot(D, x))
A9 = (np.dot(D, y) + z)
A10 = np.dot(A, B)
A11 = np.dot(B, C)
A12 = np.dot(C, D)

print("A1:", A1)
print("A2:", A2)
print("A3:", A3)
print("A4:", A4)
print("A5:", A5)
print("A6:", A6)
print("A7:", A7)
print("A8:", A8)
print("A9:", A9)
print("A10:", A10)
print("A11:", A11)
print("A12:", A12)