import numpy as np
from numba import jit, njit

def K(x, y, ret_0=False):
    ans = np.log(np.abs(x - y))
    if ret_0:
        ans = np.nan_to_num(ans)
    return ans

#right hand side
def V(x):
    return 1.0 * np.ones_like(x)

#x - сетка (x0, x1... x_n).
#RHS for exact solution u(x)=1
#В ans[i] должен быть интеграл от a до b от функции K(x[i], y)dy
def V0(x, a, b):
    ans = np.zeros_like(x)
    for i, _ in enumerate(ans):
        if not (x[i] >= a and x[i] <= b):
            raise "x is out of range"
        if np.isclose(x[i], a):
            ans[i] = (b - x[i]) * np.log(b - x[i]) + a - b
        elif np.isclose(x[i], b):
            ans[i] = (x[i] - a) * np.log(x[i] - a) + a - b
        else:
            ans[i] = (b - x[i]) * np.log(b - x[i]) + (x[i] - a) * np.log(x[i] - a) + a - b
    return ans

def V0pol(x, a, b, alpha):
    ans = np.zeros_like(x)
    for i, _ in enumerate(ans):
        if not (x[i] >= a and x[i] <= b):
            raise "x is out of range"
        ans[i] = (x[i] - a) ** (alpha + 1) / (alpha + 1) + (b - x[i]) ** (alpha + 1) / (alpha + 1)
    return ans

def V0_y(x, a, b):
    ans = np.zeros_like(x)
    for i, _ in enumerate(ans):
        if not (x[i] >= a and x[i] <= b):
            raise "x is out of range"
        if np.isclose(x[i], a):
            ans[i] = -0.5 * (b - x[i]) ** 2 * (0.5 - np.log(b - x[i]))
        elif np.isclose(x[i], b):
             ans[i] = 0.5 * (x[i] - a) ** 2 * (0.5 - np.log(x[i] - a))
        else:
            ans[i] = 0.5 * (x[i] - a) ** 2 * (0.5 - np.log(x[i] - a)) - 0.5 * (b - x[i]) ** 2 * (0.5 - np.log(b - x[i]))

    return ans

def V_y(x, a, b):
    return np.arange(x.shape[0]) * (b - a) / (x.shape[0])

#x, y - числа, вернуть функцию K(x,y)=|x-y|^alpha; alpha=0.1
def Kpol(x, y, alpha=-0.5,  ret_0=False):
    if np.isclose(x, y):
        return 0.0
    else:
        return (np.abs(x - y))**alpha


#x - сетка (x0, x1... x_n).
#RHS for exact solution u(x)=1
#В ans[i] должен быть интеграл от a до b от функции K(x[i], y)dy
"""
def polyint(x, y, alpha=-0.5):
    print('raised polyint')
    return (np.abs(x - y))**alpha * (y - x) / (alpha + 1)

def Vpol(x, a, b, alpha=0.1):
    return polyint(x, b, alpha) - polyint(x, a, alpha)
"""

def get_A(N, X, x_cap):
    A = np.zeros((len(x_cap), len(x_cap)))
    h = x_cap[1] - x_cap[0]
    #A is calculated from exact integration formula
    for j, _ in enumerate(x_cap):
        for i, _ in enumerate(x_cap):
            #if i == j or i+1 == 1:
            eps = 1e-5
            if np.abs(x_cap[j] - X[i]) <= eps:
                A[j,i] = (X[i+1] - x_cap[j]) * np.log(np.abs(x_cap[j] - X[i+1])) + X[i] - X[i+1]
            elif np.abs(x_cap[j] - X[i+1]) <= eps:
                A[j,i] = (x_cap[j] - X[i]) * np.log(np.abs(x_cap[j] - X[i])) + X[i] - X[i+1]
            else:
                A[j,i] = (x_cap[j] - X[i]) * np.log(np.abs(x_cap[j] - X[i])) + (X[i+1] - x_cap[j]) * \
                         np.log(np.abs(x_cap[j] - X[i+1])) + X[i] - X[i+1]
    return A

def get_A_pol(N, X, x_cap, alpha):
    A = np.zeros((len(x_cap), len(x_cap)))
    h = x_cap[1] - x_cap[0]
    for j, _ in enumerate(x_cap):
        for i, _ in enumerate(x_cap):
            if x_cap[j] > X[i+1]:
                A[j, i] = (x_cap[j] - X[i]) ** (alpha + 1) / (alpha + 1) - (x_cap[j] - X[i+1]) ** (alpha + 1) / (alpha + 1)
            elif x_cap[j] < X[i]:
                A[j, i] = (X[i+1] - x_cap[j]) ** (alpha + 1) / (alpha + 1) - (X[i] - x_cap[j]) ** (alpha + 1) / (alpha + 1)
            else:
                A[j, i] = (x_cap[j] - X[i]) ** (alpha + 1) / (alpha + 1) + (X[i+1] - x_cap[j]) ** (alpha + 1) / (alpha + 1)
    return A

def get_A_ji_pol(x_cap_j, X_i, X_ip1, alpha):
    if x_cap_j > X_ip1:
        return (x_cap_j - X_i) ** (alpha + 1) / (alpha + 1) - (x_cap_j - X_ip1) ** (alpha + 1) / (alpha + 1)
    elif x_cap_j < X_i:
        return (X_ip1 - x_cap_j) ** (alpha + 1) / (alpha + 1) - (X_i - x_cap_j) ** (alpha + 1) / (alpha + 1)
    else:
        return (x_cap_j - X_i) ** (alpha + 1) / (alpha + 1) + (X_ip1 - x_cap_j) ** (alpha + 1) / (alpha + 1)

#Xs/Xe - Xstart, Xend
#@jit(nopython=True)
def ex_int(Xs, Xe, Ys, Ye, x0, x1):
    if np.isclose(Ye - x1, 0) and np.isclose(Xs - x0, 0):
        raise ZeroDivisionError
    elif np.isclose(Ye - x1, 0):
        s1 = np.sign(Xs - x0)
    elif np.isclose(Xs - x0, 0):
        s1 = np.sign(Ye - x1)
    else:
        s1 = np.sqrt((Xs - x0) ** 2 + (Ye - x1) ** 2) / (Ye - x1) / (Xs - x0)
        """
        v1 = np.sqrt((Xs - x0) ** 2 + (Ye - x1) ** 2) / (Ye - x1) / (Xs - x0)
        v2 = 0.5 * (np.sign(Xs - x0) / (Ye - x1) * np.sqrt(1 + ((Ye - x1) / (Xs - x0)) ** 2) +
                   np.sign(Ye - x1) / (Xs - x0) * np.sqrt(1 + ((Xs - x0) / (Ye - x1)) ** 2))
        assert np.isclose(v1, v2)

        s1 = 0.5 * (np.sign(Xs - x0) / (Ye - x1) * np.sqrt(1 + ((Ye - x1) / (Xs - x0)) ** 2) +
                   np.sign(Ye - x1) / (Xs - x0) * np.sqrt(1 + ((Xs - x0) / (Ye - x1)) ** 2))
        """
    if np.isclose(Xe - x0, 0) and np.isclose(Ye - x1, 0):
        raise ZeroDivisionError
    elif np.isclose(Xe - x0, 0):
        s2 = np.sign(Ye - x1)
    elif np.isclose(Ye - x1, 0):
        s2 = np.sign(Xe - x0)
    else:
        s2 = np.sqrt((Xe - x0) ** 2 + (Ye - x1) ** 2) / (Ye - x1) / (Xe - x0)

    if np.isclose(Xs - x0, 0) and np.isclose(Ys - x1, 0):
        raise ZeroDivisionError
    elif np.isclose(Xs - x0, 0):
        s3 = np.sign(Ys - x1)
    elif np.isclose(Ys - x1, 0):
        s3 = np.sign(Xs - x0)
    else:
        s3 = np.sqrt((Xs - x0) ** 2 + (Ys - x1) ** 2) / (Ys - x1) / (Xs - x0)

    if np.isclose(Xe - x0, 0) and np.isclose(Ys - x1, 0):
        raise ZeroDivisionError
    elif np.isclose(Xe - x0, 0):
        s4 = np.sign(Ys - x1)
    elif np.isclose(Ys - x1, 0):
        s4 = np.sign(Xe - x0)
    else:
        s4 = np.sqrt((Xe - x0) ** 2 + (Ys - x1) ** 2) / (Ys - x1) / (Xe - x0)

    return -(s1 - s2 - s3 + s4)

def rhs_2D(X, a, b):
    ans = np.zeros((len(X[0]), len(X[1])))
    for i0, i0_val in enumerate(X[0]):
        for i1, i1_val in enumerate(X[1]):
            if not (a[0] <= i0_val <= b[0] and a[1] <= i1_val <= b[1]):
                raise "x is out of range"
            ans[i0, i1] = ex_int(a[0], b[0], a[1], b[1], i0_val, i1_val)

    return ans.reshape(-1, 1)

#1 - начало соответствующего ребра, x2 - конец
def ex_int3D(bx1, bx2, by1, by2, bz1, bz2, cx, cy, cz, k):
    """
    #Вычисление интеграла от 1 / R
    def arctan_base(args):
        if len(args) != 4:
            raise "Incorrect number of arguments"
        return args[3] * args[0] ** 2 * np.arctan(args[1] * args[2] / args[0] / np.linalg.norm([args[0], args[1], args[2]]))

    def log_base(args):
        if len(args) != 4:
            raise "Incorrect number of arguments"
        root = np.linalg.norm([args[0], args[1], args[2]])
        return args[3] * args[0] * args[1] * np.log(np.abs((root + args[2]) / (root - args[2])))

    arctan_base_args = [(z1, x1, y1, 1), (y1, x1, z1, 1),
                       (x1, y1, z1, 1), (z1, x2, y1, -1),
                       (y1, x2, z1, -1), (x2, y1, z1, -1),
                       (z1, x1, y2, -1), (y2, x1, z1, -1),
                       (x1, y2, z1, -1), (z1, x2, y2, 1),
                       (y2, x2, z1, 1), (x2, y2, z1, 1),
                       (z2, x1, y1, -1), (y1, x1, z2, -1),
                       (x1, y1, z2, -1), (z2, x2, y1, 1),
                       (y1, x2, z2, 1), (x2, y1, z2, 1),
                       (z2, x1, y2, 1), (y2, x1, z2, 1),
                       (x1, y2, z2, 1), (z2, x2, y2, -1),
                       (y2, x2, z2, -1), (x2, y2, z2, -1)]

    log_base_args = [(y1, z1, x1, -1), (x1, z1, y1, -1),
                    (x1, y1, z1, -1), (y1, z1, x2, 1),
                    (x2, z1, y1, 1), (x2, y1, z1, 1),
                    (y2, z1, x1, 1), (x1, z1, y2, 1),
                    (x1, y2, z1, 1), (y2, z1, x2, -1),
                    (x2, z1, y2, -1), (y1, z2, x1, -1),
                    (x1, z2, y1, 1), (x1, y1, z2, 1),
                    (y2, z2, x1, -1), (x1, z2, y2, -1),
                    (x1, y2, z2, -1), (y1, z2, x2, -1),
                    (x2, z2, y1, -1), (x2, y1, z2, -1),
                    (x2, y2, z1, 1), (y2, z2, x2, 1),
                    (x2, z2, y2, 1), (x2, y2, z2, 1)]

    I1 = 0.0
    for arg in arctan_base_args:
        I1 += arctan_base(arg)
    for arg in log_base_args:
        I1 += log_base(arg)
    I1 /= 2
    """
    x1 = bx1 - cx
    x2 = bx2 - cx
    y1 = by1 - cy
    y2 = by2 - cy
    z1 = bz1 - cz
    z2 = bz2 - cz

    I1 = (1/2)*(z1**2*np.arctan((x1*y1)/(z1*np.sqrt(x1**2 + y1**2 + z1**2))) + y1**2*np.arctan((x1*z1)/(y1*np.sqrt(x1**2 + y1**2 + z1**2))) + x1**2*np.arctan((y1*z1)/(x1*np.sqrt(x1**2 + y1**2 + z1**2))) - z1**2*np.arctan((x2*y1)/(z1*np.sqrt(x2**2 + y1**2 + z1**2))) - y1**2*np.arctan((x2*z1)/(y1*np.sqrt(x2**2 + y1**2 + z1**2))) - x2**2*np.arctan((y1*z1)/(x2*np.sqrt(x2**2 + y1**2 + z1**2))) -
   z1**2*np.arctan((x1*y2)/(z1*np.sqrt(x1**2 + y2**2 + z1**2))) - y2**2*np.arctan((x1*z1)/(y2*np.sqrt(x1**2 + y2**2 + z1**2))) - x1**2*np.arctan((y2*z1)/(x1*np.sqrt(x1**2 + y2**2 + z1**2))) +
   z1**2*np.arctan((x2*y2)/(z1*np.sqrt(x2**2 + y2**2 + z1**2))) + y2**2*np.arctan((x2*z1)/(y2*np.sqrt(x2**2 + y2**2 + z1**2))) + x2**2*np.arctan((y2*z1)/(x2*np.sqrt(x2**2 + y2**2 + z1**2))) -
   z2**2*np.arctan((x1*y1)/(z2*np.sqrt(x1**2 + y1**2 + z2**2))) - y1**2*np.arctan((x1*z2)/(y1*np.sqrt(x1**2 + y1**2 + z2**2))) - x1**2*np.arctan((y1*z2)/(x1*np.sqrt(x1**2 + y1**2 + z2**2))) +
   z2**2*np.arctan((x2*y1)/(z2*np.sqrt(x2**2 + y1**2 + z2**2))) + y1**2*np.arctan((x2*z2)/(y1*np.sqrt(x2**2 + y1**2 + z2**2))) + x2**2*np.arctan((y1*z2)/(x2*np.sqrt(x2**2 + y1**2 + z2**2))) +
   z2**2*np.arctan((x1*y2)/(z2*np.sqrt(x1**2 + y2**2 + z2**2))) + y2**2*np.arctan((x1*z2)/(y2*np.sqrt(x1**2 + y2**2 + z2**2))) + x1**2*np.arctan((y2*z2)/(x1*np.sqrt(x1**2 + y2**2 + z2**2))) -
   z2**2*np.arctan((x2*y2)/(z2*np.sqrt(x2**2 + y2**2 + z2**2))) - y2**2*np.arctan((x2*z2)/(y2*np.sqrt(x2**2 + y2**2 + z2**2))) - x2**2*np.arctan((y2*z2)/(x2*np.sqrt(x2**2 + y2**2 + z2**2))) -
   2*y1*z1*np.arctanh(x1/np.sqrt(x1**2 + y1**2 + z1**2)) - 2*x1*z1*np.arctanh(y1/np.sqrt(x1**2 + y1**2 + z1**2)) - 2*x1*y1*np.arctanh(z1/np.sqrt(x1**2 + y1**2 + z1**2)) + 2*y1*z1*np.arctanh(x2/np.sqrt(x2**2 + y1**2 + z1**2)) +
   2*x2*z1*np.arctanh(y1/np.sqrt(x2**2 + y1**2 + z1**2)) + 2*x2*y1*np.arctanh(z1/np.sqrt(x2**2 + y1**2 + z1**2)) + 2*y2*z1*np.arctanh(x1/np.sqrt(x1**2 + y2**2 + z1**2)) + 2*x1*z1*np.arctanh(y2/np.sqrt(x1**2 + y2**2 + z1**2)) +
   2*x1*y2*np.arctanh(z1/np.sqrt(x1**2 + y2**2 + z1**2)) - 2*y2*z1*np.arctanh(x2/np.sqrt(x2**2 + y2**2 + z1**2)) - 2*x2*z1*np.arctanh(y2/np.sqrt(x2**2 + y2**2 + z1**2)) - 2*x2*y2*np.arctanh(z1/np.sqrt(x2**2 + y2**2 + z1**2)) +
   2*y1*z2*np.arctanh(x1/np.sqrt(x1**2 + y1**2 + z2**2)) + 2*x1*z2*np.arctanh(y1/np.sqrt(x1**2 + y1**2 + z2**2)) + 2*x1*y1*np.arctanh(z2/np.sqrt(x1**2 + y1**2 + z2**2)) - 2*y1*z2*np.arctanh(x2/np.sqrt(x2**2 + y1**2 + z2**2)) -
   2*x2*z2*np.arctanh(y1/np.sqrt(x2**2 + y1**2 + z2**2)) - 2*x2*y1*np.arctanh(z2/np.sqrt(x2**2 + y1**2 + z2**2)) - 2*y2*z2*np.arctanh(x1/np.sqrt(x1**2 + y2**2 + z2**2)) - 2*x1*z2*np.arctanh(y2/np.sqrt(x1**2 + y2**2 + z2**2)) -
   2*x1*y2*np.arctanh(z2/np.sqrt(x1**2 + y2**2 + z2**2)) + 2*y2*z2*np.arctanh(x2/np.sqrt(x2**2 + y2**2 + z2**2)) + 2*x2*z2*np.arctanh(y2/np.sqrt(x2**2 + y2**2 + z2**2)) + 2*x2*y2*np.arctanh(z2/np.sqrt(x2**2 + y2**2 + z2**2)))

    #Вычисление интеграла без особенности
    y = np.array([(bx1 + bx2) / 2, (by1 + by2) / 2, (bz1+ bz2) / 2]) #где берётся функция для вычисления интеграла
    x_cap = np.array([cx, cy, cz]) #узел коллокации
    R = np.linalg.norm(y - x_cap)

    I2 = 0.0
    #ik = 1j * k

    #С точностью до R^2
    #h = 0.01
    #I2 = ik + ik ** 2 * R / 2 + ik ** 3 * R ** 2 / 6 + ik ** 4 * R ** 3 / 24 + ik ** 5 * R ** 4 / 120 + ik ** 6 * R ** 5 / 720 + ik ** 7 * \
    #    R ** 6 / 720 / 7
    #I2 *= (bx2 - bx1) * (by2 - by1) * (bz2 - bz1)
    I2 = k * 1j * np.sinc(k * R / np.pi) - k ** 2 * R / 2 * np.sinc(k * R / 2 / np.pi) ** 2
    I = (I1 + I2) / (4 * np.pi)

    return I * k ** 2


#def exact_int(Xs: npt.ArrayLike, Xe: npt.ArrayLike, Ys: npt.ArrayLike, Ye: npt.ArrayLike, x0: npt.ArrayLike, x1: npt.ArrayLike) -> np.ndarray:
def exact_int(Xs, Xe, Ys, Ye, x0, x1) -> np.ndarray:
    calc_type = np.double
    atol = 1e-12
    rtol = 1e-8
    s1 = np.empty(shape=(Xs.shape), dtype=calc_type)
    s2 = np.empty(shape=(Xs.shape), dtype=calc_type)
    s3 = np.empty(shape=(Xs.shape), dtype=calc_type)
    s4 = np.empty(shape=(Xs.shape), dtype=calc_type)

    Xsx0_c = np.empty(shape=(Xs.shape), dtype=np.bool_)
    Xsx1_c = np.empty(shape=(Xs.shape), dtype=np.bool_)
    Xex0_c = np.empty(shape=(Xs.shape), dtype=np.bool_)
    Xex1_c = np.empty(shape=(Xs.shape), dtype=np.bool_)
    Ysx0_c = np.empty(shape=(Xs.shape), dtype=np.bool_)
    Ysx1_c = np.empty(shape=(Xs.shape), dtype=np.bool_)
    Yex0_c = np.empty(shape=(Xs.shape), dtype=np.bool_)
    Yex1_c = np.empty(shape=(Xs.shape), dtype=np.bool_)

    Xsx0_c = np.isclose(Xs - x0, 0.0, rtol=rtol, atol=atol)
    Xsx1_c = np.isclose(Xs - x1, 0.0, rtol=rtol, atol=atol)
    Xex0_c = np.isclose(Xe - x0, 0.0, rtol=rtol, atol=atol)
    Xex1_c = np.isclose(Xe - x1, 0.0, rtol=rtol, atol=atol)
    Ysx0_c = np.isclose(Ys - x0, 0.0, rtol=rtol, atol=atol)
    Ysx1_c = np.isclose(Ys - x1, 0.0, rtol=rtol, atol=atol)
    Yex0_c = np.isclose(Ye - x0, 0.0, rtol=rtol, atol=atol)
    Yex1_c = np.isclose(Ye - x1, 0.0, rtol=rtol, atol=atol)

    s1[Yex1_c] = np.sign((Xs - x0)[Yex1_c])
    s1[Xsx0_c] = np.sign((Ye - x1)[Xsx0_c])
    s1[np.logical_and(np.logical_not(Yex1_c), np.logical_not(Xsx0_c))] = np.sqrt((Xs - x0) ** 2 + (Ye - x1) ** 2) / (Ye - x1) / (Xs - x0)

    s2[Yex1_c] = np.sign((Xe - x0)[Yex1_c])
    s2[Xex0_c] = np.sign((Ye - x1)[Xex0_c])
    s2[np.logical_and(np.logical_not(Yex1_c), np.logical_not(Xex0_c))] = np.sqrt((Xe - x0) ** 2 + (Ye - x1) ** 2) / (Ye - x1) / (Xe - x0)

    s3[Ysx1_c] = np.sign((Xs - x0)[Ysx1_c])
    s3[Xsx0_c] = np.sign((Ys - x1)[Xsx0_c])
    s3[np.logical_and(np.logical_not(Ysx1_c), np.logical_not(Xsx0_c))] = np.sqrt((Xs - x0) ** 2 + (Ys - x1) ** 2) / (Ys - x1) / (Xs - x0)

    s4[Ysx1_c] = np.sign((Xe - x0)[Ysx1_c])
    s4[Xex0_c] = np.sign((Ys - x1)[Xex0_c])
    s4[np.logical_and(np.logical_not(Ysx1_c), np.logical_not(Xex0_c))] = np.sqrt((Xe - x0) ** 2 + (Ys - x1) ** 2) / (Ys - x1) / (Xe - x0)

    return -(s1 - s2 - s3 + s4)
