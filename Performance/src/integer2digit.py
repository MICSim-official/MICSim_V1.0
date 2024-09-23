import numpy as np
import torch
def dec2digit_unsign(x, n, N):
    y = x.clone()
    out = []
    scale_list = []
    unit = 2 ** n
    rest = y
    for i in range(int(np.ceil(N / n))):
        y = rest % unit
        rest = rest // unit
        out.append(y)
        scale_list.append(unit ** i)
    return out, scale_list


def dec2digit_sign_2s(x, n, N):
    y = x.clone()
    out = []
    scale_list = []
    base = 2 ** (N - 1)
    y[x >= 0] = 0
    y[x < 0] = 1
    rest = x + base * y
    out.append(y)
    scale_list.append(-base)
    unit = 2 ** n
    for i in range(int(np.ceil((N - 1) / n))):
        y = rest % unit
        rest = rest // unit
        out.append(y)
        scale_list.append(unit ** i)

    return out, scale_list


def dec2digit_sign_np(x, n, N):
    N = N - 1
    y_p =  x.clone()
    y_n =  x.clone()
    y_p[y_p < 0] = 0
    y_n[y_n > 0] = 0
    out = []
    scale_list = []
    unit = 2 ** n
    rest_p = y_p
    rest_n = y_n
    for i in range(int(np.ceil(N / n))):
        y_p = rest_p % unit
        y_n = rest_n % unit
        rest_p = rest_p // unit
        rest_n = rest_n // unit
        out.append([y_p,y_n])
        scale_list.append([unit ** i,-unit ** i])
    return out, scale_list
