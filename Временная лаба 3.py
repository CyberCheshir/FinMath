from math import log, sqrt, exp
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.stats import norm
import numpy as np
import math

from optionprice import Option

cdf = norm.cdf
ASSET_PRICE = 70
STRIKE_PRICE = 70
SIGMA = 0.4
R = 0.05
T = 2

N = np.arange(20, 401)


def d1(s, x, r, sigma, t):
    return (log(s / x) + (r + pow(sigma, 2) / 2) * t) / (sigma * sqrt(t))


def d2(s, x, r, sigma, t):
    return d1(s, x, r, sigma, t) - sigma * sqrt(t)


# Справедливая цена для европейского Колл опциона
def fair_value_call(s=ASSET_PRICE, x=STRIKE_PRICE, r=R, sigma=SIGMA, t=T):
    return cdf(d1(s, x, r, sigma, t)) * s - cdf(d2(s, x, r, sigma, t)) * x * exp(-r * t)


# Справедливая цена для европейского Пут опциона
def fair_value_put(s=ASSET_PRICE, x=STRIKE_PRICE, r=R, sigma=SIGMA, t=T):
    return cdf(-d2(s, x, r, sigma, t)) * x * exp(-r * t) - cdf(-d1(s, x, r, sigma, t)) * s


# Payoff для Call опциона
def payoff_call(s=ASSET_PRICE, x=STRIKE_PRICE):
    return max(s - x, 0)


# Payoff для Put опциона
def payoff_put(s=ASSET_PRICE, x=STRIKE_PRICE):
    return max(x - s, 0)


# Расчет цены опциона на i-ом шаге по известным цена на i+1-ом шаге
def price_i(v_up, v_down, p, r, t):
    return exp(-r * t) * (p * v_up + (1 - p) * v_down)


# -------------------------


def binominal_call(s=ASSET_PRICE, x=STRIKE_PRICE, r=R, sigma=SIGMA, t=T, n=N):
    """Биномиальная модель для европейского Call опциона"""
    dt = t / n
    return (exp(-r * t) *
            np.sum([comb(n, i) * p(sigma, dt, r) ** i * (1 - p(sigma, dt, r)) ** (n - i) *
                    payoff_call(s * u(sigma, dt) ** i * d(sigma, dt) ** (n - i), x=x) for i in range(n)]))


def binominal_put(s=ASSET_PRICE, x=STRIKE_PRICE, r=R, sigma=SIGMA, t=T, n=N):
    """Биномиальная модель для европейского Put опциона"""
    dt = t / n
    return (exp(-r * T) *
            np.sum([comb(n, i) * p(sigma, dt, r) ** i * (1 - p(sigma, dt, r)) ** (n - i) *
                    payoff_put(s * u(sigma, dt) ** i * d(sigma, dt) ** (n - i), x) for i in range(n)]))


# Функции для расчета парметров u, d, p
def u(sigma, dt):
    return exp(sigma * sqrt(dt))


def d(sigma, dt):
    return exp(-sigma * sqrt(dt))


def p(sigma, dt, r):
    return (exp(r * dt) - d(sigma, dt)) / (u(sigma, dt) - d(sigma, dt))


# Функция цены Американского Put опциона при помощи биномиальной модели
def binominal_american_price(s=ASSET_PRICE, x=STRIKE_PRICE, n=N, sigma=SIGMA, r=R, t=T, type=None):
    # Расчитываем коэффициенты u, d, p, dt
    dt = t / N
    u_value = u(sigma, dt)
    d_value = d(sigma, dt)
    p_value = p(sigma, dt, r)

    # Создаем дерево длинной N
    tree = {i: [] for i in range(n + 1)}
    tree[0].append(s)

    # Прямой ход заполнения дерева
    for i in range(1, n + 1):
        for j in range(len(tree[i - 1])):
            if j == 0:
                tree[i].append(tree[i - 1][j] * u_value)
                tree[i].append(tree[i - 1][j] * d_value)
            else:
                tree[i].append(tree[i - 1][j] * d_value)

    if type == 'Put':
        # Расчет Payoff для итоговых значений опциона в момент Maturity Date
        for i in range(n + 1):
            tree[N][i] = payoff_put(tree[n][i], x)

        # Обратный ход заоплнения дерева
        for i in range(n - 1, -1, -1):
            for j in range(len(tree[i])):
                tree[i][j] = max(payoff_put(tree[i][j], x),
                                 price_i(tree[i + 1][j], tree[i + 1][j + 1], p_value, r, dt))

    elif type == 'Call':
        # Расчет Payoff для итоговых значений опциона в момент Maturity Date
        for i in range(N + 1):
            tree[n][i] = payoff_call(tree[n][i], x)

        # Обратный ход заоплнения дерева
        for i in range(n - 1, -1, -1):
            for j in range(len(tree[i])):
                tree[i][j] = max(payoff_call(tree[i][j], x),
                                 price_i(tree[i + 1][j], tree[i + 1][j + 1], p_value, r, dt))

    return tree[0][0]
