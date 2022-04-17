# %%
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.stats import norm
import numpy as np
import math

from optionprice import Option

# %% [markdown]
# ***Муров Олег Владимирович***
#
# ***Вариант №8***

# %% [markdown]
# ### Задание №1
#
# ***Условие:***
#
# *Пусть цена **Европейского Put** опциона с* $X=500\$$ *равна* $3\$,$ *а текущая цена акции равна* $S=493\$.$ *Найдите цену **Американского Call** опциона с тем же страйком, если* $r = 1\%$ *и* $T = 3.$ *Ответ округлите до целых.*
#
# ***Решение:***
#
# *Цены Европейских Put и Call опционов с одинаковыми параметрами связаны формулой:* $$C-P=S(t)-Xe^{-r(T-t)}$$
#
# $$C=S(t)-Xe^{-r(T-t)}+P$$

# %%


# %%
from math import exp


def ex_1():
    """Задание 1. Вычисление Call опциона"""
    put = 3
    s = 493
    x = 500
    r = 0.01
    dt = 3

    return round(s - x * exp(-r * dt) + put, 2)


# %%
ex_1()


# %% [markdown]
# *Цена **Европейского Call** опциона равна* $10,78\$.$
#
# *Однако цена **Американского Call** опциона равна цене **Европейского**, т.е. в каждый момент времени, при любой цене базового актива, досрочное исполнение **Call** опциона не является оптимальным.*
#
# ***Ответ:*** *цена **Американского Call** опциона при заданных параметрах **Европейского Put** опциона равна* $10,78\$$

# %% [markdown]
# ### Задание №2
#
# <u>***2.1*** *Реализуйте следующие функции подсчета:*</u>
#
# - *Цен **Европейских** опционов*

# %%
def d1(S, X, r, sigma, T, t):
    return (math.log(S / X) + (r + sigma ** 2 / 2) * (T - t)) / (sigma * math.sqrt(T - t))


def d2(S, X, r, sigma, T, t):
    return d1(S, X, r, sigma, T, t) - sigma * math.sqrt(T - t)


# Функции цен Европейских Call и Put опционов через уравнение Блэка-Шоулза
def EuropianCallPrice(S, X, r, sigma, T, t):
    return norm.cdf(d1(S, X, r, sigma, T, t)) * S - norm.cdf(d2(S, X, r, sigma, T, t)) * X * math.e ** (-r * (T - t))


def EuropianPutPrice(S, X, r, sigma, T, t):
    return norm.cdf(-d2(S, X, r, sigma, T, t)) * X * math.e ** (-r * (T - t)) - norm.cdf(-d1(S, X, r, sigma, T, t)) * S


# %% [markdown]
# - *Payoff-ов*

# %%
# Функция Payoff для Европейских Call и Put опционов
def payoffCall(S, X):
    return max(S - X, 0)


def payoffPut(S, X):
    return max(X - S, 0)


# %% [markdown]
# - *Цены на i-том шаге из цен i+1-го шага*
#
# <center>$V=e^{-rdt}(pV^++(1-p)V^-)$</center>

# %%
# Функция для расчета цены опциона на i-ом шаге по известным цена на i+1-ом шаге
def iPrice(VUp, VDown, p, r, dt):
    return math.e ** (-r * dt) * (p * VUp + (1 - p) * VDown)


# %% [markdown]
# - *Цен **Европейских** опционов при помощи биномиальной модели*
#
# <center>$C=e^{-rT}\cdot\sum_{i=0}^{n}{n \choose i}p^{i}(1-p)^{n-i}\cdot max(Su^{i}d^{n-i}-X,0)$</center>
#
# <center>$P=e^{-rT}\cdot\sum_{i=0}^{n}{n \choose i}p^{i}(1-p)^{n-i}\cdot max(X-Su^{i}d^{n-i},0)$</center>

# %%
# Функции для расчета парметров u, d, p
def u(sigma, dt):
    return math.e ** (sigma * math.sqrt(dt))


def d(sigma, dt):
    return math.e ** (-sigma * math.sqrt(dt))


def p(sigma, dt, r):
    return (math.e ** (r * dt) - d(sigma, dt)) / (u(sigma, dt) - d(sigma, dt))


# %%
# Функции цен Европейских Call и Put опционов при помощи биномиальной модели
def EuropianCallPriceBin(S, X, r, sigma, T, N):
    dt = T / N
    return (math.e ** (-r * T) *
            np.sum([comb(N, i) * p(sigma, dt, r) ** i * (1 - p(sigma, dt, r)) ** (N - i) *
                    payoffCall(S * u(sigma, dt) ** i * d(sigma, dt) ** (N - i), X) for i in range(N)]))


def EuropianPutPriceBin(S, X, r, sigma, T, N):
    dt = T / N
    return (math.e ** (-r * T) *
            np.sum([comb(N, i) * p(sigma, dt, r) ** i * (1 - p(sigma, dt, r)) ** (N - i) *
                    payoffPut(S * u(sigma, dt) ** i * d(sigma, dt) ** (N - i), X) for i in range(N)]))


# %% [markdown]
# - *Цены **Американского Put** опциона при помощи биномиальной модели*
#
# 1) *Рассчитываем δt, u, d, p*
#
# 2) *Рассчитываем цены опциона, соответствующие ценам базового актива на последнем уровне дерева (т.е. просто подставим* $S_{j,i}=Su^{i}d^{j-i}$ *в функцию выплат, получая* $C_{n,i}$ *или* $P_{n,i}$)
#
# 3) *Из цен опционов на шаге j+1 получаем состояние на шаге j:*
# $$C_{i,j}=max[Su^{i}d^{j-i}-X, e^{-rδt}(pC_{j+1,i+1}+(1-p)C_{j+1,i})]$$
# $$P_{i,j}=max[X-Su^{i}d^{j-i}, e^{-rδt}(pP_{j+1,i+1}+(1-p)P_{j+1,i})]$$

# %%
# Функция цены Американского Put опциона при помощи биномиальной модели
def AmericanPriceBin(S, X, N, sigma, r, T, Type=None):
    # Расчитываем коэффициенты u, d, p, dt
    dt = T / N
    U = u(sigma, dt)
    D = d(sigma, dt)
    P = p(sigma, dt, r)

    # Создаем дерево длинной N
    OptionTree = {i: [] for i in range(N + 1)}
    OptionTree[0].append(S)

    # Прямой ход заполнения дерева
    for i in range(1, N + 1):
        for j in range(len(OptionTree[i - 1])):
            if j == 0:
                OptionTree[i].append(OptionTree[i - 1][j] * U)
                OptionTree[i].append(OptionTree[i - 1][j] * D)
            else:
                OptionTree[i].append(OptionTree[i - 1][j] * D)

    if Type == 'Put':
        # Расчет Payoff для итоговых значений опциона в момент Maturity Date
        for i in range(N + 1):
            OptionTree[N][i] = payoffPut(OptionTree[N][i], X)

        # Обратный ход заоплнения дерева
        for i in range(N - 1, -1, -1):
            for j in range(len(OptionTree[i])):
                OptionTree[i][j] = max(payoffPut(OptionTree[i][j], X),
                                       iPrice(OptionTree[i + 1][j], OptionTree[i + 1][j + 1], P, r, dt))

    elif Type == 'Call':
        # Расчет Payoff для итоговых значений опциона в момент Maturity Date
        for i in range(N + 1):
            OptionTree[N][i] = payoffCall(OptionTree[N][i], X)

        # Обратный ход заоплнения дерева
        for i in range(N - 1, -1, -1):
            for j in range(len(OptionTree[i])):
                OptionTree[i][j] = max(payoffCall(OptionTree[i][j], X),
                                       iPrice(OptionTree[i + 1][j], OptionTree[i + 1][j + 1], P, r, dt))

    return OptionTree[0][0]


# %% [markdown]
# <u>***2.2*** *Постройте графики зависимости цен в биномиальной модели от числа шагов* $N$</u>
#
# $S=70\$, X=70\$, T=2, r=5\%, σ=40\%, N=[20:40].$
#
# *Сравните полученные результаты:*
#
#    *- Для **Европейских** опционов - с точным решением (**Black-Sholes**)*
#
#    *- Для **Американского Put** - со сторонней реализацией биномиальной модели с количеством шагов $N=2000$.*

# %%
# Зададим исходные параметры опциона
assetPrice = 70
strikePrice = 70
sigma = 0.4
rate = 0.05
Time = 2

N = np.arange(20, 401)

# %%
# Построим графики зависимости цен в биномиальной модели от числа шагов N
BinPutAmerican, BinCallAmerican = [], []
BinPutEuropian, BinCallEuropian = [], []
BlackSholesPut, BlackSholesCall = [], []

for i in N:
    # Американские опционы - биномиальная модель
    BinPutAmerican.append(AmericanPriceBin(assetPrice, strikePrice, i, sigma, rate, Time, Type='Put'))
    BinCallAmerican.append(AmericanPriceBin(assetPrice, strikePrice, i, sigma, rate, Time, Type='Call'))

    # Европейские опционы - биномиальная модель
    BinPutEuropian.append(EuropianPutPriceBin(assetPrice, strikePrice, rate, sigma, Time, i))
    BinCallEuropian.append(EuropianCallPriceBin(assetPrice, strikePrice, rate, sigma, Time, i))

    # Точная цена Black-Sholes
    BlackSholesPut.append(euro_put_price(assetPrice, strikePrice, rate, sigma, Time, 0))
    BlackSholesCall.append(EuropianCallPrice(assetPrice, strikePrice, rate, sigma, Time, 0))

# %%
# Расчет цены Американского опциона при помощи библиотеки Option-Price при N=2000
OptionPrice_results = []

# Так как по условию задачи T = 2, то установим в Option t = 730, что равняется количествую дней за 2 года
for i in N:
    OptionPrice_results.append(
        Option(european=False, kind='put', s0=70, k=70, r=0.05, sigma=0.4, dv=0, t=730).getPrice(method='BT',
                                                                                                 iteration=2000))

# %%
plt.figure(figsize=(20, 10))

plt.subplot(2, 2, 1)
plt.scatter(N, BinPutAmerican, c='blue', alpha=0.4, s=30, label='Американский Put опцион')
plt.plot(N, OptionPrice_results, c='black', ls='--', alpha=0.4,
         label='Цена Американского Put опциона (пакет Option-Price, N=2000)')
plt.title('Цена Американского Put опциона', fontweight="bold")
plt.legend()

plt.subplot(2, 2, 2)
plt.scatter(N, BinCallAmerican, c='blue', alpha=0.4, s=30, label='Американский Call опцион')
plt.title('Цена Американского Call опциона', fontweight="bold")
plt.legend()

plt.subplot(2, 2, 3)
plt.scatter(N, BinPutEuropian, c='red', alpha=0.4, s=30, label='Европейский Put опцион')
plt.plot(N, BlackSholesPut, c='black', ls='--', alpha=0.4, label='Точное решение (Black-Sholes)')
plt.title('Цена Европейского Put опциона', fontweight="bold")
plt.legend()

plt.subplot(2, 2, 4)
plt.scatter(N, BinCallEuropian, c='red', alpha=0.4, s=30, label='Европейский Call опцион')
plt.plot(N, BlackSholesCall, c='black', ls='--', alpha=0.4, label='Точное решение (Black-Sholes)')
plt.title('Цена Европейского Call опциона', fontweight="bold")
plt.legend()

# %% [markdown]
# *Из получившихся результатов видно, что цена **Американского** и **Европейского Call** опционов совпадает. Цена **Американского Put** опциона во всех точках выше, чем цена **Европейского Put** опциона. Кроме того, можно заметить, что при увеличении количества шагов цена **Европейских** опционов стремится к своему точному значению, рассчитанному по формуле **Блэка-Шоулза**.*

# %% [markdown]
# <u>***2.3*** *Поменяйте страйк на* $X=63\$$ *и постройте график цен для **Европейского Call** опциона. Как изменилось поведение? Каковы могут быть причины таких изменений?*</u>

# %%
BinCallEuropian2 = []
BlackSholesCall2 = []

for i in N:
    BinCallEuropian2.append(EuropianCallPriceBin(assetPrice, 63, rate, sigma, Time, i))
    BlackSholesCall2.append(EuropianCallPrice(assetPrice, 63, rate, sigma, Time, 0))

plt.figure(figsize=(15, 8))
plt.scatter(N, BinCallEuropian2, c='red', alpha=0.4, s=30, label='Европейский Call опцион')
plt.plot(N, BlackSholesCall2, c='black', ls='--', alpha=0.4, label='Точное решение (Black-Sholes)')
plt.title('Цена Европейского Call опциона (X=63$)', fontweight="bold")
plt.legend()

# %%
