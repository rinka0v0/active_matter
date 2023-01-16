# 相転移の指数を見積もる

from random import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anime
from matplotlib import rc

N = 0
IMPACT_RADIUS = 0.07
V0 = 0.05
NOISE = 0.01

# TODO: Rcをどうやって決めるか
Rc = 0.035

fig, ax = plt.subplots() # figオブジェクトの作成

# x 軸のラベルを設定する。
ax.set_xlabel("R - Rc")
# y 軸のラベルを設定する。
ax.set_ylabel("order parameter")

# 両対数グラフにする
ax.set_yscale('log')
ax.set_xscale('log')

time_limit = 300
progress = 0
RADIUSES = []
ORDER_PARAMETERS = []

def exponent_func (c, b, r):
    return (c * r)** b

N = 900
xy_lim = (1.0 * N / 300) ** 0.5
x = np.random.rand(N, 2) * 2 - 1
v = (np.random.rand(N, 2) * 2 - 1 ) * V0
dv_impact = np.empty((N,2))
dv_boundary = np.empty((N,2))

for r in range(1,10):
    IMPACT_RADIUS = r * 0.001 + Rc
    print('r:',r)
    order_parameter = 0.0


    for time in range(time_limit):
        for i in range(N):
            x_this = x[i]
            v_this = v[i]
            # それ以外の個体の位置と速度の配列
            x_that = np.delete(x, i, axis=0)
            v_that = np.delete(v, i, axis=0)
            # 粒子間の距離を求める
            distance = np.linalg.norm(x_that - x_this, axis=1)
            angle = np.arccos(np.dot(v_this, (x_that-x_this).T) / (np.linalg.norm(v_this) * np.linalg.norm((x_that-x_this), axis=1)))
            # 影響範囲内に存在する粒子（の速度）を取り出す
            impact_v = v_that[(distance < IMPACT_RADIUS)]
            # 粒子iの速度の変化量を計算
            random_array = np.array([np.random.normal(),np.random.normal()])
            dv_impact[i] = np.average(impact_v, axis=0) + random_array * NOISE if (len(impact_v) > 0) else random_array
            dv_impact[i] /= np.linalg.norm(dv_impact[i], ord=2)
            dv_impact[i] *= V0 

        v += dv_impact
        # 速度を一定にする
        for i in range(N):
            v_abs = np.linalg.norm(v[i])
            if (v_abs != V0):
                v[i] = V0 * v[i] / v_abs
            
        # 位置のアップデート
        x += v

        # 範囲から飛び出した粒子の処理
        for i in range(N):
            # 左右上下
            if abs(x[i][0]) >= xy_lim and abs(x[i][1]) >= xy_lim:
                if x[i][0] <= -xy_lim and x[i][1] <= -xy_lim:
                    x[i] += 2 * xy_lim
                elif x[i][0] >= xy_lim and x[i][1] >= xy_lim:
                    x[i] -= 2 * xy_lim
                elif x[i][0] <= -xy_lim and x[i][1] >= xy_lim:
                    x[i][0] += 2 * xy_lim
                    x[i][1] -= 2 * xy_lim
                else:
                    x[i][0] -= 2 * xy_lim
                    x[i][1] += 2 * xy_lim

            # 上下 
            if abs(x[i][1]) >= xy_lim and abs(x[i][0]) <= xy_lim:
                if x[i][1] >= xy_lim:
                    x[i][1] -= 2 * xy_lim
                else:
                    x[i][1] += 2 * xy_lim
            # 左右
            if abs(x[i][0]) >= xy_lim and abs(x[i][1]) <= xy_lim:
                if x[i][0] >= xy_lim:
                    x[i][0] -= 2 * xy_lim
                else:
                    x[i][0] += 2 * xy_lim

    # 十分に衝突させた後にorder parameterを計算する 
    if time == time_limit - 1:
        for vi in v:
            order_parameter += vi
        order_parameter = np.linalg.norm(order_parameter, ord=2) / (N * V0)
        plt.scatter(IMPACT_RADIUS - Rc, order_parameter, c='blue')
        RADIUSES.append(IMPACT_RADIUS - Rc)
        ORDER_PARAMETERS.append(order_parameter)

c, b = np.polyfit(RADIUSES, ORDER_PARAMETERS, 1)
for p in range(len(ORDER_PARAMETERS)):
    ORDER_PARAMETERS[p] = exponent_func(c, b, RADIUSES[p])

plt.plot(RADIUSES, ORDER_PARAMETERS, label="order parameter")
plt.plot([], [], label="c = {}".format(c))
plt.plot([], [], label="β = {}".format(b))

ax.legend(loc='lower right')
plt.show()
plt.savefig("exponent.png")
