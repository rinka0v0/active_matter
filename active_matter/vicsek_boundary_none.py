from random import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anime
from matplotlib import rc

N = 1000
IMPACT_RADIUS = 0.18
IMPACT_ANGLE = 10

# 速度の上限/下限
MIN_VEL = 0.005
MAX_VEL = 0.03

V0 = 0.03
NOISE = 0.01

x = np.random.rand(N, 2) * 2 - 1
v = (np.random.rand(N, 2) * 2 - 1 ) * MIN_VEL

dv_impact = np.empty((N,2))
dv_boundary = np.empty((N,2))

# グラフをプロットするための準備
xy_lim = 1.0                      # グラフのx,y軸の表示制限
# time_position_x = xy_lim - xy_lim / 2     # 時間表示の位置（x座標）
# time_position_y = xy_lim - xy_lim / 4     # 時間表示の位置（y座標）
time_position_x = xy_lim   # 時間表示の位置（x座標）
time_position_y = xy_lim + 0.8   # 時間表示の位置（y座標）
graph_title = "Vicsek model (Standerd ver)"   # グラフのタイトル

fig, ax = plt.subplots() # figオブジェクトの作成
ims = []               # gifのための複数画像を入れる入れ物

time_limit = 600
progress = 0
for time in range(time_limit):
    if time % (time_limit / 100) == 0:
        progress += 1
        print(progress,'%計算完了')
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
        impact_v = v_that[ (distance < IMPACT_RADIUS) & (angle < IMPACT_ANGLE) ]
        # 粒子iの速度の変化量を計算
        # TODO: 速度を一定にする
        random_array = np.array([np.random.normal(),np.random.normal()])
        dv_impact[i] = V0 * np.average(impact_v, axis=0) / abs(np.average(impact_v, axis=0)) + random_array * NOISE if (len(impact_v) > 0) else random_array

    # 速度のアップデートと上限/下限のチェック
    v += dv_impact

    # for i in range(N):
    #     v_abs = np.linalg.norm(v[i])
    #     if (v_abs < MIN_VEL):
    #         v[i] = MIN_VEL * v[i] / v_abs
    #     elif (v_abs > MAX_VEL):
    #         v[i] = MAX_VEL * v[i] / v_abs

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

    # 各時間での散布図をプロット
    im = ax.scatter(x[:, 0], x[:, 1], s=2, c='b')                                # x軸 実部, y軸 虚部 として固有値の散布図をプロット
    time_label = ax.text(time_position_x, time_position_y, f'time={time}')      # グラフ中に時間を表示
    ims.append([im, time_label])                                                # 生成した散布図をimsに入れる

print('gif画像を生成開始')
# 生成していた散布図(ims)をつなげてgif画像を生成
anim = anime.ArtistAnimation(fig, ims, interval=100)
rc('animation', html='jshtml')
anim

print('画像保存開始')
# gif画像の保存
anim.save('Hakumon_R=0.18_N=1000.gif', writer="pillow")