import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anime
from matplotlib import rc

N = 100
IMPACT_RADIUS = 0.1

# 速度の上限/下限
MIN_VEL = 0.005
MAX_VEL = 0.03

# 境界で働く力（0にすると自由境界）
BOUNDARY_FORCE = 0.1

V0 = 0.1
NOISE = 0.05

x = np.random.rand(N, 2) * 2 - 1
v = (np.random.rand(N, 2) * 2 - 1 ) * MIN_VEL


dv_impact = np.empty((N,2))
dv_boundary = np.empty((N,2))

# グラフをプロットするための準備
xy_lim = 1.5                            # グラフのx,y軸の表示制限
time_position_x = xy_lim - xy_lim/2     # 時間表示の位置（x座標）
time_position_y = xy_lim - xy_lim/4     # 時間表示の位置（y座標）
graph_title = "Vicsek model (Standerd ver)"   # グラフのタイトル

fig, ax = plt.subplots() # figオブジェクトの作成
ims = []               # gifのための複数画像を入れる入れ物

time_limit = 300
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
        # 影響範囲内に存在する粒子（の速度）を取り出す
        impact_v = v_that[ (distance < IMPACT_RADIUS) ]
        # 粒子iの速度の変化量を計算
        dv_impact[i] = V0 * np.average(impact_v, axis=0) / abs(np.average(impact_v, axis=0)) + np.array([np.random.normal(),np.random.normal()]) * NOISE if (len(impact_v) > 0) else 0

        dist_center = np.linalg.norm(x_this) # 原点からの距離
        dv_boundary[i] = - BOUNDARY_FORCE * x_this * (dist_center - 1) / dist_center if (dist_center > 1) else 0

    # 速度のアップデートと上限/下限のチェック
    v += dv_impact + dv_boundary
    for i in range(N):
        v_abs = np.linalg.norm(v[i])
        if (v_abs < MIN_VEL):
            v[i] = MIN_VEL * v[i] / v_abs
        elif (v_abs > MAX_VEL):
            v[i] = MAX_VEL * v[i] / v_abs
    
    # 位置のアップデート
    x += v

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
anim.save('VicsekModel_Standard_2DSample.gif', writer="pillow")