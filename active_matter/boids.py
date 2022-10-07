#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import vispy
from vispy.scene import SceneCanvas
from vispy.scene import visuals


class SwarmVisualizer(object):
    """docstring for SwarmVisualizer."""
    ARROW_SIZE = 20

    def __init__(self, width=600, height=600):
    # def __init__(self, width=1000, height=1000):
        self._canvas = SceneCanvas(size=(width, height), position=(0,0), keys='interactive', title="ALife book "+self.__class__.__name__)
        self._view = self._canvas.central_widget.add_view()
        self._view.camera = 'arcball'
        # self._view.camera = 'turntable'
        self._axis = visuals.XYZAxis(parent=self._view.scene)
        self._arrows = None
        self._markers = None
        self._canvas.show()

    def update(self, position, direction):
        assert position.ndim == 2 and position.shape[1] in (2,3)
        assert direction.ndim == 2 and direction.shape[1] in (2,3)
        assert position.shape[0] == direction.shape[0]
        if self._arrows == None:
            self._arrows = visuals.Arrow(arrow_size=self.ARROW_SIZE, arrow_type='triangle_30', parent=self._view.scene)
        # arrow_coordinate[0::2] is position of arrow and
        # arrow_coordinate[1::2] is direction of tail (length is ignored)
        arrow_coordinate = np.repeat(position, 2, axis=0)
        arrow_coordinate[::2] -=  direction
        self._arrows.set_data(arrows=arrow_coordinate.reshape((-1, 6)))
        self._canvas.update()
        vispy.app.process_events()

    def set_markers(self, position):
        assert position.ndim == 2 and position.shape[-1] in (2,3)
        if self._markers is None:
            self._markers = visuals.Markers(parent=self._view.scene)
        self._markers.set_data(position, face_color=(1,0,0), size=20)
        self._canvas.update()
        vispy.app.process_events()

    def __bool__(self):
        return not self._canvas._closed

# visualizerの初期化 (Appendix参照)
visualizer = SwarmVisualizer()

# シミュレーションパラメタ
# N = 256
N = 100
# 力の強さ
COHESION_FORCE = 0.8
SEPARATION_FORCE = 0.4
ALIGNMENT_FORCE = 0.06
# 力の働く距離
COHESION_DISTANCE = 0.5
SEPARATION_DISTANCE = 0.05
ALIGNMENT_DISTANCE = 0.1
# 力の働く角度
COHESION_ANGLE = np.pi / 2
SEPARATION_ANGLE = np.pi / 2
ALIGNMENT_ANGLE = np.pi / 3
# 速度の上限/下限
MIN_VEL = 0.005
MAX_VEL = 0.03
# 境界で働く力（0にすると自由境界）
BOUNDARY_FORCE = 0.001

# 位置と速度
x = np.random.rand(N, 3) * 2 - 1
v = (np.random.rand(N, 3) * 2 - 1 ) * MIN_VEL

# cohesion, separation, alignmentの３つの力を代入する変数
dv_coh = np.empty((N,3))
dv_sep = np.empty((N,3))
dv_ali = np.empty((N,3))
# 境界で働く力を代入する変数
dv_boundary = np.empty((N,3))

while visualizer:
    for i in range(N):
        # ここで計算する個体の位置と速度
        x_this = x[i]
        v_this = v[i]
        # それ以外の個体の位置と速度の配列
        x_that = np.delete(x, i, axis=0)
        v_that = np.delete(v, i, axis=0)
        # 個体間の距離と角度
        distance = np.linalg.norm(x_that - x_this, axis=1)
        angle = np.arccos(np.dot(v_this, (x_that-x_this).T) / (np.linalg.norm(v_this) * np.linalg.norm((x_that-x_this), axis=1)))
        # 各力が働く範囲内の個体のリスト
        coh_agents_x = x_that[ (distance < COHESION_DISTANCE) & (angle < COHESION_ANGLE) ]
        sep_agents_x = x_that[ (distance < SEPARATION_DISTANCE) & (angle < SEPARATION_ANGLE) ]
        ali_agents_v = v_that[ (distance < ALIGNMENT_DISTANCE) & (angle < ALIGNMENT_ANGLE) ]
        # 各力の計算
        dv_coh[i] = COHESION_FORCE * (np.average(coh_agents_x, axis=0) - x_this) if (len(coh_agents_x) > 0) else 0
        dv_sep[i] = SEPARATION_FORCE * np.sum(x_this - sep_agents_x, axis=0) if (len(sep_agents_x) > 0) else 0
        dv_ali[i] = ALIGNMENT_FORCE * (np.average(ali_agents_v, axis=0) - v_this) if (len(ali_agents_v) > 0) else 0
        dist_center = np.linalg.norm(x_this) # 原点からの距離
        dv_boundary[i] = - BOUNDARY_FORCE * x_this * (dist_center - 1) / dist_center if (dist_center > 1) else 0
    # 速度のアップデートと上限/下限のチェック
    v += dv_coh + dv_sep + dv_ali + dv_boundary
    for i in range(N):
        v_abs = np.linalg.norm(v[i])
        if (v_abs < MIN_VEL):
            v[i] = MIN_VEL * v[i] / v_abs
        elif (v_abs > MAX_VEL):
            v[i] = MAX_VEL * v[i] / v_abs
    # 位置のアップデート
    x += v
    visualizer.update(x, v)