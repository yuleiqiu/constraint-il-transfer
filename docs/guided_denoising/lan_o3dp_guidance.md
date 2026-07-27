# LAN-O3DP Guidance：按官方代码整理

这份文档只用于理清 LAN-O3DP 的实际 guidance 实现，以及它如何写进我们的
cost-based formulation。它不是最终论文表述。

## 1. 障碍物距离

对轨迹中第 $h$ 个 EEF 点 $p_h$ 和第 $j$ 个障碍物中心 $c_j$，LAN-O3DP
只考虑 XY 平面距离：

$$
d_{h,j}
=
\left\|p_h^{xy}-c_j^{xy}\right\|_2.
$$

其中 $r_j$ 是对应障碍物的安全半径。

## 2. LAN-O3DP 实际使用的梯度场

当轨迹点位于安全半径内，即 $d_{h,j}\le r_j$ 时，代码计算距离关于轨迹点的梯度：

$$
g_{h,j}
=
\nabla_{p_h^{xy}} d_{h,j}
=
\frac{p_h^{xy}-c_j^{xy}}
{\left\|p_h^{xy}-c_j^{xy}\right\|_2}.
$$

这个向量从障碍物中心指向轨迹点，因此是向外的排斥方向。在安全半径外，梯度为零：

$$
g_{h,j}=0,
\qquad d_{h,j}>r_j.
$$

多个障碍物的梯度直接相加：

$$
g_h=\sum_j g_{h,j}.
$$

代码只保留 $g_h$ 的 XY 分量；Z、旋转和 gripper 对应分量均为零。

## 3. 等价的 deployment cost

LAN-O3DP 的代码直接构造向外的梯度场，没有显式定义 cost。为了和我们的
formulation 对齐，可以将它写成未平方的 penetration cost：

$$
C_{\mathrm{LAN}}(P)
=
\sum_{h,j}
\max\left(
0,
r_j-\left\|p_h^{xy}-c_j^{xy}\right\|_2
\right).
$$

当 $d_{h,j}<r_j$ 时：

$$
-\nabla_{p_h^{xy}} C_{\mathrm{LAN}}
=
\frac{p_h^{xy}-c_j^{xy}}
{\left\|p_h^{xy}-c_j^{xy}\right\|_2}.
$$

所以，对这个 cost 做梯度下降，与 LAN-O3DP 代码中加入向外距离梯度的方向一致。

## 4. Reverse-sample update

LAN-O3DP 先执行标准 DDIM reverse step：

$$
\widetilde A_{t-1}
=
\operatorname{DDIMStep}(A_t,\epsilon_\theta(A_t,t)).
$$

然后只修改 sample 的 XY 分量：

$$
A_{t-1}^{xy}
=
\widetilde A_{t-1}^{xy}
+
\frac{0.03}{\sqrt{\bar\alpha_{t-1}}}
g_t^{xy}.
$$

用 deployment cost 表达，同一个更新方向可以写成：

$$
A_{t-1}^{xy}
=
\widetilde A_{t-1}^{xy}
-
\frac{0.03}{\sqrt{\bar\alpha_{t-1}}}
\nabla_{A^{xy}}C_{\mathrm{LAN}}.
$$

这里的 $0.03$ 是官方实现中的硬编码 guidance scale。

## 5. 梯度在哪里计算

在第 $t$ 个 denoising step 中，scheduler 同时给出 predicted clean action：

$$
\widehat A_0^{(t)}.
$$

LAN-O3DP 将它反归一化到物理 action 空间：

$$
\widehat A_{0,\mathrm{phys}}^{(t)}
=
\operatorname{Unnormalize}\left(\widehat A_0^{(t)}\right),
$$

然后在这个物理空间预测上计算障碍物梯度：

$$
g_t
=
-\nabla C_{\mathrm{LAN}}
\left(\widehat A_{0,\mathrm{phys}}^{(t)}\right).
$$

不过，官方循环是在完成当前 scheduler step 后才计算 $g_t$，所以这个梯度实际在
下一次 reverse step 中使用，存在一个 denoising step 的延迟。

## 6. 对我们第一版实现的含义

如果第一版采用“LAN-O3DP 式 direct reverse-sample update”，建议保留以下核心设计：

1. 在 predicted clean delta-EEF action 上重建 EEF 轨迹；
2. 使用 XY 未平方 penetration cost；
3. 在安全半径内产生近似单位大小的向外梯度；
4. 直接修改 DDIM reverse sample，而不是重新训练 policy；
5. 第一版只引导实际执行的 future action slice。

不应机械照抄的实现细节：

- 官方代码对梯度调用普通 action `normalize()`；梯度是向量，原则上只应进行尺度变换，
  不应加入 normalizer offset；
- 官方真实机器人代码将离当前 EEF 最近的障碍物点当成中心，并硬编码约 $0.12$ m
  的安全半径；这不一定适合我们的 robosuite 几何；
- 官方实现的一步梯度延迟更像代码结构的结果，而不是 formulation 的必要条件。

## 7. 官方实现位置

- Policy 与 `gradient_field`：<https://github.com/qqianfeng/lan-o3dp/blob/main/diffusion_policy/policy/diffusion_unet_pcd_policy.py>
- 定制 DDIM update：<https://github.com/qqianfeng/lan-o3dp/blob/main/scheduling_ddim.py>
- 真实障碍物预处理：<https://github.com/qqianfeng/lan-o3dp/blob/main/diffusion_policy/real_world/real_data_conversion.py>
