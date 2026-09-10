# WP-03：显式 pupil 与独立源模式

依据用户提供的开发规范 v1.0，承接 `TEM_P0_IMPLEMENTATION.md`。本批是样品入射面上的部分相干接口，不宣称已经完成电子枪至样品的任意相干场重建。

## 使用入口

`Sample → Wave imaging settings → Configure illumination pupil / source modes…`

默认继续使用 `ray_conditioned_reduced_order`。新建或载入旧项目不会自动更换物理模型。显式模型是 `specimen_entrance_pupil_modes`，配置保存在 `sample.wave_illumination`，同时支持项目状态与 operating profile TOML。

对话框可生成椭圆或矩形 pupil，设置两个半轴、横向角偏移与旋转，以及独立的位置、方向、能量 Gaussian 求积。下面的 JSON 可直接编辑离散模式和权重；也支持任意仿射基底、二维采样振幅/强度和显式相位。生成按钮会用上方参数替换 JSON，确认保存前会校验模式及其总数。

配置中的 `positions_nm` 每行是 `[x, y, weight]`，`angles_mrad` 是 `[tx, ty, weight]`，`energies_ev` 是 `[delta_energy, weight]`。每个维度的权重各自和为 1；三个维度的笛卡尔积上限为 256 个源模式。frozen phonon 独立组合，原有配置数与种子继续生效。

位置指样品入射平面上的实验室坐标，STEM 中另叠加原来的 raster 坐标。能量值是相对当前名义束流能量的偏差，不是电子枪高压的新设定。Gaussian 参数采用 RMS / 标准差，不是 FWHM；非零展宽至少使用两个求积节点。

## 数值与物理约定

- `PupilState` 明确形状、输入为振幅还是强度、坐标基底、偏移、参考面、来源与实际采样带宽。强度输入转为平方根振幅，并使用单独声明的相位；不会从强度猜测相位。
- pupil 是条件入射源的角分布形状，不是再执行一次硬件光阑。上游透过率不能由它推断。振幅只在定义入射电子时归一化；物理光阑和数值带宽损失随后保留。
- 角坐标采用近轴笛卡尔方向角（mrad），不是厂家孔径标定。支持范围限在 200 mrad 内。解析 pupil 在频率格点上直接计算；采样 pupil 使用复振幅双线性插值和零外延。
- 频域容纳不下完整的显式 pupil 时拒绝计算，不先截掉一部分再恢复单位通量。少于四个有效频率格点也拒绝。TEM 位置移出计算 FOV 时拒绝周期回绕。
- `alpha_edge_rad`、`alpha_95_current_rad` 与 `alpha_99_current_rad` 分开。旧射线模型不知道真实孔径边缘，`alpha_edge_rad` 保持 null；兼容字段 `convergence_edge_rad` 仍是有限射线样本最大角，并明确标注。新模型报告声明的支持边界和实际离散 pupil 的电流分位角；混合模式不伪造一个全局分位角，各模式另有记录。
- 各源模式分别传播、分别通过样品和记录面，然后作固定先验的强度加权。TEM checkpoint 保留源模式 × phonon 的独立波、权重和能量，改变后柱或光阑后仍可正确重投影。
- 每个能量节点更新电子波长、相互作用常数、传播算子和后柱传输。镜片/线圈控制与几何保持原工作点；不会自动重新聚焦。修正了原场像差能量差分给只读 `beam_voltage_kv` 赋值的问题。
- STEM 采用所有源位置的共同材料 ROI，避免随模式移动晶体。衍射支撑估计考虑较窄的解析 pupil 轴及集合中的最长波长。
- `BeamState` 表达同一电子参考下的 `WaveMode` 混合，并复用带范数检查的 `PlaneWave` / LCT。只有相同的物理格点才可直接加强度，不能在不同仿射格点上静默相加。

独立强度积分的约定参考 [abTEM partial coherence](https://abtem.readthedocs.io/en/latest/user_guide/tutorials/partial_coherence.html)。本实现的能量节点直接更新传播参数；并未把教程中的离焦分布近似当作完整能量传播的替代。

## 已知边界

旧 TEM 保留束心、二阶矩和指定像差的降阶模型；旧 STEM 保留以 95% 电流分位角构造圆盘的近似。它们不恢复任意非规则振幅、未建模高阶相位或有限发射度混合态。新模型也只从明确声明的样品入射模式出发，不推断上游色差导致的位置/方向/能量相关性。

本批显式模式支持 TEM 和弹性 STEM 检测器图像。与射线 Rutherford 尾部的耦合、混合模式体吸收、以及需要跨模式角度标定的多模式/非名义能量 4D-STEM capture 会明确拒绝。单模式名义能量的既有 4D-STEM 通道继续可用。GPU 完整衍射数据常驻仍属于后续工作包。

强度加权后的源求积差异不是 iid 随机误差，不能拿来当 frozen-phonon 标准误。多源集合的联合随机不确定度没有在本批估计；界面和记录显示 source/energy convergence 未评估，直到用户针对目标条件执行加密验证。有限周期计算域、任意采样 pupil 的远场尾部、材料真实性及实验标定仍需分别检查。没有把一组解析夹具的通过泛化成所有源条件的收敛。

## 验证

```powershell
.venv/Scripts/python.exe scripts/validate_illumination.py --extended --output outputs/illumination
```

生成 `pytest.log`、`pytest.xml` 和 `report.json`。报告保存代码 SHA256、环境版本、测试列表、命令及位置/能量求积的实际比较值；使用三个逐渐加密的阶数，要求最后两次比较均满足 `absolute_floor + relative_target × reference_scale`。

| 规范条目 | 本批验收 |
| --- | --- |
| AT-11 | 均匀圆盘的 95% / 99% 电流分位角与解析边缘关系 |
| AT-12 | 偏心、旋转矩形进入生产 TEM 衍射；二维强度/振幅语义与采样边界 |
| AT-13 | 两模式生产 TEM 及 STEM 与分开计算的强度加权一致；BeamState 与相干干涉结果不同；TEM checkpoint 重放 |
| AT-14 | 制造的有限材料边缘及相位光栅上，独立细化位置和能量求积，比较 BF 概率及材料重叠 |
| AT-15 | 同平面解析高斯经实际 LCT 后的束心、宽度、总相位曲率与权重 |

另外验证状态和 profile 保存、波缓存失效而 incident 缓存保留、GUI 配置校验、每能量波长/散射参数更新与硬件不变。已有 TEM 通量、STEM 重叠、串联探测器、4D-STEM 和 profile 回归另列在报告中。

远程 CI 加入本批核心验收，但本地创建工作流不代表远程 CI 已执行。独立软件/实验材料对照和新模式 GPU 物理对照在报告中保持 `NOT_RUN`。

2026-09-10 本地结果：扩展回归 **237 passed**（219.68 s），见 [wp03-extended.json](wp03-extended.json)。随后修正重投影记录的作用范围、能量参考只应用一次及界面参数回显，再作核心和补充审计 **48 passed**，见 [wp03-current.json](wp03-current.json)。两次有重叠，不合计为独立测试数。后一次报告验证测试期间源文件没有变化。

位置 RMS 0.05 nm 的 3→5→7 阶比较，最大概率变化分别约 `2.17e-6` 与 `4.95e-10`；能量 RMS 1000 eV 的同阶比较达到浮点舍入量级。该能量夹具在这一区间平滑，不能据此认定能量敏感边界或其他工作点也收敛。另以 10% 能量偏移验证实际透镜一阶矩阵发生变化且励磁不变。

补充检查包括多模式共同 STEM ROI、拒绝不支持的 capture 时尚未写入数据，以及原始光学能量与私有传播能量分离。已通过 Python 编译与 `git diff --check`；新对话框已使用应用样式离屏渲染并检查控件与文字没有截断。
