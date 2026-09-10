# TEM 开发规范第一批落实记录

依据用户提供的 `TEM_Simulator_v2_Development_Spec_v1.0.md`（2026-09-10）。本批范围为 WP-00、WP-01、WP-02，以及对应 WP-06/08 的测试与记录。后续源模式、A5、完整 GPU 4D-STEM 与材料外部验证不属于本批完成项。

## 基线与风险复核（WP-00）

开始修改时 HEAD 为 `efc648ddbdcfa8d008c26048dde93ef52613fffe`，与规范基线一致，工作区干净。保留了已有 3D 建模、配件复制、材料与场模型工作。

| 风险 | 当前状态 | 证据与边界 |
| --- | --- | --- |
| F-01 | TEM 路径已解决 | `wave_imaging` 不再预先施加居中角度孔；`objective_aperture` 明确选择唯一策略，`FluxLedger` 防止同一分支重复执行物理 ID。 |
| F-02 | 已复现并修复该 TEM 边界 | 原 20% 已知透过率输出约 100%；修复后保持 20%。不据此推断原 EDS、STEM 或所有计数都错误。 |
| F-03 | 仍存在，明确标注 | TEM 的矩统计重建和 STEM 圆形 pupil 仍是 `ray_conditioned_reduced_order`，不是任意源波前传播。 |
| F-04 | 仍存在 | `EffectiveAberrationSet` 与 `field_aberrations` 有 C5，无 A5。 |
| F-05 | 仍存在 | 现有拟合包含 RMS、条件数和有限孔径；本批未完成步长/孔径/场网格/独立射线联合收敛。 |
| F-06 | 仍存在 | 已有安匝与场映射来源，不等于每个线圈都有校准电流。 |
| F-07 | 部分已有，仍需推进 | `parameter_impact.py` 已能查询几何对求解的影响；不代表所有孔槽与任意变换都进入磁场和碰撞求解。 |
| F-08 | 仍存在 | `stem_wave_imaging` 的完整 diffraction sink 仍切回 NumPy。 |
| F-09 | 当前 TEM 说明已修复 | 界面 scope 与导出读取实际执行 manifest，不再声称“尚未应用中间光阑”。其他模块说明尚未做全项目逐条审计。 |
| F-10 | 本地骨架已建立，外部证据不足 | 增加 CPU CI 与环境约束、机器可读报告；未运行远程 GitHub CI，也未完成独立材料物理验证。 |

修复前记录：[tem-p0-baseline.json](tem-p0-baseline.json)。运行的是新增验收夹具配合**未修改的生产代码**，不是事后推测旧结果：32×32、1 Å 间距，0 与 4 阶已知傅里叶分量；正则傅里叶传输 B=0.01 m；物镜孔半径 0.003 mm。

| 验收 | 理论值 | 未修改基线 |
| --- | ---: | ---: |
| AT-06，选择零级分量 | 0.2 | 0.9999999999999998（FAIL） |
| AT-02，偏移孔选择非零级分量 | 0.8 | 0（FAIL） |
| AT-04，完全关闭 | 0 | 0（PASS） |

## 光阑作用归属（WP-01）

默认 `sample.wave_objective_aperture_strategy = "physical_plane"`。在实际位置依次作相干 LCT，使用实际半径、横向偏移、安装/插入/启用状态。半径为零是有效的全阻挡结果。禁用、移出、未安装的光阑不参与传播；位于当前记录面之后的光阑也不作用。

`equivalent_pupil` 是显式选择的受限特例：光阑必须位于傅里叶共轭面，且没有在它之前需要排序执行的其他物理孔。用实际带符号的 B 矩阵与仿射偏移将傅里叶坐标映射到开口，支持偏心孔；不使用半径/轴向距离代替传输。位置项 A 所引入的误差上限必须小于映射频率像素的 `1e-8`；不满足时明确报告 unsupported，要求使用实际平面策略。记录中保留判断阈值与实际矩阵。

等效策略会排除随后对同一物理孔的平面 mask。执行账本使用 `aperture:<component key>`，每个 frozen-phonon 分支最多一次。数值带宽裁剪没有物理元件 ID。

残余像差相位保留；已有场传输中的一阶聚焦不再另算。生产 TEM 的相机入口统一保留相干平面传播，包括没有中间光阑的情况；旧独立相机 API 的粗像素兼容捷径不影响新 TEM 路径。

Checkpoint 保存每个组态在**物镜孔之前**的波与原入射范数。偏移、扩孔和策略切换可从该点重放，不从已删掉的振幅“恢复”。旧 checkpoint 缺少参考范数时要求重新计算。机械编辑的自动缓存仍保守保留整个位于样品处的装配模块几何依赖；显式重放已通过与新算结果对照。策略变化可复用源 checkpoint，D/I/P 变化沿用既有重投影缓存。

## 范数和通量（WP-02）

使用一个明确参考：`specimen_entrance_conditional_zero_loss`，即以样品入射处、条件零损分支的一电子为参考。没有将源电流冒充样品电流，也没有重复乘柱透过率或零损概率。样品吸收、非弹性射线和 EDS 光子仍是独立模型和账本。

归一化清点与处理：

| 位置 | 语义与本批处理 |
| --- | --- |
| `wave_imaging._incident_wave` | 既有 mean-intensity=1 形状；保存其**截断前**离散范数作为参考，保留兼容。 |
| TEM→相机适配 | `WaveMode.from_legacy` 将当前范数转入电子权重，单位形状与权重成对变化。密度→格点振幅因子为 `sqrt(abs(det(basis_m)))`。 |
| `camera_wave._normalise_wave` | 仅保留在显式标记的旧 `legacy_unit_shape` 独立 API；新 TEM 使用 `weighted_density_per_m`，不会在这里恢复损失。零波合法。 |
| multislice / PlaneWave / phase FFT | 保留范数变化；带宽损失单独记账。无损 LCT 与残余相位有守恒检查，不通过归一化修复。 |
| frozen phonon | 各配置先验固定为 1/N，分别传播后做强度平均。0.5×0.2 与 0.5×0.8 分别贡献 0.1 和 0.4。 |
| 傅里叶强度 | 旧 `linear_diffraction_probability` 仍是表示带内的条件分布，新增 `absolute_diffraction_probability` 保留相对参考电子的概率。 |
| 显示 | 百分位与 log 显示缩放不进入原始输出和计数。 |
| 像素沉积 | 边缘插值权重归一化属于保守像素沉积，不是对波的重新归一化。 |
| STEM | `_normalised_shifted_probe` 在样品/光阑之前定义单位入射探针；本批未改动其计数模型。 |
| EFTEM | 现有 separable 光谱乘法直接使用未显示归一化的 TEM 强度，不再作单位形状归一化；其能量与空间可分离假设仍保留。 |

`FluxLedger` 记录参考面、节点、物理 ID、组态、输入/输出权重、互斥性与损失类别。区别 `physical_stop`、`numerical_bandwidth`、`missed_detector` 和 `detector_response_crop`。在容差内的正向数值残差明确记录，不把它静默修正成物理守恒。有限周期计算域的遗漏没有被独立估计，manifest 明示这一点，不能把剩余差额都归因于吸收。

CPU complex128 范数检查 `rtol=1e-10, atol=1e-14`；含既有 complex64 GPU 波的边界使用显式 `rtol=2e-5`，不会据此宣称 GPU 物理对照已完成。解析验收概率使用 `atol=1e-10`。

`expected_electron_counts` 使用 `I_ref × exposure / e × pixel_probability`；调用者必须提供与概率同参考面的电流。当前阶段没有把所有电子和光子通道强行迁入一个统一源模式模型；完整 BeamState/PupilState 与部分相干扩展留在 WP-03。

## 界面、导出与记录（WP-08 骨架）

Illuminating Image 的 **Export raw TEM…** 导出当前实际显示的结果（包括 Advanced bank），而不是读取此刻已改变的仪器参数。NPZ 可用 `numpy.load(..., allow_pickle=False)` 打开，包含：

- PSF 前后原始概率密度及每像素到达概率；
- 截止物镜孔之前的绝对衍射概率；
- 坐标轴和单独标识的显示数组；
- JSON 执行 manifest、参考面、后柱矩阵和通量账本。

GUI 提交层将既有不可变 `CalculationManifest` 绑定到结果，不在波求解函数中重新序列化可变仪器。独立函数调用没有提交 manifest 时明确保留 null；交互重放保留源 request digest，新读出 request 由调用层提供，不伪装成原请求。

新增 TEM 专用缓存版本 `physical-aperture-pre-loss-flux-v1`，使旧 TEM 和相关 EFTEM 数值结果重新计算。没有修改共享 specimen、STEM、EDS 或 incident 求解 schema。

## 验收与执行

```powershell
.venv/Scripts/python.exe scripts/validate_tem_p0.py --extended --output outputs/tem-p0
```

脚本输出 `report.json`、`pytest.xml`、`pytest.log`；报告包括 HEAD、环境版本、命令、每项测试与 PASS/FAIL/NOT_RUN。独立材料对照和 CUDA 物理对照明确标为 NOT_RUN。

AT-01～06：`test_tem_flux_contract.py` 从 `simulate_wave_image()` 进入，使用解析波/场传输夹具，实际执行 FFT、光阑、相机与 PSF。AT-03 的新夹具检查映射变化；既有真实镜柱重投影测试另检查透镜励磁改变后的响应。

AT-07：已知模式权重缩放与电流/曝光线性计数。AT-08：真正运行两个相位光栅的 multislice，零级强度分别为 0.2/0.8。AT-09：无损范数、故意损坏的传播、实际 anti-alias 损失分类。AT-10：既有 `test_record_plane.py` 和 `test_record_plane_detector_masks.py` 检查物理串联拦截与可重叠的非阻挡观察。

新增 Windows Python 3.12 CPU CI。`requirements/validation-cpu-constraints.txt` 固定本地已测主数值/GUI依赖版本；完整安装版本另在报告中保存。它不是跨平台完整传递依赖锁。远程 CI 要在代码上传后才会执行，本批不把工作流文件的存在写成 CI 已通过。

本地最终结果见同目录 `tem-p0-current.json`。科学结果可能与旧截图明显不同：偏移孔现在可以选到正确分量，物理截断与数值带宽造成的损失会保留。显示对比度仍可自动调整，因此不能仅凭显示亮暗比较绝对通量。

2026-09-10 本地执行结果：扩展相关回归 **178 passed**（222.68 s）；补充审计后的核心复验 **49 passed**（20.55 s），覆盖最后增加的相邻通量节点连续性和元件状态记录。两组测试有重叠，不把次数相加当成独立测试数量。另已通过 Python 编译检查与 `git diff --check`。
