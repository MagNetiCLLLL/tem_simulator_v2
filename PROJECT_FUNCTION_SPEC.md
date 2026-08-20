# TEM Simulator v2 — 当前功能与需求活规范

> 文档用途：本文件是后续大规模功能修改的唯一“活规范”入口。
>
> 当前状态：与工作区代码同步（2026-08-13）。
>
> 程序入口：`main.py`；项目版本：`0.1.0`。

## 1. 文档管理规则

### 1.1 本文件的职责

本文件同时承担以下职责：

1. 记录用户已经提出的需求，不因后续改写、重构、替代或停用而丢失。
2. 描述项目当前已经存在的用户功能、物理模型、输入、输出和限制。
3. 建立功能到代码、配置和测试的映射，供后续修改时核对影响范围。
4. 作为后续变更的比较基线：先读本文件并识别新增或改写的需求，再修改代码，最后更新本文件的实现状态和修订记录。

`README.md`仍可作为快速介绍，`HANDOFF.md`仍可作为开发交接记录，`CHANGELOG.md`仍可作为版本变化摘要；若三者与本文件的明确需求冲突，应先确认用户最新修改，再以本文件为需求基线。

### 1.2 不允许删除需求

以下规则是永久规则：

- 已分配编号的用户需求不得从本文件删除。
- 允许改善排版、语法、术语和章节位置，但必须保留原始意图。
- 允许把一条需求拆分成多条；拆分后的条目必须反向引用原编号。
- 允许用新需求替代旧需求；旧需求必须保留并标记为“已替代”，同时指出替代它的新编号。
- 允许取消尚未实现的需求；原条目必须保留并标记为“用户取消”，不得物理删除。
- 如果用户编辑本文件时意外移除了已有编号，后续整理时应恢复该编号并记录冲突，不得默认接受删除。
- 功能从代码中移除时，本文件中的记录仍须保留，状态改为“已停用”或“已移除”，并写明原因和替代方案。
- 代码实现细节可以重构，但不得借重构之名改变需求语义。

### 1.3 状态定义

- **已实现**：当前工作区已经存在该功能，并有代码或测试证据。
- **部分实现**：核心路径存在，但仍有明确缺口或只使用近似模型。
- **待实现**：需求已记录，当前没有满足它的实现。
- **约束**：后续所有改动都必须持续遵守。
- **已替代**：需求内容仍保留，但执行以所引用的新需求为准。
- **已停用**：历史功能或需求不再启用，但记录不得删除。

### 1.4 后续变更流程

后续用户直接修改本文件后，实施流程固定为：

1. 读取完整文件，不只读取“新需求”一节。
2. 使用稳定编号、文档修订记录和可用的 Git 差异识别新增、改写和冲突。
3. 保留所有历史需求；为没有编号的新需求分配新编号。
4. 给出受影响的 GUI、状态模型、配置、物理计算、输出和测试范围。
5. 修改代码和必要配置。
6. 运行与风险相称的测试；物理拓扑或公共状态变更需运行完整测试。
7. 更新对应条目的状态、实现位置、限制和验收结果。
8. 在文末追加修订记录，不覆盖历史记录。

## 2. 用户需求永久台账

本节只允许追加、改写排版或改变状态，不允许删除已有编号。

| 编号 | 保留后的需求表述 | 当前状态 | 当前落实位置 |
|---|---|---|---|
| UR-001 | 明确 TEM Wave Image 的用途，并使其作为真实样品高精度波成像结果，而不是装饰性图片。 | 已实现 | `physics/wave_imaging.py`、GUI `TEM Wave Image` 页 |
| UR-002 | 明确 Transverse X-Y 显示当前轴向平面的电子束横截面，并解释其颜色含义。 | 已实现 | `diagnostic_tabs.TransverseBeamView`；颜色表示初始 X-Y 象限 |
| UR-003 | Real sample 模式不得生成人为定义的衍射束；只有 Virtual sample 可以配置人为衍射、散射和吸收通道。 | 已实现，约束 | `physics/simulation.py`、`specimen/virtual.py` |
| UR-004 | Ray Diagram 支持两种互补颜色语义：不同 convergence semi-angle 用同一色相的深浅表示；不同 interaction 类型用不同色相表示。 | 已实现 | `physics/simulation.py`、`gui/visualization.py` |
| UR-005 | 用户选择任意轴向平面后，显示该平面各种 interaction 电子的比例，并依据物理概率而不是任意显示权重计算。 | 已实现 | `physics/interaction_budget.py`、Ray Diagram 选定平面表格 |
| UR-006 | Real sample 加入真实非弹性输运，包括 plasmon、ionisation、其他非弹性、复数碰撞及有效 absorption/removal。 | 已实现为概率守恒的紧凑输运模型 | `specimen/inelastic.py`、材料 TOML、Energy Filter、STEM |
| UR-007 | 修改后必须检查项目仍可启动，并验证主要功能链。 | 已实现，约束 | 离屏 GUI 冒烟检查、编译、定向与全量测试 |
| UR-008 | 在项目仍能正常启动时，不主动处理假设性的兼容性问题；只有出现实际兼容性错误时才针对错误处理。 | 约束 | 后续开发策略；不因警告或推测改动兼容层 |
| UR-009 | 把当前所有功能详细整理到一个 Markdown 文件；以后用户可大量修改该文件，实施方读取、比较、修改项目并回写文件。 | 已实现 | 本文件 |
| UR-010 | 用户需求可以重新排版和重写，但不允许删除。 | 约束 | 本文件第 1.2 节及本台账 |

## 3. 系统范围与总体结构

### 3.1 启动链

```text
main.py
  -> temsim.app.run()
     -> QApplication
     -> MainWindow
        -> TOML 仪器目录与默认装配
        -> 运行状态 State
        -> Preview / High accuracy 计算控制器
        -> CalculationResult
        -> 九个中央可视化页面
```

- `main.py`保持轻量启动入口，不承载仪器几何或物理算法。
- `temsim.app`创建或复用进程级 `QApplication`，应用 Fusion 样式并显示主窗口。
- `MainWindow`负责装配选择、状态、菜单、工具栏、后台计算、直接对准和结果分发。
- `simulation_pipeline.calculate()`统一执行状态规范化、物理布局、射线传播、可选 TEM 波成像、Energy Filter、scan/descan、STEM 帧、交叉点和 aperture stop 记录。

### 3.2 权威来源

| 信息类型 | 权威来源 | 规则 |
|---|---|---|
| 用户需求和功能语义 | 本文件 | 不得删除既有需求 |
| 静态仪器结构、机械尺寸、部件隶属、光学参考面 | `configs/instruments/*.toml` | Python 不保存第二份结构权威 |
| 装配选择 | `configs/instruments/catalog.toml` | 每次选择一个 gun 和 column；Energy Filter recording system 固定安装 |
| 工作模式和 Direct Alignment 目标 | `configs/operating_modes/catalog.toml` | 目标、范围、耦合设备、容差和来源均由 TOML 定义 |
| 样品预设及材料锚点 | `configs/specimens/*.toml` | 自定义 CIF 不得静默借用另一材料的数据 |
| 运行时可编辑值 | `State`及其组件 | 普通重算应保留用户运行值 |
| 保存的操作配置 | Operating profile TOML | 只保存允许的运行参数，不复制 TOML 静态结构 |
| 算法、验证和绘图行为 | `src/temsim` | 必须服从上述权威和本文件需求 |

### 3.3 当前装配目录

代码审计值：**10 个模块 TOML、466 条变体级部件定义、192 个逻辑部件键、15 种可选无冲突装配组合**。

Gun 选择：

- `FEG`：冷场发射枪。
- `FEG + Mono`：冷场发射枪加 Wien monochromator。
- `Thermionic`：热发射枪。

Column 选择：

- `C2`
- `C3`
- `C3 + Probe Corrector`
- `C3 + Image Corrector`
- `C3 + Probe Corrector + Image Corrector`

Recording system 固定为 `Energy Filter`，不在 Instrument Setup 中显示安装选择；
旧 `No Energy Filter` 模块只保留为历史几何资料，旧操作配置加载时自动迁移到
`Energy Filter`。默认选择为 `FEG + C3 + Probe Corrector + Energy Filter`。

### 3.4 运行模式

- Illumination：`Microprobe (TEM)`或`Nanoprobe (STEM)`。
- Projector：`Image`或`Diffraction`。
- Specimen：`Real sample (atomic)`或`Virtual sample (virtual)`。
- Sample holder：`inserted`或`retracted`。
- 计算质量：交互 `Preview`或一次性 `High accuracy`。

## 4. 安装、启动和主窗口

### 4.1 环境

- 支持的项目解释器范围为 Python `>=3.12,<3.13`。
- `setup_env.py`创建或复用 `.venv`，安装项目、开发依赖和可编辑包。
- 主要依赖包括 PySide6、PyQtGraph、NumPy、SciPy、Numba、Matplotlib、ASE、abTEM、Pillow、ImageIO、tifffile 和 tomli-w。
- CuPy CUDA 是可选 `gpu` extra；缺失时 CPU 路径仍可使用。
- 按 UR-008，项目可以启动且没有实际兼容性错误时，不进行预防性兼容改造。

### 4.2 启动与退出

- 启动命令：`.venv\Scripts\python.exe main.py`。
- 主窗口标题为 `TEM Simulator v2`，默认尺寸 `1500 x 920`。
- 窗口几何和 dock 状态通过 `QSettings`保存和恢复。
- 退出时清理计算线程池，并等待最多 3 秒完成后台任务。

### 4.3 菜单

File：

- `Open operating profile...`（Ctrl+O）
- `Save operating profile...`（Ctrl+S）
- `Reload and validate TOML catalog`（F5）
- `Exit`（Ctrl+Q）

View：

- 显示/隐藏 instrument dock。
- 显示/隐藏 calculation log dock。
- 恢复默认工作区布局。

### 4.4 计算工具栏

- `Recalculate preview`：49 条射线、2.5 mm 默认步长、后台快速预览。
- High-accuracy ray count：范围 1,000–1,000,000，默认 15,000。
- High-accuracy step：范围 0.01–1.0 mm，默认 0.1 mm。
- Compute backend：`Auto (GPU / CPU)`、`CPU`、`Numba CPU`、`CUDA GPU`。
- `Run high-accuracy once`：按当前设置运行一次完整计算。
- 提交前估算内存；高精度计算采用 24 GiB 应用预算，目标机器配置为 32 GiB。
- 内存估计包括射线工作数组、历史数组、真实非弹性分支、TEM 波网格、atomistic slices 和 frozen-phonon configurations。

### 4.5 后台执行和事务行为

- Preview、High accuracy 和 Direct Alignment 都不得阻塞 GUI 主线程。
- 新状态会使旧计算结果失效；陈旧结果不能覆盖当前状态。
- 计算中显示进度状态，完成后报告耗时、模式、射线后端和波后端。
- 失败通过状态栏、日志和错误对话框报告。

## 5. 主界面页面

中央工作区包含九个页面：

1. `Ray Diagram`
2. `Sample`
3. `Physical Layout`
4. `Magnetic Field`
5. `Optical Transfer`
6. `Energy Filter`
7. `Transverse X-Y`
8. `STEM`
9. `TEM Wave Image`

左侧 instrument dock 包含：

- 装配模块选择。
- Probe/illumination 和 projector operating preset。
- `Optical`组件树。
- `Mechanical`组件树。
- `Direct Alignment`页面。
- 选中组件的 `Operating`和`TOML`参数页。

底部 log dock记录启动、目录审计、装配、后端、计算、Direct Alignment、Energy Filter 匹配和错误。

## 6. 仪器配置、编辑和状态持久化

### 6.1 TOML 目录验证

- 启动时验证模块格式、模块类型、唯一文件、唯一模块键、选择签名和目录完整性。
- 每个模块内验证部件键、顺序、结构字段、机械嵌套、光学参考和磁场极性来源。
- 每个可选装配都验证运行键冲突和布局有效性。
- 每个活动部件具有稳定 definition ID：`<module TOML>::parts[<canonical key>]`。

### 6.2 Optical 与 Mechanical 树

- Optical 树按 lenses、apertures、stigmators、deflectors、corrector elements、recording planes、gun 和 Energy Filter 等功能分类。
- Mechanical 树显示模块和所有机械部件，包括 housing、yoke、coil、pole、liner、holder、detector housing 等。
- 组件可从树中选择，也可从 Physical Layout、Magnetic Field 或 Energy Filter 图中点击反向导航。
- Sample 不在左侧重复出现；点击 sample 会打开中央 Sample 页面。

### 6.3 参数编辑

- `Operating`页只编辑运行参数，例如 excitation、enable、offset、scan、slit 和用户模型开关。
- `TOML`页编辑静态结构或来源字段。
- TOML 保存采用先写入、再重建并验证装配的事务流程；验证失败时恢复原文件。
- 普通运行参数变化会安排 debounce Preview。
- Lens excitation 限制为 0–100%；需要更强场时应修改经过依据支持的 100% field calibration，而不是输入超过 100%。

### 6.4 Operating profile

- Profile 当前格式版本为 2。
- 保存内容包括装配选择、允许的运行设备参数、样品 quaternion、zone/in-plane axis、Virtual interaction 表、Virtual region 表和 per-element frozen-phonon RMS。
- 静态结构、位置和 TOML-owned 字段不写入 profile。
- 写入采用临时文件、flush、fsync 和原子替换。
- 加载先在候选状态应用，再重新施加 TOML 结构，避免 profile 夺取静态几何权威。
- 未识别参数会报告为 skipped，不静默改写结构。

## 7. Electron gun

### 7.1 公共输出契约

所有 gun 最终提供统一的电子相空间：

- `x_m`, `y_m`
- `tx_rad`, `ty_rad`
- `energy_offset_ev`
- 每条射线的权重和稳定 `ray_id`
- alive/blocked 状态
- 共享 Z 网格路径、equal-time history 和关键平面 arrival time

### 7.2 Cold FEG

- 具有 cold field emitter、extractor、electrostatic gun lens、accelerator stages、DPA/gun aperture、deflector、stigmator 和 C1 aperture。
- 采用确定性的低差异采样生成位置、角度和能量分布。
- 冷 FEG 能量尾保持正动能，同时保持请求的均值和 FWHM。
- 有限电场和磁场中的轨迹使用相对论 Boris 积分。
- 物理 bore 和 aperture 可截断电子，并记录第一拦截原因。

### 7.3 FEG + monochromator

- 在 FEG 路径中加入有限 crossed-field Wien element 和能量 slit。
- 支持 electric/magnetic field、soft edge、slit crossing 和能量选择。
- 输出继续遵守公共 gun-exit 契约。

### 7.4 Thermionic gun

- 包含 cathode、Wehnelt、gun lens、accelerator、anode aperture、deflector、stigmator 和 C1 aperture。
- 发射边界组合 Richardson–Laue–Dushman supply、Schottky barrier lowering 和 Child–Langmuir space-charge limit。
- 位置和速度使用 flux-weighted planar Maxwell–Boltzmann 分布。
- 后续有限场传播与 FEG 使用共同的相对论追迹路径。

## 8. Column、电子光学和射线传播

### 8.1 坐标与传播

- 电子沿实验室 `+Z`传播。
- 横向状态顺序为 `(x, y, theta_x, theta_y)`。
- 传播网格保留请求终点的精确 Z；最后一步可缩短，不能把 sample 或 detector 平面四舍五入到显示网格。
- CPU、Numba CPU 和 CUDA ray backend 使用一致的区间步长定义。
- 射线历史保留位置、斜率、alive、blocked Z、blocked key、能量偏移和权重。

### 8.2 Magnetic lenses

- Round lens 场由各 lens 的 Bz profile、excitation、校准场强和 `field_polarity`计算。
- excitation 始终为非负 0–100%；Bz 正负由独立 polarity 决定。
- 每个 lens TOML保存 polarity、status 和 source。
- 支持 focal length、Cs、Cc、Larmor rotation、signed field integral、field support 和 peak field diagnostics。
- 机械 housing/yoke/coil/pole 不产生重复光学元件，也不截断数学磁场支持。

### 8.3 Correctors 和 multipoles

- 支持 probe corrector、image corrector、hexapole、quadrupole、twelve-pole 和有限 multipole field。
- Corrector 组件保持各自机械结构和光学 interaction plane。
- nonlinear hexapole/aberration 路径用于生产射线；一阶 Jacobian 计算会明确关闭非线性项。
- Corrector crossover 和残余球差有单独诊断。

### 8.4 Deflectors 和 stigmators

- 支持 gun、condenser、beam shift/tilt、corrector、image/diffraction、AC scan 和 descan deflector。
- Paired deflector 使用上、下两个 TOML interaction plane；即使虚拟平面重合，也不人为制造机械间隙。
- Stigmator 具有 X/Y strength 和 enable 控制。

### 8.5 Apertures、recording devices 和 walls

- Aperture 使用圆形 hard edge、半径、X/Y offset、enable 和 installed 状态。
- 图中区分机械 body centre 和实际 optical stop plane。
- 启用的 aperture 绘制两段实体阻挡区域及中间开口；禁用时保留非阻挡参考。
- Vacuum wall 使用 position-dependent circular X/Y cutoff。
- Wall 只是机械截止，不停止真空中的数学传播，也不裁剪 lens field。
- Aperture、wall、screen、camera 和 detector 竞争时保留最早物理交点。
- 每条被拦截射线记录 Z、X、Y、radius 和 cause。

### 8.6 Crossovers 和 beam statistics

- 检测 gun waist、C1/C2/C3 crossover、各 lens 后 crossover 和 corrector crossover。
- 报告 axial Z、RMS radius 和相关状态。
- Sample beam statistics包含 chief ray、RMS/95%/99%/edge convergence、95% illuminated diameter、wavefront curvature 和 waist offset。

## 9. Direct Alignment

### 9.1 用户级控制

| 编号 | 控制 | 范围 | 耦合对象 | 目标 |
|---|---|---:|---|---|
| DA-001 | Nanoprobe convergence semi-angle | 20–40 mrad | C2、C3 | 以 95% current radial containment 定义 convergence，并约束 waist 到 sample |
| DA-002 | Microprobe illuminated-area diameter | 0.5–2.2 µm | C2、C3 | 95% current diameter，同时约束 wavefront curvature 和最大 0.5 mrad semi-angle |
| DA-003 | Image magnification | 10–1,000,000× | Objective、D、I、P1、P2 | 活动 recording stop 上满足 `B=0`，显示 `|A|` |
| DA-004 | Effective camera length | 0.01–5 m 请求范围 | D、I、P1、P2 | relay live Objective back-focal plane |

### 9.2 求解规则

- 目标、范围、设备集、种子、容差和 calibration provenance 全部来自 operating-mode TOML。
- 求解在独立 state snapshot 和 Qt worker 中执行。
- 只有 target 和 conjugate constraint 都通过精细 production validation，且 live state 未改变时，才能一次性提交全部 lens 值。
- 失败、不可达、设备集不匹配、越界或 stale 结果不改变任何 lens。
- Image 使用等效 thin-lens engineering calibration，同时保留 signed Larmor rotation；这是 non-OEM 模型。
- Diffraction 使用分布场/BFP relay，不使用虚构的单 lens magnification。
- 当前 5 m 请求可能在 P2 达到 100% 时只能连续到约 2.59 m，因此请求范围不等于保证可达范围。

## 10. Sample 公共功能

### 10.1 Finite sample envelope

- 控制 inserted/retracted、mode、size X/Y、thickness、sample centre X/Y 和 scan origin X/Y。
- Sample Z 由活动 instrument TOML 决定。
- Retracted 时仍保留 sample Z 作为 probe reference plane，但 sample interaction thickness 为零。
- Retracted 时不访问 dormant/invalid CIF，也不执行 diffraction、inelastic 或 atomistic interaction。
- Sample snapshot 同时携带 finite box、scan FOV、calculation ROI、probe、orientation 和 Virtual regions。

### 10.2 Real sample 结构与方向

- 可选择 specimen TOML preset 或导入 CIF/MCIF。
- 当前 preset：Vacuum、Silicon [110]、Gold [001]、Amorphous carbon (model)。
- 一个规范化 `(w,x,y,z)` quaternion 是唯一物理方向状态。
- Zone axis `[uvw]`映射到实验室 `+Z`，独立 non-collinear in-plane direction 映射到 `+X`。
- 支持 zone-axis 对准、XYZ incremental tilt 和显式 mouse-drag draft orientation。
- 默认 mouse drag 只旋转观察相机；只有启用 physical edit 后才修改 draft，且必须 Apply 才影响计算。

### 10.3 Sample 结构显示

- 支持 PyQtGraph OpenGL/PyOpenGL 3-D 显示；不支持时使用安全 2-D ball-stick 投影。
- 显示 finite sample box、cell、atoms、bonds、`+Z` beam、scan FOV 和 calculation ROI。
- ASE covalent neighbours 生成 bonds，ASE/Jmol colours 和缩小 covalent radii 生成 element balls。
- 旁置 legend列出当前显示元素。
- 默认 2,500 atom soft rendering limit只裁剪显示窗口，不改变 multislice ROI。
- 用户选择超过 3,000 atoms 时 OpenGL 使用 point-sphere level of detail。
- ROI-local pre-crop structure 有 5,000,000 atom safety limit。

### 10.4 Real 与 Virtual 的强制隔离

- Real sample 不允许人工 `+g/-g`、diffuse ring 或其他用户自定义 diffraction ray branches。
- Real coherent elastic diffraction/scattering 只属于 high-accuracy wave/multislice。
- Real ray branches只表示材料导出的 energy-loss populations及有效 removal。
- Virtual sample 不调用真实样品 IAM/multislice diffraction；其 angular channels完全来自用户表。
- 旧 profile 中遗留的 Real qualitative diffraction 字段可继续 round-trip，但不得影响 Real ray calculation。

## 11. Real sample 非弹性输运

### 11.1 当前通道

| Key | 含义 | 代表能量/角度 |
|---|---|---|
| `real_zero_loss` | 未发生随机 energy-loss；可同时存在 coherent elastic redistribution | 0 eV、0 interaction kick |
| `real_plasmon` | 单次 bulk plasmon / low-loss event | 材料或用户代表 loss，relativistic characteristic angle |
| `real_ionisation` | 单次 aggregate core-ionisation event | 材料或用户代表 binding/loss energy |
| `real_other_inelastic` | 用户提供的其他非弹性通道 | 用户 MFP 和代表 loss |
| `real_plural_inelastic` | 两次或更多非弹性事件 | conditional mean loss 和 RMS angle quadrature |
| effective absorption/removal | 从 tracked transmitted population移除 | 不生成 outgoing branch |

这里的 absorption/removal 不是 60–300 keV TEM 电子在表面的字面“adsorption”。

### 11.2 概率模型

每个独立通道 `k`：

```text
mu_k = thickness / lambda_k
mu   = sum(mu_k)
P_zero_loss       = exp(-mu)
P_single_k        = exp(-mu) * mu_k
P_plural_2_or_more = 1 - exp(-mu) * (1 + mu)
```

若启用 effective absorption MFP：

```text
S_absorbed_survival = exp(-thickness / lambda_abs)
P_absorbed          = 1 - S_absorbed_survival
P_tracked_channel   = S_absorbed_survival * P_channel
```

最终强制检查：

```text
sum(P_tracked_channel) + P_absorbed = 1
```

### 11.3 材料锚点

| Preset | 200 keV total IMFP | Plasmon-component IMFP | Plasmon loss | Ionisation representative loss | 状态 |
|---|---:|---:|---:|---:|---|
| Silicon [110] | 145 nm | 168 nm | 16.7 eV | 99.2 eV | 测量锚点 |
| Gold [001] | 84 nm | 120 nm | 9.0 eV | 84.0 eV | 测量锚点，low/core 分离近似 |
| Amorphous carbon | 150 nm | 154 nm | 25.0 eV | 284.2 eV | density-scaled 近似，建议按膜实测覆盖 |
| Vacuum | disabled | disabled | — | — | 无 interaction |

- Total 与 plasmon IMFP锚点来自材料 TOML并保留来源和适用性说明。
- Aggregate ionisation rate在参考能量由 `1/lambda_total - 1/lambda_plasmon`取得。
- Plasmon 电压变化使用 relativistic log-angle factor，相对测量锚点缩放。
- Ionisation 电压变化使用 BEB `U=B`近似，只做相对缩放，不声称由 BEB 得到绝对截面。
- Characteristic angle用于紧凑 ray quadrature，不是完整 differential cross section。

### 11.4 用户覆盖和 Custom CIF

- Plasmon、ionisation、other 和 absorption MFP可输入。
- Plasmon、ionisation 和 other representative loss可输入。
- 内建材料的 0 值 plasmon/ionisation覆盖表示使用 material default。
- Other 和 absorption 的 0 值表示 disabled。
- Custom CIF 不得借用当前选择 preset 的 inelastic constants。
- Custom CIF 的 plasmon 或 ionisation每个通道都必须同时提供 MFP 和 loss energy；不完整参数对会被忽略并给出 warning。
- Custom CIF 可以只启用一个完整通道，也可以只启用 explicit other/absorption。

### 11.5 输运连接

- 每个 tracked energy state形成一个 absolute-probability ray population。
- 非零 loss population在 source rays之间均匀采样 characteristic-angle azimuth ring。
- Branch energy为原 source energy offset减去代表 loss。
- 新能量进入 Objective chromatic kick、后续磁场传播和 Energy Filter。
- Ray batch有 4,096 post-ray上限以限制峰值内存。
- TEM wave结果明确是 conditional zero-loss coherent observable；elastic 和 inelastic可以在同一电子历史中共存，不能强行合并成互斥标签。

## 12. Virtual sample

### 12.1 Interaction table

每行具有 enabled、name、kind、absolute probability 和 JSON parameters。支持：

- `diffraction_spots`
- `diffuse_ring`
- `gaussian_diffuse`
- `arbitrary_angular`
- `user_screened_power_law`
- `physical_rutherford`
- `absorption`

规则：

- 所有概率为 absolute probability，不自动归一化。
- enabled interaction与 absorption总和不得超过 1。
- Direct/transmitted beam 是精确余量 `1 - sum(enabled probabilities)`。
- Angular quadrature对各通道内部归一化，但不会改变通道的绝对总概率。
- `physical_rutherford`使用 screened relativistic Rutherford、用户 Z、areal density、screening 和角范围；使用 `P=1-exp(-N_areal*sigma)`。
- Rutherford angular integration包含 `2*pi*sin(theta)dtheta` solid-angle Jacobian。
- 该模型明确不是完整 Mott scattering。

### 12.2 Finite regions

- Region 表支持 rectangle、ellipse 和 grayscale map。
- Map格式支持 NPY、PNG、TIF、TIFF。
- Density 被限制到 `[0,1]`；图像 row 0与实验室 `+Y`方向正确转换。
- Region 只在 finite sample slab中生效，外部为 vacuum。
- 可选择用计算得到的 probe对 density做 convolution。
- 选定平面预算按每条 source ray的位置求 density，不再次重复卷积 probe。

## 13. Ray Diagram

### 13.1 几何显示

- 显示 source-to-recording electron paths、活动部件中心、apertures、paired deflectors、sample plane、crossovers、column walls 和 first intercept stops。
- 支持连续 transverse view angle；旋转投影不改变 Z坐标、当前 zoom 或 Z=0屏幕位置。
- 支持 wheel zoom、drag pan、右键菜单、fit、component auto-focus、component labels随 zoom逐步显示。
- 轴向 cursor可拖动，也可从其他 axial plot双击跳转。
- 图中不使用横纵相同比例；提示同时报告最大物理 X angle和 transverse display magnification，避免把示意图误认为 90°电子偏转。

### 13.2 Interaction hue

| Interaction | 基础颜色语义 |
|---|---|
| Incident | 蓝青色 |
| Vacuum/reference | 中性灰蓝 |
| Real zero loss | 浅灰蓝 |
| Real plasmon/low loss | 青色 |
| Real ionisation | 红橙色 |
| Real other inelastic | 黄色 |
| Real plural inelastic | 紫色 |
| Virtual transmitted | 绿色 |
| Virtual diffraction spots | 紫蓝色 |
| Virtual diffuse ring | 橙色 |
| Virtual Gaussian diffuse | 黄色 |
| Virtual arbitrary angular | 青绿色 |
| Virtual screened power law | 粉色 |
| Virtual physical Rutherford | 红色 |

### 13.3 Convergence shade

- 每种 interaction hue内部使用五个 dark-to-bright bins。
- 每条 ray的 convergence定义为 sample plane上相对该 branch current-weighted 3-D chief ray的 semi-angle。
- 亮度在 incident bundle的 weighted 99% convergence semi-angle处饱和。
- Real inelastic characteristic-angle kick先被扣除，不能被误算为 illumination convergence。
- Hue和shade因此是两个独立维度：hue回答“发生什么 interaction”，shade回答“该 illumination ray的 convergence多大”。

### 13.4 任意 Z 平面 interaction budget

选定或拖动 Z 后，使用所有 ray weights计算，而不是只使用图中最多48条显示 ray。显示：

- 当前 Z相对 sample的位置。
- 到达当前 Z的 source fraction。
- 到达 sample的 source fraction。
- 每个 interaction在 sample incident中的 conditional probability。
- 每个 interaction到达 Z的 source fraction。
- 每个 interaction在当前 Z surviving population中的 composition。
- Representative energy loss。
- Pre-sample stops。
- Sample absorption/removal。
- Downstream stops。
- 总概率 conservation error。
- Real material name、`t/lambda`和 combined IMFP。
- 若存在 TEM wave结果，附加 non-exclusive conditional-zero-loss elastic redistribution observable。

## 14. Transverse X-Y

- 该页显示选定 component centre或指定轴向平面的电子束横截面，不是 diffraction pattern。
- X/Y使用相同物理比例，单位为 mm。
- 选择 sample之前的平面时使用 incident branch；sample之后使用 `000`或第一个可用 outgoing branch。
- 排除在该平面上游已被拦截的 rays。
- 最多显示2,000条 rays，但统计以 surviving selected rays计算。
- 红、绿、蓝、黄颜色表示每条 ray在 bundle起始面所属的四个 X-Y象限，用于观察 round-lens image rotation。
- Summary报告 branch、surviving ray数、RMS radius和相对 bundle起点的 orientation rotation。
- Transverse X-Y颜色与 Ray Diagram interaction/convergence颜色是不同体系，不得混用解释。

## 15. TEM wave image 和 wave/multislice

### 15.1 TEM Wave Image 的作用

- 只在 Real sample、Microprobe (TEM)、启用 `TEM image / diffraction`且运行 High accuracy时计算。
- 左图是 specimen-to-Objective CTF的局部 TEM image，显示时做 percentile clipping和 `[0,1]`归一化。
- 右图是 exit-wave diffraction intensity的 log display。
- 它不是最终 projector/camera plane，也不包含 curved Energy Filter branch。
- 它用于观察样品 projected/atomistic potential、multislice propagation、Objective defocus/Cs/aperture和 coherent elastic diffraction对局部图像的影响。
- 线性 diffraction probability另行保留供物理统计；不能从 log display像素直接读取概率。

### 15.2 Potential

- 支持 analytic continuous projected columns和 finite atomistic IAM slices。
- Silicon [110]与Gold [001]提供 atomistic crystal definitions。
- ASE建立结构；abTEM 1.0.10生成 Lobato–Van Dyck neutral-atom independent-atom potentials。
- Custom CIF会 orthogonalise并周期扩展到 `scan ROI + probe padding`与 finite sample的交集，不创建宏观全样品 supercell。
- Custom CIF要求 atomistic IAM和multislice；失败时不得静默换成另一 preset材料。

### 15.3 Multislice

- CPU参考路径使用 complex128 NumPy symmetric split operator。
- 可选 CUDA路径使用 complex64 CuPy。
- 支持 rectangular grid、独立 X/Y sampling、2-D projected potential和显式 `(Z,Y,X)` slices。
- 采用明确 Å / Å⁻¹ FFT约定、2/3 anti-alias bandwidth、uniform或nonuniform slice geometry。
- 报告每 slice最大 phase、初末 integrated intensity、最大 intensity change和sampling support。
- CUDA发生 allocation、propagation、FFT或detector integration错误时，丢弃全部partial result并从头以CPU reference重算。

### 15.4 Frozen phonons

- 支持 enable、configuration count 1–64、global one-axis RMS sigma、per-element RMS table和seed。
- Preset可提供source-qualified thermal sigma；Custom CIF必须有global或per-element explicit RMS。
- 使用可复现的 independent isotropic Gaussian displacement（Einstein approximation）。
- TEM与STEM平均 configuration intensities，不平均 complex exit-wave amplitudes。
- 报告 finite-ensemble relative standard error，但不声称存在通用的“已收敛配置数”。

### 15.5 波模型边界

- 波函数是 conditional zero-loss coherent elastic模型。
- 不包含 bonded charge redistribution、correlated phonons、absorptive/inelastic multislice potential、magnetic specimen field或spin。
- 不提供完整 energy-differential EELS或dielectric response。
- 严格 reciprocal-space support外的 intensity不重新归一化。
- 可选 high-angle Rutherford tail只从严格 wave support之外开始，并单独报告。

## 16. STEM、AC Scan 和 Descan

### 16.1 Raster controls

- AC Scan与Descan显示相同的 enable、pixel size、derived FOV X/Y、frame period、pixels X、lines Y、upper gain和derived lower coupling。
- Pixel size范围0.001 nm到1 mm。
- Pixels/lines范围2–4096。
- `FOV = pixel count * pixel size`表示完整像素 footprint；中心到中心 span为`(count-1)*pixel size`。
- 两者共享 raster clock、pixel count、line count和pixel pitch。
- 超过 physical coil limit或遇到singular transfer时明确失败并回滚。

### 16.2 Scan/descan calibration

- AC upper/lower foils通过active signed first-order optics求解，使sample plane的一阶angular response为零，形成pure shift。
- Descan接收AC command的精确负值。
- Descan lower coupling针对Selected Area Aperture image-reference station求解，使扫描chief ray在该站尽量静止。
- AC和Descan TOML几何关于sample镜像；验证会拒绝破坏该对称性的配置。
- 当前 Objective Aperture、Selected Area Aperture及计算的first image/diffraction plane均按实时 Jacobian分类为`image`、`diffraction`或`mixed`。

### 16.3 STEM Geometry页

- 显示 sample-plane raster和用户选择的downstream recording plane trajectory。
- 报告requested/preview raster、pixel/FOV、sample span、drift pivot、foil symmetry、coupling matrices和residuals。
- Scan geometry和真实射线显示使用同一物理foil planes。

### 16.4 STEM Images页

- 同时显示HAADF、DF和BF detector images。
- 图像坐标为实验室scan X/Y，物理单位相等，允许独立pan/zoom。
- 每幅图显示detector TOML Z、inner/outer active size和由完整signed 2x2 sample-to-detector transfer得到的collection angle。
- 若transfer anisotropic，报告inner/outer angle range而不是单一假精确值。
- Preview使用`geometric_detector_interception`，其polygon/wedge只表示detector clipping boundary，不是atom contrast。
- High accuracy可使用angle-resolved wave/multislice signal。
- Virtual sample使用finite density和absolute interaction probabilities生成signal。
- 提示FOV超出finite sample时外部像素是vacuum；CIF pixel pitch粗于最短atom spacing一半时提示undersampling。

### 16.5 Detector integration和输出

- Physical detector `hit_mask`在完整signed transfer之后求值。
- Detectors按axial order依次截获，upstream hit不能在downstream重复计数。
- 每个STEM结果包含：source fraction image、pA、expected electrons per dwell、可选seeded Poisson counts、dwell time、uncollected、absorbed、truncated和separate high-angle tail。
- 不保存完整4D-STEM cube。
- Real inelastic absorption从source current中显式分离；tracked populations保持概率守恒。
- High-accuracy wave目前为所有tracked energy-loss populations复用coherent elastic angular distribution；紧凑inelastic characteristic angles在Ray Diagram/Energy Filter中输运。这一近似必须继续明确标记。

### 16.6 Playback

- 每次计算只生成一幅完整frame并缓存。
- AC Scan启用时，GUI timer只按frame period逐行播放缓存，不反复运行物理计算。
- 停止scan后保留最后完整frame。
- Ray Diagram复用缓存的AC/Descan basis，随frame time移动scan position；用户仍可旋转view angle而不重算column。

## 17. Energy Filter

### 17.1 安装模式

- Iliad Energy Filter 视为永久安装，不提供无过滤器装配选择。
- 硬件始终安装，但光学 branch 仍可独立 enable；并支持 EELS/EFTEM operating mode、selected loss、energy window、optical integration、MultiEELS、alignment 和 ray tracing controls。
- 可按当前high tension匹配magnetic rigidity、sector field、M12和multipole scales。

### 17.2 当前拓扑

- Entrance aperture。
- Large tapered prism。
- Ten independently powered multipoles `M01–M10`；这些是稳定simulator index，不声称是制造商生产名称。
- XO crossover / optional EFTEM energy slit。
- 独立fast electrostatic shutter。
- Dynamic-focus electrostatic quadrupole mechanical placeholder。
- MultiEELS bias tube。
- Zebra camera deflector。
- Optional EFTEM output plane。
- Zebra EELS detector。

### 17.3 物理追迹

- 从main-column entrance提取带位置、方向、energy offset、colour和absolute source fraction的representative rays。
- Real plasmon、ionisation和plural branch的energy loss进入filter kinetic energy。
- 使用continuous relativistic Boris ray tracing通过sector和multipoles。
- 记录到达/通过slit、EFTEM output、EELS plane和Zebra的状态及stop key。
- Slit transmission按calibrated dispersion、selected loss、window width和blade travel计算。
- 输出entrance、slit、camera和EELS transmitted fraction及current pA。
- Absolute branch weights不因absorption缺失部分而重新归一化。

### 17.4 Energy Filter页面

- 使用独立curved branch view，不把内部部件压扁为main-column轴向标记。
- 显示entrance、prism clear path、M01–M10、XO/slit、electrostatic envelopes、EFTEM output和Zebra active plane。
- X和Z可独立缩放。
- Component label、centre marker和body均可点击导航到左侧编辑器。
- Dashed leaders只作视觉引导，不截获对body的点击。

### 17.5 已知边界

- Prism radius/bend/gap、多数multipole位置和envelope仍为parameterized non-OEM starting values。
- Dynamic-focus quadrupole只有mechanical placeholder，尚无validated field model。
- Straight-column `J_img/J_diff`链只到Energy Filter entrance；curved sector、M01–M10和Zebra坐标尚未进入一阶orientation transfer。

## 18. 诊断页面

### 18.1 Physical Layout

- 按物理比例绘制hollow-cylinder投影、vacuum bores、optical references、sample/stage/holder和recording devices。
- 使用动态screen-space callout packer将名称分配到多行；leader连接到部件中心或外边缘。
- Callout移动只改变显示，不改变TOML geometry。
- 点击名称、marker或body可定位到对应Optical/Mechanical编辑器。

### 18.2 Magnetic Field

- 绘制solver-identical total Bz和每个lens Bz。
- 支持显示/隐藏individual lens curve和rotation labels。
- Tooltip报告peak、excitation、formula、signed integral、polarity/status/source、single-lens和cumulative Larmor rotation。
- 标记image planes及sample-to-plane orientation。
- 选中lens时高亮field support并向parameter panel提供field/focal/Cs diagnostics。
- 双击axial位置可同步Ray Diagram cursor。

### 18.3 Optical Transfer

- 明确显示关系：`r_plane = J_img @ r_sample + J_diff @ theta_sample`。
- `J_img`无量纲；`J_diff`为m/rad，数值上也等于mm/mrad。
- 使用reference ray加四个transverse basis rays，减去reference消除affine beam shift。
- 报告rotation、reflection/handedness、anisotropy、equivalent magnification/camera length和conjugacy residual。
- 可在同一plane分别capture Image和Diffraction状态，计算normalized diffraction-vector-to-image-direction map。
- 结果保留完整signed 2x2 map，不用单一绝对X系数推断rotation或handedness。
- Camera detector axes当前为`uncalibrated_identity` placeholder，不能据此声称绝对crystal orientation。

### 18.4 TEM Wave Image

- 显示Objective CTF image与exit-wave diffraction。
- Summary包含preset、model、slice count、potential model、configuration count、thermal sigma、backend、FOV、pixel和surviving rays。
- Warning覆盖sampling truncation、intensity conservation、CUDA fallback、atomistic fallback和frozen-phonon uncertainty。

## 19. 计算后端、性能和失败策略

### 19.1 Ray backend

- NumPy CPU是基础路径。
- Numba CPU为parallel kernel路径。
- Numba CUDA用于足够大的independent-ray RK4 column propagation。
- Auto在ray数较小时保持CPU，在达到阈值后选择Numba或CUDA。

### 19.2 Wave backend

- NumPy complex128为reference。
- CuPy complex64为可选CUDA路径。
- Auto只在work items足够大时使用GPU，避免小任务launch/transfer成本。
- Resident STEM CUDA pipeline将scan positions、potential configurations、probe formation、multislice、FFT和detector masks留在device，只返回final detector arrays。
- Reusable plan缓存frequency grid、anti-alias mask和slice propagators。

### 19.3 Fallback

- 后端不可用或真实执行失败时，报告原因并使用允许的CPU路径。
- Resident CUDA operation是atomic：任何阶段失败都不能混用partial GPU和CPU结果。
- UR-008不禁止这种已发生错误后的fallback；它禁止在没有实际错误时因猜测而重写兼容性代码。

## 20. 物理定义和概率守恒

- 所有source fractions以发射source current归一化。
- Ray weight必须finite、non-negative并匹配bundle。
- Absolute branch总和不得超过1。
- Real inelastic、Virtual interaction、selected-plane budget、Energy Filter和STEM都必须保留absorption/removal缺失部分，不能通过归一化把它抹掉。
- Elastic coherent scattering与inelastic energy state不是互斥类别；显示时必须注明non-exclusive。
- Nanoprobe用户控制使用weighted 95% radial containment。
- Wave pupil使用更保守的weighted 99% angular containment。
- Transverse display使用X/Y真实比例；Ray Diagram为轴向示意比例。

## 21. 当前明确限制和暂定假设

以下内容不得被文案误称为已经实现：

- 没有完整energy-differential EELS spectrum或dielectric loss function。
- 没有absorptive/inelastic complex potential multislice。
- 没有bonded-charge potential、correlated phonons、magnetic specimen scattering或spin。
- Real inelastic ray angle和energy loss是compact representative quadrature，不是完整line shape。
- High-accuracy STEM对tracked inelastic populations复用zero-loss coherent angular distribution。
- Screened Rutherford不是full Mott elastic scattering。
- Amorphous carbon inelastic preset是density-scaled approximate model，不代表所有carbon film。
- Current atomistic potential是neutral-atom IAM。
- TEM Wave Image只到Objective CTF，不是最终camera image。
- Wave-supported角度外的强度默认不补偿；optional tail必须单独报告。
- Dynamic-focus Energy Filter quadrupole field未实现。
- Curved Energy Filter branch未纳入straight-column first-order orientation map。
- Detector/display绝对轴没有测量校准。
- 多数magnetic-lens field polarities仍是provisional model assumptions。
- Projector/electron-optical calibration和大量mechanical dimensions是non-OEM engineering reconstruction。
- Mechanical pole geometry目前不反向重塑analytic Bz profile。
- Geometric STEM Preview不是样品原子对比。
- 请求范围不代表Direct Alignment每个目标都一定可达。

## 22. 错误处理与安全行为

- 无效TOML、重复键、结构缺失或装配冲突阻止装配应用。
- 无效runtime参数在赋值前拒绝。
- Direct Alignment失败不改变lens。
- Scan/descan calibration失败恢复两个组件的完整旧状态。
- Virtual probabilities超过1时拒绝，不自动归一化。
- Real probability conservation失败时抛出运行错误。
- Custom CIF缺失、不合法或超出atom safety limit时明确报告。
- Memory estimate超过预算时拒绝High accuracy，而不是尝试耗尽系统内存。
- CUDA/CuPy真实失败后完整重算，不交付partial observable。
- Profile和manifest写入使用可恢复或原子操作。
- Compatibility只有出现实际错误时才进入修复范围，见UR-008。

## 23. 代码功能映射

| 功能域 | 主要代码 |
|---|---|
| 启动和主窗口 | `main.py`, `src/temsim/app.py`, `gui/main_window.py` |
| 装配目录和TOML | `assembly_catalog.py`, `module_manifest.py`, `manifest_editor.py`, `column/*` |
| 运行状态和profile | `optics/model.py`, `runtime_parameters.py`, `profile_io.py`, `state.py` |
| Electron gun | `optics/electron_gun/*` |
| Lenses/fields/aberrations | `optics/*lens*.py`, `physics/core.py`, `physics/magnetic_lens_aberration.py` |
| Correctors/multipoles | `optics/probe_corrector.py`, `optics/image_corrector.py`, `physics/*multipole*.py` |
| Deflectors/scan | `optics/*deflector*.py`, `physics/scan_geometry.py` |
| Ray simulation | `physics/simulation.py`, `physics/acceleration.py` |
| Walls/stops/apertures | `physics/column_wall.py`, `physics/aperture_clipping.py`, `physics/recording_clipping.py` |
| Beam/crossover diagnostics | `physics/beam_statistics.py`, `beam_waist.py`, `crossovers.py`, `all_lens_crossovers.py` |
| Direct Alignment | `optics/direct_alignment.py`, `gui/direct_alignment_*` |
| Signed optical transfer | `physics/first_order.py`, `gui/diagnostic_tabs.py` |
| Sample geometry/orientation | `specimen/geometry.py`, `gui/sample_panel.py` |
| Preset/atomistic sample | `specimen/presets.py`, `specimen/atomistic.py`, `configs/specimens/*` |
| Real inelastic | `specimen/inelastic.py`, `physics/interaction_budget.py` |
| Virtual sample | `specimen/virtual.py` |
| TEM wave | `physics/wave_imaging.py`, `physics/multislice.py`, `physics/wave_fft.py` |
| STEM wave/CUDA | `physics/stem_wave_imaging.py`, `physics/stem_cuda_pipeline.py`, `physics/cuda_multislice_plan.py` |
| STEM signal | `detector/stem_signal.py`, `gui/scan_panel.py` |
| Recording devices | `detector/*` |
| Energy Filter | `optics/energy_filter*.py`, `detector/eels_camera.py` |
| 所有中央可视化 | `gui/visualization.py`, `gui/diagnostic_tabs.py` |
| 统一计算结果 | `simulation_pipeline.py` |

## 24. 测试映射与当前验证状态

### 24.1 测试功能域

- Atomistic/CIF/frozen phonon：`test_atomistic_specimen.py`, `test_multislice.py`, `test_wave_imaging.py`。
- Real inelastic和概率：`test_real_inelastic.py`, `test_sample_plane_boundary.py`。
- Virtual sample：`test_virtual_specimen.py`, `test_sample_model_v2.py`, `test_sample_profile_v2.py`。
- GUI和Sample：`test_gui_shell.py`, `test_sample_page.py`。
- Ray/field/corrector：`test_mvp_core.py`, `test_corrector_calibration.py`, `test_magnetic_lens_aberration.py`。
- Direct Alignment/first order：`test_direct_alignment.py`, `test_first_order_transfer.py`。
- Scan/STEM：`test_scan_system.py`, `test_stem_observables_v2.py`, `test_stem_cuda_pipeline.py`。
- CUDA/FFT：`test_compute_backend.py`, `test_cuda_multislice_plan.py`, `test_wave_fft.py`。
- TOML/layout：`test_toml_authority.py`, `test_manifest_editing.py`, `test_column_wall.py`, `test_field_polarity_manifest.py`。
- Detector/Energy Filter：`test_detector_orientation_manifest.py`, `test_energy_filter_physical_layout.py`。
- Gun/timing：`test_electron_gun_timing.py`。

### 24.2 最近验证

- 完整测试：`275 passed, 12 skipped`，无失败，耗时约11分46秒。
- 文档建立前的关键定向测试：77个用例中76通过、1个按环境条件跳过。
- Python 3.12.4环境中`pip check`无依赖冲突。
- `main.py`导入成功。
- Offscreen环境中主窗口成功构建、显示并关闭。
- `compileall`成功。
- `git diff --check`成功；LF/CRLF提示当前不构成启动或功能错误，按UR-008不处理。

## 25. 后续需求编辑区

用户可直接复制以下模板追加需求。不要复用既有编号；没有编号时由实施方分配。

```markdown
### CR-NEW — 变更标题

- 状态：待分析
- 关联既有需求：UR-xxx / 功能章节
- 用户需求：
  - 在这里写需要新增或修改的行为。
- 不允许改变：
  - 在这里写必须保持的已有行为。
- 输入/参数：
  - ...
- 预期输出/界面：
  - ...
- 验收条件：
  1. ...
  2. ...
- 备注或物理依据：
  - ...
```

比较和实施时，不要求用户严格使用模板；自然语言修改仍然有效，但既有需求不能删除。

## 26. 追加式修订记录

| 日期 | 修订 | 结果 |
|---|---|---|
| 2026-08-13 | 建立当前功能与需求活规范；整理启动、装配、GUI、ray、sample、Real/Virtual interaction、wave、STEM、Energy Filter、诊断、限制和测试；建立UR-001至UR-010永久需求台账。 | 文档建立，待以后持续追加 |
| 2026-08-20 | Energy Filter 改为永久安装；移除 Instrument Setup 的 recording system 选择；旧无过滤器配置自动迁移。 | 15 种 gun/column 可选装配统一使用 Energy Filter；历史 TOML 保留验证。 |
