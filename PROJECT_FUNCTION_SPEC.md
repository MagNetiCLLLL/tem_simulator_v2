# TEM Simulator v2 — 当前功能与需求活规范

> 文档用途：本文件是后续大规模功能修改的唯一“活规范”入口。
>
> 当前状态：持续与工作区代码同步；总体建模原则更新于 2026-08-30。
>
> 程序入口：`main.py`；项目版本：`0.1.0`。

## 0. 项目总体目标与建模原则

TEM Simulator v2 的目标不是复制某一台现有仪器的固定操作档位，而是建立一个**以真实 TEM 为结构和物理基线、以理想连续设计变量探索新型 TEM 的研究模拟器**。

### 0.1 真实基线与理想设计自由度

- **真实基线**：在有照片、公开资料、可追溯参数或可靠物理模型时，尽量保留真实 TEM 的部件顺序、机械拓扑、相对比例、真空通道、光学作用、电子传播、孔径截断、像差、样品相互作用和探测过程。
- **理想设计自由度**：真实仪器因加工、档位、行程、电源、发热、磁饱和、磁场溢出或其他工程因素而受限的参数，在模拟器中可以作为连续设计变量开放，用于探索现有硬件不能直接实现的新型 TEM 方案。
- 理想化只移除明确声明的工程限制，不得把一个仅改变标签、却不进入传播或成像模型的控件称为“可调”。参数改变后，所有已实现且相关的物理计算仍必须使用该值。
- 新增或修改 `mechanical_only` 配件时，先依据结构证据、装配关系和机械间隙建立实体；已有光路、共轭面、aperture stop 或 preset lens strength 不得成为添加条件，也不得因此自动重算。只有用户明确要求计算光路，或该配件被显式升级为参与电子传播的光学组件时，才建立相应光学约束。
- 真实证据、非 OEM 工程重建和理想设计变量必须分别标注；不得把照片比例、暂定值或理想可调范围描述成制造商标定或现实可达性能。

### 0.2 连续 aperture 与理想 lens 参数

- 所有圆形 aperture 的有效开口直径或半径均保持连续可调，不量化、吸附或限制为真实 holder 上的若干固定孔位。Insert/retract 可以保留为独立机械状态，但不能改变开口尺寸连续可调的原则。
- C2 aperture holder 照片中的四个机械选择位置只用于理解 holder、带孔片、螺钉和连接杆的真实结构；它们不是模拟器的四档 aperture 运行状态，也不要求按照四个位置重建当前模型。
- Lens strength、lens axial position 及同类设计参数应能作为连续变量探索。理想设计模式不把线圈温升、磁饱和、磁场溢出、电源额定值、机械行程或现有仪器档位当作不可越过的物理上限，除非用户明确启用相应的真实硬件约束模型。
- “无限可调”表示不受已知实机工程额定范围限制，并不表示向数值算法传入数学上的 `inf`。有限浮点范围、求解器收敛、内存和防止无效状态所需的数值保护仍然有效，但必须标记为数值边界，而不是现实 TEM 的性能边界。
- 如果以后加入真实硬件约束模式，它必须是显式、可识别且可关闭的模型；不得在理想设计模式中静默 clamp、snap 或恢复到现有仪器的离散档位。

### 0.3 EDS 证据与理想化边界

- EDS detector 是样品附近的离轴 X-ray collection system，不是轴上电子记录面，也不应因 Physical Layout 绘图而成为 electron propagation stop。
- 当前机器只安装一套通用名称的 EDS detector array：角接受参数由 `configs/detectors/eds/EDS.toml` 单点定义，五种 column TOML 只保存样品平面安装位置引用。产品资料只作为几何来源和 provenance，不把 Ultra-X 作为新信号系统、GUI 或结果的名称，也不添加第二套 EDS 或型号切换器。
- 其他产品的公开晶片数据不得移植到当前 EDS。当前仅使用可追溯的 windowless、六段支持证据、立体角和单台仪器参考取出角；active area、sensor distance、crystal shape 和 package envelope 保持 unknown，等待用户横截面/组件图。
- Solid angle 及由其派生的 equivalent-cone angle 属于物理接受量；为看清结构而选取的探头头部大小和绘图距离属于 display-only geometry，两者不得混用。
- EDS 实体示意不得与已解析的 Objective pole-piece 材料轮廓重叠；当前 1 mm 仅为 display-only separation，不是产品间隙、真实 collimator clearance 或无阴影证明。角中心线/接受边界不是实体，可在轴对称二维截面上穿过磁极投影。
- EDS 信号计算必须与 detector 品牌身份分离。样品和支架中 primary 或 elastically-scattered electron track segment 都可以产生 K/L/M 壳层空位。默认显式点计算必须由有限三维几何中的逐事件弹性 Monte Carlo 生成路径；总 core-loss MFP 不得代替元素弹性或 EDS 电离截面。直线路径只能作为清楚标记的参考模式。
- 当前离线弹性 provider 是 100–300 keV 的 relativistic screened-Rutherford 暂定模型，不是 ELSEPA/full-Mott 或晶体 channeling 模型。遇到 `Z > 30` 必须报告精度警告；未来高精度 provider 应使用有合法来源的 ELSEPA differential cross section，不得直接打包受再分发限制的 NIST SRD 64 数据表。

### 0.4 P2 后方探测区及机械修改边界

- Titan 公开资料可约束 P2 后方 viewing/detector section 和 HAADF、主屏、DF、BF、Camera 的相对顺序，不能把示意图比例转换成 OEM 绝对轴向尺寸。
- 当前 P2 下端到 HAADF、主屏、DF、BF、Camera 有效面的距离分别为 7.25、127.25、217.25、287.25、399.75 mm。只有 HAADF 极近；7.25 mm 与 chamber 尺寸均保持明确的非 OEM 暂定值。
- `post_projector_detector_chamber` 只是 TOML 权威的机械上下文，不拥有新光学面或真空 cutoff。单个机械组件的新增、移动或外形修改不得自动重算 preset lens strength；只有用户明确要求计算光路或机械改动实际改变光学约束时才进入相应光学求解任务。
- column 与 projection chamber 的边界必须单独显示固定的 `projection_chamber_dpa_aperture`，位于 P2 后、HAADF 前；它不能与下游 `energy_filter_entrance_aperture` 合并。当前新增阶段只建立机械真空限流孔及证据，不要求满足既有光学共轭约束、不加入 ray clipping，也不触发 preset lens-strength 重算。

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
| UR-002 | 明确 Transverse X-Y 显示当前轴向平面的电子束横截面，并解释其颜色含义。 | 已实现 | `diagnostic_tabs.TransverseBeamView`；颜色连续表示相对束流质心的初始极角 |
| UR-003 | Real sample 模式不得生成人为定义的衍射束；只有 Virtual sample 可以配置人为衍射、散射和吸收通道。 | 已实现，约束 | `physics/simulation.py`、`specimen/virtual.py` |
| UR-004 | Ray Diagram 支持两种互补颜色语义：不同 convergence semi-angle 用同一色相的深浅表示；不同 interaction 类型用不同色相表示。 | 已实现 | `physics/simulation.py`、`gui/visualization.py` |
| UR-005 | 用户选择任意轴向平面后，显示该平面各种 interaction 电子的比例，并依据物理概率而不是任意显示权重计算。 | 已实现 | `physics/interaction_budget.py`、Ray Diagram 选定平面表格 |
| UR-006 | Real sample 加入真实非弹性输运，包括 plasmon、ionisation、其他非弹性、复数碰撞及有效 absorption/removal。 | 已实现为概率守恒的紧凑输运模型 | `specimen/inelastic.py`、材料 TOML、Energy Filter、STEM |
| UR-007 | 修改后必须检查项目仍可启动，并验证主要功能链。 | 已实现，约束 | 离屏 GUI 冒烟检查、编译、定向与全量测试 |
| UR-008 | 在项目仍能正常启动时，不主动处理假设性的兼容性问题；只有出现实际兼容性错误时才针对错误处理。 | 约束 | 后续开发策略；不因警告或推测改动兼容层 |
| UR-009 | 把当前所有功能详细整理到一个 Markdown 文件；以后用户可大量修改该文件，实施方读取、比较、修改项目并回写文件。 | 已实现 | 本文件 |
| UR-010 | 用户需求可以重新排版和重写，但不允许删除。 | 约束 | 本文件第 1.2 节及本台账 |
| UR-011 | Transverse X-Y 不使用简单的四象限离散颜色；必须使用类似 DPC 的连续 360 度旋转色盘，并在页面内直观显示原始方向图例。 | 已实现 | `InitialDirectionColourWheel`、`TransverseBeamView`；+X 为 0°，朝 +Y 逆时针增加 |
| UR-012 | Transverse X-Y 必须允许选择某个配件的中心 Z，也必须允许选择任意 Z；最后一次选择立即更新图谱并在重新计算后保持。 | 已实现 | `VisualizationWorkspace.jump_to_ray_position`、`TransverseBeamView.focus_component/focus_z`、可移动 Z 游标 |
| UR-013 | 在 Camera、荧光屏和 BF/DF/HAADF 等物理记录面加入 point-spread response；保留原始射线，不把任意 Z 或物镜 CTF 错当成探测器 PSF。 | 已实现首阶段 | TOML `point_spread_*`、`detector/point_spread.py`、`detector_response_image`、Transverse X-Y 响应叠层；Zebra/EFTEM 输出仍待后续接入 |
| UR-014 | D、I、P1、P2 的实体包络之间不得保留大段空白，真空通道内径必须一致；所有 detector/camera 使用上游上表面作为信号收集平面，所有标记必须落在该表面。 | 已实现 | recording TOML 的 5 mm 包络间隙与 20 mm vacuum ID；`signal_collection_surface = upstream_top_surface`；Physical Layout、Ray Diagram、Transverse X-Y |
| UR-015 | 所有真实圆磁透镜必须显示可追溯的本征 Cs/Cc；只在 probe/sample 与 Objective/image 系统维护完整有效像差列表，并真实比较校正前后，不为校正器四极/六极重复添加 Cs。 | 已实现原理模型 | `optics/aberrations.py`、GUI `Aberrations` 页、TEM/STEM wave phase；未提供实机/OEM 标定 |
| UR-016 | 项目必须以真实 TEM 的结构和物理为基线，同时把现实中仅受工程条件限制的参数开放为理想连续设计变量，用于未来新型 TEM 设计；两类信息必须明确区分来源和适用边界。 | 约束 | 本文件第 0 节；后续所有 GUI、状态、配置、求解器和验证设计 |
| UR-017 | 所有圆形 aperture 的有效开口尺寸必须连续可调；C2 holder 照片中的四个机械位置只作结构参考，不得成为四档选择、尺寸吸附或运行时量化依据。 | 已实现，约束 | Aperture 浮点 radius/diameter 控件与 clipping；Physical Layout 的 Pt 带孔片、螺钉和连接杆仅作结构示意 |
| UR-018 | Lens strength、lens position 及同类设计参数应支持不受实机温升、磁饱和、磁场溢出、电源额定值、机械行程或离散档位限制的连续设计探索；只保留明确标注的数值安全边界，真实硬件限制只能作为显式可选模型。 | 部分实现，约束 | 当前 lens/position 使用连续数值；现有 0–100% excitation、Direct Alignment 范围和部分位置编辑仍是待解耦的实现边界 |

| UR-019 | 在 Objective/sample 区域加入一套离轴 EDS 探测阵列；系统、GUI 和结果统一使用通用 EDS 名称，不以 Ultra-X 标记新系统。制造商立体角、论文/实测段数与单台仪器取出角仍须分级记录，未公开的晶片面积、距离和外壳尺寸不得由其他产品参数或专利范围代替。 | 已实现机械/角接受阶段 | `configs/detectors/eds/EDS.toml` 单一几何定义、column 安装位置引用、Physical Layout 两方位投影、`detector/eds_geometry.py`；产品名只可留在 provenance |
| UR-020 | EDS 的实体 Physical Layout 示意必须与 Objective pole-piece 轮廓无材料重叠，但不得为了适配未知 detector 外壳而擅改磁极形状或伪造产品尺寸。 | 已实现，保持证据边界 | EDS solid polygon 由解析磁极最大外半径加 1 mm display-only separation 定位；测试检查全部 active-face/housing 顶点，真实 3D clearance/shadowing 待横截面 |
| UR-021 | 核查 Titan 中 P2 到 HAADF 及后续探测器的合理距离；Physical Layout 应显示独立 viewing/detector chamber，并把相对拓扑证据与绝对暂定尺寸分开。 | 已实现非 OEM 机械阶段 | 两套 recording TOML 的 `post_projector_detector_chamber`、manifest 包含/边界验证、Physical Layout chamber；保留全部 active plane 和 preset |
| UR-022 | 在 P2 与 projection chamber 的边界加入独立 differential-pumping aperture；使用正确名称并与 Iliad spectrometer entrance aperture 分离。新增机械配件时不受当前光学共轭约束阻止，也不自动计算 preset 透镜强度。 | 已实现机械阶段，光学耦合待显式需求 | 两套 recording TOML 的 `projection_chamber_dpa_aperture`、manifest 边界/证据验证、Physical Layout；0.2 mm 仅为 Tecnai/Talos 系列参考，Titan 尺寸未确认；无 clipping/optical reference/preset 重算 |
| UR-023 | 引入通用 EDS 信号模拟：支持 3.05 mm 圆形 Cu/Au 商业方孔网和真空虚拟支架、不同 mesh 数；样品与支架中的 primary 及弹性散射后电子均可产生 X-ray，元素/壳层产额必须来自物理截面和原子数据库。 | 已实现弹性轨迹与特征线阶段 | `specimen/elastic_transport.py` 在有限样品、连续方孔/侧壁、网杆、rim 中按指数自由程逐事件生成 3-D 路径；历史数严格等于本次 column calculation 中实际到达 physical sample plane 的射线数，不再单独输入。每条射线的 X/Y、入射 tx/ty（含旋转）、能量偏移和权重均进入输运，并保留 straight reference。`detector/eds_atomic.py` 与 `detector/eds_signal.py` 计算 Bote–Salvat K/L/M 特征线、自吸收、立体角、理想效率与 Poisson。当前 screened-Rutherford 对 Z>30 仅为暂定近似；bremsstrahlung、电子减速、跨层吸收、vacancy cascade、secondary fluorescence、ELSEPA 和晶体 channeling 尚未实现。 |
| UR-024 | Scanning Image 必须允许暂停图像刷新；暂停时固定显示上一幅完整 frame，而不是当前半幅 raster。扫描时钟和 Ray Diagram 播放继续运行。 | 已实现 | `gui/scan_panel.py` 的 `stemPauseImageRefresh`、独立 paused-display frame 缓存及回归测试 |
| UR-025 | `Mode` 是样品结构来源的唯一开关：Real sample 只能导入 CIF/MCIF；Silicon [110] 等理想 TOML reference sample 必须属于 Virtual sample。不得在 Real 区域重复提供 TOML/CIF 来源选择。 | 已实现 | `Sample.specimen_mode`、`specimen/source.py`、Sample 模式专属控件、schema/profile 迁移与验证 |
| UR-026 | 深色主题中的所有 checkbox 必须具有清楚可辨的选中状态，不能因系统默认 indicator 与背景颜色接近而难以识别。 | 已实现 | `app.APPLICATION_STYLE` 全局 indicator 样式；SVG 分别覆盖未选中、选中、半选、hover 和 disabled 状态 |

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

代码审计值：**10 个模块 TOML、480 条变体级部件定义、196 个逻辑部件键、15 种可选无冲突装配组合**。

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

中央工作区包含九个顶层页面：

1. `Ray Diagram`
2. `Physical Layout`
3. `Energy Filter`
4. `Transverse X-Y`
5. `Sample`
6. `EDS`
7. `Scanning Image`
8. `Illuminating Image`
9. `Optical Transfer`

`Magnetic Field` 是 `Ray Diagram` 下方可开关面板，不再占用顶层页。`Scanning Image` 使用固定水平左右分栏：左栏标签为 `Scanning Parameters` 与 `Probe Aberrations`，右栏标签为 `Geometry` 与 `Images`；不得把已经包含 Geometry/Images 的完整 ScanControlView 再嵌入 Scanning Parameters。`Illuminating Image` 内含成像结果与 `Image Aberrations` 子页。Optical Transfer 列表固定排在 Illuminating Image 之后。

全局深色主题必须使用高对比 checkbox indicator：未选中是白边深色框，选中是青蓝底白色勾，半选是紫色底白色横线；hover 与 disabled 使用各自独立状态，不能依赖低对比的操作系统默认图形。

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
- 当前 0–100% 是现有实现使用的归一化校准坐标，不是项目对未来理想 lens strength 的实机硬上限；后续扩展必须遵守 UR-018，并把数值安全范围与真实硬件额定范围分开。
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
- 所有圆形 aperture 的有效半径/直径是连续浮点设计变量；真实 holder 的离散孔位数量不进入尺寸量化、吸附或 preset 选择。
- C2 aperture holder 的四个位置仅证明真实机械 carrier 拥有多个可选孔位。模拟器继续使用单个连续可调 opening，并保留照片支持的 holder、Pt 带孔片、螺钉和连接杆拓扑。
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

- 控制 inserted/retracted、mode、envelope shape、size/diameter、thickness、sample centre X/Y 和 scan origin X/Y。
- 新建状态的默认样品是 Virtual `Silicon [110]`：直径 3 mm 的圆片、厚度 10 nm，zone axis `[110] -> +Z`，面内 `[1 -1 0] -> +X`（与预设横向晶胞轴一致）。3 mm 宏观包络不替代独立的 wave calculation FOV，因此不会为整张圆片构造原子超胞。
- `disk` 圆边界真实参与直线路径、弹性 Monte Carlo、EDS、Virtual density 和 wave potential 的有限包络裁剪；不是把 3 mm 写入方形 X/Y 后仅改变标签。GUI 的单个 Diameter 控件同步 X/Y；需要设计型样品时仍可选择 Rectangle。
- Sample Z 由活动 instrument TOML 决定。
- Retracted 时仍保留 sample Z 作为 probe reference plane，但 sample interaction thickness 为零。
- Retracted 时不访问 dormant/invalid CIF，也不执行 diffraction、inelastic 或 atomistic interaction。
- Sample snapshot 同时携带 finite disk/box、scan FOV、calculation ROI、probe、orientation 和 Virtual regions。

### 10.2 Real sample 结构与方向

- Real sample 只显示并使用 `Imported CIF / MCIF`，没有 TOML preset 或第二个 structure-source 控件。
- 未选择 CIF/MCIF 时，Real sample 保留光学 sample reference plane，但样品波动、EDS 和非弹性材料交互不可用；不得回退到 Virtual preset。
- 旧 profile/state 的 retired source 若为 CIF，迁移为 Real；若为 preset，迁移为 Virtual。
- 一个规范化 `(w,x,y,z)` quaternion 是唯一物理方向状态。
- Zone axis `[uvw]`映射到实验室 `+Z`，独立 non-collinear in-plane direction 映射到 `+X`。
- 支持 zone-axis 对准、XYZ incremental tilt 和显式 mouse-drag draft orientation。
- 默认 mouse drag 只旋转观察相机；只有启用 physical edit 后才修改 draft，且必须 Apply 才影响计算。

### 10.2.1 Virtual reference sample

- TOML reference sample 只位于 Virtual sample：Vacuum、Silicon [110]、Gold [001]、Amorphous carbon (model)。
- 选中的 reference 为 high-accuracy TEM/STEM wave 与 EDS 提供理想结构/材料；Virtual Ray Diagram 的用户定义 angular channels 仍是独立的理想化概率模型，不伪称由 TOML 晶体自动推导。
- 用户定义 angular channels 默认关闭并显式 opt-in，避免把 reference crystal 与人工概率表混为一体，也避免默认高精度任务为未请求的方位采样分配大量 ray history。
- `specimen_preset_key` 与 `cif_path` 可作为切换模式时的休眠设置保留，但物理计算只能读取当前 mode 所拥有的一个来源。

### 10.3 Sample 结构显示

- 支持 PyQtGraph OpenGL/PyOpenGL 3-D 显示；不支持时使用安全 2-D ball-stick 投影。
- 显示 finite sample disk/box、cell、atoms、bonds、`+Z` beam、scan FOV 和 calculation ROI。
- ASE covalent neighbours 生成 bonds，ASE/Jmol colours 和缩小 covalent radii 生成 element balls。
- 旁置 legend列出当前显示元素。
- 默认 2,500 atom soft rendering limit只裁剪显示窗口，不改变 multislice ROI。
- 用户选择超过 3,000 atoms 时 OpenGL 使用 point-sphere level of detail。
- ROI-local pre-crop structure 有 5,000,000 atom safety limit。

### 10.4 Real 与 Virtual 的强制隔离

- Real sample 不允许人工 `+g/-g`、diffuse ring 或其他用户自定义 diffraction ray branches。
- Real coherent elastic diffraction/scattering 只属于 high-accuracy wave/multislice。
- Real ray branches只表示材料导出的 energy-loss populations及有效 removal。
- Virtual sample 的 Ray Diagram angular channels 完全来自用户表；启用 high-accuracy wave 时，TEM/STEM 图像可独立使用所选 TOML reference 的 IAM/multislice，二者不得混称为同一个散射模型。
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
- 每条 ray 的颜色连续表示其在 bundle 起始面相对束流质心的极角；+X 为 0°，朝 +Y 逆时针增加到 360°，用于观察 round-lens image rotation。
- 页面内显示与 ray 完全相同映射的 DPC 风格 360° 环形色盘，并明确标注 +X/+Y/-X/-Y 与 0°/90°/180°/270°；不再使用四象限离散颜色。
- 恰好位于 bundle 起始质心、因而没有可定义方位角的 ray 使用中性灰色。
- 选择配件时显示该配件 centre Z；Go to Z、轴向图选择或拖动青色 Z 游标时显示任意 Z。游标拖动期间实时更新，最后一次选择在重新计算后保持。
- 选择 Camera、Fluorescent Screen、BF、DF 或 HAADF 时，在原始方向着色 ray 点下方叠加峰值归一化的灰度记录面响应。先按实际 disk/annulus/square `hit_mask` 接受电子，再做正向二维 Gaussian PSF 卷积，最后再次应用有限敏感区边界；进入孔区或越过外缘的扩散响应会丢失并报告 retained weight。
- PSF 使用记录面物理单位 mm；TOML 分别管理 `sigma_x`、`sigma_y` 和主轴逆时针 rotation。当前数值明确标为 `provisional_model_parameter`，可编辑但不是仪器测量标定。
- 任意 Z 和非记录面配件只显示几何 ray 截面，不显示 PSF。PSF 不修改 ray trajectory，也不卷积 specimen-to-Objective CTF 的 TEM Wave Image。
- Summary报告 branch、surviving ray数、RMS radius和相对 bundle起点的 orientation rotation。
- Transverse X-Y颜色与 Ray Diagram interaction/convergence颜色是不同体系，不得混用解释。

## 15. TEM wave image 和 wave/multislice

### 15.1 TEM Wave Image 的作用

- 在存在活动结构来源（Real CIF/MCIF 或 Virtual TOML reference）、Microprobe (TEM)、启用 `TEM image / diffraction`且运行 High accuracy时计算。
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
- `Pause refresh`只冻结HAADF/DF/BF图像刷新，并固定显示上一幅完整frame；不得冻结在半幅raster。扫描时钟与Ray Diagram playback继续运行，恢复刷新后显示当前frame进度。
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

### 18.2 Magnetic Field（Ray Diagram 内嵌可开关面板）

- Ray Diagram 主图、Selected-plane interaction budget 和 Magnetic Field 使用纵向 splitter；两个高对比度水平手柄均可拖动。首次显示以 Ray Diagram 为主要空间，开关 Magnetic Field 后保留用户本次调整的相对高度。
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

### 18.4 Illuminating Image

- 显示Objective CTF image与exit-wave diffraction。
- Summary包含preset、model、slice count、potential model、configuration count、thermal sigma、backend、FOV、pixel和surviving rays。
- Warning覆盖sampling truncation、intensity conservation、CUDA fallback、atomistic fallback和frozen-phonon uncertainty。

### 18.5 EDS

- EDS 是 Sample 右侧的独立顶层页；Sample 页只负责样品结构、包络、取向和相互作用模型。
- 三维弹性 Monte Carlo 使用本次 column calculation 实际存活到 sample plane 的完整 ray bundle，不提供独立轨迹数输入。
- 样品面 X/Y、tx/ty、累积旋转、energy offset 和 source-current weight 均参与输运；上游 survival fraction 只作用一次。
- GUI 用同一批三维历史的 U-Z/V-Z 正交投影展示轨迹和碰撞；`U=X cosφ+Y sinφ`，`V=-X sinφ+Y cosφ`。φ 与 Ray Diagram 双向同步，旋转仅重投影已保存的 X/Y，不重跑输运。EDS spectrum/line list 位于单独子页。
- EDS 仍由 `Calculate point EDS` 显式触发；单个机械参数修改不得自动重算 EDS 或 preset lens strength。
- EDS 页另有 `Run sample-region high accuracy` 手动按钮。入口平面读取本次整柱计算缓存的上游相空间；样品/支架内部只运行有限几何弹性输运和 EDS；前向终态电子在 sample reference 重新注入真实 Objective/下游透镜、aperture、recording plane 与 column wall 传播，出口平面保存为显式交接诊断。相关参数变化必须使旧结果失效，不得自动运行。
- 主 Ray Diagram 用独立颜色叠加样品区 primary、backscatter、screened-Rutherford 弹性事件、零权重定性 secondary marker、生成的 characteristic X-ray 和进入 EDS 角接受的 X-ray。X-ray 方向按球面均匀抽样；因 active face/distance 未公开，只能使用与已知 aggregate solid angle 面积严格相同的角接受 surrogate，路径显示端点不得伪称 sensor intersection。
- Rutherford 是当前 elastic scattering provider，不是与 elastic 并列的独立粒子类别；channeling 属于 coherent multislice/wave，不能作为随机截面再次加入或与 MC 尾部重复计数。

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
- 当前 EDS 弹性轨迹按 bulk density/mass fraction 处理为 independent-atom、amorphous-style transport；没有晶体 channeling/coherent diffraction，也没有从 multislice wave 提取经典路径。
- 当前 EDS 初始电子来自 physical sample plane 的 geometrical ray phase space，包含 source sampling、probe convergence、X/Y tilt/rotation、energy offset 与权重；它不把 coherent aberrated probe wave 或晶体 channeling强行解释成唯一经典路径。弹性碰撞不改变单条历史的 kinetic energy，也没有 nuclear recoil 或 inelastic angular kick。
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
- 当前 0–100% lens excitation、Direct Alignment 目标范围、aperture maximum radius 和部分静态位置编辑仍是实现/校准边界；它们不得被解释为本项目最终接受的实机物理上限，UR-018 所述理想连续设计模式尚未完整实现。

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
| Energy Filter | `optics/energy_filter*.py`, `optics/energy_filter_detector.py` |
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
- Detector/Energy Filter：`test_detector_orientation_manifest.py`, `test_detector_point_spread.py`, `test_energy_filter_physical_layout.py`, `test_eds_detector_geometry.py`, `test_post_projector_detector_chamber.py`。
- EDS/弹性轨迹：`test_eds_signal.py`, `test_elastic_transport.py`, `test_specimen_support.py`。
- Gun/timing：`test_electron_gun_timing.py`。

### 24.2 最近验证

- 2026-08-31 Scanning Image pause-refresh 与 Real-sample 来源互斥完成后，分组离屏回归共 `196 passed`：41 项 scan/sample/profile/aberration、41 项完整 GUI shell、114 项 specimen/wave/EDS/core。既有 16 ms projection timer 测试单独通过；一次组合重负载运行超过其 1 s Qt event-loop timeout，未修改阈值。`compileall` 与 `git diff --check` 通过。
- 2026-08-31 弹性轨迹阶段新增/更新的截面、CDF、几何、输运、EDS、profile 与 GUI 定向测试全部通过。随后运行非 Nanoprobe 重标定全套：`405 passed, 1 skipped, 6 deselected`，零失败；六个 deselection 仍是按用户要求不重算的两组 C2/C3 Nanoprobe 标定 family。`compileall`、`main.py` import smoke、`pip check` 与 `git diff --check` 通过。
- 2026-08-31 样品面 ray-bundle EDS 与 GUI 重排完成后，最终非重标定全套为 `408 passed, 1 skipped, 6 deselected`，零失败；6 个 deselection 仍为同两组 Nanoprobe C2/C3 live-solve family。定向 EDS/profile/GUI 测试、串行 `compileall`、`pip check`、offscreen `MainWindow` 启动及精确主标签顺序 smoke、`git diff --check` 均通过。
- 2026-08-30 projection-chamber DPA 机械实现完成后共收集 386 项测试；按“不为机械修改重算光路”的要求，运行非 Nanoprobe 重标定集合，合计 `379 passed, 1 skipped, 6 deselected`。六个隔离参数点未重跑；其最近一次结果仍为 `2 passed, 4 failed`。
- 四个已知失败均属于 C2 长度/场标定和 aperture 位置改变后、按用户要求尚未重算的 Nanoprobe live-solve 参数点：100 µm→30 mrad、200 µm→60 mrad、60 µm→18 mrad、140 µm→42 mrad。本次 EDS 机械修改不改写这些 preset 或 warm starts。
- Ultra-X 五种 column TOML、证据边界、角接受派生量、Physical Layout、窄窗口和“非轴向真空壁/非光学组件”测试全部通过。
- EDS solid polygon 对 Objective pole-piece 最大外半径的 1 mm display-only separation、两套 recording TOML 的 post-P2 chamber 边界/包含关系、P2 到五个有效面的距离和离屏渲染测试通过；相关组合回归 `142 passed`。
- Projector 包络、detector 上表面、PSF、orientation 与其余 GUI 定向测试通过。
- Python 3.12.3环境中`pip check`无依赖冲突。
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
| 2026-08-20 | Transverse X-Y 初始方向颜色由四象限离散编码改为 DPC 风格连续 360° 色盘，并在页面内加入方向图例。 | 离轴束以自身起始质心为颜色圆心；方向、半径、能量和强度含义保持分离。 |
| 2026-08-20 | 接通配件中心与任意 Z 到 Transverse X-Y 的统一选择链。 | 最后一次配件或任意 Z 选择立即生效、游标拖动实时更新，并跨重新计算保持。 |
| 2026-08-20 | 为 Camera、荧光屏和 BF/DF/HAADF 记录面加入 TOML 权威的二维 point-spread response。 | 原始射线保持不变；Transverse X-Y 仅在选择物理记录面时叠加有限敏感区 PSF 响应并报告保留权重；参数明确为非实机标定的可调默认值。 |
| 2026-08-20 | 收紧 D-I-P1-P2 实体包络并统一真空管，同时把所有 detector/camera 的信号面固定到上游上表面。 | 外壳/磁轭相邻间隙均为 5 mm、全栈 vacuum ID 为 20 mm；保留已验证磁场中心；Physical Layout、Ray Diagram、Transverse X-Y 标记统一使用信号面。 |
| 2026-08-20 | 加入分层混合像差模型。 | 所有圆透镜显示本征 Cs/Cc 及来源；probe/image 系统使用 C1/A1/B2/A2/C3/S3/A3/C5/Cc，有向项带方位角；校正比较只切换非线性六极场，TEM/STEM 波相位使用同一有效系数。 |
| 2026-08-30 | 确立“真实 TEM 基线 + 理想连续设计变量”的总体目标，并记录 C2 aperture holder 的解释边界。 | 新增 UR-016 至 UR-018；四个真实 holder 位置仅作结构证据，所有圆形 aperture 保持连续可调；lens strength/position 的实机工程限制不得成为未来理想设计模式的静默硬上限。 |
| 2026-08-30 | 记录 projector 磁极拓扑研究，并在 Objective/sample 区域加入 Ultra-X EDS。 | D/I/P1/P2 的独立双磁极结构继续标为 provisional；Ultra-X 以离轴六段 aggregate 和证据分级角接受模型加入 Physical Layout，未知晶片/距离/外壳尺寸不作伪造；机械修改未重算 preset 透镜强度。 |
| 2026-08-30 | 将 EDS 配置收敛为单一型号 TOML。 | 新增 `configs/detectors/eds/UltraX.toml`；五种 column 只引用这一产品定义，Physical Layout 只显示一套 Ultra-X，不建立 Super-X 实例或切换器。 |
| 2026-08-30 | 消除 EDS 实体示意与 Objective pole-piece 的二维材料重叠，并核查 P2 后方探测器距离。 | Ultra-X 实体 polygon 使用 1 mm display-only separation，未改磁极或产品参数；两套 recording TOML 新增非 OEM post-P2 viewing/STEM-detector chamber。Titan 资料支持 HAADF-first 拓扑，不支持把当前 7.25 mm 写成 OEM 尺寸；全部 active plane 与 preset 保持不变。 |
| 2026-08-30 | 在 P2/projection-chamber 边界加入独立 differential-pumping aperture，并明确机械新增不受当前光学约束阻止。 | 新增 UR-022；两套 recording TOML 和 Physical Layout 使用 `projection_chamber_dpa_aperture`，与 Iliad entrance aperture 分离。0.2 mm 只标记为 Tecnai/Talos 系列参考；无 ray clipping、optical reference 或 preset 重算。 |
| 2026-08-30 | 开始引入通用 EDS 信号系统，并取消新系统的 Ultra-X 品牌标记。 | 修订 UR-019/UR-020，新增 UR-023；几何定义改为 `EDS.toml`、所有用户界面和结果使用 EDS。加入 Cu/Au/真空支架与 50–500 mesh catalog、Bote–Salvat K/L/M 电离、xraylib 直接空位弛豫、特征线/自吸收/立体角/Poisson 点谱；弹性轨迹生成与连续谱仍明确待实现。 |
| 2026-08-31 | 将 EDS 下一阶段重点转为真实弹性散射轨迹。 | 新增有限样品/连续方孔网几何中的事件驱动 3-D Monte Carlo、指数自由程、元素散射体抽样、屏蔽 Rutherford 偏转、可复现 seed、终态/截断统计、EDS 路径汇总及 X-Z 代表轨迹图。保留 straight reference；明确 Z>30/ELSEPA、晶体 channeling、能损与连续谱边界；未重算任何 preset 透镜强度。 |
| 2026-08-31 | 让 EDS 使用真实样品面 ray bundle，并重组中央图像/诊断页。 | 移除 EDS 独立轨迹数；每条到达 sample plane 的射线携带 X/Y、tx/ty、旋转、能量与权重进入三维输运。EDS 从 Sample 拆为右侧独立页并显示 X-Z/Y-Z 投影和 spectrum。主标签重排为 Ray Diagram、Physical Layout、Energy Filter、Transverse X-Y、Sample、EDS、Scanning Image、Illuminating Image、Optical Transfer；Magnetic Field 内嵌可开关，probe/image aberrations 分别进入对应图像子页。未重算任何 preset 透镜强度。 |
| 2026-08-31 | 加入手动样品局部高精度边界、X-ray 路径和下游电子交接。 | EDS 页增加入口/出口平面与手动运行按钮；材料 MC 返回真实 flight 与所有 terminal electron state，前向电子重新进入真实下游 column propagation。Ray Diagram 叠加球面均匀 characteristic X-ray、exact-aggregate-solid-angle 角接受、elastic/Rutherford/backscatter 和零权重 secondary marker；channeling 继续由 wave/multislice 独占，未伪造独立随机路径。未重算任何 preset 透镜强度。 |
| 2026-08-31 | 将默认样品设为 3 mm disk、10 nm 厚的 Si `[110]`。 | 新状态显式选择 Virtual Silicon `[110]`、`[1 -1 0]` 面内方向和 3,000,000 nm 圆片直径；圆边界统一进入显示、Virtual density、wave 裁剪、直线 EDS 与弹性 Monte Carlo 侧壁求交。旧 schema 无 shape 的状态按原矩形语义迁移；未重算任何 preset 透镜强度。 |
| 2026-08-31 | 允许调整 Ray Diagram 内部各栏目高度。 | 主 ray panel、selected-plane interaction budget 与可开关 Magnetic Field 改为三段纵向 splitter；主 ray 默认占最大比例，手柄使用深色主题高对比样式，隐藏并重新打开 Magnetic Field 时保留用户调整。未改变 ray、field 或 lens 计算。 |
