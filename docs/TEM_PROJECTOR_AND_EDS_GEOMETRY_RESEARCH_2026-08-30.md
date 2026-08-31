# TEM 投影镜与 Super-X / Ultra-X EDS 几何研究记录

日期：2026-08-30

本记录保存当前公开资料研究结论，供后续用户提供投影系统横截面、EDS
组件图和带尺度照片后继续修订。证据分为：制造商公开规格、专利/论文所示
结构、单台仪器数据，以及非 OEM 工程重建。后两类不得写成制造商尺寸。

## 1. 投影镜磁极结论

1. D、I、P1、P2 在光学模型中都是轴对称磁圆透镜。产生局域轴向磁场需要
   磁路和极面/磁隙；因此“投影镜完全没有 pole piece”不是合适的物理描述。
2. 但是，软件中的每一个透镜名称并不必然对应一套机械上完全独立的
   “上极靴 + 下极靴 + 独立磁轭”。公开专利明确存在三磁极、双磁隙、双线圈
   的 projector lens；中间磁极可同时界定相邻的两个磁隙。也存在 compound
   pole-piece assembly。由此不能从控制通道名称反推独立零件数量。
3. 当前 TOML 将 D、I、P1、P2 各自画成 `two_pole_single_gap`，适合作为能与
   当前分布场模型对应的可编辑原理模型，但仍是 provisional topology，不是
   FEI/Thermo Fisher 生产横截面。
4. 在拿到服务横截面或拆解组件图以前，不合并相邻磁极，也不删除现有
   projector pole-piece children。未来图纸应优先判断：共享中间磁极、共享
   磁轭、线圈数量、各磁隙位置、真空管连续性和实际装配边界。

主要依据：

- [US4450357A：三磁极、两磁隙、两线圈 projector lens](https://patents.google.com/patent/US4450357A/en)
- [US2472315A：objective/projector compound pole-piece assembly](https://patents.google.com/patent/US2472315A/en)
- [US5304801A：多级 intermediate/projector excitation channels](https://patents.google.com/patent/US5304801A/en)

这些资料证明可行的磁路拓扑，不证明当前 Thermo Fisher 仪器采用其中某一
专利实施例的具体尺寸。

## 2. Super-X：可确认的几何

公开论文中的 FEI Super-X 模型给出：

- 四个无窗 silicon-drift detector（SDD）围绕样品对称布置；
- 每个晶片有效面积 30 mm²，总有效面积 120 mm²；
- 总立体角随 objective pole-piece geometry 为约 0.7 sr 或 0.9 sr；
- 一篇经实验核对的几何模型使用 18° polar/take-off angle，方位角为
  45°、135°、225°、315°；该角度属于论文所建仪器几何，不应扩展成所有
  Super-X 安装的统一 OEM 数值；
- 样品杆、网格、样品倾转和极靴遮挡会分别改变四个探头的实际收集率。

尺寸解释：30 mm² 是 active sensor area，不是 detector housing 的横截面积。
若仅为了直观比较，把面积换算成圆形等效晶片，直径为
`2*sqrt(30/pi) = 6.18 mm`。若再假设圆片正对样品、无准直器并平均分配总
立体角，则 0.7/0.9 sr 对应约 12.8/11.2 mm 的等效样品到晶片距离。这个距离
高度依赖形状、倾角和准直，不能作为 Super-X 机械尺寸写入配置。

主要依据：

- [Ultramicroscopy 164 (2016), 51-61：四探头、30 mm²、18°及四方位几何](https://doi.org/10.1016/j.ultramic.2016.02.004)
- [US8410439B2：TEM 物镜内部环形多探头、约 30 mm² SDD、低取出角及极靴安装](https://patents.google.com/patent/US8410439B2/en)

## 3. Ultra-X：已知量、未知量和本项目取值

制造商公开确认：

- windowless Ultra-X Detection System；
- 无样品杆遮挡时立体角大于 4.45 sr；分析双倾样品杆条件为 4.04 sr；
- 零倾转灵敏度约为 Super-X 的 6 倍、Dual-X 的 2.5 倍；
- 与大间隙 S-TWIN objective pole piece 配套。

六段信息由公开的 Spectra Ultra 实验方法报告和一份用户实测 EMD 的六个
SpectrumStream 信号共同支持；Thermo Fisher 当前公开 datasheet 本身没有写
segment count。因此配置状态为“公开仪器报告 + 用户数据支持，但非 OEM
datasheet 明示”。用户数据中的 detector elevation/take-off metadata 为
约 32.055°-32.065°，项目只取 32.06°作为该单台仪器的参考绘图中心线。

当前没有找到可归属于量产 Ultra-X 的公开数值：

- 每段 active area 或晶片长宽/形状；
- 样品到晶片距离；
- collimator、cold shield、支撑环及外部 package 尺寸；
- 六段的绝对方位角相位。

因此项目不把 Super-X 的 30 mm²复制给 Ultra-X，也不把专利实施例中的
50-100 mm²范围冒充产品规格。配置仅保存六段对称的 60°间隔；0°相位是可替换
的非 OEM 绘图约定。

用于检查量级的角度派生量：

- 4.45 sr 占完整 `4*pi` 球面的 35.4%；4.04 sr 占 32.1%；
- 假设六段平均分配，分别为至少 0.7417 sr/段和 0.6733 sr/段；
- 换算成相同立体角的圆锥，半角分别约 28.1°和 26.8°。

圆锥半角只是 acceptance summary，不反演晶片面积或距离。当前 Physical
Layout 只画两个相反方位的二维投影、中心取出方向和等效接受边界；探头头部
大小与距离是 display-only schematic，不写回 TOML，也不参与电子光路。旧的
显示距离把未知尺寸的实体头部画进了 Objective pole-piece 的二维轮廓。现在
实体 active-face/housing 顶点由上下磁极最大外半径加 1 mm **显示间隙**反求
位置；这只消除错误的二维材料重叠。1 mm 不是 Ultra-X 安装尺寸，中心线和
接受边界仍可穿过轴对称截面，因为它们表示取出方向/角接受而不是实体材料。
在拿到 Ultra-X 横截面以前，不能据此声称真实 3D collimator clearance 或
无 pole-piece/holder shadowing，也不应为适配示意图而改写现有物镜磁极间隙。

主要依据：

- [Thermo Fisher Spectra Ultra datasheet](https://documents.thermofisher.com/TFS-Assets/MSD/Datasheets/spectra-ultra-ds0361-materials-science-datasheet.pdf)
- [Thermo Fisher Iliad Ultra / Ultra-X datasheet](https://documents.thermofisher.com/TFS-Assets/MSD/Datasheets/iliad-ultra-datasheet-ds0510.pdf)
- [公开 Spectra Ultra 实验方法：six-segmented Ultra-X](https://doi.org/10.1021/acsami.4c18243)
- [US8410439B2：多探头在 TEM lens 内、朝向样品并可由 pole piece 支撑](https://patents.google.com/patent/US8410439B2/en)

## 4. P2 后方 detector 距离与 viewing chamber

公开 Titan 纵向示意图能够确认的顺序是 P2 后进入 viewing/recording section，
其中 HAADF 在最上游，之后依次出现 main fluorescent screen、DF/BF detector
stack 和 camera。Fischione 的公开资料也确认 STEM detector 是带真空波纹管的
可缩回机构，但这些资料均不给出 P2 外壳末端到 detector active plane 的生产
尺寸。因此“HAADF 紧接 P2”在拓扑上合理，不能把当前 7.25 mm 当作 OEM 尺寸。

当前两套 recording TOML 的 P2 外壳下端均为 local Z = 772.5 mm；有效信号面
相对该下端的距离为：

| 记录面 | P2 下端到有效面 |
|---|---:|
| HAADF | 7.25 mm |
| Fluorescent Screen | 127.25 mm |
| DF | 217.25 mm |
| BF | 287.25 mm |
| Camera | 399.75 mm |

所以真正“非常靠近”的只有 HAADF；后续屏和探测器已经相距 127--400 mm。
Physical Layout 现加入 `post_projector_detector_chamber`，从 P2 下端连续到
local Z = 1100 mm，包含 HAADF、screen、DF、BF；180/200 mm ID/OD 和末端位置
是可替换的非 OEM 绘图包络。Camera 保留在该示意 chamber 下游。这个部件是
`mechanical_only` 且 `axial_vacuum_context_only`，不会新增传播面、改变真空
cutoff、移动 active plane 或触发 preset lens-strength 重算。

主要依据：

- [University of Antwerp Titan³ column Figure 1.4](https://repository.uantwerpen.be/docman/irua/6e3afa/130499.pdf)
- [University of Oregon Titan column topology](https://scholarsbank.uoregon.edu//bitstreams/d56b2b6e-e2d1-4fbe-abe0-eafe58bfee67/download)
- [Thermo Fisher MiCo detector layout manual](https://documents.thermofisher.com/TFS-Assets/MSD/manuals/mico-user-guide-1-16.pdf)
- [Fischione Model 3000 retractable HAADF detector brochure](https://sfilev2.f-static.com/image/users/390742/ftp/my_files/PB3000.pdf?id=26921072)

## 5. 当前实现边界

- 当前机器只安装一套 `eds_detector_system = "ultra_x"`。产品与角接受数据
  只在 `configs/detectors/eds/UltraX.toml` 定义一次；五种 column TOML 只保存
  各自 sample plane 上的零厚度安装行并引用该文件。
- Super-X 仅作为位置和量级研究对照，不存在 Super-X TOML、第二套 EDS
  aggregate 或型号切换器。
- Ultra-X 是 Objective assembly 的 transverse mechanical child，并与 sample
  plane 共用一个零厚度 aggregate anchor。
- 它不是 aperture、轴向电子 detector 或 propagation stop；本次机械修改不
  重算任何 preset lens strength。
- 现在只实现结构、来源和角接受几何。EDS X-ray generation、吸收、holder
  shadowing、探测效率、脉冲堆积、能量响应和谱定量尚未实现。
- 后续横截面/组件图应尽量包含标尺或一个已知尺寸，并标明探头相对样品杆
  轴、objective upper/lower pole、aperture port 和冷却/真空接口的方向。

## 6. Ultra-X 取出线、z 位移与 Objective 极靴约束

本轮把用户数据中的 `32.06°` 当作该台仪器的参考取出角，并保留制造商公开的
`4.45 sr`（无 holder 遮挡）和 `4.04 sr`（分析双倾 holder）数据。结论如下：

1. 仅凭角度和总立体角不能得到毫米位置。若额外假设每段都是正对样品的圆片，半径为
   `a`，则每段 `4.45/6 sr` 的等效圆锥半角为 `28.1203°`，距离才可写为
   `d = a/tan(28.1203°)`。生产 Ultra-X 的每段 active area、形状和距离没有公开，
   所以当前 TOML 不写入 `d`。
2. 当前 Physical Layout 中 4.5 mm 宽的 active-face 线段只是显示符号。若错误地把它
   当成直径 4.5 mm 的圆形敏感面，会得到 `d = 4.2103 mm`、轴向偏移 `2.2349 mm`、
   径向位置 `3.5682 mm`，它会落入极靴/样品台区域。反过来，在当前约 59.58 mm 的
   显示距离上维持同一等效圆锥，需要每段约 63.68 mm 等效直径和 3185 mm² 面积，
   同样证明显示头部不能作为物理尺寸。
3. 在保持 32.06° 时，探头沿同一取出线前后移动不会改变该直线与极靴的交点；只改 z
   而固定径向位置又会改变取出角，并在敏感面固定时改变立体角。因此不存在一个独立的
   “移动 z、其余量全部不变”的解。
4. 对原 5.4 mm 极靴间隙、96 mm 极靴外径、10 mm 平头 OD 和默认 23.56 mm 锥鼻，
   中心线在极面处半径为 4.3109 mm，小于 5 mm 平头半径；在锥肩处也有约
   6.0730 mm 径向侵入。原轮廓确实遮挡参考中心线。
5. 现在保留 5.4 mm 间隙、96 mm 外径和 5.76 mm bore，使用非 OEM 约束设计：
   平头 OD 8.0 mm，锥鼻轴向长度 27.5584 mm，名义锥面角 57.94°（相对轴）。锥面与
   32.06° 参考取出线平行，在平头和锥肩之间得到约 0.3109 mm 的恒定径向中心线间隙。
   零余量极限分别是平头 OD 不大于 8.6217 mm、锥鼻长度不小于 27.3637 mm。
6. 这只证明二维轴对称截面中的中心线可通，不证明完整 4.45 sr 接受域无遮挡。六个
   60° 方位间隔、32.06° elevation 的轴线在球面上最小只相隔 50.1427°，而每段等效
   圆锥直径为 56.2406°；六个等效圆锥会互相重叠，所以它们只能是 aggregate angular
   summary，不是六个真实、互不重叠的圆形准直孔。完整验证仍需 Ultra-X 敏感面、
   collimator、上下分布和 pole-piece 三维窗口/切口图。

本轮只改机械轮廓和显示诊断，不重算 Objective 或 preset 透镜强度。

补充依据：

- [Thermo Fisher Spectra Ultra datasheet：S-TWIN 与 Ultra-X 4.45/4.04 sr](https://documents.thermofisher.com/TFS-Assets/MSD/Datasheets/spectra-ultra-ds0361-materials-science-datasheet.pdf)
- [Thermo Fisher Iliad Ultra datasheet：当前 Ultra-X 公开规格](https://documents.thermofisher.com/TFS-Assets/MSD/Datasheets/iliad-ultra-datasheet-ds0510.pdf)
- [Zaluzec：XPAD 实验立体角和 holder penumbra](https://www.osti.gov/servlets/purl/1894230)
- [XPAD 早期结果：custom ZTwin pole piece](https://er-c.org/wp-content/uploads/2021/05/pico2021-programme.pdf)

## 7. Projection-chamber differential-pumping aperture

FEI/Thermo Fisher 的 column/vacuum 手册把 projection chamber 与上游 column
之间的小孔明确称为 **differential pumping aperture**。它位于 projection
chamber 顶部，在 D、I、P1、P2 全部投影透镜之后、HAADF 和 viewing/recording
detector 之前。Tecnai/Talos 系列公开手册给出的参考孔径为 200 µm；该值只用于
当前非 OEM 系列参考，不能写成 Titan/Iliad 生产尺寸。材料和轴向厚度没有可靠
公开值。

它与 post-column spectrometer entrance aperture 是两个部件：前者是 column /
projection-chamber 真空分区的固定限流孔，后者位于更下游的 EELS 能谱仪入口并
定义能谱接受。FEI 模式手册还表明 DPA 的共轭角色随 projector 模式交换：TEM
image/EFTEM image 条件可在这里形成 diffraction/cross-over plane；TEM
diffraction 或 STEM-EELS image-coupling 条件则可在这里形成 image plane，而
衍射图样位于 spectrometer entrance aperture。因此机械部件本身不能被静态命名为
永远的 diffraction plane。

当前实现采用：

- 名称：`Projection-Chamber Differential-Pumping Aperture`；
- 键：`projection_chamber_dpa_aperture`；
- 位置：两套 recording TOML 均为 local Z = 772.5 mm，即 P2 housing 末端与
  `post_projector_detector_chamber` 起点；HAADF 有效面在其后 7.25 mm；
- 几何：0.2 mm 系列参考 bore、20 mm 当前周围通道/示意 plate OD；未知实体厚度
  用零长度边界和独立 schematic display thickness 表示；
- 状态：固定、不可插拔，但 TOML 直径仍可作为未来设计变量编辑；
- 光学边界：本阶段为 `mechanical_only`，没有 `optical_reference_local_z_mm`，
  不进入 `APERTURE_KEYS`、ray clipping 或 Direct Alignment，也不重算任何 preset。

主要依据：

- [Thermo Fisher Talos Basic General Information：200 µm DPA 位于 projection chamber 与 column 之间](https://imf.ucmerced.edu/sites/g/files/ufvvjh1081/f/documents/tem_talosbasicguide_1.pdf)
- [FEI Tecnai column description：projection-chamber differential pumping boundary](https://www.dartmouth.edu/emlab/docs/fei_tecnai_f20_column_description_doc.pdf)
- [FEI Tecnai Modes：DPA、EELS entrance aperture 与模式相关共轭关系](https://www.dartmouth.edu/emlab/docs/fei_tecnai_f20_modes_doc.pdf)
- [Titan TIA on-line help：STEM-EELS image coupling 中 DPA 与入口孔径的不同平面角色](https://www.manuallib.com/download/2023-10-20/Titan%20on-line%20help%20manual%20--%20TIA.pdf)
