# DF 排除直射照明盘

是否重叠同时取决于收束半角、有效 camera length、探测器尺寸、Z 位置和偏心。
本项目按当前镜头计算的完整一阶映射检查：

```text
r_detector = J_img · r_sample + J_diff · theta_sample + offset
```

还包含每个扫描点的 AC/Descan 位移。Jdiff 的奇异值给出角度到探测器半径的
有效 camera length 范围；不能只用一个固定探测器尺寸判断是否重叠。

## 软件中的入口

重启后，Images 的 DF 照明盘重叠提示下有 **Exclude direct beam…**。
它根据匹配当前完整计算结果的镜筒状态提出内径、外径，显示角域、有效 camera
length、直射盘间隙和目标 TOML。点击 **Save DF dimensions** 才会将两项尺寸
一起写入当前使用的记录模块，经过原有装配校验并重新加载。

默认内角在输入照明半角外增加 `max(5 mrad, 20% × 半角)` 的余量；现有外径
仍能提供至少 5 mrad 环宽时保留，否则向外扩展，并检查机械腔体内径。角域
建议相对 probe chief，常规图像标题的参考角域仍按原来的光轴参考显示。
这些检查针对当前一阶传递模型和给定的照明盘，不把 95% 束流统计半角当作
所有原始射线的绝对硬边缘。

上游 HAADF 允许遮挡下游 DF。计算沿实际部件顺序扣除已截获电子，既不移动
HAADF，也不补偿被它截获的信号。因此 DF 的几何角域可以与 HAADF 重叠。

改变 camera length、投影镜强度、收束或几何后，旧尺寸建议不能继续保存；
需要重新计算和检查。保存尺寸后，原图片保留并标记为过期，运行 High accuracy
才会得到新图。历史 bank、旧暂停帧和外部已经改写的源 TOML 不能套用旧建议。

## 本次 Si [110] 重算

有效结果目录：`si110_32px_004nm_df60_100_final`。

| 参数 | 原存档 | 本次 |
| --- | ---: | ---: |
| DF 内径 / 外径 | 2 / 14 mm | 60 / 100 mm |
| DF 接收表面 Z | 2804.15 mm | 2804.15 mm |
| DF Jdiff 有效 camera length | 0.998850626 m | 0.998850626 m |
| DF 光轴参考内角 / 外角 | 1.00115 / 7.00806 mrad | 30.03452 / 50.05754 mrad |
| 照明 95% 半角 | 24.80467 mrad | 24.80467 mrad |
| 原始射线边缘半角 | 26.37510 mrad | 26.37510 mrad |
| 扫描 | 64 × 64，0.02 nm/像素 | 32 × 32，0.04 nm/像素 |
| 视场 | 1.28 × 1.28 nm | 1.28 × 1.28 nm |

样品沿用用户的 Si CIF，[110] 取向、10 nm 直径、5 nm 厚度，300 kV；
1024² 波网格、4 nm 波窗口、4 个 frozen-phonon 配置、seed 707。
镜头强度、位置、原有近焦状态和采集周期保留。探测器内外径属于机械 TOML，
本次在独立 `instrument_inputs` 装配中修改，没有将这组尺寸设成所有 camera
length 通用的全局默认值。

`mask_preflight.json` 检查全部 1024 个扫描位置、360 个方位和 55 个 0–27 mrad
径向点，DF 直射盘命中为零。30–50 mrad 是本组镜筒的参考角域，不是通用标准。
全部真实上下游 aperture 和 detector 遮挡均保留。

`haadf_df_bf_comparison.png` 和 `haadf_df_bf_absolute_scale.png` 为本次重新计算
的信号；前者各通道自动对比度，后者从零开始。`raw_scan.npz`、TIFF 保存原始
数值。另有 100 pA、10 µs/像素、seed 42 的 Poisson 读出示例；它是单独的计数
后处理，未用该曝光时间替换光学扫描过程的原始周期。

这次新扫描用于验证调整后的 DF，不代表空间网格或热振动配置已经收敛。
高角端仍使用有限波网格和近似 screened-Rutherford 尾模型；DF 排除直射盘
也不意味着其晶体散射对比变为完全非相干。

## 复现与数据来源

在项目根目录执行，输出必须使用一个新的目录：

```powershell
.\.venv\Scripts\python.exe scripts/run_df_clearance_scan.py --output outputs/df_direct_beam_clearance/reproduction
```

原始归档由脚本的 `--archive` 指定，默认是
`outputs/si110_cif_5nm_64px_002nm/acquisition_gpu_1024_final`。
原操作 profile 必须配套独立 `instrument_inputs` catalog 才能复现机械尺寸；
仅在主界面导入普通 operating profile 不会导入这组 DF 尺寸。

本目录的 `ui_*_layout_fixture.png` 是明确的合成布局测试截图，不是实际
Si 数据或其尺寸建议；测试没有保存配置。带 `PREPARATION_ONLY.txt` 的早期
准备目录不含有效扫描结果，仅保留在本地并由 `.gitignore` 排除，不随仓库分发。
