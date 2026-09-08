# STEM 计数显示验证

Images 页面新增 Display 选择框：Ideal intensity（理想强度）、Expected
electrons（期望电子数）、Poisson counts（泊松计数）。勾选 Scanning Parameters
底部的 Generate seeded Poisson counts 后会自动切到计数显示；运行 High accuracy
生成计数。若当前图像暂停刷新，需要取消 Pause refresh 才能显示新结果。

期望电子数与 Poisson 计数按每个探测器共用同一色阶，单位为电子/像素。
切换显示和逐行播放不会重新抽样。旧结果没有存储对应数据时会显示 Unavailable。
Poisson 开关/seed 改变可以复用已有 STEM 强度；改变扫描周期则仍需检查时间相关的
扫描/消扫描物理依赖。

## 本次对比图

这是已保存 Si [110] 信号的计数后处理，用来验证新显示功能。未重新求解波传播，
也未修改正在运行的软件或用户当前剂量。采用先前修正尾部散射重叠后的近焦结果：

- 数据：`outputs/si110_underfocus_100nm/near_focus_tail_corrected/raw_scan.npz`
- 原采集参数：`outputs/si110_cif_5nm_64px_002nm/acquisition_gpu_1024_final/parameters.json`
- 样品：Si [110]，直径 10 nm、厚度 5 nm。
- 扫描：64 × 64，0.02 nm/像素，视场 1.28 × 1.28 nm。
- 本次计数演示：有效源电流 100 pA，10 µs/像素，seed 42。
- 用原采集的探测角区间；这不代表用户当前镜筒/探测器设置。

这里原 DF 的接收角约为 1–7 mrad，处于约 24.8 mrad 的照明盘内。
它展示的是该低角环形探测器的原有信号，不是排除了直射盘的纯暗场信号。
本次 Poisson 后处理没有调整探测器几何；后续排直射盘的 DF 几何验证和新采集
单独保存在 `outputs/df_direct_beam_clearance`，原图保留以记录其实际来源。

| 通道 | 平均期望电子数/像素 | 平均抽样计数/像素 | 零计数像素数 |
| --- | ---: | ---: | ---: |
| HAADF | 5.00507 | 4.99658 | 38 |
| DF | 273.16721 | 273.16479 | 0 |
| BF | 2.21745 | 2.25098 | 716 |

`ideal_expected_poisson.png` 按行展示理想强度、期望电子数和计数。理想图采用
自动对比度；下两行共用各通道的计数色阶，因此明暗程度不应直接与第一行比较。
`ui_ideal.png`、`ui_expected.png`、`ui_poisson.png` 是实际 Qt 控件的离屏截图。
`counts.npz` 保存原始强度、期望值和整数计数；`verification.json` 记录输入散列、
参数和检查结果；`readout_state.json` 仅记录本次后处理的上下文。

原归档含有限波网格与尾部散射近似；本次验证不证明其角域、空间网格或热振动
采样已收敛，也不包括探测器电子学噪声。界面继续显示原采集的角域诊断。

复现（在项目根目录运行）：

```powershell
.\.venv\Scripts\python.exe scripts/render_stem_poisson_demo.py --capture-ui
```
