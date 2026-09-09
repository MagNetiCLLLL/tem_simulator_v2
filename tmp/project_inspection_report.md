# 项目检查报告

> Historical audit of commit `704ee3f`, not the current verification status.
> Maintenance on 2026-09-07 removed the one-off reproduction scripts after
> confirming their coverage in the formal tests linked below. Original logs
> are retained; obsolete scripts remain recoverable from commit `12ff7e3`.
> Generated packaging/pytest directories have now been cleaned successfully.

日期：2026-09-07。项目：`E:\tem_simulator_v2`。检查基线：`704ee3f`。

发现 4 个可复现的功能问题。未修改项目源代码、测试代码或仪器配置；本次新增文件均为检查日志、复现脚本和临时验证产物。

## 1. [P1] 对中状态下波动 STEM 计算失败

位置：[stem_wave_imaging.py:930](E:/tem_simulator_v2/src/temsim/physics/stem_wave_imaging.py:930)。

存在 `record_plane_plan` 时，探测器掩码是 `(batch, y, x)` 三维数组；当 `detector_center_shifts_mrad` 为空，积分代码却执行 `diffraction[:, mask]`，使三维衍射数组被按四维索引，抛出 `IndexError`。

生产路径 `calculate_stem_scan_frame` 会构造并传入 plan，而对中状态的 detector shifts 可以是 `None`。因此会导致这类 STEM 计算失败。GUI worker 会捕获并显示计算错误，并非整个应用进程崩溃。

用真实求解函数、32×32 真空网格、1×2 扫描及合成探测平面复现，无 monkeypatch。报错：`too many indices for array: array is 3-dimensional, but 4 were indexed`。

建议根据实际掩码维度处理积分，覆盖 plan + 无偏移及 plan + 有偏移两条路径。

[Maintained recording regressions](../tests/test_stem_recording_deflection.py) · [Historical log](project_inspection_repro_stem.log)

## 2. [P1] 探测平面路径漏掉 Descan 位移

位置：[stem_wave_imaging.py:825](E:/tem_simulator_v2/src/temsim/physics/stem_wave_imaging.py:825)；相关路径：[first_order.py:262](E:/tem_simulator_v2/src/temsim/physics/first_order.py:262)。

使用 plan 后，代码绕过原来的逐扫描位置 detector shift 处理；构造 plan 的传递矩阵又未向传播器传入偏转器 kick events，因此 plan 的仿射偏移不能补上 Descan 的贡献。这会使波动 STEM 的物理探测器积分忽略相关偏转，产生错误信号。

两项独立证据：

- 等效 BF 探测器从 0 改为 100 mrad shift 时，无 plan 的积分从 `[[1,1]]` 变为 `[[0,0]]`；带 plan 时两次均为 `[[1,1]]`。
- 使用项目真实 Descan 组件，把静态 X kick 从 0 改为 1 mrad，同一状态的显式射线传播产生约 22.5 µm 的 X 位移，但 plan 使用的传递函数仍给出零位移。

现有 `test_angle_resolved_stem_applies_per_probe_descan_detector_shift` 单测通过，但没有传入 plan，未覆盖此组合。建议在新路由中正确纳入静态和逐扫描位置的偏转贡献，避免遗漏或重复计入。

[Maintained Descan/record-plane regressions](../tests/test_stem_recording_deflection.py) · [Wave regressions](../tests/test_wave_imaging.py) · [Historical Descan log](project_inspection_repro_descan.log)

## 3. [P2] 保存配置丢失透镜像差估算模式

位置：[profile_io.py:56](E:/tem_simulator_v2/src/temsim/profile_io.py:56)。

GUI 用 `cs_mm = cc_mm = None` 表示“Focal-length estimate (provisional)”模式；保存 profile 时直接省略所有 `None` 字段。加载流程先应用装配默认值，因此缺失字段恢复为固定系数，用户选择的估算模式丢失且没有提示。

按 GUI 完整加载顺序复现，Objective 有效 Cc 从 `2.7996476885 mm` 变为 `2.0 mm`，状态从 `provisional principle model` 变为 `configured`，跳过字段列表仍为空。建议明确持久化估算模式标记，并在装配默认值加载后恢复；不能依靠省略字段表达此状态。

[Maintained profile regressions](../tests/test_profile_optional_values.py) · [Historical log](project_inspection_repro_profile.log)

## 4. [P2] 拒绝超限缓存写入时丢失旧缓存

位置：[artifact_store.py:916](E:/tem_simulator_v2/src/temsim/artifact_store.py:916)。

`_prune_to_quota` 未先确认新对象自身能否放入配额，就开始淘汰旧缓存；新对象最终仍超限时，又删除新对象并抛出错误。因此一次失败的写入可以清空此前有效缓存。

在合法的 1 MiB 配额下先保存小数组，再写入 2 MiB 数组：收到 `ArtifactTooLargeError` 后，旧缓存读回为 `None`，引用数量从 1 变为 0。建议先检查新对象及必需元数据独立占用是否超额，再执行历史缓存淘汰。

[Maintained quota regressions](../tests/test_artifact_quota.py) · [Historical log](project_inspection_repro_cache.log). The retired script deliberately returned exit code 1 when reproducing the former defect.

## 验证结果

- 环境：Python 3.12.4，Qt / PySide6 6.8.3；`pip check` 和核心模块导入通过。
- 337 个现有 Python 文件语法检查通过；`git diff --check` 通过。
- 共收集 1,241 个测试；选取 22 个文件、322 项重点回归，包含配置、缓存、后台计算、离屏 Qt 显示、EDS、4D-STEM 和扫描采样等路径。
- 复核后的唯一测试结果为 **317 通过、2 失败、3 跳过**。3 项跳过来自 CUDA / CuPy 后端不可用。
- 首次批次为 316 通过、3 失败、3 跳过。其中多进程锁测试因检查包装器从标准输入启动而失败（Windows spawn 无法重新打开 `<stdin>`）；改用标准 `python -m pytest` 单独重跑通过，不计为项目缺陷。
- 剩余两项失败均位于 [test_mvp_core.py:809](E:/tem_simulator_v2/tests/test_mvp_core.py:809)：测试要求说明中包含旧字符串 `not an OEM production drawing`，而两个 recording TOML 已换用新的非 OEM 来源说明。独立重跑确认是文案断言未同步，不能据此判断几何计算错误。
- 隔离构建 wheel 成功；模拟标准 Windows 安装布局后，配置、材料、支持网格和 9 个 SVG 均可定位。30 种装配组合的默认操作参数保存/加载无差异；第三项缺陷需要用户主动选择像差估算模式才会触发。
- 测试还出现 Thermionic 路径浮点溢出/无效减法警告，以及 pyqtgraph 销毁信号警告；本次未将这些警告确认为新的独立缺陷。

未运行全部 1,241 项测试或大规模 GPU 扫描。四项功能缺陷分别通过小型独立脚本验证，不能用现有重点回归大部分通过推断这些问题不存在。

[主回归日志](E:/tem_simulator_v2/tmp/project_inspection_tests.log) · [多进程测试标准启动重跑](E:/tem_simulator_v2/tmp/project_inspection_multiprocess.log) · [文案断言独立重跑](E:/tem_simulator_v2/tmp/project_inspection_projector_failure.log)

临时打包目录 `tmp/inspection_packaging` 的清理被自动审批策略阻止，工具仅返回“blocked by policy”；因此本次临时验证目录和日志均予保留。
