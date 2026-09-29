# Physical source and cache requirements

These requirements were explicitly specified by the user on 2026-09-11.

- Preserve existing physical components and capabilities. A numerical method
  may change, but it must not skip extraction, acceleration, focusing, apertures,
  or other already modelled gun/column operations.
- For the FEG, custom electron-source inputs belong only to emission from the
  tip (current/brightness, spatial/angular distribution and energy spread).
  Do not introduce an independently configurable accelerated, gun-exit,
  specimen-plane or other downstream source.
- The user explicitly permits adding new FEG tip parameters, including a
  physically defined coherence/phase model. This does not permit defining an
  independently configurable state after extraction or acceleration.
- An equivalent beam state is a cache of an executed upstream calculation,
  not a new source. Its identity must include the consumed upstream optics,
  source, model and numerical inputs. Changing a relevant input invalidates
  reuse. A label, digest binding or manual recalibration is not transport.
- Keep historical results readable without admitting prohibited historical
  exit-source inputs to active calculations. Do not silently convert profiles.
- Missing coherent tip-to-gun-exit physics must remain explicit. Rejecting an
  unsupported request is not completion of that physics or full TEM/STEM
  acceptance. Isolated mathematical fixtures do not qualify the full chain.
- Preserve the full modelled electron/wave state across stages. Users may
  choose optional observables and numerical budgets, but deselecting a
  readout must not remove phase, physical interactions or detector absorption.
- Detector phase readout is explicitly requested. Preserve per-mode complex
  fields and phase references; do not invent one aggregate phase for an
  incoherent mixture or confuse simulated phase with direct hardware readout.
- Dynamic scan and outgoing inelastic waves must use that same tip-origin chain.
  Segment expensive execution and release completed wave buffers. The user
  permits higher configurable cache budgets for the 96 GB host; numerical
  resource choices must not silently remove modelled physical effects.
- Integrate the energy filter last. A detector physically before its entrance
  does not traverse it; a requested path reaching it must not bypass it.

## Current scope and generated data (2026-09-13)

- Coherent tip-to-column wave development is paused at the user's request.
  Do not restart long wave calculations or silently enable a coherent source.
  Keep historical wave code and profiles readable. Current work uses classical
  particle emission with editable physical tip geometry and local emission.
- Curvature radius, cone angle and emitting-cap angle are distinct quantities.
  Derived patch diameter, depth, surface area and arc length are not independent
  downstream source inputs. Preserve extraction, acceleration and apertures.
- Do not commit generated calculation caches or numerical array outputs.
  Keep them locally; retain lightweight reports, source code and input settings.
  Actual microscope acquisition records are not calculation caches and remain
  eligible for version control. Do not rewrite published Git history without
  explicit authorization.

## Vacuum calculation policy (2026-09-14)

- Vacuum scattering / attenuation is opt-in and disabled by default. Users
  normally choose it before the first Preview; later changes remain allowed.
- Changing vacuum participation or active vacuum settings may invalidate all
  calculation stages. This broad cache invalidation is explicitly permitted.
- Preserve explicit on/off choices in saved maps, profiles and snapshots.

## Scientific scope and naming (2026-09-18)

- The target is physically correct mechanisms and qualitative parameter-response
  trends for the simulator's own mechanical structure. Reproducing numerical
  settings or absolute performance of a commercial microscope is not required.
- Instrument records may inform topology, interactions and validation hypotheses.
  Do not import their currents, sensitivities or magnifications as authoritative
  settings for a different geometry. Compare trends only after matching coordinate
  conventions, operating regime and held/fitted controls.
- Use scientific or functional equipment names in the simulator's interface,
  component labels and explanatory text. Do not label simulated components with
  commercial instrument or product names. Name the sensor technology only when
  the implemented model supports that distinction.
- Preserve original acquisition metadata, reference URLs, historical files and
  compatibility identifiers. Present functional labels without rewriting the
  underlying evidence or changing device selection semantics.
- Define the range and controls held fixed for each trend check. Do not assume
  global monotonicity across crossovers, saturation or changes of optical mode.
  Qualitative agreement does not waive unit, conservation or numerical-convergence
  checks, and does not by itself qualify the complete microscope chain.

## Particle performance and continuation (2026-09-19)

- Numerical work may use at most half of the logical CPUs available to the
  process, with at least one worker and respect for explicitly lower limits.
  Apply the budget in calculation workers and prevent nested library pools
  or simultaneous numerical jobs from multiplying that budget.
- Keep user-selected cutoff planes, full-precision executed checkpoints and
  dependency-checked continuation available for classical particle work.
  A saved cutoff is an executed upstream state, never a configurable source.
- Automatically archive each accepted completed particle calculation locally.
  Show the actual calculated endpoint, resumable endpoint, quality, population,
  save time and path. Failed, cancelled or stale work must not be reported as
  a successful current archive. Generated archives remain excluded from Git.

---

# 项目维护与验收规则（2026-09-29 新增）

> 用途：本文件是仓库根目录 `AGENTS.md` 的完整更新版，供 coding agent 执行开发、修复、清理和验收任务时读取。上方英文部分保留原文件全文；以下规则补充维护流程，不撤销既有物理要求，也不授权自动开始全库重构。
>
> 编制基准：`MagNetiCLLLL/tem_simulator_v2`，提交 `37b15119bfc9dad461a43fa3a29900b9f8ab635b`。原 `AGENTS.md` 的 Git blob 为 `a0910c26331e36baa3d580dfb48b46f418de4d3d`。下述文件路径及 scope 名称来自这个版本；每次任务开始必须重新核对实际分支。本文不是测试通过报告。

## 1. 适用范围与工作原则

**目标是保持软件可用、物理要求不变、验证可复现，而不是只让 GitHub 变绿。**

- 遵守运行环境的上层指令和用户当前明确任务。这里的流程不能被解释为用户授权发布、合并、删除数据或改变物理模型。
- 先阅读本文件，再阅读当前目录适用的其他 `AGENTS.md`、相关源码、测试和需求说明；`PROJECT_FUNCTION_SPEC.md`、README 及相关设计文档存在时，阅读与任务相关的部分。
- 自主完成任务范围内的定位、最小修复和验证；不要为已能从源码或日志确定的问题反复请求确认。遇到真正影响物理需求或破坏性操作的歧义，停止该部分并说明，不擅自决定。
- 一次修改解决一个问题或一组紧密相关的问题。不要把故障修复、无关依赖升级、格式化全库、文件迁移和新功能混在一起。
- 不把旧提交、旧报告或文档中的测试数量当作当前版本的验收结果。读取当前 HEAD、工作区差异和当前 CI 记录。
- 新规则以“必须、禁止”表述时是执行约束；“建议、候选”是待评估事项，不是已经实现的功能。

## 2. 每次修改前：记录基线与影响范围

按顺序完成以下工作：

1. 确认仓库根目录、当前分支、HEAD 和工作区状态，使用 `git status --short`、`git branch --show-current`、`git rev-parse HEAD`、`git diff --stat`，并检查相关暂存区差异。
2. 保留用户已有修改和未跟踪文件。不执行 `git reset --hard`、`git clean -fd`，不自动丢弃、覆盖或暂存无关修改。需要对比干净版本时，使用独立 worktree 或副本；不得破坏当前工作区。
3. 写出本次需求、正确行为依据、受影响模块、拟运行的验证范围。区分运行时代码、测试代码、验收工具、配置、历史兼容和生成文件。
4. 先运行受影响的最小检查，记录原有失败、异常、跳过及环境限制；同一问题修复后使用可比的环境和输入复测。无法复现时如实记录，不猜测为“已修复”。
5. 修改涉及当前红灯时，先读具体失败步骤和报告，不仅看任务名称或最终退出码。将问题归类为实现、测试、环境、打包、执行资源或证据收集问题。

当前工作区已有修改时，不将它直接称为“原始提交基线”。报告 HEAD 的同时记录 dirty 状态、相关差异和源码标识；需要归因时进行隔离对比。

## 3. 现有检查链路：优先维护，不重复建设

以下为编制基准中的入口。它们是项目代码的一部分，不是 GitHub 为显微镜制定的外部合格标准。

| 位置 | 职责 | 修改时的注意事项 |
|---|---|---|
| `.github/workflows/tem-p0.yml` | 触发、Windows/Python 环境、任务矩阵、安装包检查及报告上传 | 同步检查事件、命令、路径、状态检查名称和退出码 |
| `requirements/validation-cpu-lock.txt` | CPU 验证使用的依赖版本清单 | 与项目依赖及实际平台共同验证，不盲目升级 |
| `pyproject.toml` | 包依赖、入口、配置资源打包及 pytest 设置 | 修改后检查安装包，不只检查源码目录启动 |
| `scripts/validate_classical_scope.py` | 有限范围运行、线程预算、超时、日志及验收报告 | 保持证据完整性，不能吞掉失败 |
| `src/temsim/acceptance.py` | scope、测试文件清单及通过条件 | 新测试必须进入相应清单；不能删验收项掩盖失败 |
| `src/temsim/acceptance_pytest.py`、`src/temsim/validation_process.py` | 测试收据和有界进程执行 | 改动时验证中断、缺失报告、子进程失败和清理 |
| `scripts/installation_diagnostic_smoke.py` | 隔离 wheel 安装后的有限启动验证 | 不得改成引用源码目录的 editable 安装来“通过” |
| `tests/` | 功能、数值、界面、持久化和验收工具测试 | 以实际需求为依据；测试不是可随意清理的运行垃圾 |

**不要另建一套与现有 scope 和报告格式脱节的“快速验收系统”。** 对现有工具有明确不足时，优先作局部改进并增加回归测试。

### 3.1 按变更选择验证范围

| 变更内容 | 至少评估运行的现有 scope |
|---|---|
| 源准入、核心状态、参数失效、后台任务或计算协调 | `classical` |
| 针尖、静电场、场边界或场缓存 | `gun-fields`，以及受影响的 `classical` / `electron-execution` |
| 虚拟电子轨迹、编译执行、进程、取消或资源回收 | `electron-execution`；涉及界面时加 `field-ui` |
| 场可视化、Hardware tuning、虚拟电子界面或结果交互 | `field-ui`；涉及核心状态时加 `classical` |
| 断点、续算、材料阶段复用、结果文件或归档 | `particle-continuation`，以及受影响的核心范围 |
| 测量工具、基准测试、缓存统计或性能记录 | `performance-observation`，加被修改功能所属 scope |
| scope 选择、收据、运行器、超时或 CI 执行规则 | `acceptance-policy`，加受执行方式影响的范围 |
| 依赖、包入口、配置资源、跨模块重构或大范围清理 | 隔离 wheel 检查，以及全部受影响的活跃 scope；通常需要全部七组 |

这张表用于选择，不是完整依赖图。修改共享组件必须向其调用方扩大验证。只改文档时可进行路径、指令和格式核对；不得把未运行的功能测试写成 PASS。

当前注册的七个活跃 scope 为：`classical`、`gun-fields`、`electron-execution`、`field-ui`、`particle-continuation`、`performance-observation`、`acceptance-policy`。以实际 `ACCEPTANCE_SCOPES` 为准。

### 3.2 新测试的纳入要求

- 新增测试文件后，检查 `src/temsim/acceptance.py` 的清单并将其加入合适的 scope。仅放入 `tests/` 不表示它会被现有分组工作流执行。
- 在合适的已有文件中增加用例时，确认运行报告确实收集到新用例。重命名、迁移和删除文件时，同步修复清单及相关引用。
- 定位时允许运行单个测试或使用过滤条件；正式声明某个 scope 通过时，必须运行它的完整清单，不能把过滤后的结果冒充完整验收。
- 原则上新增功能增加功能与边界测试，修复缺陷增加能捕捉该缺陷的回归测试。可行时证明回归用例在修复前失败、修复后通过。
- 测试预期应来自需求、独立推导或可信参考，不得仅复制当前实现的输出。Mock 可以证明软件控制逻辑，但不能替代真实数值执行或硬件验证。

## 4. CI 失败的处理纪律

### 4.1 根据证据修复

先保存失败的 commit、run/job 标识、步骤、环境、日志和错误信息，再作判断：

| 原因 | 正确处理 |
|---|---|
| 实现不满足未变更的需求 | 修复实现，保留或增强原有验证 |
| 经确认的需求变化使旧预期过时 | 同步更新需求说明、实现和测试；说明旧预期为何不再适用 |
| 测试单位、边界、预期或平台假设错误 | 修复测试并给出独立依据，不能只因实现输出不同就改预期 |
| 依赖、配置资源或独立安装问题 | 修复环境或打包，重新进行隔离安装验证 |
| 进程、线程、计时或报告不完整 | 修复执行及收据链路；保留原始错误和失败语义 |
| 功能暂停、资源不可用或验证未实施 | 明确记录 NOT_RUN / BLOCKED / INCOMPLETE，不声称通过 |

现有安装 smoke 的 `report.json` 可包含 `traceback`；scope 输出目录包含 `report.json` 和带运行标识的 `pytest-*.log`、`pytest-*.xml`。先读这些实际产物；下载不到报告时说明缺少什么证据。

### 4.2 禁止为“变绿”采取的措施

- 删除仍有效的测试、断言、验收标准或工作流；取消必需检查；用 `continue-on-error`、`|| true` 或成功退出码掩盖真实失败。
- 无依据添加 `skip` / `xfail`，通过过滤排除失败测试，或将空收集、超时、取消、缺失收据当作完成。
- 无独立数值依据放宽容差、修改基准值，或降低粒子数、分辨率、路径长度、物理过程，使测试不再检验原要求。
- 用假数据、固定成功响应或绕过真实调用来替代应验证的执行路径；把过去的报告复制为当前结果。
- 修改源码或验收定义的同时继续使用该次运行作为稳定版本的验收。发生变化必须以新源码重新运行。

确需改变容差、预算、测试选择或依赖时，必须记录原因、影响、替代证据及适用范围；变更不得撤销原物理要求。

### 4.3 区分偶发错误与真实回归

CI-only 失败优先比较操作系统、Python、锁定依赖、路径、编码、进程启动、无显示器 Qt、线程预算和随机输入。最多进行有目的的有限重跑并保留每次结果；禁止“反复重试直到绿灯”。性能问题记录预热、冷/热缓存和环境差异，不使用开发机结果替代 CI 实测。

原有且与本次无关的失败可以单独报告，但必须有基线证据；不得把“原本就失败”当作无需调查新增影响的理由，也不得宣称整个项目通过。

## 5. 删除、重构与性能优化的边界

**“GUI 不直接调用”或“搜索不到 import”不能单独作为删除依据。**

- 删除前检查动态导入、反射、字符串注册、Qt 信号/槽、入口命令、配置文件、安装包资源、测试、兼容性读取以及文档中的调用方式。
- 对候选文件或函数记录：职责、调用方、拟删除理由、替代路径、受影响功能及验证证据。无充分证据的候选标为“待确认”，不要直接删除。
- 工作流、测试、锁定依赖和验收脚本属于维护代码；历史波计算模块属于已暂停能力，不是自动可删除项。参考资料、真实采集记录和历史文件兼容代码不能与生成缓存混为一谈。
- 合并重复函数必须比较单位、坐标约定、参数默认值、异常、返回类型、副作用、缓存身份及序列化兼容。保留公共行为；需要兼容转接时使用薄封装，不复制两套实现。
- 大范围清理按模块分批验证，不在同一批同时改算法、删除历史数据和重写文件格式。不要为减少行数牺牲可读性或可追踪性。
- 性能改进必须在相同输入、物理过程、数值精度及资源预算下测量，说明误差、内存和冷/热缓存差异。未经验证的“应该更快”只可作为假设。

## 6. 物理、缓存、持久化与 GUI 回归要求

以下是对上方原始约束的验证补充，不增加未经授权的物理模型。

**物理与数值：** 修改相关算法时核对单位、坐标方向、边界、守恒量及收敛性。趋势验证写明固定参数和有效区间。独立均匀场或解析算例只能支持其覆盖的机制，不能证明完整显微镜链路。维持相干计算暂停状态；不得为验收擅自启动长波计算。

**缓存与续算：** 验证相同有效输入可复用、相关输入变化会失效、无关显示变化不会误改物理状态。保留已执行上游状态及其依赖；缓存不能成为独立可配置的下游电子源。错误、取消、陈旧结果不能标成当前成功计算。

**文件与兼容：** 保存读取应验证状态、数组、单位、版本及来源信息；继续区分只读历史记录和可用于活动计算的有效状态。缺失资源、损坏文件或不支持版本应给出明确错误，不静默套用默认值。不改变既有格式和迁移语义来掩盖失败。

**GUI 与任务生命周期：** 保持耗时计算不阻塞界面；验证取消、关闭窗口、连续编辑、迟到结果及失败恢复。旧任务不得覆盖新状态。打开或浏览控制面板不应意外启动计算或修改物理参数，除非该动作本来就是明确的计算命令。

**验证层级：** 离屏 Qt 测试不等于原生桌面/OpenGL 体验验证；涉及显示、线程、关闭或交互的大改动，应增加实际桌面操作核对，或者明确标为未验证。CPU、模拟 GPU 策略和真实 GPU 科学一致性必须分开报告。

## 7. 环境与独立安装检查

- 默认复现当前 CI 的 Windows + Python 3.12 CPU 路径，实际要求以工作流为准；不要在用户的 base 或其他科研环境中盲目安装、删除或升级依赖。
- 使用专用环境及 `requirements/validation-cpu-lock.txt`，核对 `pyproject.toml` 的依赖要求。不要将不可安装的固定版本默默替换成其他版本；先记录问题再作有依据的调整。
- 非 Windows 环境的结果可以作为补充，不等同于复现 Windows CI。无法运行目标平台或真实 GPU 时如实注明。
- 验收数值计算采用现有脚本的单线程策略。直接执行测试或新子进程也要控制底层线程；应用正常计算仍遵守原文件的“最多一半可用逻辑 CPU”上限及更低的用户限制。
- 不让多个本地数值任务绕过总预算。不擅自改变 GPU 选择/回退策略，不掩盖不应重试的后端错误。
- 安装包相关变更必须构建新 wheel，在新的非 editable 环境安装，使用安装后的解释器及 `-I` 执行 smoke。不得通过注入源码路径或复制未打包配置来绕过安装缺陷。
- 使用全新输出目录，保留原报告和诊断归档；不能反复覆盖同一 smoke 报告。验收输出放在临时位置或明确忽略的目录，提交前核对不会误收录。

编制基准的工作流写死了 `tem_simulator_v2-0.1.0-py3-none-any.whl`。涉及版本或安装流程维护时，应改为从干净构建目录定位本次唯一目标 wheel，并检查包身份，避免旧包或版本号变更造成错误。这是待维护事项，不代表该问题已经导致当前失败或已经修复。

## 8. 可执行命令参考：以当前源码为准

以下为 Windows PowerShell 示例，均在仓库根目录开始。它们是执行参考，不是已完成的验证记录。先选择专用 Python 3.12 环境；未安装或无网络时记录阻碍，不自动修改系统设置。

### 8.1 安装环境并运行所需 scope

```powershell
$ErrorActionPreference = "Stop"
$Repo = (Get-Location).Path
if (-not (Test-Path (Join-Path $Repo "pyproject.toml"))) {
    throw "请先进入项目根目录。"
}
$Python = (Get-Command python -ErrorAction Stop).Source
& $Python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version; print(sys.executable)'
if ($LASTEXITCODE -ne 0) { throw "请选择专用 Python 3.12 环境。" }

& $Python -m pip install -r requirements/validation-cpu-lock.txt -e ".[dev]"
if ($LASTEXITCODE -ne 0) { throw "验证环境安装失败；先处理依赖问题。" }
& $Python -m pip check
if ($LASTEXITCODE -ne 0) { throw "依赖一致性检查失败。" }

$RunRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("temsim-validation-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $RunRoot | Out-Null
Write-Host "本轮证据目录：$RunRoot"

# 根据第 3 节选择。跨模块维护通常应选择全部七个活跃 scope。
$Scopes = @("electron-execution", "classical")
$FailedScopes = @()
foreach ($Scope in $Scopes) {
    $ScopeOutput = Join-Path $RunRoot $Scope
    & $Python scripts/validate_classical_scope.py --scope $Scope --output $ScopeOutput --timeout-seconds 1800
    $Code = $LASTEXITCODE
    if ($Code -ne 0) { $FailedScopes += "$Scope (exit=$Code)" }
}
if ($FailedScopes.Count -gt 0) {
    throw ("以下范围未通过；读取本轮报告：" + ($FailedScopes -join ", "))
}
```

上例保留其他所选 scope 的运行机会，但最终不会吞掉失败。正式复现当前 `cpu-acceptance` 时，另按工作流的原始命令和默认超时执行；上例显式 1800 秒属于本地设置，不能声称与该任务完全相同。单个 scope 通过不能代替其余范围。

临时调试单个测试时，应使用相同专用环境、离屏 Qt 及单线程限制；完成后再运行完整 scope。不要无选择地启动全库 pytest 或旧的完整开发验收脚本，以免触发已暂停计算。

### 8.2 构建并检查全新的独立安装包

以下代码承接上节成功安装环境后的 `$Repo`、`$Python`、`$RunRoot`。即使 scope 失败，也可在记录失败后单独执行安装诊断；不应因此忽略先前失败。

```powershell
$InstallRoot = Join-Path $RunRoot ("installed-" + [guid]::NewGuid().ToString("N"))
$WheelDir = Join-Path $InstallRoot "wheel"
$VenvDir = Join-Path $InstallRoot "venv"
$SmokeOutput = Join-Path $InstallRoot "smoke/report.json"
$SmokeScript = Join-Path $Repo "scripts/installation_diagnostic_smoke.py"
$LockFile = Join-Path $Repo "requirements/validation-cpu-lock.txt"
New-Item -ItemType Directory -Path $WheelDir | Out-Null

# 在仓库中构建；干净的输出目录避免误装上次构建。
& $Python -m pip wheel --no-deps --no-build-isolation --wheel-dir $WheelDir $Repo
if ($LASTEXITCODE -ne 0) { throw "wheel 构建失败。" }
$Wheels = @(Get-ChildItem -Path $WheelDir -Filter "tem_simulator_v2-*.whl" -File)
if ($Wheels.Count -ne 1) { throw "未找到本次唯一的目标 wheel；不得任选旧包。" }

& $Python -m venv $VenvDir
if ($LASTEXITCODE -ne 0) { throw "独立安装环境创建失败。" }
$InstalledPython = Join-Path $VenvDir "Scripts/python.exe"
& $InstalledPython -m pip install -r $LockFile $Wheels[0].FullName
if ($LASTEXITCODE -ne 0) { throw "独立 wheel 安装失败。" }
& $InstalledPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "独立安装环境依赖检查失败。" }

# 在源码目录之外执行；使用脚本的绝对路径，不把源码加入 Python 搜索路径。
Push-Location $InstallRoot
try {
    & $InstalledPython -I $SmokeScript --output $SmokeOutput
    $SmokeCode = $LASTEXITCODE
} finally {
    Pop-Location
}
if ($SmokeCode -ne 0) { throw "安装后检查失败；读取 $SmokeOutput 中的 traceback。" }
Write-Host "安装后检查完成，报告：$SmokeOutput"
```

保留脚本关于模块安装位置、配置路径、粒子来源、保存读取和界面状态的断言。此流程检查有限的安装后功能，不证明完整 TEM/STEM 成像或真实 GPU 正确性。

## 9. CI 配置、状态与验收证据

- 保持核心检查在应触发的 push / pull request 事件中实际执行；改过滤器、条件、矩阵、依赖或任务名时，检查有无遗漏及对应 required checks 是否仍匹配。
- 基准中的 `full-scope-status` 只在 `workflow_dispatch` 执行，调用 `--scope full-report`。它是报告暂停/未完成范围的通道，不运行完整成像验证；普通推送跳过该任务不是缺陷，也不应将它设为要求全系统通过的门槛。
- 不将 `skipped`、`neutral`、`cancelled` 或只有报告上传成功解释为实际测试通过。GitHub 界面的可合并状态不能代替对必要测试及产物的核对。
- 保留失败时的报告上传，确保报告路径与命令一致。不得让汇总任务在依赖任务失败、未运行或收据缺失时错误宣称全通过。
- 若任务要求设置分支保护、ruleset 或必需检查，先核对当前配置、授权及实际工作流名称；没有授权只提出配置建议，不自动修改仓库权限或保护规则。
- 依赖更新、Action 版本更新和工作流改造应作为可复核的小改动，验证兼容性与权限；不要使用不必要的凭据权限，不输出令牌或敏感环境变量。

每轮证据至少包含：commit/dirty 状态、实际命令、平台/Python/依赖、线程与后端、输入或 scope、时间、退出码、测试数及失败/跳过、报告路径。涉及性能或物理结果时增加计算预算、单位、误差、收敛性及独立参考。

采用已有报告字段及语义，不在没有迁移需求时另造不兼容状态：软件 scope 的 PASS 不得升级为完整物理资格；`UNQUALIFIED`、`NOT_RUN`、`BLOCKED`、`INCOMPLETE` 必须准确保留。缺少真实 GPU、原生桌面或实验校准证据时，在最终说明中单独列出。

## 10. 提交、合并与版本维护

- 优先在用户指定分支或独立任务分支工作。已有 agent 管理分支时沿用，不盲目切回 `master`。不得未经授权重写已发布历史、force-push、发布版本或自动合并。
- 只有任务或已授权工作流包含提交/推送时才执行；提交前检查 `git diff --check`、逐文件差异及暂存区。精确暂存任务文件，避免 `git add .` 收入用户数据和无关修改。
- 代码修复、测试更新及必要的说明应形成可理解、可回退的提交。生成缓存、数值数组、归档、环境、完整运行日志和本地绝对路径不得因方便而进入仓库；保留有价值的轻量报告时先审核内容。
- 合并前审阅当前提交对应的 CI，而不是旧提交的绿灯。必需的活跃检查失败或未实际运行时，不宣布 merge-ready。
- 发布前核对版本、安装包、配置资源、入口、兼容性及变更说明。Python 版本声明不等于所有声明版本都已验证；未覆盖的平台和后端应列明。
- 本次无权或无条件提交、推送、运行 CI 时，交付本地变更和明确验证结果；不声称 GitHub 已更新或持续在后台监视。

## 11. 首次采用本文件时的维护顺序

只有用户要求“整体维护”或“修复 CI”时，才将本节作为执行任务；普通功能任务仅遵循前面的相关规则。

**阶段 A：建立现状。** 读取当前 HEAD 和 Actions，确认仍存在的失败任务及具体报告，建立可复现基线。不要把本文件编制日的失败永远当成当前问题。

**阶段 B：最小修复。** 先修活跃范围的真实失败和安装问题，再复测对应 scope，区分历史问题与新增回归。独立安装失败不能用源码目录成功替代。

**阶段 C：防止复发。** 补充针对根因的回归测试、同步 scope 清单；必要时改善 wheel 选择、日志定位或失败产物保存。不要同时扩大物理计算范围。

**阶段 D：有证据地清理。** 在基线可信后，分批移除确认无用的代码，保留维护工具和历史兼容；每批执行影响范围内的验证。

**阶段 E：汇总与交付。** 审阅差异、核对工作区和产物，提交任务允许的变更，报告尚未解决的问题及未验证范围。无法完成整个阶段时交付已完成部分，不伪称全部验收。

## 12. 每次任务的最终交付模板

```text
任务与变更范围：
- 用户要求；实际修改的文件和行为；明确未改的相关物理约束。

基线与根因：
- HEAD、工作区状态、原有失败；根因及证据；无法确认的部分。

实际验证：
- 环境、命令、scope/用例、退出码、通过/失败/跳过数量、报告路径。
- 隔离 wheel 检查是否执行；真实 GPU/桌面/物理参考是否执行。

兼容性与风险：
- 安装、配置、缓存、保存读取、续算、GUI、数值和资源预算影响。
- 仍未解决的问题及 NOT_RUN / BLOCKED / INCOMPLETE 范围。

仓库状态：
- 修改未提交 / 已提交及 SHA / 已推送 / PR 与当前 CI 状态，按事实填写。
- 不把“建议运行”“预计通过”写成“已运行”“已通过”。
```

### 完成检查

- [ ] 原有物理链路、暂停范围、CPU 上限和用户数据均已保留。
- [ ] 基线与修改后结果可以区分；没有用旧报告证明新代码。
- [ ] 测试或验收变更有依据；新用例确实进入应运行的范围。
- [ ] 相应活跃检查及必要安装检查已完成，或明确报告未完成原因。
- [ ] 没有通过删测试、吞错误、无依据放宽容差或绕过物理过程消除失败。
- [ ] 源码、配置、打包、兼容性和相关文档保持一致；无无关大范围改动。
- [ ] 提交与最终说明没有夹带生成数据、凭据或未经核实的通过声明。

## 13. 编制依据与后续更新

本文件的原始要求来自开头标明的 `AGENTS.md` Git blob；检查链路以同一基准提交中的 `.github/workflows/tem-p0.yml`、`scripts/validate_classical_scope.py`、`scripts/installation_diagnostic_smoke.py`、`src/temsim/acceptance.py` 和 `requirements/validation-cpu-lock.txt` 为依据。后续路径、scope、环境或流程改变时，同步维护相关段落，避免继续执行过期命令。

GitHub 状态语义参考官方文档《Status checks》：`https://docs.github.com/en/pull-requests/reference/status-checks`。pytest 退出码参考官方文档《Exit codes》：`https://doc.pytest.org/en/latest/reference/exit-codes.html`。实际执行仍以项目锁定版本和本轮原始日志为准。

不要因为 agent 遇到难以通过的检查而自动修改本文件的要求。需要改变既定物理目标、暂停决定、破坏性操作权限或验收范围时，明确记录用户授权和相应影响。
