# BGFM：按 ICLR 标准的研究与实现审计

审计日期：2026-09-08。源快照：`e00597bc461348026ef81ad768f1052bf16801c6`。
这是代码、理论条件和原始结果记录的审计，不是对所有历史作业、原始数据来源或全部参考文献的认证。

## 判断

**现有课题尚未达到我愿意推荐提交 ICLR 的程度。问题在方法与证据，而不只是表达。**
确实存在一个可重算的局部排序现象，但现在不能把它说成“生成器学到了 Boltzmann 密度”，
也不能说它改善了生成分子。旧稿的问题不能靠强调正面结果、隐藏积分精度实验或把接口错误改称
“某种 readout 约定”解决。

有价值的基础是：多种子实验记录、真实/零值/打乱能量对照、独立 xTB 评价、显式失败记录，
以及可保留的 bond-free OMol25 + FlowMol3 工程骨架。下一步应在这个基础上做一个更小但可成立的实验。

## 必须先解决的问题

### P0-1：能量积分器使用了错误的向量场

`cfm_mol/bgfm_density.py::make_position_velocity_fn` 直接返回
`model.vector_field(...)["x"]`，反向状态更新也使用这个返回值。
实际依赖 `flowmol/models/vector_field.py::EndpointVectorField.forward` 明确预测终点坐标 D；
采样器在另一个 helper 中计算 `alpha'/(1-alpha) * (D-x)` 才得到速度。
对 CTMC 输入，旧积分器因此没有积分它声称的 positional flow。

一个直接反例：若 D(x,t)=x，正确速度恒为 0，密度应保持先验；旧代码却积分 dx/dt=x，产生膨胀流。
已有 58 项测试大多提供真正的速度 mock，因此全绿并未检查到生产接口的语义。
新增测试直接实例化 CTMCVectorField，确认输出不同于其应转换出的速度，并验证修正路径可反向传播。

### P0-2：质心维度与散度不一致

旧 prior 使用 3(N-1) 维归一化，但 head 默认 `remove_com=False`，被微分的场也没有投影。
平移等变的坐标预测满足 D(x+u)=D(x)+u，平移方向的导数是 1，并不满足附录假设的 J u=0。
数值坐标已经居中不能消除微分中的平移方向，反向轨迹还可能离开零质心子空间。
仅仅把 prior 常数减 3 维，无法证明这是内禀归一化密度。

修正路径实现 `P v(Px,t)`，对状态和输出均按 graph 去质心，并在该场内计算导数。
COM-free 解析高斯测试同时检查了归一化、3(N-1) 散度和积分误差阶。

### P0-3：梯度不是所显示损失的完整梯度

每个旧反向步骤都 detach，状态更新处于 no_grad，先验在参数方向也是常数。
实际梯度只经过散度。对散度恒为零、但会改变密度的参数化平移场，它会漏掉全部学习信号。
新路径保留轨迹和 prior 的梯度，并以固定噪声有限差分核对；另外保留旧路径用于追溯结果。

真实训练还启用了随机 self-conditioning，旧代码对同一时间点的散度和状态更新分别前向，
两次调用可能使用不同的随机分支。新密度定义暂时切到确定性 eval 行为而不禁用 autograd，
执行完恢复每个子模块原来的 training 标志。

**修正后的 q_T 仍然不是联合 CTMC + 历史自条件 + steric retraction 采样器的密度。**
这是需要明确建模的对象差异，不能用“discrete endpoints clamped”一句话解决。

### P0-4：现有主要结果对求解精度敏感

完整重算见 [evidence.md](evidence.md)、[evidence.json](evidence.json) 和 [图](evidence_figure.pdf)。
脚本核对 `(group_id, pert_id)` 和存储的原 readout，再接能量；仅比较三档都存在的 checkpoint。
所有档位使用相同的旧 93-parent mask，移除未扰动参考结构。主表中的 5/4 个种子与本表 5/3 个种子
是不同可用集合，不混在一起比较。

| 旧积分步数 | FM，5 seeds | Value，3 seeds | Scrambled，3 seeds | Value − FM |
|---|---:|---:|---:|---:|
| 12 | 0.2221 ± 0.0091 | 0.3972 ± 0.0161 | 0.2216 ± 0.0151 | +0.1751 |
| 24 | 0.1973 ± 0.0156 | 0.2745 ± 0.0023 | 0.2032 ± 0.0173 | +0.0772 |
| 48 | 0.1989 ± 0.0111 | 0.2413 ± 0.0034 | 0.1907 ± 0.0087 | +0.0424 |

误差是训练种子间 SEM。残余差异仍为正；不能说“完全消失”，但只剩原效应约 24%。
同时不能称 48 步为真值：未建立收敛，也没有完成多组 probe repeats。
中点时间没有截断积分区间；它也没有把 start-state Euler 积分变成二阶中点法。
旧解析测试只检查常/线性时间散度，不能证明一般状态依赖散度的二阶误差。

### P0-5：生成改善与校准没有得到支持

旧 93-parent 主端点的 r=0.2000→0.3693、top-1=20.2%→29.6% 可重算，
但每 seed 的组内 NRV 中位数再平均为 1.4017→2.1441；常数 readout 基准为 1。
相关性提升不能替代幅度校准。

另有 11 seeds × 500 个生成结构的 xTB 实验。每 seed 的单位原子能量下降中位数再平均：
FM 5.43±0.59、value 5.68±0.81、scrambled 5.51±0.65 kcal/mol/atom，未提供改善证据。
13.0%、11.1%、12.9% 是数值失败率；若以真正的优化收敛标志计，未收敛比例达到
43.3%、42.6%、47.6%。旧脚本把“有最终能量”记作成功，这不等于找到了局部极小值。
新汇总分别报告有限能量和收敛计数；既不删除不利样本，也不由相似失败率推断不存在选择偏差。

### P1-1：力对照并没有检验一般的 force/score matching

旧力臂默认直接把 D 填进 velocity-score 公式；它是正确 endpoint readout 的 (1-t) 倍，
而多个 t、Huber 化与 norm cap 使单一 lambda 不能完整抵消此问题。
`force_endpoint_to_velocity: true` 已存在，但旧发表数字不是这样产生的。
更深的条件也没满足：Gaussian prior 与 endpoint 必须独立；生产配置有 alignment、路径有 retraction；
有限 t 的平滑分布 score 不一般等于 endpoint force。即便修好尺度，负面结果也不能推广为“力不如能量”。

### P1-2：理论应限定于理想 residual，不能证成当前训练

1. FM 的数据最优密度不一般等于 Boltzmann 密度；三个目标不会仅凭无限容量就相容。
2. 有限权重/无限权重都不自动保证存在零能量残差的可行解。
3. 有限点的零方差只给有限点约束，不给区域、全局或 basin mass 的一致性。
4. 分组 residual 的 offset 自由度是在不受限可测函数类中的可识别性问题。
   光滑模型或接触支撑的连续性可能额外耦合常数；“不连通因此任意光滑模型都有任意 mass”过强。
5. Boltzmann 分配函数还需要有限积分和明确的状态域。去平移不会约束相互解离的片段。
6. `NRV_min=1-r^2` 是允许有符号缩放的结论。只许正温度缩放且 r≤0 时，下确界为 1。

已补一份审计后的推导 `notes/bgfm_method.md`，原 April 文件保存于 `notes/archive/`。
新增的有限点 graph-Laplacian spectral-gap 界给出可测的 bridge-group 设计条件，
但明确是标准线性代数推论，不包装成新颖性，也不声称连续 basin 概率保证。

### P1-3：数据与实验协议仍需重建

- 旧主实验 energy shard 含 validation 部分。93-parent 修正依赖历史 group mapping，
  本次使用了这份 mask，并没有重新证明结构/身份层面的严格不交叠。
- 划分是在接受的结构上随机进行，未建立 parent/chemical-identity 级隔离；构象或相近结构泄漏仍需核查。
- charge 被放到第一个原子且截断 [-2,3]，spin 没有完整保留。不能由数据集有 83 个元素推出模型获得
  83 元素、任意电荷/自旋的热力学能力。
- `p0_Dfix` 中检查到部分配置又加载 val shard；不能因为文件名带 fix 就当作 clean matched grid。
- 仅约 0.12 epoch 的训练和 objective-dependent failure 使训练充分性与幸存者效应未分离。

### P1-4：相关工作与贡献边界

本次核查的关键原始来源：

- [FlowMol3](https://arxiv.org/abs/2508.12629)：endpoint parameterisation、alignment 和 self-conditioning
  都必须按实际接口处理，不能当普通 velocity network。
- [LDR, 2026](https://arxiv.org/abs/2602.03729)：已明确研究 off-policy、energy-labelled、与数据训练组合的
  log-dispersion regularisation。原稿漏掉了这个直接近邻；已补入正文和参考文献。
- [Adjoint Sampling, ICML 2025](https://proceedings.mlr.press/v267/havens25a.html)：已用神经能量模型做跨系统
  amortised conformer sampling。宽化学空间本身不足以替代方法比较。
- [OMol25](https://arxiv.org/abs/2505.08762)：多样的电荷、自旋和反应结构不是单一温度的平衡样本。

目前真正可辩护的问题是：“给一个 bond-free de novo generator 加分组物理值约束，
能否在一个定义正确、数值可控的密度/样本评价上带来改善？”本次没有把这个问题写成已解答。

## 本次已经完成的改动与验证

- 新 `clamped_density.py`：实际 schedule 的 endpoint→velocity，COM 微分投影，确定性执行，
  state/divergence 中点积分，完整轨迹与 prior 梯度，明确 terminal_time 和 density 对象。
- 训练 hook 可显式选择新路径；旧路径为复现实验保留并警告。主 batch 缺 force labels 不再阻断独立 energy shards。
- 修复 exact divergence 最后一个坐标漏参数梯度、常数场求导异常、全无效/单例分组以 NaN×0 生成 NaN 的边界情况。
- 增加只依赖标准库的证据重算脚本、输入 SHA256、每 seed 指标、科学绘图与收敛计数。
- 74 项测试通过；新增检查包括真实 CTMC 的 forward/backward、解析 Gaussian density、二阶收敛、
  有限差分参数梯度、只存在 prior 路径的参数、缺力标签时 energy hook 仍执行。
- 对真实 FM checkpoint 的一个 9 原子 parent 和三个固定几何，CPU 运行了新 q_0.95 的 8/16/32/64/128 步积分。
  结果有限，见 `checkpoint_smoke.json`；64→128 步的最大 centered logq 变化约 0.087。
  同一 checkpoint 的 2 步完整反传产生 420 组有限参数梯度，未执行 optimizer update。
  **这些只证明运行与梯度可计算，没有建立分子性能或跨分子数值收敛。**
- 原稿编译正文 10 页、零 LaTeX error，但旧 build 返回成功。工作稿重写了摘要/方法/结论，恢复求解精度和生成反证，
  并区分旧 scalar 与理想 density；调整后正文达到 9 页。新 build 超页或 unresolved refs 会返回非零。
  另用未改动的旧稿运行新 build：识别到 10/9 页并返回 exit 1，负向门禁已实测。
- 未运行新的分子训练，未提交 SLURM 作业，未改已有 checkpoint、共享环境或结果记录。

## 我愿意接受的下一轮研究方案

### 第一道关：定义与数值正确性

先固定究竟评价哪个采样器的密度。若选择 q_T，就必须提供与 q_T 对应的确定性 clamped sampler，
并明确条件生成的边界；若坚持 de novo 联合采样主张，就用独立生成结构指标作为主终点，
不能继续把 q_T 当作其联合条件 likelihood。

在固定 checkpoint、固定 parent/geometry 上做 16/32/64/128 步和 trace-repeat/exact-divergence 检验，
比较 **组内中心化值与 pairwise log ratios**，不只看每分子的总 offset。
浮点精度、terminal time、完整/截断梯度作为明确因素。无偏 trace 不代表平方后的 energy variance 无偏；
还必须量化随机 trace 为损失增加的噪声惩罚。事先定义数值误差应远小于欲检测的臂间效应。
若还不稳定，先解决这个问题，不扩大数据或重复旧配置。

### 第二道关：真正匹配且可追溯的实验

用 train-only shards 和一个从未用于选择模型/指标的 test 集；保留原始 molecule/parent ID、total charge、spin、
coordinate hash 与 source split。训练、验证、测试按分子身份分组，事先检查交叠。
在同一个充分训练的 FM 起点上固定至少五个 seed，比较 FM、zero-value、scrambled-value、true-value；
另加定义正确的 force comparator 和 full-vs-frozen gradient 对照。
所有臂保持 batch、步数、oracle预算和 solver 一致，报告总算力；失败不补种子、不用后发成功 run 替换。

### 第三道关：与论文主张对应的终点

预先确定 raw NRV/斜率与独立结构质量指标，排序只能是其中一项。
生成评价至少报告全部样本的有效性、收敛率、计算失败、单位原子 strain 的 mean/median/tail，
同时分层控制 size/composition/charge；不能只看可优化的幸存样本。
训练 seed 是算法复制单位，parent 是另一层随机性；不要把 93×8 个点当作独立训练重复。
如果要 basin mass，加入真实桥接组/参考系综并测试 occupancy 或权重比；目前没有这些证据。

### 投稿决定

在以上关卡之前，我建议暂停“可靠 Boltzmann generator”这个投稿结论，而不是暂停代码工作。
若新实验只恢复局部 readout 排序、没有独立采样收益，应收窄为诚实的方法诊断研究；
是否足够 ICLR 还取决于方法洞见、近邻基线和外部复现，不能由页数或更多语言润色补足。

[ICLR 2027 官方作者指南](https://iclr.cc/Conferences/2027/AuthorGuidelines) 当前列出摘要截止
2026-09-18、全文截止 2026-09-25（AoE），投稿正文最多 9 页。
此次修订满足页面预算不等于满足科学标准；也未执行任何投稿或对外发布。
