# Round2 建议处理记录（2026-09-24）

本轮读取了 `suggestions/round2` 的全部 16 个文件。原件保存在
`runs/round2_20260924/input_archive`，文件清单及 SHA256 公开在
`research/evidence/round2_suggestion_inventory_v1.json`。截图和带个人信息的原件不随代码发布。

## 采用的写作修改

- 以 Overleaf 的合作者提交 `2bfca98` 为最新来源，保留标题
  **LatentSpring: Harmonic Sources and Physical Corrections for Molecular Generation**。
  ZIP 是更早版本，没有用 ZIP 覆盖后来的协作修改。
- 按微信意见把 introduction 末尾的章节导览改为三条完整贡献：组成条件下的谐振源、
  从 endpoint forces 学习的矫正、生成质量与跨目标权重迁移的实验。
- 保留合作者采用的正式学术叙述，删除重复解释；直接给出设计、定义和结果。
  把 “normalize coordinates” 改为准确的“减去平均位置”：没有缩放 Å 坐标。
  把 FM 的 “denoising network” 改为 velocity network，修正输入符号的排版错误。
- 四个实验 subsection 仍然是 setup、四方法主比较、FM 组件分析、迁移。
  主比较继续展示 Gaussian FM、EDM、GAGA、LatentSpring。GAGA 自己训练的 head
  作为迁移图中的 own-head 对照，避免与未经矫正的主比较混成一个设定。

## 采用的图形设计

- 方法图区分源分布与生成、离线 force-target 学习、FM/扩散模型的 endpoint 接口。
  输入写明组成、电荷和自旋；输出明确为原始三维坐标，并配二维连接图。
- 主比较用 graph validity 与 joint yield 的配对点和连接线显示质量差距，
  旁边给出基于全部存档力分数的阈值曲线。没有用平滑插值制造趋势。
- 组件图展示四个组合，并单独画出 harmonic source 的增益和实际 crossed bootstrap 区间。
  原先两个训练重复均保留，source-only 的不确定性也保留。
- 迁移图展示五个训练对的 parent、own head、transferred head。
  早期两对与新增三对分开编码，两个迁移方向使用同一尺度。
- 增大面板之间、图例与轴标题之间的间距；矢量文字与真实分子渲染分开保存。
  在独立图片、论文实际尺寸和灰度版本检查。未靠缩小标签凑页数。

## 化学结构与数据

旧图的问题包含真实几何问题。旧方法例子不通过键长和内部距离检查；旧画廊四个
例子中三个不通过新增的几何筛查。保留原始坐标和全部结果，论文以简短结果说明
替代重复的旧画廊。旧动画展示的是历史 paired-update 变体，保留在仓库归档，
不再作为当前主模型的 Overleaf 附件。

对主实验全部 20,480 个原始输出和 512 个准备好的验证集参考结构进行了检查。
采用 PoseBusters 0.4.4 的几何子集，加上零指定自由基电子要求；没有采用它的
参考身份比较或构象能量比，因此不称为完整的 PoseBusters-valid 指标。

| 设置 | 图有效率 | 几何子集通过率 | 几何子集且力阈值通过率 |
|---|---:|---:|---:|
| FM parent | 20.12% | 3.48% | 1.64% |
| FM + correction | 24.86% | 4.61% | 4.43% |
| GAGA parent | 22.60% | 4.71% | 2.40% |
| GAGA + correction | 26.25% | 6.64% | 6.45% |

每行分母均为 5,120。图推断失败仍计入分母。闭壳层条件和几何检查的定义、
逐输出记录、参考记录及失败类别均公开。原主实验的 graph-plus-force 指标不变。

主图另用了四个验证集组成、每个 32 个输出的独立展示面板。组成按参考结构和
原始行顺序预先选定；全部 128 个输出保存并评分，其中 79 个图有效、15 个通过几何
子集。展示样本是条件 2、输出 27（零起始），C9H9NO2，连接图
`N=Cc1ccccc1C1OC1O`，力 RMS 2.886 eV/Å。它通过所声明检查且无指定形式电荷或自由基。
最初排除全部三、四元环的展示筛选没有合格输出；展示选择随后允许饱和环氧环，
此过程已在附录记录。没有删去任何生成结果，也没有改变 benchmark 的率。

源坐标由原始 RNG 精确重放，输出直接读取存档，二者使用相同相机和比例。
RDKit 二维图只说明推断连接，不改变三维坐标。PyMOL 的芳香键 1.5 取整错误已修复，
用芳香环往返检查及实际场景的键阶、原子、坐标核对验证。主图采用明确 Kekulé 表示。
建议中的 AI 生成分子图只用于理解布局，没有作为科学数据或结构图使用。

## 已完成的联合迁移实验

原来等待中的作业 47380645 已完成。两种设计一起迁移到重新训练的扩散模型后，
EDM joint yield 为 0.00%，GAGA 为 0.05%；对应两次训练的原始对照是 8.15% 和 9.47%。
论文和状态文件已填入结果，不再留下 pending。完整设定和成本在附录。
已验证为正向的 frozen-head 迁移是另一项实验，其结果保持不变。

## 未采纳为已完成结论的建议

- 没有把单次网络评估父模型的四组合研究写成最终 self-conditioned 模型的完整重训练消融。
- 没有把“图有效且力较低”改称为化学稳定、可合成或达到 Boltzmann 分布。
- 没有宣称两种设计能够联合改善所有 baseline；实测联合扩散适配没有成功。
- 没有为作图优化坐标、替换为参考分子，或对主实验筛选分子后重新计算成功率。

下一项算法工作应针对重原子连接和局部平面性，并在冻结的主评估上验证。
本轮完成了可复核的化学诊断、渲染修复和稿件修改，未把它们包装成新权重带来的性能提升。

## 文件入口

- 论文：`paper/tree_working.tex`、`paper/latentspring.pdf`
- 新主图：`research/figures/round2_method_v1`、`research/figures/round2_results_v4`
- 化学检查：`cfm_mol/chemical_geometry_review.py`、`scripts/research/audit_round2_chemistry.py`
- 联合迁移：`research/evidence/joint_design_transfer_audit_v1.json`
- 化学诊断：`research/evidence/round2_chemical_geometry_audit_v1.json`
- 新展示面板：`research/evidence/round2_illustration_protocol_v1.json`
- 编译与发布：`research/evidence/publication_build_v28.json`、`publication_sync_v28.json`
