# Figure1 重绘记录

本轮专门落实 round2 的方法图建议，不改模型、数据或统计结果。

- 上半幅让真实初态和生成分子成为视觉主体，减少模块框；保持相同相机与比例。
- 比较 outline、soft、polished、sculpted、balanced 五种 PyMOL 渲染，采用中等柔和阴影的 balanced。
- 补回 FM/扩散的共同终点接口；几何头先给出 g 和 H_g，物理头再给出 f，两者形成 ΔH。
- 单独标明离线学习：扰动参考监督几何头，初始模型终点上的 eSEN 力监督物理头。
- 两头参数合计14,853；采样不调用能量模型。
- 保留 C7H15N 原始输出及二维推断连接图。全部23个原子、键级及坐标核验通过。
- 画布5.5×4.85英寸，最小标注8.1pt。检查独立彩色图、灰度、论文第3页及第9页。
- 本地与独立Overleaf源均编译通过，9页主文、50页总计、33项引用。

入口：research/figures/figure1_redesign_v1/figure1.png，figure1.pdf，figure1.svg。
脚本：scripts/research/compose_figure1_redesign.py。
科学与视觉核验：research/evidence/figure1_redesign_verification_v1.json。
发布状态：publication_build_v32.json / publication_sync_v32.json。
