# ICLR2027匿名投稿格式核对

以朋友的Overleaf提交f09901e为基底，保留其新增引用、训练/推理解释和附录换页。
官方2027样式包已下载并核对，四个实际编译样式文件逐字节一致。
删除内部draft页眉覆盖，恢复匿名review模式。附录从参考文献之后第12页另起。
AI/reproducibility采用官方示例的无编号subsection级标题；符号表caption移到表上方。
新增引用保留并使用natbib括号格式；OpenReview纯文本摘要去除无法解析的cite宏。
科学表述仅修正元素编码/固定Q0S1，以及完整模型在推理时使用校正速度。

AI声明按实际辅助范围简写，未写成“仅语言润色”。官方政策明确要求披露方法实施、
研究/实验设计、结果解释等用途，不能用删掉GitHub自动共同署名替代准确披露。

仓库名改为LatentSpring。默认分支清理21条Claude自动共同署名；562个commit的
文件树及原作者/提交者/时间戳均保持不变，原始出处由provenance标签与提交映射保留。
只改变共同署名元数据，不删除实验记录、不重写过去的协议哈希。

最终构建v35为8页主文/45页总计/33项引用，本地和独立Overleaf源的PDF文本一致。
格式记录iclr2027_submission_format_v1.json；同步记录publication_sync_v35.json。
正式上传文件为paper/latentspring.pdf。尚未代用户上传OpenReview。
