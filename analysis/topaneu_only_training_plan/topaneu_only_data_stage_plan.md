# TopAneu-only 分阶段训练方案（证据快照）

本方案严格限定只使用当前 TopAneu release 的 416 个序列及其影像、52 类位置 mask、3 类动脉瘤 type mask、36 类组织方血管银标准 mask。开发阶段不使用外部病例或外部任务权重。

## 数据与切分约束

- 416 个序列中 305 个阳性、111 个阴性；52 个最终位置类中 9 类全局零样本，24 类不超过 3 个阳性病例。
- 现有五折为 332/84、333/83、333/83、333/83、333/83；各折验证集阳性/阴性分别为 61/23、61/22、61/22、61/22、61/22。
- center4 的 7 组纵向复查均完整落在同一验证折，没有患者级泄漏。
- 每个训练折仍有 10–12 个最终位置类别零样本、25–28 类不超过 3 个样本。因此最终 52 类不能作为普通独立 softmax/BCE 的唯一监督。
- OOF 开发时，每一阶段只能使用该折 train 部分训练；不能先在全部 416 例上训练 vessel teacher 再评价五折。模型选型完成后才允许用全部 416 例训练最终提交模型。

## 阶段设计

0. 全图 ROI 定位：从原始 brain-cropped 影像预测 36 类血管或血管 union，以预测血管包围盒加物理 margin 生成 160³ ROI。当前由 GT 派生 ROI 只能用于上限实验，不能代表端到端成绩。
1. 独立血管 teacher：影像到 36 类血管分割。使用全部 fold-train 病例的银标准 vessel mask；禁用镜像，保留轻度旋转、缩放与强度增强。先用 Dice+CE，按稀有血管类别采样；union-vessel clDice 只作为后续低权重消融。
2. 二值瘤 proposal：初始化 Stage 1 encoder，预测 merged aneurysm mask；52 类 mask 合并成二值目标，3 类 type mask作为低权重辅助监督。正负病例平衡，并强化含瘤 patch/小病灶采样。主目标是提高 lesion recall，同时控制纯空间 FP。
3. component 层级位置分类：先用 GT aneurysm component 训练，输入局部 encoder feature、坐标、side、与 36 类血管的距离/接触/中心线关系；预测 territory、side、parent vessel、trunk/junction/bifurcation/distal 属性，再通过固定表映射为 52 类。最终 52-way head 只作 masked residual，不承担全部学习压力。
4. 预测输入替换：依次评估 GT component+GT/silver vessel、GT component+pred vessel、pred component+GT/silver vessel、pred component+pred vessel，定位误差来源。只有前三项上限达标才进入局部精修。
5. 高分辨率局部 refiner：围绕预测 component 在原始分辨率裁 32–48 mm 物理视野，输入影像、粗瘤概率和母血管概率，细化 binary boundary。训练中心混合 GT 抖动中心、预测真阳性中心和假阳性中心，避免只在完美中心上训练。
6. 端到端校准：还原原始空间并直接调用官方 task2 evaluator；阈值和 component 过滤参数只能在训练折/其他 OOF 折选择，不能在被报告的验证折上调参。

## 不可回避的限制

只用当前数据时，9 个全局零样本最终类别没有任何正类监督，神经网络不可能学会这些类别。可通过可组合的 side/parent-vessel/subtype 属性和确定性解剖映射产生候选，但效果无法在本地验证。这是数据上限，不是换 loss 能解决的问题。

