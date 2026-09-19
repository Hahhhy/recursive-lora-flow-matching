# Scale-RAE Recursive LoRA：第一轮可判别实验

目标不是立刻追最好指标，而是用最少 GPU 时间回答：指标下降来自 LoRA 插入点、
循环粒度，还是训练/推理深度不匹配。

## 必须分开的实验变量

- `lora_target`: `attention_output`（`attn.proj`）或 `conditioning`
  （`adaLN_modulation.1`，以真实模型审计结果为准）。
- `loop_granularity`: `layerwise`（每个 block 独立循环）或 `rangewise`
  （blocks 12--27 作为一个完整函数循环）。
- `train_k` 与 `infer_k`: 不得只写一个 `K`。
- 其他因素固定：同一 checkpoint、数据、seed、blocks 12--27、rank 8、
  `alpha=8`、总步长 `lambda_total=1`、优化器和训练步数。

“AdaLN-layerwise”不是充分的实验名称，它至少要展开成：LoRA 是否插在 AdaLN、
哪些精确 Linear 路径、哪些 blocks、训练 K、推理 K、循环算子和总步长。

## Gate 0：不训练（已有结果 + 正式模型待补）

| ID | LoRA | loop | 用途 |
|---|---|---|---|
| B0 | 无 | K=1 | 原模型基线 |
| B1 | 无 | layerwise K=4 | training-free loop 基线 |
| B2 | 无 | rangewise K=4 | 整段循环基线 |

## Gate 1：单批反向传播（先做，不算论文结果）

每个候选仅运行一个 batch，要求：loss 有限、只有 LoRA 可训练、LoRA 梯度有限、
显存不溢出、block 调用数符合定义、W&B offline 日志包含完整配置。

| ID | target | granularity | train K |
|---|---|---|---:|
| S1 | attn.proj | layerwise | 4 |
| S2 | attn.proj | rangewise | 4 |
| S3 | AdaLN Linear | layerwise | 4 |
| S4 | AdaLN Linear | rangewise | 4 |

如果显存不足，先将 blocks 缩到 12--15 验证机制；这叫工程 smoke test，不能与
正式 blocks 12--27 的结果混报。

## Gate 2：短程过拟合/趋势检查

Gate 1 全部通过后，每个设置使用同一小数据子集、同一 seed，先跑 100--500 steps。
记录 raw loss 与 smoothed loss、grad norm、learning rate、显存、吞吐量和样例图。
若某设置 loss 不下降或生成明显崩坏，立即停，不进入完整训练。

## Gate 3：最小训练/推理深度矩阵

对 Gate 2 中最可靠的 target × granularity，运行：

| train K | infer K | 解释 |
|---:|---:|---|
| 1 | 1 | ordinary LoRA |
| 1 | 4 | 推理深度错配 |
| 4 | 1 | 去掉训练时递归后的退化 |
| 4 | 4 | Recursive LoRA 主设置 |

只有在这四格都固定相同 checkpoint/data/steps/seed 时，才能判断 Recursive LoRA
是否解决 inference loop 的 train-test mismatch。

## 向同学索取的最小证据

1. 完整命令或 config；2. LoRA 精确模块路径；3. trainable parameter 列表与数量；
4. train/infer K；5. layerwise/rangewise 的 forward 定义；6. checkpoint；
7. 指标名称、方向、评估脚本版本和 seed。没有这些信息，其下降结果只能作为风险提示，
不能作为可复现实验结论。

## 版本身份

- RAEv2 代理实验代码：DiffusionBench Git commit
  `4848f0c9991fbe13b33a7ed458edeeba1b8884e8`。
- RAEv2 权重 revision：`a4a0f5c22e70f7f253f47a5250b478ae5084acd7`；
  它是 Hugging Face revision，不是上述 GitHub commit。
- 正式实验必须另行记录 Scale-RAE、loop 方法代码和训练代码的 commit。

