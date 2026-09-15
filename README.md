# Fixed-depth Recursive LoRA v0

这个目录是开发线的最小原型，不是论文结果，也暂不直接修改 Scale-RAE 官方代码。

## v0 要回答的问题

在冻结的 Transformer block 中给选定的 `nn.Linear` 加 LoRA，并让同一组 LoRA
参数在每次 inner loop 中共享，能否：

1. 保持原模型初始行为（LoRA 的 B 矩阵零初始化）；
2. 只更新 LoRA 参数，不更新 backbone；
3. 在固定 `K=4` 的 Euler 递归路径上正常反向传播；
4. 保存、加载后得到完全相同的 LoRA 输出；
5. 用同一实现支持公平对照：`K=1` 普通 LoRA，`K=4` Recursive LoRA。

## 数学定义

对一个冻结线性层 `W`，使用

```text
Linear_LoRA(x) = W x + (alpha / rank) B A x
```

其中只训练 `A, B`。若被适配后的 block 记为 `F_phi`，固定深度递归为

```text
h_0 = h
h_{j+1} = h_j + (lambda_total / K) * (F_phi(h_j, cond) - h_j)
j = 0, ..., K-1
```

注意：这里共享的是 LoRA 参数 `phi`，不是为每个循环分别复制一套参数。原方法配置
中的 `lambda_value=1.0` 是总强度，实际每轮步长为 `1/K`。

## v0 的实验身份

| 设置 | 训练时 K | 推理时 K | 用途 |
|---|---:|---:|---|
| frozen B0 | 1 | 1 | 原模型 |
| training-free loop | 未训练 | 4 | 原论文方法 |
| ordinary LoRA | 1 | 1 | 排除“一般微调”收益 |
| mismatch control | 1 | 4 | 直接观察训练/推理深度错配 |
| Recursive LoRA | 4 | 4 | v0 主方法 |

## 当前边界

- 原型只实现 Dense、layer-wise Euler loop；Sparse 和 Loop Guidance 属于推理
  baseline，不是 v0 训练机制。
- `target_paths` 只是接口；正式应适配 Scale-RAE 的真实模块名后再锁定。
- 候选优先级是 attention output projection，其次 QKV，再其次 MLP；不能在未审计
  Scale-RAE 官方模型前宣称最终插入点。
- loss 仍应复用 Scale-RAE 原生 flow-matching 训练目标；本目录不虚构数据或 loss。

## 在有 PyTorch 的环境运行

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

若 Ubuntu/WSL 缺少 `python3-venv`，可安装系统的 venv 组件，或使用
[PyPA virtualenv.pyz](https://bootstrap.pypa.io/virtualenv.pyz) 在本地创建 `.venv`。

测试通过只表示参数共享、冻结、梯度与 checkpoint 机制正确，不能表示生成质量提高。

可用 toy training 检查优化器、递归反向传播和 LoRA-only checkpoint 的完整闭环：

```bash
.venv/bin/python toy_train.py --num-loops 4 --steps 100
```

该 loss 是合成的回归目标，不是 Scale-RAE flow-matching loss，只能作为工程
sanity check。

2026-09-15 本机 CPU 环境已验证 5/5 项单元测试通过。该环境位于本目录
的 `.venv` 中，仅用于轻量开发测试，不用于加载正式图像生成权重。

## Scale-RAE 真实插入点审计

官方源码显示主生成器层级为 `model.diff_head.model.dit_blocks`，block 包含
`attn.qkv`、`attn.proj`、`mlp.*` 和 `adaLN_modulation.*`。`target_audit.py`
对已加载的模型枚举精确路径和维度，并按 rank 估算 LoRA 参数量。

在集群 Scale-RAE 环境中运行（会加载正式权重，不要在本机 CPU 运行）：

```bash
python development/recursive_lora/audit_loaded_scale_rae.py \
  --scale-rae-root baseline_workspace/Scale-RAE \
  --model-path nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B \
  --layers {12..27} --rank 8 --device cuda --dtype bfloat16 \
  --output runs/scale_rae_lora_targets.json
```

第一个建议烟测候选是仅注入 blocks 12--27 的 `attn.proj`，因为它是单一、
易归因的残差输出路径。这是工程起点，不是已证明的最佳位置。

官方 Stage 1/2 脚本给出 2.4B DiT 的配置为 hidden size 2048、32 层、32 heads，
DDT encoder depth 2。因此，rank 8 时仅给 blocks 12--27 的 16 个 `attn.proj`
注入 LoRA，理论新增参数为
`16 * 8 * (2048 + 2048) = 524,288`，约 0.52M。正式运行仍以 checkpoint
审计 JSON 为准，而不仅依赖训练脚本。
