# Fast-WAM：动机与方法整理

> 论文：**Fast-WAM: Do World Action Models Need Test-time Future Imagination?**  
> 核心问题：World Action Models 的收益到底来自 **训练时的视频建模**，还是来自 **推理时显式生成未来视频**？

---

## 1. 论文动机

### 1.1 研究背景：从 VLA 到 WAM

具身智能体需要根据视觉观测和语言指令执行动作。传统 Vision-Language-Action（VLA）模型通常直接学习从当前观测到动作的映射：

```math
p(a_{1:H} \mid o, l)
```

其中，$o$ 表示当前视觉观测，$l$ 表示语言任务指令，$a_{1:H}$ 表示长度为 $H$ 的动作块。

这种直接策略形式简单高效，但它没有显式建模物理世界在动作作用下如何变化。也就是说，模型主要学习「看见什么 → 做什么」，而不是「做了之后世界会怎么变」。

World Action Models（WAMs）试图补上这一点。WAM 不只预测动作，还引入未来视觉预测，让模型学习未来观测如何随任务和动作演化。其核心直觉是：如果模型能预测未来图像或视频，就可能学到更强的物理动态、接触关系、物体运动和任务时序结构。

---

### 1.2 现有 WAM 的典型范式：imagine-then-execute

大多数已有 WAM 遵循 **imagine-then-execute** 范式，即：

1. 先根据当前观测和语言指令生成未来视觉观测；
2. 再根据想象出的未来视觉结果预测动作。

论文将该过程形式化为：

```math
p(a_{1:H} \mid o, l)
=
\int p(v_{1:T} \mid o, l)\,
p(a_{1:H} \mid o, l, v_{1:T})\,
dv_{1:T}
```

其中，$`v_{1:T}`$ 表示预测得到的未来视觉观测序列。

这个分解可以理解为：

- $p(v_{1:T} \mid o, l)$：世界模型，根据当前状态和任务想象未来；
- $p(a_{1:H} \mid o, l, v_{1:T})$：动作模型，根据想象出的未来反推动作。

这种方法很直观，因为未来图像可以作为一种 visual plan 或 visual goal，使动作预测更接近 inverse dynamics 问题。

---

### 1.3 现有问题：测试时未来想象代价高

虽然 imagine-then-execute 很自然，但它在测试时通常需要迭代式视频生成，尤其是 diffusion video denoising，因此推理延迟很高。

对真实机器人控制来说，这个问题很关键。机器人需要实时闭环控制，如果每一步动作都要先生成未来视频，计算成本会显著增加。

因此，论文提出一个核心问题：

> WAM 真的需要在测试时显式生成未来视频吗？

更具体地说，WAM 的收益可能来自两个因素：

### 因素一：训练时的视频预测目标

训练时的视频预测目标可能迫使模型学习更好的世界表征，例如：

- 物体如何移动；
- 机器人动作如何改变场景；
- 接触、变形、遮挡等物理变化；
- 任务完成过程中的时序结构。

这种收益来自 representation learning。

### 因素二：推理时的显式未来生成

推理时生成未来图像或视频，可能为动作预测提供显式 foresight，使策略知道任务未来应该往哪里发展。

这种收益来自 test-time imagination。

---

### 1.4 论文的核心假设

现有 WAM 通常把这两个因素绑定在一起：训练时做视频预测，测试时也生成未来视频。因此无法判断到底哪个因素更重要。

Fast-WAM 的核心假设是：

> WAM 的主要收益可能并不来自测试时显式生成未来视频，而是来自训练时视频预测目标对 latent world representation 的塑造。

也就是说，模型可能不需要在部署时真的“想象未来图像”。只要训练时通过预测未来视频学到了世界动态，推理时就可以直接利用这种 latent world representation 来生成动作。

---

### 1.5 Fast-WAM 的基本思想

Fast-WAM 解耦了两个因素：

- 训练时：保留 video co-training，即继续用未来视频预测作为辅助训练目标；
- 推理时：跳过 future video prediction，不显式生成未来视频；
- 动作生成：直接从当前观测、语言和视频 backbone 编码出的 latent world representation 中预测动作。

因此，Fast-WAM 在推理接口上类似普通 VLA：

```math
p_\theta(a_{1:H} \mid o, l)
```

但它不是普通 VLA，因为其视觉 backbone 在训练时受到了 WAM-style future video prediction 的监督。

---

## 2. 方法

## 2.1 问题形式化

标准 visuomotor policy 直接建模：

```math
p(a_{1:H} \mid o, l)
```

其中：

- $o$：当前视觉观测；
- $l$：任务语言指令；
- $a_{1:H}$：未来 $H$ 步动作块。

WAM 引入未来视觉观测 $v_{1:T}$，典型形式为：

```math
p(a_{1:H} \mid o, l)
=
\int p(v_{1:T} \mid o, l)
p(a_{1:H} \mid o, l, v_{1:T})
dv_{1:T}
```

Fast-WAM 不在推理时显式采样或去噪 $v_{1:T}$。它直接预测动作：

```math
p_\theta(a_{1:H} \mid o, l)
```

但该动作分布由一个 latent world representation 参数化。设视频 backbone 根据当前观测和语言得到：

```math
z(o,l)
```

则 Fast-WAM 的动作分布可以写成：

```math
p_\theta(a_{1:H} \mid o, l)
=
p_\theta(a_{1:H} \mid z(o,l))
```

这里的关键区别是：

- imagine-then-execute WAM：推理时显式生成未来视频 $v_{1:T}$；
- Fast-WAM：推理时只通过一次前向传播得到 $z(o,l)$，不生成未来视频。

---

## 2.2 模型架构

### 2.2.1 总体结构

Fast-WAM 的目标是：

> 保留 world modeling 的训练收益，同时移除测试时显式 future imagination 的推理成本。

训练时，模型同时学习：

1. 动作预测；
2. 未来视频 latent 建模。

推理时，模型只保留第一帧观测的 clean latent tokens，通过视频 backbone 得到 latent world features，再交给 action expert 生成动作。

---

### 2.2.2 Backbone

Fast-WAM 基于 **Wan2.2-5B** 的 Video Diffusion Transformer（Video DiT）构建。

论文复用了 Wan2.2-5B 的三个部分：

- **Video DiT**：作为 world modeling backbone；
- **Text Encoder**：使用内置 T5 encoder 编码语言指令；
- **Video VAE**：将图像观测编码为 video latent tokens。

在此基础上，作者加入一个 **Action Expert DiT**，专门负责生成动作块。

整体模型采用 **Mixture-of-Transformer（MoT）** 架构，包括：

- video branch；
- action branch；
- shared attention 机制。

---

### 2.2.3 输入 token 组成

Fast-WAM 将 token 分为三类：

### 第一类：clean first-frame latent tokens

这些 token 来自当前观测的第一帧，经 Video VAE 编码后得到。它们作为当前场景的视觉锚点。

### 第二类：noisy future video latent tokens

这些 token 对应未来视频帧的 latent 表示。它们只在训练阶段使用，用于视频建模目标。

推理阶段不再构造这些 token。

### 第三类：action tokens

这些 token 表示要生成的动作块，由 action expert DiT 处理。

所有 token 都可以通过 cross-attention 访问语言 embedding，因此语言指令会影响视频建模和动作预测。

---

## 2.3 结构化 Attention Mask

Fast-WAM 使用结构化 attention mask 控制不同 token 之间的信息流。

训练时：

- future video tokens 可以在 video branch 内部双向 attention；
- future video tokens 可以访问 clean first-frame tokens；
- action tokens 可以在 action branch 内部双向 attention；
- action tokens 可以访问 clean first-frame tokens；
- action tokens 不能 attend 到 future video tokens；
- clean first-frame tokens 不 attend 到其他 token。

这一设计的关键作用是：

> 动作分支不能直接偷看真实未来视频 token，因此动作预测的提升不能简单归因于 future video leakage。

这使得 Fast-WAM 可以更干净地测试：训练时的视频建模目标是否能单独提升动作策略。

---

## 2.4 推理过程

推理时，Fast-WAM 移除整个 future video branch：

1. 输入当前观测和语言；
2. 使用 Video VAE 得到当前帧 latent tokens；
3. Video DiT 对当前视觉上下文做一次前向传播；
4. 产生 latent world representation；
5. Action Expert DiT 基于该表示生成动作块。

推理时不执行：

- future video token 初始化；
- future video denoising；
- explicit future frame generation。

因此，Fast-WAM 显著降低推理延迟。

论文报告 Fast-WAM 在真实任务中约为 **190 ms** 延迟，而显式 imagine-then-execute 变体明显更慢。

---

## 2.5 训练目标

Fast-WAM 使用 flow matching objective，同时训练动作生成和视频建模。

给定目标变量 $y$，它可以是动作块 $a_{1:H}$，也可以是未来视频 latent $z_{1:T}$。

采样高斯噪声：

```math
\epsilon \sim \mathcal{N}(0,I)
```

采样时间步：

```math
t \in (0,1)
```

构造插值样本：

```math
y_t = (1-t)y + t\epsilon
```

模型学习预测 velocity field：

```math
L_{FM}(y)
=
\mathbb{E}_{y,\epsilon,t}
\left[
\|f_\theta(y_t,t,o,l)-(\epsilon-y)\|_2^2
\right]
```

---

### 2.5.1 动作损失

对动作生成，令：

```math
y = a_{1:H}
```

动作损失为：

```math
L_{act}=L_{FM}(a_{1:H})
```

该目标训练 action expert 从噪声动作逐步恢复真实动作块。

---

### 2.5.2 视频联合训练损失

对视频建模，令：

```math
y = z_{1:T}
```

其中 $z_{1:T}$ 是未来视频帧经过 pretrained VAE 得到的 latent tokens。

视频损失为：

```math
L_{vid}=L_{FM}(z_{1:T})
```

该损失的目的不是让模型在推理时一定生成未来视频，而是在训练阶段塑造 video backbone 的物理动态表征。

---

### 2.5.3 总损失

总训练目标为：

```math
L = L_{act} + \lambda L_{vid}
```

其中 $\lambda$ 用于平衡动作学习和视频联合训练。

---

## 2.6 受控变体设计

为了区分「训练时 video co-training」和「推理时 future imagination」的作用，论文设计了多个受控变体。

这些变体尽量共享相同的：

- backbone；
- tokenization；
- training recipe；
- action horizon；
- flow matching formulation。

这样可以更公平地比较不同设计因素。

---

### 2.6.1 Fast-WAM

这是论文主方法。

特点：

- 训练时保留视频联合训练；
- 推理时不生成未来视频；
- 动作生成不 attend future video；
- 视频预测只作为训练目标。

该方法用于验证：仅靠训练时 video co-training 是否足够获得强性能。

---

### 2.6.2 Fast-WAM-Joint

Fast-WAM-Joint 对应 joint-modeling WAM 范式。

特点：

- 未来视频 tokens 和动作 tokens 一起去噪；
- 推理时保留未来视频生成；
- action generation 与 video generation 在去噪过程中耦合。

它对应论文图 1(A)：Joint future video & action prediction。

---

### 2.6.3 Fast-WAM-IDM

Fast-WAM-IDM 对应 video-then-action 范式。

特点：

1. 先根据当前观测和语言生成未来视频；
2. 再让动作模型以生成的未来视频表示为条件预测动作。

它对应论文图 1(B)：Future video prediction followed by action prediction。

---

### 2.6.4 Fast-WAM without video co-training

该变体保持 Fast-WAM 的架构和推理流程不变，但去掉视频建模损失 $L_{vid}$。

也就是说，总损失变成：

```math
L = L_{act}
```

这个变体用于直接回答：

> 如果没有训练时的视频预测目标，模型性能会下降多少？

实验结果显示，该变体性能下降明显，说明 video co-training 是 Fast-WAM 性能的重要来源。

---

## 3. 方法核心对比

| 方法 | 训练时视频建模 | 推理时显式未来视频 | 动作是否依赖未来视频 | 推理效率 |
|---|---|---|---|---|
| Fast-WAM | 有 | 无 | 不直接依赖 | 高 |
| Fast-WAM-Joint | 有 | 有 | 依赖联合去噪过程 | 较低 |
| Fast-WAM-IDM | 有 | 有 | 依赖先生成的未来视频 | 最低 |
| w.o. video co-train | 无 | 无 | 不依赖 | 高 |

这个对比体现了论文的核心实验设计：将 video co-training 和 test-time future imagination 解耦。

---

## 4. 实验结论与方法解释

### 4.1 仿真基准

在 RoboTwin 和 LIBERO 上，Fast-WAM 在没有 embodied pretraining 的情况下取得了接近或达到强基线的性能。

关键发现是：

- Fast-WAM 与 Fast-WAM-Joint、Fast-WAM-IDM 性能接近；
- 去掉 video co-training 后性能下降更明显。

这说明，WAM 的主要收益更可能来自训练时的视频预测目标，而不是推理时显式生成未来视频。

---

### 4.2 真实机器人任务

论文在真实毛巾折叠任务上验证 Fast-WAM。该任务具有：

- 长时序；
- 可变形物体；
- 闭环控制需求；
- 对执行效率要求高。

结果显示：

- Fast-WAM 保持较强成功率；
- Fast-WAM 推理延迟约为 190 ms；
- Fast-WAM-IDM 等 imagine-then-execute 变体延迟更高；
- 去掉 video co-training 后成功率和完成时间显著恶化。

这说明 Fast-WAM 在真实部署中具有更好的性能-效率折中。

---

## 5. 总结

Fast-WAM 重新审视了 World Action Models 中的一个关键假设：

> 测试时是否必须显式生成未来观测？

论文的结论是：不一定。

Fast-WAM 表明，WAM 的主要价值可能来自训练时的视频建模目标。该目标可以塑造更好的 world-grounded latent representations，使模型在推理时无需显式生成未来视频，也能获得强动作预测能力。

一句话总结：

> **Fast-WAM 在训练时通过预测未来视频学习世界动态，在推理时跳过未来视频生成，直接用 latent world representation 生成动作，从而在保持性能的同时显著降低延迟。**
