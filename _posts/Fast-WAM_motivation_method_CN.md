# Fast-WAM：动机与方法整理

> 论文：**Fast-WAM: Do World Action Models Need Test-time Future Imagination?**  
> 核心问题：**World Action Models 的收益主要来自测试时显式想象未来，还是来自训练时的视频建模？**

---

## 1. 论文动机

### 1.1 背景：为什么提出 World Action Models？

通用具身智能体不仅需要根据当前观测输出动作，还需要理解物理世界如何在交互中变化。传统的 Vision-Language-Action（VLA）模型通常直接学习从视觉观测和语言指令到动作的映射：

```math
p(a_{1:H}\mid o,l)
```

其中：

- \(o\)：当前视觉观测；
- \(l\)：语言任务指令；
- \(a_{1:H}\)：长度为 \(H\) 的动作块。

这种方式本质上是一个直接策略模型：看当前图像和语言，直接输出动作。问题是，它不一定显式建模“如果机器人这么做，世界会如何变化”。

World Action Models（WAMs）试图弥补这一点。WAMs 通常把未来视觉预测和动作建模结合起来，让模型不仅预测动作，还学习未来视觉状态的演化。相比普通 VLA，WAM 的优势在于：未来视频建模可能帮助模型学习物理动态、物体交互和任务时序结构。

---

### 1.2 现有 WAM 的主流范式：imagine-then-execute

大多数已有 WAM 采用 **imagine-then-execute** 范式：

1. 先根据当前观测和语言指令生成未来视觉观测；
2. 再基于生成的未来视觉结果预测动作。

论文将这种分解写成：

```math
p(a_{1:H}\mid o,l)
=
\int p(v_{1:T}\mid o,l)\,
p(a_{1:H}\mid o,l,v_{1:T})\,dv_{1:T}
```

其中：

- \(v_{1:T}\)：未来视觉观测序列；
- \(p(v_{1:T}\mid o,l)\)：未来视觉预测模型；
- \(p(a_{1:H}\mid o,l,v_{1:T})\)：基于未来视觉结果的动作预测模型。

这种设计很直观：如果模型先想象出任务完成过程或未来状态，那么动作预测可以变成类似逆动力学的问题。

---

### 1.3 关键问题：测试时显式想象未来是否必要？

imagine-then-execute 的问题是推理成本高。许多 WAM 需要在测试时执行迭代式视频去噪，显式生成未来视频，然后再预测动作。这会带来较高延迟，不利于实时机器人控制。

论文指出，现有 WAM 的收益可能来自两个不同因素：

1. **训练时的视频预测目标**  
   视频预测可能让模型学习到更好的物理先验和动作条件表征。

2. **推理时的显式未来生成**  
   测试时生成未来视频可能为动作预测提供额外 foresight。

但现有 WAM 通常同时包含这两个因素，因此很难判断：性能提升到底来自训练时的视频建模，还是来自推理时显式生成未来视频。

因此，论文提出核心问题：

> WAM 是否真的需要在测试时显式想象未来？  
> 还是说，它们的主要收益其实来自训练阶段的视频建模？

---

### 1.4 Fast-WAM 的核心想法

Fast-WAM 的核心思想是解耦这两个因素：

- 训练时：保留视频联合训练目标；
- 推理时：跳过未来视频生成，直接预测动作。

也就是说，Fast-WAM 不在测试时生成未来视频，而是把视频预测作为训练时的辅助目标，用它来塑造更好的 latent world representation。

如果 world modeling 的主要价值在于训练阶段学习更好的世界表征，那么模型应该可以在不进行测试时未来视频生成的情况下，仍然保留大部分性能收益。

---

### 1.5 论文贡献

论文的主要贡献包括：

1. **提出一个关键研究问题**  
   系统研究 WAM 的收益究竟主要来自训练时 video modeling，还是来自测试时 explicit future imagination。

2. **提出 Fast-WAM 架构**  
   Fast-WAM 在训练时保留 video co-training，但在推理时去掉 future prediction，从而实现实时动作生成。

3. **通过受控实验验证结论**  
   作者构建多个 Fast-WAM 变体，包括保留测试时未来生成的版本，以及去掉 video co-training 的版本。实验表明，Fast-WAM 与 imagine-then-execute 变体性能接近，而去掉 video co-training 会导致更大性能下降。

---

## 2. 方法

## 2.1 问题形式化

设：

- \(o\)：当前观测；
- \(l\)：语言指令；
- \(a_{1:H}\)：动作块；
- \(v_{1:T}\)：未来视觉观测。

标准 visuomotor policy 建模为：

```math
p(a_{1:H}\mid o,l)
```

WAM 则引入未来视觉变量：

```math
p(a_{1:H}\mid o,l)
=
\int p(v_{1:T}\mid o,l)
p(a_{1:H}\mid o,l,v_{1:T})
dv_{1:T}
```

Fast-WAM 的做法不同。它训练时仍然保留 world modeling 目标，但推理时不显式生成 \(v_{1:T}\)，而是直接预测：

```math
p_\theta(a_{1:H}\mid o,l)
```

不过，这个直接策略并不是普通 VLA。它使用由视频 backbone 产生的 latent world representation。设：

```math
z(o,l)
```

表示视频 backbone 根据当前观测和语言产生的 latent world representation，则 Fast-WAM 建模为：

```math
p_\theta(a_{1:H}\mid o,l)
=
p_\theta(a_{1:H}\mid z(o,l))
```

关键区别是：

- imagine-then-execute WAM：推理时显式采样或去噪未来视频；
- Fast-WAM：推理时只做一次前向编码，得到 latent world representation，然后直接生成动作。

---

## 2.2 模型架构

### 2.2.1 总体架构

Fast-WAM 的目标是：

> 保留 world modeling 在训练阶段带来的表征学习收益，同时移除推理阶段显式未来视频生成的成本。

训练时，Fast-WAM 同时学习：

- 动作预测；
- 未来视频建模。

推理时，Fast-WAM 不生成未来观测，只保留第一帧观测的 clean latent tokens，通过视频模型一次前向传播得到 latent world features，再用于动作生成。

因此，Fast-WAM 在测试时具有类似直接策略的接口，但它的表征是在 WAM-style 视频监督下训练得到的。

---

### 2.2.2 Backbone

Fast-WAM 基于 **Wan2.2-5B** 的 video Diffusion Transformer（Video DiT）构建。该 Video DiT 作为 world modeling backbone。

模型还复用了 Wan2.2-5B 的：

- **text encoder**：用于编码语言任务指令；
- **video VAE**：用于将图像观测编码为 latent video tokens。

在 video DiT 之外，作者引入一个 **action expert DiT**，用于动作块生成。

整体模型是一个 **Mixture-of-Transformer（MoT）** 架构，包含：

- video DiT branch；
- action expert DiT branch；
- 二者之间通过 shared attention 交互。

---

### 2.2.3 输入 token 组成

Fast-WAM 将输入 token 分为三类：

1. **clean latent tokens of the first observation frame**  
   第一帧观测的干净 latent token，作为共享视觉锚点。

2. **noisy latent tokens of future video frames**  
   未来视频帧的带噪 latent token，只在训练阶段用于视频建模。

3. **action tokens**  
   动作 token，由 action expert 处理，用于生成动作块。

所有 token 组都通过 cross-attention 访问语言 embedding，因此语言指令会同时影响视频建模和动作预测。

---

### 2.2.4 结构化 attention mask

Fast-WAM 使用 structured attention mask 控制不同 token 之间的信息流。

训练阶段：

- future noisy video tokens 可以在 video branch 内部双向 attention，并可以访问 clean first-frame tokens；
- action tokens 可以在 action branch 内部双向 attention，并可以访问 clean first-frame tokens；
- action tokens 不能 attend 到 future video tokens；
- clean first-frame tokens 不 attend 其他 token。

这个设计非常关键。它保证 action branch 不能直接偷看未来视频 token。因此，动作预测不能依赖真实未来视频信息泄漏，而只能通过共享的当前视觉上下文和训练出的 latent world representation 获益。

---

### 2.2.5 推理阶段

推理时，Fast-WAM 完全移除未来视频分支：

- 不实例化 future noisy video tokens；
- 不执行未来视频去噪；
- 只保留第一帧 clean latent tokens；
- video backbone 对当前观测做一次前向传播；
- action expert 根据 latent world features 进行动作去噪。

因此，Fast-WAM 的推理成本显著低于 imagine-then-execute WAM。

---

## 2.3 训练目标

Fast-WAM 使用 joint flow matching objective，同时用于动作 token 和未来视频 latent。

给定目标变量 \(y\)，它可以是：

- 动作块 \(a_{1:H}\)；
- 未来视频 latent \(z_{1:T}\)。

采样高斯噪声：

```math
\epsilon \sim \mathcal{N}(0,I)
```

采样时间步：

```math
t\in(0,1)
```

构造插值样本：

```math
y_t=(1-t)y+t\epsilon
```

模型学习预测 velocity field：

```math
L_{FM}(y)
=
\mathbb{E}_{y,\epsilon,t}
\left[
\left\|f_\theta(y_t,t,o,l)-(\epsilon-y)\right\|_2^2
\right]
```

---

### 2.3.1 动作损失

对于动作预测，令：

```math
y=a_{1:H}
```

得到动作损失：

```math
L_{act}=L_{FM}(a_{1:H})
```

该损失训练 action expert 从带噪动作样本恢复真实动作块。

---

### 2.3.2 视频联合训练损失

对于视频联合训练，令：

```math
y=z_{1:T}
```

其中 \(z_{1:T}\) 是由预训练 VAE 编码得到的未来视频帧 latent tokens。

视频损失为：

```math
L_{vid}=L_{FM}(z_{1:T})
```

这个损失的目的不是为了测试时生成未来视频，而是为了在训练阶段塑造 video backbone 的世界表征，使其学习运动、交互和物理动态。

---

### 2.3.3 总损失

总训练目标为：

```math
L=L_{act}+\lambda L_{vid}
```

其中 \(\lambda\) 用于平衡动作学习和视频联合训练。

---

## 2.4 受控变体

为了回答核心问题，作者设计了一组受控变体。所有变体尽量共享相同的 backbone、tokenization 和训练 recipe，从而隔离不同因素的作用。

---

### 2.4.1 Fast-WAM

Fast-WAM 是论文主方法。

特点：

- 训练时保留 video co-training；
- 推理时不生成未来视频；
- action tokens 不 attend future video tokens；
- 通过一次前向传播得到 latent world representation；
- 直接进行动作预测。

它对应论文图 1(C)：video prediction only as objective, action prediction without attending future video。

---

### 2.4.2 Fast-WAM-Joint

Fast-WAM-Joint 模拟 joint-generation WAM。

特点：

- future video tokens 和 action tokens 在同一个模型中联合去噪；
- 动作生成在整个去噪过程中与未来视频建模耦合；
- 推理时保留显式未来视频生成。

它对应论文图 1(A)：joint future video & action prediction。

---

### 2.4.3 Fast-WAM-IDM

Fast-WAM-IDM 模拟 video-then-action WAM。

流程：

1. 先从当前观测和语言上下文生成未来视频 tokens；
2. 再以生成的未来表示为条件预测动作。

它对应论文图 1(B)：future video prediction followed by action prediction。

---

### 2.4.4 Fast-WAM without video co-training

该变体保持架构和推理流程不变，但训练时移除视频建模目标。

也就是说，它去掉：

```math
L_{vid}
```

只保留动作学习目标。该变体用于验证 video co-training 本身的重要性。

实验结果显示，去掉 video co-training 会导致明显性能下降，说明训练时的视频建模目标是 WAM 收益的重要来源。

---

## 3. 方法逻辑总结

Fast-WAM 的整体逻辑可以概括为：

1. 训练时使用未来视频预测目标，让 video backbone 学习物理世界动态；
2. 训练时同时学习动作生成；
3. 通过 structured attention mask 防止动作分支直接偷看未来视频；
4. 推理时完全移除未来视频生成；
5. 只用当前观测产生 latent world representation；
6. 基于该 representation 直接生成动作。

因此，Fast-WAM 并不是否定 world modeling，而是把 world modeling 的作用从“测试时生成未来视频”转移到“训练时塑造世界表征”。

---

## 4. 实验结论与方法对应关系

论文实验显示：

- Fast-WAM 在 LIBERO 和 RoboTwin 上不使用 embodied pretraining 也能达到接近 SOTA 的表现；
- Fast-WAM 与 Fast-WAM-Joint、Fast-WAM-IDM 性能接近；
- 去掉 video co-training 后性能明显下降；
- 在真实机器人毛巾折叠任务中，Fast-WAM 保持较好性能，同时推理延迟仅约 190 ms；
- imagine-then-execute 变体延迟明显更高，例如 Fast-WAM-IDM 达到约 810 ms。

这些结果说明：

> WAM 的主要价值可能更多来自训练时的视频预测目标，而不是测试时显式生成未来观测。

---

## 5. 一句话总结

**Fast-WAM 的核心思想是：训练时让模型学习预测未来视频，从而获得更好的世界表征；推理时不再生成未来视频，而是直接利用这种表征预测动作，实现接近 WAM 的性能和接近 VLA 的推理效率。**
