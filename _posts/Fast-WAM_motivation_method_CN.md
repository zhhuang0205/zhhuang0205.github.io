# Fast-WAM：动机与方法整理翻译

> 原文：**Fast-WAM: Do World Action Models Need Test-time Future Imagination?**  
> 本文档整理并翻译论文中与 **motivation（研究动机）** 和 **method（方法）** 相关的核心内容，重点保留论文的技术逻辑、公式和结构，而不是逐句机械翻译全文。

---

## 1. 研究动机

### 1.1 背景：为什么需要 World Action Models？

构建通用具身智能体要求策略模型不仅能够把视觉观测映射到机器人动作，还需要理解物理世界在交互过程中的演化规律。传统的 Vision-Language-Action（VLA）模型通常直接建模：

\[
p(a_{1:H}\mid o,l)
\]

其中 \(o\) 是当前视觉观测，\(l\) 是语言任务指令，\(a_{1:H}\) 是长度为 \(H\) 的动作块。也就是说，标准 VLA 更像是一个直接策略模型：输入当前图像和语言，输出未来动作。

World Action Models（WAMs）进一步引入未来视觉预测，把「世界会怎么变化」显式纳入动作建模过程。与标准 VLA 相比，WAM 的吸引力在于：通过预测未来视觉观测，模型可能学到更强的物理动态、接触关系和任务相关的时序结构。

---

### 1.2 现有 WAM 的主流范式：imagine-then-execute

大多数现有 WAM 遵循一种 **imagine-then-execute** 范式：

1. 先生成未来视觉观测，例如未来图像或视频；
2. 再基于当前观测、语言指令和想象出的未来视觉结果预测动作。

论文将这种形式写成：

\[
p(a_{1:H}\mid o,l)
=
\int p(v_{1:T}\mid o,l)\,p(a_{1:H}\mid o,l,v_{1:T})\,dv_{1:T}
\]

其中：

- \(v_{1:T}\)：未来视觉观测序列；
- \(p(v_{1:T}\mid o,l)\)：根据当前观测和语言指令想象未来；
- \(p(a_{1:H}\mid o,l,v_{1:T})\)：根据想象出的未来生成动作。

这个思路很直观：如果模型先知道「任务完成后的世界应该长什么样」，动作预测就可以转化为某种逆动力学或轨迹跟踪问题。

---

### 1.3 现有问题：测试时显式生成未来视频很慢

虽然 imagine-then-execute 很自然，但它有一个明显缺点：**测试/部署时需要显式生成未来视频，带来很高延迟**。

许多 WAM 需要在推理时执行迭代式视频去噪，例如 diffusion video generation。这个过程计算成本高，不利于真实机器人实时控制。

因此，论文提出一个核心疑问：

> WAM 的性能提升，真的来自测试时显式“想象未来”吗？  
> 还是主要来自训练时的视频建模目标，让模型学到了更好的世界表征？

换句话说，WAM 的收益可能来自两个不同因素：

1. **训练时的视频预测目标**  
   该目标可能促使模型学习物理上有意义的 latent representation，例如运动、交互、物体变化和任务时序结构。

2. **推理时的显式未来生成**  
   显式生成未来图像/视频可能为动作预测提供额外 foresight。

现有 WAM 通常把这两个因素绑定在一起：训练时学视频预测，推理时也生成未来视频。因此，很难判断到底是哪个因素真正贡献了性能。

---

### 1.4 Fast-WAM 的核心观点

Fast-WAM 的出发点是解耦这两个因素：

- **训练时保留 video co-training**，也就是继续用未来视频预测目标塑造模型的世界表征；
- **推理时跳过 future prediction**，不再显式生成未来视频，而是直接从当前观测和语言产生动作。

论文的核心假设是：

> 如果 world modeling 的主要价值在于训练阶段塑造更好的 latent representation，那么 WAM 应该可以在不进行测试时未来视频合成的情况下保留大部分收益。

因此，Fast-WAM 试图回答：

> World Action Models 是否真的需要 test-time future imagination？  
> 还是说训练时的视频建模才是关键？

---

### 1.5 论文贡献

论文的贡献可以整理为三点：

1. **提出并研究一个基础问题**  
   WAM 的收益主要来自训练时的视频建模，还是来自推理时显式想象未来？

2. **提出 Fast-WAM 架构**  
   Fast-WAM 在训练时保留视频联合训练目标，但在测试时去掉未来视频生成，从而实现实时推理。

3. **通过受控对比实验验证结论**  
   作者构建了多个受控变体，包括保留/去除视频联合训练、保留/去除测试时未来生成。实验显示：Fast-WAM 与显式 imagine-then-execute 变体性能接近，但去除 video co-training 会导致更大性能下降。这说明 WAM 的主要收益可能来自训练时的视频预测目标，而不是测试时显式生成未来观测。

---

## 2. 方法

## 2.1 问题形式化

论文考虑从视觉观测和语言指令中学习具身策略。设：

- \(o\)：当前观测；
- \(l\)：语言任务指令；
- \(a_{1:H}\)：长度为 \(H\) 的动作块。

标准视觉运动策略建模为：

\[
p(a_{1:H}\mid o,l)
\tag{1}
\]

即直接从当前感知上下文预测动作序列。

WAM 则引入未来视觉观测 \(v_{1:T}\)，很多现有方法采用 imagine-then-execute 分解：

\[
p(a_{1:H}\mid o,l)
=
\int p(v_{1:T}\mid o,l)
p(a_{1:H}\mid o,l,v_{1:T})
dv_{1:T}
\tag{2}
\]

这意味着模型先预测未来观测，再以该未来观测为条件生成动作。

Fast-WAM 的关键改变是：训练时仍然使用 world modeling 作为联合训练信号，但推理时不再显式生成未来观测。推理时直接建模：

\[
p_\theta(a_{1:H}\mid o,l)
\tag{3}
\]

但这个直接策略并不是普通 VLA。它的动作分布由训练时被视频建模目标塑造过的 latent world representation 参数化。设：

\[
z(o,l)
\]

表示视频 backbone 根据当前上下文产生的 latent world representation，则：

\[
p_\theta(a_{1:H}\mid o,l)
=
p_\theta(a_{1:H}\mid z(o,l))
\tag{4}
\]

与 imagine-then-execute WAM 的关键区别是：

- 传统 WAM：推理时显式采样或去噪未来视频 \(v_{1:T}\)；
- Fast-WAM：推理时只做一次前向编码，得到 \(z(o,l)\)，再直接生成动作。

---

## 2.2 模型架构

### 2.2.1 总体设计

Fast-WAM 的设计目标是：

> 保留 world modeling 在训练中带来的表征学习收益，同时移除推理时显式未来想象的成本。

训练时，Fast-WAM 同时学习：

1. 动作预测；
2. 未来视频建模。

这样可以鼓励视觉 backbone 捕捉物理上有意义的运动结构和交互结构。

推理时，Fast-WAM 不生成未来观测。它只保留第一帧观测的 clean latent tokens，让视频模型做一次前向传播，产生 latent world features，然后用于动作生成。

因此，Fast-WAM 在测试时看起来像一个直接策略模型，但它的内部表征是在 WAM-style 视频监督下训练出来的。

---

### 2.2.2 Backbone 与专家结构

Fast-WAM 构建在 **Wan2.2-5B** 的 video Diffusion Transformer（Video DiT）之上。该 Video DiT 作为 world modeling backbone。

论文还复用了 Wan2.2-5B 的两个组件：

- **text encoder**：使用内置 T5 encoder 编码任务语言；
- **video VAE**：将视觉观测映射为 latent video tokens。

在此基础上，作者引入一个 **action expert DiT**，用于生成动作块。

整体架构是一个 **Mixture-of-Transformer（MoT）** 结构，其中包括：

- video DiT 分支；
- action expert DiT 分支；
- 二者通过 shared attention 交互。

---

### 2.2.3 输入 token 分组

Fast-WAM 将输入 token 分成三组：

1. **第一帧观测的 clean latent tokens**  
   这些 token 作为共享视觉锚点，提供当前场景上下文。

2. **未来视频帧的 noisy latent tokens**  
   这些 token 只在训练阶段用于视频建模，不在测试时使用。

3. **action tokens**  
   这些 token 由 action expert 处理，用于动作块生成。

所有 token 组都会通过 cross-attention 访问语言 embedding，因此语言指令会同时影响视频建模和动作预测。

---

### 2.2.4 结构化 attention mask

Fast-WAM 使用 structured attention mask 控制信息流。其目的有两个：

1. 让视频建模和动作预测共享同一个视觉上下文；
2. 防止未来信息泄漏到动作分支中，从而保证受控对比的公平性。

训练时的信息流设计如下：

- 未来 noisy video tokens 可以在 video branch 内部双向 attention，并且可以访问 clean first-frame tokens；
- action tokens 可以在 action branch 内部双向 attention，并且可以访问 clean first-frame tokens；
- **action tokens 不能 attend 到 future video tokens**；
- clean first-frame tokens 不 attend 其他 token。

这一点非常重要：Fast-WAM 的动作分支在训练时不能直接偷看未来视频 token，因此它不是通过显式未来帧来预测动作，而是通过共享的世界表征学习受益。

---

### 2.2.5 推理过程

推理时，Fast-WAM 完全移除 future video branch：

- 不实例化未来 noisy video tokens；
- 不执行未来视频去噪；
- 只保留第一帧 clean latent tokens；
- video backbone 对当前观测上下文做一次前向传播；
- action expert 基于得到的 latent world features 进行动作去噪。

因此，Fast-WAM 的推理成本显著低于标准 imagine-then-execute WAM。

论文报告，在真实机器人任务中 Fast-WAM 延迟约为 **190 ms**，显著快于需要显式未来生成的变体。

---

## 2.3 训练目标

Fast-WAM 使用 joint flow matching objective，同时作用于动作 tokens 和未来视频 latents。

给定目标变量 \(y\)，它可以是：

- 动作块 \(a_{1:H}\)；
- 未来视频 latent \(z_{1:T}\)。

采样高斯噪声：

\[
\epsilon \sim \mathcal{N}(0,I)
\]

以及时间步：

\[
t\in(0,1)
\]

构造插值样本：

\[
y_t=(1-t)y+t\epsilon
\tag{5}
\]

模型学习预测对应的 velocity field，标准 flow matching loss 为：

\[
L_{FM}(y)
=
\mathbb{E}_{y,\epsilon,t}
\left[
\|f_\theta(y_t,t,o,l)-(\epsilon-y)\|_2^2
\right]
\tag{6}
\]

---

### 2.3.1 动作生成损失

对于动作预测，令：

\[
y=a_{1:H}
\]

则动作损失为：

\[
L_{act}=L_{FM}(a_{1:H})
\tag{7}
\]

这使 action expert 学会从带噪动作样本逐步去噪到真实动作块。

---

### 2.3.2 视频联合训练损失

对于视频联合训练，令：

\[
y=z_{1:T}
\]

其中 \(z_{1:T}\) 是由预训练 VAE 得到的未来视频帧 latent tokens。视频损失为：

\[
L_{vid}=L_{FM}(z_{1:T})
\tag{8}
\]

这个损失的作用不是为了推理时生成未来视频，而是为了在训练阶段塑造 video backbone 的世界表征，使其编码运动、交互和物理动态。

---

### 2.3.3 总损失

总训练目标为：

\[
L=L_{act}+\lambda L_{vid}
\tag{9}
\]

其中 \(\lambda\) 用于平衡动作学习和视频联合训练。

---

## 2.4 受控变体设计

为了回答“WAM 的收益来自训练时 video co-training 还是推理时 future imagination”这一核心问题，作者设计了一组受控变体。所有变体尽量共享相同 backbone、tokenization 和训练配方，从而隔离不同因素的影响。

---

### 2.4.1 Fast-WAM-Joint

**Fast-WAM-Joint** 对应联合生成范式。

它模拟现有 joint-modeling WAM：

- 未来视频 tokens 和动作 tokens 在同一个模型中一起去噪；
- action generation 在整个去噪过程中始终与未来视频建模耦合；
- 推理时保留未来视频生成。

这个变体代表图 1(A) 中的 WAM 设计：future video and action joint denoising。

---

### 2.4.2 Fast-WAM-IDM

**Fast-WAM-IDM** 对应 video-then-action 范式。

它模拟现有 causal / inverse-dynamics-style WAM：

1. 先从当前观测和语言上下文生成未来视频 tokens；
2. 再以生成的未来表示为条件进行动作预测。

这个变体代表图 1(B) 中的 WAM 设计：future video prediction followed by action prediction。

---

### 2.4.3 Fast-WAM without video co-training

第三个变体是 **去掉视频联合训练的 Fast-WAM**。

它保持：

- 架构不变；
- 推理流程不变；
- action branch 不变。

唯一改变是：训练时移除 video modeling objective，也就是不使用 \(L_{vid}\)。

这个变体直接用于测试：

> 如果没有训练时的视频预测目标，性能会下降多少？

论文实验显示，该变体性能下降显著，说明 video co-training 本身对 WAM 性能非常关键。

---

## 3. 图 1 的三类 WAM 范式解释

论文图 1 对比了三种代表性 WAM 设计。

### 3.1 Joint future video & action prediction

对应图 1(A)。

这种方法在训练和推理时都让未来视频和动作 tokens 一起去噪。它的特点是：

- video 和 action 通过 shared attention 耦合；
- 动作预测可以在去噪过程中直接利用未来视频表示；
- 但推理时需要处理视频分支，因此开销较高。

---

### 3.2 Future video prediction followed by action prediction

对应图 1(B)。

这种方法先进行未来视频去噪，再把生成的未来视频表示交给 action module。它的特点是：

- 符合直观的 imagine-then-execute；
- 未来视频显式作为动作预测条件；
- 推理需要先 denoise video，再 denoise action，延迟更高。

---

### 3.3 Fast-WAM: action prediction without attending future video

对应图 1(C)。

Fast-WAM 的核心设计是：

- 训练时仍然有视频预测目标；
- 推理时不生成未来视频；
- 动作预测不 attend future video；
- video prediction 只作为训练目标，用于塑造 latent world representation。

也就是说，Fast-WAM 把视频预测的价值从“推理时显式想象”转移到“训练时表征学习”。

---

## 4. 实验结论对应方法动机

论文的实验结果支持其核心假设。

### 4.1 仿真结果

在 RoboTwin 和 LIBERO 上，Fast-WAM 不使用 embodied pretraining 也能达到接近 SOTA 的表现。

更重要的是，受控变体显示：

- Fast-WAM 与 Fast-WAM-Joint、Fast-WAM-IDM 性能接近；
- 去掉 video co-training 后性能大幅下降。

这说明：显式 test-time future imagination 并不是 WAM 性能提升的唯一关键，甚至可能不是主要关键；训练时的视频预测目标更重要。

---

### 4.2 真实机器人结果

论文在真实毛巾折叠任务上评估。该任务具有长时序、可变形物体和闭环控制难点。

结果表明：

- Fast-WAM 具备较强成功率；
- Fast-WAM 的推理延迟约为 190 ms；
- imagine-then-execute 变体更慢，例如 Fast-WAM-IDM 延迟约 810 ms；
- 去掉 video co-training 后成功率和完成时间都明显恶化。

因此，Fast-WAM 在真实部署上具有更好的性能-效率折中。

---

## 5. 总结

Fast-WAM 重新审视了 World Action Models 中一个关键设计假设：

> 测试时是否必须显式生成未来观测？

论文的答案是：不一定。

Fast-WAM 表明，WAM 的主要价值可能来自训练时的视频建模目标，因为该目标能让模型学习更好的 world-grounded latent representations。只要训练时保留 video co-training，推理时可以跳过显式未来视频生成，直接用 latent world representation 预测动作，从而获得接近 imagine-then-execute WAM 的性能，同时显著降低延迟。

用一句话概括：

> **Fast-WAM 不在测试时“想象未来视频”，而是在训练时通过预测未来视频学会更好的世界表征；推理时直接用这个表征生成动作。**

