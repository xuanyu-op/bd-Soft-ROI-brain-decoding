"""
Copied from
https://github.com/lucidrains/flamingo-pytorch/blob/main/flamingo_pytorch/flamingo_pytorch.py


1. 去除了 Gated Cross-Attention 与 MaskedCrossAttention
原版 flamingo-pytorch 中包含用于将视觉信息注入 LLM 的 GatedCrossAttentionBlock，带有 tanh gate 控制语义注入
UMBRAE 的 perceiver.py 中无此 gated attention 结构，仅保留压缩与重采样模块，专注于多模态 token 的统一编码。

3. 去除媒体相关的逻辑或模块
原版包含对“media_locations” mask 的处理，Later layers 控制 gated attention，仅在 LLM 中使用。UMBRAE 则完全不涉及这些机制，简化多模态，聚焦 encoding
"""

import torch
'''
爱因斯坦求和约定函数（Einstein Summation Convention）：
用字符串来指定复杂的张量操作（如矩阵乘法、点积、批量注意力等）
'''
from torch import nn, einsum

'''
rearrange：
用简洁字符串语法重新排列张量的维度
repeat：
扩展张量以增加维度，常用于复制 latent token
'''
from einops import rearrange, repeat
#可以对多个张量 同时进行相同的 rearrange 操作
from einops_exts import rearrange_many


#用于快速判断一个对象是否存在（非 None）
def exists(val):
    return val is not None
'''
这个函数 FeedForward 是一个标准的前馈神经网络模块（FeedForward Network, FFN），用于 Transformer 或 Perceiver 模块中，起到非线性特征变换和信息融合的作用。它将输入特征扩展到更高维，经过激活函数处理后再映射回原始维度。
这个模块是 Transformer/Perceiver 架构中的标准构件，一般搭配注意力层交替堆叠
✅ 整体功能概述
定义一个返回 nn.Sequential(...) 的函数，构建如下结构：
输入 (dim)
 ↓
LayerNorm
 ↓
Linear(dim → dim × mult)
 ↓
GELU 非线性激活
 ↓
Linear(dim × mult → dim)
 ↓
输出 (dim)
它用于增强模型表达能力，通常与 Attention 模块交替堆叠。

dim: 输入特征的维度
mult: 扩展倍数，默认值为 4，即中间层维度是输入维度的 4 倍

 inner_dim = int(dim * mult)
计算中间层的维度：
通常 FFN 会有一个瓶颈结构，先升维再降维
例如：dim = 512 → inner_dim = 2048

return nn.Sequential(
构建一个顺序神经网络容器（模块串联执行）

 nn.LayerNorm(dim),
层归一化：对输入进行标准化，提升训练稳定性
常见于 Transformer、Perceiver 架构中
归一化形状为 [batch_size, *, dim]，对最后一个维度做归一化

 nn.Linear(dim, inner_dim, bias = False),
线性映射：从 dim 升维到 inner_dim
不使用偏置项（bias=False），因为 LayerNorm 已处理偏移
形状变化：[batch, dim] → [batch, dim × mult]

  nn.GELU(),
激活函数：GELU

n.Linear(inner_dim, dim, bias = False)
线性降维：把激活后的大维度特征映射回原维度

🔁 使用示例
ff = FeedForward(512)     # 创建一个输入/输出为 512，内部维度为 2048 的 FF 模块
x = torch.randn(1, 512)   # 输入：batch=1, dim=512
y = ff(x)                 # 输出：shape 仍为 [1, 512]
'''
'''
🧬 FeedForward 的作用（结合位置解释）
在每一层 PerceiverResampler 中，结构是：
    latents ← latents + Attention(x, latents)
    latents ← latents + FeedForward(latents)
也就是每个 latent token 都会经历：
    Cross-Attention 更新
    让 latent 和外部输入（如图像或脑信号）交互
    得到含有语义信息的新 latent 表征
    FeedForward 非线性处理 ✅
    对每个 latent 进行维度内部的变换
    提升表达能力，让 latent token 自主提取、组合信息

🔍 为什么要有 FeedForward？
注意力（Attention）只负责 token 与 token 之间的关系建模，但：
它本身是线性的（点积 + 加权和）；
缺乏 token 内部维度之间的非线性交互能力；
所以我们需要 FeedForward 来增加：
特征变换深度；
多维之间的复杂组合关系；
表达多样性（尤其是跨模态映射时）；
这正是 Transformer/Perceiver 中的经典结构：“Attention + FeedForward” 的残差堆叠。

🧪 举个类比：
假设 latent token 是你的一组“笔记”，Attention 就像是你去查资料、从外部图像中找到相关信息填进笔记，而 FeedForward 就像是你独立消化这些信息并组织语言。
没有 FFN，你只能复制外部输入；加上 FFN，latent token 本身具备了“理解”和“组合”的能力。

✅ 总结一句话：
FeedForward 在 PerceiverResampler 中是用于每层 latent token 的维度内非线性变换，它补充了注意力机制不能提供的非线性特征处理能力，是整个 token 压缩和语义提炼过程的重要组成部分。
'''
def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )


'''
🧠 整体功能概述
这个类实现了一个 Cross-Attention 模块，它的作用是：
使用一组 learnable latent tokens 作为 Query，从外部输入（如图像 patch 特征或 fMRI token）中提取关键信息（作为 Key/Value），通过注意力机制聚合语义信息，从而不断优化 latent 表征。
它的结构类似于 Transformer 的 Multi-Head Attention，但 Query 和 Key/Value 来自不同来源，属于 Cross-Attention 类型。
'''


class PerceiverAttention(nn.Module):
    '''
    初始化函数：
        dim: 输入特征的维度
        dim_head: 每个注意力头的维度（默认 64）
        heads: 注意力头的数量（默认 8）
        注意 * 表示之后的参数必须用关键字方式传入，例如：PerceiverAttention(dim=512)。
    '''
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8
    ):
        '''
        self.heads = heads
            存储注意力头数。
         inner_dim = dim_head * heads
            计算多头注意力后的拼接维度。例如：8 个 head × 每个 head 64 维 = 总共 512 维。
        self.scale = dim_head ** -0.5
        在注意力机制中的分母
        self.norm_media = nn.LayerNorm(dim)
            对输入的媒体特征（如图像 patch 或 fMRI token）进行 LayerNorm，提升数值稳定性。
        self.norm_latents = nn.LayerNorm(dim)
        对 latent tokens 做 LayerNorm。
        这两个归一化操作是标准 Transformer 实践，避免数值爆炸、提升训练收敛速度。

         self.to_q = nn.Linear(dim, inner_dim, bias = False)
用于将 normalized latent token 映射成 Query 向量，形状：[batch, num_latents, dim] → [batch, num_latents, inner_dim]。这里不加 bias，是为了匹配点积 attention 的对称性。

           self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
用于将 normalized media 特征同时映射为 Key 和 Value：
输入：[batch, seq_len, dim]
输出：[batch, seq_len, inner_dim * 2]
然后再 reshape 拆分成 K 和 V

         self.to_out = nn.Linear(inner_dim, dim, bias = False)
将多个 head 拼接后的注意力输出还原回原始维度 dim，用于残差连接和下一层处理。如：latents = attn(x, latents) + latents  # 残差连接更新 latent tokens 保留原始表示 + 加入增量更新
        '''
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)
    '''
    对 latent tokens（Query）和媒体输入（Key/Value）执行 Cross-Attention 操作，输出更新后的 latent 表征。
    每个 latent token 会从输入 x 中获取语义信息，提升自己的表示能力。
    '''
    def forward(self, x, latents):
        """
        einstein notation
        b - batch
        t - time	输入 token 数（例如图像 patch/fMRI 分片数）
        n - sequence	latent token 数
        d - dimension每个 token 的维度

  x = self.norm_media(x)
对输入 x（图像或 fMRI token）做 LayerNorm，提升训练稳定性。
注意：这里归一化的是每个输入 token 的最后一维（特征维度）。

 latents = self.norm_latents(latents)
对 latent tokens 做 LayerNorm。
这一点很重要，因为 latent 是 learnable 参数，在 attention 之前需要数值归一化，确保稳定学习。

  b = x.shape[0]  # batch size
    m = x.shape[1]  # 输入 token 数（比如 patch/fMRI 分片数）
    h = self.heads  # 多头数量
这三个变量后续用于 reshape 或 attention 运算。

    q = self.to_q(latents)
将 latent tokens 送入 to_q 线性层（Linear(dim, heads × dim_head)），输出 Query：
输入 shape：[b, n, d]（n 是 latent token 数）
输出 shape：[b, n, h × dim_head]
这一步完成 latent token → 多头 Query 映射，是 cross-attention 的第一步。
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        '''
        ⚠️ 与原始 Perceiver 的区别：
        原始 Perceiver 中：
        K/V 仅来自媒体输入（如图像或 fMRI）
        而在 UMBRAE 中：
        K/V 来自媒体输入 + latent token → 允许模型“自我引用”，在 attending 的同时保留 latent 中已有的知识。
        📐 输入维度：
        x: [B, T, D]
        latents: [B, L, D]
        拼接后：kv_input: [B, T + L, D]
        多了一种“让自己也成为参考信息”的机制，这是一种轻微但重要的结构增强。
        
        kv_input = torch.cat((x, latents), dim = -2)
将媒体输入 x 和 latent tokens 在 sequence 维（即 token 数）上拼接起来，作为 Key 和 Value 的输入。

        k, v = self.to_kv(kv_input).chunk(2, dim = -1)
将拼接后的 kv_input 通过一个线性层 to_kv，生成 Key 和 Value 向量，并一分为二。

        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h = h)
用 einops_exts.rearrange_many 将 Q/K/V 都 reshape 成多头格式，准备做注意力点积。
📐 变换前后：
张量	变换前维度	变换后维度	含义
q	[B, L, H×d_h]	[B, H, L, d_h]	latent token 做 query
k	[B, T+L, H×d_h]	[B, H, T+L, d_h]	输入 + latent 作为 key
v	[B, T+L, H×d_h]	[B, H, T+L, d_h]	同上
t 是被 attention 的 token 数，即 T + L
n 是 query 数，即 L（latent token 数）

    q = q * self.scale
对 Query 向量进行缩放，避免后续点积过大（使 softmax 更稳定）。

        '''
        kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h = h)

        q = q * self.scale

        # attention注意力计算与输出构造
        '''
        sim = einsum('... i d, ... j d  -> ... i j', q, k)
            计算 Query 与 Key 之间的点积相似度（similarity score），用于后续 softmax 注意力权重。
            输入：
            q: [B, H, L, d_h]
            k: [B, H, T+L, d_h]
            输出：
            sim: [B, H, L, T+L]
            也就是说：
            每个 latent token（Query）去和所有 media + latent token（Key）做点积；
            得到每个 Query 与所有 Key 的匹配程度（score）；
        
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
            📌 功能：
            为了 防止 softmax 溢出（数值不稳定），进行标准的稳定化处理：从每一行中减去最大值。
            sim.amax(dim=-1) 是每个 attention 行（query 对应所有 keys）中的最大值；
            .detach() 是防止这个值对梯度产生影响；
            Softmax 依然保持不变，因为 softmax 对平移是不变的（只是数值更稳定）；
            
        attn = sim.softmax(dim = -1)
            📌 功能：
            对 sim 做 softmax → 得到注意力权重。
            形状为 [B, H, L, T+L]
            每个 latent token（在每个 head 中）都会对所有输入 token（图像+latent）产生一组注意力分布
        '''
        sim = einsum('... i d, ... j d  -> ... i j', q, k)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        #将多头输出重新 reshape 回原来的形状：out: [B, H, L, d_h] → [B, L, H × d_h] 这样做是为了后面通过线性层映射回原始维度。
        out = rearrange(out, 'b h t n d -> b t n (h d)', h = h)
        '''
        最后一层线性映射：Linear(H × d_h → D)
        作用：把多头 concat 的输出映射回原始维度 D；
        返回值：更新后的 latent 表征，形状为 [B, L, D]
        '''
        return self.to_out(out)


'''
将输入（图像 patch/fMRI token）编码成一组固定数量的 latent tokens 表征，通过多层 Cross-Attention 与 FeedForward 层更新，输出语义压缩后的表示。
| 参数名                | 含义                               |
| ------------------ | -------------------------------- |
| `dim`              | 输入与 latent token 的维度（如 512）      |
| `depth`            | Cross-Attention + FF 层堆叠的层数      |
| `dim_head`         | 每个注意力头的维度                        |
| `heads`            | 多头注意力的头数                         |
| `num_latents`      | latent tokens 的数量（即最终输出 token 数） |
| `num_media_embeds` | 输入模态的最大媒体数量（如图像帧/fMRI 段数）        |表示最大支持的媒体片段数目，每个片段都有一个独特的位置向量
| `ff_mult`          | FF 层内部维度扩张倍数（默认是 dim × 4）        |
🧠 类的整体作用
将输入的图像 patch/fMRI token 加上位置信息；
使用 latent token 作为 Query 与输入 Cross-Attend；
反复堆叠多层 attention + FF；
输出一组高质量 latent token 表征，作为语义摘要；

'''

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_latents = 64,
        num_media_embeds = 4,
        ff_mult = 4
    ):
        '''
         self.latents = nn.Parameter(torch.randn(num_latents, dim))
初始化一组 latent tokens，维度 [num_latents, dim]，作为可学习参数。
这些 tokens 是 Query 的来源；

        self.media_pos_emb = nn.Parameter(torch.randn(num_media_embeds, 1, dim))
为多模态输入设置 可学习的位置编码（positional embedding）。
维度 [num_media_embeds, 1, dim]；
假设每个输入模态（图像、fMRI）都带多个片段/帧，可以为每帧添加唯一的位置信息；
会在 forward 中加到输入 token 上。

self.layers = nn.ModuleList([])
创建一个空的模块列表，用于存放多层 cross-attention + FF 层。

  for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))
为每一层构建：
一个 PerceiverAttention 层（Cross-Attention）
一个 FeedForward 层
最终形成一个深度为 depth 的堆叠结构，每层先 attention 后 FF。

  self.norm = nn.LayerNorm(dim)
在最后输出前加一个 LayerNorm，帮助稳定 latent token 表示。
        '''
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.media_pos_emb = nn.Parameter(torch.randn(num_media_embeds, 1, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = nn.LayerNorm(dim)
    '''
    接收输入的媒体特征序列（如图像 patch token 或 fMRI token），加上位置编码，与 latent tokens 执行多层 Cross-Attention + FeedForward 操作，输出融合后的一组 latent 表征（语义摘要）。
    
    def forward(self, x):
定义前向传播方法，输入：
x：媒体输入，shape 通常为 [B, T, D] 或 [B, M, T, D]（如多段 fMRI/图像 patch）

 if x.ndim == 3:
        x = rearrange(x, 'b n d -> b 1 n d')
📝 说明：
如果输入是 [B, T, D]（单段 token），就加一维变成 [B, 1, T, D]，表示 batch 中每个样本有 1 段输入
统一处理方便后面为每段加位置编码

times = x.shape[1]
        x = x + self.media_pos_emb[:times]
从 self.media_pos_emb 中取出前 times 个媒体位置编码，加到每段 token 上
x 现在是带位置信息的输入，shape 为 [B, M, T, D]（M 是媒体段数）
📌 作用：
保留顺序/结构特征，使模型知道第几段媒体（比如第几帧图像或第几段 fMRI）

   latents = repeat(self.latents, 'n d -> b m n d', b = x.shape[0], m = x.shape[1])
📝 功能：
将 learnable 的 latent tokens 扩展到和输入 x 同样的 batch size 和媒体段数
得到 latent token tensor，shape 为 [B, M, L, D]
也就是说每个段（图像/fMRI）都拥有一套独立的 latent tokens。

  for attn, ff in self.layers:
        latents = attn(x, latents) + latents
        latents = ff(latents) + latents
📝 功能：
    多层堆叠的 Perceiver 模块，每层包含：
    Cross-Attention（attn）
    输入 latent token（作为 Query）和媒体 token（作为 Key/Value）
    得到融合媒体语义后的 latent 表征
    残差连接：+ latents 保留原始表示，防止信息丢失，稳定训练
    FeedForward（ff）
    对 latent token 做 MLP 非线性变换
    再加上残差，进一步 refine 表征
    这个循环会执行 depth 次。

 res = self.norm(latents)
对最终的 latent tokens 做 LayerNorm，标准操作，用于归一化输出，提高稳定性。

  if res.ndim == 4:
        res = res.squeeze(1)
如果输出的维度是 [B, 1, L, D]，说明只有 1 段媒体 → 去掉那一维
输出变成 [B, L, D]，每个样本有 L 个语义 token


    '''
    def forward(self, x):
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')

        times = x.shape[1]
        x = x + self.media_pos_emb[:times]

        latents = repeat(self.latents, 'n d -> b m n d', b = x.shape[0], m = x.shape[1])

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        res = self.norm(latents)

        if res.ndim == 4:
            res = res.squeeze(1)

        return res