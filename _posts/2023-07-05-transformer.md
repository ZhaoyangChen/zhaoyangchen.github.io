---
title: Transformer：自然语言处理领域的变形金刚
tags: [自然语言处理，算法]
article_header:
  type: cover
---
Transformer是一种基于注意力机制的神经网络架构，被广泛应用于自然语言处理任务，图像领域。是近年来很多令人惊艳的大模型的基石结构之一。与传统的循环神经网络（RNN）相比，Transformer采用了注意力机制，使其能够更好地捕捉输入序列中的长距离依赖关系。
<!--more-->

### 1. Transformer 的由来
Transformer 是一种神经网络架构，由 Vaswani 等人于 2017 年在论文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 中首次提出。该论文引入了一种全新的模型结构，用于解决自然语言处理中的序列到序列（sequence-to-sequence）任务，如机器翻译。

Transformer 一词常被直译为 "变换器"，有些地方也被翻译为"变形金刚"。考虑到Transfomer的强大以及在自然语言等领域的重要里程碑地位，我更偏爱称之为 "变形金刚"。

在之前的序列到序列任务中，循环神经网络（Recurrent Neural Networks, RNN）是主要的模型架构。然而，RNN 在处理长距离依赖关系时存在困难，且在并行计算方面表现较差。为了克服这些限制，Transformer 提出了一种全新的方式来编码和解码序列信息，主要基于注意力机制。

### 2. 尝试了解Transformer

初学者在想要了解Transformer结构时，高冷的专家们常常会直接建议初学者直接阅读[Attention Is All You Need](https://arxiv.org/abs/1706.03762)，甚至直接抛出论文中的这张图：

<img src="/assets/images/articles/transformer_model_arch.png" alt="transformer model arch" width="500"/>

虽然这张图画的非常清晰，经常被各种文章引用。但仅仅是用来解释Transformer的架构，并不涉及细节原理。大部分初学者盯着这张图看再久，也无法对Transformer有一个透彻的认识。

接下来，本文尽量用通俗的语言介绍Transformer，尽量做到，即使是仅有人工神经网络前置知识的读者也能看懂。

### 3. Transformer VS RNN

NLP领域很长一段时间都在被RNN（循环神经网络）类模型所主导。有关于RNN的细节本文不再赘述，后续我会单独写一篇文章介绍RNN。

这里仅罗列RNN的主要问题：
* **难捕捉语义长期依赖**：当序列很长时，信息传递和梯度更新在时间上会变得非常困难，这导致了长期依赖关系难以捕捉的问题。尽管RNN存在LSTM这样的变体，但是在捕捉长期依赖方面仍然逊色于Transformer。
* **无法并行**：RNN以顺序方式处理序列数据，一次处理一个元素。这限制了并行计算的能力，因为下一个元素的计算依赖于前一个元素的计算结果。这也使得RNN在处理长序列时效率较低。

### 4. Transformer 中 数据是怎样流动的

这里，以NLP领域常见的翻译问题为例。

假设我们需要做中文到英文的翻译，需要将中文序列"我爱你"翻译为"I love you"。


#### 4.1 word embedding

第一步需要将中文序列映射成向量，因为中文是无法直接进行数学运算的，只有将中文以某种数学形式表示之后，才能够被当作输入进入神经网络。这一步就是我们常说的 word embedding。

word embedding 的方式有很多，包括




















