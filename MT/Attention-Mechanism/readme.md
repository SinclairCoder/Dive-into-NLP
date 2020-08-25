
# 机器翻译中的Attention机制

Neural Machine Translation By Joint Learning to Align and Translate   ICLR 2014


代码是基于Dive into DL Pytorch的复现，更多请参见：https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter10_natural-language-processing/10.12_machine-translation


# Abstract
与传统的统计机器翻译有所不同，神经机器翻译致力于构建一个神经网络，可以共同调整以最大化翻译性能。最近提出的神经机器翻译模型都是属于Encoder-Decoder系列，将源语言句子编码成为固定长度的向量，然后解码生成目标语言的翻译结果。在这篇论文中我们猜测固定长度的向量表示是提升基础encoder-decoder架构性能的瓶颈，为此，作者允许模型自行搜索与目标词汇相关的上下文，即Attention机制。使用这个新方法，作者在英法翻译任务上达到了SOTA。通过定性分析发现，模型发现的对齐方式符合我们的直觉。
# 1. Introduction


首先介绍了神经机器翻译的提出，不像传统的基于短语的机器翻译一样，包含了很多可以单独调整的小组件，神经机器翻译尝试去构建一个神经网络模型，输入一个句子，输出一个正确的翻译句子。
大部分提出的神经机器翻译模型都是属于Encoder-Decoder系列，整个encoder-decoder系统包含一个针对语言对的编码器和解码器，二者一起训练来最大化给定的源语言句子产生对应正确的翻译句子的概率。
有个潜在的问题是需要将源语言的句子的信息整个压缩到一个固定长度的向量，但这对长句子来说可能有些困难，Cho, K等学者发现随着输入句子长度的增加，基础的encoder-decoder架构的性能会急速下降。
为了解决这个问题，作者介绍一种encoder-decoder的扩展——共同学习去对齐和翻译，在翻译的过程中每次生成一个单词，模型将会搜索源语言句子中与目标词汇最相关的信息，然后模型基于上下文向量预测目标词汇。
作者提出的方法与之前的方法最大的区别就是不会去尝试去把整个句子编码成固定长度的句子，相反把源语言句子编码成一系列向量，然后在解码的过程中自适应的选择向量的子集去翻译，这对于处理长句子很有效。
在这篇文章中，作者提出的共同学习对齐和翻译的方法性能超过了原有的encoder-decoder方法，这个提升对于长句子是更为明显的。在英法翻译的任务上，提出的方法单模型翻译性能远超传统的翻译模型。更进一步，通过定性分析发现，作者提出的模型在源句子和相应的目标句子之间找到一种语言上合理的（软）对齐方式。


# 2. Background：Nerual Machine Translation


从概率的视角，机器翻译就是已知源语言句子![](https://cdn.nlark.com/yuque/__latex/9dd4e461268c8034f5c8564e155c67a6.svg#card=math&code=x%0A&height=12&width=9)，和目标语言句子![](https://cdn.nlark.com/yuque/__latex/415290769594460e2e485922904f345d.svg#card=math&code=y&height=14&width=8)来![](https://cdn.nlark.com/yuque/__latex/25a2434e779a13b17e120cacf1de9579.svg#card=math&code=argmax_y%20p%28y%7Cx%29&height=21&width=109)
作为一种新方法，神经机器翻译已经表现出不俗的效果，Sutskever等学者发现基于带LSTM单元的RNNs在英法翻译任务上已经超过了传统基于短语的机器翻译的系统，取得了SOTA的效果。


## 2.1 RNN Encoder-Decoder
首先简要的介绍一下RNN Encoder-Decoder这个底层的框架：
编码器读入一个句子，![](https://cdn.nlark.com/yuque/__latex/e4f31c02f0e62895ac3ee27324280dd1.svg#card=math&code=X%20%3D%20%28x_1%2C...x_%7BT_x%7D%29%0A&height=20&width=122),将这个向量变成一个向量![](https://cdn.nlark.com/yuque/__latex/4a8a08f09d37b73795649038408b5f33.svg#card=math&code=c&height=12&width=7)，最常用的方法就是RNN，![](https://cdn.nlark.com/yuque/__latex/3c1d73c185c0c29a7e8d2520f13f64c5.svg#card=math&code=h_t%20%3D%20f%28x_t%2Ch_%7Bt-1%7D%29&height=20&width=114), ![](https://cdn.nlark.com/yuque/__latex/1202ceae08b46089dfeab8b0102cd943.svg#card=math&code=c%20%3D%20q%28%7Bh_1%2C...%2Ch_%7BT_%7D%7D%29&height=20&width=130),其中![](https://cdn.nlark.com/yuque/__latex/15dbcc48b2af8e5733b81c2caa1bf395.svg#card=math&code=f%EF%BC%8Cq&height=24&width=32)都是非线性的激活函数。
解码器通过给定上下文向量![](https://cdn.nlark.com/yuque/__latex/4a8a08f09d37b73795649038408b5f33.svg#card=math&code=c&height=12&width=7)和之前的已经生成的词汇![](https://cdn.nlark.com/yuque/__latex/9a622a0506491d4495286d5a3de475a4.svg#card=math&code=%5C%7By_1%2C...%2Cy_%7Bt%27-1%7D%5C%7D&height=20&width=103)来预测下一时刻的词汇![](https://cdn.nlark.com/yuque/__latex/2acc8c1f901be0c85f72a3e3bba62156.svg#card=math&code=y_t%27&height=20&width=14)，换句话说解码器定义了翻译结果y的概率分布：![](https://cdn.nlark.com/yuque/__latex/547ca5c0dbd73958f3cab1973e1baa77.svg#card=math&code=p%28y%29%20%3D%20%5Cprod_%7Bt%3D1%7D%5E%7BT%7Dp%28y_t%7C%5C%7By_1%2C...y_%7Bt-1%7D%5C%7D%2Cc%29&height=53&width=224)，其中![](https://cdn.nlark.com/yuque/__latex/987e91374f9be1d4af160af18bb4617c.svg#card=math&code=y%20%3D%20%28y_1%2C...%2Cy_%7BT_y%7D%29&height=23&width=119),有了RNN，每个条件概率可以被描述成 ![](https://cdn.nlark.com/yuque/__latex/13a34c15a76c31621b39ef7da970e3e5.svg#card=math&code=p%28y%29%20%3D%20%5Cprod_%7Bt%3D1%7D%5E%7BT%7Dp%28y_t%7C%5C%7By_1%2C...y_%7Bt-1%7D%5C%7D%2Cc%29%20%3D%20g%28y_%7Bt-1%7D%2Cs_t%2Cc%29&height=53&width=333),其中![](https://cdn.nlark.com/yuque/__latex/b2f5ff47436671b6e533d8dc3614845d.svg#card=math&code=g&height=14&width=8)是一个非线性激活函数，输出![](https://cdn.nlark.com/yuque/__latex/a568bf104397bd8311073893dff24222.svg#card=math&code=y_t%0A&height=14&width=14)的概率，![](https://cdn.nlark.com/yuque/__latex/86ad9159785a8f6f1c1a74c4eac26365.svg#card=math&code=s_t&height=14&width=13)是RNN的一个隐层状态。

# 3. Learning to Align and Translate
本节，作者提出了一个新的神经机器翻译的结构，其中包含双向的RNN作为编码器，和一个解码器，解码器在翻译的过程中搜索源语言句子。


## 3.1 Decoder：General Description
在新的模型结构里，我们定义条件概率为：![](https://cdn.nlark.com/yuque/__latex/1cb866fb3555067ae1b96da1f0b0f47a.svg#card=math&code=p%28y_i%7Cy_1%2C...%2Cy_%7Bi-1%7D%2Cx%29%20%3D%20g%28y_%7Bi-1%7D%2Cs_i%2Cc_i%29&height=20&width=255),，其中![](https://cdn.nlark.com/yuque/__latex/e406ac4d7c470823a8619c13dd7101be.svg#card=math&code=s_&height=14&width=13)是RNN的i时刻的隐层状态，![](https://cdn.nlark.com/yuque/__latex/8ada11a7553d9e37d55cd9368b359b8c.svg#card=math&code=s_i%20%3D%20f%28s_%7Bi-1%7D%2Cy_%7Bi-1%7D%2Cc_i%29&height=20&width=144)。跟之前不一样的是这里对于每个目标词汇![](https://cdn.nlark.com/yuque/__latex/8d62e469fb30ed435a668eb5c035b1f6.svg#card=math&code=y_&height=14&width=13)都有一个上下文向量![](https://cdn.nlark.com/yuque/__latex/96fafac0c054b9eb47d3f630ed02c289.svg#card=math&code=c_&height=14&width=12)。
上下文向量其实是注解序列![](https://cdn.nlark.com/yuque/__latex/57339b77b3af15427b7154f4daf8a223.svg#card=math&code=h_i&height=18&width=15)的线性组合：
![](https://cdn.nlark.com/yuque/__latex/0a863ccda0bc1babd8710274a39e5795.svg#card=math&code=c_i%20%3D%20%5Csum_%7Bj%3D1%7D%5E%7BT_x%7Da_%7Bij%7Dh_&height=55&width=98)
其中的权重![](https://cdn.nlark.com/yuque/__latex/9a59294d35ec9247478796ffb89359eb.svg#card=math&code=a_%7Bij%7D&height=16&width=19)的计算方式：
![](https://cdn.nlark.com/yuque/__latex/b76ee5c5ac53a585c3cbfe20cdb4f50f.svg#card=math&code=a_%7Bij%7D%20%3D%20%5Cfrac%7Bexp%28e_%7Bij%7D%29%7D%7B%5Csum_%7Bk%3D1%7D%5E%7BT_x%7Dexp%28e_%7Bik%7D%29%7D&height=52&width=150)，其中![](https://cdn.nlark.com/yuque/__latex/6a773c9ee2234bbc8f0569254be16920.svg#card=math&code=e_%7Bij%7D%20%3D%20a%28s_%7Bi-1%7D%2Ch_j%29&height=21&width=115)

在本文中作者使用输入连结后通过含单隐藏层的多层感知机变换:
![](https://cdn.nlark.com/yuque/__latex/db3b8ea48f629def7e69066068dde82b.svg#card=math&code=a%28s%2Ch%29%20%3D%20v%5ET%20tanh%28W_ss%2BW_hh%29&height=23&width=221)
 其中![](https://cdn.nlark.com/yuque/__latex/9e3669d19b675bd57058fd4664205d2a.svg#card=math&code=v&height=12&width=8),![](https://cdn.nlark.com/yuque/__latex/bc09fe19e1165de9c3bdd48f49ab36a1.svg#card=math&code=W_&height=18&width=23),![](https://cdn.nlark.com/yuque/__latex/9ed4343f2b3bf4b5edb500f877223c18.svg#card=math&code=W_h&height=18&width=24)都是可以学习的模型参数
![](https://cdn.nlark.com/yuque/__latex/6fd64a8eafc5224488e3523dd225bb7b.svg#card=math&code=e_%7Bij%7D&height=16&width=18)是一个对齐模型，衡量输入句子的第![](https://cdn.nlark.com/yuque/__latex/865c0c0b4ab0e063e5caa3387c1a8741.svg#card=math&code=i&height=16&width=5)个词汇与输出结果的第![](https://cdn.nlark.com/yuque/__latex/363b122c528f54df4a0446b6bab05515.svg#card=math&code=j&height=18&width=7)个词汇之间的相关匹配程度，其中![](https://cdn.nlark.com/yuque/__latex/5c4f27766a2cc040879c7abca87af831.svg#card=math&code=s_%7Bi-1%7D%0A&height=14&width=28)是RNN的隐层状态，![](https://cdn.nlark.com/yuque/__latex/7891fa1c2293f9c8b0796c28c083c500.svg#card=math&code=h_j%0A&height=20&width=16)是输入序列第![](https://cdn.nlark.com/yuque/__latex/363b122c528f54df4a0446b6bab05515.svg#card=math&code=j&height=18&width=7)个位置的状态序列。
需要注意的是在该模型中对齐模型和翻译模型一起训练。
![](https://cdn.nlark.com/yuque/__latex/9a59294d35ec9247478796ffb89359eb.svg#card=math&code=a_%7Bij%7D&height=16&width=19)：可以认为是目标词汇![](https://cdn.nlark.com/yuque/__latex/8d62e469fb30ed435a668eb5c035b1f6.svg#card=math&code=y_i%0A&height=14&width=13)从源语言词汇![](https://cdn.nlark.com/yuque/__latex/1f89889020cdc84d9e1c35237cb62f65.svg#card=math&code=x_j%0A&height=16&width=16)翻译或对齐来的概率；
![](https://cdn.nlark.com/yuque/__latex/96fafac0c054b9eb47d3f630ed02c289.svg#card=math&code=c_i&height=14&width=12)：也就是所有状态序列在![](https://cdn.nlark.com/yuque/__latex/9a59294d35ec9247478796ffb89359eb.svg#card=math&code=a_%7Bij%7D&height=16&width=19)为概率下期望；
![](https://cdn.nlark.com/yuque/__latex/9a59294d35ec9247478796ffb89359eb.svg#card=math&code=a_%7Bij%7D%0A%0A&height=16&width=19)或者![](https://cdn.nlark.com/yuque/__latex/6fd64a8eafc5224488e3523dd225bb7b.svg#card=math&code=e_%7Bij%7D&height=16&width=18)反映了源语言状态序列![](https://cdn.nlark.com/yuque/__latex/7891fa1c2293f9c8b0796c28c083c500.svg#card=math&code=h_j&height=20&width=16)和目标语言上一时刻的隐层输出![](https://cdn.nlark.com/yuque/__latex/5c4f27766a2cc040879c7abca87af831.svg#card=math&code=s_%7Bi-1%7D&height=14&width=28)对于产生下一个隐状态![](https://cdn.nlark.com/yuque/__latex/e406ac4d7c470823a8619c13dd7101be.svg#card=math&code=s_i%0A&height=14&width=13)和生成![](https://cdn.nlark.com/yuque/__latex/8d62e469fb30ed435a668eb5c035b1f6.svg#card=math&code=y_i&height=14&width=13)的重要性程度，这就实现了在解码器端的注意力机制。
![](https://cdn.nlark.com/yuque/__latex/ae3673d2708346c99dcb547cb741fe53.svg#card=math&code=%5Calpha%28.%29&height=20&width=31):可以看作是目标语言和源语言表示的一种统一化，即把源语言和目标语言表示映射在同一个语义空间，比如向量乘法、向量夹角、线性模型等等。

**注意力机制的解读：**
> 实际上，目标语位置![](https://cdn.nlark.com/yuque/__latex/363b122c528f54df4a0446b6bab05515.svg#card=math&code=j&height=18&width=7)本质上是一个查询，我们希望从源语言端找到与之最匹配的源语言位置，并返回相应的表示结果，为了描述这个问题，可以建立一个查询系统，里面包含若干key-value单元，每次查询时就把query与key逐个进行匹配，如果匹配成果就返回value，对应到机器翻译中不总是能够完全匹配上，注意力机制就采用了一个“模糊”匹配的方法，对key和value之间定义一个0~1的匹配度，记为![](https://cdn.nlark.com/yuque/__latex/cd0f1069db14b3485b705eb04d3e58a4.svg#card=math&code=%5Calpha_i&height=14&width=16),查询的结果也不是简单的value，而是所有value用![](https://cdn.nlark.com/yuque/__latex/cd0f1069db14b3485b705eb04d3e58a4.svg#card=math&code=%5Calpha_i&height=14&width=16)的加权和，也就是说所有的value都会对查询结果有贡献，只是贡献不同罢了，使用![](https://cdn.nlark.com/yuque/__latex/cd0f1069db14b3485b705eb04d3e58a4.svg#card=math&code=%5Calpha_i&height=14&width=16)来捕捉key和query之间的相关性，相关性越大贡献越大。
> 从统计学的角度，![](https://cdn.nlark.com/yuque/__latex/cd0f1069db14b3485b705eb04d3e58a4.svg#card=math&code=%5Calpha_i&height=14&width=16)作为每个 ![](https://cdn.nlark.com/yuque/__latex/400b5d286500a6bef31912f22e152f6a.svg#card=math&code=value_i&height=18&width=45) 出现的概率的某种估计,新的value就是![](https://cdn.nlark.com/yuque/__latex/400b5d286500a6bef31912f22e152f6a.svg#card=math&code=value_&height=18&width=45)在对应的概率![](https://cdn.nlark.com/yuque/__latex/cd0f1069db14b3485b705eb04d3e58a4.svg#card=math&code=%5Calpha_i&height=14&width=16)下的期望，从这个观点看注意力机制是得到了![](https://cdn.nlark.com/yuque/__latex/2063c1608d6e0baf80249c42e2be5804.svg#card=math&code=value%0A%0A&height=16&width=39)的一种期望。
> 参考：《机器翻译：统计建模与深度学习方法》 肖桐 朱靖波著

## 3.2 Encoder：Bidirectional RNN for Annotating Sequences
以往的RNN，都是从句子第一个符号读到最后一个符号。然而，我们想要让注释不仅囊括之前的信息，还要包含之后的信息，所以我们采用双向RNN。BiRNN包含前向RNN和反向RNN，前向RNN读入序列![](https://cdn.nlark.com/yuque/__latex/b1e427fd2bcf3855ac762cde2dc893f8.svg#card=math&code=x_1...x_%7BT_x%7D&height=16&width=64),得到前向的隐层状态序列![](https://cdn.nlark.com/yuque/__latex/d6799601cfb25ff73019a571a9330563.svg#card=math&code=%28%5Coverrightarrow%7Bh_1%7D%2C...%5Coverrightarrow%7Bh_%7BT_x%7D%7D%29%0A&height=31&width=87),反向的RNN读入序列![](https://cdn.nlark.com/yuque/__latex/8f1cc65d206d7f8a0fd253f4d439fde9.svg#card=math&code=x_%7BT_x%7D...x_1&height=16&width=64),得到前向的隐层状态序列![](https://cdn.nlark.com/yuque/__latex/eee6c6dc18137b72553b9ccae59560ef.svg#card=math&code=%28%5Coverleftarrow%7Bh_1%7D%2C...%5Coverleftarrow%7Bh_%7BT_x%7D%7D%29%0A&height=31&width=88),当对于每个词汇![](https://cdn.nlark.com/yuque/__latex/1f89889020cdc84d9e1c35237cb62f65.svg#card=math&code=x_j%0A&height=16&width=16)产生的状态表示![](https://cdn.nlark.com/yuque/__latex/7891fa1c2293f9c8b0796c28c083c500.svg#card=math&code=h_j&height=20&width=16)包含前向和反向的对于的状态表示。这个状态向量会在对齐模型和解码器中用来计算上下文向量。

# 4. Experiment Settings


## 4.1 Datasets
在英法翻译任务上做评估，使用ACL WMT'14提供的双语并行语料....

## 4.2 Models
作者训练了两种模型，一种是RNN Encoder-Decoder，另一种是RNNsearch(即本文提出的模型)，然后分别在最大句子长度为30和50下进行训练。


RNN Encoder-Decoder 模型结构：
encoder和decoder都是1000个隐层单元的RNN(GRU)

RNNsearch 模型结构：
encoder：1000个隐层单元的BiRNN
decoder：1000个隐层单元


上述两种模型结构都使用一个带maxout的多层的神经网络来计算每个目标词的条件概率
maxout的论文：[Maxout Networks](https://arxiv.org/pdf/1302.4389.pdf)     讲的比较清楚的博客：[Maxout网络学习](https://blog.csdn.net/hjimce/article/details/50414467)
使用批梯度下降算法，使用Adadelta算法进行优化，![](https://cdn.nlark.com/yuque/__latex/92e4da341fe8f4cd46192f21b6ff3aa7.svg#card=math&code=%5Cepsilon&height=12&width=6)= 10−6 and ![](https://cdn.nlark.com/yuque/__latex/d2606be4e0cd2c9a6179c8f2e3547a85.svg#card=math&code=%5Crho&height=16&width=8) = 0.95每个minibatch的大小是80个sentences, ![](https://cdn.nlark.com/yuque/__latex/b2eaac203fb38bc6f0271a2ea71f1ddc.svg#card=math&code=L_2%7B-%7Dnorm&height=18&width=73),训练每个模型大概需要5天。


一旦模型训练好，作者使用集束搜索的算法去近似的最大和条件概率来生产对应的翻译结果。

集束搜索：是一种启发式图搜索算法，通常用在图的解空间比较大的情况下，为了减少搜索所占用的空间和时间，在每一步深度扩展的时候，剪掉一些质量比较差的结点，保留下一些质量较高的结点。这样减少了空间消耗，并提高了时间效率，但缺点就是有可能存在潜在的最佳方案被丢弃，因此Beam Search算法是不完全的，一般用于解空间较大的系统中。

# 5 Results
具体数据省略..


作者提出该方法的动机就是克服固定长度的上下文向量的弊端，并猜测这限制了模型在长句子上的表现，而本文提出的RNNsearch-30和RNNsearch-50均有较好的鲁棒性，即使在一些长度超过50的句子上，仍有不错的表现。

具体的定量分析略..



