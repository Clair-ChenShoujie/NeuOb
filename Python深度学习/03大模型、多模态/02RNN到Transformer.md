
> [!NOTE] 理解RNN和LSTM
> https://blog.csdn.net/v_JULY_v/article/details/89894058
> 
> 这个博主的其他文章总结：
> https://blog.csdn.net/v_JULY_v/article/details/6543438?fromshare=blogdetail&sharetype=blogdetail&sharerId=6543438&sharerefer=PC&sharesource=Annnclaire&sharefrom=from_link
# 1. RNN原理
某些场景中，单个输入不足以满足业务需求，处理**序列数据**需要RNN：

传统神经网络的结构比较简单：输入层 – 隐藏层 – 输出层
f(x)  = wx + b

![[2907c24280f0e922bb818c4fcc21673f.png|225]]
RNN**每次都会将前一次隐藏层的输出结果，带到下一次的隐藏层中，一起训练**。
![[Pasted image 20250722152116.png|108]]
当前隐藏层输出：![[Pasted image 20250224152742.png]]

短期的记忆影响较大，长期的记忆影响很小
如果一条序列足够长，那它们将很难将信息从较早的时间步传送到后面的时间步。

# 2. LSTM
这个是RNN![[Pasted image 20250224165436.png|450]]
这是LSTM![[Pasted image 20250224165641.png|425]]
LSTM的关键就是细胞状态，只有一些少量的线性交互。信息在上面流传保持不变会很容易。

## 1. 忘记门
输入：上一个输出$h_{t-1}$，和当前输入$x_{t}$
操作：$Sigmoid$ 的非线性映射
输出：向量$f_{t}$表示“忘记程度”
![[Pasted image 20250224170636.png]]
## 2. 输入门
$i_t$：
输入：上一个输出$h_{t-1}$，和当前输入$x_{t}$
操作：$Sigmoid$ 的非线性映射
输出：$i_t$表示输入程度

$\tilde{C}_t$
输入：上一个输出$h_{t-1}$，和当前输入$x_{t}$
操作：$tanh$ 的非线性映射
输出：$\tilde{C}_t$表示候选输入信息
![[Pasted image 20250224172122.png]]
## 3. 更新细胞状态
![[Pasted image 20250224171944.png]]

## 4.输出门
$sigmod$函数得到输出的隐藏层权重
本层输出经过tanh函数，点乘权重，得到$h_t$隐藏层输出

![[Pasted image 20250224172746.png]]
# 3. 补充知识
[[NNLM和Word2Vec]]

> [!NOTE] word2vec就是典型的word embedding，最终输出是固定维度的词向量
> 
> 词嵌入就是将复杂的高维的word和映射到低维的向量

针对Seq2Seq序列问题（输入序列映射到输出序列），比如翻译一句话，可以通过Encoder-Decoder模型来解决。
# 4. 注意力机制
通过计算相似性得出权重最后加权求和

query：当前任务的向量，可以理解为需求
key：每条数据的特征
value：特征对应的实际的值
要查询的query和key求相似度，点积或者余弦相似度，得到每个key的权重
通过 SoftMax 将权重转换为概率分布，确保权重和为 1。
根据权重对 Value 进行加权求和，得到注意力的最终输出。