首先要了解数据的形状大小，怎么用这个数据

# 一、BOLD信号是什么、怎么处理

- **BD_HC.npy**：常规表示静态功能连接矩阵（一般392个样本 × 116 × 116）。
- **BD_HC_label.npy**：标签（一维 392）。
- **BD_HC_sig.npy**：**原始BOLD信号**（ shape：(392, 950, 116) ）

BOLD信号是MRI扫描时采集下来的时间序列信号，原始BOLD信号通常维数为NMT "样本数 × 区域数 × 时间帧"

# 二、搞清楚数据形状

BD_HC.npy这个是邻接矩阵。392个被试，116个脑区，两两对应的功能连接强度。

>矩阵中的每个元素 (i, j) 表示脑区 i 和脑区 j 之间的功能连接强度，通常是通过计算BOLD信号的时间序列相关性（如皮尔逊相关系数）得到的。

![[Pasted image 20250525002212.png|550]]

这个邻接矩阵不是原始数据。
原始数据是NMT `(样本数, 节点数, 时间序列长度)`

BD_HC_label.npy是一维的，392个样本，健康对照组 和 双向情感障碍组
BD_HC_sig.npy有392个样本，950个时间节点，116个脑区
```python
import numpy as np

data = np.load('./datasets/BD_HC.npy')
label = np.load('./datasets/BD_HC_label.npy')
sig = np.load('./datasets/BD_HC_sig.npy')

print(data.shape)    # (样本数, 脑区数, 脑区数)
print(label.shape)   # (样本数,)    或 (样本数, 1)
print(sig.shape)     # (样本数，时间节点，脑区数)

# (392, 116, 116)
# (392,)
# (392, 950, 116)
```


做模型前，要将这些**信号**转换为描述**大脑功能网络结构**也就是上面的功能连接矩阵，作为模型的输入特征

# 三、预处理数据

### 标准化、归一化

### 降噪
### 维度调整
NTM --> NMT
将 X 从 (392, 950, 116) 转换为 (392, 116, 950)。
```python
print('loading data...')
X = np.load(f'./Dataset/new/{source_data}_HC_sig.npy')
X = torch.from_numpy(X).permute([0, 2, 1]).numpy()
Y = np.load(f'./Dataset/new/{source_data}_HC_label.npy')
```


![[BrainUSL.py]]
![[BrainGSL-pytorch.py]]

现在已经有了邻接矩阵（BD_HC.npy）和潜在的特征矩阵 （可以从 BD_HC_sig.npy 提取），接下来需要将这些数据输入 GNN 模型，完成分类任务（例如，区分健康对照组 HC 和双相情感障碍 BD）

# 四、复现BrainUSL

![[train_val_plot.png]]

但是复现效果不好

## 输入
每个受试者的 fMRI 信号和标签

## 预处理
数据转置操作，适配后续卷积操作的需求
NTM --> NMT

## Step1，无监督训练
### 处理1：通过SigCNN模块，动态生成邻接矩阵A
从 BOLD 信号中学习一个优化的图结构，表示大脑区域之间的功能连接，而不是依赖传统的皮尔逊相关系数等预定义方法。

**输入**NMT的fmri，提取特征，计算余弦相似度，学习稀疏邻接矩阵，**输出**图结构A，大小NMM

### 处理2，通过SigCNN和E2E模块，获得节点特征矩阵X
**输入**：SigCNN 输出的邻接矩阵 A
通过十字形卷积捕获拓扑信息，然后提取特征
**输出**：节点特征矩阵X，形状为 (N, 116, 48)

损失函数
- 视图一致性损失：确保不同视图（例如原始信号和生成的图结构）的特征表示一致。
- 对比损失：根据图结构相似性定义正负样本对，增强特征的区分性。
- L1 稀疏损失：控制 A 的稀疏性。

生成高质量的 A 和 X

## Step2，监督微调

**输入**：与预训练相同的 fMRI 时间序列数据 (N, T, M)，以及对应的标签 (N,)。
用交叉熵损失，衡量模型预测概率与真实标签之间的差异。
**输出**：每个受试者的分类概率

代码如下
怎么**用**数据进行训练？详见Step6和Step9
![[BrainUSL_run.ipynb]]
