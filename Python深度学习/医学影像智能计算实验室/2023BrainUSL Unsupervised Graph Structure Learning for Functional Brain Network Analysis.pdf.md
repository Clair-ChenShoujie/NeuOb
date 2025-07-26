# 1.文献基本信息

| 项目             | 内容                                                                                                                                                     |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **文章主题**       | 无监督图结构学习在功能性脑网络分析中的应用                                                                                                                                  |
| **题目**         | BrainUSL:功能性脑网络分析的无监督图结构学习                                                                                                                             |
| **英文题目**       | BrainUSL: Unsupervised Graph Structure Learning for Functional Brain Network Analysis                                                                  |
| **关键词**        | 功能连接性分析、图结构学习、无监督学习、功能性磁共振成像                                                                                                                           |
| **作者信息**       | 第一作者:Pengshuai Zhang等,通讯作者:Peng Cao和Fei Wang                                                                                                           |
| **期刊名称**       | MICCAI 2023会议论文                                                                                                                                        |
| **期刊类型**       | 会议论文                                                                                                                                                   |
| **发表时间**       | 2023                                                                                                                                                   |
| **参考文献格式**     | Zhang P. et al. (2023) BrainUSL: Unsupervised Graph Structure Learning for Functional Brain Network Analysis. In: MICCAI 2023, LNCS 14227, pp. 205-214 |
| **阅读时间**       | 2024年1月                                                                                                                                                |
| **文章摘要**       | 提出了一个端到端的无监督图结构学习方法BrainUSL,用于从fMRI数据中学习功能性脑网络的结构特征。该方法包含图生成模块和拓扑感知编码模块,并设计了视图一致性和相关性引导的对比正则化。在双相障碍(BD)和重度抑郁症(MDD)的诊断应用中取得了优于现有方法的表现。                  |
| **文章基本结构**     | 1. 引言<br>2. 方法(图生成模块、拓扑感知编码器、目标函数)<br>3. 实验与结果(数据集、分类结果、功能连接分析、疾病关联分析)<br>4. 结论                                                                        |
| **文章结论**       | 1. 提出的BrainUSL方法能够生成稀疏和具有判别性的图结构<br>2. 在BD和MDD诊断任务上优于现有方法<br>3. 能够识别疾病相关的生物标志物并提供疾病关联的证据                                                               |
| **文章结论如何得到印证** | 1. 在NMU数据集上进行了5折交叉验证实验<br>2. 与多个SOTA方法进行了对比<br>3. 进行了消融实验验证各组件的有效性<br>4. 通过可视化分析验证了模型的可解释性                                                             |
| **文章的创新之处**    | 1. 首次尝试使用无监督图结构学习构建功能性脑网络<br>2. 提出了相关性引导的对比损失来建模图之间的相关性<br>3. 提供了疾病可解释性分析和BD/MDD疾病关联分析的新视角                                                             |
| **自己对文章的评价**   | 优点:<br>1. 方法新颖,首次将无监督图结构学习应用于脑网络分析<br>2. 实验充分,包含多个验证角度<br>3. 具有良好的可解释性和实际应用价值<br><br>不足:<br>1. 数据集规模相对较小<br>2. 缺乏与其他无监督方法的对比<br>3. 参数敏感性分析不够充分         |
背景
**脑疾病识别**可以看作是具有细化图结构的**图分类**
GSL（Graph Structure Learning）数据存在**噪声**or**不完整**，依赖人工注释
→
无监督GSL


### 1. 图生成模块 (Graph Generation Module)

**目的**：从fMRI的BOLD信号中生成优化的稀疏图结构

**具体实现**：
- 使用堆叠的卷积层进行BOLD信号特征聚合
- 特征学习公式：E(l+1)f(u) = ∑(U-1)(s=0) E(l)f(u-s) * K(l)(s)
  - E(l)f：第l层的特征
  - K(l)：第l层的卷积核
  - U：BOLD信号在脑区输入x中的元素
- 基于学习到的特征计算节点间的相关性，生成优化图AG


1. 图生成的核心组件 `SigCNN` 类：
```python
class SigCNN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SigCNN, self).__init__()
        # 一维卷积层
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=10)
        self.conv2 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=15)
        # 批归一化层
        self.bn = nn.BatchNorm1d(116, affine=True)
        self.bn2 = nn.BatchNorm1d(116, affine=True)
        
        # 可学习的邻接矩阵
        self.w = nn.Parameter(torch.FloatTensor(116, 116), requires_grad=True)
        torch.nn.init.uniform_(self.w, a=0, b=1)
```

2. 图生成的处理流程：
```python
def forward(self, x):
    # 1. 输入数据归一化
    x = self.bn(x)
    
    # 2. 通过两层一维卷积提取特征
    x = self.conv1(x)
    x = self.conv2(x).reshape(x.shape[0], self.in_channel, -1)
    
    # 3. 计算余弦相似度生成初始图结构
    x = cos_similar(x, x)
    
    # 4. 生成最终的图结构
    w = self._normalize(self.w)
    w = F.relu(self.bn2(w))
    w = (w + w.T) / 2  # 确保对称性
    
    # 5. 计算L1正则化
    l1 = torch.norm(w, p=1, dim=1).mean()
    
    # 6. 将学习到的图结构应用到特征上
    x = w * x
    return x, l1
```

3. 图生成相关的辅助函数：
```python
def _normalize(self, mx):
    # 对邻接矩阵进行归一化
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1 / 2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    mx = mx @ r_mat_inv
    return mx

def cos_similar(p, q):
    # 计算余弦相似度
    sim_matrix = p.matmul(q.transpose(-2, -1))
    a = torch.norm(p, p=2, dim=-1)
    b = torch.norm(q, p=2, dim=-1)
    sim_matrix /= a.unsqueeze(-1)
    sim_matrix /= b.unsqueeze(-2)
    sim_matrix = torch.where(torch.isnan(sim_matrix), torch.full_like(sim_matrix, 0), sim_matrix)
    return sim_matrix
```

图生产模块的工作流程：
1. 输入数据首先经过批归一化处理
2. 通过两层一维卷积提取时序特征
3. 计算特征间的余弦相似度，生成初始图结构
4. 使用可学习的邻接矩阵 W 对图结构进行调整
5. 对邻接矩阵进行归一化和对称化处理
6. 将最终的图结构应用到特征上

### 2. 拓扑感知编码器 (Topology-Aware Encoder)

为什么要使用拓扑感知编码器？

1. **保持图结构信息**：
    - 传统的卷积操作可能会丢失图的拓扑结构信息
    - 拓扑感知编码器通过同时考虑行和列方向的特征，保持了节点间的连接关系
2. **捕获多尺度特征**：
    - 通过多层E2E结构，可以捕获不同尺度的图结构特征
    - 有助于理解节点间的局部和全局关系
3. **增强特征表示**：
    - 双向卷积操作提供了更丰富的特征表示
    - 有助于提取更有区分性的图特征
  
**目的**：捕获图的拓扑信息，通过分层的局部(边)到全局(节点)策略

**包含两个主要操作**：
1. 边聚合(Edge Aggregation)：
   - 使用十字形滤波器(水平和垂直方向)
   - Hg = Eg(A) = ∑∑(A(i,·)wr + A(·,j)wh)
   - wr, wh：水平和垂直卷积核的学习向量
   - M：ROI的数量

2. 节点聚合(Node Aggregation)：
   - 通过1D卷积滤波器聚合相关边的信息
   - Hn = En(Hg) = ∑H(i,·)g wn
   - Hn：节点嵌入
   - wn：学习的滤波器向量

1. **预处理阶段**：
    - 首先通过SigCNN生成图结构
    - 计算图的相似度矩阵和分组标签
2. **特征提取阶段**：
       - 使用E2E编码器提取拓扑特征
    - 通过多层E2E结构捕获多尺度特征
3. **特征转换阶段**：
       - 边级特征转换为节点级特征(e2n)
    - 节点级特征聚合为图级特征(n2g)
4. **分类阶段**：
       - 通过全连接层进行最终的分类
    - 使用softmax得到分类概率
### 3. 目标函数设计

模型使用三个损失函数来约束图结构学习：

1. **稀疏诱导范数**：
   - 使用l1范数移除无关连接
   - 保持生成图的稀疏性

2. **视图一致性正则化**：
   - Lvc = ∑sim(ei, êi)
   - 确保固定图结构AP和可学习图结构AG之间的节点嵌入一致性
   - sim(·,·)为余弦相似度度量

3. **相关性引导的对比损失**：
   - 通过图核方法构建相关矩阵S
   - 使用阈值θ进行二值化处理
   - Lcc = -∑log(exp(sim(ei, ej)/τ) / ∑exp(sim(ei, ej)/τ))
   - τ为温度因子

**最终的优化目标**：
L = Lvc + αLcc + β||AG||1
- α和β为权衡超参数
- 用于平衡各个损失项的贡献

### 4. 下游任务

- 基于生成的图结构和预训练的拓扑感知编码器
- 使用多层感知器(MLP)进行图分类
- 采用交叉熵损失进行优化

### 5. 模型的优势

1. **端到端学习**：从原始BOLD信号直接学习图结构
2. **无监督学习**：不依赖标签信息
3. **可解释性**：能够识别疾病相关的功能连接
4. **灵活性**：可用于不同的脑疾病诊断任务

这个模型通过结合图生成、拓扑特征提取和多重约束优化，实现了对功能性脑网络更准确的建模和分析。其创新点在于采用无监督方式学习图结构，并通过多个损失函数的设计来确保生成的图结构既稀疏又具有判别性。