### **Team**
团队名称： **VIPLab**
A榜排名： 22
### **Introduction**
我们的整体实现思路与baseline一致，分两阶段先预测骨骼再预测皮肤权重。在复现baseline时我们发现骨骼的预测结果差强人意，从而影响后续皮肤权重的预测。所以我们首要关注的目标就是骨骼预测任务，通过更换更强大的模型，数据增强，增加先验以及约束来优化这个任务。
##### Skeleton Model
Skeleton Model是我们主要改动的部分，分为Encoder和Decoder两部分。
我们使用[3DShape2Vec[1]](http://arxiv.org/abs/2301.11445) 作为Encoder，以获取鲁棒的潜在三维模型表征latent。受到[DETR[2]](http://arxiv.org/abs/2005.12872) 的启发，Decoder部分我们使用NumJoints个可学习向量作为Query，以latent作为KV，进行交叉注意力，最后通过一个final layer得到最后的结果。
根据实验结果，我们还进行了以下优化：
+ 增加点云顶点的采样数，增强三维空间结构输入。
+ 在数据处理中添加`随机缩放`、`随机旋转`和`随机姿态变换矩阵`三种数据增强方法，以增强对少量奇怪姿态的理解能力。
+ 在训练过程中除了`MSE`和`J2J`约束外我们还添加了`骨长对称性约束`、`平面对称性约束`、`拓扑一致性约束`、`跟关节位置约束`和`模型内部约束`等解剖约束和先验约束，详细参数可以在训练代码中查看。
##### Skin Model
经过上述步骤后，我们可以获得鲁棒的骨骼预测结果。Skin Model采用和baseline一致的训练方法，沿用GT输入（我们尝试过增加随机噪声扰动以模拟预测骨骼的误差，但是效果不是很好，仍需要调整），PCT[3] 作为模型框架的策略，我们简单调整了训练参数，数据增强方法同上。
##### DATA
除此以外，我们依次人工浏览了训练集的数据，挑选出了极少数难以拟合的样本（27个，约占0.5%），我们在训练时丢弃了这些bad case。

### 代码结构
``` txt
.
|-- checkpoint      # 提交的checkpoint 
|-- data            # 数据集
|-- dataset       
|   |-- asset.py  
|   |-- dataset.py  # 新增数据增强方法，包括随机缩放、随机旋转、随机姿态变换
|   |-- ...
|   `-- utils.py
|-- launch
|-- models
|   |-- PCT                # PCT相关代码实现
|   |-- metrics.py         # 新增对称约束、骨长对称约束、拓扑一致性约束、跟关节约束、模型内部约束
|   |-- pointnet_ops.py    # jittor FPS采样cuda加速实现
|   |-- sal_perceiver.py   # 3DShape2Vec jittor实现
|   |-- skeleton.py        #　skeleton model 新增sal方法
|   |-- skin.py            # skin model
|   `-- transformers.py    # Transformer MHSA CrossAttention 等相关操作实现
|-- scripts                
|   |-- pipeline.sh          # 训练脚本
|   |-- predict_skeleton.sh  # 骨骼预测脚本
|   `-- predict_skin.sh      # 皮肤权重预测脚本
|-- predict_skeleton.py
|-- predict_skin.py
|-- train_skeleton.py
|-- train_skin.py
|-- drop_list.txt         # 人工筛选排除掉数据集中的bad case使训练更稳定
|-- requirements.txt
|-- Dockerfile
|-- LICENSE
`-- README.md
```

### 环境配置：
1. 安装NVIDIA Container Toolkit
2. 当前用户加入docker组, 命令 `sudo usermod -aG docker $USER`
3. 刷新，命令：`newgrp docker`
#### Docker：
##### Docker Bulid 
**(展示我们的创建过程，无需执行)**
```bash
bash dockerbuild.sh
```
`Dockerfile`和`dockerbuild.sh`可以在我们提供的docker项目文件中找到。

##### 加载镜像：
```bash
docker load -i contest2_viplab_22.tar.gz
```
##### 容器启动
同时挂载数据到容器
数据集即比赛官方提供的[数据集](https://cloud.tsinghua.edu.cn/f/676c582527f34793bbac/?dl=1),可以由此下载解压缩到本地。
```bash
docker run --gpus all -it --rm  -v  [path/data]:/workspace/project/data contest2_viplab_22:submit
# 示例：
docker run --gpus all -it --rm  -v /home/hxgk/MoGen/jittor-comp-human/data:/workspace/project/data contest2_viplab_22:submit
```
或者
```bash
docker run --gpus all -it  --rm contest2_viplab_22:submit
# 再添加data到 /workspace/project 即 workspace/project/data: 命令如下：
docker cp /path/to/local/file <container_id>:/path/in/container/
```
进入容器后项目路径位于`/workspace/project/`，进入容器后默认位于该目录下。checkpoint位于`checkpoint/`目录下，同时其中包含我们的训练记录log，需要运行的脚本位于`scripts/`目录下。

### 运行步骤
#### Train
```bash
bash scripts/pipeline.sh
```
训练过程记录以及结果将保存在`output/`目录下。
我们的训练日志可以在`checkpoint/`目录下找到。
**参数说明：**
数据集参数
`--num_samples`: 模型的采样数，包含顶点采样和面采样
`--vertex_samples`: 采样器在顶点上的采样数
`--rotation_range`:数据增强随机旋转的角度
`--scaling_range`:数据增强随机缩放的比例包含两个值分别为最小和最大值
`--aug_prob`:每个sample被数据增强的概率，默认0.5，每种数据增强方式独立判断
`--drop_bad`:是否丢弃bad case
`--pose_angle_range`: 数据增强随机姿态变换的角度，随机作用在每个骨骼关节处

模型参数
`--model_name`: 模型类型 新增`sal`选项
`--wnormals`: 是否法向量作为embeding，仅`sal`
`--num_tokens`: 潜空间的token数，仅`sal`
`--feat_dim`: 模型的默认维度
`--encoder_layers`: 使用的Transformer encoder层数，仅`sal`
`--pct_feat_dim`: PCT模型的默认维度，区别于`feat_dim`，仅`pct` `pct2`

训练参数
`--batch_size`: 批大小
`--optimizer`: 优化器类型，新增`adamw` 
`--learning_rate`: 学习率
`--lr_scheduler`: 学习率更新策略，包含`step`周期性衰减和`cosine`余弦退火学习率调度策略
`--lr_min`: 最小的学习率，仅`consine`
`--sym_loss_weight`: 对称约束权重占比，默认0.05
`--bone_length_symmetry_weight`: 骨长对称性约束占比，默认0.5
`--J2J_loss_weight`:J2J约束权重占比，默认1.0
`--topo_loss_weight`:拓扑一致性约束权重占比，默认0.1
`--rel_pos_loss_weight`:跟关节一致性约束权重占比，默认0.1
`--mesh_interior_weight`:模型内部约束权重占比，约束预测的骨骼位于mesh模型内部，默认0.5
`--interior_margin`: 模型内部约束的边缘惩罚项，使骨骼预测结果远离mesh模型边缘，默认0.01
`--terminal_interior_loss`: 是否取消骨骼终端约束，即是否将模型内部约束应用于双手和双脚
`--use_normals_interior`: 是否使用法向量计算的方法计算模型内部约束
`--interior_k_neighbors`: 指定计算内部约束的K个最近邻居数，默认50
更多参数可以在`train_skeleton.py`和`trian_skin.py`代码中查找定义。
#### Inference
+ 预测骨架
```bash
bash scripts/predict_skeleton.sh
```
+ 预测权重
```bash
bash scripts/predict_skin.sh
```
推理结果将保存在`predict/`目录下。

### Checkpoint 说明
我们提供当前榜上排名最优(rank 22)的checkpoint，包含skeleton和skin两个模型分别位于`checkpoint/skeleton` 和 `checkpoint/skin`下，另分别附我们的训练log。
+ skeleton model best训练500轮，验证集最优出现在470轮，验证集mse Loss: 0.0016 J2J Loss: 0.0162
+ skin model best训练1000轮，验证集最优出现在858轮，验证集mse: 0.0044 l1: 0.0115

基于下面出现的不确定性情况，我们另外提供我们当前验证集上最好性能的骨骼模型于`checkpoint/skeleton_best` ，另附训练log。
+ skeleton_best model best训练500轮， 验证集最优在493轮，验证集mse Loss: 0.0011 J2J Loss: 0.0147
更多训练细节可以在我们的训练log中查看。
### 其他补充
由于我们的模型中包含随机过程，具体参见`models/sal_perceiver.py line 186-190`，以及随机Dropout和BatchNorm操作，就在最近我们发现预测代码中没有调用`model.eval()`方法，尽管我们固定了随机种子，但是预测过程中仍存在随机过程。基于上述原因我们未能完全复现系统中提交的预测结果，在取消`model.eval()`的情况下，每次运行的结果存在一定差异。现在我们提供的代码中已经修复这个问题，每次预测的结果会保持一致，但是预测结果可能和提交的代码存在差异，由于提交系统已经关闭我们无法验证当前的效果，对此我们表示歉意。您也可以通过注释`predict_skeleton.py line 34`以及`predict_skin.py line line 34`来进行随机测试。
我们同时提供当前可能效果更好的skeleton模型于`checkpoint/skin_best/`目录中同时附带训练过程的log，您也可以通过调整`scripts/predict_skeleton.sh`以及`scripts/predict_skin.sh`中被注释掉的部分来预测。

### Reference
[1] Zhang B, Tang J, Niessner M, et al. 3dshape2vecset: A 3d shape representation for neural fields and generative diffusion models[J]. ACM Transactions On Graphics (TOG), 2023, 42(4): 1-16.
[2] Carion N, Massa F, Synnaeve G, et al. End-to-end object detection with transformers[C]//European conference on computer vision. Cham: Springer International Publishing, 2020: 213-229.
[3] Guo M H, Cai J X, Liu Z N, et al. Pct: Point cloud transformer[J]. Computational visual media, 2021, 7(2): 187-199.
