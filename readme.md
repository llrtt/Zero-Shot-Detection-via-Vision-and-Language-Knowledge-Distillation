# Zero-Shot Detection via Vision and Language Knowledge Distillation论文复现代码
此工程用于复现论文Zero-Shot Detection via Vision and Language Knowledge Distillation中的部分效果
## 环境
> Ubuntu 20.04

> pytorch 1.8.1

> clip
## clip模型介绍
论文中用到了openai开源的clip模型作为teacher model做KD(Knowledge Distillation)，clip由网上收集而来的4亿image-text pairs训练而成，意在将image以及其对应的text分别通过image-encoder和text-encoder映射到一个相互匹配的空间(embedding)，其中image-encoder负责用image生成region-embedding,text-encoder负责将带有标签的句子生成text-embedding，最后将两个embedding求余弦相似度再进行softmax得到各类的概率。
![CLIP 图标](imgs/CLIP.png)
```cpp
result = (region_embedding @ text_embedding.T)/(torch.linalg.norm(region_embedding, ord=1)*torch.linalg.norm(text_embedding, ord=1))
result = nn.functional.softmax(result)
```

## 论文复现过程
### Faster_RCNN training
论文中采用的是Mask_RCNN来生成proposals
![Faster_RCNN 图标](imgs/faster_rcnn.jpg)
而本次复现使用的是用ResNet-50-FPN作为backbone的单类Faster_RCNN，Faster_RCNN用修改过的pytorch源码实现，并且用ROIalign替代ROIpooling(排除了ROIpooling因量化而对bbox regression造成的误差)，没有用到Mask_RCNN中的FCN支路，在保证检测效果的同时又减少了训练成本。训练时，为达到class-agnostic的效果，所有背景的label设置为0，其它所有类为1。

### Crop regions && Get image_embedding
目标由Faster_RCNN定位,提取ROIalign之后的proposals和features，**直接提取出来的proposals和features还不能直接用，要经过比例调整、边界限制，再筛除较小的proposal和features**，再将proposals和1.5倍大小的proposals经过crop和resize后输入CLIP模型的image_encoder中获得两种image_embdding，两种image_embdding进行相加后归一化得到最后的image_embdding，

```cpp
image_embedding = model.encode_image(proposal)
image_embedding = image_embedding/torch.linalg.norm(image_embeddings, ord=1, dim=2)
```
crop和resize操作由clip.load()函数返回的preprocess函数进行，用到1.5倍大小的proposals是因为其含有更多的信息，但是由于显存限制，这里只用到原尺寸的proposals。

### Generate text_embedding
首先用将训练的类别与'a photo of a {类名}'组合输入text_encoder获得text_embedding(每一个类对应一个text_embedding)，text_encoder本质是一个transformer模型，用于将sentence映射到高维的空间，寻找句子中词之间的联系，**由于CLIP模型中没有'backgound'对应的数据，故代码里用了一个1x512的可训练tensor代替，再加入text_embedding中一起计算,用nn.Parameter()来确保backgound向量可以被当作参数保存。**
```cpp
self.background = nn.Parameter(torch.rand(1, 512))
```
### Get proposals label
为计算text_loss，我们需要知道每一个proposals对应的label，论文中关于这一部分只是一笔带过，没有过多赘述，我的第一反应是计算proposal和标注的boundingbox之间的IOU，当IOU超过一定阈值则是为proposal中含有此类目标，但是训练时发现模型难以收敛。一番思考后，我发现用IOU确定label的不恰当之处，当proposal很大而boundingbox非常小时，IOU是很小的，就算proposal完整包含了目标，也会当作不包含，这不符合分类的原则(text_embedding的作用就是分类)，所以我采用了另外的方式来确定label。
```cpp
iot = proposal和bbox相交面积/bbox面积
```
当iot超过一定阈值则视为包含当前目标，这样就可以避免上面描述的情况，最终测试得用IOT作为阈值的模型可以收敛。

### Calculate text_loss
得到text_embedding之后，为了找到可以在高维空间中与之对应的region_embedding，故将上一步中的features输入自行搭建的网络中，得到输出为1x512的向量(text_embedding以及image_embedding的维度也为1x512)。而为了计算text_embedding和region_embedding的相似度，论文里面采用了计算余弦相似度的方法，其中两个向量的余弦相似度越大代表两向量越相似，**下面用sim指代余弦相似度**。将单个feature和所有的text_embedding计算余弦相似度，结果保存成一个向量，将向量整体除于温度T后(**T是Kownleadge Distillation里面的知识，用于改变teacher model中小概率结果对student model的影响程度**)，对向量以及proposals对应的label求交叉熵，便得到了proposal中每一类的概率，其中主要过程的实现代码如下:
```cpp
import torch
Zr = sim(region_embedding, text_embeddings)
loss_t = torch.nn.CrossEntropyLoss()
loss = loss_t(Zr/T, torch.nonzero(label)#torch.nonzero()用于确定label的位置
```

### Calculate image_loss
image_loss的计算相对简单，直接计算image_embedding和region_embeddin相减的一范数即可

### Train VILD
训练的时候将数据分为两类，base类和novel类，base类用于计算text_loss和image_loss，novel类则只用于计算image_loss，因为计算image_loss不需要label，而最后训练好的模型可以检测到novel类里面的目标，也就体现了zero-shot-learning的特性

## Code example
'''
'''
