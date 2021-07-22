# Zero-Shot Detection via Vision and Language Knowledge Distillation论文复现代码
此工程用于复现论文Zero-Shot Detection via Vision and Language Knowledge Distillation中的部分效果
## 环境
> Ubuntu 20.04

> pytorch 1.8.1

> clip
## clip模型介绍
论文中用到了openai开源的clip模型作为teacher model做KD(Knowledge Distillation)，clip由网上收集而来的4亿image-text pairs训练而成，意在将image以及其对应的text分别通过image-encoder和text-encoder映射到一个相互匹配的空间(embedding)，其中image-encoder负责用image生成region-embedding,text-encoder负责将带有标签的句子生成text-embedding，最后将两个embedding求余弦相似度再进行softmax得到各类的概率。
```cpp
result = (region_embedding @ text_embedding.T)/(torch.linalg.norm(region_embedding, ord=1)*torch.linalg.norm(text_embedding, ord=1))
result = nn.functional.softmax(result)
```

## 论文复现过程
### crop regions
当目标由一个two stage detector定位之后（代码中使用的是用ResNet-50-FPN作为backbone的单类Faster rcnn）