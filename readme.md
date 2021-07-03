# Zero-Shot Detection via Vision and Language Knowledge Distillation论文复现代码
此工程用于复现论文Zero-Shot Detection via Vision and Language Knowledge Distillation中的部分效果
## 环境
> Ubuntu 20.04

> pytorch 1.8.1

> clip
## clip模型介绍
论文中用到了openai开源的clip模型作为teacher model做KD(Knowledge Distillation)，clip由网上收集而来的4亿image-text pairs训练而成，意在将image以及其对应的text分别通过image-encoder和text-encoder映射到一个相互匹配的空间(embeding)，其中image-encoder