from matplotlib import image
from torch import nn
from torch.nn.modules.conv import Conv2d
from torch.utils.data.dataset import T
from torchvision.models.detection import backbone_utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import dataloader
import numpy as np
import vocDataset
import clip
import torch
import Modified
import faster_rcnn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import nms
import utils

device = "cuda" if torch.cuda.is_available() else "cpu"
train_class = 10


class VILD(nn.Module):
    def __init__(self):
        super(VILD, self).__init__()

        self.backbone = faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=True)
        self.backbone.box_nms_thresh = 0.9
        self.backbone.roi_heads.box_predictor = Modified.FastRCNNPredictor(
            1024, 2)
        self.backbone.load_state_dict(torch.load("maskrcnn5.pt"))

        box_head = Modified.TwoMLPHead(
            self.backbone.backbone.out_channels *
            self.backbone.roi_heads.box_roi_pool.output_size[0] ** 2,
            1024)

        box_head.fc6 = self.backbone.roi_heads.box_head.fc6
        box_head.fc7 = self.backbone.roi_heads.box_head.fc7

        self.backbone.roi_heads.box_head = box_head
        self.backbone.eval().to(device)

        self.region = backbone_utils.resnet.resnet18()
        self.region.conv1 = nn.Conv2d(256, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.region.fc = nn.Linear(512, 512)
        self.region.to(device)

        self.background = nn.Parameter(torch.rand(1, 512))
        self.background.requires_grad = True

    def VILD_image(self, images, model, preprocess):

        # 获得变换之后的图片和原图的尺寸比例
        image_sizes = torch.tensor(
            self.backbone.transform(images)[0].image_sizes)
        x_ratio = torch.tensor([image.shape[1] for image in images]) / \
            torch.tensor([image_size[0] for image_size in image_sizes])
        y_ratio = torch.tensor([image.shape[2] for image in images]) / \
            torch.tensor([image_size[1] for image_size in image_sizes])
        x_ratio = x_ratio.unsqueeze(1).repeat(
            [1, self.proposals.shape[1]]).to(device)
        y_ratio = y_ratio.unsqueeze(1).repeat(
            [1, self.proposals.shape[1]]).to(device)

        # 将proposal变换为原来的图片对应的尺寸及1.5倍原来尺寸
        self.proposals[:, :, 0] = self.proposals[:, :, 0] * x_ratio
        self.proposals[:, :, 1] = self.proposals[:, :, 1] * y_ratio
        self.proposals[:, :, 2] = self.proposals[:, :, 2] * x_ratio
        self.proposals[:, :, 3] = self.proposals[:, :, 3] * y_ratio

        # proposals1_5 = proposals*1.5
        for image, i in zip(images, range(self.proposals.shape[0])):
            self.proposals[i][:, 0] = self.proposals[i][:, 0].clamp(
                0, image.shape[2]-2)
            self.proposals[i][:, 1] = self.proposals[i][:, 1].clamp(
                0, image.shape[1]-2)
            self.proposals[i][:, 2] = self.proposals[i][:, 2].clamp(
                0, image.shape[2]-2)
            self.proposals[i][:, 3] = self.proposals[i][:, 3].clamp(
                0, image.shape[1]-2)
        self.proposals = self.proposals.long()


        # 将变换后的proposals输入clip获得50x512的image_embedding
        postprocess = transforms.ToPILImage()
        image_embeddings = torch.empty(
            len(images), self.proposals.shape[1], 512)

        for image, j in zip(images, range(len(images))):
            for i in range(self.proposals.shape[1]):
                image_input = preprocess(postprocess(
                    image[:, self.proposals[j, i, 1]:self.proposals[j, i, 3], self.proposals[j, i, 0]:self.proposals[j, i, 2]])).unsqueeze(0).to(device)
                image_embedding = model.encode_image(image_input)
                image_embeddings[j, i] = image_embedding

        # 归一化
        v = torch.linalg.norm(image_embeddings, ord=1,
                              dim=2).unsqueeze(2).repeat([1, 1, 512])
        image_embeddings = image_embeddings / v

        return image_embeddings

    def get_label(self, proposal, target, image):
        """
        计算每个proposal的标签
        """
        text_label = torch.zeros([1, train_class+1])
        for i in range(len(target['labels'])):
            if nms.iot(proposal.to('cpu'), target['boxes'][i]) >= 0.3:
                text_label[0, target['labels'][i]+1] = 1
        return text_label

    def VILD_text(self, model):
        """
        生成text_embedding，训练时只用到VOC数据集中前十个类
        """
        if self.training:
            text_inputs = torch.cat(
                [clip.tokenize(f"a photo of a {c}") for c in vocDataset.classes[:train_class]]).to(device)
        else:
            text_inputs = torch.cat(
                [clip.tokenize(f"a photo of a {c}") for c in vocDataset.classes[:]]).to(device)
        text_embedding = model.encode_text(text_inputs)
        return text_embedding

    def sim(self, a, b):
        """
        计算两向量的余弦相似度(cosine simarility)
        """
        return (a @ b.T)/(torch.linalg.norm(a, ord=1)*torch.linalg.norm(b, ord=1))

    def Loss_text(self, text_embedding, region_embedding, targets):
        """
        计算VILD_text的loss，其中background_embedding为1x512的可训练tensor
            Arguments:
                text_embedding:由text_encoder输出
                region_embedding:由经过修改的resnet_18输出，输入为prposals
                tergets:prposals对应的标签
            Return:
                losses:多个proposals的region_embedding和text_embedding之间的损失
        """
        # 将backgound_embedding和text_embedding拼接，方便计算
        text_embedding = torch.cat([self.background, text_embedding], dim=0)
        losses = 0

        for regions, i in zip(region_embedding, range(targets.shape[0])):
            for region, j in zip(regions, range(targets.shape[1])):
                # 计算余弦相似度
                Zr = self.sim(region.unsqueeze(0), text_embedding.float())
                print("region:{}".format(nn.functional.softmax(Zr)))
                print("label:{}".format(targets[i, j]))
                loss_t = nn.CrossEntropyLoss()
                if(torch.nonzero(targets[i, j]).shape[0] == 0):
                    losses += 0.2*loss_t(Zr/5, torch.tensor([0]).to(device))
                else:
                    losses += loss_t(Zr/5,
                                     torch.nonzero(targets[i, j])[0].to(device))
        return losses

    def Loss_image(self, image_embedding, region_embedding):
        """
        计算VILD_image loss,其中image_embedding经过了归一化，但是region_embedding没有，所以要进行归一化之后才能进行相减求一范数
        """
        v = torch.linalg.norm(region_embedding, ord=1,
                              dim=2).unsqueeze(2).repeat([1, 1, 512])
        loss = image_embedding.to(device) - region_embedding/v
        return torch.sum(torch.linalg.norm(loss, ord=1, dim=2))

    def filter_proposals(self, proposals, region_embedding):
        """
        筛选掉部分proposals
        """
        pro = []
        reg = []
        width = self.proposals[:, :, 3] - self.proposals[:, :, 1]
        height = self.proposals[:, :, 2] - self.proposals[:, :, 0]
        self.proposals = self.proposals.long()
        for i in range(proposals.shape[0]):
            for j in range(proposals.shape[1]):
                if (width[i, j] <= 10 or height[i, j] <= 10):
                    continue
                pro.append(proposals[i, j])
                reg.append(region_embedding[i, j])
        return pro, reg

    def forward(self, images, targets):
        self.backbone.eval()
        # 加载clip模型
        model, preprocess = clip.load('ViT-B/32', device, jit=False)

        # backbone输入应该为image数组
        detection = self.backbone(images)
        print(len(self.backbone.roi_heads.box_head.features))
        self.proposals = torch.stack(
            [proposal for proposal in self.backbone.proposals])
        region_embedding = self.region(
            self.backbone.roi_heads.box_head.features).reshape(int(len(images)), int(len(self.backbone.roi_heads.box_head.features)/len(images)), 512)

        # 筛选proposals
        self.proposals, region_embedding = self.filter_proposals(
            self.proposals, region_embedding)
        self.proposals = torch.stack(
            [proposal for proposal in self.proposals]).unsqueeze(0)
        region_embedding = torch.stack(
            [reg for reg in region_embedding]).unsqueeze(0)
        
        # 训练或者推理时都需要计算text_embedding
        text_embedding = self.VILD_text(model)
        if self.training:
            images_embedding = self.VILD_image(images, model, preprocess)
            proposal_labels = torch.empty(
                [self.proposals.shape[0], self.proposals.shape[1], train_class+1])

            # 获取每个proposal对应的one-hot标签
            for i in range(self.proposals.shape[0]):
                for p, j in zip(self.proposals[i], range(self.proposals.shape[1])):
                    proposal_labels[i, j] = self.get_label(
                        p, targets[i], images[i])

            # 计算VILD_text loss 和VILD_image loss
            loss = self.Loss_text(
                text_embedding, region_embedding, proposal_labels)
            loss += 0.5*self.Loss_image(images_embedding, region_embedding)
            avg_losses = loss / len(self.backbone.roi_heads.box_head.features)
            return avg_losses

        return region_embedding


if __name__ == '__main__':
    # 主要作用是将图片数据转成tensor传入显存
    preprocess = vocDataset.get_transform(False)
    postprocess = transforms.ToPILImage()

    # 加载数据
    dataset = vocDataset.vocData(
        "/home/llrt/文档/VOCdevkit/VOC2012", transform=preprocess)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn, pin_memory=True)
    model0 = VILD().to(device)
    # model0.load_state_dict(torch.load("VILD.pt"))
    for p in model0.backbone.parameters():
        p.requires_grad = False
    params = [p for p in model0.parameters() if p.requires_grad]
    model0.train()

    # 将需要训练的参数输入优化器中
    optimizer = torch.optim.SGD(
        params, lr=0.006, momentum=0.8, weight_decay=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=500,
                                                   gamma=0.9)

    model0.to(device)
    num_epoch = 2

    # 训练开始
    for i in range(num_epoch):
        for imgs, targets in loader:
            images = list([img.to(device) for img in imgs])
            if(int(len(images)) == 0 or int(len(targets[0]["boxes"])) == 0):
                continue
            target = [{k: v for k, v in target.items()}
                      for target in targets]
            losses = model0(images, target)
            print('epoch:{} loss:{}'.format(i, losses))
            # print(model0.backbone.backbone.state_dict())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            lr_scheduler.step()
            # if j % 30 ==0:

        torch.save(model0.state_dict(), 'VILD.pt')
