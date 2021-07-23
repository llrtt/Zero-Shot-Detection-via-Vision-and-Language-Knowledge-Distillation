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
train_class = 2


class VILD(nn.Module):
    """
    Arguments:
        train_calss:训练时用到的类的数目，推理时用到全部类别
        iot_thresh:用于计算prposals对应label的阈值
    """

    def __init__(self, train_class=2, iot_thresh=0.3):
        super(VILD, self).__init__()
        self.train_class = train_class
        self.iot_thresh = iot_thresh

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

        # 用简单的卷积加全连接层做生成region_embedding
        self.region = nn.Sequential(nn.Conv2d(256, 64, kernel_size=(
            3, 3), stride=1, padding=(3, 3), bias=False), nn.Flatten(), nn.Linear(7744, 512))

        self.background = nn.Parameter(torch.rand(1, 512))

    def process_proposals(self, images):
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

        # 将proposal变换为原来的图片对应的尺寸
        self.proposals[:, :, 0] = self.proposals[:, :, 0] * x_ratio
        self.proposals[:, :, 1] = self.proposals[:, :, 1] * y_ratio
        self.proposals[:, :, 2] = self.proposals[:, :, 2] * x_ratio
        self.proposals[:, :, 3] = self.proposals[:, :, 3] * y_ratio

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

    def VILD_image(self, images, model, preprocess):
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
        text_label = torch.zeros([1, self.train_class+1])
        for i in range(len(target['labels'])):
            if nms.iot(proposal.to('cpu'), target['boxes'][i]) >= self.iot_thresh:
                text_label[0, target['labels'][i]+1] = 1
        return text_label

    def VILD_text(self, model):
        """
        生成text_embedding，训练时只用到VOC数据集中前十个类
        """
        if self.training:
            text_inputs = torch.cat(
                [clip.tokenize(f"a photo of a {c}") for c in vocDataset.classes[:self.train_class]]).to(device)
        else:
            text_inputs = torch.cat(
                [clip.tokenize(f"a photo of a {c}") for c in vocDataset.classes[:self.train_class]]).to(device)
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
                # print("region:{}".format(nn.functional.softmax(Zr)))
                # print("label:{}".format(targets[i, j]))
                loss_t = nn.CrossEntropyLoss()
                if(torch.nonzero(targets[i, j]).shape[0] == 0):
                    losses += loss_t(Zr/5, torch.tensor([0]).to(device))
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
        训练前筛选掉部分proposals
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

    def filter_result(self, result):
        """
        筛除背景概率较大的proposal
        """
        pro = []
        res = []
        all_classes = []
        all_classes_pre = []
        for i in range(int(self.proposals.shape[0])):
            all_classes.append([])
            for j in range(self.train_class+1):
                all_classes[i].append([])

        for i in range(int(self.proposals.shape[0])):
            all_classes_pre.append([])
            for j in range(self.train_class+1):
                all_classes_pre[i].append([])

        for proposals, reses in zip(self.proposals, result):
            p_s = []
            res_s = []
            for p, r in zip(proposals, reses):
                if(r[0][0] >= 0.05):
                    continue
                else:
                    p_s.append(p)
                    res_s.append(r)
            pro.append(p_s)
            res.append(res_s)

        # 将proposals分类，依据是result中概率最大对应的数组下标
        for p, r, i in zip(pro, res, range(int(self.proposals.shape[0]))):
            for p_s, r_s in zip(p, r):
                val, ind = r_s.max(1)
                all_classes[i][ind.long()].append([b.item()
                                                   for b in p_s.cpu().detach()])
                all_classes_pre[i][ind.long()].append(
                    val[0].cpu().detach().item())
        # 对每类proposal进行NMS
        for i in range(int(self.proposals.shape[0])):
            for j in range(self.train_class+1):
                all_classes[i][j], all_classes_pre[i][j] = nms.nms(
                    np.array(all_classes[i][j]), np.array(all_classes_pre[i][j]))

        return all_classes, all_classes_pre

    def forward(self, images, targets):
        """
        Arguments:
            images:tensor([num_images,3,height,width])
            targets:由vocDataset输出的标注
        """
        # 冻结backbone的参数
        self.backbone.eval()
        # 加载clip模型
        model, preprocess = clip.load('ViT-B/32', device, jit=False)

        # backbone输入应该为image数组
        detection = self.backbone(images)
        self.proposals = torch.stack(
            [proposal for proposal in self.backbone.proposals])

        # 将ROIalign之后的features输入网络生成region_embedding
        region_embedding = self.region(
            self.backbone.roi_heads.box_head.features).reshape(int(len(images)), int(len(self.backbone.roi_heads.box_head.features)/len(images)), 512)

        # 筛选proposals
        self.proposals, region_embedding = self.filter_proposals(
            self.proposals, region_embedding)
        self.proposals = torch.stack(
            [proposal for proposal in self.proposals]).unsqueeze(0).repeat([int(len(images)), 1, 1])
        region_embedding = torch.stack(
            [reg for reg in region_embedding]).unsqueeze(0).repeat([int(len(images)), 1, 1])
        self.process_proposals(images)

        # 训练或者推理时都需要计算text_embedding
        text_embedding = self.VILD_text(model)
        if self.training:
            images_embedding = self.VILD_image(images, model, preprocess)
            proposal_labels = torch.empty(
                [self.proposals.shape[0], self.proposals.shape[1], self.train_class+1])

            # 获取每个proposal对应的one-hot标签
            for i in range(self.proposals.shape[0]):
                for p, j in zip(self.proposals[i], range(self.proposals.shape[1])):
                    proposal_labels[i, j] = self.get_label(
                        p, targets[i], images[i])

            # 计算VILD_text loss 和VILD_image loss
            loss = self.Loss_text(
                text_embedding, region_embedding, proposal_labels)
            loss += 0.5*self.Loss_image(images_embedding, region_embedding)

            # 单个proposal的平均loss,去除了proposals数量不同的影响
            avg_losses = loss / len(self.backbone.roi_heads.box_head.features)
            return avg_losses, loss

        # 计算分类结果
        result = []
        text_embedding = torch.cat([self.background, text_embedding], dim=0)
        for regions in region_embedding:
            r = []
            for region in regions:
                # 计算余弦相似度
                Zr = self.sim(region.unsqueeze(0), text_embedding.float())
                r.append(nn.functional.softmax(Zr))
            result.append(r)

        # 筛除背景
        self.proposals, result = self.filter_result(result)

        return result, self.proposals


def VILD_train(epoch=2, train_class=2, pretrain=False):
    # 主要作用是将图片数据转成tensor传入显存
    preprocess = vocDataset.get_transform(False)

    # 加载数据
    dataset = vocDataset.vocData(
        "/home/llrt/文档/VOCdevkit/VOC2012", transform=preprocess)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn, pin_memory=True)

    # 加载模型
    model = VILD(train_class)
    if pretrain:
        model.load_state_dict(torch.load("VILD.pt"))
    for p in model.backbone.parameters():
        p.requires_grad = False

    # 提取出可训练参数
    params = [p for p in model.parameters() if p.requires_grad]
    model.train()

    # 将需要训练的参数输入优化器中
    optimizer = torch.optim.SGD(
        params, lr=0.0006, momentum=0.8, weight_decay=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=500,
                                                   gamma=0.9)

    model.to(device)

    # 训练开始
    for i in range(epoch):
        for imgs, targets in loader:
            images = list([img.to(device) for img in imgs])
            target = [{k: v for k, v in target.items()}
                      for target in targets]

            # 防止images为空导致训练出错
            if(int(len(images)) == 0 or int(len(targets[0]["boxes"])) == 0):
                continue
            avg_loss, losses = model(images, target)
            print('epoch:{} sum_loss:{} avg_loss'.format(i, avg_loss, avg_loss))
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            lr_scheduler.step()

        torch.save(model.state_dict(), 'VILD.pt')
    print('-------------------------------------------train complete------------------------------------------------')


def VILD_eval(model_path="VILD.pt"):
    # 主要作用是将图片数据转成tensor传入显存
    preprocess = vocDataset.get_transform(False)
    postprocess = transforms.ToPILImage()
    # 加载数据
    dataset = vocDataset.vocData(
        "/home/llrt/文档/VOCdevkit/VOC2012", transform=preprocess)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn, pin_memory=True)

    # 加载模型
    model = VILD(train_class)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    for imgs, targets in loader:
        images = list([img.to(device) for img in imgs])
        target = [{k: v for k, v in target.items()}
                  for target in targets]

        # 防止images为空导致出错
        if(int(len(images)) == 0 or int(len(targets[0]["boxes"])) == 0):
            continue
        plt.cla()
        fig = plt.imshow(postprocess(images[0].to('cpu')))
        result, proposals = model(images, target)
        for p, r in zip(proposals, result):
            for pro, re in zip(p, r):
                for bbox, core in zip(pro, re):
                    fig.axes.add_patch(
                        vocDataset.bbox_to_rect(bbox, 'blue'))
        plt.pause(1)


if __name__ == '__main__':
    VILD_eval()
