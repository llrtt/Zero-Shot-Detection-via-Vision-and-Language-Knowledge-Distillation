import vocDataset
from VILD import VILD
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms


def collate_fn(batch):
    return tuple(zip(*batch))


def VILD_train(epoch=2, train_class=2, pretrain=False, dataset_path="/home/llrt/文档/VOCdevkit/VOC2012"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 主要作用是将图片数据转成tensor传入显存
    preprocess = vocDataset.get_transform(False)

    # 加载数据
    dataset = vocDataset.vocData(
        data_path=dataset_path, train_class=train_class, transform=preprocess)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn, pin_memory=True)

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
            print('epoch:{} sum_loss:{} avg_loss:{}'.format(i, losses, avg_loss))
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            lr_scheduler.step()

        torch.save(model.state_dict(), 'VILD.pt')
    print('-------------------------------------------train complete------------------------------------------------')


def VILD_eval(model_path="VILD.pt", train_class=2, dataset_path="/home/llrt/文档/VOCdevkit/VOC2012"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 主要作用是将图片数据转成tensor传入显存
    preprocess = vocDataset.get_transform(False)
    postprocess = transforms.ToPILImage()
    # 加载数据
    dataset = vocDataset.vocData(
        train_class=train_class, data_path=dataset_path, transform=preprocess)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn, pin_memory=True)

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
        # 清空每一次显示的矩形框，防止残留
        plt.cla()
        fig = plt.imshow(postprocess(images[0].to('cpu')))
        result, proposals = model(images, target)
        for p, r in zip(proposals, result):
            for pro, re, i in zip(p, r, range(len(p))):
                for bbox, core in zip(pro, re):
                    if(core < 0.9):
                        continue
                    fig.axes.add_patch(vocDataset.bbox_to_rect(bbox, 'blue'))
                    plt.text(bbox[0], bbox[1],
                             vocDataset.classes[i-1]+': '+str(round(core, 2)))
        plt.pause(1)


if __name__ == '__main__':
    VILD_train(epoch=2, train_class=2, pretrain=False,
               dataset_path="/home/llrt/文档/VOCdevkit/VOC2012")
