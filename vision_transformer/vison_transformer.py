# _*_coding:utf-8_*_
# _*_coding:utf-8_*_
import pandas as pd
import numpy as np
import cv2


import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from easydict import EasyDict
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn import CrossEntropyLoss, NLLLoss
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# %matplotlib inline

train = pd.read_csv("digit-recognizer/train.csv")
test = pd.read_csv("digit-recognizer/test.csv")
submission = pd.read_csv("digit-recognizer/sample_submission.csv")

train_images = train.iloc[:, 1:].values.reshape(-1, 28, 28)
train_labels = train.iloc[:, 0].values
test_images = test.values.reshape(-1, 28, 28)

#以下时vision_transformer需要用到的常量
model_dim = 20
patch_size = 4
patch_depth = patch_size * patch_size * 1
weight = torch.randn(patch_depth,model_dim).to("cuda:0")
kernel = weight.transpose(0,1).reshape((-1,1,patch_size,patch_size))

def get_transform(image_size, train=True):

    if train:
        return A.Compose([
                A.HorizontalFlip(p=0.5),
                 A.VerticalFlip(p=0.5),
                 A.RandomBrightnessContrast(p=0.2),
                 A.Resize(*image_size, interpolation=cv2.INTER_LANCZOS4),
                 A.Normalize(0.1310, 0.30854),

                 ToTensorV2(),
 ])
    else:
        return A.Compose([
            A.Resize(*image_size, interpolation=cv2.INTER_LANCZOS4),
            A.Normalize(0.1310, 0.30854),
            ToTensorV2(),
 ])



CONFIG = EasyDict({
 "backbone": "convnext_small",
 "num_class": 10,
 "image_size": (56,56),
 "pretrained": True,
 "epochs": 5,
 "batch_size": 56,
 "num_workers": 0,
 "device": "cuda:0"
})


def image2emb_conv(image, kernel, stride):
    conv_output = F.conv2d(input=image, weight=kernel, stride=stride)
    #print(conv_output.shape)
    batch_size, output_channels, output_height, output_wide = conv_output.shape
    patch_embadding = conv_output.reshape(batch_size, output_channels, (output_height * output_wide)).transpose(-1, -2)
    cls_token_embadding = torch.randn(CONFIG.batch_size, 1, model_dim, requires_grad=True)
    return patch_embadding , cls_token_embadding

# patch_embadding_conv = image2emb_conv(image,kernel=kernel,stride=patch_size)

# cls_token_embadding = torch.randn(CONFIG.batch_size, 1, model_dim, requires_grad=True)



def fa(patch_embadding_conv,model_dim,cls_token_embadding):
    #print(patch_embadding_conv.shape,cls_token_embadding.shape)
    token_embadding = torch.cat([cls_token_embadding, patch_embadding_conv], dim=1)
    #print(token_embadding.shape,'token_embadding.shape')
    max_num_token = 220
    position_embadding_table = torch.randn(max_num_token, model_dim, requires_grad=True).to("cuda:0")
    #print('position_embadding_table.shape',position_embadding_table.shape)
    seq_len = token_embadding.shape[1]
    #print(seq_len,'seq_len')
    #print(position_embadding_table[:seq_len].shape,'position_embadding_table[:seq_len].shape')
    position_embadding = torch.tile(position_embadding_table[:seq_len], [token_embadding.shape[0], 1, 1]).to("cuda:0")
    #print(token_embadding.shape,position_embadding.shape)
    token_embadding = token_embadding + position_embadding
    return token_embadding
# 注意：max_num_token 一定要要大于 token_embadding 的第一个维度
# 不然切片操作是无效的 并且 程序也不报错

class MiniDataSet(Dataset):

    def __init__(self, images, labels=None, transform=None):
        self.images = images.astype("float32")   #据pytorch要求 输入的图片的数据要转换成 float32
        # print(images.shape)

        self.labels = labels
        self.transform = transform



    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        ret = {}

        img = self.images[idx]

        if self.transform is not None:
            img = self.transform(image=img)["image"]
        ret["image"] = img


        if self.labels is not None:
            ret["label"] = self.labels[idx]

        return ret

#测试不同的模型
#主要就是改这个地方
#定义初始化方法
#定义forward方法 也就是模型的前向推理方法

class MiniModel(nn.Module):

    def __init__(self,num_class,):
        #同时将pretrained改为False  该为true的话 是自己下载
        #backbone_ckpt 若是使用自己找来的权重文件 或者是 自己训练的权重文件 这里直接传入权重文件的地址就好了

        super().__init__()  #写作规范
        # self.backbone = timm.create_model(backbone, pretrained=pretrained,
        #                               checkpoint_path=backbone_ckpt, in_chans=1)
        #这里的conv_next是用image_net训练的 所以默认是三通道的 但是我们这里的图片的通道数为1
        #注意，不同的网络的分类层 是不一样的 需要具体更改
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=10)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=6)
        self.head = nn.Linear(in_features=model_dim,out_features=num_class)

        self.loss_fn = NLLLoss()

    def forward(self, image, label=None):
        # image = image.unsqueeze(dim=1)
        # RuntimeError: Expected
        # 4 - dimensional
        # input
        # for 4 - dimensional weight[96, 1, 4, 4], but got 5-dimensional input of size[128, 1, 1, 56, 56] instead
        # ToTensor已经起作用了

        patch_embadding_conv,cls_token_embadding = image2emb_conv(image,kernel=kernel,stride=patch_size)
        patch_embadding_conv,cls_token_embadding = patch_embadding_conv.to("cuda:0"),cls_token_embadding.to("cuda:0")
        token_embadding = fa(patch_embadding_conv,model_dim=model_dim,cls_token_embadding=cls_token_embadding)
        # print(token_embadding.shape)
        token_embadding = token_embadding.to("cuda:0")
        encoder_output = self.transformer_encoder(token_embadding)

        cls_token_output = encoder_output[:, 0, :]

        logit = self.head(cls_token_output)  #分类预测的结果

        if label is not None:
            logit_logsoftmax = torch.log_softmax(logit, 1)
            loss = self.loss_fn(logit_logsoftmax, label)
            return {"prediction": logit, "loss": loss}
        return {"prediction": logit}

from tqdm import tqdm


def train_one_epoch(model, optimizer, dataloader, device):
    model.to(device)
    model.train()
    train_loss = []

    for batch in tqdm(dataloader):

        output = model(batch["image"].to(device), batch["label"].to(device))
        optimizer.zero_grad()
        output["loss"].backward()
        optimizer.step()
        train_loss.append(output['loss'].item())

    return np.mean(train_loss)


def eval_one_epoch(model, dataloader, device):
    model.to(device)
    model.eval()
    eval_loss = []

    for step, batch in enumerate(dataloader):
        output = model(batch["image"].to(device), batch["label"].to(device))
        eval_loss.append(output['loss'].item())

    return np.mean(eval_loss)


def predict(model, dataloader, device):
    model.to(device)
    model.eval()
    predictions = []

    for step, batch in enumerate(dataloader):
        output = model(batch["image"].to(device))
        prediction = torch.argmax(output['prediction'], 1)
        predictions.append(prediction.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    return predictions




train_transform = get_transform(CONFIG.image_size, True)
valid_transform = get_transform(CONFIG.image_size, False)
train_ds = MiniDataSet(train_images[:40000], train_labels[:40000], train_transform)
val_ds = MiniDataSet(train_images[40000:], train_labels[40000:], valid_transform)

test_ds = MiniDataSet(test_images, transform=valid_transform)

train_dl = DataLoader(
 train_ds,
 batch_size=CONFIG.batch_size,
 num_workers=CONFIG.num_workers,
 shuffle=True,
 drop_last=True)
val_dl = DataLoader(
 val_ds,
 batch_size=CONFIG.batch_size,
 num_workers=CONFIG.num_workers,
 shuffle=False,
 drop_last=True)   #这里 False 要改为 True
test_dl = DataLoader(
 test_ds,
 batch_size=CONFIG.batch_size,
 num_workers=CONFIG.num_workers,
 shuffle=False,
 drop_last=True)  #这里 False 要改为 True    #但是会减少预测数目啊！！！

model = MiniModel(num_class=CONFIG.num_class)

optimizer = Adam(model.parameters(), lr=0.05)

if __name__ == '__main__':
    for epoch in range(CONFIG.epochs):
        print("-----第{}轮训练开始-----".format(epoch + 1))  # 为了符合阅读习惯 这里要加一

        train_loss = train_one_epoch(model, optimizer, train_dl, CONFIG.device)
        val_loss = eval_one_epoch(model, val_dl, CONFIG.device)
        print(f"Epoch_{epoch+1}, train loss {train_loss:.4f}, val loss {val_loss:.4f}")

        val_prediction = predict(model, val_dl, device=CONFIG.device)

        A = accuracy_score(train_labels[40000:41960], val_prediction)
        print("accuracy: ",A)

        torch.save(model.state_dict(), "model_epoch{}_{}.pt".format(epoch,A))

    test_prediction = predict(model, test_dl, device=CONFIG.device)


    #这个没有保存模型啊 得整一整

# Traceback (most recent call last):
#   File "D:/pytorch code/sota_test/vison_transformer.py", line 243, in <module>
#     model = MiniModel(num_class=CONFIG.num_class)
#   File "D:/pytorch code/sota_test/vison_transformer.py", line 141, in __init__
#     encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=8)
#   File "E:\anaconda\lib\site-packages\torch\nn\modules\transformer.py", line 264, in __init__
#     self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
#   File "E:\anaconda\lib\site-packages\torch\nn\modules\activation.py", line 874, in __init__
#     assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
# AssertionError: embed_dim must be divisible by num_heads



#   File "D:/pytorch code/sota_test/vison_transformer.py", line 88, in fa
#     token_embadding = torch.cat([cls_token_embadding, patch_embadding_conv], dim=1)
# RuntimeError: Sizes of tensors must match except in dimension 0. Got 16 and 64 (The offending index is 0)
# torch.Size([64, 20, 14, 14])
# torch.Size([64, 196, 20]) torch.Size([64, 1, 20])
# torch.Size([64, 20, 14, 14])
# torch.Size([64, 196, 20]) torch.Size([64, 1, 20])
# torch.Size([64, 20, 14, 14])
# torch.Size([64, 196, 20]) torch.Size([64, 1, 20])
# torch.Size([64, 20, 14, 14])
# torch.Size([64, 196, 20]) torch.Size([64, 1, 20])
# torch.Size([64, 20, 14, 14])
# torch.Size([64, 196, 20]) torch.Size([64, 1, 20])
# torch.Size([64, 20, 14, 14])
# torch.Size([64, 196, 20]) torch.Size([64, 1, 20])
# torch.Size([64, 20, 14, 14])
# torch.Size([64, 196, 20]) torch.Size([64, 1, 20])
# torch.Size([64, 20, 14, 14])
# torch.Size([64, 196, 20]) torch.Size([64, 1, 20])
# torch.Size([64, 20, 14, 14])
# torch.Size([64, 196, 20]) torch.Size([64, 1, 20])
# torch.Size([64, 20, 14, 14])
# torch.Size([64, 196, 20]) torch.Size([64, 1, 20])
# torch.Size([64, 20, 14, 14])
# torch.Size([64, 196, 20]) torch.Size([64, 1, 20])
# torch.Size([64, 20, 14, 14])
# torch.Size([64, 196, 20]) torch.Size([64, 1, 20])
# torch.Size([64, 20, 14, 14])
# torch.Size([64, 196, 20]) torch.Size([64, 1, 20])
# torch.Size([64, 20, 14, 14])
# torch.Size([64, 196, 20]) torch.Size([64, 1, 20])
# torch.Size([64, 20, 14, 14])
# torch.Size([64, 196, 20]) torch.Size([64, 1, 20])
# torch.Size([16, 20, 14, 14])
# torch.Size([16, 196, 20]) torch.Size([64, 1, 20])
# 最后一个不满足 64 所以不符合
# 应该对其舍弃
