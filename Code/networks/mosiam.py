# 1017 时间 修改了原本会造成nan数据的decoder 对比loss部分
# 1027 修改了decoder 的损失代码

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch.cuda.amp import autocast



class Channel_Max_2dPooling(nn.Module):
    def __init__(self, kernel: int, stride: int):
        super(Channel_Max_2dPooling, self).__init__()
        self.kernel_size = (1, kernel)
        self.stride = (1, stride)
        self.max_pooling = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
        """
        x shape is [bs, c, w, h]
        """
        x = x.transpose(1, 3) # bs h w c
        x = self.max_pooling(x)
        out = x.transpose(1, 3)
        return out

class Channel_Adaptive_Max_2dPooling(nn.Module):
    def __init__(self, channel_size: int, w: int):
        """
        channel_size: the kernel size after pooling
        w: the intput tensor size of tensor[-1]
        """
        super(Channel_Adaptive_Max_2dPooling, self).__init__()
        self.channel_size = channel_size
        self.adp_max_pooling = nn.AdaptiveMaxPool2d((w, channel_size))

    def forward(self, x):
        x = x.transpose(1, 3)
        x = self.adp_max_pooling(x)
        x = x.transpose(1, 3)
        return x

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, feature, feature2, label):
        # input feature shape [bs, c, w, h]
        # input label shape [bs, w, h]
        kernels = feature.permute(0, 2, 3, 1)# [bs, w, h, c]
        kernels_1 = feature2.permute(0, 2, 3, 1)
        kernels = kernels.reshape(-1, feature.shape[1], 1, 1)# [bs*w*h, c, 1, 1] 反映的是空间位置的相似性 该位置和其他的空间位置的相似性
        kernels_1 = kernels_1.reshape(-1, feature2.shape[1], 1, 1)

        logits = torch.div(F.conv2d(feature, kernels), self.temperature)# [bs, bs*w*h, w, h]
        logits = logits.permute(1, 0, 2, 3)# [bs*w*h, bs, w, h]
        logits = logits.reshape(logits.shape[0], -1)# [bs*w*h, bs*w*h] # 原本是一个位置 有m种相似性  现在是该位置相似的其他的位置

        logits2 = torch.div(F.conv2d(feature2, kernels_1), self.temperature)  # [bs, bs*w*h, w, h]
        logits2 = logits2.permute(1, 0, 2, 3)  # [bs*w*h, bs, w, h]
        logits2 = logits2.reshape(logits2.shape[0], -1)  # [bs*w*h, bs*w*h] # 原本是一个位置 有m种相似性  现在是该位置相似的其他的位置

        #print("loggits shape is {}, the logits2 shape is {}".format(logits.shape, logits2.shape))

        # sim_logits = torch.matmul(logits, logits2.T)
        sim_logits = F.cosine_similarity(logits.unsqueeze(1), logits2.unsqueeze(0), dim=2)
        #print("sim_logits shape is {}".format(sim_logits.shape))  # [bs*w*h, bs*w*h]
        sim_logits = (sim_logits+1)/2
        label = label.contiguous().view(-1, 1)# [bs*w*h, 1] 每一个位置是什么
        # all = label.sum()
        #print("label shape is ", label.shape)
        mask = torch.eq(label, label.T).float() # [bs*w*h, bs*w*h]

        bg_bool = torch.eq(label.squeeze().cpu(), torch.zeros(label.squeeze().shape))
        non_bg_bool = ~bg_bool
        non_bg_bool = non_bg_bool.int().unsqueeze(1).cuda()

        sim_logits = torch.div(sim_logits, self.temperature)
        sim_max = sim_logits.max()
        exp_logits = torch.exp(sim_logits)
        mask_sim_exp_logits = mask*exp_logits# 获取正样本的相似度
        positive_sim_exp_logits = torch.sum(mask_sim_exp_logits, dim=-1, keepdim=True)
        all_sim_positive_exp_logits = torch.sum(exp_logits, dim=-1, keepdim=True)
        loss_prob = -torch.log(torch.div(positive_sim_exp_logits, all_sim_positive_exp_logits+1e-6)+1e-6)
        max_loss_prob = loss_prob.max()
        loss = (loss_prob*non_bg_bool).sum()/non_bg_bool.sum()
        return loss


class BlockConLoss(nn.Module):
    def __init__(self, temperature=0.7, block_size=10):
        super(BlockConLoss, self).__init__()
        self.block_size = block_size
        self.supconloss = SupConLoss(temperature=temperature)

    def forward(self, features, label):
        # features [bs, v, c, w, h]
        # features [bs, w, h]
        shape = features.shape
        image_size = shape[-1]
        div_num = image_size // self.block_size
        losses = []
        for i in range(div_num):
            for j in range(div_num):
                block_features = features[:, :, :, i*self.block_size:(i+1)*self.block_size,
                                 j*self.block_size:(j+1)*self.block_size]
                block_label = label[:, i*self.block_size:(i+1)*self.block_size,
                                 j*self.block_size:(j+1)*self.block_size]

                block_feature_lst = torch.unbind(block_features, dim=1)
                block_feature1, block_feature2 = block_feature_lst
                if block_label.sum() == 0:
                    continue

                non_bg_loss = self.supconloss(block_feature1, block_feature2, block_label)
                losses.append(non_bg_loss)

        if len(losses) == 0:
            loss = torch.tensor(0).float()
            return loss

        loss = torch.stack(losses).mean()
        return loss


class MoCLR(nn.Module):
    def __init__(self, model, emodel, args):
        """
        just working for transunet
        """
        super(MoCLR, self).__init__()
        self.model = model
        self.emodel = emodel
        self.args = args
        # self.blockconloss = BlockConLoss()
        self.labeled_bs = self.args.label_batch_size
        # self.ce_loss = nn.CrossEntropyLoss()
        # self.dice_loss = DiceLoss(self.args.num_classes)

        mlp_dim = self.model.mlp_dim
        self.model.fc = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(inplace=True),
            self.model.fc
        )
        self.emodel.fc = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(inplace=True),
            self.emodel.fc
        )

        # 初始化ema_model参数, 并且让ema_model停止梯度
        for parm_1, parm_2 in zip(self.model.parameters(), self.emodel.parameters()):
            parm_2.data.copy_(parm_1.data)
            parm_2.require_grad = False

    @torch.no_grad()
    def _momentum_update_emmodel(self, iter_num):
        alpha = min(1 - 1 / (iter_num + 1), self.args.ema_decay)
        for ema_param, param in zip(self.emodel.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def flat_loss(self, feature):
        """
        feature [2*bs x 768]
        """
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()
        feature1 = feature.unsqueeze(1)
        feature2 = feature.unsqueeze(0)

        similarity_matrix = F.cosine_similarity(feature1, feature2, dim=2)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        positive = similarity_matrix[labels.bool()].view(similarity_matrix.shape[0], -1)
        negetive = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        labels = torch.zeros(positive.shape[0], dtype=torch.long).cuda()
        logits = (negetive - positive) / self.args.temperature
        return logits, labels

    # def forward(self, x1, x2, x3, iter_num, label=None):
    def forward(self, x1, x2):
        # x1 x2 是经过了数据增强之后的view x3[bs * 8, c, w, h] 用来计算一致性的
        # outputs(bs, num_class, w, h) embedding (bs, 768)
        # labels shape [bs, w, h]
        # 计算的相似度计算cross_entropy
        outputs, embedding, _ = self.model(x1)
        embedding = F.normalize(embedding, dim=1)

        # 计算ema 结果
        with torch.no_grad():
            # self._momentum_update_emmodel(iter_num)
            ema_outputs, ema_embedding, _ = self.emodel(x2)
            ema_embedding = F.normalize(ema_embedding, dim=1)

        # 计算encoder部分对比损失
        features = torch.cat([embedding, ema_embedding], dim=0)
        with autocast(enabled=self.args.fp16_precision):
            # 默认为true
            logits, labels = self.flat_loss(features)  # 2*bs

        return logits, labels, outputs, ema_outputs

if __name__ == '__main__':
    a = torch.randn((2, 3, 3), requires_grad=False)
    b = torch.randn(2, 3, 3)
    a = a.unsqueeze(1)
    b = b.unsqueeze(1)
    c = torch.cat([a, b], dim=1)
    print(c.shape)
