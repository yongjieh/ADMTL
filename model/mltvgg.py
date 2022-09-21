import torchvision.models as models
import torch.nn as nn
from typing import Union, List, cast
import torch
import torch.nn.init as init
import math
import torchvision.models as pmodels


class mlt_vgg(nn.Module):
    def __init__(self, num_classes):
        super(mlt_vgg, self).__init__()

        cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

        self.features = make_layers(cfg)
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mlts, pre=None):
        mmdlosses = 0
        masklosses = 0
        pre_x = x
        mlt_x = [x for _ in range(len(mlts) + 1)]  # 0为要网络的网络输出，其他为其他多任务网络输出
        for name, m in self.features.named_modules():  # 遍历全部模块
            if isinstance(m, nn.Conv2d):
                mlt_x[0] = m(mlt_x[0])
                if pre is not None:
                    pre_x = run_module(pre_x, pre, name)
                    pre_x, mlt_x[0], maskloss, mmdloss = self.pre_transfer(pre_x, mlt_x[0], name)  # 处理迁移学习
                    masklosses += maskloss  # mask损失
                    mmdlosses += mmdloss  # mmd损失
                for idx, mlt in enumerate(mlts):
                    mlt_x[idx + 1] = run_module(mlt_x[idx + 1], mlt.features, name)
            elif isinstance(m, nn.MaxPool2d):
                mlt_x[0] = m(mlt_x[0])
                if pre is not None:
                    pre_x = run_module(pre_x, pre, name)
                self.share_unit(mlt_x, mlts, name)  # 特征共享
            elif not isinstance(m, nn.Sequential):
                mlt_x[0] = m(mlt_x[0])
                if pre is not None:
                    pre_x = run_module(pre_x, pre, name)
                for idx, mlt in enumerate(mlts):
                    mlt_x[idx + 1] = run_module(mlt_x[idx + 1], mlt.features, name)
            else:
                pass
        # out = self.avgpool(mlt_x[0])
        out = torch.flatten(mlt_x[0], 1)
        out = self.classifier(out)
        return out, mmdlosses, masklosses

    def init(self, pre, mlts):
        self.make_mask(pre)  # 产生mask
        self.make_share_matrix(mlts)  # 产生共享矩阵

    def make_mask(self, pre, fea_size=[112, 56, 28, 28, 14, 14, 7, 7]):  # 产生mask
        idx = 0
        for pre_name, pre_m in pre.named_modules():
            if not isinstance(pre_m, nn.Sequential) and isinstance(pre_m, nn.Conv2d):
                # channels = pre_m.weight.shape[0]
                temp_val = nn.Parameter(torch.FloatTensor(fea_size[idx], fea_size[idx]))
                init.kaiming_uniform_(temp_val, a=math.sqrt(5))
                # init.constant_(temp_val, 1)
                # init.uniform_(temp_val)
                setattr(self, 'mask_' + pre_name, temp_val)
                idx += 1
        return

    def make_share_matrix(self, mlts):  # 产生共享矩阵
        for name, m in self.features.named_modules():
            if isinstance(m, nn.MaxPool2d):
                temp_val = nn.Parameter(torch.FloatTensor(len(mlts) + 1, len(mlts) + 1))
                nn.init.constant_(temp_val, 1)
                setattr(self, 'share_' + name, temp_val)
        return

    def share_unit(self, mlt_x, mlts, name):  # 特诊共享
        shares = []
        n, c, w, h = mlt_x[0].shape
        shares.append(mlt_x[0].reshape((-1)).unsqueeze(0))
        for idx, mlt in enumerate(mlts):
            mlt_x[idx + 1] = run_module(mlt_x[idx + 1], mlt.features, name)
            shares.append(mlt_x[idx + 1].reshape((-1)).unsqueeze(0))
        shares = torch.cat(shares, dim=0)
        share_matrix = getattr(self, 'share_' + name)
        shareds = torch.mm(share_matrix, shares)
        for i, shared in enumerate(shareds):
            mlt_x[i] = shared.reshape((n, c, w, h))
        return

    def pre_transfer(self, pre_x, x, name):  # 迁移学习

        mask = getattr(self, 'mask_' + name)

        mask = torch.mul(mask, hard_mask(mask))

        pre_mask_x = torch.mul(pre_x.clone(), mask)

        maskloss = torch.norm(mask, p=2)

        x = torch.mul(x, pre_mask_x.clone())

        return pre_x, x, maskloss, 0

def normalization(data):
    _range = torch.max(data) - torch.min(data)
    return (data - torch.min(data)) / _range


def hard_mask(input: torch.Tensor):
    input = input.detach()
    input = normalization(input)  # 归一化
    return torch.bernoulli(input)  # 获得硬mask


def run_module(input, model, name):
    for name_, m in model.named_modules():
        if name_ == name:
            return m(input)
    return None


def make_layers2():
    pre_vgg = pmodels.vgg11_bn(pretrained=True)
    return pre_vgg.features


def make_layers(cfg: List[Union[str, int]]) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = v
    return nn.Sequential(*layers)


if __name__ == '__main__':
    pre_vgg = models.vgg11_bn(pretrained=True)
    # pre_vgg.eval()
    pre_vgg.requires_grad_(False)
    pre_vgg.cuda()
    mlt_vgg1 = models.vgg11_bn()
    mlt_vgg1.cuda()
    mlt_vgg2 = models.vgg11_bn()
    mlt_vgg2.cuda()
    m_vgg = mlt_vgg(2)
    m_vgg.eval()
    m_vgg.cuda()
    m_vgg.init(pre_vgg.features, [mlt_vgg1, mlt_vgg2])
    m_vgg(torch.ones((4, 3, 112, 112)).cuda(), [mlt_vgg1, mlt_vgg2], pre_vgg.features)
