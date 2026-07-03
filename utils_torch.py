import os
import torch
import numpy as np
import pandas as pd
from PIL import Image


# 映射模型名称到所需的输入尺寸
image_size_dict = {
    'inception_v1': 299,
    'inception_v2': 299,
    'inception_v3': 299,
    'inception_v4': 299,
    'inception_resnet_v2': 299,
    'resnet_v1_50': 224,
    'resnet_v1_101': 224,
    'resnet_v1_152': 224,
    'resnet_v1_200': 224,
    'resnet_v2_50': 299,  # 原版 TF 中 ResNet-V2 是 299
    'resnet_v2_101': 299,
    'resnet_v2_152': 299,
    'resnet_v2_200': 299,
    'vgg_16': 224,
    'vgg_19': 224,
}

# 映射均值和标准差 (用于组装 PreprocessingModel)
mean_dict = {
    'inception_v1': [0.5, 0.5, 0.5],
    'inception_v2': [0.5, 0.5, 0.5],
    'inception_v3': [0.5, 0.5, 0.5],
    'inception_v4': [0.5, 0.5, 0.5],
    'inception_resnet_v2': [0.5, 0.5, 0.5],
    'resnet_v1_50': [0.485, 0.456, 0.406],
    'resnet_v1_101': [0.485, 0.456, 0.406],
    'resnet_v1_152': [0.485, 0.456, 0.406],
    'resnet_v1_200': [0.485, 0.456, 0.406],
    'resnet_v2_50': [0.5, 0.5, 0.5],
    'resnet_v2_101': [0.5, 0.5, 0.5],
    'resnet_v2_152': [0.5, 0.5, 0.5],
    'resnet_v2_200': [0.5, 0.5, 0.5],
    'vgg_16': [0.485, 0.456, 0.406],
    'vgg_19': [0.485, 0.456, 0.406],
}

std_dict = {
    'inception_v1': [0.5, 0.5, 0.5],
    'inception_v2': [0.5, 0.5, 0.5],
    'inception_v3': [0.5, 0.5, 0.5],
    'inception_v4': [0.5, 0.5, 0.5],
    'inception_resnet_v2': [0.5, 0.5, 0.5],
    'resnet_v1_50': [0.229, 0.224, 0.225],
    'resnet_v1_101': [0.229, 0.224, 0.225],
    'resnet_v1_152': [0.229, 0.224, 0.225],
    'resnet_v1_200': [0.229, 0.224, 0.225],
    'resnet_v2_50': [0.5, 0.5, 0.5],
    'resnet_v2_101': [0.5, 0.5, 0.5],
    'resnet_v2_152': [0.5, 0.5, 0.5],
    'resnet_v2_200': [0.5, 0.5, 0.5],
    'vgg_16': [0.229, 0.224, 0.225],
    'vgg_19': [0.229, 0.224, 0.225],
}

# PyTorch torchvision 和 timm 的多数分类模型输出均为 1000 类 (无背景类偏移),因此不需要offset
offset_dict = {
}


# ==========================================
# 2.AdvDataset
# ==========================================
class AdvDataset(torch.utils.data.Dataset):
    def __init__(self, model_name, input_dir=None, output_dir=None, targeted=False, target_class=None, eval=False):
        self.targeted = targeted
        self.target_class = target_class
        self.data_dir = input_dir

        # 动态获取当前模型的尺寸，彻底抛弃硬编码
        self.image_size = image_size_dict.get(model_name, 299)
        self.f2l = self.load_labels(os.path.join(self.data_dir, 'labels.csv'))

        if eval:
            self.data_dir = output_dir
            print(f'=> Eval mode: evaluating on {self.data_dir} (Resize to {self.image_size})')
        else:
            self.data_dir = os.path.join(self.data_dir, 'images')
            print(f'=> Train mode: training on {self.data_dir} (Resize to {self.image_size})')
            if output_dir:
                print(f'Save images to {output_dir}')

    def __len__(self):
        return len(self.f2l.keys())

    def __getitem__(self, idx):
        filename = list(self.f2l.keys())[idx]
        filepath = os.path.join(self.data_dir, filename)

        # 仅进行一次 Resize，避免双重插值造成的伪影破坏对抗梯度
        image = Image.open(filepath).convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # 统一归一化到 [0, 1] 区间，对抗扰动 Delta 将直接累加在此空间
        image = np.array(image).astype(np.float32) / 255.0  # 在真实加载模型的时候，针对inc模型会进行Normalize，得到[-1,1]的值域输入给模型
                                                            # 这里做法是先在[0,1]施加扰动，然后再转回[-1,1]，在数据上等价于原TF
        image = torch.from_numpy(image).permute(2, 0, 1)  # 把 NumPy 格式的图像数组转成 PyTorch tensor,并把维度顺序从 HWC 改成 CHW

        label = self.f2l[filename]
        return image, label, filename

    def load_labels(self, file_name):
        dev = pd.read_csv(file_name)
        f2l = {}
        for i in range(len(dev)):
            img_filename = str(dev.iloc[i]['ImageId']) + '.png'

            # 必须强制 -1，将 NIPS 数据集的 1-1000 标签对齐至 PyTorch 的 0-999 索引
            if self.targeted:
                if self.target_class:
                    f2l[img_filename] = [dev.iloc[i]['TrueLabel'] - 1, self.target_class - 1]
                else:
                    f2l[img_filename] = [dev.iloc[i]['TrueLabel'] - 1, dev.iloc[i]['TargetClass'] - 1]
            else:
                f2l[img_filename] = dev.iloc[i]['TrueLabel'] - 1  # 所有模型都是1000类，因此统统-1

        return f2l


# ==========================================
# 3. 辅助函数
# ==========================================
def save_images(output_dir, adversaries, filenames):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 强制进行 clamp 截断，防止对抗扰动溢出导致反推 uint8 时出现花屏 (噪点)
    adversaries = torch.clamp(adversaries, 0.0, 1.0)
    adversaries = (adversaries.detach().permute((0, 2, 3, 1)).cpu().numpy() * 255.0).astype(np.uint8)

    for i, filename in enumerate(filenames):
        Image.fromarray(adversaries[i]).save(os.path.join(output_dir, filename))