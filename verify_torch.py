import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import timm
import warnings

warnings.filterwarnings("ignore")


class VerifyDataset(Dataset):
    def __init__(self, ori_dir, adv_dir):
        self.ori_dir = os.path.join(ori_dir, 'images')
        self.adv_dir = adv_dir
        self.labels_df = pd.read_csv(os.path.join(ori_dir, 'labels.csv'))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        filename = str(row['ImageId']) + '.png'

        ori_path = os.path.join(self.ori_dir, filename)
        adv_path = os.path.join(self.adv_dir, filename)

        ori_img = Image.open(ori_path).convert('RGB')
        ori_tensor = self.to_tensor(ori_img)

        if os.path.exists(adv_path):
            adv_img = Image.open(adv_path).convert('RGB')
        else:
            adv_img = ori_img
        adv_tensor = self.to_tensor(adv_img)

        label = row['TrueLabel'] - 1

        return ori_tensor, adv_tensor, label, filename


def get_model_and_transform(model_name):
    tf_transform = transforms.Compose([
        transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    # PyTorch 原生模型：需要 ImageNet 归一化和 224 尺寸
    pt_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if model_name == 'inception_v3':
        return timm.create_model('inception_v3.tf_in1k', pretrained=True), tf_transform
    elif model_name == 'inception_v4':
        return timm.create_model('inception_v4.tf_in1k', pretrained=True), tf_transform
    elif model_name == 'inception_resnet_v2':
        return timm.create_model('inception_resnet_v2.tf_in1k', pretrained=True), tf_transform
    elif model_name == 'resnet50':
        return models.resnet50(pretrained=True), pt_transform
    elif model_name == 'vgg16':
        return models.vgg16(pretrained=True), pt_transform
    elif model_name == 'inception_v3_adv':
        return timm.create_model('adv_inception_v3', pretrained=True), tf_transform
    elif model_name == 'inception_resnet_v2_ens_adv':
        return timm.create_model('ens_adv_inception_resnet_v2', pretrained=True), tf_transform
    else:
        raise ValueError(f"Not supported: {model_name}")


def verify(model_name, dataloader, device):
    model, transform = get_model_and_transform(model_name)
    model = model.to(device).eval()

    ori_pre_list = []
    adv_pre_list = []
    ground_truth_list = []

    with torch.no_grad():
        for ori_imgs, adv_imgs, labels, _ in dataloader:
            ori_inputs = transform(ori_imgs).to(device)
            adv_inputs = transform(adv_imgs).to(device)

            ori_preds = model(ori_inputs).argmax(dim=1).cpu().numpy()
            adv_preds = model(adv_inputs).argmax(dim=1).cpu().numpy()

            ori_pre_list.extend(ori_preds)
            adv_pre_list.extend(adv_preds)
            ground_truth_list.extend(labels.numpy())

    ori_pre = np.array(ori_pre_list)
    adv_pre = np.array(adv_pre_list)
    ground_truth = np.array(ground_truth_list)

    return ori_pre, adv_pre, ground_truth


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading data (Batch Size: {args.batch_size})...")

    dataset = VerifyDataset(args.ori_path, args.adv_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    total_images = len(dataset)

    model_names = ['inception_v3','inception_v4','inception_resnet_v2']

    print("==================================================")
    for model_name in model_names:
        print(model_name)
        ori_pre, adv_pre, ground_truth = verify(model_name, dataloader, device)

        ori_accuracy = np.sum(ori_pre == ground_truth) / total_images
        adv_accuracy = np.sum(adv_pre == ground_truth) / total_images
        adv_successrate = np.sum(ori_pre != adv_pre) / total_images  # 公式1: 对抗图与原图预测不同
        adv_successrate2 = np.sum(ground_truth != adv_pre) / total_images  # 公式2: 对抗图预测不是真实标签

        print('ori_acc:{:.1%}/adv_acc:{:.1%}/adv_suc:{:.1%}/adv_suc2:{:.1%}'.format(
            ori_accuracy, adv_accuracy, adv_successrate, adv_successrate2))
    print("==================================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_path', type=str, default=r'D:\pycharm\Project_1\attribution\NAA\NAA-master\image')
    parser.add_argument('--adv_path', type=str, default=r'D:\pycharm\Project_1\attribution\NAA\NAA-master\adv_new\NAA_inc_Res_v2_PD')
    parser.add_argument('--batch_size', type=int, default=50)
    args = parser.parse_args()
    main(args)
