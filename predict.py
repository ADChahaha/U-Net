import torch
import numpy as np
from model import UNet
import argparse
import os
import re
import torchvision
import cv2

parser = argparse.ArgumentParser("Predict parser")
parser.add_argument("image_path", type=str, help="The path of image which will be predicted", default="")
parser.add_argument("-c", "--checkpoint_path", type=str, help="Checkpoint path", default="")
parser.add_argument("-s", "--save_path", type=str, help="Result save path", default="")

args = parser.parse_args()

if args.checkpoint_path == "" or args.save_path == "":
    print("checkpoint_path or save_path is empty!")
    exit(-1)

feature_maps = []
def hook(module, input, output):
    feature_maps.append(output)

def TensorToCv(image):
    #转换为numpy数组
    result = image.numpy()
    # 变换为 HWC 格式（Height, Width, Channels）
    result = np.transpose(result, (1, 2, 0))
    # 将像素值映射到 [0, 255] 并转换为 uint8
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    # 如果是 RGB 转换为 BGR
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    return result


with torch.no_grad():
    #加载模型
    model = UNet()
    conv1_handle = model.conv1.register_forward_hook(hook)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    #转换为tensor
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((256, 256)), 
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize(mean=0,std=1)])
    image = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)
    image = image.unsqueeze(0)
    predict = model(image)
    #二值化
    predict = predict.squeeze(0)
    predict = predict > 0.5
    result = TensorToCv(predict)
    #保存结果
    pattern = ".*\/(.*)"
    image_name = re.match(pattern, args.image_path).group(1)
    result_path = os.path.join(args.save_path, image_name)
    cv2.imwrite(result_path, result)
    for output in feature_maps:
        output.squeeze_(0)
        for i in range(output.shape[0]):
            print(output.shape)
            channel = TensorToCv(output[i].unsqueeze(0))
            cv2.namedWindow("channel")
            cv2.imshow("channel", channel)
            cv2.waitKey(1000)
    conv1_handle.remove()
