import torch.utils.tensorboard as tensorboard
import time
from tqdm import tqdm
import torchmetrics.classification
from dataset import KvasirSegDataset
from model import UNet
from torch.utils.data import DataLoader
import torchvision
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tensorboard
import torch
import json
import torchmetrics


def DiceLoss(predict, label, epsilon=1e-7):
    predict = torch.nn.functional.sigmoid((predict - 0.5) * 10000)
    label = torch.nn.functional.sigmoid((label - 0.5) * 100)
    intersection = torch.sum(predict * label)
    total = torch.sum(predict) + torch.sum(label)
    dice = (2.0 * intersection + epsilon) / (total + epsilon)
    return 1 - dice

if __name__ == "__main__":
    parser = argparse.ArgumentParser("U-Net trainer")
    parser.add_argument("--config_path", type=str, help="The config json path")
    parser.add_argument('-c', '--checkpoint', type=str, help="The checkpoint path",default='')
    args = parser.parse_args()
    config = {}
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    #训练需要
    device = torch.device(config['device'])
    model = UNet().to(device)
    # criterion = torchmetrics.classification.Dice(average='micro',multiclass=False).to(device)
    criterion = DiceLoss
    current_epoch = 0
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 20, 2)
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint)
        current_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    current_epoch += 1
    
    #tensorboard可视化
    logger = tensorboard.writer.SummaryWriter()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((256, 256)), 
         torchvision.transforms.ToTensor(), 
         torchvision.transforms.Normalize(mean=0,std=1)])
    #Dataloader
    train_dataset = KvasirSegDataset(config['train_image_path'], config['train_label_path'], device, transform)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataset = KvasirSegDataset(config['val_image_path'], config['val_label_path'], device, transform)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=True)

    #训练部分
    max_f1_score = 0
    for epoch in range(current_epoch, config['epoch']):
        model.train()
        #前向传播
        average_loss = 0
        pbar = tqdm(train_dataloader, desc="batch process")
        for i, (image, label) in enumerate(pbar):
            #清空grad
            optimizer.zero_grad()
            #预测
            predict = model(image)
            #得到损失
            loss = criterion(predict, label)
            #反向传播
            loss.backward()
            optimizer.step()
            average_loss += loss.item()
            #可视化loss
        average_loss /= len(train_dataloader)
        print(f"average_loss: {average_loss}")
        print(f"lr: {optimizer.param_groups[0]['lr']}")
        logger.add_scalar("loss/train", average_loss, epoch)
        logger.add_scalar("lr/train",optimizer.param_groups[0]['lr'], epoch)
        scheduler.step()
        #保存checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(checkpoint, f=f"{config['weights_path']}/epoch{epoch}.pth")

        #val
        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                val_pbar = tqdm(val_dataloader, desc="val process")
                accuracy_calculater = torchmetrics.classification.Accuracy(task='binary').to(device)
                precision_calculater = torchmetrics.classification.Precision(task='binary').to(device)
                recall_calculater = torchmetrics.classification.Recall(task='binary').to(device)
                f1score_calculater = torchmetrics.classification.F1Score(task='binary').to(device)
                for i, (image, label) in enumerate(val_pbar):
                    predict = model(image)
                    #可视化图片
                    if 0 == i:
                        logger.add_image('predict', predict[0], epoch)
                        logger.add_image('label', label[0], epoch)
                        # print(predict[0])
                    #移除channel，转换为batch_size*height*width
                    predict = predict.squeeze(1)
                    predict = predict.view(-1)
                    label = label.view(-1)
                    #评价效果
                    predict = predict > 0.5
                    label = label > 0.5
                    batch_accuracy = accuracy_calculater.update(predict, label)
                    batch_precision = precision_calculater.update(predict, label)
                    batch_recall = recall_calculater.update(predict, label)
                    batch_f1 = f1score_calculater.update(predict, label)
                accuracy = accuracy_calculater.compute()
                precision = precision_calculater.compute()
                recall = recall_calculater.compute()
                f1 = f1score_calculater.compute()
                print(f"epoch: {epoch}, accuracy: {accuracy}, precision: {precision},recall: {recall}, f1-score: {f1}")
                logger.add_scalar("accuracy/val", accuracy, epoch)
                logger.add_scalar("precision/val", precision, epoch)
                logger.add_scalar("recall/val", recall, epoch)
                logger.add_scalar("f1-score/val", f1, epoch)
                if f1 > max_f1_score:
                    max_f1_score = f1
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()
                    }
                    torch.save(checkpoint, f=f"outputs/best_f1.pth")
    logger.close()
        

