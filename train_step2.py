import os
import sys

import torch
from torch.optim import Adam
from torch.optim import lr_scheduler
import dataloader
import torch.nn.functional as F
from utils.args_parser import args_parser, print_args
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from models import network
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def main():
    global args, exp_dir, best_result, device, tb_freq

    args = args_parser()
    print('\n===> Starting a new experiment for 2nd phase')
    print_args(args)
    start_epoch = 0

    exp_dir = os.path.join('workspace/', args.workspace, args.exp)
    assert os.path.isdir(exp_dir), 'exp_path is wrong'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sys.path.append(exp_dir)
    num_cls = args.num_cls
    num_obj = args.num_obj
    vector_dim = args.vector_dim

    Gridnet = network.RGB2GRAY(vector_dim).to(device)
    Gridnet = nn.DataParallel(Gridnet)
    Encoder = network.Encoder(vector_dim, num_cls, num_obj).to(device)
    Encoder = nn.DataParallel(Encoder)

    print('\n==> Model was loaded successfully!')
    checkpoint = torch.load(os.path.join(exp_dir, 'step1.pth.tar'))
    Encoder.load_state_dict(checkpoint['state_dict'])
    proxies = checkpoint['state_dict_loss']['proxies']
    proxies = torch.nn.functional.normalize(proxies, p=2, dim=1)
    dataset_names = dataloader.get_dataset(args.dataset_step2)

    train_set = dataset_names(root_dir=args.dataset_path, split='train')
    val_set = dataset_names(root_dir=args.dataset_path, split='val')
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.step2_train_batch_size,
        num_workers=args.workers, shuffle=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.step2_test_batch_size,
        num_workers=args.workers, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(Gridnet.parameters(), lr=args.lr)
    lr_decayer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_psnr = 0
    best_ssim = 0
    best_epoch = 0

    if args.resume:
        print("\nLoading Pretrained Model ##################3")
        checkpoint = torch.load(os.path.join(exp_dir, 'step2_latest.pth.tar'))
        start_epoch = checkpoint['epoch'] +1
        if 'best_psnr' in checkpoint.keys():
            best_psnr = checkpoint['best_psnr']
            best_ssim = checkpoint['best_ssim']
            best_epoch = checkpoint['best_epoch']
        Gridnet.load_state_dict(checkpoint['state_dict'])
        for i in range(start_epoch):
            lr_decayer.step()

    for epoch in range(start_epoch, args.epochs):
        print('\n==> Training Epoch [{}] (lr={})'.format(epoch, optimizer.param_groups[0]['lr']))

        train(train_loader, Gridnet, Encoder, proxies, optimizer, epoch)
        lr_decayer.step()
        if epoch % 10 == 0:
            psnr, ssim = valid(val_loader, Gridnet, Encoder, proxies, epoch)

            if psnr > best_psnr:
                best_psnr = psnr
                best_ssim = ssim
                best_epoch = epoch
                torch.save({'state_dict': Gridnet.state_dict()},
                           os.path.join(exp_dir, 'step2_best.pth.tar'))
            torch.save({'state_dict': Gridnet.state_dict(), 'epoch': epoch, 'best_psnr': best_psnr, 'best_ssim': best_ssim,
                        'best_epoch': best_epoch},
                       os.path.join(exp_dir, 'step2_latest.pth.tar'))

            print(
                'Best validation epoch: {0} PSNR: {top_psnr:.3f} TOP SSIM: {top_ssim:.4f} '.format(
                    best_epoch,
                    top_psnr=best_psnr, top_ssim=best_ssim))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


############ TRAINING FUNCTION ############
def train(train_loader, Gridnet, Encoder, proxies, optimizer, epoch):
    losses = AverageMeter()

    Encoder.eval()
    Gridnet.train()

    for data in tqdm(train_loader, mininterval=5):
        rgb_img = data[0].to(device)
        gt_img = data[1].to(device)
        gray_img = data[2].to(device)
        style_idx = data[-1].to(device)
        target_proxy = proxies.to(device)
        source_proxy, _, _ = Encoder(gray_img)
        target_proxy = target_proxy[style_idx]
        for i in range(len(style_idx)):
            if style_idx[i] // 4 == 2:
                target_proxy[i] = source_proxy[i]
        output, _ = Gridnet(rgb_img, source_proxy, target_proxy, gray_img)
        _ , style_ind, _ = Encoder(output)
        loss1 = F.l1_loss(gt_img, output)
        loss2 = F.cross_entropy(style_ind, style_idx //4)
        loss = loss1 + loss2*0.01
        losses.update(loss.item(), args.step2_train_batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Train [{0}] \t Loss: {loss:.5f}'.format(epoch, loss=losses.avg))


def valid(val_loader, Gridnet, Encoder, proxy, epoch):
    top1_psnr_o = AverageMeter()
    top1_psnr = AverageMeter()
    top1_ssim_o = AverageMeter()
    top1_ssim = AverageMeter()
    torch.cuda.empty_cache()
    Encoder.eval()
    Gridnet.eval()
    for data in tqdm(val_loader, mininterval=5):
        rgb_img = data[0].to(device)
        gt_img = data[1].to(device)
        gray_img = data[2].to(device)
        style_idx = data[-1].to(device)
        target_proxy = proxy.to(device)
        count_o = 0

        with torch.no_grad():
            source_proxy, _, _ = Encoder(gray_img)
            target_proxy = target_proxy[style_idx]
            for i in range(len(style_idx)):
                if style_idx[i] // 4 == 2:
                    count_o += 1
                    target_proxy[i] = source_proxy[i]

            output, _ = Gridnet(rgb_img, source_proxy, target_proxy, gray_img)
        psnr, ssim, psnr_o, ssim_o = psnr_ssim(output, gt_img, style_idx, args.eval_border)
        if count_o != 0:
            top1_psnr_o.update(psnr_o, count_o)
            top1_ssim_o.update(ssim_o, count_o)

        if len(gt_img) - count_o != 0:
            top1_psnr.update(psnr, len(gt_img) - count_o)
            top1_ssim.update(ssim, len(gt_img) - count_o)

    print(
        'Valid [{0}] \t PNSR: {top1_psnr:.3f}, SSIM: {top1_ssim:.4f}, PNSR_O: {top1_psnr_o:.3f}, SSIM_O: {top1_ssim_o:.4f}'.format(
            epoch, top1_psnr=top1_psnr.avg, top1_ssim=top1_ssim.avg, top1_ssim_o=top1_ssim_o.avg, top1_psnr_o=top1_psnr_o.avg))

    return top1_psnr.avg, top1_ssim.avg


def psnr_ssim(gt_img, predict_img, style_idx, eval_border):
    gt_img = gt_img[:,:,eval_border:-eval_border,eval_border:-eval_border]
    predict_img = predict_img[:,:,eval_border:-eval_border,eval_border:-eval_border]
    gt_np = gt_img.permute([0, 2, 3, 1]).detach().cpu().numpy()
    output_np = predict_img.permute([0, 2, 3, 1]).detach().cpu().numpy()
    gt_np = np.clip(gt_np * 255.0, 0, 255).astype(np.uint8)
    output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
    psnr = 0
    psnr_o = 0
    ssim = 0
    ssim_o = 0
    count_o = 0
    for j in range(len(output_np)):
        if style_idx[j] // 4 == 2:
            count_o += 1
            psnr_o += compare_psnr(output_np[j], gt_np[j], data_range=255)
            ssim_o += compare_ssim(output_np[j], gt_np[j], data_range=255, multichannel=True)

        else:
            psnr += compare_psnr(output_np[j], gt_np[j], data_range=255)
            ssim += compare_ssim(output_np[j], gt_np[j], data_range=255, multichannel=True)

    if count_o != 0:
        psnr_o = psnr_o / count_o
        ssim_o = ssim_o / count_o
    else:
        psnr_o = 0
        ssim_o = 0

    if len(output_np) - count_o != 0:
        psnr = psnr / (len(output_np) - count_o )
        ssim = ssim / (len(output_np) - count_o )
    else:
        psnr = 0
        ssim = 0

    return psnr, ssim, psnr_o, ssim_o


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
