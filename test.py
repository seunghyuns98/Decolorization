import os
import sys
import torch
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
    print_args(args)

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
    dataset_names = dataloader.get_dataset(args.dataset_test)

    test_set = dataset_names(root_dir=args.dataset_path, split='test')

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1,
        num_workers=args.workers, shuffle=True, drop_last=True)

    checkpoint = torch.load(os.path.join(exp_dir, 'step2_best_l1_cls.pth.tar'))
    Gridnet.load_state_dict(checkpoint['state_dict'])

    test(test_loader, Gridnet, Encoder, proxies)


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


def test(test_loader, Gridnet, Encoder, proxy):
    top1_psnr_o = AverageMeter()
    top1_psnr = AverageMeter()
    top1_ssim_o = AverageMeter()
    top1_ssim = AverageMeter()
    torch.cuda.empty_cache()
    Encoder.eval()
    Gridnet.eval()
    for data in tqdm(test_loader, mininterval=5):
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
        'Test  \t PNSR: {top1_psnr:.3f}, SSIM: {top1_ssim:.4f}, PNSR_O: {top1_psnr_o:.3f}, SSIM_O: {top1_ssim_o:.4f}'.format(
            top1_psnr=top1_psnr.avg, top1_ssim=top1_ssim.avg, top1_ssim_o=top1_ssim_o.avg, top1_psnr_o=top1_psnr_o.avg))

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


if __name__ == '__main__':
    main()
