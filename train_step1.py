import os
import sys
import torch
import dataloader
import torch.nn.functional as F
from utils.args_parser import args_parser, print_args
from loss.losses import get_loss_fn
from tqdm import tqdm
import torch.nn as nn
from models import network


def main():
    global args, exp_dir, best_result, device, tb_freq

    args = args_parser()
    print('\n==> Starting a new experiment')
    print_args(args)
    start_epoch = 0

    exp_dir = os.path.join('workspace/', args.workspace, args.exp)
    assert os.path.isdir(exp_dir), 'exp_path is wrong'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sys.path.append(exp_dir)

    num_cls = args.num_cls
    num_obj = args.num_obj
    vector_dim = args.vector_dim
    model = network.Encoder(vector_dim, num_cls, num_obj).to(device)
    model = nn.DataParallel(model)
    print('\n==> Model was loaded successfully!')

    # Dataloader
    dataset_names = dataloader.get_dataset(args.dataset_step1)
    train_set = dataset_names(root_dir=args.dataset_path)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.step1_train_batch_size,
        num_workers=args.workers, shuffle=True, drop_last=True)

    # Create Loss
    loss = get_loss_fn(args)(num_cls, num_obj, 512)

    # Create Optimizer
    param_groups = [{'params': model.parameters(), 'lr': args.lr}, {'params': loss.parameters(), 'lr': args.lr*100}]
    optimizer = torch.optim.Adam(param_groups)
    lr_decayer = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.resume:
        print("\nLoading Pretrained Model ##################3")
        checkpoint = torch.load(os.path.join(exp_dir, 'step1.pth.tar'))
        start_epoch = checkpoint['epoch'] +1
        model.load_state_dict(checkpoint['state_dict'])
        for i in range(start_epoch):
            lr_decayer.step()

    for epoch in range(start_epoch, args.epochs+1):
        print('\n==> Training Epoch [{}] (lr={})'.format(epoch, optimizer.param_groups[0]['lr']))

        train(train_loader, model, optimizer, epoch, loss, args.styl_w, args.ce_w)

        torch.save({'state_dict': model.state_dict(),
                    'state_dict_loss': loss.state_dict(),
                    'epoch': epoch},
                   os.path.join(exp_dir, 'step1.pth.tar'))



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
def train(train_loader, model, optimizer, epoch, objectives, w_1, w_2):
    losses = AverageMeter()
    model.train()
    objectives.train()
    for (source_img, style_id, object_id, s_o_id) in tqdm(train_loader):

        source_img = source_img.to(device)
        style_id = style_id.to(device)
        object_id = object_id.to(device)
        s_o_id = s_o_id.to(device)
        embeddings, style_Ind, object_Ind = model(source_img)
        loss, _, _, _, _ = objectives(embeddings, style_Ind, object_Ind, style_id, object_id, s_o_id, w_1, w_2)
        losses.update(loss.item(), args.step1_train_batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Train [{0}] \t Loss: {loss:.2f}'.format(epoch, loss=losses.avg))
    return losses.avg


if __name__ == '__main__':
    main()
