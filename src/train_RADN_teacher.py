import argparse

import os



import torch

from tensorboardX import SummaryWriter

from torch import nn, optim

from torch.autograd import Variable

from torch.utils.data import DataLoader


from losses import EdgeLoss


from utils import SSIM

from torch.optim.lr_scheduler import MultiStepLR

from dataset.select_dataset import define_Dataset

from model.select_model import define_model

from utils import  batch_PSNR



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"






def test(model, opt, epoch):

    print('start test:')

    model.eval()


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = SSIM().to(device)

    dataset_val = define_Dataset("val", opt.dataset)

    loader_val = DataLoader(dataset=dataset_val, num_workers=0, batch_size=1, shuffle=False,

                            drop_last=False)

    save_file = os.path.join('save', opt.model, opt.dataset, 'logs.txt')

    f = open(save_file, 'a+')

    ssim_sum = 0

    psnr_sum = 0

    with torch.no_grad():

        for i, (input_train, target_train) in enumerate(loader_val, 0):
            input_train, target_train = Variable(input_train), Variable(target_train)

            input = input_train.to(device)

            target = target_train.to(device)

            input = input - target

            out_train, _ = model(input)
            ssim = criterion(input, out_train)

            out_train = torch.clamp(out_train, 0., 1.)  # 裁剪函数
            psnr_train = batch_PSNR(out_train, input, 1.)

            ssim_sum = ssim_sum + ssim
            psnr_sum = psnr_sum + psnr_train



    print(f'the {epoch}th test is over, result: ')

    print(f'psnr = {psnr_sum / len(dataset_val)}, ssim = {ssim_sum / len(dataset_val)}')

    f.write(f'the {epoch}th test result:\n')

    f.write(f'psnr = {psnr_sum / len(dataset_val)} \n')

    f.write(f'ssim: {ssim_sum / len(dataset_val)}\n')



def main():

    '''



       # ----------------------------------------



       # Step--1 (prepare opt)



       # ----------------------------------------



    '''

    parser = argparse.ArgumentParser(description="model_train")

    parser.add_argument("--batch_size", type=int, default=25, help="Training batch size")

    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")

    parser.add_argument("--milestone", type=int, default=[50,100,200,500], help="When to decay learning rate")

    parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")

    parser.add_argument("--save_freq", type=int, default=1, help='save intermediate model')

    parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')

    parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')

    parser.add_argument("--iter_val", type=int, default=1000, help='iter val')

    parser.add_argument("--iter_epoch", type=int, default=1, help='iter epoch')

    parser.add_argument("--dataset", type=str, default='rain800', help='dataset name(rain100L)')

    parser.add_argument('--model', type=str, default='Recurrent_T', help='model_name')

    parser.add_argument('--use_cl', type=bool, default=False, help='use curriculum learning')

    parser.add_argument("--lambda_wav", type=float, default=10, help='weight of wav loss ')

    parser.add_argument("--lambda_vgg", type=float, default=0.1, help='weight of vgg loss ')

    parser.add_argument("--lambda_edge", type=float, default=0.1, help='weight of edge loss ')

    opt = parser.parse_args()

    if opt.use_gpu:

        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

    save_path = os.path.join('save', opt.model, opt.dataset)

    if not os.path.isdir(save_path):

        os.makedirs(save_path)



    '''
           # ----------------------------------------



           # Step--2 (prepare dataset)



           # ----------------------------------------
    '''

    print('Loading dataset ...\n')

    dataset_train = define_Dataset("train", opt.dataset)

    loader_train = DataLoader(dataset=dataset_train, num_workers=0, batch_size=opt.batch_size, shuffle=True,

                              drop_last=True)

    print("# of training samples: %d\n" % int(len(dataset_train)))

    '''



           # ----------------------------------------



           # Step--3 (prepare module)



           # ----------------------------------------



    '''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = define_model(opt).to(device)

    # loss function
    criterion1 = nn.MSELoss(size_average=True).to(device)
    edge_loss = EdgeLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.5)  # learning rates
    scheduler.step()
    # record training

    writer = SummaryWriter(save_path)
    '''
             # ----------------------------------------



             # Step--3 (start training)



             # ----------------------------------------
      '''

    step = 0

    p = 0

    train_loss_sum = 0

    psnr_value = 0

    for epoch in range(opt.epochs):

        current_psnr = 0

        scheduler.step(step)

        for param_group in optimizer.param_groups:

            print('learning rate %f' % param_group["lr"])

        ## epoch training start

        for i, (input, target) in enumerate(loader_train, 0):

            model.train()

            model.zero_grad()

            optimizer.zero_grad()

            input_train, target_train = Variable(input).to(device), Variable(target).to(device)

            # get the rain streaks
            input_train = input_train - target_train

            out,_ = model(input_train)

            pixel_metric = criterion1(input_train, out)

            edge_los = edge_loss(input_train, out)

            loss =  pixel_metric + opt.lambda_edge * edge_los

            loss.backward()

            optimizer.step()

            if i % 2 == 0:  # 每两次训练，loss累加

                p = p + 1

                train_loss_sum = train_loss_sum + loss.item()

            out1_train = torch.clamp(out, 0., 1.)

            psnr_train = batch_PSNR(out1_train, input_train, 1.)

            current_psnr += psnr_train

            print("[epoch %d][%d/%d] loss: %.4f,PSNR_train: %.4f,mse=%.4f" %

                  (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train, pixel_metric.item() ))

        model.eval()

        # log the images

        if step % 10 == 0:

            # Log the scalar values

            writer.add_scalar('loss', loss.item(), step)

            writer.add_scalar('PSNR on training data', psnr_train, step)


        if step % opt.iter_epoch == 0:

            # test

            test(model,opt,epoch)

            # Log the scalar values

            epoch_loss = train_loss_sum / p

            writer.add_scalar('epoch_loss', epoch_loss, step)

            p = 0

            train_loss_sum = 0

        step += 1

        if epoch % opt.save_freq == 0:

            torch.save(model.state_dict(), os.path.join(save_path, 'net_latest.pth'))

            torch.save(model.state_dict(), os.path.join(save_path, 'net_step%d.pth' % (step)))

        if psnr_value < current_psnr:

            psnr_value = current_psnr

            torch.save(model.state_dict(), os.path.join(save_path, 'best.pth'))

            print(f'The best performence is the {epoch + 1}th train')

    print('finish')



if __name__ == '__main__':

    main()