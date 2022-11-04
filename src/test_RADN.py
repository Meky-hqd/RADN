import argparse
import os


import torch
from PIL import Image
from torch.utils.data import DataLoader
from dataset.select_dataset import define_Dataset
from model.select_model import define_model
from utils import find_last_model, batch_PSNR, SSIM
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss


def pad_img(img,scale=4):
    b, c, h, w = img.shape
    if h % scale != 0:
        pad_size = scale - (h - h//scale *scale)
        h_pad = torch.zeros(b, c, pad_size, w)
        img = torch.cat((img, h_pad), 2)
        h += pad_size
        pass
    if w % scale != 0:
        pad_size = scale - (w - w // scale * scale)
        w_pad = torch.zeros(b, c, h, pad_size)
        img = torch.cat((img, w_pad), 3)

    return img
        
def main():
    '''

       # ----------------------------------------

       # Step--1 (prepare opt)

       # ----------------------------------------

    '''
    parser = argparse.ArgumentParser(description="model_test")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--milestone", type=int, default=[150, 300], help="When to decay learning rate")
    parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
    parser.add_argument("--save_freq", type=int, default=10, help='save intermediate model')
    parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
    parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
    parser.add_argument("--iter_val", type=int, default=1000, help='iter val')
    parser.add_argument("--iter_epoch", type=int, default=500, help='iter epoch')
    parser.add_argument("--dataset", type=str, default='rain100H', help='dataset name(rain100L)')
    parser.add_argument('--model', type=str, default='Recurrent_S', help='model_name')


    opt = parser.parse_args()
    save_path = os.path.join('save', opt.model, opt.dataset)

    if not os.path.isdir(save_path):
        raise FileExistsError('模型文件未找到')
    # 创建结果文件夹
    save_result = os.path.join('save', opt.model, opt.dataset, 'result')
    if not os.path.isdir(save_result):
        os.makedirs(save_result)
    '''

           # ----------------------------------------

           # Step--2 (prepare dataset)

           # ----------------------------------------

    '''
    print('Loading val dataset ...\n')
    dataset_val = define_Dataset("val", opt.dataset)
    loader_val = DataLoader(dataset=dataset_val, num_workers=0, batch_size=opt.batch_size, shuffle=False,
                            drop_last=False)
    '''

           # ----------------------------------------

           # Step--3 (get model)

           # ----------------------------------------

    '''
    trained_model = find_last_model(save_path)
    model_path = os.path.join(save_path, trained_model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = define_model(opt).to(device)
    model.load_state_dict(torch.load(model_path))
    '''

           # ----------------------------------------

           # Step--4 (start val)

           # ----------------------------------------

     '''
    criterion = SSIM().to(device)
    ssim_sum = 0
    psnr_sum = 0
    with torch.no_grad():

        for i, (input, target) in enumerate(loader_val, 0):
            predicted_image, _ = model(input.to(device))
            target = target.to(device)

            predicted_image = torch.clamp(predicted_image, 0., 1.)
            psnr = batch_PSNR(predicted_image, target, 1.)
            ssim = criterion(target, predicted_image)
            # 数据处理
            predicted_image = predicted_image[0,:,:,:]
            predicted_image = predicted_image.permute(1, 2, 0)
            predicted_image = predicted_image.cpu().detach().numpy()
            predicted_image = np.clip(predicted_image * 255, 0, 255).astype('uint8')

            # 数据保存
            img = Image.fromarray(predicted_image)
            img.save('{0}/{1}'.format(save_result,'norain'+ '-'+ str(i+1) +'.png'))
            print(f'save the {i} img:')
            print("SSIM = %.4f" % ssim)
            print("PSNR = %.4f" % psnr)
            ssim_sum += ssim
            psnr_sum += psnr
    avg_ssim = ssim_sum / len(dataset_val)
    avg_psnr = psnr_sum / len(dataset_val)
    print("---- Average SSIM = %.4f----" % avg_ssim)
    print("---- Average PSNR = %.4f----" % avg_psnr)


if __name__ == '__main__':

    main()