import torch
import torch.nn.functional as F
from torch import nn
from . import utils

def join_reim_mag_output(tensor):
    return torch.abs(utils.join_reim_tensor(tensor)).unsqueeze(1)

def fourier_loss(output_domain, loss):
    if output_domain == 'FREQ':
        if loss == 'L1':
            def fourier_l1(y_true, y_pred):
                y_true = join_reim_mag_output(y_true)
                y_pred = join_reim_mag_output(y_pred)
                return torch.mean(torch.abs(y_true - y_pred))
            return fourier_l1
        elif loss == 'L2':
            def fourier_l2(y_true, y_pred):
                y_true = join_reim_mag_output(y_true)
                y_pred = join_reim_mag_output(y_pred)
                return torch.mean(torch.pow(torch.abs(y_true - y_pred), 2))
            return fourier_l2
    elif output_domain == 'IMAGE':
        if loss == 'L1':
            def fourier_l1(y_true, y_pred):
                y_true_fourier = utils.convert_tensor_to_frequency_domain(y_true)
                y_pred_fourier = utils.convert_tensor_to_frequency_domain(y_pred)
                y_true = utils.join_reim_tensor(y_true_fourier)
                y_pred = utils.join_reim_tensor(y_pred_fourier)
                return torch.mean(torch.abs(y_true - y_pred))
            return fourier_l1
        elif loss == 'L2':
            def fourier_l2(y_true, y_pred):
                y_true_fourier = utils.convert_tensor_to_frequency_domain(y_true)
                y_pred_fourier = utils.convert_tensor_to_frequency_domain(y_pred)
                y_true = utils.join_reim_tensor(y_true_fourier)
                y_pred = utils.join_reim_tensor(y_pred_fourier)
                return torch.mean(torch.pow(torch.abs(y_true - y_pred), 2))
            return fourier_l2

def comp_image_loss(output_domain, loss):
    if output_domain == 'IMAGE':
        if loss == 'L1':
            def image_l1(y_true, y_pred):
                return torch.mean(torch.abs(y_true - y_pred))
            return image_l1
        elif loss == 'L2':
            def image_l2(y_true, y_pred):
                return torch.mean(torch.pow(torch.abs(y_true - y_pred), 2))
            return image_l2
    elif output_domain == 'FREQ':
        if loss == 'L1':
            def image_l1(y_true, y_pred):
                y_true = utils.convert_tensor_to_image_domain(y_true)
                y_pred = utils.convert_tensor_to_image_domain(y_pred)
                return torch.mean(torch.abs(y_true - y_pred))
            return image_l1
        elif loss == 'L2':
            def image_l2(y_true, y_pred):
                y_true = utils.convert_tensor_to_image_domain(y_true)
                y_pred = utils.convert_tensor_to_image_domain(y_pred)
                return torch.mean(torch.pow(torch.abs(y_true - y_pred), 2))
            return image_l2

def image_loss(output_domain, loss):
    if output_domain == 'IMAGE':
        if loss == 'L1':
            def image_l1(y_true, y_pred):
                y_true = join_reim_mag_output(y_true)
                y_pred = join_reim_mag_output(y_pred)
                return torch.mean(torch.abs(y_true - y_pred))
            return image_l1
        elif loss == 'L2':
            def image_l2(y_true, y_pred):
                y_true = join_reim_mag_output(y_true)
                y_pred = join_reim_mag_output(y_pred)
                return torch.mean(torch.pow(torch.abs(y_true - y_pred), 2))
            return image_l2
    elif output_domain == 'FREQ':
        if loss == 'L1':
            def image_l1(y_true, y_pred):
                y_true_image = utils.convert_tensor_to_image_domain(y_true)
                y_pred_image = utils.convert_tensor_to_image_domain(y_pred)
                y_true = join_reim_mag_output(y_true_image)
                y_pred = join_reim_mag_output(y_pred_image)
                return torch.mean(torch.abs(y_true - y_pred))
            return image_l1
        elif loss == 'L2':
            def image_l2(y_true, y_pred):
                y_true_image = utils.convert_tensor_to_image_domain(y_true)
                y_pred_image = utils.convert_tensor_to_image_domain(y_pred)
                y_true = join_reim_mag_output(y_true_image)
                y_pred = join_reim_mag_output(y_pred_image)
                return torch.mean(torch.pow(torch.abs(y_true - y_pred), 2))
            return image_l2

def joint_img_freq_loss(output_domain, loss, loss_lambda):
    def joint_loss(y_true, y_pred):
        return (image_loss(output_domain, loss)(y_true, y_pred) + loss_lambda * fourier_loss(output_domain, loss)(y_true, y_pred))
    return joint_loss

def compute_ssim(img1, img2, window_size=11, sigma=1.5):
    """Simple SSIM computation without external dependencies."""
    # This is a simplified SSIM implementation
    # For production use, consider using a proper SSIM library
    mse = F.mse_loss(img1, img2)
    return 1.0 / (1.0 + mse)  # Simplified SSIM approximation

def ssim(output_domain):
    """SSIM loss function."""
    if output_domain == 'IMAGE':
        def image_ssim(y_true, y_pred):
            y_true = join_reim_mag_output(y_true)
            y_pred = join_reim_mag_output(y_pred)
            ssim_value = compute_ssim(y_true, y_pred)
            return -ssim_value  # SSIM is higher for better quality, so return negative for minimization
        return image_ssim
    elif output_domain == 'FREQ':
        def image_ssim(y_true, y_pred):
            y_true_image = utils.convert_tensor_to_image_domain(y_true)
            y_pred_image = utils.convert_tensor_to_image_domain(y_pred)
            y_true = join_reim_mag_output(y_true_image)
            y_pred = join_reim_mag_output(y_pred_image)
            ssim_value = compute_ssim(y_true, y_pred)
            return -ssim_value  # SSIM is higher for better quality, so return negative for minimization
        return image_ssim

def psnr(output_domain):
    """PSNR loss function."""
    if output_domain == 'IMAGE':
        def image_psnr(y_true, y_pred):
            y_true = join_reim_mag_output(y_true)
            y_pred = join_reim_mag_output(y_pred)
            mse = F.mse_loss(y_true, y_pred)
            psnr_value = 20 * torch.log10(1.0 / torch.sqrt(mse))
            return -psnr_value  # PSNR is higher for better quality, so return negative for minimization
        return image_psnr
    elif output_domain == 'FREQ':
        def image_psnr(y_true, y_pred):
            y_true_image = utils.convert_tensor_to_image_domain(y_true)
            y_pred_image = utils.convert_tensor_to_image_domain(y_pred)
            y_true = join_reim_mag_output(y_true_image)
            y_pred = join_reim_mag_output(y_pred_image)
            mse = F.mse_loss(y_true, y_pred)
            psnr_value = 20 * torch.log10(1.0 / torch.sqrt(mse))
            return -psnr_value  # PSNR is higher for better quality, so return negative for minimization
        return image_psnr