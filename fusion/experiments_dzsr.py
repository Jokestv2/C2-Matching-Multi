import os
import numpy as np
import mmcv
from fusion.non_nn_fuse import dst_aware_fuse_y_v3
from util.utils import ImgFetcher, RefImgFetcher, RefImgFetcherWithRange, ImgFetcherWithRange, \
    resize_bgr_img_bicubic, resize_y_img_bicubic, resize_img_cv2, get_time_str
from util.eval_results import comp_psnr_ssim
from mmsr.data.transforms import mod_crop
import mmsr.utils.metrics as metrics
import util.visualize as visual

def evaluate_dst_aware_fuse_y_v3_CUFED5(saved_folder, beta2_list, beta_list, plot_images=False, save_metrics=False):
    num_input = 126
    num_ref = 5

    # create fetcher for RefSR images
    img_sr_path = \
        r'E:\CodeProjects\super_resolution\C2-Matching-Multi\results\C2_matching_gan_multi\visualization\sr_full_401'
    sr_name_format = '{idx:03d}_{ref_idx}_x4_DZSR.png'
    img_sr_fetcher = RefImgFetcher(num_input=num_input, num_ref=num_ref, img_path=img_sr_path,
                                   name_format_str=sr_name_format)

    # create fetcher for ground truth images
    img_gt_path = r'E:\CodeProjects\super_resolution\dataset\CUFED_SRNTT\CUFED5'
    gt_name_format = '{idx:03d}_0.png'
    img_gt_fetcher = ImgFetcher(num_input=num_input, img_path=img_gt_path, name_format_str=gt_name_format)

    # make root folder for fused images
    fused_folder_name = saved_folder
    fused_root_path = os.path.join(os.path.dirname(img_sr_path), fused_folder_name)
    dir_paths = os.path.expanduser(fused_root_path)
    os.makedirs(dir_paths, mode=0o777, exist_ok=True)

    # fuse and save image
    opts = {}
    opts['fused_root_path'] = fused_root_path
    opts['fused_filename_format'] = '{idx:03d}_x4_DZSR_fused.png'
    opts['scale'] = 4
    opts['plot_weight_mask'] = plot_images
    opts['plot_result_sr'] = plot_images

    avg_psnr_mat = np.zeros((len(beta2_list), len(beta_list)))
    avg_psnr_y_mat = np.zeros((len(beta2_list), len(beta_list)))
    avg_ssim_y_mat = np.zeros((len(beta2_list), len(beta_list)))
    for (i, beta2) in enumerate(beta2_list):
        for (j, beta) in enumerate(beta_list):
            avg_psnr, avg_psnr_y, avg_ssim_y \
                = dst_aware_fuse_y_v3(img_sr_fetcher, img_gt_fetcher, opts, beta=beta, beta2=beta2)
            print(f"For beta2: {beta2}, beta: {beta}\n")
            print(f" PSNR: {avg_psnr}\n PSNR_Y: {avg_psnr_y}\n SSIM_Y: {avg_ssim_y}\n")
            avg_psnr_mat[i, j] = avg_psnr
            avg_psnr_y_mat[i, j] = avg_psnr_y
            avg_ssim_y_mat[i, j] = avg_ssim_y
    # save metrics results
    time_str = get_time_str()
    if save_metrics:
        np.save(os.path.join(fused_root_path, f'{time_str}_avg_psnr_mat'), avg_psnr_mat)
        np.save(os.path.join(fused_root_path, f'{time_str}_avg_psnr_y_mat'), avg_psnr_y_mat)
        np.save(os.path.join(fused_root_path, f'{time_str}_avg_ssim_y_mat'), avg_ssim_y_mat)
        np.save(os.path.join(fused_root_path, f'{time_str}_beta2_list'), beta2_list)
        np.save(os.path.join(fused_root_path, f'{time_str}_beta_list'), beta_list)
        print(time_str)
    return time_str


def do_experiements_dzsr():
    saved_folder = 'sr_full_401_fused'
    beta2_list = [3]
    beta_list = [8]
    plot_images = True
    save_metrics = False
    time_str = evaluate_dst_aware_fuse_y_v3_CUFED5(saved_folder, beta2_list, beta_list, plot_images, save_metrics)
    visual.visual_beta_metrics(saved_folder, time_str)


if __name__ == '__main__':
    do_experiements_dzsr()