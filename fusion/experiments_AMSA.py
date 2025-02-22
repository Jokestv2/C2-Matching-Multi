import os
import numpy as np
import mmcv
from fusion.non_nn_fuse import dst_aware_fuse_y_v3_AMSA
from util.utils import ImgFetcher, RefImgFetcher, RefImgFetcherWithRange, ImgFetcherWithRange, \
    resize_bgr_img_bicubic, resize_y_img_bicubic, resize_img_cv2, get_time_str
from util.eval_results import comp_psnr_ssim
from mmsr.data.transforms import mod_crop
import mmsr.utils.metrics as metrics
import util.visualize as visual

default_img_sr_path = \
    r'E:\CodeProjects\super_resolution\C2-Matching-Multi\results\amsa_gan_multi\visualization\CUFED5'
default_sr_name_format = '{idx:03d}_{ref_idx}_AMSA_gan.png'

# default_img_sr_path = \
#     r'E:\CodeProjects\super_resolution\C2-Matching-Multi\results\C2_matching_gan_multi\visualization\CUFED5'
# default_sr_name_format = '{idx:03d}_{ref_idx}_C2_matching_gan_multi.png'


def evaluate_dst_aware_fuse_y_v3_CUFED5(saved_folder, beta2_list, beta_list, plot_images=False, save_metrics=False):
    num_input = 126
    num_ref = 3

    # create fetcher for RefSR images
    img_sr_path = default_img_sr_path
    sr_name_format = default_sr_name_format
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
    opts['fused_filename_format'] = '{idx:03d}_fused.png'
    opts['scale'] = 4
    opts['plot_weight_mask'] = plot_images
    opts['plot_result_sr'] = plot_images

    avg_psnr_mat = np.zeros((len(beta2_list), len(beta_list)))
    avg_psnr_y_mat = np.zeros((len(beta2_list), len(beta_list)))
    avg_ssim_y_mat = np.zeros((len(beta2_list), len(beta_list)))
    for (i, beta2) in enumerate(beta2_list):
        for (j, beta) in enumerate(beta_list):
            avg_psnr, avg_psnr_y, avg_ssim_y \
                = dst_aware_fuse_y_v3_AMSA(img_sr_fetcher, img_gt_fetcher, opts, beta=beta, beta2=beta2)
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


def evaluate_dst_aware_fuse_y_v3_CUFED5_num_sr(saved_folder, beta2_list, beta_list, plot_images=False, save_metrics=False, num_ref=5):
    num_input = 126

    # create fetcher for RefSR images
    img_sr_path = default_img_sr_path
    sr_name_format = default_sr_name_format
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
    opts['fused_filename_format'] = '{idx:03d}_fused.png'
    opts['scale'] = 4
    opts['plot_weight_mask'] = plot_images
    opts['plot_result_sr'] = plot_images

    avg_psnr_mat = np.zeros((len(beta2_list), len(beta_list)))
    avg_psnr_y_mat = np.zeros((len(beta2_list), len(beta_list)))
    avg_ssim_y_mat = np.zeros((len(beta2_list), len(beta_list)))
    for (i, beta2) in enumerate(beta2_list):
        for (j, beta) in enumerate(beta_list):
            avg_psnr, avg_psnr_y, avg_ssim_y \
                = dst_aware_fuse_y_v3_AMSA(img_sr_fetcher, img_gt_fetcher, opts, beta=beta, beta2=beta2)
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


def evaluate_dst_aware_fuse_y_v3_CUFED5_single_img(saved_folder, img_idx, beta2_list, beta_list,
                                                   plot_images=False, save_metrics=False, num_ref=5):
    num_input = 126
    num_ref = 5
    lower_bound = img_idx
    upper_bound = img_idx + 1

    # create fetcher for RefSR images
    img_sr_path = default_img_sr_path
    sr_name_format = default_sr_name_format
    img_sr_fetcher = RefImgFetcherWithRange(lower_bound=lower_bound, upper_bound=upper_bound, num_ref=num_ref,
                                            img_path=img_sr_path, name_format_str=sr_name_format)

    # create fetcher for ground truth images
    img_gt_path = r'E:\CodeProjects\super_resolution\dataset\CUFED_SRNTT\CUFED5'
    gt_name_format = '{idx:03d}_0.png'
    img_gt_fetcher = ImgFetcherWithRange(lower_bound=lower_bound, upper_bound=upper_bound,
                                         img_path=img_gt_path,name_format_str=gt_name_format)

    # make root folder for fused images
    fused_folder_name = saved_folder
    fused_root_path = os.path.join(os.path.dirname(img_sr_path), fused_folder_name)
    dir_paths = os.path.expanduser(fused_root_path)
    os.makedirs(dir_paths, mode=0o777, exist_ok=True)

    # fuse and save image
    opts = {}
    opts['fused_root_path'] = fused_root_path
    opts['fused_filename_format'] = '{idx:03d}_fused.png'
    opts['scale'] = 4
    opts['plot_weight_mask'] = plot_images
    opts['plot_result_sr'] = plot_images

    avg_psnr_mat = np.zeros((len(beta2_list), len(beta_list)))
    avg_psnr_y_mat = np.zeros((len(beta2_list), len(beta_list)))
    avg_ssim_y_mat = np.zeros((len(beta2_list), len(beta_list)))
    for (i, beta2) in enumerate(beta2_list):
        for (j, beta) in enumerate(beta_list):
            avg_psnr, avg_psnr_y, avg_ssim_y \
                = dst_aware_fuse_y_v3_AMSA(img_sr_fetcher, img_gt_fetcher, opts, beta=beta, beta2=beta2)
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
        print('\n' + time_str)
    return time_str



def do_experiements_quantitative_beta():
    saved_folder = 'fused_experiments_quantitative'

    # beta2_list = [0]
    beta_list = np.arange(0, 930, 30)

    beta2_list = np.arange(0, 8.2, 0.2)
    # beta_list = [0]

    plot_images = False
    save_metrics = True
    time_str = evaluate_dst_aware_fuse_y_v3_CUFED5(saved_folder, beta2_list, beta_list, plot_images, save_metrics)
    # visual.visual_beta_metrics(saved_folder, time_str)

def do_experiements_quantitative_num_sr():
    saved_folder = 'fused_experiments_quantitative_num_sr'

    beta2_list = [8]
    # beta_list = np.arange(0, 840, 30)
    #
    # beta2_list = np.arange(0, 8.2, 0.2)
    beta_list = [90]
    plot_images = False
    save_metrics = True

    time_str_list = []
    for i in range(1, 6):
        time_str = evaluate_dst_aware_fuse_y_v3_CUFED5_num_sr(saved_folder, beta2_list, beta_list, plot_images, save_metrics, i)
        time_str_list.append(time_str)
    print(time_str_list)
    # visual.visual_beta_metrics(saved_folder, time_str)

def do_experiments_qualitative():
    saved_folder = 'fused_experiments_qualitative'
    # img_idx = 39
    # img_idx = 10
    # img_idx = 80
    # img_idx = 93
    # img_idx = 65
    img_idx = 91
    # # inital search

    # beta2_list = np.arange(0, 30, 3)
    # beta_list = np.arange(0, 200, 30)

    # # precise search 1
    # beta2_list = np.arange(0, 6, 0.3)
    # beta_list = np.arange(0, 30, 3)

    # precise search 2
    # beta2_list = np.arange(0, 6, 1)
    # beta_list = np.arange(330, 390, 10)

    # beta2_list = np.arange(1, 4, 0.5)
    # beta_list = np.arange(0, 10, 1)

    beta2_list = [3.6]
    beta_list = [0]

    plot_images = True
    # plot_images = False
    save_metrics = True
    for num_ref in [3]:
        time_str = evaluate_dst_aware_fuse_y_v3_CUFED5_single_img(saved_folder, img_idx, beta2_list, beta_list,
                                                              plot_images, save_metrics, num_ref)
    # visual.visual_beta_metrics(saved_folder, time_str)


if __name__ == '__main__':
    do_experiements_quantitative_beta()
    # do_experiements_quantitative_num_sr()
    # do_experiments_qualitative()