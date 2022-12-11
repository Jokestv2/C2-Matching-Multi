import os
import numpy as np
from fusion.non_nn_fuse import dst_aware_fuse_y_v3
from util.utils import get_time_str, RefImgFetcherByIndexList, ImgFetcherByIndexList

# default_img_sr_path = \
#     r'E:\CodeProjects\super_resolution\C2-Matching-Multi\results\amsa_gan_multi\visualization\CUFED5'
# default_sr_name_format = '{idx:03d}_{ref_idx}_AMSA_gan.png'

CUFED5_spec = {
    'img_sr_path': r'E:\CodeProjects\super_resolution\C2-Matching-Multi\results\C2_matching_gan_multi\visualization\CUFED5',
    'sr_name_format': '{idx:03d}_{ref_idx}_C2_matching_gan_multi.png',
    'img_gt_path': r'E:\CodeProjects\super_resolution\dataset\CUFED_SRNTT\CUFED5',
    'gt_name_format': '{idx:03d}_0.png'
}

Landmark_spec = {
    'img_sr_path': r'E:\CodeProjects\super_resolution\C2-Matching-Multi\results\C2_matching_gan_multi\visualization\Landmark',
    'sr_name_format': '{idx:03d}_{ref_idx}_C2_matching_gan_multi.png',
    'img_gt_path': r'E:\CodeProjects\super_resolution\dataset\Landmark\testset',
    'gt_name_format': '{idx:03d}_0.png'
}


def evaluate_fuse_method(img_sr_fetcher, img_gt_fetcher, beta2_list, beta_list, opts):
    opts = opts.copy()
    saved_folder = opts['saved_folder']
    save_metrics = opts['save_metrics']
    plot_images = opts['plot_images']
    img_sr_path = opts['img_sr_path']

    # make root folder for fused images
    fused_folder_name = saved_folder
    fused_root_path = os.path.join(os.path.dirname(img_sr_path), fused_folder_name)
    dir_paths = os.path.expanduser(fused_root_path)
    os.makedirs(dir_paths, mode=0o777, exist_ok=True)

    # fuse and save image (this opts entries are set for the fusing method to use)
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


def do_experiements():
    saved_folder = 'fused_experiments_quantitative'
    dataset_spec = Landmark_spec

    index_list = range(0, 150)
    # index_list = [0]
    num_ref_list = [4]

    beta2_list = np.arange(0, 20, 1)
    beta_list = np.arange(0, 820, 60)

    plot_images = False
    save_metrics = True

    # beta2_list = np.arange(0, 8.2, 0.2)
    # beta_list = [0]

    # Normally you don't neeed to modity the lines behind
    img_sr_path = dataset_spec['img_sr_path']
    sr_name_format = dataset_spec['sr_name_format']
    img_gt_path = dataset_spec['img_gt_path']
    gt_name_format = dataset_spec['gt_name_format']

    opts = {}
    opts['img_sr_path'] = img_sr_path
    opts['saved_folder'] = saved_folder
    opts['plot_images'] = plot_images
    opts['save_metrics'] = save_metrics

    time_str_list = []
    for num_ref in num_ref_list:
        # create fetcher for RefSR images
        img_sr_fetcher = RefImgFetcherByIndexList(index_list=index_list, num_ref=num_ref, img_path=img_sr_path,
                                                  name_format_str=sr_name_format)

        # create fetcher for ground truth images
        img_gt_fetcher = ImgFetcherByIndexList(index_list=index_list, img_path=img_gt_path, name_format_str=gt_name_format)

        time_str = evaluate_fuse_method(img_sr_fetcher, img_gt_fetcher, beta2_list, beta_list, opts)
        time_str_list.append(time_str)
    print(saved_folder)
    print(time_str_list)


if __name__ == '__main__':
    do_experiements()
