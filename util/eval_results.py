"""
Evaluate the performance of a model by calculating PSNR and SSIM of results
"""
from util.utils import ImgFetcher
import mmsr.utils.metrics as metrics
from mmsr.data.transforms import mod_crop

single_ref_sr_path = r'E:\CodeProjects\super_resolution\C2-Matching-Multi\results\C2_matching_gan_multi\visualization\CUFED5_fusion\A_Y'
single_ref_sr_file_format = '{idx}.png'
naive_fuse_path = r'E:\CodeProjects\super_resolution\C2-Matching-Multi\results\C2_matching_gan_multi\visualization\CUFED5_fused\naive_fuse'
naive_fuse_file_format = '{idx:03d}_fused.png'
dst_aware_fuse_path = r'E:\CodeProjects\super_resolution\C2-Matching-Multi\results\C2_matching_gan_multi\visualization\CUFED5_fused\dst_aware_fuse'
dst_aware_fuse_file_format = '{idx:03d}_fused.png'
img_in_path_file_format_dict = {
    'single_ref_sr':[single_ref_sr_path, single_ref_sr_file_format],
    'naive_fuse':[naive_fuse_path, naive_fuse_file_format],
    'dst_aware_fuse': [dst_aware_fuse_path, dst_aware_fuse_file_format]
}


def comp_psnr_ssim(img_gt, img_in, scale):
    psnr = metrics.psnr(img_in * 255, img_gt * 255, crop_border=scale)

    img_in_y = metrics.bgr2ycbcr(img_in, only_y=True)
    img_gt_y = metrics.bgr2ycbcr(img_gt, only_y=True)
    psnr_y = metrics.psnr(img_in_y * 255, img_gt_y * 255, crop_border=scale)

    ssim_y = metrics.ssim(img_in_y * 255, img_gt_y * 255, crop_border=scale)

    return psnr, psnr_y, ssim_y


def eval_psnr_ssim(img_in_path, in_name_format, num_input, scale):
    img_gt_path = r'E:\CodeProjects\super_resolution\dataset\CUFED_SRNTT\CUFED5'
    gt_name_format = '{idx:03d}_0.png'

    img_in_fetcher = ImgFetcher(num_input=num_input, img_path=img_in_path, name_format_str=in_name_format)
    img_gt_fetcher = ImgFetcher(num_input=num_input, img_path=img_gt_path, name_format_str=gt_name_format)

    avg_psnr = 0.
    avg_psnr_y = 0.
    avg_ssim_y = 0.
    for ((idx, img_in), (_, img_gt)) in zip(img_in_fetcher, img_gt_fetcher):
        # To be consistent with the process in RefCUFEDDataset.__getitem__()
        # Note that img_in has already been applied mod_crop() before saving, so no need to crop here
        img_gt = mod_crop(img_gt, scale=scale)

        psnr, psnr_y, ssim_y = comp_psnr_ssim(img_gt, img_in, scale)
        avg_psnr += psnr
        avg_psnr_y += psnr_y
        avg_ssim_y += ssim_y

    avg_psnr /= num_input
    avg_psnr_y /= num_input
    avg_ssim_y /= num_input
    print(f" PSNR: {avg_psnr}\n PSNR_Y: {avg_psnr_y}\n SSIM_Y: {avg_ssim_y}\n")


def main():
    scale = 4
    num_input = 126
    img_in_path = img_in_path_file_format_dict['dst_aware_fuse'][0]
    in_name_format = img_in_path_file_format_dict['dst_aware_fuse'][1]

    for beta2 in [3]:
        for beta in [6]:
            print(f"beta={beta}")
            # img_in_path_beta = img_in_path + f'_{beta}'
            # img_in_path_beta = img_in_path + f'_y_v2_{beta}'
            img_in_path_beta = img_in_path + f'_y_v3_{beta2}_{beta}'
            eval_psnr_ssim(img_in_path_beta, in_name_format, num_input, scale)


if __name__ == '__main__':
    main()