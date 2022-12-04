"""
Fuse outputs from RefSR (each output corresponds to the use of different Ref image)
based on non-neural-network-based approaches
"""
import os
import numpy as np
import mmcv
from util.utils import mkdirs, ImgFetcher, RefImgFetcher, resize_bgr_img_bicubic, resize_y_img_bicubic
from mmsr.data.transforms import mod_crop
import mmsr.utils.metrics as metrics


def get_original_input(img_gt_fetcher, opts):
    """
    Get original input, that is, usually the downsampled version of ground truth during
    """
    root_path = opts['fused_root_path']
    original_filename_format = '{idx:03d}_original_lr.png'
    scale = opts['scale']

    fused_path = os.path.join(root_path, 'original_lq')
    os.mkdir(fused_path)
    for (idx, img_gt) in img_gt_fetcher:
        img_gt = mod_crop(img_gt, scale=scale)

        gt_h, gt_w, _ = img_gt.shape
        lq_h, lq_w = gt_h // scale, gt_w // scale
        img_gt_lq = resize_bgr_img_bicubic(img_gt, lq_h, lq_w)
        img_gt_lq = (img_gt_lq * 255).astype(np.uint8)

        dst_file_name = original_filename_format.format(idx=idx)
        dst_file_path = os.path.join(fused_path, dst_file_name)
        mmcv.imwrite(img_gt_lq, dst_file_path)


def naive_fuse(img_in_fetcher, opts):
    """
    fuse images by simply taking the average of their intensities
    """
    fused_root_path = opts['fused_root_path']
    fused_filename_format = opts['fused_filename_format']

    fused_path = os.path.join(fused_root_path, 'naive_fuse')
    os.mkdir(fused_path)
    for input_idx, img_in_list in img_in_fetcher:
        img_fused = np.average(img_in_list, axis=0)
        img_fused = (img_fused * 255).astype(np.uint8)

        dst_file_name = fused_filename_format.format(idx=input_idx)
        dst_file_path = os.path.join(fused_path, dst_file_name)
        mmcv.imwrite(img_fused, dst_file_path)


def dst_aware_fuse_rgb(img_in_fetcher, img_gt_fetcher, opts, beta):
    """
    Destination Aware Fuse
    For each image to be fused, weight for each pixel is calculated according to the similarity between
    the local region to the input LR image
    beta controls the penalty given to incorrect RefSR pixels, beta should be larger than 0
    """
    fused_root_path = opts['fused_root_path']
    fused_filename_format = opts['fused_filename_format']
    scale = opts['scale']

    fused_path = os.path.join(fused_root_path, f'dst_aware_fuse_{beta}')
    os.mkdir(fused_path)
    for ((idx, img_gt), (_, img_in_list)) in zip(img_gt_fetcher, img_in_fetcher):
        img_gt = mod_crop(img_gt, scale=scale)
        gt_h, gt_w, _ = img_gt.shape
        lq_h, lq_w = gt_h // scale, gt_w // scale
        img_gt_lq = resize_bgr_img_bicubic(img_gt, lq_h, lq_w)
        weight_mask_list = []
        for img_sr in img_in_list:
            img_sr_lq = resize_bgr_img_bicubic(img_sr, lq_h, lq_w)
            # calculate average weight factor
            delta = np.abs(img_sr_lq - img_gt_lq)
            weight_mask_lq = np.exp(-beta * delta)
            weight_mask = resize_bgr_img_bicubic(weight_mask_lq, gt_h, gt_w)
            weight_mask_list.append(weight_mask)
        # normalize weight_mask by each pixel, so they sum to 1
        weight_mask_sum = np.sum(weight_mask_list, axis=0)
        img_fused = np.zeros_like(img_gt)
        for (img_sr, weight_mask) in zip(img_in_list, weight_mask_list):
            img_fused += img_sr * weight_mask / weight_mask_sum
        img_fused = (img_fused * 255).astype(np.uint8)

        dst_file_name = fused_filename_format.format(idx=idx)
        dst_file_path = os.path.join(fused_path, dst_file_name)
        mmcv.imwrite(img_fused, dst_file_path)


def dst_aware_fuse_y(img_in_fetcher, img_gt_fetcher, opts, beta):
    """
    Destination Aware Fuse
    For each image to be fused, weight for each pixel is calculated according to the similarity between
    the local region to the input LR image
    beta controls the penalty given to incorrect RefSR pixels, beta should be larger than 0
    """
    fused_root_path = opts['fused_root_path']
    fused_filename_format = opts['fused_filename_format']
    scale = opts['scale']

    fused_path = os.path.join(fused_root_path, f'dst_aware_fuse_y_{beta}')
    if not os.path.exists(fused_path):
        os.mkdir(fused_path)
    for ((idx, img_gt), (_, img_in_list)) in zip(img_gt_fetcher, img_in_fetcher):
        dst_file_name = fused_filename_format.format(idx=idx)
        dst_file_path = os.path.join(fused_path, dst_file_name)

        img_gt = mod_crop(img_gt, scale=scale)
        gt_h, gt_w, _ = img_gt.shape
        lq_h, lq_w = gt_h // scale, gt_w // scale
        img_gt_lq = resize_bgr_img_bicubic(img_gt, lq_h, lq_w)
        weight_mask_list = []
        for img_sr in img_in_list:
            img_sr_lq = resize_bgr_img_bicubic(img_sr, lq_h, lq_w)
            # calculate average weight factor
            img_sr_lq_y = metrics.bgr2ycbcr(img_sr_lq.copy(), only_y=True)
            img_gt_lq_y = metrics.bgr2ycbcr(img_gt_lq.copy(), only_y=True)

            delta = np.abs(img_sr_lq_y - img_gt_lq_y)
            weight_mask_lq = np.exp(-beta * delta)

            weight_mask = resize_bgr_img_bicubic(weight_mask_lq, gt_h, gt_w)
            weight_mask_list.append(weight_mask)

        # normalize weight_mask by each pixel, so they sum to 1
        weight_mask_sum = np.sum(weight_mask_list, axis=0)

        img_fused = np.zeros_like(img_gt)
        for (img_sr, weight_mask) in zip(img_in_list, weight_mask_list):
            img_fused += (img_sr * weight_mask / weight_mask_sum)
        img_fused = (img_fused * 255).astype(np.uint8)

        mmcv.imwrite(img_fused, dst_file_path)


def dst_aware_fuse_y_v2(img_in_fetcher, img_gt_fetcher, opts, beta):
    """
    Destination Aware Fuse
    The difference between v2 and v1 is that, v2 uses binary weight mask (determined by taking the
    max of pixel values) for each RefSR image
    """
    fused_root_path = opts['fused_root_path']
    fused_filename_format = opts['fused_filename_format']
    scale = opts['scale']
    plot_weight_mask = opts['plot_weight_mask']

    fused_path = os.path.join(fused_root_path, f'dst_aware_fuse_y_v2_{beta}')
    if not os.path.exists(fused_path):
        os.mkdir(fused_path)
    for ((idx, img_gt), (_, img_in_list)) in zip(img_gt_fetcher, img_in_fetcher):
        dst_file_name = fused_filename_format.format(idx=idx)
        dst_file_path = os.path.join(fused_path, dst_file_name)

        img_gt = mod_crop(img_gt, scale=scale)
        gt_h, gt_w, _ = img_gt.shape
        lq_h, lq_w = gt_h // scale, gt_w // scale
        img_gt_lq = resize_bgr_img_bicubic(img_gt, lq_h, lq_w)
        weight_mask_list = np.zeros((gt_h, gt_w, len(img_in_list)))
        for (idx_sr, img_sr) in enumerate(img_in_list):
            img_sr_lq = resize_bgr_img_bicubic(img_sr, lq_h, lq_w)
            # calculate average weight factor
            img_sr_lq_y = metrics.bgr2ycbcr(img_sr_lq.copy(), only_y=True)
            img_gt_lq_y = metrics.bgr2ycbcr(img_gt_lq.copy(), only_y=True)

            delta = np.abs(img_sr_lq_y - img_gt_lq_y)
            weight_mask_lq = np.exp(-beta * delta)
            # weight_mask_lq = 1 - delta

            weight_mask = resize_y_img_bicubic(weight_mask_lq, gt_h, gt_w)

            weight_mask_list[:, :, idx_sr] = weight_mask

        # for each pixel in the weight_mask_list, set the largest to be 1 and others to be 0
        weight_mask_binary_list = np.zeros_like(weight_mask_list)
        max_idx = np.argmax(weight_mask_list, axis=-1)
        for idx_sr in range(np.shape(weight_mask_binary_list)[-1]):
            weight_mask_binary_list[:, :, idx_sr] = (max_idx == idx_sr)
        weight_mask_sum = np.sum(weight_mask_binary_list, axis=-1)

        if plot_weight_mask:
            for idx_sr in range(np.shape(weight_mask_binary_list)[-1]):
                mask = weight_mask_binary_list[:, :, idx_sr]
                mmcv.imwrite(mask * 255, dst_file_path[:-4] + f"_wm_{idx_sr + 1}.png")
            mmcv.imwrite(weight_mask_sum * 255, dst_file_path[:-4] + "_wm_sum.png")

        img_fused = np.zeros_like(img_gt)
        for idx_sr in range(np.shape(weight_mask_binary_list)[-1]):
            img_sr = img_in_list[idx_sr]
            weight_mask = weight_mask_binary_list[:, :, idx_sr]
            img_sr_masked = img_sr * (weight_mask / weight_mask_sum)[..., None]
            img_fused += img_sr_masked
        img_fused = (img_fused * 255).astype(np.uint8)

        mmcv.imwrite(img_fused, dst_file_path)


def comp_local_adptive_lq_masks(img_sr_lq_y, img_gt_lq_y, beta):
    delta = np.square(img_sr_lq_y - img_gt_lq_y)
    weight_mask_lq = np.exp(-beta * delta)
    return weight_mask_lq


def comp_binary_weight_masks(weight_masks):
    num_img_sr = np.shape(weight_masks)[-1]
    b_weight_masks = np.zeros_like(weight_masks)
    max_idx = np.argmax(weight_masks, axis=-1)
    for idx_sr in range(num_img_sr):
        b_weight_masks[:, :, idx_sr] = (max_idx == idx_sr)
    return b_weight_masks


def comp_ref_quality_weights(b_weight_masks, beta2):
    num_img_sr = np.shape(b_weight_masks)[-1]
    ref_quality_weights = np.zeros(num_img_sr)
    for idx_sr in range(num_img_sr):
        ref_quality_weights[idx_sr] = np.sum(b_weight_masks[:, :, idx_sr])
    ref_quality_weights /= np.sum(ref_quality_weights)
    ref_quality_weights = np.exp(beta2 * ref_quality_weights)
    ref_quality_weights /= np.sum(ref_quality_weights)
    return ref_quality_weights


def dst_aware_fuse_y_v3(img_sr_fetcher, img_gt_fetcher, opts, beta, beta2):
    """
    Destination Aware Fuse
    The difference between v3 and v2 is that, v2 only uses Adaptive Locally Weight Mask while v3 also use
    an additional Overall Weight.
    The Adaptive Locally Weight Mask is still determined by the softmax function.
    The Overall Weight is determined by first sum up the binary weight mask of each RefSR and then perform softmax.
    """
    fused_root_path = opts['fused_root_path']
    fused_filename_format = opts['fused_filename_format']
    scale = opts['scale']
    plot_result_sr = opts['plot_result_sr']
    plot_weight_mask = opts['plot_weight_mask']

    # create folders to store images
    fused_path = os.path.join(fused_root_path, f'dst_aware_fuse_y_v3_{beta2}_{beta}')
    if not os.path.exists(fused_path):
        os.mkdir(fused_path)

    for ((idx, img_gt), (_, img_sr_list)) in zip(img_gt_fetcher, img_sr_fetcher):
        # crop ground truth image to the same dimension as RefSR results
        img_gt = mod_crop(img_gt, scale=scale)
        gt_h, gt_w, _ = img_gt.shape
        lq_h, lq_w = gt_h // scale, gt_w // scale
        img_gt_lq = resize_bgr_img_bicubic(img_gt, lq_h, lq_w)
        num_img_sr = len(img_sr_list)

        # compute weight masks
        weight_masks = np.zeros((gt_h, gt_w, num_img_sr))
        for (idx_sr, img_sr) in enumerate(img_sr_list):
            # preprocess img_gt and img_sr for weight mask calculation
            img_sr_lq = resize_bgr_img_bicubic(img_sr, lq_h, lq_w)
            img_sr_lq_y = metrics.bgr2ycbcr(img_sr_lq.copy(), only_y=True)
            img_gt_lq_y = metrics.bgr2ycbcr(img_gt_lq.copy(), only_y=True)

            # compute Locally Adaptive weight mask
            weight_mask_lq = comp_local_adptive_lq_masks(img_sr_lq_y, img_gt_lq_y, beta)
            weight_mask = resize_y_img_bicubic(weight_mask_lq, gt_h, gt_w)
            weight_masks[:, :, idx_sr] = weight_mask

        # compute binary_weight_mask, that is
        # for each pixel in the weight_masks, set the largest to be 1 and others to be 0
        b_weight_masks = comp_binary_weight_masks(weight_masks)

        # second-layer weight that reflects how good a reference image is
        ref_quality_weights = comp_ref_quality_weights(b_weight_masks, beta2)

        # combine each layer of masks and normalize
        final_weight_masks = np.copy(weight_masks)
        for idx_sr in range(num_img_sr):
            final_weight_masks[:, :, idx_sr] *= ref_quality_weights[idx_sr]
        weight_mask_sum = np.sum(final_weight_masks, axis=-1)
        final_weight_masks /= weight_mask_sum[..., None]

        # apply the masks and fuse the image
        img_fused = np.zeros_like(img_gt)
        for idx_sr in range(num_img_sr):
            # apply mask & fuse images
            img_fused += img_sr_list[idx_sr] * final_weight_masks[:, :, idx_sr][..., None]

        # (optional) save weight_masks as images
        dst_file_name = fused_filename_format.format(idx=idx)
        dst_file_path = os.path.join(fused_path, dst_file_name)
        if plot_weight_mask:
            print(f"{comp_ref_quality_weights = }")
            for idx_sr in range(num_img_sr):
                mmcv.imwrite(weight_masks[:, :, idx_sr] * 255, dst_file_path[:-4] + f"_wm_{idx_sr + 1}.png")
                mmcv.imwrite(b_weight_masks[:, :, idx_sr] * 255, dst_file_path[:-4] + f"_bwm_{idx_sr + 1}.png")
                mmcv.imwrite(final_weight_masks[:, :, idx_sr] * 255, dst_file_path[:-4] + f"_fwm_{idx_sr + 1}.png")

        # (optional) save the fused image
        if plot_result_sr:
            mmcv.imwrite(img_fused * 255, dst_file_path)


def main():
    num_input = 126
    num_ref = 5

    # create fetcher for RefSR images
    img_sr_path = \
        r'E:\CodeProjects\super_resolution\C2-Matching-Multi\results\C2_matching_gan_multi\visualization\CUFED5'
    sr_name_format = '{idx:03d}_{ref_idx}_C2_matching_gan_multi.png'
    img_sr_fetcher = RefImgFetcher(num_input=num_input, num_ref=num_ref, img_path=img_sr_path,
                                   name_format_str=sr_name_format)
    fused_filename_format = '{idx:03d}_fused.png'

    # create fetcher for ground truth images
    img_gt_path = r'E:\CodeProjects\super_resolution\dataset\CUFED_SRNTT\CUFED5'
    gt_name_format = '{idx:03d}_0.png'
    img_gt_fetcher = ImgFetcher(num_input=num_input, img_path=img_gt_path, name_format_str=gt_name_format)

    # make root folder for fused images
    fused_folder_name = 'CUFED5_fused'
    fused_root_path = os.path.join(os.path.dirname(img_sr_path), fused_folder_name)
    dir_paths = os.path.expanduser(fused_root_path)
    os.makedirs(dir_paths, mode=0o777, exist_ok=True)

    # fuse and save image
    opts = {}
    opts['fused_root_path'] = fused_root_path
    opts['fused_filename_format'] = fused_filename_format
    opts['scale'] = 4
    opts['plot_weight_mask'] = False
    opts['plot_result_sr']= False

    # get_original_input(img_gt_fetcher, opts)

    # naive_fuse(img_sr_fetcher, opts)

    # for beta in [5, 10, 20, 30, 40, 80, 100, 140, 150, 160, 200]:
    beta2 = 2
    for beta in [40, 80, 140]:
        dst_aware_fuse_y_v3(img_sr_fetcher, img_gt_fetcher, opts, beta=beta, beta2=beta2)


if __name__ == '__main__':
    main()
