"""
Convert results generated by RefSR to a format suitable for fusion model
"""
import os
import shutil
from util.utils import mkdirs


def main():
    result_path = \
        r'E:\CodeProjects\super_resolution\C2-Matching-Multi\results\C2_matching_gan_multi\visualization\CUFED5'
    gt_path = r'E:\CodeProjects\super_resolution\dataset\CUFED_SRNTT\CUFED5'
    num_input = 126
    num_ref = 5
    fusion_subfolders = ['A_Y', 'B_Y', 'C_Y', 'D_Y', 'E_Y']
    fusion_folder_name = 'CUFED5_fusion'

    # make root folder for fusion dataset
    fusion_path = os.path.join(os.path.dirname(result_path), fusion_folder_name)
    mkdirs(fusion_path)

    # Copy ground-truth images
    fusion_gt_path = os.path.join(fusion_path, 'GT')
    os.mkdir(fusion_gt_path)

    for input_idx in range(num_input):
        src_file_name = f'{input_idx:03d}_0.png'
        src_file_path = os.path.join(gt_path, src_file_name)
        dst_file_name = f'{input_idx}.png'
        dst_file_path = os.path.join(fusion_gt_path, dst_file_name)
        shutil.copyfile(src_file_path, dst_file_path)

    # Copy RefSR results
    for (fusion_subfolder, ref_idx) in zip(fusion_subfolders, range(1, 1+num_ref)):
        fusion_subpath = os.path.join(fusion_path, fusion_subfolder)
        os.mkdir(fusion_subpath)
        for input_idx in range(num_input):
            src_file_name = f'{input_idx:03d}_{ref_idx}_C2_matching_gan_multi.png'
            src_file_path = os.path.join(result_path, src_file_name)
            dst_file_name = f'{input_idx}.png'
            dst_file_path = os.path.join(fusion_subpath, dst_file_name)
            shutil.copyfile(src_file_path, dst_file_path)
    print("Done!")


if __name__ == '__main__':
    main()