import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

default_root_path = r'E:\CodeProjects\super_resolution\C2-Matching-Multi\results\C2_matching_gan_multi\visualization'


def visual_beta_metrics(saved_folder, time_str='20221205_221331'):
    if len(saved_folder) < 1:
        saved_folder = 'CUFED5_fused_experiments_quantitative'
    root_path = os.path.join(default_root_path, saved_folder)

    avg_psnr_mat = np.load(os.path.join(root_path, f'{time_str}_avg_psnr_mat.npy'))
    avg_psnr_y_mat = np.load(os.path.join(root_path, f'{time_str}_avg_psnr_y_mat.npy'))
    avg_ssim_y_mat = np.load(os.path.join(root_path, f'{time_str}_avg_ssim_y_mat.npy'))
    beta2_list = np.load(os.path.join(root_path, f'{time_str}_beta2_list.npy'))
    beta_list = np.load(os.path.join(root_path, f'{time_str}_beta_list.npy'))

    print(f"Shape of avg_psnr_y_mat is: {np.shape(avg_psnr_y_mat)}")
    print(f"Shape of avg_ssim_y_mat is: {np.shape(avg_ssim_y_mat)}")
    print(f"Shape of beta2_list is: {np.shape(beta2_list)}")
    print(f"Shape of beta_list is: {np.shape(beta_list)}")

    plt.plot(beta_list, avg_psnr_y_mat.squeeze(), 'o-b', label=f'PSNR_Y - C2-Matching')
    beta_list_x_axis = beta_list[::2]
    plt.xticks(beta_list_x_axis, labels=[f'{x:.0f}' for x in beta_list_x_axis], rotation=45)
    plt.xlabel(r'$\beta$')
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(root_path, f'{time_str}_avg_psnr_y_line.png'), dpi=300, bbox_inches='tight')
    plt.show()


def visual_beta_and_beta2_metrics():
    time_str = '20221204_193424'
    root_path = \
        r'E:\CodeProjects\super_resolution\C2-Matching-Multi\results\C2_matching_gan_multi\visualization\CUFED5_fused_experiments_quantitative'

    avg_psnr_mat = np.load(os.path.join(root_path, f'{time_str}_avg_psnr_mat.npy'))
    avg_psnr_y_mat = np.load(os.path.join(root_path, f'{time_str}_avg_psnr_y_mat.npy'))
    avg_ssim_y_mat = np.load(os.path.join(root_path, f'{time_str}_avg_ssim_y_mat.npy'))
    beta2_list = np.load(os.path.join(root_path, f'{time_str}_beta2_list.npy'))
    beta_list = np.load(os.path.join(root_path, f'{time_str}_beta_list.npy'))

    plt.imshow(avg_psnr_y_mat, interpolation='none')
    plt.colorbar()
    plt.xlabel('beta')
    plt.xticks(np.arange(len(beta_list)), labels=[f'{x:.1f}' for x in beta_list])
    plt.ylabel('beta2')
    plt.yticks(np.arange(len(beta2_list)), labels=[f'{y:.1f}' for y in beta2_list])
    plt.title('avg_psnr_y_mat')
    plt.tight_layout()
    plt.savefig(os.path.join(root_path, f'{time_str}_avg_psnr_y_mat.png'), dpi=300, bbox_inches='tight')
    plt.show()


def main():
    visual_beta_metrics('')


if __name__ == '__main__':
    main()