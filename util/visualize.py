import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

c2_root_path = r'E:\CodeProjects\super_resolution\C2-Matching-Multi\results\C2_matching_gan_multi\visualization'
amsa_root_path = r'E:\CodeProjects\super_resolution\C2-Matching-Multi\results\amsa_gan_multi\visualization'

root_paths = {
    'C2-Matching': c2_root_path,
    'AMSA': amsa_root_path
}

psnr_y_line_fmt = {
    'C2-Matching': '-b',
    'AMSA': '-b'
}
ssim_line_fmt = {
    'C2-Matching': '--g',
    'AMSA': '--g'
}

# model_used = ['C2-Matching', 'AMSA']
model_used = ['C2-Matching']
# model_used = ['AMSA']


def main():
    # saved_folder = 'fused_experiments_qualitative'
    # beta
    # time_str = {
    #     'C2-Matching': '20221206_012156',
    #     'AMSA': '20221206_222104'
    # }
    # x_label = r'$\beta$'
    # visual_beta_metrics('', x_label, time_str)

    # beta2
    # time_str = {
    #     'C2-Matching': '20221206_014131',
    #     'AMSA': '20221206_224530'
    # }
    # x_label = r'$\beta_g$'
    # visual_beta2_metrics('', x_label, time_str)

    saved_folder = 'fused_experiments_quantitative'
    time_str = '20221210_221023'
    root_path = c2_root_path
    visual_beta_and_beta2_metrics(root_path, saved_folder, time_str)

    # num_sr
    # time_str = {
    #     'C2-Matching': '20221206_125204',
    #     'AMSA': '20221206_224530'
    # }
    # time_str_list = {
    #     'C2-Matching': ['20221206_141955', '20221206_142005', '20221206_142016', '20221206_142029', '20221206_142044'],
    #     'AMSA': ['20221206_225632', '20221206_225641', '20221206_225652', '20221206_225704', '20221206_225718'],
    # }
    # visual_num_sr_metrics('', time_str_list)

def visual_beta_metrics(saved_folder, x_label, time_str_dict='20221206_012156'):
    if len(saved_folder) < 1:
        saved_folder = 'fused_experiments_quantitative'

    root_path = {}
    for ds in model_used:
        root_path[ds] = os.path.join(root_paths[ds], saved_folder)

    avg_psnr_mat = {}
    avg_psnr_y_mat = {}
    avg_ssim_y_mat = {}
    for ds in model_used:
        time_str = time_str_dict[ds]
        avg_psnr_mat[ds] = np.load(os.path.join(root_path[ds], f'{time_str}_avg_psnr_mat.npy'))
        avg_psnr_y_mat[ds] = np.load(os.path.join(root_path[ds], f'{time_str}_avg_psnr_y_mat.npy'))
        avg_ssim_y_mat[ds] = np.load(os.path.join(root_path[ds], f'{time_str}_avg_ssim_y_mat.npy'))
    beta2_list = np.load(os.path.join(root_path[model_used[0]], f'{time_str_dict[model_used[0]]}_beta2_list.npy'))
    beta_list = np.load(os.path.join(root_path[model_used[0]], f'{time_str_dict[model_used[0]]}_beta_list.npy'))

    # print(f"Shape of avg_psnr_y_mat is: {np.shape(avg_psnr_y_mat)}")
    # print(f"Shape of avg_ssim_y_mat is: {np.shape(avg_ssim_y_mat)}")
    # print(f"Shape of beta2_list is: {np.shape(beta2_list)}")
    # print(f"Shape of beta_list is: {np.shape(beta_list)}")

    saved_path = os.path.join(root_path[model_used[0]], f'{time_str_dict[model_used[0]]}_beta_metrics.png')
    plot_psnr_and_ssim(beta_list, x_label, avg_psnr_y_mat, avg_ssim_y_mat, saved_path)


def visual_beta2_metrics(saved_folder, x_label, time_str_dict='20221206_004426'):
    if len(saved_folder) < 1:
        saved_folder = 'fused_experiments_quantitative'

    root_path = {}
    for ds in model_used:
        root_path[ds] = os.path.join(root_paths[ds], saved_folder)

    avg_psnr_mat = {}
    avg_psnr_y_mat = {}
    avg_ssim_y_mat = {}
    for ds in model_used:
        time_str = time_str_dict[ds]
        avg_psnr_mat[ds] = np.load(os.path.join(root_path[ds], f'{time_str}_avg_psnr_mat.npy'))
        avg_psnr_y_mat[ds] = np.load(os.path.join(root_path[ds], f'{time_str}_avg_psnr_y_mat.npy'))
        avg_ssim_y_mat[ds] = np.load(os.path.join(root_path[ds], f'{time_str}_avg_ssim_y_mat.npy'))
        avg_psnr_y_mat[ds] = avg_psnr_y_mat[ds].transpose()
        avg_ssim_y_mat[ds] = avg_ssim_y_mat[ds].transpose()
    beta2_list = np.load(os.path.join(root_path[model_used[0]], f'{time_str_dict[model_used[0]]}_beta2_list.npy'))
    beta_list = np.load(os.path.join(root_path[model_used[0]], f'{time_str_dict[model_used[0]]}_beta_list.npy'))



    saved_path = os.path.join(root_path[model_used[0]], f'{time_str_dict[model_used[0]]}_beta2_metrics.png')
    plot_psnr_and_ssim(beta2_list, x_label, avg_psnr_y_mat, avg_ssim_y_mat, saved_path)


def plot_psnr_and_ssim(x_values, x_label, psnr_values, ssim_values, img_save_path):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    for ds in model_used:
        ln1 = ax1.plot(x_values, psnr_values[ds].squeeze(), psnr_y_line_fmt[ds],
                       label=f'PSNR_Y - {ds}')
        ln2 = ax2.plot(x_values, ssim_values[ds].squeeze(), ssim_line_fmt[ds],
                       label=f'SSIM - {ds}')

    x_values_for_axis = x_values[::2]
    ax1.set_xticks(x_values_for_axis, labels=[f'{x:.1f}' for x in x_values_for_axis], rotation=45)
    ax1.set_xlabel(x_label)

    ax1.set_ylabel('PSNR_Y')
    ax2.set_ylabel('SSIM')

    ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))

    lns = ln1 + ln2
    ln_labels = [l.get_label() for l in lns]
    ax1.legend(lns, ln_labels, loc='lower right')

    plt.tight_layout()
    plt.savefig(img_save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visual_beta_and_beta2_metrics(root_path, saved_folder, time_str='20221206_004426'):
    if len(saved_folder) < 1:
        saved_folder = 'fused_experiments_quantitative'
    root_path = os.path.join(root_path, saved_folder)

    avg_psnr_mat = np.load(os.path.join(root_path, f'{time_str}_avg_psnr_mat.npy'))
    avg_psnr_y_mat = np.load(os.path.join(root_path, f'{time_str}_avg_psnr_y_mat.npy'))
    avg_ssim_y_mat = np.load(os.path.join(root_path, f'{time_str}_avg_ssim_y_mat.npy'))
    beta2_list = np.load(os.path.join(root_path, f'{time_str}_beta2_list.npy'))
    beta_list = np.load(os.path.join(root_path, f'{time_str}_beta_list.npy'))

    plt.imshow(avg_psnr_y_mat, interpolation='none')
    plt.colorbar()
    plt.xlabel(r'$\beta$')
    beta_list_for_axis = beta_list[::1]
    plt.xticks(np.arange(0, len(beta_list), 1), labels=[f'{x:.0f}' for x in beta_list_for_axis], rotation=45)
    plt.ylabel(r'$\beta_g$')
    beta2_list_for_axis = beta2_list[::2]
    plt.yticks(np.arange(0, len(beta2_list), 2), labels=[f'{y:.1f}' for y in beta2_list_for_axis])
    plt.tight_layout()
    plt.savefig(os.path.join(root_path, f'{time_str}_avg_psnr_y_mat_beta_beta2.png'), dpi=300, bbox_inches='tight')
    plt.show()

    plt.imshow(avg_ssim_y_mat, interpolation='none')
    plt.colorbar()
    plt.xlabel(r'$\beta$')
    beta_list_for_axis = beta_list[::1]
    plt.xticks(np.arange(0, len(beta_list), 1), labels=[f'{x:.0f}' for x in beta_list_for_axis], rotation=45)
    plt.ylabel(r'$\beta_g$')
    beta2_list_for_axis = beta2_list[::2]
    plt.yticks(np.arange(0, len(beta2_list), 2), labels=[f'{y:.1f}' for y in beta2_list_for_axis])
    plt.tight_layout()
    plt.savefig(os.path.join(root_path, f'{time_str}_avg_ssim_y_mat_beta_beta2.png'), dpi=300, bbox_inches='tight')
    plt.show()


def visual_num_sr_metrics(saved_folder, time_str_list):
    if len(saved_folder) < 1:
        saved_folder = 'fused_experiments_quantitative_num_sr'
    root_path = {}
    for ds in model_used:
        root_path[ds] = os.path.join(root_paths[ds], saved_folder)

    RefNum = [1,2,3,4,5]
    psnr_y_list = {}
    ssim_y_list = {}
    for ds in model_used:
        psnr_y_list[ds] = []
        ssim_y_list[ds] = []
        for t_str in time_str_list[ds]:
            avg_psnr_y_mat = np.load(os.path.join(root_path[ds], f'{t_str}_avg_psnr_y_mat.npy'))
            avg_ssim_y_mat = np.load(os.path.join(root_path[ds], f'{t_str}_avg_ssim_y_mat.npy'))
            psnr_y_list[ds].append(avg_psnr_y_mat.item())
            ssim_y_list[ds].append(avg_ssim_y_mat.item())

    x_values = RefNum
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    for ds in model_used:
        hline_y = [psnr_y_list[ds][0] for x in x_values]
        ln1 = ax1.plot(x_values, psnr_y_list[ds], 'o' + psnr_y_line_fmt[ds],
                       label=f'PSNR_Y - {ds}')
        ln2 = ax2.plot(x_values, ssim_y_list[ds], 'o' + ssim_line_fmt[ds],
                       label=f'SSIM - {ds}')
        ln3 = ax1.plot(x_values, hline_y, 'r--',
                       label=f"PSNR_Y/SSIM - Single Ref {ds}")

    ax1.set_xticks(x_values, labels=[f'{x:.0f}' for x in x_values])
    ax1.set_xlabel('Number of RefSR Images Fused')

    ax1.set_ylabel('PSNR_Y')
    ax2.set_ylabel('SSIM')

    ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))

    lns = ln3 + ln1 + ln2
    ln_labels = [l.get_label() for l in lns]
    ax1.legend(lns, ln_labels, loc='lower right')

    plt.tight_layout()

    saved_path = os.path.join(root_path[model_used[0]], f'{time_str_list[ds][0]}_{time_str_list[ds][-1]}_num_sr.png')
    plt.savefig(saved_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()