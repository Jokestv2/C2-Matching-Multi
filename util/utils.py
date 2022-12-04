import os
import time
import mmcv
import numpy as np
import cv2
from PIL import Image
from mmsr.utils import FileClient


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def mkdirs(dir_path, mode=0o777):
    if dir_path == '':
        return
    if os.path.exists(dir_path):
        new_name = dir_path + '_archived_' + get_time_str()
        print(f'Warning: Path already exists. Rename it to {new_name}', flush=True)
        os.rename(dir_path, new_name)
    dir_paths = os.path.expanduser(dir_path)
    os.makedirs(dir_paths, mode=mode, exist_ok=True)


def resize_bgr_img_bicubic(img, h, w):
    """
    resize an img array read by mmcv.imfrombytes(img_bytes, channel_order='bgr').astype(np.float32) / 255.
    """
    img_pil = img * 255
    img_pil = Image.fromarray(cv2.cvtColor(img_pil.astype(np.uint8), cv2.COLOR_BGR2RGB))
    img_lq = img_pil.resize((w, h), Image.BICUBIC)
    img_lq = cv2.cvtColor(np.array(img_lq), cv2.COLOR_RGB2BGR)
    img_lq = img_lq.astype(np.float32) / 255.
    return img_lq


def resize_y_img_bicubic(img, h, w):
    """
    resize the y channel of an img array read by
    mmcv.imfrombytes(img_bytes, channel_order='bgr').astype(np.float32) / 255.
    """
    img_pil = img * 255
    img_pil = Image.fromarray(img_pil.astype(np.uint8))
    img_lq = img_pil.resize((w, h), Image.BICUBIC)
    img_lq = np.array(img_lq, dtype=np.float32) / 255.
    return img_lq


class RefImgFetcher:
    """
    Iterable object fetching a group of images, using different Ref images, from the RefSR outputs
    """
    def __init__(self, num_input, num_ref, img_path, name_format_str):
        self.num_input = num_input
        self.num_ref = num_ref
        self.img_path = img_path
        self.name_format_str = name_format_str

    def __iter__(self):
        self.cur_idx = 0
        return self

    def __next__(self):
        if self.cur_idx < self.num_input:
            fcli = FileClient()
            img_in_list = []
            for ref_idx in range(1, 1 + self.num_ref):
                src_file_name = self.name_format_str.format(idx=self.cur_idx, ref_idx=ref_idx)
                src_file_path = os.path.join(self.img_path, src_file_name)
                img_bytes = fcli.get(src_file_path, 'in')
                img_in = mmcv.imfrombytes(img_bytes, channel_order='bgr').astype(np.float32) / 255.
                img_in_list.append(img_in)
            self.cur_idx += 1
            return self.cur_idx - 1, img_in_list
        else:
            raise StopIteration


class ImgFetcher:
    """
    Iterable object fetching images
    """
    def __init__(self, num_input, img_path, name_format_str):
        self.num_input = num_input
        self.img_path = img_path
        self.name_format_str = name_format_str

    def __iter__(self):
        self.cur_idx = 0
        return self

    def __next__(self):
        if self.cur_idx < self.num_input:
            fcli = FileClient()
            src_file_name = self.name_format_str.format(idx=self.cur_idx)
            src_file_path = os.path.join(self.img_path, src_file_name)
            img_bytes = fcli.get(src_file_path, 'in')
            img_in = mmcv.imfrombytes(img_bytes, channel_order='bgr').astype(np.float32) / 255.
            self.cur_idx += 1
            return self.cur_idx - 1, img_in
        else:
            raise StopIteration
