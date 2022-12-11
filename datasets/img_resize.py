from PIL import Image
import glob
import os

src_folder_path = r'E:\CodeProjects\super_resolution\dataset\Landmark\analyzed_dataset'
dst_folder_path = r'E:\CodeProjects\super_resolution\dataset\Landmark\testset'

img_paths = glob.glob(src_folder_path + '/*.jpg')
for src_img_path in img_paths:
    file_name = os.path.basename(src_img_path)
    img = Image.open(src_img_path)

    # w, h = img.size[0], img.size[1]
    img.thumbnail((500, 500), Image.Resampling.LANCZOS)

    new_file_name = file_name[:-4] + '.png'
    img.save(os.path.join(dst_folder_path, new_file_name))
