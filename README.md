# Group Project

![Python 3.7](https://img.shields.io/badge/python-3.9.7-green.svg?style=plastic)
![pytorch](https://img.shields.io/badge/pytorch-1.9.0-green.svg?style=plastic)


## Overview
This project allows you to:
- Run the state-of-art single-reference-based super-resolution model C2-Matching (pretrained weight included), the scripts for it to run on CUFED5 with each individual reference image are included.
- Run scripts to fuse the outputs into one final result. These scripts implement the methods mentioned in our report.
- Run scripts to do qualitative and quantitative experiments on the result.



## Dependencies and Installation

- Python == 3.9.7
- PyTorch == 1.9.0
- CUDA >= 10.2
- GCC >= 7.5.0


1. Clone Repo

   ```bash
   git clone git@github.com:yumingj/C2-Matching.git
   ```

1. Create Conda Environment

   ```bash
   conda create --name c2_matching python=3.9.7
   conda activate c2_matching
   ```

1. Install Dependencies

   ```bash
   cd C2-Matching
   pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
   pip install mmcv==0.4.4
   pip install -r requirements.txt
   ```

1. Install MMSR and DCNv2

    ```bash
    python setup.py develop --user
    cd mmsr/models/archs/DCNv2
    python setup.py build develop --user
    ```


## Dataset Preparation

- Train Set: [CUFED Dataset](https://drive.google.com/drive/folders/1hGHy36XcmSZ1LtARWmGL5OK1IUdWJi3I)
- Test Set: [WR-SR Dataset](https://drive.google.com/drive/folders/16UKRu-7jgCYcndOlGYBmo5Pp0_Mq71hP?usp=sharing), [CUFED5 Dataset](https://drive.google.com/file/d/1Fa1mopExA9YGG1RxrCZZn7QFTYXLx6ph/view)

Please refer to [Datasets.md](datasets/DATASETS.md) for pre-processing and more details.

## Get Started

### Pretrained Models
Downloading the pretrained models from this [link](https://drive.google.com/drive/folders/1dTkXMzeBrHelVQUEx5zib5MdmvqDaSd9?usp=sharing) and put them under `experiments/pretrained_models folder`.

### Test

We provide quick test code with the pretrained model.

1. Modify the paths to dataset and pretrained model in the following yaml files for configuration.

    ```bash
    ./options/test/test_C2_matching_multi.yml
    ```

1. Run test code for models trained using **GAN loss**.

    ```bash
    python mmsr/test.py -opt "options/test/test_C2_matching.yml"
    ```

   Check out the results in `./results`.



### Train (Note for TA: this part can be skipped)

All logging files in the training process, *e.g.*, log message, checkpoints, and snapshots, will be saved to `./experiments` and `./tb_logger` directory.

1. Modify the paths to dataset in the following yaml files for configuration.
   ```bash
   ./options/train/stage1_teacher_contras_network.yml
   ./options/train/stage2_student_contras_network.yml
   ./options/train/stage3_restoration_gan.yml
   ```

1. Stage 1: Train teacher contrastive network.
   ```bash
   python mmsr/train.py -opt "options/train/stage1_teacher_contras_network.yml"
   ```

1. Stage 2: Train student contrastive network.
   ```bash
   # add the path to *pretrain_model_teacher* in the following yaml
   # the path to *pretrain_model_teacher* is the model obtained in stage1
   ./options/train/stage2_student_contras_network.yml
   python mmsr/train.py -opt "options/train/stage2_student_contras_network.yml"
   ```

1. Stage 3: Train restoration network.
   ```bash
   # add the path to *pretrain_model_feature_extractor* in the following yaml
   # the path to *pretrain_model_feature_extractor* is the model obtained in stage2
   ./options/train/stage3_restoration_gan.yml
   python mmsr/train.py -opt "options/train/stage3_restoration_gan.yml"

   # if you wish to train the restoration network with only mse loss
   # prepare the dataset path and pretrained model path in the following yaml
   ./options/train/stage3_restoration_mse.yml
   python mmsr/train.py -opt "options/train/stage3_restoration_mse.yml"
   ```

## The Fusion Module
Check out the scripts in `./fusion`. The fusion method is implemented in the `non_nn_fuse.py`, and the script to run the experiments are written in `experiments_C2_mathcing.py` and `experiments_AMSA.py`.
For a visualization of the quantitative experimental results, check ou the code in `./util/visualize.py`

## License and Acknowledgement
This project is mainly modified from [C2-Matching](https://github.com/yumingj/C2-Matching), please cite their paper if you find the C2-Matching Module of this project useful.

The original project is open sourced under MIT license. The code framework is mainly modified from [BasicSR](https://github.com/xinntao/BasicSR) and [MMSR](https://github.com/open-mmlab/mmediting) (Now reorganized as MMEditing). Please refer to the original repo for more usage and documents.


## Contact

If you have any question, please feel free to contact us via `kke.zhao@mail.utoronto.ca`.
