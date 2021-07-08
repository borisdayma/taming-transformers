## Instructions for mini-DALLE project

### Set up environment

- [set up TPU VM](https://cloud.google.com/tpu/docs/pytorch-quickstart-tpu-vm)
- `git clone -b feat-wandb https://github.com/borisdayma/taming-transformers.git`
- `conda env create -f environment.yaml`
- `conda activate taming`
- `pip install -e .`
- `wandb login`

### Set up data & model

- have a folder that contains:

  - CC3M
  - CC12M
  - some of [YFCC100M OpenAI subset](https://huggingface.co/datasets/flax-community/YFCC100M_OpenAI_subset) - you can download a few archives manually and extract (or I can scp)
  - make sure we still have at least 100-200 GB of space left

- prepare train/validation set

  - run `find DATA_PATH -name "*.jpg" > train.txt` (make sure we have data from the 3 datasets)
  - move some lines to `test.txt` (better from the OpenAI dataset to make sure there is no overlap - maybe 1000 images for a reliable validation metric)

- download [pretrained vqgan_imagenet_f16_16384](https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/)

- update `configs/dalle_vqgan.yaml`

  - ckpt_path needs to point to pretrained checkpoint
  - training_images_list_file points to `train.txt`
  - test_images_list_file points to `test.txt`
  - batch_size will need to be increased manually to maximize use of TPU
  - do not change the other values as it could cause conflict with pre-trained model (except maybe num_workers if data loading seems to be the bottleneck)

- run the script

  - on TPU: `python main.py --base configs/dalle_vqgan.yaml -t True --tpu_cores 8`
  - on GPU: `python main.py --base configs/dalle_vqgan.yaml -t True --gpus 0,` or `--gpus 0,1,`, etc (don't forget the last `,`)

  Note: depending on the time per epoch, we can adjust (it uses a default of 1000 epochs with auto logging of top 3 models based on `val/rec_loss`)
