# Inference models
There are two models that we have tried, [DeepCrack](https://github.com/qinnzou/DeepCrack) and [LECSFormer](https://github.com/ZhaoNan1/LECSFormer.git), and the folders are adapted from their original repo. We appreciate their foundational work on the models. 

### Suggested requirements
We suggested the followings for the hardware to train the models:
- **GPU Requirement**
   - A GPU is required for this project. Ensure that your machine is equipped with an NVIDIA GPU that supports CUDA.
- **CUDA Toolkit**
   - This project requires CUDA version **11.8** or higher.
- **Driver Requirements**
   - Ensure that your NVIDIA drivers are up to date. The required version should be compatible with CUDA 11.8.

As a reference, the original DeepCrack code was run on the Intel Core Xeon E5-2630@2.3GHz, 64GB RAM and two GeForce GTX TITAN-X GPUs.

### Environment setup
For the libraries required to run the models, please refer to [requirements](/Inference_models/LECSFormer/requirements.txt) in [LECSFormer](/Inference_models/LECSFormer/)

To install the required libraries, please create a virtual environment, activate and run the following:
 ```
pip install -r requirements.txt
```

### Model training and testing

All commands below should be run from the model's directory.

#### DeepCrack

**Train** — configure `train_data_path` and `test_data_path` in `config.py`, then:
```bash
cd Inference_models/DeepCrack
python train.py
```

**Test**
```bash
cd Inference_models/DeepCrack
python test.py \
  --test_data_path data/CRKWH100/test.txt \
  --pretrained_model weights/DeepCrack_CT260_FT1.pth \
  --save_path output/deepcrack_results/
```

#### LECSFormer

**Train**
```bash
cd Inference_models/LECSFormer
python train.py \
  --root_path path/to/dataset/ \
  --output_dirs output/LECSFormer/
```

**Test**
```bash
cd Inference_models/LECSFormer
python test.py \
  --root_path path/to/dataset/ \
  --checkpoints output/CrackTree260/best_model.pth \
  --output_dir output/CrackTree260/test_results/
```
## weights and dataset
The weights that we have trained the used can be found in [weights](https://buffalo.box.com/s/skwflwdikwu8edad7pue1bgh92cbm9cp)

The Training dataset that we used for LECSFormer can be found in [Training_datasets](https://buffalo.box.com/s/i8pel3yw2krbl6i7pm10o44t0pdqrda5)

The Testing dataset that we used for LECSFormer can be found in [Testing_datasets](https://buffalo.box.com/s/h0yvuij543rrl36v0nc6werditajxlgn)