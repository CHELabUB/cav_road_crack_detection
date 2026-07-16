
# YOLOv8 Checkered Box Detection

[@chaozheUB comment] This work was originally done by @deepa355. It is heavily over fitted but good attempt.

This folder contains code to train a YOLOv8 model for detecting checkered boxes in images.


## Files

### `checkered_box_model.py`

- Loads the YOLOv8n pre-trained model.

- Trains the model on the dataset after splitting it into train and test specified in `data.yaml`.

- Training parameters:

- Epochs: 15 (since it was for small size dataset)

- Image size: 640x640

- Batch size: 4

- Device: CPU (due to Mac M1 restrictions on GPU support)

- Saves training results under:

`checkered_board/runs/detect/run1`


### `data.yaml`
Configuration file for the dataset:

-  `path`: Base path to the dataset directory (`./split_data`)

-  `train`: Relative path to training images folder (`images/train`)

-  `val`: Relative path to validation images folder (`images/test`) [ I've used the test set for validation ]

-  `names`: Class names (a single class called `"checkered_box"`)

-  `nc`: Number of classes (1)


### Model Weights

Model weights have been uploaded to UB Box:

(https://buffalo.box.com/s/1dfumi0m2gsxo6dvci1littqipg2u5dr)


### Dataset

Dataset has been uploaded to UB Box:

(https://buffalo.box.com/s/fqf0oivxk88y8mw9ytq6e3bmf965wgs8)


### Predictions on Test set

Predicted results on the test set are also available:

(https://buffalo.box.com/s/sxgqhsb0f6kk5kxhqil3oyreof5u0piq)


## Dataset Structure

split_data/

- images/

-  - train/

-  -  - img1.jpg

-  -  - ...

-  - test/

-  -  - img1.jpg

-  -  - ...

- labels/

-  - train/

-  -  - img1.txt

-  -  - ...

-  - test/

-  -  - img1.txt

-  -  - ...


## How to run

1. Make sure you have the required dependencies installed:

```bash

pip  install  ultralytics  torch

```

2. Organize your dataset as shown above or you can access the dataset uploaded to UB Box.


3. Run the training script.

```bash
python checkered_box_model.py
```

4. Training results and weights will be saved in the checkered_board/runs/detect/run1 directory.

5. To obtain the predictions on a test set, run the following command.

```bash

model = YOLO("update path to the model weights (best.pt)")
results = model.predict(source="update path to the test set",  save=True)

```

6. You will be able to access the results in runs/detect/predict directory
