# HitDetector
A machine learning pipeline to detect holes/marks/hits on images. Designed to distinguish marks from blank areas using neural network classification.

## Quick start
- Make sure you have **Python 3.9+** and **Poetry** installed
  `curl -sSL https://install.python-poetry.org | python3 -`
- Clone the project:  
  `git clone https://github.com/andrewromanenco/hit-detector`
- Navigate to the project directory:  
  `cd hit-detector` & `poetry install`
- Run inference:  
  `poetry run inference model.pt <path-to-image> <path-to-save-result-image>`

## ✨ Summary
- Detects only holes/isolated marks (no background or environmental noise)
- Contains script for annotation, samples extraction, training, and inference 
- [Sample input image](https://github.com/andrewromanenco/hit-detector/blob/main/images/sample1.png)
- [Sample processed image](https://github.com/andrewromanenco/hit-detector/blob/main/images/sample1-out.png)
- The included model looks at areas of 24x24 pixels

## 📌 Future ideas
- Use active learning to expand the training dataset based on model confidence scores
- Prioritize reducing false positives, including both inked regions and paper overlays
- Train the main model (or a supporting model) to identify and localize the target region within an image

## 🧠 How It Works
1. Load an image into the annotation tool.
2. Annotate holes/hits and blanks by clicking on the image. Each click is recorded into a CSV file.
3. Use the patch extraction tool to crop sample regions from images based on the annotation data.
4. Train a neural network on the extracted image patches.
5. Use a sliding window algorithm for inference to find hits in new images.

## 🧬 Algorithm Details
-   Binary classification of image patches
-   Sliding window approach for inference
-   Metrics tracked during training:
    -   Accuracy (used for early stop)
    -   Precision
    -   Recall
    -   F1 Score
-   Positional class balancing

## 🧰 Environment
- Runs in a Docker container
- Requires [XQuartz](https://www.xquartz.org/) (for UI on macOS)
- Uses (inside the docker container):
  - `cv2` for image manipulation
  - `PyTorch` for training and inference

## 🛠️ Installation & Setup

### 1. Install XQuartz (for GUI apps inside Docker)
> 💡 Used for annotation only. Not needed for training or inference.

```bash
brew install --cask xquartz
```

Then, inside an XQuartz terminal (rerun xhost on every XQuartz restart):
```bash
export DISPLAY=:0
xhost + 127.0.0.1
```

### 2. Clone the Project
```bash
git clone https://github.com/andrewromanenco/hit-detector.git
cd hit-detector
```

### 3. Run the Docker Container
```bash
docker run -e DISPLAY=host.docker.internal:0 \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           -it --rm \
           -v $PWD:/appx:rw \
           romanenco/python-tool-chain /bin/bash
```
> 💡 This uses a temporary container. You may want to build your own image later.

### 4. Install Dependencies
> 💡 libgl1 is needed only for annotation tool.
```bash
apt update && apt install -y libgl1
cd /appx
poetry install
```

## 🧭 Step-by-Step Guide

### 1. Annotate Source Images

```bash
poetry run marker images/8-full-top.jpg hits-8-full-top.csv
poetry run marker images/8-full-top.jpg blanks-8-full-top.csv
```
> ✅ This saves click coordinates to CSV and a copy of the image with  `_marked`  suffix.

### 2. Extract Patches
Extract 24×24 patches around each coordinate for hits and blanks:
```bash
poetry run extract-patches images/8-full-top.jpg hits-8-full-top.csv 24 patches/hits
poetry run extract-patches images/8-full-top.jpg blanks-8-full-top.csv 24 patches/blanks
```

> 📁 You'll now have images in:
> 
> -   `patches/hits/`
> -   `patches/blanks/`
>     

### 3. Train the Model
Train using extracted patches:
```bash
poetry run train --hits-dir patches/hits --blanks-dir patches/blanks --model-path model.pt
```

### 4. Run Inference
Run the trained model on a new image:
```bash
mkdir result
poetry run inference model.pt images/8-mod-bottom.jpg result/8-mod-bottom.png
poetry run inference model.pt images/8-full-top.jpg result/8-full-top.png
poetry run inference model.pt images/7.5-mod-bottom.jpg result/7.5-mod-bottom.png
poetry run inference model.pt images/7.5-full-top.jpg result/7.5-full-top.png
```

## 📊 Sample Training Log
```
Class  balance:  0  ->  2736,  1  ->  1608,  pos_weight  =  0.59
Epoch  1/1000  -  Loss:  0.3956  -  Accuracy:  0.7065  -  Precision:  0.8535  -  Recall:  0.2500  -  F1-score:  0.3867
Epoch  2/1000  -  Loss:  0.1840  -  Accuracy:  0.8980  -  Precision:  0.9088  -  Recall:  0.8053  -  F1-score:  0.8539
Epoch  3/1000  -  Loss:  0.1427  -  Accuracy:  0.9256  -  Precision:  0.9509  -  Recall:  0.8427  -  F1-score:  0.8935
Epoch  4/1000  -  Loss:  0.1171  -  Accuracy:  0.9388  -  Precision:  0.9583  -  Recall:  0.8725  -  F1-score:  0.9134
Epoch  5/1000  -  Loss:  0.1050  -  Accuracy:  0.9445  -  Precision:  0.9640  -  Recall:  0.8831  -  F1-score:  0.9218
Epoch  6/1000  -  Loss:  0.0966  -  Accuracy:  0.9503  -  Precision:  0.9671  -  Recall:  0.8961  -  F1-score:  0.9303
Epoch  7/1000  -  Loss:  0.0843  -  Accuracy:  0.9558  -  Precision:  0.9733  -  Recall:  0.9055  -  F1-score:  0.9381
Epoch  8/1000  -  Loss:  0.0817  -  Accuracy:  0.9572  -  Precision:  0.9785  -  Recall:  0.9042  -  F1-score:  0.9399
Epoch  9/1000  -  Loss:  0.0655  -  Accuracy:  0.9664  -  Precision:  0.9784  -  Recall:  0.9297  -  F1-score:  0.9534
Epoch  10/1000  -  Loss:  0.0699  -  Accuracy:  0.9634  -  Precision:  0.9776  -  Recall:  0.9223  -  F1-score:  0.9491
Epoch  11/1000  -  Loss:  0.0589  -  Accuracy:  0.9728  -  Precision:  0.9850  -  Recall:  0.9409  -  F1-score:  0.9625
Epoch  12/1000  -  Loss:  0.0562  -  Accuracy:  0.9740  -  Precision:  0.9857  -  Recall:  0.9434  -  F1-score:  0.9641
Epoch  13/1000  -  Loss:  0.0502  -  Accuracy:  0.9756  -  Precision:  0.9877  -  Recall:  0.9459  -  F1-score:  0.9663
Epoch  14/1000  -  Loss:  0.0454  -  Accuracy:  0.9802  -  Precision:  0.9910  -  Recall:  0.9552  -  F1-score:  0.9728
Epoch  15/1000  -  Loss:  0.0364  -  Accuracy:  0.9846  -  Precision:  0.9936  -  Recall:  0.9646  -  F1-score:  0.9789
Epoch  16/1000  -  Loss:  0.0338  -  Accuracy:  0.9853  -  Precision:  0.9949  -  Recall:  0.9652  -  F1-score:  0.9798
Epoch  17/1000  -  Loss:  0.0314  -  Accuracy:  0.9871  -  Precision:  0.9936  -  Recall:  0.9714  -  F1-score:  0.9824
Epoch  18/1000  -  Loss:  0.0244  -  Accuracy:  0.9910  -  Precision:  0.9981  -  Recall:  0.9776  -  F1-score:  0.9877
Epoch  19/1000  -  Loss:  0.0245  -  Accuracy:  0.9922  -  Precision:  0.9987  -  Recall:  0.9801  -  F1-score:  0.9893
Epoch  20/1000  -  Loss:  0.0258  -  Accuracy:  0.9899  -  Precision:  0.9975  -  Recall:  0.9751  -  F1-score:  0.9862
Epoch  21/1000  -  Loss:  0.0212  -  Accuracy:  0.9901  -  Precision:  0.9981  -  Recall:  0.9751  -  F1-score:  0.9865
Epoch  22/1000  -  Loss:  0.0212  -  Accuracy:  0.9929  -  Precision:  0.9987  -  Recall:  0.9820  -  F1-score:  0.9903
Epoch  23/1000  -  Loss:  0.0166  -  Accuracy:  0.9947  -  Precision:  0.9987  -  Recall:  0.9869  -  F1-score:  0.9928
Epoch  24/1000  -  Loss:  0.0194  -  Accuracy:  0.9933  -  Precision:  0.9987  -  Recall:  0.9832  -  F1-score:  0.9909
Epoch  25/1000  -  Loss:  0.0201  -  Accuracy:  0.9919  -  Precision:  0.9962  -  Recall:  0.9820  -  F1-score:  0.9890
Epoch  26/1000  -  Loss:  0.0139  -  Accuracy:  0.9959  -  Precision:  0.9994  -  Recall:  0.9894  -  F1-score:  0.9944
Epoch  27/1000  -  Loss:  0.0155  -  Accuracy:  0.9940  -  Precision:  0.9987  -  Recall:  0.9851  -  F1-score:  0.9919
Epoch  28/1000  -  Loss:  0.0123  -  Accuracy:  0.9961  -  Precision:  1.0000  -  Recall:  0.9894  -  F1-score:  0.9947
Epoch  29/1000  -  Loss:  0.0114  -  Accuracy:  0.9965  -  Precision:  1.0000  -  Recall:  0.9907  -  F1-score:  0.9953
Epoch  30/1000  -  Loss:  0.0179  -  Accuracy:  0.9936  -  Precision:  0.9981  -  Recall:  0.9845  -  F1-score:  0.9912
Epoch  31/1000  -  Loss:  0.0117  -  Accuracy:  0.9959  -  Precision:  0.9994  -  Recall:  0.9894  -  F1-score:  0.9944
Epoch  32/1000  -  Loss:  0.0132  -  Accuracy:  0.9965  -  Precision:  1.0000  -  Recall:  0.9907  -  F1-score:  0.9953
Epoch  33/1000  -  Loss:  0.0135  -  Accuracy:  0.9961  -  Precision:  1.0000  -  Recall:  0.9894  -  F1-score:  0.9947
Epoch  34/1000  -  Loss:  0.0109  -  Accuracy:  0.9965  -  Precision:  1.0000  -  Recall:  0.9907  -  F1-score:  0.9953
⏹️  Early  stopping  at  epoch  34.  Best  accuracy:  0.9965
✅  Model  saved  to  /appx/model.pt
```

## License
- MIT