# Handwritten Character Recognition (CNN)

Recognises handwritten digits (MNIST) and letters A-Z (EMNIST) using a Convolutional Neural Network built with PyTorch.

## Results
| Model | Dataset | Test accuracy |
|---|---|---|
| CNN | MNIST (digits) | ~99% |
| CNN | EMNIST (letters) | ~93.4% |

## Setup
```
pip install -r requirements.txt
```

## Train
```
python3 train.py --dataset mnist
python3 train.py --dataset emnist
```
Saves `model_<dataset>.pth`, training curves and sample predictions.

## Predict on your own image
```
python3 predict.py my_digit.png --dataset mnist
python3 predict.py my_letter.png --dataset emnist --save-debug
```
The script removes shadows, crops to the character, thickens thin lines and centres it before prediction. Any pen colour works. `--save-debug` saves the cleaned 28x28 image the model sees.

## Tested on my own handwriting
| Image | Predicted | Confidence |
|---|---|---|
| My_5 | 5 | 100.0% |
| My_A (rounded style) | H (wrong) | 64.6% |
| My_A2 (standard style) | A | 98.7% |

Writing style matters: the rounded A was misread, while a standard pointed A was read correctly.

## Future work
Extend to words and sentences with a CRNN and CTC loss.