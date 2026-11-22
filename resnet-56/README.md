# ResNet-56 on CIFAR-10

A PyTorch implementation of the CIFAR-10 ResNet-56 architecture, with 56
weighted layers and 853,018 trainable parameters.
[Deep Residual Learning for Image Recognition, §4.2](https://arxiv.org/pdf/1512.03385#page=7)

## Notebooks

1. [Architecture and first implementation](notebooks/01_paper_and_model.ipynb)
   explains residual blocks, shortcut options, and the full network.
2. [Train on CIFAR-10](notebooks/02_train_resnet56.ipynb)
   prepares the data, walks through an SGD update, and trains the model.
3. [Evaluate and interpret the results](notebooks/03_evaluate_resnet56.ipynb)
   plots the learning curves and evaluates the saved model on the test set.

## Results

| Metric | Result |
| --- | ---: |
| Best validation accuracy (epoch 176) | 93.52% |
| Test accuracy | 92.41% |
| Test error | 7.59% |
| Test cross-entropy loss | 0.3440 |

The model trained for 64,000 SGD updates on 45,000 CIFAR-10 images with seed
42. We selected the checkpoint by accuracy on 5,000 validation images. It
correctly classified 9,241 of the 10,000 held-out test images. The paper reports 6.97%
test error. It used a 45,000/5,000 split to choose the training schedule.
[ResNet paper, §4.2 and Table 6](https://arxiv.org/pdf/1512.03385#page=7)

## Reproduce the run

Follow the [environment setup](../README.md#setup), then run Notebook 2 with
`RUN_MODE = "start"`. Use `"resume"` to continue from the latest saved
checkpoint. Notebook 3 reads `best.pth` for evaluation and `latest.pth` for
the full metric history. The checkpoints are stored locally in
`checkpoints/sgd-paper/` and ignored by Git. Notebook 3 includes the saved
plots and test results.
