# Highway Networks — Experimental Summary

## Experiment 1: Basic Highway CNN

**Notebook:** `03_cifar10_training.ipynb`

We implemented a CNN with four Highway blocks and trained it on CIFAR-10.

**Results:**

- Test accuracy: 55.79%
- Trainable parameters: 75,210
- Training epochs: 10

**Lesson:** A convolutional Highway Network can be trained end to end with backpropagation.

## Experiment 2: Shallow Highway vs. Standard CNN

We trained both architectures using 45,000 training images and 5,000 validation images.

**Results:**

- Highway CNN: 56.84% validation accuracy
- Standard CNN: 55.84% validation accuracy

**Lesson:** Highway gates produced only a modest improvement at shallow depth.

## Experiment 3: Increasing Network Depth

**Notebook:** `04_depth_experiments.ipynb`

We compared networks with 4, 10, 20, 50, and 100 internal blocks.

| Blocks | Standard CNN | Highway CNN |
| ------ | -----------: | ----------: |
| 4      |       56.68% |      57.34% |
| 10     |       61.18% |      63.92% |
| 20     |       46.82% |      63.44% |
| 50     |       10.34% |  **66.06%** |
| 100    |        9.86% |       9.86% |

Values are final validation accuracies.

The 100-block Highway CNN initially reached 36.48% validation accuracy but subsequently collapsed.

**Lesson:** Highway Networks showed a substantial optimization advantage at 50 blocks, although the 100-block architecture remained unstable.

## Experiment 4: Gate Bias and Gradient Propagation

We investigated transform gate biases of $-2$, $-3$, $-4$, and $-5$.

At 100 blocks:

| Bias | Final Activation RMS |
| ---- | -------------------: |
| -2   |             0.015535 |
| -3   |             0.018929 |
| -4   |             0.092563 |
| -5   |             0.233284 |

We also found that changing the gate bias from $-2$ to $-5$ greatly increased early-layer gradient magnitudes at initialization.

A separate training run with bias $-5$ initially improved, reaching 32.42% validation accuracy at epoch 2, but experienced instability at epoch 3.

**Lesson:** Gate initialization strongly influences information and gradient propagation, but does not guarantee stable training.

## Experiment 5: Improved Multi-Stage Highway CNN

**Notebook:** `05_improved_highway_cifar10.ipynb`

We implemented a stronger CNN with three feature stages, spatial downsampling, normalization, regularization, and a learning-rate schedule.

**Results:**

- Training accuracy: 93.30%
- Validation accuracy: 88.80%
- Test accuracy: **88.07%**
- Parameters: 870,250
- Training epochs: 20

**Lesson:** A stronger image classification architecture and training procedure substantially improved our results.

## Experiment 6: Final Highway vs. Standard CNN Comparison

**Notebook:** `06_final_model_comparison.ipynb`

We compared the improved Highway CNN with a Standard CNN using the same overall feature stages and training procedure.

| Metric              | Highway CNN | Standard CNN |
| ------------------- | ----------: | -----------: |
| Parameters          |     870,250 |      482,730 |
| Training accuracy   |      93.30% |       91.39% |
| Validation accuracy |  **88.80%** |       88.42% |
| Validation loss     |      0.3456 |       0.3515 |
| Training time       |    18.5 min |      9.0 min |

**Lesson:** Both models achieved similar accuracy, but the Highway CNN required more parameters and computation.

## Overall Research Conclusion

Our results suggest that the most distinctive advantage of Highway Networks appears when optimization becomes difficult as network depth increases.

For a strong multi-stage CIFAR-10 classifier, Highway gates provided only a small improvement over conventional convolutional blocks.

Our experiments support the qualitative motivation of the original Highway Networks paper, but they are not an exact reproduction and do not establish a statistically consistent gating advantage.

**Main achievement:** We implemented Highway Networks, investigated information and gradient flow, analyzed optimization failures, and achieved 88.07% CIFAR-10 test accuracy with an improved model.
