# CS229-Final-Project: Tennis Serve Classification

## Overview

`logreg/` implements a multi-class logistic regression model (softmax regression) to classify tennis serves into 9 categories based on pose keypoints extracted from video data. The implementation is adapted from the CS229 PS1 binary logistic regression code.

## Serve Result Classes (9 categories)
- **4**: Wide serve
- **4n**: Fault, net (wide serve)
- **4d**: Fault, deep (wide serve)
- **4w**: Fault, wide (wide serve)
- **4x**: Fault, deep and wide (wide serve)
- **5**: Body serve
- **6**: Down the T serve
- **6d**: Fault, deep (Down the T serve)
- **6n**: Fault, net (Down the T serve)

## Model Architecture

### Feature Extraction
For each serve (sequence of frames):
1. Load all keypoints from JSON files (25 keypoints × 3 values = 75 values per frame)
2. Compute statistics across frames:
   - Mean of all keypoint values
   - Standard deviation
   - Maximum values
   - Minimum values
3. Result: **300 features** (75 keypoints × 4 statistics)

### Softmax Regression
```
Input (300 features) → Linear Transform → Softmax → Probabilities (9 classes)
```

### Training Process
1. Load and filter data (exclude "Test" rows)
2. Split into train (80%) and validation (20%)
3. Normalize features (zero mean, unit variance)
4. Add intercept term
5. Initialize weights to zeros
6. Iteratively update weights using gradient descent
7. Monitor loss and accuracy every 100 iterations
8. Stop when convergence threshold is met or max iterations reached

**Generated Files:**
- `test_predictions.txt`: Predictions on test set
- `confusion_matrix_test.png`: Test set confusion matrix
