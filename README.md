# CS229-Final-Project: Tennis Serve Classification

## Overview

`logreg/` implements a multi-class logistic regression model (softmax regression) to classify tennis serves into 9 categories based on pose keypoints extracted from video data. The implementation is adapted from the CS229 PS1 binary logistic regression code.

### Serve Result Classes (9 categories)
- **4**: Wide serve
- **4n**: Fault, net (wide serve)
- **4d**: Fault, deep (wide serve)
- **4w**: Fault, wide (wide serve)
- **4x**: Fault, deep and wide (wide serve)
- **5**: Body serve
- **6**: Down the T serve
- **6d**: Fault, deep (Down the T serve)
- **6n**: Fault, net (Down the T serve)