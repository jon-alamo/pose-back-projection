# Padel Player 3D Pose Back-Projection

This project provides a Python package (`bppose`) and a script to infer 3D pose coordinates from 2D COCO17 keypoints for padel players. It utilizes a pretrained **VideoPose3D** model to perform back-projection from 2D pixel coordinates to a 3D space.

## Installation

### Requirements
- Python 3.7+
- torch
- pandas
- numpy

Install dependencies:
```bash
pip install -r requirements.txt
```

To install the package in editable mode:
```bash
pip install -e .
```

## Usage

### As a Library
You can import `bppose` and use the `back_projection_from_coco17` function directly.

```python
import pandas as pd
import bppose

# Load your dataframe with 2D keypoints
df = pd.read_csv('path/to/data.csv')

# Define player identifiers (suffixes in your column names)
player_ids = ['low_drive', 'low_left', 'up_drive', 'up_left']

# Perform inference
df_3d = bppose.back_projection_from_coco17(df, player_ids)
```

### Running the Script
The repository includes a `main.py` script that processes a sample dataset found in `datasets/`.

```bash
python main.py
```
This will generate an `output_3d.csv` file with the results.

## Input Data Format

The input must be a CSV file containing columns for 2D keypoints and bounding boxes. 

**Bounding Boxes**: `player_<id>_{x, y, w, h}`  
**Keypoints**: `player_<id>_{keypoint}_{x, y}`  

Keypoints supported: `nose`, `left_eye`, `right_eye`, `left_ear`, `right_ear`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`, `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`.

## Output Data Format

The output is a pandas DataFrame identical to the input but with additional columns for the 3D coordinates.

For each player identifier and for each of the 17 keypoints, three columns are added:
- `player_<id>_<keypoint>_3d_x`
- `player_<id>_<keypoint>_3d_y`
- `player_<id>_<keypoint>_3d_z`

### Coordinate System and Units
- **Reference Frame**: The 3D poses are **centered at the root** (the midpoint between the left and right hip). This means the coordinate $(0, 0, 0)$ corresponds to the player's mid-hip position for every frame.
- **Scale/Units (Normalized Body Space)**: 
    - The output coordinates are in an arbitrary **normalized model space**. 
    - Because the input is normalized using the bounding box ($[-1, 1]$), the output (3D skeleton) **does not shrink or grow** when the player moves away or towards the camera. The size of the reconstructed skeleton remains roughly constant regardless of the player's distance from the camera.
    - **To convert to Real-World Units (e.g., meters)**: You cannot use the bounding box size directly because it depends on distance and perspective. Instead, you should calculate a scaling factor based on a known physical dimension. A common approach is **Bone Length Calibration**:
        1. Calculate the height of the inferred 3D skeleton (e.g., distance from ankle to head) or a sum of fixed bone lengths (thigh + shin).
        2. obtain the ratio between the Real World Height of the player (e.g., 1.80m) and the Inferred Skeleton Height.
        3. Multiply all 3D coordinates by this ratio.

## Checkpoints

The system relies on a pretrained model checkpoint located at `bppose/checkpoints/pretrained_h36m_cpn.bin`. This file is included in the package distribution.