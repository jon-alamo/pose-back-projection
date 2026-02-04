# Padel Player 3D Pose Back-Projection

This project provides a Python package (`bppose`) and a script to infer 3D pose coordinates from 2D COCO17 keypoints for padel players. It utilizes a pretrained **VideoPose3D** model to perform back-projection from 2D pixel coordinates to a 3D space.

## Installation

### Requirements
- Python 3.7+
- torch
- pandas
- numpy
- matplotlib

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

## Visualization and Diagnostics

The package includes visualization tools to compare 2D input poses with inferred 3D poses. This is useful for verifying inference quality.

### Single Frame Comparison
Compare 2D and 3D pose side-by-side for a specific frame:

```python
import bppose

# After running inference
bppose.visualize_pose_comparison(df_3d, player_id='low_drive', frame_idx=100)

# Save to file instead of displaying
bppose.visualize_pose_comparison(df_3d, 'low_drive', frame_idx=100, save_path='frame_100.png')
```

### Sequence Visualization
Visualize multiple consecutive frames in a grid (2D on top row, 3D on bottom):

```python
bppose.visualize_sequence(
    df_3d,
    player_id='low_drive',
    start_frame=100,
    num_frames=5,
    output_dir='diagnostics/sequences'  # Optional: saves to file
)
```

### Animated Visualization
Create an animated GIF showing the pose evolution over time:

```python
bppose.visualize_sequence_animated(
    df_3d,
    player_id='low_drive',
    start_frame=0,
    num_frames=30,
    output_path='animation.gif',
    fps=10
)
```

### Diagnostic Report
Generate a diagnostic report that samples frames evenly across the dataset:

```python
bppose.generate_diagnostic_report(
    df_3d,
    player_id='low_drive',
    sample_frames=5,
    output_dir='diagnostics/low_drive'
)
```

### Running Full Diagnostics
A convenience script `visualize_poses.py` is included to generate diagnostics for all players:

```bash
python visualize_poses.py
```

This creates diagnostic images in the `diagnostics/` folder organized by player.

## Input Data Format

The input must be a CSV file containing columns for 2D keypoints and bounding boxes. 

**Bounding Boxes**: `player_<id>_{x, y, w, h}`  
**Keypoints**: `player_<id>_{keypoint}_{x, y}`  

Keypoints supported: `nose`, `left_eye`, `right_eye`, `left_ear`, `right_ear`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`, `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`.

Missing keypoints (value 0) are handled automatically:
- Nose is interpolated as the midpoint between the ears
- Eyes are inferred as points between the nose and ears

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
    - Because the input is normalized using the bounding box, the output (3D skeleton) **does not shrink or grow** when the player moves away or towards the camera. The size of the reconstructed skeleton remains roughly constant regardless of the player's distance from the camera.
    - **To convert to Real-World Units (e.g., meters)**: You cannot use the bounding box size directly because it depends on distance and perspective. Instead, you should calculate a scaling factor based on a known physical dimension. A common approach is **Bone Length Calibration**:
        1. Calculate the height of the inferred 3D skeleton (e.g., distance from ankle to head) or a sum of fixed bone lengths (thigh + shin).
        2. Obtain the ratio between the Real World Height of the player (e.g., 1.80m) and the Inferred Skeleton Height.
        3. Multiply all 3D coordinates by this ratio.

## Technical Details

### Skeleton Format Conversion
The VideoPose3D model was trained on Human3.6M dataset which uses a different skeleton format than COCO17. The package automatically handles the conversion:
- **COCO17 → H36M**: Maps matching joints and infers missing ones (hip center, spine, thorax) from available keypoints
- **H36M → COCO17**: Converts the 3D output back to COCO format, approximating face keypoints from head position

### Normalization
Keypoints are normalized relative to each player's bounding box before inference. This removes perspective scale differences between players at different distances from the camera.

## Checkpoints

The system relies on a pretrained model checkpoint located at `bppose/checkpoints/pretrained_h36m_cpn.bin`. This file is included in the package distribution.