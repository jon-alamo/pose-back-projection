# bppose

A Python package for 3D pose back-projection of Padel players from COCO17 2D keypoints using VideoPose3D.

## Installation

You can install the package directly from source:

```bash
pip install .
```

## Usage

The main entry point for the package is `back_projection_from_coco17`.

```python
import pandas as pd
import bppose

# Load your dataframe containing 2D keypoints and bboxes
df = pd.read_csv('path/to/your/dataset.csv')

# Define the player IDs derived from columns (e.g. player_low_drive_...)
player_ids = ['low_drive', 'low_left', 'up_drive', 'up_left']

# Run the 3D back-projection
df_3d = bppose.back_projection_from_coco17(df, player_ids)

# Save result
df_3d.to_csv('output_3d.csv', index=False)
```

## Input Data Format

The input `DataFrame` must contain columns following this convention for each player:
- Bounding Box: `player_<id>_x`, `player_<id>_y`, `player_<id>_w`, `player_<id>_h`
- Keypoints (COCO17): `player_<id>_<keypoint>_x`, `player_<id>_<keypoint>_y`

Keypoints supported: `nose`, `left_eye`, `right_eye`, `left_ear`, `right_ear`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`, `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`.

## Output Data Format

The output is a pandas DataFrame identical to the input but with additional columns for the 3D coordinates.

For each player identifier and for each of the 17 keypoints, three columns are added:
- `player_<id>_<keypoint>_3d_x`
- `player_<id>_<keypoint>_3d_y`
- `player_<id>_<keypoint>_3d_z`

### Coordinate System and Units
- **Reference Frame**: The 3D poses are **centered at the root** (the midpoint between the left and right hip). This means the coordinate $(0, 0, 0)$ corresponds to the player's mid-hip position for every frame.
- **Scale/Units**: The coordinates are inferred from 2D keypoints that have been normalized relative to the bounding box dimensions (-1 to 1). Therefore, the 3D units are **relative** and dimensionless (not absolute meters). They represent the reconstructed 3D structure scaled proportionally to the player's size in the image.

## Checkpoints

The package includes pre-trained weights (`pretrained_h36m_cpn.bin`) which are automatically loaded.