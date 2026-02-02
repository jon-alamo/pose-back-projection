# Instructions

When developing the application, what is defined in this file must be respected. Any change that modifies or adds any type of functionality must also be reflected in this file. For simplicity, some technical details such as function signatures, restrictions, or implementation details may be defined in the code itself in the form of docstrings. Avoid using comments whenever possible, prioritizing documentation in docstrings or this file for general topics. Always take into account what is already implemented and treat it with great caution, always trying to keep modifications minimal if necessary. Furthermore, it is not necessary to maintain any type of backward compatibility in favor of simple and readable code. When a change is implemented that could affect previous versions, the old code is cleaned up.


# Style

- The code must always be self-explanatory, avoiding unnecessary comments. It must be well encapsulated, and the use of pure functions will be prioritized over classes for greater readability, maintenance, and ease of testing. 
- It is fundamental to respect the single responsibility principle applicable to any unit of code, whether it be a function, a method, a class, or a module.
- The names of classes, functions, and variables must always be as descriptive as possible, and in general, a Pythonic code style will be respected. Avoid very dense statements in single lines. Do things explicitly and clearly, avoiding, for example, list comprehensions. 
- Add docstrings to every function, class, or module.
- Always use typehints.


# Program Specifications

The current application consists of an application of the VideoPose3D model to obtain a 3D back-projection of the positions of padel players from their plane coordinates of their corresponding elements with COCO17. The input file is a .csv with each x and y coordinate in pixels in columns as shown in the following section. Some of the coordinates for some of the keypoints may not be available, such as the nose or eyes when players have their backs to the camera. In those cases, the data is presented with a 0, but the application must interpolate the nose as halfway between the two ears and infer the position of the eyes as something between the nose and the ears.

Temporal smoothing (rolling average) with a window of 5 frames is applied to reduce noise in predictions.
The pretrained model weights are contained in the file `checkpoints/pretrained_h36m_cpn.bin`.


# Entrypoint as a package
As a python package, I need it to be importable as "import bppose" and I need the package to have a function as follows:
```python
import pandas as pd
def back_projection_from_coco17(df: pd.DataFrame, player_ids: list) -> pd.DataFrame:
    ...
    return dr_3_data
```


# Input Dataset

## Pose Dataset (dataset/<dataset_name>.csv)

### Columns:

- index: the cleaned frame position.
- source_frame_file, file_name: redundant columns with the original frame file name.
- source_frame: the frame number in the original video before cleaning repeated frames.

For each one of the fourth players, being theirs ids: low_drive, low_left, up_drive, up_left:

bboxes given by:
- player_<player_id>_x: x axis position in pixels for the upper left corner in screen, from left to right.
- player_<player_id>_y: y axis position in pixels for the upper left corner in screen, from up to bottom.
- player_<player_id>_w: horizontal width of the bbox
- player_<player_id>_h: vertical height of the bbox

keypoints for each player:
- player_<player_id>_nose_x: x axis position in pixels for the player is nose in screen.
- player_<player_id>_nose_y: y axis position in pixels for the player is nose in screen.
- player_<player_id>_left_eye_x: x axis position in pixels for the player is left_eye in screen.
- player_<player_id>_left_eye_y: y axis position in pixels for the player is left_eye in screen.
- player_<player_id>_right_eye_x: x axis position in pixels for the player is right_eye in screen.
- player_<player_id>_right_eye_y: y axis position in pixels for the player is right_eye in screen.
- player_<player_id>_left_ear_x: x axis position in pixels for the player is left_ear in screen.
- player_<player_id>_left_ear_y: y axis position in pixels for the player is left_ear in screen.
- player_<player_id>_right_ear_x: x axis position in pixels for the player is right_ear in screen.
- player_<player_id>_right_ear_y: y axis position in pixels for the player is right_ear in screen.
- player_<player_id>_left_shoulder_x: x axis position in pixels for the player is left_shoulder in screen.
- player_<player_id>_left_shoulder_y: y axis position in pixels for the player is left_shoulder in screen.
- player_<player_id>_right_shoulder_x: x axis position in pixels for the player is right_shoulder in screen.
- player_<player_id>_right_shoulder_y: y axis position in pixels for the player is right_shoulder in screen.
- player_<player_id>_left_elbow_x: x axis position in pixels for the player is left_elbow in screen.
- player_<player_id>_left_elbow_y: y axis position in pixels for the player is left_elbow in screen.
- player_<player_id>_right_elbow_x: x axis position in pixels for the player is right_elbow in screen.
- player_<player_id>_right_elbow_y: y axis position in pixels for the player is right_elbow in screen.
- player_<player_id>_left_wrist_x: x axis position in pixels for the player is left_wrist in screen.
- player_<player_id>_left_wrist_y: y axis position in pixels for the player is left_wrist in screen.
- player_<player_id>_right_wrist_x: x axis position in pixels for the player is right_wrist in screen.
- player_<player_id>_right_wrist_y: y axis position in pixels for the player is right_wrist in screen.
- player_<player_id>_left_hip_x: x axis position in pixels for the player is left_hip in screen.
- player_<player_id>_left_hip_y: y axis position in pixels for the player is left_hip in screen.
- player_<player_id>_right_hip_x: x axis position in pixels for the player is right_hip in screen.
- player_<player_id>_right_hip_y: y axis position in pixels for the player is right_hip in screen.
- player_<player_id>_left_knee_x: x axis position in pixels for the player is left_knee in screen.
- player_<player_id>_left_knee_y: y axis position in pixels for the player is left_knee in screen.
- player_<player_id>_right_knee_x: x axis position in pixels for the player is right_knee in screen.
- player_<player_id>_right_knee_y: y axis position in pixels for the player is right_knee in screen.
- player_<player_id>_left_ankle_x: x axis position in pixels for the player is left_ankle in screen.
- player_<player_id>_left_ankle_y: y axis position in pixels for the player is left_ankle in screen.
- player_<player_id>_right_ankle_x: x axis position in pixels for the player is right_ankle in screen.
- player_<player_id>_right_ankle_y: y axis position in pixels for the player is right_ankle in screen.


# Output Dataset
The output must be the 3D coordinates for every COCO17 keypoint infered from the model and expressed in relation to a part of the body, usually, the center of the hip. It must contain the following columns:

- player_<player_id>_nose_3d_x: x axis of the position in relative coordinates to the system of reference
- player_<player_id>_nose_3d_y: y axis of the position in relative coordinates to the system of reference
- player_<player_id>_nose_3d_z: z axis of the position in relative coordinates to the system of reference
- player_<player_id>_left_eye_3d_x: x axis of the position in relative coordinates to the system of reference
- player_<player_id>_left_eye_3d_y: y axis of the position in relative coordinates to the system of reference
- player_<player_id>_left_eye_3d_z: z axis of the position in relative coordinates to the system of reference
- player_<player_id>_right_eye_3d_x: x axis of the position in relative coordinates to the system of reference
- player_<player_id>_right_eye_3d_y: y axis of the position in relative coordinates to the system of reference
- player_<player_id>_right_eye_3d_z: z axis of the position in relative coordinates to the system of reference
- player_<player_id>_left_ear_3d_x: x axis of the position in relative coordinates to the system of reference
- player_<player_id>_left_ear_3d_y: y axis of the position in relative coordinates to the system of reference
- player_<player_id>_left_ear_3d_z: z axis of the position in relative coordinates to the system of reference
- player_<player_id>_right_ear_3d_x: x axis of the position in relative coordinates to the system of reference
- player_<player_id>_right_ear_3d_y: y axis of the position in relative coordinates to the system of reference
- player_<player_id>_right_ear_3d_z: z axis of the position in relative coordinates to the system of reference
- player_<player_id>_left_shoulder_3d_x: x axis of the position in relative coordinates to the system of reference
- player_<player_id>_left_shoulder_3d_y: y axis of the position in relative coordinates to the system of reference
- player_<player_id>_left_shoulder_3d_z: z axis of the position in relative coordinates to the system of reference
- player_<player_id>_right_shoulder_3d_x: x axis of the position in relative coordinates to the system of reference
- player_<player_id>_right_shoulder_3d_y: y axis of the position in relative coordinates to the system of reference
- player_<player_id>_right_shoulder_3d_z: z axis of the position in relative coordinates to the system of reference
- player_<player_id>_left_elbow_3d_x: x axis of the position in relative coordinates to the system of reference
- player_<player_id>_left_elbow_3d_y: y axis of the position in relative coordinates to the system of reference
- player_<player_id>_left_elbow_3d_z: z axis of the position in relative coordinates to the system of reference
- player_<player_id>_right_elbow_3d_x: x axis of the position in relative coordinates to the system of reference
- player_<player_id>_right_elbow_3d_y: y axis of the position in relative coordinates to the system of reference
- player_<player_id>_right_elbow_3d_z: z axis of the position in relative coordinates to the system of reference
- player_<player_id>_left_wrist_3d_x: x axis of the position in relative coordinates to the system of reference
- player_<player_id>_left_wrist_3d_y: y axis of the position in relative coordinates to the system of reference
- player_<player_id>_left_wrist_3d_z: z axis of the position in relative coordinates to the system of reference
- player_<player_id>_right_wrist_3d_x: x axis of the position in relative coordinates to the system of reference
- player_<player_id>_right_wrist_3d_y: y axis of the position in relative coordinates to the system of reference
- player_<player_id>_right_wrist_3d_z: z axis of the position in relative coordinates to the system of reference
- player_<player_id>_left_hip_3d_x: x axis of the position in relative coordinates to the system of reference
- player_<player_id>_left_hip_3d_y: y axis of the position in relative coordinates to the system of reference
- player_<player_id>_left_hip_3d_z: z axis of the position in relative coordinates to the system of reference
- player_<player_id>_right_hip_3d_x: x axis of the position in relative coordinates to the system of reference
- player_<player_id>_right_hip_3d_y: y axis of the position in relative coordinates to the system of reference
- player_<player_id>_right_hip_3d_z: z axis of the position in relative coordinates to the system of reference
- player_<player_id>_left_knee_3d_x: x axis of the position in relative coordinates to the system of reference
- player_<player_id>_left_knee_3d_y: y axis of the position in relative coordinates to the system of reference
- player_<player_id>_left_knee_3d_z: z axis of the position in relative coordinates to the system of reference
- player_<player_id>_right_knee_3d_x: x axis of the position in relative coordinates to the system of reference
- player_<player_id>_right_knee_3d_y: y axis of the position in relative coordinates to the system of reference
- player_<player_id>_right_knee_3d_z: z axis of the position in relative coordinates to the system of reference
- player_<player_id>_left_ankle_3d_x: x axis of the position in relative coordinates to the system of reference
- player_<player_id>_left_ankle_3d_y: y axis of the position in relative coordinates to the system of reference
- player_<player_id>_left_ankle_3d_z: z axis of the position in relative coordinates to the system of reference
- player_<player_id>_right_ankle_3d_x: x axis of the position in relative coordinates to the system of reference
- player_<player_id>_right_ankle_3d_y: y axis of the position in relative coordinates to the system of reference
- player_<player_id>_right_ankle_3d_z: z axis of the position in relative coordinates to the system of reference