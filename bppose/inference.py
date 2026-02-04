import pandas as pd
import numpy as np
import torch
import os
from typing import Optional, Tuple
from .parser import interpolate_head_points, smooth_player_data, extract_keypoints, extract_bboxes
from .model import TemporalModel


# Human3.6M joint indices (what the model expects)
H36M_JOINTS = {
    'hip': 0,           # Root/Pelvis (center)
    'right_hip': 1,
    'right_knee': 2,
    'right_ankle': 3,
    'left_hip': 4,
    'left_knee': 5,
    'left_ankle': 6,
    'spine': 7,         # Between hips and thorax
    'thorax': 8,        # Chest/upper spine
    'neck_nose': 9,     # Neck or nose position
    'head': 10,         # Top of head
    'left_shoulder': 11,
    'left_elbow': 12,
    'left_wrist': 13,
    'right_shoulder': 14,
    'right_elbow': 15,
    'right_wrist': 16,
}

# COCO17 joint indices (what we have)
COCO_JOINTS = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16,
}


def convert_coco_to_h36m(coco_keypoints: np.ndarray) -> np.ndarray:
    """
    Converts COCO17 keypoints to Human3.6M format for VideoPose3D.
    
    COCO17 has face keypoints but no spine. H36M has spine but no face details.
    We map matching joints and infer missing ones.
    
    Args:
        coco_keypoints: Array of shape (N, 17, 2) with COCO keypoints.
        
    Returns:
        Array of shape (N, 17, 2) with H36M format keypoints.
    """
    n_frames = coco_keypoints.shape[0]
    h36m_keypoints = np.zeros((n_frames, 17, 2), dtype=np.float32)
    
    # Direct mappings for limbs
    h36m_keypoints[:, H36M_JOINTS['left_hip']] = coco_keypoints[:, COCO_JOINTS['left_hip']]
    h36m_keypoints[:, H36M_JOINTS['right_hip']] = coco_keypoints[:, COCO_JOINTS['right_hip']]
    h36m_keypoints[:, H36M_JOINTS['left_knee']] = coco_keypoints[:, COCO_JOINTS['left_knee']]
    h36m_keypoints[:, H36M_JOINTS['right_knee']] = coco_keypoints[:, COCO_JOINTS['right_knee']]
    h36m_keypoints[:, H36M_JOINTS['left_ankle']] = coco_keypoints[:, COCO_JOINTS['left_ankle']]
    h36m_keypoints[:, H36M_JOINTS['right_ankle']] = coco_keypoints[:, COCO_JOINTS['right_ankle']]
    h36m_keypoints[:, H36M_JOINTS['left_shoulder']] = coco_keypoints[:, COCO_JOINTS['left_shoulder']]
    h36m_keypoints[:, H36M_JOINTS['right_shoulder']] = coco_keypoints[:, COCO_JOINTS['right_shoulder']]
    h36m_keypoints[:, H36M_JOINTS['left_elbow']] = coco_keypoints[:, COCO_JOINTS['left_elbow']]
    h36m_keypoints[:, H36M_JOINTS['right_elbow']] = coco_keypoints[:, COCO_JOINTS['right_elbow']]
    h36m_keypoints[:, H36M_JOINTS['left_wrist']] = coco_keypoints[:, COCO_JOINTS['left_wrist']]
    h36m_keypoints[:, H36M_JOINTS['right_wrist']] = coco_keypoints[:, COCO_JOINTS['right_wrist']]
    
    # Inferred joints
    # Hip center (root) = midpoint of left and right hip
    h36m_keypoints[:, H36M_JOINTS['hip']] = (
        coco_keypoints[:, COCO_JOINTS['left_hip']] + 
        coco_keypoints[:, COCO_JOINTS['right_hip']]
    ) / 2
    
    # Thorax = midpoint of shoulders
    thorax = (
        coco_keypoints[:, COCO_JOINTS['left_shoulder']] + 
        coco_keypoints[:, COCO_JOINTS['right_shoulder']]
    ) / 2
    h36m_keypoints[:, H36M_JOINTS['thorax']] = thorax
    
    # Spine = midpoint between hip center and thorax
    hip_center = h36m_keypoints[:, H36M_JOINTS['hip']]
    h36m_keypoints[:, H36M_JOINTS['spine']] = (hip_center + thorax) / 2
    
    # Neck/Nose position = use COCO nose
    h36m_keypoints[:, H36M_JOINTS['neck_nose']] = coco_keypoints[:, COCO_JOINTS['nose']]
    
    # Head = midpoint of ears (top of head approximation)
    h36m_keypoints[:, H36M_JOINTS['head']] = (
        coco_keypoints[:, COCO_JOINTS['left_ear']] + 
        coco_keypoints[:, COCO_JOINTS['right_ear']]
    ) / 2
    
    return h36m_keypoints


def convert_h36m_to_coco(h36m_3d: np.ndarray) -> np.ndarray:
    """
    Converts Human3.6M 3D output back to COCO17 format.
    
    Some COCO joints (eyes, ears) don't exist in H36M, so we approximate them.
    
    Args:
        h36m_3d: Array of shape (N, 17, 3) with H36M 3D coordinates.
        
    Returns:
        Array of shape (N, 17, 3) with COCO17 format 3D coordinates.
    """
    n_frames = h36m_3d.shape[0]
    coco_3d = np.zeros((n_frames, 17, 3), dtype=np.float32)
    
    # Direct mappings
    coco_3d[:, COCO_JOINTS['left_hip']] = h36m_3d[:, H36M_JOINTS['left_hip']]
    coco_3d[:, COCO_JOINTS['right_hip']] = h36m_3d[:, H36M_JOINTS['right_hip']]
    coco_3d[:, COCO_JOINTS['left_knee']] = h36m_3d[:, H36M_JOINTS['left_knee']]
    coco_3d[:, COCO_JOINTS['right_knee']] = h36m_3d[:, H36M_JOINTS['right_knee']]
    coco_3d[:, COCO_JOINTS['left_ankle']] = h36m_3d[:, H36M_JOINTS['left_ankle']]
    coco_3d[:, COCO_JOINTS['right_ankle']] = h36m_3d[:, H36M_JOINTS['right_ankle']]
    coco_3d[:, COCO_JOINTS['left_shoulder']] = h36m_3d[:, H36M_JOINTS['left_shoulder']]
    coco_3d[:, COCO_JOINTS['right_shoulder']] = h36m_3d[:, H36M_JOINTS['right_shoulder']]
    coco_3d[:, COCO_JOINTS['left_elbow']] = h36m_3d[:, H36M_JOINTS['left_elbow']]
    coco_3d[:, COCO_JOINTS['right_elbow']] = h36m_3d[:, H36M_JOINTS['right_elbow']]
    coco_3d[:, COCO_JOINTS['left_wrist']] = h36m_3d[:, H36M_JOINTS['left_wrist']]
    coco_3d[:, COCO_JOINTS['right_wrist']] = h36m_3d[:, H36M_JOINTS['right_wrist']]
    
    # Nose from neck_nose
    coco_3d[:, COCO_JOINTS['nose']] = h36m_3d[:, H36M_JOINTS['neck_nose']]
    
    # Approximate eyes and ears from head and neck positions
    head = h36m_3d[:, H36M_JOINTS['head']]
    neck = h36m_3d[:, H36M_JOINTS['neck_nose']]
    
    # Eyes: slightly below head, between neck and head
    eye_pos = neck + 0.7 * (head - neck)
    head_width = 0.05  # Approximate offset for left/right
    
    coco_3d[:, COCO_JOINTS['left_eye']] = eye_pos.copy()
    coco_3d[:, COCO_JOINTS['left_eye'], 0] -= head_width
    
    coco_3d[:, COCO_JOINTS['right_eye']] = eye_pos.copy()
    coco_3d[:, COCO_JOINTS['right_eye'], 0] += head_width
    
    # Ears: at head level, offset to sides
    ear_offset = 0.08
    coco_3d[:, COCO_JOINTS['left_ear']] = head.copy()
    coco_3d[:, COCO_JOINTS['left_ear'], 0] -= ear_offset
    
    coco_3d[:, COCO_JOINTS['right_ear']] = head.copy()
    coco_3d[:, COCO_JOINTS['right_ear'], 0] += ear_offset
    
    return coco_3d


def normalize_to_bbox(keypoints: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    """
    Normalize keypoints relative to their bounding box.
    
    This removes perspective scale differences between players at different
    distances from the camera. The normalization centers keypoints at the
    bbox center and scales by the bbox size.
    
    Args:
        keypoints: Array of shape (N, 17, 2) with pixel coordinates.
        bboxes: Array of shape (N, 4) with (x, y, w, h) for each frame.
        
    Returns:
        Normalized keypoints in range approximately [-1, 1].
    """
    n_frames = keypoints.shape[0]
    kps_norm = np.zeros_like(keypoints, dtype=np.float32)
    
    bbox_x = bboxes[:, 0]
    bbox_y = bboxes[:, 1]
    bbox_w = bboxes[:, 2]
    bbox_h = bboxes[:, 3]
    
    cx = bbox_x + bbox_w / 2
    cy = bbox_y + bbox_h / 2
    
    scale = np.maximum(bbox_w, bbox_h)
    
    for j in range(17):
        kps_norm[:, j, 0] = (keypoints[:, j, 0] - cx) / scale
        kps_norm[:, j, 1] = (keypoints[:, j, 1] - cy) / scale
    
    return kps_norm


def prepare_input_batched(
    keypoints: np.ndarray, 
    receptive_field: int
) -> torch.Tensor:
    """
    Prepares input for the TemporalModel with proper padding for receptive field.
    
    The model uses dilated convolutions and needs a sequence of frames as context.
    We pad the sequence at both ends to get output for all frames.
    
    Args:
        keypoints: Array of shape (N, 17, 2) with normalized coordinates.
        receptive_field: The receptive field of the model in frames.
        
    Returns:
        Tensor of shape (1, N + padding, 17, 2).
    """
    n_frames = keypoints.shape[0]
    pad = receptive_field // 2
    
    # Pad by repeating edge frames
    padded = np.pad(
        keypoints,
        ((pad, pad), (0, 0), (0, 0)),
        mode='edge'
    )
    
    input_tensor = torch.from_numpy(padded).float().unsqueeze(0)
    return input_tensor


def center_pose(prediction: np.ndarray) -> np.ndarray:
    """
    Centers the 3D pose prediction around the hip midpoint.
    
    Args:
        prediction: Array of shape (N, 17, 3) with 3D coordinates.
        
    Returns:
        Centered 3D coordinates with hip midpoint at origin.
    """
    root = (prediction[:, COCO_JOINTS['left_hip'], :] + prediction[:, COCO_JOINTS['right_hip'], :]) / 2
    return prediction - root[:, np.newaxis, :]


def estimate_image_dimensions(df: pd.DataFrame, player_ids: list) -> Tuple[int, int]:
    """
    Estimates image dimensions from the keypoint data.
    
    Args:
        df: DataFrame with keypoint columns.
        player_ids: List of player identifiers.
        
    Returns:
        Tuple of (width, height) in pixels.
    """
    max_x = 0
    max_y = 0
    
    for player in player_ids:
        prefix = f"player_{player}"
        x_cols = [c for c in df.columns if c.startswith(prefix) and c.endswith('_x')]
        y_cols = [c for c in df.columns if c.startswith(prefix) and c.endswith('_y')]
        
        for col in x_cols:
            max_x = max(max_x, df[col].max())
        for col in y_cols:
            max_y = max(max_y, df[col].max())
    
    width = int(np.ceil(max_x / 100) * 100)
    height = int(np.ceil(max_y / 100) * 100)
    
    if width == 0:
        width = 1920
    if height == 0:
        height = 1080
        
    return width, height


def back_projection_from_coco17(
    df: pd.DataFrame, 
    player_ids: list,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None
) -> pd.DataFrame:
    """
    Applies VideoPose3D model to infer 3D coordinates from COCO17 2D keypoints.
    
    The model was trained on Human3.6M format, so we convert COCO → H36M for
    inference, then convert the output back to COCO format.
    
    Args:
        df: Input DataFrame containing 2D keypoints and bounding boxes.
        player_ids: List of player identifiers suffixes (e.g. ['low_drive']).
        image_width: Unused, kept for API compatibility.
        image_height: Unused, kept for API compatibility.
        
    Returns:
        DataFrame with added 3D coordinate columns.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(base_path, 'checkpoints', 'pretrained_h36m_cpn.bin')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}.")
        
    model = TemporalModel(
        num_joints_in=17, 
        in_features=2, 
        num_joints_out=17, 
        filter_widths=[3, 3, 3, 3, 3],  # 5 elements = 1 expand + 4 residual blocks
        causal=False, 
        dropout=0.25, 
        channels=1024
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if 'model_pos' in checkpoint:
        model.load_state_dict(checkpoint['model_pos'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    
    receptive_field = model.receptive_field()
    
    result_df = df.copy()
    new_3d_columns = {}
    
    keypoints_order = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    for player in player_ids:
        prefix = f"player_{player}"
        
        result_df = interpolate_head_points(result_df, prefix)
        result_df = smooth_player_data(result_df, prefix, window_size=5)
        
        # Extract COCO keypoints
        coco_kps = extract_keypoints(result_df, player)
        bboxes = extract_bboxes(result_df, player)
        
        # Convert COCO → H36M format
        h36m_kps = convert_coco_to_h36m(coco_kps)
        
        # Normalize for model input
        h36m_kps_norm = normalize_to_bbox(h36m_kps, bboxes)
        input_tensor = prepare_input_batched(h36m_kps_norm, receptive_field)
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
            
        # Output is in H36M format, convert back to COCO
        h36m_3d = output.squeeze(0).numpy()
        coco_3d = convert_h36m_to_coco(h36m_3d)
        
        # Center the pose
        output_3d = center_pose(coco_3d)
        
        for i, kp in enumerate(keypoints_order):
            new_3d_columns[f"{prefix}_{kp}_3d_x"] = output_3d[:, i, 0]
            new_3d_columns[f"{prefix}_{kp}_3d_y"] = output_3d[:, i, 1]
            new_3d_columns[f"{prefix}_{kp}_3d_z"] = output_3d[:, i, 2]
            
    if new_3d_columns:
        new_df = pd.DataFrame(new_3d_columns, index=result_df.index)
        result_df = pd.concat([result_df, new_df], axis=1)
            
    return result_df
