import pandas as pd
import numpy as np
import torch
import os
from .parser import interpolate_head_points, smooth_player_data, extract_keypoints, extract_bboxes
from .model import TemporalModel

def normalize_points(keypoints, bboxes):
    """
    Normalize keypoints relative to the bounding box.
    Maps keypoints so that the center of the bbox is (0,0) and the scale is relative to bbox dimensions.
    
    keypoints: (N, 17, 2)
    bboxes: (N, 4) -> x, y, w, h
    """
    bbox_x = bboxes[:, 0][:, np.newaxis] # (N, 1)
    bbox_y = bboxes[:, 1][:, np.newaxis]
    bbox_w = bboxes[:, 2][:, np.newaxis]
    bbox_h = bboxes[:, 3][:, np.newaxis]
    
    # Center of bbox
    cx = bbox_x + bbox_w / 2
    cy = bbox_y + bbox_h / 2
    
    scale = np.maximum(bbox_w, bbox_h)
    
    kps_norm = keypoints.copy()
    kps_norm[..., 0] = 2 * (keypoints[..., 0] - cx) / scale
    kps_norm[..., 1] = 2 * (keypoints[..., 1] - cy) / scale
    
    return kps_norm

def prepare_input(keypoints):
    """
    Prepares input for the TemporalModel.
    Input keypoints: (N, 17, 2)
    Output: (1, 34, N) -> (Batch, Channels, SeqLen)
    """
    # Flatten last dimension: (N, 34)
    N, J, D = keypoints.shape
    kps_flat = keypoints.reshape(N, -1) # (N, 34)
    
    # Transpose to (34, N)
    kps_transposed = kps_flat.T
    
    # Add batch dimension
    input_tensor = torch.from_numpy(kps_transposed).float().unsqueeze(0) # (1, 34, N)
    
    return input_tensor

def center_pose(prediction):
    """
    Centers the 3D pose prediction around a specific joint (usually hip).
    prediction: (N, 17, 3)
    """
    # Keypoint 11 (left_hip) and 12 (right_hip) in COCO
    # We use the midpoint of hips as root
    root = (prediction[:, 11, :] + prediction[:, 12, :]) / 2 # (N, 3)
    return prediction - root[:, np.newaxis, :]

def back_projection_from_coco17(df: pd.DataFrame, player_ids: list) -> pd.DataFrame:
    """
    Applies VideoPose3D model to infer 3D coordinates from COCO17 2D keypoints.
    
    Args:
        df: Input DataFrame containing 2D keypoints and bounding boxes.
        player_ids: List of player identifiers suffixes (e.g. ['low_drive']).
        
    Returns:
        DataFrame with added 3D coordinate columns.
    """
    # Determine path to checkpoint relative to this file
    base_path = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(base_path, 'checkpoints', 'pretrained_h36m_cpn.bin')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please verify package installation.")
        
    # Load Model
    # The checkpoint corresponds to a model with 8 convolution layers (4 blocks).
    model = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3, 3], causal=False, dropout=0.25, channels=1024)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_pos' in checkpoint:
            model.load_state_dict(checkpoint['model_pos'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    model.eval()
    
    # We work on a copy to avoid unintended side effects on the input DF if not desired,
    # and to ensure we return the full augmented dataset.
    result_df = df.copy()
    
    # Dictionary to collect new 3D columns to avoid DataFrame fragmentation
    new_3d_columns = {}
    
    keypoints_order = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    for player in player_ids:
        prefix = f"player_{player}"
        
        # 1. Processing (Interpolate & Smooth)
        result_df = interpolate_head_points(result_df, prefix)
        result_df = smooth_player_data(result_df, prefix, window_size=5)
        
        # 2. Preparation
        kps = extract_keypoints(result_df, player)
        bboxes = extract_bboxes(result_df, player)
        kps_norm = normalize_points(kps, bboxes)
        input_tensor = prepare_input(kps_norm)
        
        # 3. Inference
        with torch.no_grad():
            output = model(input_tensor)
            
        # 4. Post-processing
        output_np = output.squeeze(0).numpy().T # (N, 51)
        output_3d = output_np.reshape(-1, 17, 3) # (N, 17, 3)
        output_3d = center_pose(output_3d)
        
        # 5. Populate Result
        for i, kp in enumerate(keypoints_order):
            new_3d_columns[f"{prefix}_{kp}_3d_x"] = output_3d[:, i, 0]
            new_3d_columns[f"{prefix}_{kp}_3d_y"] = output_3d[:, i, 1]
            new_3d_columns[f"{prefix}_{kp}_3d_z"] = output_3d[:, i, 2]
            
    # Concatenate all new columns
    if new_3d_columns:
        new_df = pd.DataFrame(new_3d_columns, index=result_df.index)
        result_df = pd.concat([result_df, new_df], axis=1)
            
    return result_df
