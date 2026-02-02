import pandas as pd
import numpy as np

def interpolate_head_points(player_data: pd.DataFrame, player_prefix: str) -> pd.DataFrame:
    """
    Interpolates missing nose and eye keypoints.
    Logic:
    - Nose: midpoint between left_ear and right_ear.
    - Left Eye: midpoint between nose and left_ear.
    - Right Eye: midpoint between nose and right_ear.
    
    This function modifies the DataFrame in place for the specified columns if they are 0.
    """
    # Define column names
    nose_x, nose_y = f"{player_prefix}_nose_x", f"{player_prefix}_nose_y"
    lear_x, lear_y = f"{player_prefix}_left_ear_x", f"{player_prefix}_left_ear_y"
    rear_x, rear_y = f"{player_prefix}_right_ear_x", f"{player_prefix}_right_ear_y"
    leye_x, leye_y = f"{player_prefix}_left_eye_x", f"{player_prefix}_left_eye_y"
    reye_x, reye_y = f"{player_prefix}_right_eye_x", f"{player_prefix}_right_eye_y"

    # Vectorized interpolation
    # Only apply where nose is missing (0,0 assumed as missing or close to it)
    # We assume if x is 0, y is likely 0 or invalid.
    
    # Interpolate Nose
    missing_nose = (player_data[nose_x] == 0) & (player_data[lear_x] != 0) & (player_data[rear_x] != 0)
    player_data.loc[missing_nose, nose_x] = (player_data.loc[missing_nose, lear_x] + player_data.loc[missing_nose, rear_x]) / 2
    player_data.loc[missing_nose, nose_y] = (player_data.loc[missing_nose, lear_y] + player_data.loc[missing_nose, rear_y]) / 2

    # Interpolate Left Eye (between nose and left ear)
    missing_leye = (player_data[leye_x] == 0) & (player_data[nose_x] != 0) & (player_data[lear_x] != 0)
    player_data.loc[missing_leye, leye_x] = (player_data.loc[missing_leye, nose_x] + player_data.loc[missing_leye, lear_x]) / 2
    player_data.loc[missing_leye, leye_y] = (player_data.loc[missing_leye, nose_y] + player_data.loc[missing_leye, lear_y]) / 2

    # Interpolate Right Eye (between nose and right ear)
    missing_reye = (player_data[reye_x] == 0) & (player_data[nose_x] != 0) & (player_data[rear_x] != 0)
    player_data.loc[missing_reye, reye_x] = (player_data.loc[missing_reye, nose_x] + player_data.loc[missing_reye, rear_x]) / 2
    player_data.loc[missing_reye, reye_y] = (player_data.loc[missing_reye, nose_y] + player_data.loc[missing_reye, rear_y]) / 2
    
    return player_data

def smooth_player_data(df: pd.DataFrame, player_prefix: str, window_size: int = 5) -> pd.DataFrame:
    """
    Applies temporal smoothing to the player's keypoints.
    Strategy:
    1. Identify keypoint columns.
    2. Replace 0s with NaN to avoid smearing zeros into valid data.
    3. Interpolate missing values linearly to fill gaps.
    4. Apply a rolling average filter to smooth jitter.
    5. Fill remaining NaNs with 0 (if any at start/end).
    """
    # Identify columns for this player
    # keypoints: x, y for 17 kps
    # We also have bboxes (x, y, w, h). We might want to smooth them too.
    
    cols = [c for c in df.columns if c.startswith(player_prefix) and (c.endswith('_x') or c.endswith('_y') or c.endswith('_w') or c.endswith('_h'))]
    
    player_df = df[cols].copy()
    
    # Replace 0 with NaN for valid interpolation
    # Note: 0 is exactly 0.
    player_df.replace(0, np.nan, inplace=True)
    
    # Interpolate to fill gaps
    player_df.interpolate(method='linear', axis=0, limit_direction='both', inplace=True)
    
    # Smooth
    # min_periods=1 ensures we get values even at edges
    player_df = player_df.rolling(window=window_size, center=True, min_periods=1).mean()
    
    # Update original dataframe
    # We use fillna(0) just in case
    df[cols] = player_df.fillna(0)
    
    return df

def load_and_process_data(filepath: str) -> dict:
    """
    Loads the dataset and processes it for all 4 players.
    Returns a dictionary {player_id: dataframe}.
    """
    df = pd.read_csv(filepath)
    players = ['low_drive', 'low_left', 'up_drive', 'up_left']
    processed_data = {}

    for player in players:
        prefix = f"player_{player}"
        
        # 1. Custom head interpolation (as per specs)
        df = interpolate_head_points(df, prefix)
        
        # 2. Temporal smoothing (additional filtering)
        df = smooth_player_data(df, prefix, window_size=5)
        
        # We might want to separate them now or just keep the reference
        processed_data[player] = df 
        
    return processed_data

def extract_keypoints(df: pd.DataFrame, player_id: str):
    """
    Extracts the 17 COCO keypoints (x, y) for a specific player into a numpy array (N, 17, 2).
    """
    # COCO17 Keypoint order matters for the model. 
    # Standard COCO order: 
    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear, 
    # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow, 
    # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip, 
    # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    
    keypoints_order = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    prefix = f"player_{player_id}"
    coordinates = []
    
    for kp in keypoints_order:
        x_col = f"{prefix}_{kp}_x"
        y_col = f"{prefix}_{kp}_y"
        # Stack x and y
        coordinates.append(df[[x_col, y_col]].to_numpy())
        
    # Stack to get (17, N, 2) then transpose to (N, 17, 2)
    # current list is 17 items of shape (N, 2)
    coords_array = np.stack(coordinates, axis=1) # (N, 17, 2)
    
    return coords_array

def extract_bboxes(df: pd.DataFrame, player_id: str):
    """
    Extracts bboxes (x, y, w, h) for a specific player into a numpy array (N, 4).
    """
    prefix = f"player_{player_id}"
    cols = [f"{prefix}_x", f"{prefix}_y", f"{prefix}_w", f"{prefix}_h"]
    return df[cols].to_numpy()


