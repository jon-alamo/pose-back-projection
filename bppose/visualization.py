"""
Visualization module for comparing 2D input poses with inferred 3D poses.
Provides tools to visualize frame sequences side-by-side.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional
import os


COCO17_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

COCO17_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6),  # Shoulders
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 11), (6, 12),  # Torso
    (11, 12),  # Hips
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
]

SKELETON_COLORS = {
    'head': '#FF6B6B',
    'torso': '#4ECDC4',
    'left_arm': '#45B7D1',
    'right_arm': '#96CEB4',
    'left_leg': '#FFEAA7',
    'right_leg': '#DDA0DD',
}


def get_bone_color(bone_idx: int) -> str:
    """
    Returns the color for a specific bone based on body part.
    """
    if bone_idx < 4:
        return SKELETON_COLORS['head']
    elif bone_idx == 4:
        return SKELETON_COLORS['torso']
    elif bone_idx in [5, 6]:
        return SKELETON_COLORS['left_arm']
    elif bone_idx in [7, 8]:
        return SKELETON_COLORS['right_arm']
    elif bone_idx in [9, 10, 11]:
        return SKELETON_COLORS['torso']
    elif bone_idx in [12, 13]:
        return SKELETON_COLORS['left_leg']
    else:
        return SKELETON_COLORS['right_leg']


def extract_2d_pose_for_frame(df, player_id: str, frame_idx: int) -> np.ndarray:
    """
    Extracts 2D keypoints for a single frame.
    
    Returns:
        Array of shape (17, 2) with x, y coordinates.
    """
    prefix = f"player_{player_id}"
    pose = np.zeros((17, 2))
    
    for i, kp in enumerate(COCO17_KEYPOINTS):
        x_col = f"{prefix}_{kp}_x"
        y_col = f"{prefix}_{kp}_y"
        pose[i, 0] = df.iloc[frame_idx][x_col]
        pose[i, 1] = df.iloc[frame_idx][y_col]
    
    return pose


def extract_3d_pose_for_frame(df, player_id: str, frame_idx: int) -> np.ndarray:
    """
    Extracts 3D keypoints for a single frame.
    
    Returns:
        Array of shape (17, 3) with x, y, z coordinates.
    """
    prefix = f"player_{player_id}"
    pose = np.zeros((17, 3))
    
    for i, kp in enumerate(COCO17_KEYPOINTS):
        x_col = f"{prefix}_{kp}_3d_x"
        y_col = f"{prefix}_{kp}_3d_y"
        z_col = f"{prefix}_{kp}_3d_z"
        pose[i, 0] = df.iloc[frame_idx][x_col]
        pose[i, 1] = df.iloc[frame_idx][y_col]
        pose[i, 2] = df.iloc[frame_idx][z_col]
    
    return pose


def draw_2d_skeleton(ax, pose_2d: np.ndarray, title: str = "2D Pose"):
    """
    Draws the 2D skeleton on a matplotlib axis.
    """
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    for bone_idx, (start, end) in enumerate(COCO17_SKELETON):
        if pose_2d[start, 0] != 0 and pose_2d[end, 0] != 0:
            ax.plot(
                [pose_2d[start, 0], pose_2d[end, 0]],
                [pose_2d[start, 1], pose_2d[end, 1]],
                color=get_bone_color(bone_idx),
                linewidth=2
            )
    
    valid_points = pose_2d[pose_2d[:, 0] != 0]
    if len(valid_points) > 0:
        ax.scatter(valid_points[:, 0], valid_points[:, 1], c='white', edgecolors='black', s=30, zorder=5)
    
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')


def draw_3d_skeleton(ax, pose_3d: np.ndarray, title: str = "3D Pose", elev: int = 15, azim: int = 70):
    """
    Draws the 3D skeleton on a matplotlib 3D axis.
    """
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    
    for bone_idx, (start, end) in enumerate(COCO17_SKELETON):
        ax.plot3D(
            [pose_3d[start, 0], pose_3d[end, 0]],
            [pose_3d[start, 2], pose_3d[end, 2]],
            [-pose_3d[start, 1], -pose_3d[end, 1]],
            color=get_bone_color(bone_idx),
            linewidth=2
        )
    
    ax.scatter3D(pose_3d[:, 0], pose_3d[:, 2], -pose_3d[:, 1], c='white', edgecolors='black', s=30, zorder=5)
    
    max_range = np.max(np.abs(pose_3d)) * 1.2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')


def visualize_pose_comparison(
    df,
    player_id: str,
    frame_idx: int,
    save_path: Optional[str] = None
) -> None:
    """
    Visualizes a single frame comparing 2D input pose with inferred 3D pose.
    
    Args:
        df: DataFrame containing both 2D and 3D pose data.
        player_id: Player identifier (e.g., 'low_drive').
        frame_idx: Frame index to visualize.
        save_path: Optional path to save the figure.
    """
    pose_2d = extract_2d_pose_for_frame(df, player_id, frame_idx)
    pose_3d = extract_3d_pose_for_frame(df, player_id, frame_idx)
    
    fig = plt.figure(figsize=(14, 6))
    
    ax1 = fig.add_subplot(1, 2, 1)
    draw_2d_skeleton(ax1, pose_2d, title=f"2D Pose - Frame {frame_idx}")
    
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    draw_3d_skeleton(ax2, pose_3d, title=f"3D Pose - Frame {frame_idx}")
    
    plt.suptitle(f"Player: {player_id}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_sequence(
    df,
    player_id: str,
    start_frame: int,
    num_frames: int = 5,
    output_dir: Optional[str] = None
) -> None:
    """
    Visualizes a sequence of frames showing 2D and 3D pose comparison.
    Creates a grid with 2D poses on top row and 3D poses on bottom row.
    
    Args:
        df: DataFrame containing both 2D and 3D pose data.
        player_id: Player identifier (e.g., 'low_drive').
        start_frame: Starting frame index.
        num_frames: Number of frames to visualize.
        output_dir: Optional directory to save the figure.
    """
    end_frame = min(start_frame + num_frames, len(df))
    actual_frames = end_frame - start_frame
    
    fig = plt.figure(figsize=(4 * actual_frames, 8))
    
    for i, frame_idx in enumerate(range(start_frame, end_frame)):
        pose_2d = extract_2d_pose_for_frame(df, player_id, frame_idx)
        pose_3d = extract_3d_pose_for_frame(df, player_id, frame_idx)
        
        ax_2d = fig.add_subplot(2, actual_frames, i + 1)
        draw_2d_skeleton(ax_2d, pose_2d, title=f"Frame {frame_idx}")
        
        ax_3d = fig.add_subplot(2, actual_frames, actual_frames + i + 1, projection='3d')
        draw_3d_skeleton(ax_3d, pose_3d, title="")
    
    plt.suptitle(f"Player: {player_id} - Frames {start_frame} to {end_frame - 1}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{player_id}_frames_{start_frame}_{end_frame - 1}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
        plt.close()
    else:
        plt.show()


def visualize_sequence_animated(
    df,
    player_id: str,
    start_frame: int,
    num_frames: int = 30,
    output_path: Optional[str] = None,
    fps: int = 10
) -> None:
    """
    Creates an animated visualization of the pose sequence.
    
    Args:
        df: DataFrame containing both 2D and 3D pose data.
        player_id: Player identifier (e.g., 'low_drive').
        start_frame: Starting frame index.
        num_frames: Number of frames to animate.
        output_path: Optional path to save the animation (as .gif or .mp4).
        fps: Frames per second for the animation.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    
    end_frame = min(start_frame + num_frames, len(df))
    frame_indices = list(range(start_frame, end_frame))
    
    fig = plt.figure(figsize=(14, 6))
    ax_2d = fig.add_subplot(1, 2, 1)
    ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
    
    def update(frame_num):
        ax_2d.clear()
        ax_3d.clear()
        
        frame_idx = frame_indices[frame_num]
        pose_2d = extract_2d_pose_for_frame(df, player_id, frame_idx)
        pose_3d = extract_3d_pose_for_frame(df, player_id, frame_idx)
        
        draw_2d_skeleton(ax_2d, pose_2d, title=f"2D Pose - Frame {frame_idx}")
        draw_3d_skeleton(ax_3d, pose_3d, title=f"3D Pose - Frame {frame_idx}")
        
        return ax_2d, ax_3d
    
    anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=1000 // fps, blit=False)
    
    plt.suptitle(f"Player: {player_id}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        if output_path.endswith('.gif'):
            writer = PillowWriter(fps=fps)
            anim.save(output_path, writer=writer)
            print(f"Saved animation to {output_path}")
        else:
            anim.save(output_path, fps=fps)
            print(f"Saved animation to {output_path}")
        plt.close()
    else:
        plt.show()


def generate_diagnostic_report(
    df,
    player_id: str,
    sample_frames: int = 5,
    output_dir: str = "diagnostics"
) -> None:
    """
    Generates a diagnostic report with visualizations for a player.
    Samples frames evenly across the dataset to check inference quality.
    
    Args:
        df: DataFrame containing both 2D and 3D pose data.
        player_id: Player identifier.
        sample_frames: Number of frames to sample for visualization.
        output_dir: Directory to save diagnostic images.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    total_frames = len(df)
    step = max(1, total_frames // sample_frames)
    sampled_indices = list(range(0, total_frames, step))[:sample_frames]
    
    print(f"Generating diagnostic report for player '{player_id}'...")
    print(f"Total frames: {total_frames}, sampling {len(sampled_indices)} frames")
    
    for frame_idx in sampled_indices:
        save_path = os.path.join(output_dir, f"{player_id}_frame_{frame_idx}.png")
        visualize_pose_comparison(df, player_id, frame_idx, save_path=save_path)
        print(f"  Saved frame {frame_idx}")
    
    print(f"Diagnostic report saved to '{output_dir}/'")
