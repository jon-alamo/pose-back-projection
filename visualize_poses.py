"""
Script to run pose inference and generate diagnostic visualizations.
Use this to verify that the 3D pose inference is working correctly.
"""

import bppose
import pandas as pd
import sys


def main():
    dataset_path = 'datasets/bcn-finals-2022-fem-pose-and-shot-data.csv'
    
    print(f"Loading dataset from {dataset_path}...")
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"File not found: {dataset_path}")
        sys.exit(1)

    players = ['low_drive', 'low_left', 'up_drive', 'up_left']
    
    print("Running back-projection...")
    df_3d = bppose.back_projection_from_coco17(df, players)
    
    print(f"\nDataset has {len(df_3d)} frames")
    
    # Generate diagnostic visualizations for each player
    for player in players:
        print(f"\n--- Generating diagnostics for {player} ---")
        bppose.generate_diagnostic_report(
            df_3d,
            player_id=player,
            sample_frames=5,
            output_dir=f"diagnostics/{player}"
        )
    
    # Generate sequence visualizations (5 consecutive frames)
    print("\n--- Generating sequence visualizations ---")
    for player in players:
        bppose.visualize_sequence(
            df_3d,
            player_id=player,
            start_frame=100,
            num_frames=5,
            output_dir="diagnostics/sequences"
        )
    
    print("\nDone! Check the 'diagnostics/' folder for visualizations.")


if __name__ == '__main__':
    main()
