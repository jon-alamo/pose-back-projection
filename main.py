import bppose
import pandas as pd
import sys

def main():
    dataset_path = 'datasets/bcn-finals-2022-fem-pose-and-shot-data.csv'
    output_filename = "output_3d.csv"
    
    print(f"Loading dataset from {dataset_path}...")
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"File not found: {dataset_path}")
        sys.exit(1)

    players = ['low_drive', 'low_left', 'up_drive', 'up_left']
    
    print("Running back-projection...")
    try:
        df_3d = bppose.back_projection_from_coco17(df, players)
    except Exception as e:
        print(f"An error occurred: {e}")
        # Print full trace for debugging if needed, but keeping it simple as per spec style
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"Saving results to {output_filename}...")
    df_3d.to_csv(output_filename, index=False)
    print("Done.")

if __name__ == '__main__':
    main()
