import pandas as pd

# Basic conversion
def parquet_to_csv(parquet_file, csv_file):
    """
    Convert a Parquet file to CSV
    
    Args:
        parquet_file: Path to input .parquet file
        csv_file: Path to output .csv file
    """
    df = pd.read_parquet(parquet_file)
    df.to_csv(csv_file, index=False)
    print(f"Converted {parquet_file} to {csv_file}")

# Example usage
if __name__ == "__main__":

    split = ["train", "validation", "test"]

    for split_name in split:
        input_parquet = f"data/{split_name}-00000-of-00001.parquet"
        output_csv = f"data/{split_name}.csv"
        parquet_to_csv(input_parquet, output_csv)