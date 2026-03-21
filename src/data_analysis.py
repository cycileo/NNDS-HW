import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt

def analyze_dataset(file_path):
    """
    Analyzes the Rotten Tomatoes dataset.
    
    Args:
        file_path (str): Path to the CSV file.
    """
    print(f"Loading dataset from: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Assuming columns are 'review_content' and 'review_type' based on user prompt,
    # but let's check actual column names first in case they differ (e.g., 'review_content' vs 'review_desc')
    print("\n--- Dataset Columns ---")
    print(list(df.columns))
    
    # We will try to automatically identify the text and label columns
    text_col = 'review_content' if 'review_content' in df.columns else None
    label_col = 'review_type' if 'review_type' in df.columns else None
    
    if not text_col and 'review_desc' in df.columns: text_col = 'review_desc'
    if not label_col and 'review_score' in df.columns: label_col = 'review_score' # just examples
    
    print(f"\nUsing Text Column: {text_col}")
    print(f"Using Label Column: {label_col}")

    if text_col not in df.columns or label_col not in df.columns:
        print(f"Error: Could not find required columns in {list(df.columns)}")
        print("Please specify the correct column names for 'review_content' and 'review_type'.")
        return

    print("\n--- Basic Statistics ---")
    total_reviews = len(df)
    print(f"Total reviews: {total_reviews}")

    # Label distribution
    print("\n--- Review Type Distribution ---")
    label_counts = df[label_col].value_counts(dropna=False)
    for index, val in label_counts.items():
        print(f"{index}: {val} ({val/total_reviews*100:.2f}%)")

    # Missing values
    print("\n--- Missing Values ---")
    missing_text = df[text_col].isna().sum()
    missing_label = df[label_col].isna().sum()
    print(f"Missing text (review_content): {missing_text} ({missing_text/total_reviews*100:.2f}%)")
    print(f"Missing labels (review_type): {missing_label} ({missing_label/total_reviews*100:.2f}%)")

    # Filter out missing text for length analysis
    df_clean = df.dropna(subset=[text_col])
    
    print("\n--- Text Length Analysis (in characters) ---")
    # Calculate lengths
    lengths = df_clean[text_col].astype(str).apply(len)
    
    print(f"Minimum length: {lengths.min()}")
    print(f"Maximum length: {lengths.max()}")
    print(f"Average length: {lengths.mean():.2f}")
    
    print("\n--- Text Length Analysis (in approximate words - split by space) ---")
    word_counts = df_clean[text_col].astype(str).apply(lambda x: len(x.split()))
    print(f"Minimum words: {word_counts.min()}")
    print(f"Maximum words: {word_counts.max()}")
    print(f"Average words: {word_counts.mean():.2f}")
    
    print("\n--- Sample Reviews ---")
    print("Fresh:")
    try:
        fresh_sample = df_clean[df_clean[label_col].astype(str).str.lower().str.contains('fresh', na=False)][text_col].iloc[0]
        print(f"  {fresh_sample[:150]}...")
    except:
        print("  No 'fresh' reviews found or not matching string format.")
        
    print("Rotten:")
    try:
        rotten_sample = df_clean[df_clean[label_col].astype(str).str.lower().str.contains('rotten', na=False)][text_col].iloc[0]
        print(f"  {rotten_sample[:150]}...")
    except:
        print("  No 'rotten' reviews found or not matching string format.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Rotten Tomatoes dataset")
    parser.add_argument("--data_path", type=str, default="../data/rotten_tomatoes_critic_reviews.csv", help="Path to the dataset CSV file")
    
    args = parser.parse_args()
    
    analyze_dataset(args.data_path)
