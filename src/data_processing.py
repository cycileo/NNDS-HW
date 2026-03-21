import pandas as pd
import argparse
import os
import csv

def process_dataset(input_path, output_path, num_reasoning_tokens=15):
    """
    Processes the Rotten Tomatoes dataset into the target format:
    [SOS] [review text] [SOR] [RP] x 15 [S] [F/R] [EOS]
    
    Args:
        input_path (str): Path to the input CSV.
        output_path (str): Path to save the processed text file.
        num_reasoning_tokens (int): Number of placeholder tokens to insert.
    """
    print(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)
    
    initial_len = len(df)
    df = df.dropna(subset=['review_content', 'review_type'])
    print(f"Dropped {initial_len - len(df)} rows with missing content/type. Remaining: {len(df)}")
    
    df['review_type'] = df['review_type'].astype(str).str.strip().str.lower()
    
    # Calculate stats before formatting
    fresh_count = (df['review_type'] == 'fresh').sum()
    rotten_count = (df['review_type'] == 'rotten').sum()
    total = len(df)
    print(f"\nFresh: {fresh_count / total * 100:.2f}%")
    print(f"Rotten: {rotten_count / total * 100:.2f}%")
    
    # Define special tokens
    SOS_TOKEN = "[SOS]" # Start of Sequence
    EOS_TOKEN = "[EOS]" # End of Sequence
    SOR_TOKEN = "[SOR]" # Start of Reasoning
    RP_TOKEN  = "[RP]"  # Reasoning Placeholder
    S_TOKEN   = "[SEN]"   # Sentiment Indicator
    
    # We will map 'Fresh' and 'Rotten' directly to [FRE] and [ROT]
    def format_review(row):
        # Lowercase the review text
        text = str(row['review_content']).strip().lower()
                
        sentiment = "[FRE]" if row['review_type'] == 'fresh' else "[ROT]"
        
        reasoning_segment = " ".join([RP_TOKEN] * num_reasoning_tokens)
        
        # Format mapping: [SOS] text [SOR] [RP]*15 [SEN] sentiment [EOS]
        formatted = f"{SOS_TOKEN} {text} {SOR_TOKEN} {reasoning_segment} {S_TOKEN} {sentiment} {EOS_TOKEN}"
        return formatted

    print("\nFormatting reviews...")
    df['formatted_review'] = df.apply(format_review, axis=1)
    
    print("\n--- Length Analysis of Processed Text (in words) ---")
    word_counts = df['formatted_review'].apply(lambda x: len(x.split()))
    print(f"Minimum length: {word_counts.min()} words")
    print(f"Maximum length: {word_counts.max()} words")
    print(f"Average length: {word_counts.mean():.2f} words")
    
    # Save to a new CSV file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Store labels as boolean to save space and for easy stratification later
    df['is_fresh'] = df['review_type'] == 'fresh'
    out_df = df[['formatted_review', 'is_fresh']]
    
    print(f"\nSaving formatted dataset to: {output_path}")
    out_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
            
    print("Done!")
    
    print("\nSample formatted review:")
    print(out_df.iloc[0].to_dict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset into KV Cache format")
    parser.add_argument("--input", type=str, default="../data/rotten_tomatoes_critic_reviews.csv")
    parser.add_argument("--output", type=str, default="../data/processed_reviews.csv")
    parser.add_argument("--reasoning_tokens", type=int, default=15)
    
    args = parser.parse_args()
    process_dataset(args.input, args.output, args.reasoning_tokens)
