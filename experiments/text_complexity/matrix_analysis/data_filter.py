import re

import datasets
import pandas as pd
import numpy as np
from datasets import load_dataset

def create_finetuning_subset():
    """
    Create a subset of Yelp data based on specified thresholds:
    - average token length <= 4
    - lexical density <= 0.5
    - rarity <= 0.25

    Create a subset of TATOEBA data based on specified thresholds:
    - average token length <= 3.7
    - lexical density <= 0.45
    - rarity <= 0.17
    """

    # Read the Yelp_score file
    print("Reading Yelp_score file...")
    try:
        df = pd.read_csv('D:\\textcomplexity\\twitter_output\\all_results.tsv', sep='\t')
        print(f"File loaded successfully! Total samples: {len(df)}")
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Define required columns
    required_columns = {
        'average_token_length': 'average token length (disjoint windows)',
        'lexical_density': 'lexical density',
        'rarity': 'rarity'
    }
    # Check if all required columns exist
    missing_columns = []
    for name, col in required_columns.items():
        if col not in df.columns:
            missing_columns.append(col)

    if missing_columns:
        raise ValueError(f"ERROR: Missing required columns: {missing_columns}")

    # Convert columns to numeric
    for col in required_columns.values():
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Apply threshold filters
    print("\nApplying filters:")
    print("-" * 60)
    print("average token length < 4")
    print("lexical density < 0.5")
    print("rarity < 0.25")

   # Keep samples meeting any one of these conditions
    print("\nFiltering samples...")

    # mask = (df['average token length (disjoint windows)'] <= 4.7) & \
    #        (df['lexical density'] <= 0.47) & \
    #        (df['rarity'] <= 0.3)
    mask = (df['average token length (disjoint windows)'] > 4) | \
           (df['lexical density'] > 0.4) | \
           (df['rarity'] > 0.25)


    subset_df = df[mask].copy()

    print(f"\nSamples meeting all conditions: {len(subset_df)}")

    # Check sample count and limit if necessary
    if len(subset_df) < 10000:
        print(f"WARNING: Only {len(subset_df)} samples found (less than 10k)")
    elif len(subset_df) > 20000:
        print(f"Found {len(subset_df)} samples, limiting to 20,000...")
        subset_df = subset_df.iloc[20000:21000]

    # Save the subset
    # subset_df.to_csv('yelp_finetuning_subset.csv', sep='\t', index=False)
    # print(f"\nSubset saved to: yelp_finetuning_subset.csv")

    if 'sample_idx' in subset_df.columns:
        sample_indices = subset_df['sample_idx'].tolist()
        print(f"Using sample_idx column")
        # Option 2: Extract from filename
    else:
        print("Extracting indices from filenames...")
        sample_indices = []
        for filename in subset_df['filename']:
            # Extract number from filename
            match = re.search(r'/(\d+)\.conllu', filename)
            if match:
                idx = int(match.group(1))
                sample_indices.append(idx)
            else:
                print(f"Warning: Could not extract index from {filename}")

    print(f"Extracted {len(sample_indices)} valid indices")

    # Sort and potentially limit indices
    sample_indices = sorted(sample_indices)
    # if len(sample_indices) > 20000:
    #     print(f"Limiting to first 20,000 samples...")
    #     sample_indices = sample_indices[:20000]
    # save the sample indices to a file
    with open('yelp_lexically_complex_test_indices.txt', 'w') as f:
        for idx in sample_indices:
            f.write(f"{idx}\n")
    return sample_indices
if __name__ == "__main__":
    sample_idx = create_finetuning_subset()
    with open('yelp_lexically_complex_test_indices.txt', 'r') as f:
        index_list = [int(line.strip()) for line in f]
    dataset = load_dataset('yelp_polarity')['test']
    dataset = dataset.select(index_list)
    dataset.to_csv('yelp_lexically_complex_test.csv', index=False)