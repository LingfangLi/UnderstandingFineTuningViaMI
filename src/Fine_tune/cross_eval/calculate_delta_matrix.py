import pandas as pd
import numpy as np
import os

# Configuration
INPUT_CSV = "llama2_matrix_results.csv"
OUTPUT_CSV = "llama2_delta_matrix.csv"
BASE_MODEL_ID = "Base_Model"

TASK_TYPES = {
    "absolute_diff": ["yelp", "sst2"],          # Sentiment
    "relative_imp": ["squad", "coqa", "kde4", "tatoeba"]  # QA, MT
}

PRIMARY_METRICS = {
    "yelp": "Accuracy",
    "sst2": "Accuracy",
    "squad": "F1",
    "coqa": "F1",
    "kde4": "BLEU",
    "tatoeba": "BLEU"
}

def calculate_delta(row, base_scores):
    """Calculate performance delta for a single model row relative to the base model."""
    delta_row = {}
    
    for task in row.index:
        ft_score = row[task]
        base_score = base_scores.get(task, 0)

        if pd.isna(ft_score) or pd.isna(base_score):
            delta_row[task] = np.nan
            continue

        task_lower = task.lower()

        # Sentiment: absolute difference (FT - Base) * 100
        if task_lower in TASK_TYPES["absolute_diff"]:
            delta = (ft_score - base_score) * 100

        # QA / Translation: relative improvement ((FT - Base) / Base) * 100
        elif task_lower in TASK_TYPES["relative_imp"]:
            if base_score == 0:
                delta = 0
            else:
                delta = ((ft_score - base_score) / base_score) * 100

        else:
            delta = np.nan
            
        delta_row[task] = delta
        
    return pd.Series(delta_row)

def main():
    print(f"Reading raw results from: {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        print(f"Error: File {INPUT_CSV} not found. Please run the evaluation script first.")
        return

    df = pd.read_csv(INPUT_CSV)
    
    # Merge task-specific metrics into a single score column
    def get_score(row):
        task = row["Eval_Task"]
        metric = PRIMARY_METRICS.get(task, "Accuracy") # Default fallback
        return row.get(metric, 0)

    df["Main_Score"] = df.apply(get_score, axis=1)

    raw_matrix = df.pivot(index="Model_Source", columns="Eval_Task", values="Main_Score")
    
    print("\n[Raw Score Matrix]")
    print(raw_matrix)

    if BASE_MODEL_ID not in raw_matrix.index:
        print(f"\nError: Base model '{BASE_MODEL_ID}' not found in the results!")
        print(f"Available models: {list(raw_matrix.index)}")
        return

    base_row = raw_matrix.loc[BASE_MODEL_ID]
    print(f"\n[Baseline Scores]\n{base_row.to_string()}")

    delta_matrix = raw_matrix.apply(lambda row: calculate_delta(row, base_row), axis=1)
    delta_matrix = delta_matrix.round(2)

    print("\n" + "="*50)
    print("FINAL DELTA MATRIX (Perf delta %)")
    print("="*50)
    print(delta_matrix)
    
    delta_matrix.to_csv(OUTPUT_CSV)
    print(f"\nSaved delta matrix to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()