import os
import subprocess
import sys
from pathlib import Path
from datasets import load_dataset
import pandas as pd

# Configuration
TEXTCOMPLEXITY_DIR = "D:/textcomplexity"
TXTCOMPLEXITY_EXE = "D:/Software/anaconda3/envs/textcomplexity/Scripts/txtcomplexity"
NUM_SAMPLES = 7700
OUTPUT_DIR = "D:\\textcomplexity\\yelp_test_output"


def determine_window_size(text):
    word_count = len(text.split())

    if word_count <= 2:
        return 1
    elif word_count < 5:
        return 2
    elif word_count < 10:
        return 3
    elif word_count < 30:
        return 5
    elif word_count < 50:
        return 8
    elif word_count < 100:
        return 10
    elif word_count < 200:
        return 15
    else:
        return 20


# Force UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'


def prepare_data():
    """Prepare Yelp data -- save text files only, not labels."""
    print("=== Step 1: Preparing Yelp data ===")

    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # Load dataset
    print("Loading Yelp dataset...")
    dataset = load_dataset('yelp_polarity')['test']

    # Save text files
    for i in range(475, 575):
        sample = dataset[i]
        text = sample['text'].replace('\n', ' ').strip()

        # Save text
        text_file = Path(OUTPUT_DIR) / f"text_{i}.txt"
        text_file.write_text(text + '\n', encoding='utf-8')

        if (i + 1) % 1000 == 0:
            print(f"Prepared {i + 1} samples...")

    print(f"Prepared {min(NUM_SAMPLES, len(dataset))} samples")
    return min(NUM_SAMPLES, len(dataset))


def process_stanza():
    """Process text files with Stanza to generate CoNLL-U files."""
    print("\n=== Step 2A: Processing with Stanza ===")

    # Switch to textcomplexity directory
    original_dir = os.getcwd()
    os.chdir(TEXTCOMPLEXITY_DIR)

    success_count = 0
    fail_count = 0

    # Process each file
    for i in range(475, 575):
        output_dir = Path(original_dir) / OUTPUT_DIR
        text_file = Path(original_dir) / OUTPUT_DIR / f"text_{i}.txt"
        conllu_file = output_dir

        if not text_file.exists():
            continue

        print(f"Processing sample {i} with Stanza... ", end='', flush=True)

        # Run Stanza
        stanza_cmd = f'python ./utils/run_stanza.py --language en -o "{conllu_file}" "{text_file.absolute()}"'
        stanza_result = os.system(stanza_cmd)

        # Check generated CoNLL-U file
        expected_conllu = Path(original_dir) / OUTPUT_DIR / f"text_{i}.txt.conllu"

        if stanza_result != 0 or not expected_conllu.exists() or expected_conllu.stat().st_size == 0:
            print("FAILED")
            fail_count += 1
        else:
            print("SUCCESS")
            success_count += 1

    # Restore original directory
    os.chdir(original_dir)

    print(f"\nStanza processing complete: {success_count} success, {fail_count} failed")
    return success_count, fail_count


def process_textcomplexity():
    print("\n=== Step 2B: Processing with textcomplexity ===")

    original_dir = os.getcwd()
    os.chdir(TEXTCOMPLEXITY_DIR)

    results = []
    header = None
    success_count = 0
    fail_count = 0
    dataset = load_dataset('fancyzhx/yelp_polarity')['test']
    for i in range(NUM_SAMPLES):
        # text_file = Path(original_dir) / OUTPUT_DIR / f"text_{i}.txt"
        conllu_file = Path(original_dir) / OUTPUT_DIR / "input_files" / f"{i}.conllu"
        tsv_file = Path(original_dir) / OUTPUT_DIR / f"{i}.tsv"

        if not conllu_file.exists():
            continue

        print(f"Processing sample {i} with textcomplexity... ", end='', flush=True)

        window_size = 50  # default
        # text = str(dataset['context'][i]) + str(dataset['question'][i])
        text = str(dataset['text'][i])
        window_size = determine_window_size(text)

        preset = "lexical_core"

        txt_cmd = [
            sys.executable,
            TXTCOMPLEXITY_EXE,
            "--preset", preset,
            "--lang", "en",
            "--window-size", str(window_size),
            "-i", "conllu",
            "-o", "tsv",
            str(conllu_file.absolute())
        ]

        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'

        result = subprocess.run(
            txt_cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            env=env
        )

        # Debug info
        if result.returncode != 0:
            print(f"FAILED - Command error: {result.stderr}")
            fail_count += 1
            continue

        # Save results
        tsv_file.write_text(result.stdout, encoding='utf-8')

        if not tsv_file.exists() or tsv_file.stat().st_size == 0:
            print(f"FAILED - No output. stdout: {result.stdout[:200]}... stderr: {result.stderr}")
            fail_count += 1
            continue

        # Read results
        try:
            lines = tsv_file.read_text(encoding='utf-8').strip().split('\n')
            if len(lines) > 1:
                if header is None:
                    header = lines[0]
                data = lines[1]
                results.append(f"{i}\t{data}")
                success_count += 1
                print("SUCCESS")
            else:
                print("FAILED (no data)")
                fail_count += 1
        except Exception as e:
            print(f"FAILED ({e})")
            fail_count += 1

    # Restore original directory
    os.chdir(original_dir)

    print(f"\nTextcomplexity processing complete: {success_count} success, {fail_count} failed")
    return results, header


def analyze_results(results, header):
    """Analyze results."""
    print("\n=== Step 3: Analysis ===")

    if not results:
        print("No results to analyze!")
        return

    # Save merged results
    results_file = Path(OUTPUT_DIR) / "all_results.tsv"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"sample_idx\t{header}\n")
        for r in results:
            f.write(r + "\n")

    print(f"Results saved to: {results_file}")


def main():
    """Main entry point."""
    # print("Yelp Text Complexity Analysis - Separated Processing")
    # print("=" * 50)

    # Prepare data
    # num_samples = prepare_data()

    # Step 1: Stanza processing
    # stanza_success, stanza_fail = process_stanza()

    # Step 2: textcomplexity processing
    results, header = process_textcomplexity()

    # Analyze results
    analyze_results(results, header)

    print("\n=== Complete! ===")


if __name__ == "__main__":
    main()