import os
import subprocess

ADAPTERS = ["yelp", "sst2", "squad", "coqa", "tatoeba", "kde4"]

EVAL_DATASETS = ["kde4", "tatoeba"]

BASE_ADAPTER_DIR ="<MODEL_STORAGE>/fine-tuning-project/old_fine_tuned_model/"
SUBMIT_SCRIPT = SUBMIT_SCRIPT = "<PROJECT_ROOT>/src/Fine_tune/Machine_translation/mt.sh"
LOG_DIR = "<MODEL_STORAGE>/fine-tuning-project/mt/logs"

def main():
    for adj in ADAPTERS:
        for data in EVAL_DATASETS:
            if adj == data:
                print(f"Skip {adj} on {data}")
                continue
    
            adapter_path = os.path.join(BASE_ADAPTER_DIR, f"llama2-7b-{adj}-qlora")
            job_name = f"Eval_{adj}_on_{data}"
            log_file = os.path.join(LOG_DIR, f"{job_name}_%j.out")
    
            cmd = [
                'sbatch',
                '-p', 'gpu-a-lowsmall,gpu-a100-lowbig,gpu-h100,gpu-l40s,gpu-v100',
                '--gres', 'gpu:1',
                '-J', job_name,
                '-o', log_file,
                '-t', '1-00:00:00',
                SUBMIT_SCRIPT,
                '--task', data,     
                '--adapter', adapter_path
            ]
            subprocess.run(cmd)


if __name__ == "__main__":
    main()