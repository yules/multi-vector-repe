import os
from datasets import load_dataset

def download_and_save_datasets(base_dir="./data"):
    """
    Downloads Hugging Face datasets and saves them locally to disk.
    """
    # Create the base data directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    print(f"Directory {base_dir} is ready.\n")

    # 1. Download the Malicious Baseline (walledai/HarmBench)
    print("Downloading walledai/HarmBench...")
    try:
        # HarmBench is small and fits easily in memory
        harmbench = load_dataset("walledai/HarmBench", "standard", trust_remote_code=True)
        harmbench_path = os.path.join(base_dir, "harmbench")
        
        # Save to local disk
        harmbench.save_to_disk(harmbench_path)
        print(f"✅ HarmBench successfully saved to: {harmbench_path}")
        
    except Exception as e:
        print(f"❌ Failed to download HarmBench: {e}")



    print("\nAll downloads complete! Your lab datasets are ready.")

if __name__ == "__main__":
    # Ensure you have 'datasets' installed: pip install datasets
    download_and_save_datasets()