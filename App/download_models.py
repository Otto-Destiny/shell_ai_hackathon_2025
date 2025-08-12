import os
from huggingface_hub import hf_hub_download, snapshot_download

# Target directory for models
target_dir = "Models"
os.makedirs(target_dir, exist_ok=True)

# Download specific files (Folds 1–5) from willieseun/Eagle-Team-TabPFN
print("Downloading fold models from willieseun/Eagle-Team-TabPFN...")
for i in range(1, 6):
    file_name = f"Fold_{i}_best_model.tabpfn_fit"
    model_path = hf_hub_download(
        repo_id="willieseun/Eagle-Team-TabPFN",
        filename=file_name,
        local_dir=target_dir
    )
    print(f"Downloaded: {model_path}")

# Download full snapshot from wayne-chi/Eagle_Team
print("\nDownloading snapshot from wayne-chi/Eagle_Team...")
snapshot_download(
    repo_id="wayne-chi/Eagle_Team",
    revision="main",  # Optional, default is "main"
    local_dir=target_dir,
)

print("\n✅ All models downloaded successfully to:", target_dir)

