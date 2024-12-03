from huggingface_hub import snapshot_download

repo_id="eloialonso/iris"
local_dir="pretrained_iris"

print(f"Downloading {repo_id} to {local_dir}")
snapshot_download(repo_id=repo_id, local_dir=local_dir)
print("Downloaded all checkpoints!")


repo_id="eloialonso/diamond"
local_dir="pretrained_diamond"

print(f"Downloading {repo_id} to {local_dir}")
snapshot_download(repo_id=repo_id, local_dir=local_dir)
print("Downloaded all checkpoints!")