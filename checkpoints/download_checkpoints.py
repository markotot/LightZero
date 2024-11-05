from huggingface_hub import snapshot_download

repo_id="eloialonso/iris"
local_dir="pretrained_iris"


snapshot_download(repo_id=repo_id, local_dir=local_dir)
