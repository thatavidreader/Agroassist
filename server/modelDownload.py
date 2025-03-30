from huggingface_hub import snapshot_download

# Download the model files to "models/plant-disease/"
path = snapshot_download(repo_id="linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification", cache_dir="models/plant-disease")

print("Model downloaded to:", path)
