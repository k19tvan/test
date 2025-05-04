import kagglehub

# Download latest version
path = kagglehub.dataset_download("flap1812/fisheye8k")

print("Path to dataset files:", path)