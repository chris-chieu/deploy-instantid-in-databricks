from huggingface_hub import hf_hub_download


volume_path = "/Volumes/<catalog>/<schema>/<volume>"

hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/config.json", local_dir=f"{volume_path}/checkpoints")

hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="ControlNetModel/diffusion_pytorch_model.safetensors",
    local_dir="/tmp",
)

shutil.move('/tmp/ControlNetModel/diffusion_pytorch_model.safetensors', f"{volume_path}/checkpoints/ControlNetModel/diffusion_pytorch_model.safetensors")

hf_hub_download(repo_id="InstantX/InstantID", filename="ip-adapter.bin", local_dir="/tmp")

shutil.move('/tmp/ip-adapter.bin', f"{volume_path}/checkpoints/ip-adapter.bin")
