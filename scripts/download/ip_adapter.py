#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

from huggingface_hub import cached_download, hf_hub_url

os.makedirs("data/models/image_encoder", exist_ok=True)
for hub_file in [
    "models/image_encoder/config.json",
    "models/image_encoder/pytorch_model.bin",
    "models/ip-adapter-plus_sd15.bin",
    "models/ip-adapter_sd15.bin",
]:
    path = Path(hub_file)
    local_file = cached_download(hf_hub_url(repo_id="h94/IP-Adapter", subfolder=path.parent, filename=path.name))
    shutil.copy(local_file, "data" / path)
