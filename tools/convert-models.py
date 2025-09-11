import torch
from torch.hub import load_state_dict_from_url # used to be in torchvision
import os
import boto3
from botocore.exceptions import ClientError
from ultralytics import settings


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    s3 = boto3.client('s3')

    s3.upload_file(
      source_file_name,
      bucket_name,
      destination_blob_name
    )

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

def blob_exist(bucket_name, blob_name):
    """Check if file already exists in s3 bucket"""
    s3 = boto3.client("s3")
    try:
        s3.head_object(Bucket=bucket_name, Key=blob_name)
        return True
    except s3.exceptions.NoSuchKey:
        return False
    except ClientError:
        return False

models = {
  'convnext_tiny_1k': 'https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth', # _224_1k
  'convnext_tiny_22k': 'https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth', # _224_22k
  'convnext_small_22k': 'https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth', # _224_22k
  'convnext_small_22k1k': 'https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_224.pth', # _224_22k_1k
  'convnext_base_1k': 'https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth', # _224_1k
  'convnext_base_22k': 'https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth', # _224_22k
  'convnext_large_1k': 'https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth', # _224_1k
  'convnext_large_22k': 'https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth', # _224_22k
  }

os.makedirs("models", exist_ok=True)
# yolo specifics 
os.makedirs("runs", exist_ok=True)
settings.update({"runs_dir": "runs/", "weights_dir": "models/", "sync": False})

for name, url in models.items():
  fpath = "models/" + name + ".pth"

  if blob_exist("torch-pretrained-models", f"models/vision/v3/{fpath}"):
    print(f"--- file {fpath} is already in the bucket. Bypassing conversion")

  else:
    # download from url, convert and upload the converted weights
    m = load_state_dict_from_url(url, progress=False)
    converted = {}
    
    # yolo models weights are embedded in a BaseModel per https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L309
    if name.startswith("yolo_"):
      m = m["model"].model.float().state_dict()
    
    for nm, par in m.items():
      converted.update([(nm, par.clone())])
    torch.save(converted, fpath, _use_new_zipfile_serialization=True)
    upload_blob(
      "torch-pretrained-models",
      fpath,
      "models/vision/v2/" + fpath
    )
    # free disk space
    os.remove(fpath)
