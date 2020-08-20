import torchvision
import torch
from google.cloud import storage
import os

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

models = {
  'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
}

os.mkdir("models/")

for name, url in models.items():
  m = torchvision.models.utils.load_state_dict_from_url(url, progress=False)
  converted = {}
  for nm, par in m.items():
    converted.update([(nm, par.clone())])
  fpath = "models/" + name + ".pth"
  torch.save(converted, fpath, _use_new_zipfile_serialization=True)
  upload_blob(
    "torchvision-models",
    fpath,
    "v1/" + fpath
  )
