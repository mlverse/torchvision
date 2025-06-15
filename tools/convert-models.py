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
  'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
  'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
  'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
  'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
  'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
  'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
  'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
  "mnasnet0_5": "https://download.pytorch.org/models/mnasnet0.5_top1_67.823-3ffadce67e.pth",
  "mnasnet1_0": "https://download.pytorch.org/models/mnasnet1.0_top1_73.512-f206786ef8.pth",
  'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
  'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
  'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
  'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
  'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
  'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
  'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
  'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
  'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
  'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
  'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
  'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
  'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
  'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
  'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
  'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
  'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
  'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
  'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
  'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
  'r3d_18': 'https://download.pytorch.org/models/r3d_18-b3b3357e.pth',
  'mc3_18': 'https://download.pytorch.org/models/mc3_18-a90a0ba3.pth',
  'r2plus1d_18': 'https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth',
  'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth',
  'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth',
  'deeplabv3_resnet50_coco': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth',
  'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
  'casia-webface': 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt',
  'vggface2': 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt'
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
    "v2/" + fpath
  )
