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
  'vggface2': 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt',
  'efficientnet_b0': 'https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth',
  'efficientnet_b1': 'https://download.pytorch.org/models/efficientnet_b1-c27df63c.pth', # IMAGENET1K_V2
  'efficientnet_b2': 'https://download.pytorch.org/models/efficientnet_b2_rwightman-bcdf34b7.pth',
  'efficientnet_b3': 'https://download.pytorch.org/models/efficientnet_b3_rwightman-cf984f9c.pth',
  'efficientnet_b4': 'https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth',
  'efficientnet_b5': 'https://download.pytorch.org/models/efficientnet_b5_lukemelas-1a07897c.pth',
  'efficientnet_b6': 'https://download.pytorch.org/models/efficientnet_b6_lukemelas-24a108a5.pth',
  'efficientnet_b7': 'https://download.pytorch.org/models/efficientnet_b7_lukemelas-c5b4e57e.pth',
  'efficientnet_v2_s': 'https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth',
  'efficientnet_v2_m': 'https://download.pytorch.org/models/efficientnet_v2_m-dc08266a.pth',
  'efficientnet_v2_l': 'https://download.pytorch.org/models/efficientnet_v2_l-59c71312.pth',
  'vit_b_16': 'https://download.pytorch.org/models/vit_b_16-c867db91.pth',
  'vit_b_32': 'https://download.pytorch.org/models/vit_b_32-d86f8d99.pth',
  'vit_l_16': 'https://download.pytorch.org/models/vit_l_16-852ce7e3.pth',
  'vit_l_32': 'https://download.pytorch.org/models/vit_l_32-c7638314.pth',
  'vit_h_14': 'https://download.pytorch.org/models/vit_h_14_swag-80465313.pth',
  'convnext_tiny': 'https://download.pytorch.org/models/convnext_tiny-983f1562.pth',
  'convnext_small': 'https://download.pytorch.org/models/convnext_small-0c510722.pth',
  'convnext_base': 'https://download.pytorch.org/models/convnext_base-6075fbad.pth',
  'convnext_large': 'https://download.pytorch.org/models/convnext_large-ea097f82.pth',
  'maskrcnn_resnet50': 'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
  'maskrcnn_resnet50_v2': 'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth',
  'mobilenet_v3_large': 'https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth', #IMAGENET1K_V2
  'mobilenet_v3_small': 'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth',
  'maxvit': 'https://download.pytorch.org/models/maxvit_t-bc5ab103.pth',
  'fcos_resnet50': 'https://download.pytorch.org/models/fcos_resnet50_fpn_coco-99b0c9b7.pth', 
  'retinanet_resnet50_v2': 'https://download.pytorch.org/models/retinanet_resnet50_fpn_v2_coco-5905b1c5.pth', # RetinaNet_ResNet50_FPN_V2_Weights
  'ssd300_vgg16': 'https://download.pytorch.org/models/ssd300_vgg16_coco-b556d3b4.pth',
  'ssdlite320_mobilenet_v3': 'https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth',
  'fasterrcnn_resnet50': 'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
  'fasterrcnn_resnet50_v2': 'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth',
  'fasterrcnn_mobilenet_v3_large': 'https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth',
  'fasterrcnn_mobilenet_v3_large_320': 'https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth',
  'yolo_v8_l': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l.pt',
  'yolo_v8_l_seg': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-seg.pt',
  'yolo_v8_m': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt',
  'yolo_v8_m_seg': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-seg.pt',
  'yolo_v8_n': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt',
  'yolo_v8_s': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt',
  'yolo_v8_s_seg': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-seg.pt',
  'yolo_v8_x': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt',
  'yolo_v11_l': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt',
  'yolo_v11_m': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt',
  'yolo_v11_n': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt',
  'yolo_v11_s': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt',
  'yolo_v11_s_cls': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-cls.pt',
  'yolo_v11_s_obb': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-obb.pt',
  'yolo_v11_s_pose': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt',
  'yolo_v11_s_seg': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt',
  'yolo_v11_x': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt',
  'yolo_v12_l': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12l.pt',
  'yolo_v12_m': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12m.pt',
  'yolo_v12_n': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt',
  'yolo_v12_s': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12s.pt',
  'yolo_v12_x': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12x.pt',
  }

os.makedirs("models", exist_ok=True)
# yolo specifics 
os.makedirs("runs", exist_ok=True)
settings.update({"runs_dir": "runs/", "weights_dir": "models/", "sync": False})

for name, url in models.items():
  fpath = "models/" + name + ".pth"

  if blob_exist("torch-pretrained-models", f"models/vision/v2/{fpath}"):
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
