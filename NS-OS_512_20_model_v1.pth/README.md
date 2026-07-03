---
library_name: segmentation-models-pytorch
license: mit
pipeline_tag: image-segmentation
tags:
- model_hub_mixin
- pytorch_model_hub_mixin
- segmentation-models-pytorch
- semantic-segmentation
- pytorch
languages:
- python
---
# UPerNet Model Card

Table of Contents:
- [Load trained model](#load-trained-model)
- [Model init parameters](#model-init-parameters)
- [Model metrics](#model-metrics)
- [Dataset](#dataset)

## Load trained model
```python
import segmentation_models_pytorch as smp

model = smp.from_pretrained("<save-directory-or-this-repo>")
```

## Model init parameters
```python
model_init_params = {
    "encoder_name": "mit_b3",
    "encoder_depth": 5,
    "encoder_weights": "imagenet",
    "decoder_channels": 256,
    "decoder_use_norm": "batchnorm",
    "in_channels": 3,
    "classes": 2,
    "activation": None,
    "upsampling": 4,
    "aux_params": None
}
```

## Model metrics
```json
{
    "epoch": 16,
    "iou": 0.7997635006904602,
    "dice": 0.8752717971801758,
    "loss": 0.01892707563404526,
    "loss_fnc": "wce",
    "transform": "sar_transform_512_triple",
    "pretrained weights": "imagenet"
}
```

## Dataset
Dataset name: 512_20_north_sea_all

## More Information
- Library: https://github.com/qubvel/segmentation_models.pytorch
- Docs: https://smp.readthedocs.io/en/latest/

This model has been pushed to the Hub using the [PytorchModelHubMixin](https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin)