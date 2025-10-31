# Torch Interlacer

PyTorch implementation of the Interlacer package for joint frequency- and image-space learning in Fourier imaging tasks.

## Installation

```bash
pip install -e .
```

## Usage

```python
import torch
from torch_interlacer.models import get_interlacer_residual_model

model = get_interlacer_residual_model(
    input_size=(2, 256, 256),
    nonlinearity='3-piece',
    kernel_size=9,
    num_features=32,
    num_convs=1,
    num_layers=10
)

input_data = torch.randn(1, 2, 256, 256)  # (batch_size, 2, height, width)
output = model(input_data)
```
