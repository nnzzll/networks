# networks
Pytorch implementation of benchmark models for image segmentation

- Models
    - [x] UNet
    - [x] UNet++
    - [ ] UNet3+
    - [x] UNet3D
    - [x] VNet

- Usage

        python setup.py develop

- Sample

    ```py
    import torch
    from networks import UNet

    model = UNet(1,1)
    inputs = torch.randn((1,1,512,512))
    output = model(inputs)
    print(output.shape)
    ```