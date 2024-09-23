def load_model(model_name,mode):
    assert model_name in ['VGG8','ResNet18'], model_name
    assert mode in ['WAGE', 'WAGEV2','DynamicFixedPoint','FloatingPoint','LSQ'], mode

    if model_name == 'VGG8':
        if mode == 'WAGE':
            from Accuracy.src.Network.VGG8.WAGE.network import vgg8_load
            model = vgg8_load()
            return model
        if mode == 'WAGEV2':
            from Accuracy.src.Network.VGG8.WAGE_v2.network import vgg8_load
            model = vgg8_load()
            return model
        if mode == 'DynamicFixedPoint':
            from Accuracy.src.Network.VGG8.DynamicFixedPoint.network import vgg8_load
            model = vgg8_load()
            return model
        if mode == 'FloatingPoint':
            from Accuracy.src.Network.VGG8.FloatingPoint.network import vgg8_load
            model = vgg8_load()
            return model
        if mode == 'LSQ':
            from Accuracy.src.Network.VGG8.LSQ.network import vgg8_load
            model = vgg8_load()
            return model

    if model_name == 'ResNet18':
        if mode == 'WAGE':
            from Accuracy.src.Network.ResNet18.WAGE.network import resnet18
            model = resnet18()
            return model
        if mode == 'WAGEV2':
            from Accuracy.src.Network.ResNet18.WAGE_v2.network import resnet18
            model = resnet18()
            return model
        if mode == 'DynamicFixedPoint':
            from Accuracy.src.Network.ResNet18.DynamicFixedPoint.network import resnet18
            model = resnet18()
            return model
        if mode == 'FloatingPoint':
            from Accuracy.src.Network.ResNet18.FloatingPoint.network import resnet18
            model = resnet18()
            return model
        if mode == 'LSQ':
            from Accuracy.src.Network.ResNet18.LSQ.network import resnet18
            model = resnet18()
            return model