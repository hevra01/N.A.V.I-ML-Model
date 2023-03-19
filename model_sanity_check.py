# here we will perform sanity check that the model at least outputs the correct shapes

from model import *
def model_sanity_test():
    num_classes = 20
    model = YOLOv3(num_classes=num_classes)
    img_size = 416
    # Based on chatgpt, the first dimension of the input tensor (2 in this case) represents the batch size.
    # The batch size is typically specified during the training phase when the data is loaded into the model.
    # The batch size is defined by the dataloader, which is used to load the data into the model during training.
    # The batch size is specified in the dataloader's constructor, where the batch_size parameter is passed.
    x = torch.randn((2, 3, img_size, img_size))
    # the forward method of the model will be called
    out = model(x)
    assert out[0].shape == (2, 3, img_size//32, img_size//32, 5 + num_classes)
    assert out[1].shape == (2, 3, img_size//16, img_size//16, 5 + num_classes)
    assert out[2].shape == (2, 3, img_size//8, img_size//8, 5 + num_classes)

model_sanity_test()