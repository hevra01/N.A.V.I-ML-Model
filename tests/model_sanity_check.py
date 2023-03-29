# here we will perform sanity check that the model at least outputs the correct shapes

from model import *
def model_sanity_test():
    num_classes = 20
    model = YOLOv3(num_classes=num_classes)
    img_size = 416

    # The first dimension of the input tensor (2 in this case) represents the batch size.
    # The batch size is typically specified during the training phase when the data is loaded into the model.
    # The batch size is defined by the dataloader, which is used to load the data into the model during training.
    # The batch size is specified in the dataloader's constructor, where the batch_size parameter is passed.
    x = torch.randn((2, 3, img_size, img_size))

    # the forward method of the model will be called
    out = model(x)

    # In YOLOv3 the grid sizes used are [13, 26, 52] for an image size of 416x416.
    # If you use another image size the first grid size will be the image size
    # divided by 32 and the others will be a multiple of two of the previous one.
    # the second dimension, "3" represents the number of anchors.
    # the forward function of the model, gives predictions in three different scales.
    # hence, the output has out[0], out[1], out[2]
    # it is 6 + num_classes because we added distance to model's prediction
    # 0th index is objectness, 1st to 4th is bounding box infor, 5th is dist
    assert out[0].shape == (2, 3, img_size//32, img_size//32, 6 + num_classes)
    assert out[1].shape == (2, 3, img_size//16, img_size//16, 6 + num_classes)
    assert out[2].shape == (2, 3, img_size//8, img_size//8, 6 + num_classes)

model_sanity_test()