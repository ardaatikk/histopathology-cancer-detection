import torchvision.transforms as transforms

IMAGE_SIZE = (224, 224)

RGB_MEAN = (
    0.61495719,
    0.42157306,
    0.64444248,
)

RGB_STD = (
    0.21410407,
    0.22494683,
    0.18994613,
)


def get_preprocessing_transform(
    training=False,
    mean=None,
    std=None,
):
    if mean is None:
        mean = RGB_MEAN

    if std is None:
        std = RGB_STD

    transform_list = [
        transforms.Resize(IMAGE_SIZE),
    ]

    if training:
        transform_list.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=90),
            ]
        )

    transform_list.append(
        transforms.Normalize(
            mean=mean,
            std=std,
        )
    )

    return transforms.Compose(transform_list)