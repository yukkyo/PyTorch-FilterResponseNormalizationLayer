import albumentations as albu
from albumentations.pytorch import ToTensorV2
from catalyst.data.augmentor import Augmentor


def pre_transforms(image_size=256, border_constant=0):
    # Convert the image to a square of size image_size x image_size
    # (keeping aspect ratio)
    result = [
        albu.LongestMaxSize(max_size=image_size),
        albu.PadIfNeeded(image_size, image_size, border_mode=border_constant)
    ]
    return result


def hard_transforms(border_reflect=2):
    result = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=border_reflect,
            p=0.5
        ),
        albu.IAAPerspective(scale=(0.02, 0.05), p=0.3),
        albu.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.3
        ),
        albu.RandomGamma(gamma_limit=(85, 115), p=0.3),
        albu.HueSaturationValue(p=0.3),
        albu.ImageCompression(quality_lower=80, p=0.5),
    ]

    return result


def post_transforms():
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    return [albu.Normalize(), ToTensorV2()]


def compose(transforms_to_compose):
    # combine all augmentations into one single pipeline
    result = albu.Compose([
        item for sublist in transforms_to_compose for item in sublist
    ])
    return result


def get_transform(phase, img_size):
    assert phase in {'train', 'valid'}, f'invalid phase: {phase}'
    if phase == 'train':
        transforms = compose([
            pre_transforms(image_size=img_size),
            hard_transforms(),
            post_transforms()
        ])
    else:
        transforms = compose([pre_transforms(), post_transforms()])

    data_transforms = Augmentor(
        dict_key="features",
        augment_fn=lambda x: transforms(image=x)["image"]
    )
    return data_transforms
