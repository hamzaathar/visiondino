import torchvision.transforms as transforms
from PIL import Image


class RandomGaussianBlur(transforms.RandomApply):
    def __init__(self, p=0.5, kernel_size=5, sigma=(0.1, 2)):
        gaussian_blur = transforms.GaussianBlur(
            kernel_size=kernel_size, sigma=sigma)
        super(RandomGaussianBlur, self).__init__([gaussian_blur], p=p)


class RandomColorJitter(transforms.RandomApply):
    def __init__(self, p=0.5, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1):
        color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
        super(RandomColorJitter, self).__init__([color_jitter], p=p)


class DataAugmentation:
    def __init__(self,
                 global_crops_scale=(0.4, 1),
                 local_crops_scale=(0.05, 0.4),
                 n_local_crops=8,
                 size=224
                 ):
        self.n_local_crops = n_local_crops

        flip_and_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            RandomColorJitter(p=0.5),
            transforms.RandomGrayscale(p=0.2)
        ])

        normalize = transforms.Compose([
            transforms.ToTensor(),
            # Parameters from the paper
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.global_1 = transforms.Compose([
            transforms.RandomResizedCrop(
                size,
                scale=global_crops_scale,
                interpolation=Image.BICUBIC
            ),
            flip_and_jitter,
            # always apply, p=1
            RandomGaussianBlur(1.0),
            normalize
        ])

        self.global_2 = transforms.Compose([
            transforms.RandomResizedCrop(
                size,
                scale=global_crops_scale,
                interpolation=Image.BICUBIC
            ),
            flip_and_jitter,
            RandomGaussianBlur(1.0),
            transforms.RandomSolarize(170, p=0.2),
            normalize
        ])

        self.local = transforms.Compose([
            transforms.RandomResizedCrop(
                size,
                scale=local_crops_scale,
                interpolation=Image.BICUBIC
            ),
            flip_and_jitter,
            RandomGaussianBlur(0.5),
            normalize
        ])

    def __call__(self, img):
        crops = []
        crops.append(self.global_1(img))
        crops.append(self.global_2(img))
        for _ in range(self.n_local_crops):
            crops.append(self.local(img))
        return crops
