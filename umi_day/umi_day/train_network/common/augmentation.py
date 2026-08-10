import torchvision
from omegaconf import DictConfig

class ImageAugmentation:
    def __init__(self, shape_meta, transforms: DictConfig):
        self.key_to_transform = {}

        rgb_keys = []
        for obs_key in shape_meta['obs']:
            if shape_meta['obs'][obs_key]['type'] == 'rgb':
                rgb_keys.append(obs_key)

                img_shape = shape_meta['obs'][obs_key]['shape'] # (C, H, W)
                img_size = img_shape[1]

                # compute the transforms for this RGB key
                self.transforms = []
                for transform in transforms:
                    if type(transform) == DictConfig:
                        if transform['type'] == 'RandomCrop':
                            ratio = transform.ratio
                            self.transforms.extend([
                                torchvision.transforms.RandomCrop(size=int(img_size * ratio)),
                                torchvision.transforms.Resize(size=img_size, antialias=True)
                            ])
                    else:
                        self.transforms.append(transform)

                self.key_to_transform[obs_key] = torchvision.transforms.Compose(self.transforms)
    
    def apply(self, batch):
        for key in batch:
            if key in self.key_to_transform:
                batch[key] = self.key_to_transform[key](batch[key])
        return batch
