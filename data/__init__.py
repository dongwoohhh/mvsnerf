from .llff import LLFFDataset
from .blender import BlenderDataset
from .dtu_ft import DTU_ft
from .dtu import MVSDatasetDTU
from .dataset_with_colmap import DatasetWithColmap

dataset_dict = {'dtu': MVSDatasetDTU,
                'llff':LLFFDataset,
                'blender': BlenderDataset,
                'dtu_ft': DTU_ft,
                'custom': DatasetWithColmap}