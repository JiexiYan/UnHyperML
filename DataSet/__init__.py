from .CUB200 import CUB_200_2011
from .Car196 import Cars196
from .Stanford_Online_Products import Stanford_Online_Products
from .In_shop_clothes import InShopClothes
# from .transforms import *
import os 

__factory = {
    'cub': CUB_200_2011,
    'car': Cars196,
    'product': Stanford_Online_Products,
    'shop': InShopClothes,
}


def names():
    return sorted(__factory.keys())

def get_full_name(name):
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name].__name__

def create(name, root=None, root_c=None, part_rate=0, noise_rate=0, HC=True, *args, **kwargs):
    """
    Create a dataset instance.
    """
    if root is not None:
        root = os.path.join(root, get_full_name(name))
    
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root=root, root_c=root_c, part_rate=part_rate, noise_rate=noise_rate, HC=HC, *args, **kwargs)
