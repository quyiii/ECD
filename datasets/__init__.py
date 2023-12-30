from .ucf import UCFVideo, data
from .xd import XDVideo

def get_dataset(datasetname):
    if 'ucf' in datasetname:
        return UCFVideo
    elif 'xd' in datasetname:
        return XDVideo
    else:
        raise RuntimeError(f'Unknown Dataset: {datasetname}')