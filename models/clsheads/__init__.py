from .ur import URClsHead
from .rtfm import RTFMClsHead

def get_clsHead(model_type):
    if model_type == 'ur':
        return URClsHead
    elif model_type == 'rtfm':
        return RTFMClsHead
    else:
        raise RuntimeError(f'Unknown model type: {model_type}')