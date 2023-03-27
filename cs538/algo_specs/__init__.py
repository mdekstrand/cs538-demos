from . import pop, bias, lift
from . import ials
from . import torch_imf, torch_lmf, torch_lmf_samp, torch_bpr

algorithms = {
    'POP': pop,
    'BIAS': bias,
    'LIFT': lift,
    'E-MF': None,
    'I-MF': ials,
    'TorchEMF': None,
    'UserIMF': torch_imf,
    'UserLMF': torch_lmf,
    'SampLMF': torch_lmf_samp,
    'BPR': torch_bpr,
}