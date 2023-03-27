from . import pop, bias, lift
from . import ials, eals
from . import torch_emf, torch_imf, torch_lmf, torch_lmf_samp, torch_bpr
from . import torch_tag

algorithms = {
    'POP': pop,
    'BIAS': bias,
    'LIFT': lift,
    'E-MF': eals,
    'I-MF': ials,
    'TorchEMF': torch_emf,
    'UserIMF': torch_imf,
    'UserLMF': torch_lmf,
    'SampLMF': torch_lmf_samp,
    'BPR': torch_bpr,
    'TagEMF': torch_tag,
}