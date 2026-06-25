import os

os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
import geomstats.backend as gs

import torch
torch.set_default_dtype(torch.float32)  # geomstats sets float64 as default on import
