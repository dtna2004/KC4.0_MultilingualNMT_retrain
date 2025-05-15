from models.default import MockModel
from models.transformer import Transformer
from models.mamba import Mamba

AvailableModels = {
    "MockModel": MockModel, 
    "Transformer": Transformer,
    "Mamba": Mamba
}
