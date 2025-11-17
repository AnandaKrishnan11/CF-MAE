from model import CF_MAE
from terratorch.registry import BACKBONE_REGISTRY
from transformers import AutoModel

def build_model(config,name='CF_MAE'):

    if name == 'CF_MAE':
        return CF_MAE.TemporalGeoMAE_ConvDecoder(config)
    
    elif name == 'prithvi': #PRITHVI (~100M TO 600M)
        return BACKBONE_REGISTRY.build("ibm-nasa-geospatial/Prithvi-EO-2.0-300M")
    
    elif name == 'clay': #CLAY (~632M)
        return AutoModel.from_pretrained("made-with-clay/Clay", dtype="auto")
    
    elif name == 'satmae': #SATMAE (~329M)
        return  AutoModel.from_pretrained("MVRL/satmae-vitlarge-multispec-pretrain", dtype="auto")
    
    else:
        raise NotImplementedError

