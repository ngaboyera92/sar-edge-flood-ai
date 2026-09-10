from src.models.unet import UNet


def build_model(cfg):
    a=cfg['architecture']
    out_channels=3 if a['output']=='3_logits_N_P_F' else 1
    return UNet(in_channels=len(a['input_channels']),out_channels=out_channels,base=a['base'])
