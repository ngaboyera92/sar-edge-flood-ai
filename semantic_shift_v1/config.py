import json
from pathlib import Path

EXPECTED_CONDITIONS={'B0','A0','B3','A1','A2','A3','A4','A5','TEACHER'}


def load_config(path):
    cfg=json.loads(Path(path).read_text())
    validate_config(cfg)
    return cfg


def validate_config(c):
    assert c['status']=='FROZEN'
    assert c['implementation_binding']['required_interface_version']=='semantic_shift_v1'
    assert c['condition_id'] in EXPECTED_CONDITIONS
    pp=c['preprocessing']
    assert pp['band_order']==['VV','VH']
    assert pp['clipping']=='none'
    assert pp['double_log_forbidden'] is True
    n=pp['normalization']
    assert n['VV']['mean_db']==-10.136466953246543 and n['VV']['std_db']==5.013093346521575
    assert n['VH']['mean_db']==-17.314963656896122 and n['VH']['std_db']==6.202284981963544
    assert c['epochs']==40 and c['batch_size']==8 and c['optimizer']['lr']==0.0003
    assert c['precision']['amp'] is False and c['precision']['dtype']=='FP32'
    if c['condition_id']!='TEACHER':
        assert c['seed'] in range(5)
    return True
