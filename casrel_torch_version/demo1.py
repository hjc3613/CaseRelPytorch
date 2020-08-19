import json
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(ROOT)
sys.path.append(ROOT)

import torch

from casrel_torch_version.config_util import Conf
from casrel_torch_version.casrel_model import CaseModel
from casrel_torch_version.load_data import CasRelDataset, collate_casrel
from casrel_torch_version.predict import predict


CONF = Conf().conf_data
model_path = CONF['SAVE_MODEL']

model = CaseModel()
model.load_state_dict(torch.load(CONF['SAVE_MODEL']))

with open(CONF['TEST_DATA_PATH'], encoding='utf8') as f:
    data_set = json.load(f)

result = []
for item in data_set:
    sentence = item['text']
    pred = predict(model=model, record=item)
    label = item['triple_list']
    result.append({'text':sentence, 'label':label, 'predict':pred})
with open('pred_result.json', mode='w', encoding='utf8') as f:
    json.dump(result, f, indent=4, ensure_ascii=False)
