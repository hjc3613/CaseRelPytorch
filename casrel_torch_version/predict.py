import torch
from casrel_torch_version.casrel_model import CaseModel
from casrel_torch_version.load_data import Record
from casrel_torch_version import data_utils

tokenizer = data_utils.getBertModel()[1]
id2relation, relation2id = data_utils.getRelationIDMap()

def predict(model: CaseModel, record: Record):
    # ##################预测阶段分两步##############################
    # 第一步：预测subject
    sentence_features, sub_head_pred, sub_tail_pred = model.subModel(record.token_ids.unsqueeze(0), sigmoid=True)
    sub_head_index, sub_tail_index = torch.where(sub_head_pred[0] > 0.5)[0], torch.where(sub_tail_pred[0] > 0.5)[0]
    subjects = []
    for head in sub_head_index:
        tail = sub_tail_index[sub_head_index >= head]
        if len(tail) > 0:
            tail = tail[0]
            subject = record.token_ids[head:tail + 1]
            subject = tokenizer.decode(subject)
            subjects.append((subject, head, tail))
    # 第二部：根据第一步预测出来的subject，来预测object和relation
    if subjects:
        triple_list = []
        batch_sentence_features = torch.stack([sentence_features] * len(subjects))
        sub_heads, sub_tails = torch.tensor([sub[1:] for sub in subjects]).T.view(2, -1, 1)
        obj_heads_pred, obj_tails_pred = model.objModel(batch_sentence_features, sub_heads, sub_tails, sigmoid=True)
        for i, subject in enumerate(subjects):
            # 第i个subject对应的object和relation
            ith_obj_heads, ith_obj_tails = torch.where(obj_heads_pred[i] > 0.5), torch.where(obj_tails_pred[i] > 0.5)
            sub = subject[0]
            sub = ''.join([i.lstrip("##") for i in sub])
            sub = ' '.join(sub.split('[unused1]'))
            for obj_head, head_rel_id in zip(*ith_obj_heads):
                for obj_tail, tail_rel_id in zip(*ith_obj_tails):
                    if obj_head < obj_tail and head_rel_id == tail_rel_id:
                        relation = id2relation[head_rel_id]
                        obj = record.token_ids[obj_head:obj_tail+1]
                        obj = tokenizer.decode(obj)
                        obj = ''.join([i.lstrip("##") for i in obj])
                        obj = ' '.join(obj.split('[unused1]'))
                        triple_list.append((sub, relation, obj))
                        break
        triple_list = list(set(triple_list))
        return triple_list
    else:
        return []

