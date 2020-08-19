from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
import torch
from casrel_torch_version import data_utils
from casrel_torch_version.config_util import Conf

CONF = Conf().conf_data
id2relation, relation2id = data_utils.getRelationIDMap()

class SubjectModel(Module):
    def __init__(self):
        super(SubjectModel, self).__init__()
        # 从本地加载预训练的BERT模型
        self.encoder = data_utils.getBertModel()[0]
        self.sub_head_linear = nn.Linear(768, 1)
        self.sub_tail_linear = nn.Linear(768, 1)
        pass

    def forward(self, token_id, mask=None, sigmoid=False):
        sentence_features,  = self.encoder(input_ids=token_id, attention_mask=mask)
        sub_head_pred = self.sub_head_linear(sentence_features)
        sub_tail_pred = self.sub_tail_linear(sentence_features)
        if sigmoid:
            sub_head_pred = F.sigmoid(sub_head_pred)
            sub_tail_pred = F.sigmoid(sub_tail_pred)
        return sentence_features, sub_head_pred, sub_tail_pred

class ObjectModel(nn.Module):
    def __init__(self):
        super(ObjectModel, self).__init__()
        self.relationObjHeadLinear = nn.Linear(768, len(relation2id))
        self.relationObjTailLinear = nn.Linear(768, len(relation2id))

    def gather_nd(self, tensor:torch.Tensor, coordinates:torch.Tensor):
        result = []
        for dim0, dim1 in coordinates:
            result.append(tensor[dim0, dim1])
        return torch.stack(result)
        pass

    def forward(self, sentence_feautre, sub_head, sub_tail, sigmoid=False):
        ##################获取subject所在位置的特征向量，首尾求平均#######################
        idx = torch.arange(0, sub_head.shape[0], dtype=torch.int32, device=sentence_feautre.device).unsqueeze(1)
        head_coordinates = torch.cat((idx, sub_head), -1)
        tail_coordinates = torch.cat((idx, sub_tail), -1)
        head_features = self.gather_nd(sentence_feautre, head_coordinates)
        tail_features = self.gather_nd(sentence_feautre, tail_coordinates)
        sub_features = (head_features+tail_features)/2
        ################将subject的特征向量与句子特征相加################################
        sub_features = sub_features.unsqueeze(1)
        sentence_feautre = sentence_feautre + sub_features
        #预测relation和object，句子的每个位置都输出一个num_relation维的向量，并且经过sigmoid，表示每个位置在relation[i]中可能是obj开头或结尾的可能性
        obj_head_pred = self.relationObjHeadLinear(sentence_feautre)
        obj_tail_pred = self.relationObjTailLinear(sentence_feautre)
        if sigmoid:
            obj_head_pred = F.sigmoid(obj_head_pred)
            obj_tail_pred = F.sigmoid(obj_tail_pred)
        return obj_head_pred, obj_tail_pred

class CaseModel(Module):
    def __init__(self):
        super(CaseModel, self).__init__()
        self.subModel = SubjectModel()
        self.objModel = ObjectModel()

    def forward(self,
                batch_tokens,
                batch_mask=None,
                batch_sub_head=None,
                batch_sub_tail=None,
                ):
        sentence_feature, sub_head_pred, sub_tail_pred = self.subModel(batch_tokens, batch_mask)
        obj_head_pred, obj_tail_pred = self.objModel(sentence_feature, batch_sub_head, batch_sub_tail)

        return sub_head_pred, sub_tail_pred, obj_head_pred, obj_tail_pred