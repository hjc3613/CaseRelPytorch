import json
import random
import torch
from torch.utils.data import DataLoader, Dataset
from casrel_torch_version import data_utils
from casrel_torch_version.config_util import Conf

CONF = Conf().conf_data
MAX_SENTENCE_LEN = CONF['MAX_SENTENCE_LEN']

id2relation, relation2id = data_utils.getRelationIDMap()
tokenizer = data_utils.getBertModel()[1]

class Record:
    def __init__(self, src, token_ids, mask, sub_head, sub_tail, sub_head_arr, sub_tail_arr, obj_head_arr,
                 obj_tail_arr, triple_list=None):
        self.src = src
        self.token_ids = token_ids
        self.mask = mask
        self.sub_head = sub_head
        self.sub_tail = sub_tail
        self.sub_head_arr = sub_head_arr
        self.sub_tail_arr = sub_tail_arr
        self.obj_head_arr = obj_head_arr
        self.obj_tail_arr = obj_tail_arr
        self.triple_list = triple_list


class CasRelDataset(Dataset):
    def __init__(self, path_or_json, train=True):
        if isinstance(path_or_json, str):
            with open(path_or_json) as f:
                self.data = json.load(f)
        else:
            self.data = path_or_json
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        one_record = self.data[i]
        one_record = CasRelDataset.createTensor(one_record, train=self.train)
        return one_record

    @staticmethod
    def createTensor(item, train=True):
        text = item['text']
        triple_list = item['triple_list']
        triple_list = [tuple(i) for i in triple_list]

        # 对text分词，编码
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(tokens)
        src_lst = tokenizer.decode(token_ids)
        token_ids = token_ids[:MAX_SENTENCE_LEN]
        ############################在预测阶段，只需提供token_ids即可，封装成Record返回##########################
        if not train:
            record = Record(src=src_lst,
                            token_ids=torch.tensor(token_ids, dtype=torch.long),
                            mask=None,
                            sub_head=None,
                            sub_tail=None,
                            sub_head_arr=None,
                            sub_tail_arr=None,
                            obj_head_arr=None,
                            obj_tail_arr=None,
                            triple_list=None)
            return record
        ####################### train 和 eval阶段，需要提供以下信息################################################
        # 创建subject -> object, relation 的映射
        sub2ObjRel = {}
        for sub, relation, obj in triple_list:
            sub_ids = tokenizer.encode(sub)[1:-1]  # 开头和结尾分别时[CLS]、[SEP]
            obj_ids = tokenizer.encode(obj)[1:-1]
            sub_head_index = data_utils.findIndex(token_ids, sub_ids)
            obj_head_index = data_utils.findIndex(token_ids, obj_ids)
            if sub_head_index != -1 and obj_head_index != -1:
                sub = (sub_head_index, sub_head_index + len(sub_ids) - 1)
                if sub not in sub2ObjRel:
                    sub2ObjRel[sub] = []
                sub2ObjRel[sub].append((obj_head_index, obj_head_index + len(obj_ids) - 1, relation2id[relation]))
        # 根据subject -> object, relation映射关系，创建训练数据集
        token_ids = token_ids + [0] * (MAX_SENTENCE_LEN - len(token_ids))
        mask = [int(i > 0) for i in token_ids]
        # 创建 subject 的head array, tail array，sub_head、sub_tail位置设为 1，其余为 0
        sub_head_arr, sub_tail_arr = torch.zeros(MAX_SENTENCE_LEN), torch.zeros(MAX_SENTENCE_LEN)
        for sub_head_index, sub_tail_index in sub2ObjRel.keys():
            sub_head_arr[sub_head_index] = 1
            sub_tail_arr[sub_tail_index] = 1
        # 随机选择一个subject的索引，创建该subject对应的所有object 的 head_array, tail_array， 如果subJect不存在，索引设为(0，0)
        # 这么做是为了在模型中方便计算，而且第0个索引是[CLS]，影响不大。
        obj_head_arr, obj_tail_arr = torch.zeros((MAX_SENTENCE_LEN, len(relation2id))), torch.zeros(
            (MAX_SENTENCE_LEN, len(relation2id)))
        sub_head, sub_tail = random.choice(list(sub2ObjRel.keys())) if sub2ObjRel else (0, 0)
        for (obj_head_index, obj_tail_index, relation_id) in sub2ObjRel.get((sub_head, sub_tail), []):
            obj_head_arr[obj_head_index][relation_id] = 1
            obj_tail_arr[obj_tail_index][relation_id] = 1
        record = Record(src=src_lst,
                        token_ids=torch.tensor(token_ids, dtype=torch.long),
                        mask=torch.tensor(mask, dtype=torch.int32),
                        sub_head=torch.tensor([sub_head], dtype=torch.int32),
                        sub_tail=torch.tensor([sub_tail], dtype=torch.int32),
                        sub_head_arr=sub_head_arr,
                        sub_tail_arr=sub_tail_arr,
                        obj_head_arr=obj_head_arr,
                        obj_tail_arr=obj_tail_arr,
                        triple_list=triple_list)
        return record


def collate_casrel(x):
    batch_tokens = torch.stack([i.token_ids for i in x])
    batch_mask = torch.stack([i.mask for i in x])
    batch_sub_head = torch.stack([i.sub_head for i in x])
    batch_sub_tail = torch.stack([i.sub_tail for i in x])
    batch_sub_head_arr = torch.stack([i.sub_head_arr for i in x])
    batch_sub_tail_arr = torch.stack([i.sub_tail_arr for i in x])
    batch_obj_head_arr = torch.stack([i.obj_head_arr for i in x])
    batch_obj_tail_arr = torch.stack([i.obj_tail_arr for i in x])

    return (batch_tokens,
            batch_mask,
            batch_sub_head,
            batch_sub_tail,
            batch_sub_head_arr,
            batch_sub_tail_arr,
            batch_obj_head_arr,
            batch_obj_tail_arr)


if __name__ == '__main__':
    # _, relation2id = getRelationIDMap()
    # print(relation2id)
    dataset = CasRelDataset(CONF['TRAIN_DATA_PATH'])
    # a = dataset[44]
    dataloader = DataLoader(dataset, batch_size=5, collate_fn=collate_casrel)
    for batch in dataloader:
        print(batch)
    pass
