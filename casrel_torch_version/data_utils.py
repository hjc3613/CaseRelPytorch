import json
from transformers import (DistilBertConfig, DistilBertModel, DistilBertTokenizer)
from transformers import BertTokenizer, AlbertModel
from casrel_torch_version.config_util import Conf
CONF = Conf().conf_data

relation2id_path = CONF['relation2id']
def getRelationIDMap():
    with open(relation2id_path) as f:
        id2relation, relation2id = json.load(f)
    return id2relation, relation2id

def findIndex(sentence:list, word:list):
    word_len = len(word)
    index = -1 # default
    for i in range(0, len(sentence)-word_len+1):
        if sentence[i:i+word_len]==word:
            index = i
            break
    return index

def getBertModel():
    # tokenizer = DistilBertTokenizer.from_pretrained(CONF['BERT_MODEL_PATH'])
    # model = DistilBertModel.from_pretrained(CONF['BERT_MODEL_PATH'])

    tokenizer = BertTokenizer.from_pretrained(CONF['BERT_MODEL_PATH'])
    model = AlbertModel.from_pretrained(CONF['BERT_MODEL_PATH'])
    return model, tokenizer


if __name__ == '__main__':
    sentence = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
    word = [ 'k']
    print(findIndex(sentence, word))