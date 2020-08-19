from casrel_torch_version.casrel_model import CaseModel
from casrel_torch_version.load_data import CasRelDataset
from casrel_torch_version.predict import predict

def metric(model: CaseModel, eval_dataset: CasRelDataset, exact_match=False):
    correct_num, predict_num, gold_num = 1e-05,1e-05,1e-05
    for i in range(len(eval_dataset)):
        record = eval_dataset[i]
        pred_triples = set(predict(model, record))
        gold_triples = set(record.triple_list)
        correct_num += len(pred_triples & gold_triples)
        predict_num += len(pred_triples)
        gold_num += len(gold_triples)
    precision = correct_num/predict_num
    recall = correct_num/gold_num
    f1 = 2*precision*recall/(precision+recall)
    return precision, recall, f1
    pass
