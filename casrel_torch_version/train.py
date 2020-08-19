import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(ROOT)
sys.path.append(ROOT)
from casrel_torch_version.casrel_model import CaseModel
from casrel_torch_version.config_util import Conf
from casrel_torch_version.load_data import CasRelDataset, collate_casrel
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.optim.adam import Adam
from tqdm import tqdm
from casrel_torch_version.metric import metric
import torch
from tensorboardX import SummaryWriter

import logging

CONF = Conf().conf_data

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
fileHandler = logging.FileHandler('train.log')
logger.addHandler(fileHandler)

Epochs = CONF['Epoch']


def train():
    tb_writer = SummaryWriter('tb_output')
    device = 'cuda:0' if CONF['GPU'] else 'cpu'
    model: nn.Module = CaseModel()
    # tb_writer.add_graph(model)
    model.train()
    
    train_dataset = CasRelDataset(path_or_json=CONF['TRAIN_DATA_PATH'])
    eval_dataset = CasRelDataset(path_or_json=CONF['EVAL_DATA_PATH'])
    dataloader = DataLoader(train_dataset,
                            batch_size=CONF['batch_size'],
                            shuffle=True,
                            collate_fn=collate_casrel)
    loss_func = BCEWithLogitsLoss()
    best_loss = 1e3
    optim = Adam(model.parameters(), lr=1e-5)
    global_steps = 0

    for epoch_num in range(Epochs):
        epoch_loss = 0.0
        model = model.to(device=device)
        for (batch_tokens,
             batch_mask,
             batch_sub_head,
             batch_sub_tail,
             batch_sub_head_arr,
             batch_sub_tail_arr,
             batch_obj_head_arr,
             batch_obj_tail_arr) in tqdm(dataloader, f'Epoch {epoch_num:3.0f}/{Epochs}', len(dataloader)):
            batch_tokens, batch_mask, batch_sub_head, batch_sub_tail, batch_sub_head_arr, batch_sub_tail_arr,batch_obj_head_arr, batch_obj_tail_arr = list(
                map(lambda x: x.to(device),
                    (batch_tokens, batch_mask, batch_sub_head, batch_sub_tail,batch_sub_head_arr, batch_sub_tail_arr,batch_obj_head_arr, batch_obj_tail_arr)
                    )
            )
            sub_head_pred, sub_tail_pred, obj_head_pred, obj_tail_pred = model(batch_tokens,
                                                                               batch_mask,
                                                                               batch_sub_head,
                                                                               batch_sub_tail)
            sub_head_loss = loss_func(sub_head_pred.squeeze(), batch_sub_head_arr)
            sub_tail_loss = loss_func(sub_tail_pred.squeeze(), batch_sub_tail_arr)
            obj_head_loss = loss_func(obj_head_pred, batch_obj_head_arr)
            obj_tail_loss = loss_func(obj_tail_pred, batch_obj_tail_arr)
            loss = sub_head_loss + sub_tail_loss + obj_head_loss + obj_tail_loss
            epoch_loss += loss
            logger.info(f'every batch loss: {loss}')
            global_steps += 1
            tb_writer.add_scalar('train_loss', loss, global_steps)
            optim.zero_grad()
            loss.backward()
            optim.step()
        # end one epoch

        p, r, f = metric(model.to('cpu'), eval_dataset)
        logger.info(f'epoch:{epoch_num + 1:3.0f}, precision: {p:5.4f}, recall: {r:5.4f}, f1-score: {f:5.4f}')
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_model = CONF['SAVE_MODEL']
            if not os.path.exists(os.path.dirname(save_model)):
                os.makedirs(os.path.dirname(save_model))
            torch.save(model.state_dict(), save_model)
    # end all epoch


if __name__ == '__main__':
    train()
