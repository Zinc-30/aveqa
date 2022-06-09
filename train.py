import os


def find_gpus(nums=6):
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >~/.tmp_free_gpus')
    # If there is no ~ in the path, return the path unchanged
    with open(os.path.expanduser('~/.tmp_free_gpus'), 'r') as lines_txt:
        frees = lines_txt.readlines()
        idx_freeMemory_pair = [(idx, int(x.split()[2]))
                               for idx, x in enumerate(frees)]
    idx_freeMemory_pair.sort(key=lambda my_tuple: my_tuple[1], reverse=True)
    usingGPUs = [str(idx_memory_pair[0])
                 for idx_memory_pair in idx_freeMemory_pair[:nums]]
    usingGPUs = ','.join(usingGPUs)
    print('using GPU idx: #', usingGPUs)
    return usingGPUs


os.environ['CUDA_VISIBLE_DEVICES'] = find_gpus(nums=1)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import argparse
import random
import numpy as np
import tqdm
import json
from torch import nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import BertTokenizer
from bert_model import AVEQA
from ae_pub import AEPub
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, \
    precision_recall_fscore_support
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str)
args = parser.parse_args()
device = args.device


class CustomTrainer(Trainer):
    def __init__(self, **kwargs):
        super(CustomTrainer, self).__init__(**kwargs)
        self.training_metric_dict = {'Accuracy': [],
                                     'NA_Accuracy': []}

    def dmlm_loss(self, bert_gt_output, contextual_prediction_output):
        bert_gt_sum = torch.sum(bert_gt_output, dim=-1)
        contextual_prediction_sum = torch.sum(contextual_prediction_output, dim=-1)
        bert_gt = torch.div(bert_gt_output.sqrt().t(), bert_gt_sum.sqrt()).t()
        contextual_prediction = torch.div(contextual_prediction_output.sqrt().t(), contextual_prediction_sum.sqrt()).t()
        loss = torch.sum(-1 * contextual_prediction * torch.log(bert_gt), dim=-1)
        return torch.mean(loss)

    def compute_loss(self, model, inputs, return_outputs=False, alpha=0.5, beta=0.5):
        # labels = inputs.get("labels")

        loss_function_na_loss = nn.CrossEntropyLoss()
        loss_function_begin = nn.CrossEntropyLoss()
        loss_function_end = nn.CrossEntropyLoss()
        # forward pass
        outputs = model(inputs, device)
        # have_answer_list = outputs['have_answer_idx'].cpu().tolist()
        no_answer_loss = loss_function_na_loss(outputs['no_answer_output'], inputs['answer_label'])
        NA_T, NA_F, T, F = compute_metrics(outputs, 0, 0, 0, 0)
        self.training_metric_dict['Accuracy'].append(T / (T + F))
        self.training_metric_dict['NA_Accuracy'].append(NA_T / (NA_T + NA_F))
        dmlm_loss = self.dmlm_loss(outputs['bert_gt_output'],
                                   outputs['contextual_prediction_output'])

        begin_loss = loss_function_begin(outputs['begin_output'],
                                         outputs['begin_label'])
        gt_end_idx = outputs['end_label'] - outputs['begin_label']
        # print(pred_end_idx.max())
        # print(pred_end_idx.min())
        end_loss = loss_function_end(outputs['end_output'], gt_end_idx)
        qa_loss = (begin_loss + end_loss) / 2
        total_loss = qa_loss + alpha * dmlm_loss + beta * no_answer_loss
        # print('got loss')

        return (total_loss, {'pred_begin_idx': outputs['pred_begin_idx'],
                             'pred_end_idx': outputs['pred_end_idx']}) if return_outputs else total_loss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_metrics_sample(pred):
    labels = pred.label_ids
    preds = pred.predictions
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def compute_metrics(outputs, NA_T, NA_F, T, F):
    temp_dict = {}
    # have_answer_list = outputs['have_answer_idx']
    gt_begin_idx = outputs['begin_label_ori'].cpu().tolist()
    gt_end_idx = outputs['end_label_ori'] - outputs['begin_label_ori']
    gt_end_idx = gt_end_idx.cpu().tolist()
    pred_begin_idx = torch.argmax(outputs['begin_output_ori'], dim=-1).cpu().tolist()
    pred_end_idx = torch.argmax(outputs['begin_output_ori'], dim=-1).cpu().tolist()
    gt_no_answer = outputs['answer_label'].cpu().tolist()
    pred_no_answer = torch.argmax(outputs['no_answer_output'], dim=-1).cpu().tolist()
    for j in range(len(gt_no_answer)):
        if gt_no_answer[j] == pred_no_answer[j]:
            temp_dict[j] = True
            NA_T += 1
        else:
            temp_dict[j] = False
            NA_F += 1
    for i in range(len(gt_begin_idx)):
        if temp_dict[i] is True:
            if int(pred_no_answer[i]) == 1:
                if gt_begin_idx[i] == pred_begin_idx[i] and gt_end_idx[i] == pred_end_idx[i]:
                    T += 1
                else:
                    F += 1
            else:
                T += 1
        else:
            F += 1
    return NA_T, NA_F, T, F


def generate_data(full_dataset, eval=False):
    eval_dataset = None
    if eval:
        train_size = int(0.7 * len(full_dataset))
        eval_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - eval_size - train_size
        torch.manual_seed(0)
        train_dataset, eval_dataset, test_dataset = torch.utils.data.random_split(full_dataset,
                                                                                  [train_size, eval_size, test_size])
    else:
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        torch.manual_seed(0)
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    return train_dataset, eval_dataset, test_dataset


<<<<<<< HEAD
base_dir = './aveqa_model_1e-6'


def start_train(train_set, model):
    training_args = TrainingArguments(
        output_dir=base_dir,  # 存储结果文件的目录
        overwrite_output_dir=True,
        max_steps=200000,
        # max_steps=2000,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=1e-6,
        # eval_steps=50,
        # load_best_model_at_end=True,
        # metric_for_best_model="f1",  # 最后载入最优模型的评判标准，这里选用precision最高的那个模型参数
        #weight_decay=0.0001,
        #warmup_steps=500,
=======
def start_train(train_set, model, training_config):
    training_args = TrainingArguments(
        output_dir=training_config['model_output_dir'],  # 存储结果文件的目录
        overwrite_output_dir=True,
        max_steps=training_config['max_steps'],
        # max_steps=2000,
        per_device_train_batch_size=training_config['batch_size'],
        per_device_eval_batch_size=training_config['batch_size'],
        learning_rate=training_config['learning_rate'],
        # eval_steps=50,
        # load_best_model_at_end=True,
        # metric_for_best_model="f1",  # 最后载入最优模型的评判标准，这里选用precision最高的那个模型参数
        # weight_decay=0.0001,
        # warmup_steps=500,
>>>>>>> 4551a957485020a79d2dfd08557f210b1409fff5
        # evaluation_strategy="steps",  # 这里设置每100个batch做一次评估，也可以为“epoch”，也就是每个epoch进行一次
        # logging_strategy="steps",
        # save_strategy='steps',
        save_total_limit=3,
        seed=training_config['seed'],
        logging_dir='./log',
        # label_names=['msk_index']
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        # eval_dataset=eval_set,
        # compute_metrics=compute_metrics,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # 早停Callback
    )
    trainer.train()
    with open(training_config['model_output_dir'] + '/training_metric.json', 'w') as file:
        file.write(json.dumps(trainer.training_metric_dict, indent=4))


def start_test(model, test_dataset):
    T, F = 0, 0
    NA_T, NA_F = 0, 0

    dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    for i, batch in enumerate(tqdm.tqdm(dataloader)):
        outputs = model(batch, device)
<<<<<<< HEAD
        temp_dict = {}
        # have_answer_list = outputs['have_answer_idx']
        gt_begin_idx = outputs['begin_label_ori'].cpu().tolist()
        gt_end_idx = outputs['end_label_ori'] - outputs['begin_label_ori']
        gt_end_idx = gt_end_idx.cpu().tolist()
        pred_begin_idx = torch.argmax(outputs['begin_output_ori'], dim=-1).cpu().tolist()
        pred_end_idx = torch.argmax(outputs['begin_output_ori'], dim=-1).cpu().tolist()
        gt_no_answer = batch['answer_label'].cpu().tolist()
        pred_no_answer = torch.argmax(outputs['no_answer_output'], dim=-1).cpu().tolist()
        for j in range(len(gt_no_answer)):
            if gt_no_answer[j] == pred_no_answer[j]:
                temp_dict[j] = True
                NA_T += 1
            else:
                temp_dict[j] = False
                NA_F += 1
        for i in range(len(gt_begin_idx)):
            if temp_dict[i] is True:
                if int(pred_no_answer[i]) == 1:
                    if gt_begin_idx[i] == pred_begin_idx[i] and gt_end_idx[i] == pred_end_idx[i]:
                        T += 1
                    else:
                        F += 1
                else:
                    T += 1
            else:
                F += 1
=======
        NA_T, NA_F, T, F = compute_metrics(outputs, NA_T, NA_F, T, F)
>>>>>>> 4551a957485020a79d2dfd08557f210b1409fff5

    print('Accuracy: {}, No Answer Accuracy: {}'.format(T / (T + F), NA_T / (NA_T + NA_F)))


if __name__ == '__main__':
    '''
    temp = {
        'dataset': './dataset/aePub',
        'model_output_dir': './aveqa_model_1e-6_64_updated_emb',
        'learning_rate': 1e-6,
        'max_steps': 200000,
        'batch_size': 32,
        'seed': 0
    }
    with open('./config.json', 'w') as file:
        file.write(json.dumps(temp, indent=4))
    '''
    # dataset_path = "./dataset/publish_data.txt"
    with open('./config.json', 'r') as file:
        training_config = json.load(file)
    setup_seed(training_config['seed'])
    dataset = torch.load(training_config['dataset'])
    # Tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # dataset = AEPub(dataset_path, Tokenizer)
    train_dataset, _, test_dataset = generate_data(dataset, False)
<<<<<<< HEAD
    if mode == 'train':
        model = AVEQA().to(device)
        start_train(train_dataset, model)
        torch.save(model.bert_model_contextual.state_dict(), base_dir + '/bert_state_dict')
        model.eval()
        start_test(model, test_dataset)
    else:
        base_dir = './aveqa_model_1e-6_64'
        model = AVEQA().to(device)
        model.load_state_dict(torch.load('./aveqa_model_1e-6_64/checkpoint-200000/pytorch_model.bin'))
        # model.bert_model_contextual.load_state_dict(torch.load(base_dir + '/bert_state_dict'))
        model.eval()
        start_test(model, test_dataset)
=======
    torch.save(test_dataset, training_config['test_dataset'])
    model = AVEQA().to(device)
    start_train(train_dataset, model, training_config)
    # torch.save(model.bert_model_contextual.state_dict(), training_config['model_output_dir'] + '/bert_state_dict')
    model.eval()
    start_test(model, test_dataset)
>>>>>>> 4551a957485020a79d2dfd08557f210b1409fff5
