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
Tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#num_list = ["##0", "##1", "##2", "##3", "##4", "##5", "##6", "##7", "##8", "##9"]
Tokenizer.add_special_tokens({'additional_special_tokens': ["[scinotexp]", "[DOT]"]})
#Tokenizer.add_special_tokens({'additional_special_tokens': num_list})


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
        class_dict = {0: [],  # 'brand name'
                      1: [],  # 'material'
                      2: [],  # 'color'
                      3: [],  # 'category'
                      4: [],
                      5: []}
        gt_dict = {0: [],  # 'brand name'
                   1: [],  # 'material'
                   2: [],  # 'color'
                   3: [],  # 'category'
                   4: [],
                   5: []}
        # have_answer_list = outputs['have_answer_idx'].cpu().tolist()
        no_answer_loss = loss_function_na_loss(outputs['no_answer_output'], inputs['answer_label'])
        # print(outputs['contain_valid_value'])
        if outputs['contain_valid_value'] == 1:
            NA_T, NA_F, T, F, y_true, y_pred, _ = compute_metrics(inputs, outputs, 0, 0, 0, 0, [], [], [], [], [],
                                                                  class_dict, gt_dict)
            self.training_metric_dict['Accuracy'].append(T / (T + F))
            self.training_metric_dict['NA_Accuracy'].append(NA_T / (NA_T + NA_F))
            dmlm_loss = self.dmlm_loss(outputs['bert_gt_output'],
                                       outputs['contextual_prediction_output'])
            # print(pred_end_idx.max())
            # print(pred_end_idx.min())
            # qa_loss = outputs['contextual_output_whole'].loss
            begin_loss = loss_function_begin(outputs['start_logit'],
                                             outputs['begin_label_ori'])
            # print(pred_end_idx.max())
            # print(pred_end_idx.min())
            end_loss = loss_function_end(outputs['end_logit'], outputs['end_label_ori'])
            qa_loss = (begin_loss + end_loss) / 2
            total_loss = qa_loss + alpha * dmlm_loss + beta * no_answer_loss
            return (total_loss, {'pred_begin_idx': outputs['pred_begin_idx'],
                                 'pred_end_idx': outputs['pred_end_idx']}) if return_outputs else total_loss
        else:
            print('Whole batch null')
            return (no_answer_loss, {'pred_begin_idx': -1,
                                     'pred_end_idx': -1}) if return_outputs else no_answer_loss


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


def process_bad_case(gt_begin_idx, gt_end_idx, pred_begin_idx, pred_end_idx, input_ids, input_ids_label):
    bad_case_dict = {}
    bad_case_dict['gt_idx'] = [gt_begin_idx, gt_end_idx]
    bad_case_dict['pred_idx'] = [pred_begin_idx, pred_end_idx]
    bad_case_dict['text'] = Tokenizer.decode(input_ids)
    bad_case_dict['label'] = Tokenizer.decode(input_ids_label)
    return bad_case_dict


def compute_metrics(inputs, outputs, NA_T, NA_F, T, F, y_true: list, y_pred: list, prec, rec, f1, classfi_dict: dict,
                    gt_dict: dict):
    bad_case_list = []
    class_label = inputs['class_label'].cpu().tolist()
    input_ids = inputs['input_ids'].cpu().tolist()
    input_ids_label = inputs['input_ids_label'].cpu().tolist()
    y_true += class_label
    temp_dict = {}
    # have_answer_list = outputs['have_answer_idx']
    gt_begin_idx = outputs['begin_label_ori'].cpu().tolist()
    gt_end_idx = outputs['end_label_ori'].cpu().tolist()
    pred_begin_idx = outputs['pred_begin_idx'].cpu().tolist()
    pred_end_idx = outputs['pred_end_idx'].cpu().tolist()
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
                    prec.append(1.0)
                    rec.append(1.0)
                    f1.append(1.0)
                    y_pred.append(class_label[i])
                    '''
                    classfi_dict[int(class_label[i])].append(1)
                    gt_dict[int(class_label[i])].append(1)
                    for k0 in range(6):
                        if k0 != int(class_label[i]):
                            classfi_dict[k0].append(0)
                            gt_dict[k0].append(0)
                    '''
                    T += 1
                else:
                    y_pred.append(11957)
                    pred_span = set(
                        range(int(min(pred_begin_idx[i], pred_end_idx[i])),
                              int(max(pred_begin_idx[i], pred_end_idx[i])) + 1))
                    gt_span = set(
                        range(int(min(gt_begin_idx[i], gt_end_idx[i])),
                              int(max(gt_begin_idx[i], gt_end_idx[i])) + 1))
                    common = pred_span.intersection(gt_span)
                    p = len(common) / len(pred_span)
                    # print(pred_span)
                    # print(pred_begin_idx[i])
                    # print(pred_end_idx[i])
                    r = len(common) / len(gt_span)
                    prec.append(p)
                    rec.append(r)
                    f1.append((p + r) / 2)

                    '''
                    gt_dict[int(class_label[i])].append(1)
                    for k1 in range(6):
                        classfi_dict[k1].append(0)
                        if k1 != int(class_label[i]):
                            gt_dict[k1].append(0)
                    '''
                    bad_case_list.append(
                        process_bad_case(gt_begin_idx[i], gt_end_idx[i], pred_begin_idx[i], pred_end_idx[i],
                                         input_ids[i],
                                         input_ids_label[i]))

                    F += 1
            else:
                y_pred.append(0)
                prec.append(1.0)
                rec.append(1.0)
                f1.append(1.0)
                T += 1
        else:
            y_pred.append(0)
            prec.append(0.0)
            rec.append(0.0)
            f1.append(0.0)
            bad_case_list.append(
                process_bad_case(gt_begin_idx[i], gt_end_idx[i], pred_begin_idx[i], pred_end_idx[i],
                                 input_ids[i],
                                 input_ids_label[i]))
            F += 1
    return NA_T, NA_F, T, F, y_true, y_pred, bad_case_list


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
        weight_decay=training_config['weight_decay'],
        # warmup_steps=500,
        # evaluation_strategy="steps",  # 这里设置每100个batch做一次评估，也可以为“epoch”，也就是每个epoch进行一次
        # logging_strategy="steps",
        # save_strategy='steps',
        save_steps=5000,
        save_total_limit=1,
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


def start_test(model, test_dataset, training_config):
    T, F = 0, 0
    NA_T, NA_F = 0, 0
    bad_case_list_total = []
    class_dict = {0: [],  # 'brand name'
                  1: [],  # 'material'
                  2: [],  # 'color'
                  3: [],  # 'category'
                  4: [],
                  5: []}
    gt_dict = {0: [],  # 'brand name'
               1: [],  # 'material'
               2: [],  # 'color'
               3: [],  # 'category'
               4: [],
               5: []}

    dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    y_true, y_pred = [], []
    prec, rec, f1 = [], [], []
    for i, batch in enumerate(tqdm.tqdm(dataloader)):
        outputs = model(batch, 'cuda')
        NA_T, NA_F, T, F, y_true, y_pred, bad_case_list = compute_metrics(batch, outputs, NA_T, NA_F, T, F, y_true,
                                                                          y_pred, prec, rec, f1, class_dict, gt_dict)
        bad_case_list_total += bad_case_list
    print('New metric: Precision: {}, Recall: {}, F1: {}'.format(prec, rec, f1))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    print('Accuracy: {}, No Answer Accuracy: {}, Precision: {}, Recall: {}, F1: {}'.format(T / (T + F),
                                                                                           NA_T / (NA_T + NA_F),
                                                                                           precision, recall, f1))

    '''
    precision_bn, recall_bn, f1_bn, _ = precision_recall_fscore_support(class_dict[0], gt_dict[0])
    print('Brand name: \n Precision: {}, Recall: {}, F1: {}'.format(precision_bn, recall_bn, f1_bn))

    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(class_dict[1], gt_dict[1])
    print('Material: \n Precision: {}, Recall: {}, F1: {}'.format(precision_m, recall_m, f1_m))

    precision_c, recall_c, f1_c, _ = precision_recall_fscore_support(class_dict[2], gt_dict[2])
    print('Color: \n Precision: {}, Recall: {}, F1: {}'.format(precision_c, recall_c, f1_c))

    precision_ca, recall_ca, f1_ca, _ = precision_recall_fscore_support(class_dict[3], gt_dict[3])
    print('Category: \n Precision: {}, Recall: {}, F1: {}'.format(precision_ca, recall_ca, f1_ca))
    '''
    with open(training_config['model_output_dir'] + '/bad_case.json', 'w') as file:
        file.write(json.dumps(bad_case_list_total, indent=4))


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
    torch.save(test_dataset, training_config['test_dataset'])
    print('Training begin')
    model = AVEQA(model_name=training_config['model_name']).to(device)
    start_train(train_dataset, model, training_config)
    torch.save(model.bert_model_contextual.state_dict(), training_config['model_output_dir'] + '/bert_state_dict')
    model.eval()
    start_test(model, test_dataset, training_config)
    print('Using model: {}, {}'.format(training_config['model_name'], training_config['model_output_dir']))
