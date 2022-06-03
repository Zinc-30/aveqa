import torch
import argparse
import random
import numpy as np
import tqdm
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


def compute_metrics(pred):
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


def start_train(train_set, model):
    training_args = TrainingArguments(
        output_dir='./aveqa_model',  # 存储结果文件的目录
        overwrite_output_dir=True,
        max_steps=200000,
        #max_steps=2000,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=1e-5,
        # eval_steps=50,
        # load_best_model_at_end=True,
        # metric_for_best_model="f1",  # 最后载入最优模型的评判标准，这里选用precision最高的那个模型参数
        weight_decay=0.01,
        warmup_steps=50,
        # evaluation_strategy="steps",  # 这里设置每100个batch做一次评估，也可以为“epoch”，也就是每个epoch进行一次
        # logging_strategy="steps",
        # save_strategy='steps',
        save_total_limit=3,
        seed=0,
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


def start_test(model, test_dataset):
    T, F = 0, 0
    NA_T, NA_F = 0, 0

    dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    for i, batch in enumerate(tqdm.tqdm(dataloader)):
        outputs = model(batch, device)
        temp_dict = {}
        # have_answer_list = outputs['have_answer_idx']
        gt_begin_idx = outputs['begin_label'].cpu().tolist()
        gt_end_idx = outputs['end_label'] - outputs['begin_label']
        gt_end_idx = gt_end_idx.cpu().tolist()
        pred_begin_idx = torch.argmax(outputs['begin_output'], dim=-1).cpu().tolist()
        pred_end_idx = torch.argmax(outputs['end_output'], dim=-1).cpu().tolist()
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
            if temp_dict[i]:
                if int(pred_no_answer[i]) == 1:
                    if gt_begin_idx[i] == pred_begin_idx[i] and gt_end_idx[i] == pred_end_idx[i]:
                        T += 1
                    else:
                        F += 1
                else:
                    T += 1
            else:
                F += 1

    print('Accuracy: {}, No Answer Accuracy: {}'.format(T / (T + F), NA_T / (NA_T + NA_F)))


if __name__ == '__main__':
    setup_seed(0)
    dataset_path = "./dataset/publish_data.txt"
    dataset = torch.load('./dataset/aePub')
    mode = 'train'
    # Tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # dataset = AEPub(dataset_path, Tokenizer)
    train_dataset, _, test_dataset = generate_data(dataset, False)
    if mode == 'train':
        model = AVEQA().to(device)
        start_train(train_dataset, model)
    else:
        model = AVEQA().to(device)
        model.load_state_dict(torch.load('./aveqa_model/checkpoint-2000/pytorch_model.bin'))
        model.eval()
        start_test(model, test_dataset)
