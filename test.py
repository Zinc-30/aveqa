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

import json
import torch
from bert_model import AVEQA
from ae_pub import AEPub
import tqdm
import train
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, \
    precision_recall_fscore_support
from transformers import BertTokenizer

Tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def sort_dict(input_dict: dict, k1=-1, k2=-1):
    sorted_dict = {}
    temp = sorted(input_dict.items(), key=lambda x: len(x[1]), reverse=True)
    for i in range(len(temp)):
        if k1 > 0 and k1 == i:
            break
        sorted_dict[temp[i][0]] = temp[i][1][:k2]

    return sorted_dict


def process_bad_case(gt_begin_idx, gt_end_idx, pred_begin_idx, pred_end_idx, input_ids, input_ids_label, wrong_type):
    bad_case_dict = {}
    bad_case_dict['gt_idx'] = [gt_begin_idx, gt_end_idx]
    bad_case_dict['wrong_type'] = wrong_type
    bad_case_dict['pred_idx'] = [pred_begin_idx, pred_end_idx]
    bad_case_dict['text'] = Tokenizer.decode(input_ids)
    bad_case_dict['label'] = Tokenizer.decode(input_ids_label)
    return bad_case_dict


def compute_metrics(inputs, outputs, NA_T, NA_F, T, F, y_true: list, y_pred: list, prec, rec, f1, classfi_dict: dict,
                    gt_dict: dict, bad_case_dict, good_case_dict):
    class_label = inputs['class_label'].cpu().tolist()
    input_ids = inputs['input_ids'].cpu().tolist()
    input_ids_label = inputs['input_ids_label'].cpu().tolist()
    attribute = inputs['attribute']
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
                    if attribute[i] not in good_case_dict:
                        good_case_dict[attribute[i]] = [
                            process_bad_case(gt_begin_idx[i], gt_end_idx[i], pred_begin_idx[i],
                                             pred_end_idx[i],
                                             input_ids[i],
                                             input_ids_label[i], 'true_prediction')]
                    else:
                        good_case_dict[attribute[i]].append(
                            process_bad_case(gt_begin_idx[i], gt_end_idx[i], pred_begin_idx[i],
                                             pred_end_idx[i],
                                             input_ids[i],
                                             input_ids_label[i], 'true_prediction'))
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
                    if attribute[i] not in bad_case_dict:
                        bad_case_dict[attribute[i]] = [
                            process_bad_case(gt_begin_idx[i], gt_end_idx[i], pred_begin_idx[i],
                                             pred_end_idx[i],
                                             input_ids[i],
                                             input_ids_label[i], 'wrong_prediction')]
                    else:
                        bad_case_dict[attribute[i]].append(
                            process_bad_case(gt_begin_idx[i], gt_end_idx[i], pred_begin_idx[i],
                                             pred_end_idx[i],
                                             input_ids[i],
                                             input_ids_label[i], 'wrong_prediction'))

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
            if attribute[i] not in bad_case_dict:
                bad_case_dict[attribute[i]] = [
                    process_bad_case(gt_begin_idx[i], gt_end_idx[i], pred_begin_idx[i],
                                     pred_end_idx[i],
                                     input_ids[i],
                                     input_ids_label[i], 'no_answer')]
            else:
                bad_case_dict[attribute[i]].append(
                    process_bad_case(gt_begin_idx[i], gt_end_idx[i], pred_begin_idx[i],
                                     pred_end_idx[i],
                                     input_ids[i],
                                     input_ids_label[i], 'no_answer'))
            F += 1
    return NA_T, NA_F, T, F, y_true, y_pred


def start_test(model, test_dataset, testing_config):
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
    bad_case_dict = {}
    good_case_dict = {}
    for i, batch in enumerate(tqdm.tqdm(dataloader)):
        outputs = model(batch, 'cuda')
        NA_T, NA_F, T, F, y_true, y_pred = compute_metrics(batch, outputs, NA_T, NA_F, T,
                                                           F, y_true,
                                                           y_pred, prec, rec, f1,
                                                           class_dict, gt_dict,
                                                           bad_case_dict, good_case_dict)
    # precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    precision = sum(prec) / len(prec)
    recall = sum(rec) / len(rec)
    f1 = (precision + recall) / 2

    print('Accuracy: {}, No Answer Accuracy: {}, Precision: {}, Recall: {}, F1: {}'.format(T / (T + F),
                                                                                           NA_T / (NA_T + NA_F),
                                                                                           precision, recall, f1))
    '''
    precision_bn, recall_bn, f1_bn, _ = precision_recall_fscore_support(class_dict[0], gt_dict[0], average='micro')
    print(
        'Len: {}, Brand name: \n Precision: {}, Recall: {}, F1: {}'.format(len(class_dict[0]), precision_bn, recall_bn,
                                                                           f1_bn))

    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(class_dict[1], gt_dict[1], average='micro')
    print('Len: {}, Material: \n Precision: {}, Recall: {}, F1: {}'.format(len(class_dict[1]), precision_m, recall_m,
                                                                           f1_m))

    precision_c, recall_c, f1_c, _ = precision_recall_fscore_support(class_dict[2], gt_dict[2], average='micro')
    print(
        'Len: {}, Color: \n Precision: {}, Recall: {}, F1: {}'.format(len(class_dict[2]), precision_c, recall_c, f1_c))

    precision_ca, recall_ca, f1_ca, _ = precision_recall_fscore_support(class_dict[3], gt_dict[3], average='micro')
    print('Len: {}, Category: \n Precision: {}, Recall: {}, F1: {}'.format(len(class_dict[3]), precision_ca, recall_ca,
                                                                           f1_ca))
    '''
    with open(testing_config['model_output_dir'] + '/bad_case.json', 'w') as file:
        file.write(json.dumps(sort_dict(bad_case_dict, 10, 10), indent=4))
    with open(testing_config['model_output_dir'] + '/good_case.json', 'w') as file:
        file.write(json.dumps(sort_dict(good_case_dict, 10, 10), indent=4))


if __name__ == '__main__':
    with open('./config.json', 'r') as file:
        testing_config = json.load(file)
    dataset = torch.load(testing_config['test_dataset'])
    model_path = testing_config['model_output_dir'] + '/checkpoint-' + str(
        testing_config['max_steps']) + '/pytorch_model.bin'
    model = AVEQA().to('cuda')
    model.load_state_dict(torch.load(model_path))
    model.bert_model_contextual.load_state_dict(torch.load(testing_config['model_output_dir'] + '/bert_state_dict'))
    model.eval()
    start_test(model, dataset, testing_config)
