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
import train


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


def start_test(model, test_dataset):
    T, F = 0, 0
    NA_T, NA_F = 0, 0

    dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    for i, batch in enumerate(tqdm.tqdm(dataloader)):
        outputs = model(batch, device)
        NA_T, NA_F, T, F = compute_metrics(outputs, NA_T, NA_F, T, F)

    print('Accuracy: {}, No Answer Accuracy: {}'.format(T / (T + F), NA_T / (NA_T + NA_F)))


if __name__ == '__main__':
    with open('./config.json', 'r') as file:
        testing_config = json.load(file)
    dataset = torch.load(testing_config['test_dataset'])
    model_path = testing_config['model_output_dir'] + '/checkpoint-' + str(
        testing_config['max_steps']) + '/pytorch_model.bin'
    model = AVEQA().to(device)
    model.load_state_dict(torch.load(model_path))
    # model.bert_model_contextual.load_state_dict(torch.load(testing_config['model_output_dir'] + '/bert_state_dict'))
    model.eval()
    start_test(model, test_dataset)
