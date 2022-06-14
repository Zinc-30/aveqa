import torch
import string
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tqdm
from transformers import BertTokenizer
import argparse
import json


class AEPub(Dataset):
    def __init__(self, dataset_path, tokenizer, msk):
        super().__init__()
        _, tup = self.read_txt(dataset_path, msk=msk)
        self.text = tup[0]
        self.cat_text = tup[5]
        self.text_msk = tup[1]
        self.cat_text_msk = tup[6]
        self.attribute = tup[2]
        self.label = tup[3]
        self.msk_id_list = tup[4]
        self.answer_label = tup[7]
        self.begin_label = tup[8]
        self.end_label = tup[9]
        self.attribute_word_label = tup[10]
        self.encodings = tokenizer(self.cat_text, padding='max_length', truncation=True, max_length=128,
                                   return_tensors='pt')
        self.encodings_msk = tokenizer(self.cat_text_msk, padding='max_length', truncation=True, max_length=128,
                                       return_tensors='pt')

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        # item = {}
        item = {key: val[idx] for key, val in self.encodings.items()}
        for key, val in self.encodings_msk.items():
            item[key + '_msk'] = val[idx]
        item['text'] = self.text[idx]
        item['label_word'] = self.label[idx]
        item['cat_text'] = self.cat_text[idx]
        item['text_msk'] = self.text_msk[idx]
        item['cat_text_msk'] = self.cat_text_msk[idx]
        # item['data_id'] = self.id[idx]
        item['attribute'] = self.attribute[idx]
        # item['category'] = self.category[idx]
        # item['title'] = self.title[idx]
        # item['description'] = self.description[idx]
        item['msk_index'] = self.msk_id_list[idx]
        # item['seg_label'] = self.segment_id[idx]
        # item['sep_idx'] = self.sep_idx_list[idx]
        item['answer_label'] = self.answer_label[idx]
        item['begin_label'] = self.begin_label[idx]
        item['end_label'] = self.end_label[idx]
        item['attribute_word_label'] = self.attribute_word_label[idx]
        # item['word_sequence_label_QA'] = self.modified_label[idx]

        # print(list(item.keys()))
        return item

    def read_txt(self, dataset_path: str, msk: str = 'value'):
        dataset, text, text_msk, cat_text_msk, attribute, label, msk_id_list, cat_text = [], [], [], [], [], [], [], []
        answer_label, begin_label, end_label, attribute_word_label = [], [], [], []
        max_label = 0
        with open(dataset_path, "rb+") as f:
            for line in f.readlines():
                byte_line = line.split(b'\x01')
                str_line = [item.decode('utf-8') for item in byte_line]
                str_line_word_list = str_line[0].split()
                attribute_word_list = str_line[1].split()
                attribute_word_idx = [len(str_line_word_list), len(str_line_word_list) + len(attribute_word_list)]
                cat_text.append(str_line[0] + ' ' + str_line[1])
                attribute_word_label.append(attribute_word_idx)
                if str_line[2].strip() != 'NULL':
                    answer_label.append(1)
                    msk_idx = []
                    label_words = str_line[2].strip()
                    idx_candidate = []
                    idx_result = []
                    label_words_list = label_words.split()
                    if len(label_words_list) > max_label:
                        max_label = len(label_words_list)
                    for word in label_words_list:
                        idx_candidate.append(self.get_index(str_line_word_list, word))
                    if len(idx_candidate) > 1:
                        for i, idx_list in enumerate(idx_candidate):
                            for idx in idx_list:
                                if i + 1 < len(idx_candidate):
                                    if idx + 1 in idx_candidate[i + 1]:
                                        idx_result.append(idx)
                                        idx_result.append(idx + 1)
                        idx_result = list(set(idx_result))[:len(label_words_list)]
                        msk_idx = [str(w) for w in idx_result]
                        for i in idx_result:
                            str_line_word_list[i] = '[MASK]'
                    else:
                        idx_result = idx_candidate[0]
                        msk_idx.append(str(idx_result[0]))
                        str_line_word_list[idx_result[0]] = '[MASK]'
                else:
                    answer_label.append(0)
                    msk_idx = ['-1']
                str_line_word_msk = ' '.join(str_line_word_list)
                if len(msk_idx) > 1:
                    b = int(msk_idx[0])
                    e = int(msk_idx[-1])
                    begin_label.append(b)
                    end_label.append(e)
                else:
                    b = int(msk_idx[0])
                    e = int(msk_idx[-1])
                    begin_label.append(b)
                    end_label.append(b)
                dataset.append({
                    'text': str_line[0],
                    'text_msk': str_line_word_msk,
                    'cat_text': str_line[0] + ' ' + str_line[1],
                    'cat_text_msk': str_line_word_msk + ' ' + str_line[1],
                    'attribute': str_line[1],
                    'answer_label': answer_label,
                    'label': str_line[2].strip(),
                    'msk_idx': '|'.join(msk_idx),
                    'begin_label': b,
                    'end_label': e,
                    'attribute_word_label': attribute_word_idx
                })
                text.append(str_line[0])
                text_msk.append(str_line_word_msk)
                if msk == 'attribute':
                    cat_text_msk.append(str_line[0] + ' [MASK]')
                else:
                    cat_text_msk.append(str_line_word_msk + ' ' + str_line[1])
                attribute.append(str_line[1])
                label.append(str_line[2].strip())
                msk_id_list.append('|'.join(msk_idx))
        print('Max label len: {}'.format(max_label))
        return dataset, (
            text, text_msk, attribute, label, msk_id_list, cat_text, cat_text_msk, answer_label, begin_label, end_label,
            attribute_word_label)

    def get_index(self, lst=None, item=''):
        return [index for (index, value) in enumerate(lst) if (value in item or item in value) and value != '']


if __name__ == '__main__':
    dataset_path = "./dataset/publish_data.txt"
    # ae_pub_dataset, tup = read_txt(dataset_path)
    Tokenizer = BertTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
    aePub = AEPub(dataset_path, Tokenizer, msk='attribute')
    torch.save(aePub, './dataset/aePub_squad2')
    print('Finish')
