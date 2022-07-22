import torch
import string
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tqdm
import string
from transformers import BertTokenizer
import argparse
import json
from random import *


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
        self.class_label = tup[11]
        self.encodings = tokenizer(self.cat_text, padding='max_length', truncation=True, max_length=128,
                                   return_tensors='pt')
        self.encodings_msk = tokenizer(self.cat_text_msk, padding='max_length', truncation=True, max_length=128,
                                       return_tensors='pt')
        self.encodings_label = tokenizer(self.label, padding=True, return_tensors='pt')

        # self.text_token = tokenizer.convert_ids_to_tokens(self.encodings['input_ids'])

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        # item = {}
        item = {key: val[idx] for key, val in self.encodings.items()}
        for key, val in self.encodings_msk.items():
            item[key + '_msk'] = val[idx]
        for key, val in self.encodings_label.items():
            item[key + '_label'] = val[idx]
        # item['text_token'] = self.text_token[idx]
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
        item['class_label'] = self.class_label[idx]
        # item['word_sequence_label_QA'] = self.modified_label[idx]

        # print(list(item.keys()))
        return item

    def convert_num(self, input_str: str):
        str_new = ''
        for idx, character in enumerate(input_str):
            if character in string.punctuation or character.isdigit():
                str_new += ' ' + character + ' '
            else:
                str_new += character
        '''
        str_new_list = str_new.split()
        split_str_converted = []
        for idx1, word in enumerate(str_new_list):
            if word.isdigit():
                for num in word:
                    split_str_converted.append(num)
            else:
                split_str_converted.append(word)
        return ' '.join(split_str_converted)
        '''
        return str_new

    def add_msk(self, text_word: str, begin_label, end_label):
        text_word_list = text_word.split()
        msk_idx = randint(0, len(text_word_list) - 1)
        count = 0
        while int(begin_label) <= msk_idx <= int(end_label):
            count += 1
            msk_idx = randint(0, len(text_word_list) - 1)
            if count == 100:
                print('{} {} {}'.format(begin_label, end_label, len(text_word_list)))
                break
        text_word_list[msk_idx] = '[MASK]'
        return ' '.join(text_word_list)

    def read_txt(self, dataset_path: str, msk: str = 'value'):
        dataset, text_list, text_msk, cat_text_msk, attribute, label, msk_id_list, cat_text = [], [], [], [], [], [], [], []
        answer_label, begin_label, end_label, attribute_word_label = [], [], [], []
        class_label = []
        max_label = 0
        label_dict = {}
        label_count = 1
        with open(dataset_path, "rb+") as f:
            for line in f.readlines():
                byte_line = line.split(b'\x01')
                str_line = [item.decode('utf-8') for item in byte_line]
                text_word = str_line[0].strip()
                attr = str_line[1].strip()
                str_line_word_list = text_word.split()
                attribute_word_list = attr.split()
                attribute_word_idx = [len(str_line_word_list), len(str_line_word_list) + len(attribute_word_list)]
                cat_text.append(text_word + ' [SEP] ' + attr)
                attribute_word_label.append(attribute_word_idx)
                if str_line[2].strip() != 'NULL':
                    answer_label.append(1)
                    msk_idx = []
                    label_words = str_line[2].strip()
                    if label_words.lower() not in label_dict:
                        label_dict[label_words.lower()] = label_count
                        class_label.append(label_count)
                        label_count += 1
                    else:
                        class_label.append(label_dict[label_words.lower()])

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
                        '''
                        for i in idx_result:
                            str_line_word_list[i] = '[MASK]'
                        '''
                    else:
                        idx_result = idx_candidate[0]
                        msk_idx.append(str(idx_result[0]))
                        # str_line_word_list[idx_result[0]] = '[MASK]'
                else:
                    class_label.append(0)
                    answer_label.append(0)
                    msk_idx = ['-1']
                # str_line_word_msk = ' '.join(str_line_word_list)
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
                str_line_word_msk = self.add_msk(text_word, b, e)
                dataset.append({
                    'text': str_line[0],
                    'text_msk': str_line_word_msk,
                    'cat_text': str_line[0] + ' [SEP] ' + str_line[1],
                    'cat_text_msk': str_line_word_msk + ' [SEP] ' + str_line[1],
                    'attribute': str_line[1],
                    'answer_label': answer_label,
                    'label': str_line[2].strip(),
                    'msk_idx': '|'.join(msk_idx),
                    'begin_label': b,
                    'end_label': e,
                    'attribute_word_label': attribute_word_idx,
                    'class_label': class_label
                })
                text_list.append(str_line[0])
                text_msk.append(str_line_word_msk)
                cat_text_msk.append(str_line_word_msk + ' [SEP] ' + str_line[1])
                attribute.append(str_line[1])
                label.append(str_line[2].strip())
                msk_id_list.append('|'.join(msk_idx))
        print('Max label len: {}'.format(max_label))
        print('Label count: {}'.format(label_count))
        return dataset, (
            text_list, text_msk, attribute, label, msk_id_list, cat_text, cat_text_msk, answer_label, begin_label,
            end_label,
            attribute_word_label, class_label)

    def get_index(self, lst=None, item=''):
        return [index for (index, value) in enumerate(lst) if (value in item or item in value) and value != '']


if __name__ == '__main__':
    with open('./config.json', 'r') as file:
        training_config = json.load(file)
    dataset_path = "./dataset/publish_data_filtered.txt"
    num_list = ["##0", "##1", "##2", "##3", "##4", "##5", "##6", "##7", "##8", "##9"]
    # ae_pub_dataset, tup = read_txt(dataset_path)
    # Tokenizer = BertTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
    Tokenizer = BertTokenizer.from_pretrained(training_config['model_name'])
    Tokenizer.add_special_tokens({'additional_special_tokens': num_list})
    # Tokenizer = BertTokenizer(vocab_file='./vocab.txt')
    print('encode: 1')
    print(Tokenizer('1'))
    aePub = AEPub(dataset_path, Tokenizer, msk='value')
    torch.save(aePub, training_config['dataset'])
    print('Finish')
