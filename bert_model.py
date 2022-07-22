from transformers import BertTokenizer, BertModel, BertConfig, BertForQuestionAnswering
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import Dataset, DataLoader
import re

from model.transformer import TransformerBlock
from model.embedding import BERTEmbedding


# Tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def generate_data(full_dataset):
    train_size = int(0.7 * len(full_dataset))
    eval_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - eval_size - train_size
    torch.manual_seed(0)
    train_dataset, eval_dataset, test_dataset = torch.utils.data.random_split(full_dataset,
                                                                              [train_size, eval_size, test_size])
    return train_dataset, eval_dataset, test_dataset


class AVEQA(nn.Module):
    def __init__(self, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, msk='value',
                 model_name="bert-base-uncased"):  # "deepset/bert-base-cased-squad2"
        super(AVEQA, self).__init__()
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ["[scinotexp]", "[DOT]"]})
        # self.tokenizer.add_special_tokens(
        #   {'additional_special_tokens': ["##0", "##1", "##2", "##3", "##4", "##5", "##6", "##7", "##8", "##9"]})
        self.bert_model = BertModel.from_pretrained(self.model_name)
        self.bert_model.resize_token_embeddings(len(self.tokenizer))
        # self.bert_model_contextual = BertForQuestionAnswering.from_pretrained(self.model_name)
        self.bert_model_contextual = BertModel.from_pretrained(self.model_name)
        self.bert_model_contextual.resize_token_embeddings(len(self.tokenizer))
        self.wb = nn.Linear(hidden, 1)
        self.we = nn.Linear(2 * hidden, 1)
        self.hidden = hidden
        self.config = BertConfig.from_pretrained(self.model_name)
        self.classifier = nn.Linear(hidden, 2)
        self.projector = nn.Linear(hidden, len(self.tokenizer))
        # self.projector = nn.Linear(hidden, 30522)
        self.softmax = nn.Softmax(dim=1)
        self.msk = msk

    def get_index(self, lst=None, item=''):
        return [index for (index, value) in enumerate(lst) if
                item in re.sub("#", "", value) or re.sub("#", "", value) in item]

    def forward(self, input_data, device):
        ids_list = input_data['input_ids'].cpu().tolist()
        label_token_list = input_data['input_ids_label'].cpu().tolist()
        idx_begin, idx_end, have_idx_list = [], [], []
        exception_list = []
        for index in range(len(ids_list)):
            mark = False
            token_list = self.tokenizer.convert_ids_to_tokens(ids_list[index])
            token_list_label = self.tokenizer.convert_ids_to_tokens(label_token_list[index])
            cls_idx = token_list_label.index('[CLS]')
            sep_idx = token_list_label.index('[SEP]')
            token_list_label = token_list_label[cls_idx + 1:sep_idx]
            label = ''.join(token_list_label)
            if re.sub("#", "", label.lower()) != 'null':
                have_idx_list.append(index)
                if token_list_label[0] == token_list_label[-1]:
                    idx_single = self.get_index(token_list, re.sub("#", "", token_list_label[0]))
                    if not idx_single:
                        have_idx_list = have_idx_list[0:-1]
                        exception_list.append(index)
                        idx_begin.append(0)
                        idx_end.append(0)
                        continue
                    idx_begin.append(idx_single[0])
                    idx_end.append(idx_single[0])
                else:
                    res_start = self.get_index(token_list, re.sub("#", "", token_list_label[0]))
                    # if not res_start:
                    # print(token_list)
                    # print(token_list_label[0])
                    res_end = self.get_index(token_list, re.sub("#", "", token_list_label[-1]))
                    if len(res_start) == 1 and len(res_end) == 1:
                        idx_begin.append(res_start[0])
                        idx_end.append(res_end[0])
                    else:
                        for candidate_start_index in res_start:
                            for candidate_end_index in res_end:
                                candidate_str = ''.join(
                                    token_list[candidate_start_index:candidate_end_index + 1]).lower()
                                if candidate_start_index > candidate_end_index:
                                    continue
                                elif re.sub("#", "", label.lower()) in re.sub("#", "", candidate_str):
                                    idx_begin.append(candidate_start_index)
                                    idx_end.append(candidate_end_index)
                                    mark = True
                                    break
                            if mark:
                                break
                        if not mark:
                            exception_list.append(index)
                            idx_begin.append(0)
                            idx_end.append(0)
            else:
                idx_begin.append(0)
                idx_end.append(0)
        # contextual_output_whole = self.bert_model_contextual(input_ids=input_data['input_ids_msk'].to(device),
        #                                                      # token_type_ids=input_data['token_type_ids'].to(device),
        #                                                      attention_mask=input_data['attention_mask_msk'].to(device),
        #                                                      start_positions=torch.LongTensor(idx_begin).to(device),
        #                                                      end_positions=torch.LongTensor(idx_end).to(device),
        #                                                      output_hidden_states=True)
        contextual_outputs = self.bert_model_contextual(input_ids=input_data['input_ids_msk'].to(device),
                                                        # token_type_ids=input_data['token_type_ids'].to(device),
                                                        attention_mask=input_data['attention_mask_msk'].to(device))
        contextual_output = contextual_outputs.last_hidden_state

        # ==== compute qa index ====
        start_logits = self.wb(contextual_output)
        start_logits = start_logits.squeeze(-1).contiguous()
        # print(start_logits.size())
        answer_start_index = start_logits.argmax(dim=-1)

        # generate start hidden embedding
        batch_size = contextual_output.size(0)
        token_size = contextual_output.size(1)
        emb_size = contextual_output.size(2)

        start_index = answer_start_index.unsqueeze(1).expand(batch_size, token_size)
        end_contextual_size = [batch_size, token_size, 2 * emb_size]
        end_contextual = torch.zeros(end_contextual_size)
        for batch in range(batch_size):
            hidden_start = contextual_output[batch, start_index[batch], :]
            end_contextual[batch] = torch.cat([hidden_start, contextual_output[batch]], dim=-1)

        end_logits = self.we(end_contextual.to(device)).squeeze(-1).contiguous()
        end_mask = torch.zeros(batch_size, token_size).bool().to(device)

        for batch in range(batch_size):
            for token in range(token_size):
                if token >= answer_start_index[batch]:
                    end_mask[batch, token] = True

        answer_end_index = torch.zeros(answer_start_index.size())
        for batch in range(batch_size):
            answer_end_index[batch] = token_size - end_mask[batch].sum() + torch.masked_select(end_logits[batch],
                                                                                               end_mask[batch]).argmax(
                dim=-1)

        # ==== compute no answer ====
        no_answer = self.classifier(contextual_output[:, 0, :])
        pred_label = torch.argmax(no_answer, dim=1)

        # ==== compute dmlm ====
        bert_output = self.bert_model(input_ids=input_data['input_ids'].to(device),
                                      # token_type_ids=input_data['token_type_ids'].to(device),
                                      attention_mask=input_data['attention_mask'].to(device))
        have_answer_list, msk_index_converted = self.convert_msk_index(input_data['begin_label'],
                                                                       input_data['end_label'],
                                                                       exception_list)
        # print(have_answer_list)
        # print(msk_index_converted)
        if self.msk == 'attribute' and self.training:
            msk_index_raw = (input_data['input_ids_msk'] == 103).nonzero()
            msk_index_converted = []
            for row_idx, idx_pair in enumerate(msk_index_raw.cpu().tolist()):
                if row_idx != idx_pair[0]:
                    print('Error in MASK index per row!')
                msk_index_converted.append([idx_pair[2]])
            # for label_idx in input_data['attribute_word_label'].cpu().tolist():
            #   msk_index_converted.append([i for i in range(label_idx[0], label_idx[1])])
            have_answer_list_inp = [i for i in range(pred_label.size(0))]
            bert_gt = self.flat_output(bert_output.last_hidden_state, have_answer_list_inp, msk_index_converted)
            contextual_prediction = self.flat_output(contextual_output, have_answer_list_inp, msk_index_converted)
        else:
            bert_gt = self.flat_output(bert_output.last_hidden_state, have_answer_list, msk_index_converted)
            contextual_prediction = self.flat_output(contextual_output, have_answer_list, msk_index_converted)

        if bert_gt.size(-1) == 0 or contextual_prediction.size(-1) == 0:
            bert_gt_output = bert_gt
            contextual_prediction_output = contextual_prediction
            contain_valid_value = 0

        else:
            bert_gt_output = self.softmax(self.projector(bert_gt.to(device)))
            # print(bert_gt_output.size())
            contextual_prediction_output = self.softmax(self.projector(contextual_prediction.to(device)))
            contain_valid_value = 1
        # print(contextual_prediction_output.size())
        # print(idx_begin)
        # print(idx_end)
        return {
            'contain_valid_value': contain_valid_value,
            'no_answer_output': no_answer.to(device),  #
            'answer_label': input_data['answer_label'],  #
            # 'have_answer_idx': torch.LongTensor(have_answer_list).to(device),
            # 'bert_output': bert_output.last_hidden_state.to(device),
            # 'contextual_output': contextual_output.to(device),
            # 'contextual_output_whole': contextual_output_whole,  #
            'start_logit': start_logits.to(device),
            'end_logit': end_logits.to(device),
            'bert_gt_output': bert_gt_output.to(device),  #
            'contextual_prediction_output': contextual_prediction_output.to(device),  #
            # 'begin_label': input_data['begin_label'][have_answer_list].to(device),
            # 'end_label': input_data['end_label'][have_answer_list].to(device),
            # 'msk_index': input_data['msk_index'],  # 改成length和起始id
            'pred_begin_idx': answer_start_index.to(device),  #
            'pred_end_idx': answer_end_index.to(device),  #
            # 'begin_label_ori': input_data['begin_label'].to(device),
            # 'end_label_ori': input_data['end_label'].to(device),
            'begin_label_ori': torch.Tensor(idx_begin).contiguous().type(torch.int64).to(device),  #
            'end_label_ori': torch.Tensor(idx_end).contiguous().type(torch.int64).to(device)  #
        }

    # total_loss = qa_loss + alpha * dmlm_loss + beta * no_answer_loss
    # total_loss.backward()

    def convert_msk_index(self, begin_idx: torch.Tensor, end_idx: torch.Tensor, exception_list: list):
        converted_list = []
        begin_list = begin_idx.cpu().tolist()
        end_list = end_idx.cpu().tolist()
        idx_list = []
        for i in range(len(end_list)):
            temp_list = []
            for idx in range(begin_list[i], end_list[i] + 1):
                if idx != -1 and idx not in exception_list:
                    temp_list.append(idx)
                else:
                    break
            if len(temp_list) > 0:
                idx_list.append(i)
                converted_list.append(temp_list)
        return idx_list, converted_list

    def flat_output(self, input_tensor: torch.Tensor, have_answer_idx: list, msk_index_converted: list):
        have_idx = input_tensor[have_answer_idx, :, :].tolist()
        result = []
        for idx, item in enumerate(have_idx):
            for msk_idx in msk_index_converted[idx]:
                result.append(item[msk_idx])
        return torch.Tensor(result)
