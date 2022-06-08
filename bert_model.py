from transformers import BertTokenizer, BertModel, BertConfig
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
    def __init__(self, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        super(AVEQA, self).__init__()
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        # self.bert_model_contextual = BertModel.from_pretrained("bert-base-uncased")
        self.hidden = hidden
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in
             range(n_layers)])
        self.config = BertConfig.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(hidden, 2)
        self.weight_begin = nn.Linear(hidden, 1)
        self.weight_end = nn.Linear(hidden * 2, 1)
        self.projector = nn.Linear(hidden, 30522)
        self.softmax = nn.Softmax(dim=1)
        self.embedding = BERTEmbedding(vocab_size=self.config.vocab_size, embed_size=self.hidden)

    def get_index(self, lst=None, item=''):
        return [index for (index, value) in enumerate(lst) if value == item]

    def forward(self, input_data, device):
        # print(input_data['input_ids_msk'].size())

        segment_id = torch.ones(input_data['input_ids_msk'].size(), dtype=torch.int)
        x = self.embedding(input_data['input_ids_msk'].to(device), segment_id.to(device))

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, input_data['attention_mask_msk'].to(device))
        contextual_output = x.to(device)

        # contextual_output = self.bert_model_contextual(input_ids=input_data['input_ids_msk'].to(device),
        #                                              token_type_ids=input_data['token_type_ids'].to(device),
        #                                              attention_mask=input_data['attention_mask_msk'].to(
        #                                                  device)).last_hidden_state
        no_answer = self.classifier(contextual_output[:, 0, :])
        # no_answer_loss = no_answer_loss_function(no_answer, input_data['answer_label'])
        pred_label = torch.argmax(no_answer, dim=1)
        # have_answer_idx = self.get_index(pred_label.tolist(), item=1)
        # no_answer_idx = self.get_index(pred_label.tolist(), item=0)

        bert_output = self.bert_model(input_ids=input_data['input_ids'].to(device),
                                      token_type_ids=input_data['token_type_ids'].to(device),
                                      attention_mask=input_data['attention_mask'].to(device))
        # dmlm_loss, qa_loss = 0, 0
        # begin, end = 0, 0
        # pred_begin_idx, pred_end_idx = -1, -1
        # no_answer_loss_function = nn.CrossEntropyLoss()
        # qa_loss_function = nn.CrossEntropyLoss()

        '''
        dmlm_loss = self.dmlm_loss(bert_output[have_answer_idx, :, :],
                                   contextual_output[have_answer_idx, :, :],
                                   torch.tensor(input_data['msk_index'][have_answer_idx]))
        '''
        all_batch_idx = torch.LongTensor([i for i in range(pred_label.size(0))])
        begin = []

        for idx in range(contextual_output.size(0)):
            begin_weight = self.weight_begin(contextual_output[idx, :, :])
            begin.append(torch.squeeze(begin_weight).cpu().tolist())
            # begin_list.append(torch.argmax(torch.squeeze(begin_weight), dim=0))
        begin = torch.Tensor(begin).to(device)
        # begin = self.weight_begin(torch.flatten(contextual_output, start_dim=1))

        pred_begin_idx = torch.argmax(begin, dim=1)
        # pred_begin_idx = torch.LongTensor(begin_list).to(device)
        # pred_begin_idx += torch.ones(pred_begin_idx.size(0), dtype=torch.long).to(device)
        # print('out put complete')
        # print(contextual_output.size())
        # print(pred_begin_idx)
        # print(contextual_output[all_batch_idx, pred_begin_idx, :].size())
        # print('**************')
        h_b = contextual_output[all_batch_idx, pred_begin_idx, :]
        total = []
        for plus_idx in range(32):
            plus_idx_tensor = torch.LongTensor([plus_idx] * pred_begin_idx.size(0)).to(device)
            id_sum = pred_begin_idx + plus_idx_tensor
            input_idx = torch.where(id_sum > 127, 127, id_sum)
            h_i = contextual_output[all_batch_idx, input_idx, :]
            input_tensor = torch.cat((h_i, h_b), 1)
            total.append(self.weight_end(input_tensor.to(device)).tolist())
        end = torch.squeeze(torch.Tensor(total)).t()
        if len(end.size()) < 2:
            end = torch.unsqueeze(end, dim=0)
            print(end.size())
        pred_end_idx = torch.argmax(end, dim=-1)
        end_idx = pred_begin_idx.to(device) + pred_end_idx.to(device)
        pred_end_idx = torch.where(end_idx > 127, 127, end_idx)

        # begin_loss = qa_loss_function(begin, input_data['word_sequence_label_QA'][1])
        # end_loss = qa_loss_function(end, input_data['word_sequence_label_QA'][2])
        # qa_loss = (begin_loss + end_loss) / 2

        have_answer_list, msk_index_converted = self.convert_msk_index(input_data['begin_label'],
                                                                       input_data['end_label'])
        bert_gt = self.flat_output(bert_output.last_hidden_state, have_answer_list, msk_index_converted)
        contextual_prediction = self.flat_output(contextual_output, have_answer_list, msk_index_converted)
        bert_gt_output = self.softmax(self.projector(bert_gt.to(device)))
        contextual_prediction_output = self.softmax(self.projector(contextual_prediction.to(device)))

        return {
            'begin_output': begin.to(device)[have_answer_list, :],
            'end_output': end.to(device)[have_answer_list, :],
            'no_answer_output': no_answer.to(device),
            # 'no_answer_label': input_data['answer_label'],
            'have_answer_idx': torch.LongTensor(have_answer_list).to(device),
            'bert_output': bert_output.last_hidden_state.to(device),
            'contextual_output': contextual_output.to(device),
            'bert_gt_output': bert_gt_output.to(device),
            'contextual_prediction_output': contextual_prediction_output.to(device),
            'begin_label': input_data['begin_label'][have_answer_list].to(device),
            'end_label': input_data['end_label'][have_answer_list].to(device),
            # 'msk_index': input_data['msk_index'],  # 改成length和起始id
            'pred_begin_idx': pred_begin_idx.to(device),
            'pred_end_idx': pred_end_idx.to(device),
            'begin_label_ori': input_data['begin_label'].to(device),
            'end_label_ori': input_data['end_label'].to(device),
            'begin_output_ori': begin.to(device),
            'end_output_ori': end.to(device)
        }

        # total_loss = qa_loss + alpha * dmlm_loss + beta * no_answer_loss
        # total_loss.backward()

    def convert_msk_index(self, begin_idx: torch.Tensor, end_idx: torch.Tensor):
        converted_list = []
        begin_list = begin_idx.cpu().tolist()
        end_list = end_idx.cpu().tolist()
        idx_list = []
        for i in range(len(end_list)):
            temp_list = []
            for idx in range(begin_list[i], end_list[i] + 1):
                if idx != -1:
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
