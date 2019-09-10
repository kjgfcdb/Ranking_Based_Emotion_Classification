import json
import os
import pickle
from collections import namedtuple
from datetime import datetime
from os.path import join
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, SequentialSampler
from tqdm import tqdm, trange

from tools import backup_code, setup_logger, evaluate_classification, evaluate_ranking
import prettytable as pt
from bert_config import *

Example = namedtuple("Example", ["guid", "text", "label", "score"])
Features = namedtuple("Features", ['input_ids', 'input_mask', 'segment_ids', 'label', 'score'])


class Processor:
    """Processor for preparing training and testing examples"""

    def __init__(self, args):
        with open(args.data_dir) as f:
            data = f.readlines()
        self.data = np.array(data)
        np.random.seed(1234)
        random_idx = np.random.permutation(len(data))
        self.data = self.data[random_idx]
        self.args = args

    def get_train_examples(self):
        lines = self.data[:int(len(self.data) * 0.8)]
        return self._create_examples(
            lines, "train")

    def get_test_examples(self):
        lines = self.data[int(len(self.data) * 0.8):]
        return self._create_examples(
            lines, "dev")

    def _create_examples(self, lines, set_type):
        assert isinstance(self.args, Config)
        examples = []
        for i, ln in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            obj = json.loads(ln)
            text = obj['text']
            emotion = obj['emotion']
            emotion_score, emotion_label = self.args.process_emotion(emotion)
            examples.append(Example(guid=guid, text=text,
                                    label=emotion_label, score=emotion_score))
        return examples


class AttentionLayer(nn.Module):
    def __init__(self, attention_size):
        super(AttentionLayer, self).__init__()
        self.attention_scale = nn.Linear(attention_size, 1)

    def forward(self, x, mask):
        logits = self.attention_scale(x).squeeze(-1)
        logits = (logits - logits.max()).exp()

        masked_logits = logits * mask.float()
        masked_logits_sum = masked_logits.sum(dim=1, keepdim=True)
        attentions = masked_logits.div(masked_logits_sum)

        weighted_x = torch.mul(x, attentions.unsqueeze(-1).expand_as(x))
        final_features = weighted_x.sum(1)
        return final_features


class ClassifierWithBCELoss(nn.Module):
    def __init__(self, input_size, output_size):
        super(ClassifierWithBCELoss, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.fc3 = nn.Linear(input_size, output_size)

    def forward(self, feature, labels=None):
        feature = F.relu(self.fc1(feature))
        feature = F.relu(self.fc2(feature))
        feature = self.fc3(feature)
        if labels is not None:
            bce_loss = nn.BCEWithLogitsLoss()
            return bce_loss(feature, labels)
        return feature


class RegressorWithMSELoss(nn.Module):
    def __init__(self, input_size, output_size):
        super(RegressorWithMSELoss, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.fc3 = nn.Linear(input_size, output_size)
        self.act = nn.ReLU()

    def forward(self, feature, labels=None):
        feature = self.act(self.fc1(feature))
        feature = self.act(self.fc2(feature))
        feature = self.act(self.fc3(feature))
        if labels is not None:
            mse_loss = nn.MSELoss()
            return mse_loss(feature, labels)
        return feature


class RankNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(RankNet, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.fc3 = nn.Linear(input_size, output_size)
        self.num_labels = output_size

    def forward(self, feature, labels=None):
        device = feature.device
        feature = F.relu(self.fc1(feature))
        feature = F.relu(self.fc2(feature))
        scores = F.relu(self.fc3(feature))
        # output is of shape (batch_size, num_labels)
        batch_size = scores.size(0)
        if labels is not None:
            # idx = torch.randint(self.num_labels, (batch_size, 2)).to(device)
            loss = None
            for i in range(self.num_labels):
                for j in range(i + 1, self.num_labels):
                    idx = torch.zeros(batch_size, 2).long().to(device)
                    idx[:, 0] = i
                    idx[:, 1] = j

                    sij = scores.gather(1, idx)
                    lij = labels.gather(1, idx)
                    s_diff = sij[:, 0] - sij[:, 1]
                    l_diff = lij[:, 0] - lij[:, 1]

                    mask = torch.zeros(batch_size).to(device)
                    mask[l_diff > 0] = 1
                    mask[l_diff < 0] = -1
                    if loss is None:
                        loss = (1 - mask) * s_diff / 2.0 + torch.log(1 + torch.exp(-s_diff))
                    else:
                        loss += (1 - mask) * s_diff / 2.0 + torch.log(1 + torch.exp(-s_diff))
            return loss.mean() / 5.0
        return scores


class ListNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ListNet, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, output_size)
        self.num_labels = output_size

    def forward(self, feature, labels=None):
        pred = torch.tanh(self.fc1(feature))
        pred = self.fc2(pred)
        if labels is not None:
            pred = F.softmax(pred, -1)
            labels = F.softmax(labels, -1)
            loss = self.kl_div_loss(pred, labels)
            return loss
        return pred

    def kl_div_loss(self, pred, labels):
        loss = nn.KLDivLoss(reduction='sum')
        kl_loss = loss(pred.log(), labels)
        kl_loss /= pred.size(0)  # batch mean
        return kl_loss * 5


class AttentionBert(BertModel):
    def __init__(self, config, attention_size=768):
        super(AttentionBert, self).__init__(config)
        self.attention_layer = AttentionLayer(attention_size)
        self.combine = nn.Sequential(
            nn.Linear(attention_size * 4, attention_size),
            nn.Tanh()
        )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        attention_output = self.attention_layer(sequence_output, attention_mask)
        mean_output = torch.mean(sequence_output, dim=1)
        max_output = torch.max(sequence_output, dim=1).values

        pooled_output = torch.cat((attention_output, pooled_output, mean_output, max_output), dim=1)
        pooled_output = self.combine(pooled_output)
        return pooled_output


class ClassificationModel(nn.Module):
    def __init__(self, pretrained_model_dir, num_labels, hidden_dropout_prob, hidden_size):
        super(ClassificationModel, self).__init__()
        self.num_labels = num_labels
        self.backbone = AttentionBert.from_pretrained(pretrained_model_dir)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = ClassifierWithBCELoss(hidden_size, num_labels)
        # self.ranknet = RankNet(hidden_size, num_labels)
        # self.ranknet = RegressorWithMSELoss(hidden_size, num_labels)
        self.ranknet = ListNet(hidden_size, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, scores=None):
        pooled_output = self.backbone(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        if labels is not None:
            loss1 = self.classifier(pooled_output, labels)
            loss2 = self.ranknet(pooled_output, labels)
            return loss1, loss2
        else:
            logits = self.classifier(pooled_output)
            scores = self.ranknet(pooled_output)
            return logits, scores


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for example in examples:
        tokens = tokenizer.tokenize(example.text)
        # leave space for [CLS] and [SEP]
        tokens = tokens[: min(len(tokens), max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]

        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        # use zero padding
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        label = example.label
        score = example.score

        features.append(
            Features(input_ids=input_ids,
                     input_mask=input_mask,
                     segment_ids=segment_ids,
                     label=label,
                     score=score))
    return features


def make_dataloder(examples: List, max_seq_len: int, tokenizer: BertTokenizer, batch_size: int, mode: str):
    features = convert_examples_to_features(examples, max_seq_len, tokenizer)

    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    label = torch.tensor([f.label for f in features], dtype=torch.float)
    score = torch.tensor([f.score for f in features], dtype=torch.float)

    data_source = TensorDataset(
        input_ids, input_mask, segment_ids, label, score)
    if mode == "train":
        sampler = RandomSampler(data_source)
    elif mode == "test":
        sampler = SequentialSampler(data_source)
    else:
        raise NotImplementedError()
    return DataLoader(data_source, sampler=sampler, batch_size=batch_size)


def train_net():
    # preparing
    # args = Sina_News_Config()
    args = Ren_CECps_Config()
    processor = Processor(args)
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = join(args.output_dir, time_stamp)
    os.makedirs(save_dir)
    backup_code(os.path.abspath(os.path.curdir), os.path.join(save_dir, "code"))
    # set up logger
    logger = setup_logger(save_path=join(save_dir, "logger.txt"))
    _print = logger.info

    tokenizer = BertTokenizer.from_pretrained(
        os.path.join(args.output_dir, args.pretrained_model_name, args.pretrained_model_name + "-vocab.txt"),
        do_lower_case=False)

    train_examples = processor.get_train_examples()
    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    # Prepare model
    model = ClassificationModel(os.path.join(args.output_dir, args.pretrained_model_name),
                                args.num_labels, args.dropout_prob, args.hidden_size)
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        _print("Using multi GPUs!")

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)
    # Prepare dataloader
    train_dataloader = make_dataloder(
        train_examples, args.max_seq_length, tokenizer, args.train_batch_size, mode="train")
    test_examples = processor.get_test_examples()
    with open("train.examples", "wb") as f:
        pickle.dump(train_examples, f)
    with open("test.examples", "wb") as f:
        pickle.dump(test_examples, f)
    test_dataloader = make_dataloder(
        test_examples, args.max_seq_length, tokenizer, args.train_batch_size, mode="test")
    for epoch in trange(int(args.num_train_epochs), desc="Epoch", leave=True):
        # tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training...")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, label_scores = batch
            loss1, loss2 = model(input_ids, segment_ids,
                                 input_mask, label_ids, label_scores)
            # tr_loss += loss1.mean().item() + loss2.mean().item()
            (loss1 + loss2).mean(dim=0).sum().backward()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # do evaluation
        model.eval()
        clf_pred, clf_label, rk_pred, rk_label = [], [], [], []
        for step, batch in enumerate(tqdm(test_dataloader, desc="Evaluating...")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, label_scores = batch
            with torch.no_grad():
                pred_logits, pred_scores = model(input_ids, segment_ids, input_mask)
            pred_logits = pred_logits.ge(0).float()
            clf_pred.append(pred_logits.cpu().numpy())
            clf_label.append(label_ids.cpu().numpy())
            rk_pred.append(pred_scores.cpu().numpy())
            rk_label.append(label_scores.cpu().numpy())
        clf_pred, clf_label, rk_pred, rk_label = map(lambda x: np.vstack(x), (clf_pred, clf_label, rk_pred, rk_label))
        clf_result = evaluate_classification(clf_pred, clf_label)
        rk_result = evaluate_ranking(rk_pred, rk_label, clf_label)
        tb = pt.PrettyTable(field_names=list(clf_result.keys()) + list(rk_result.keys()))
        tb.add_row(list(clf_result.values()) + list(rk_result.values()))
        tb.float_format = "0.4"
        output_str = "\n" + tb.__str__()
        tqdm.write(output_str)
        _print(output_str)
        torch.save(model.state_dict(), join(save_dir, "model_{}.pt".format(epoch)))


def evaluate_net(state_dict_path: str):
    # args = Sina_News_Config()
    args = Ren_CECps_Config()
    processor = Processor(args=args)
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = BertTokenizer.from_pretrained(
        os.path.join(args.output_dir, args.pretrained_model_name, args.pretrained_model_name + "-vocab.txt"),
        do_lower_case=False)

    if os.path.exists("test.examples"):
        with open("test.examples", "rb") as f:
            test_examples = pickle.load(f)
        print("Load test examples from previous run")
    else:
        test_examples = processor.get_test_examples()
    test_dataloader = make_dataloder(test_examples, args.max_seq_length, tokenizer, args.train_batch_size, mode="test")
    # Prepare model
    model = ClassificationModel(os.path.join(args.output_dir, args.pretrained_model_name), args.num_labels, args.dropout_prob,
                                args.hidden_size)
    model.to(device)
    model = nn.DataParallel(model)
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    # Do evaluation
    model.eval()
    clf_pred, clf_label, rk_pred, rk_label = [], [], [], []
    for step, batch in enumerate(tqdm(test_dataloader, desc="Evaluating...")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, label_scores = batch
        with torch.no_grad():
            pred_logits, pred_scores = model(input_ids, segment_ids, input_mask)
        pred_logits = pred_logits.ge(0).float()
        clf_pred.append(pred_logits.cpu().numpy())
        clf_label.append(label_ids.cpu().numpy())
        rk_pred.append(pred_scores.cpu().numpy())
        rk_label.append(label_scores.cpu().numpy())
    clf_pred, clf_label, rk_pred, rk_label = map(lambda x: np.vstack(x), (clf_pred, clf_label, rk_pred, rk_label))
    clf_result = evaluate_classification(clf_pred, clf_label)
    rk_result = evaluate_ranking(rk_pred, rk_label, clf_label)
    tb = pt.PrettyTable(field_names=list(clf_result.keys()) + list(rk_result.keys()))
    tb.add_row(list(clf_result.values()) + list(rk_result.values()))
    tb.float_format = "0.4"
    print(tb)


if __name__ == "__main__":
    train_net()
    # evaluate_net("path/to/model.pt")
