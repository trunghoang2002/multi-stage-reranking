import random
import time
import argparse
import numpy as np
import json
import io, sys
from tqdm import tqdm
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    set_seed,
)

class ScorePredictor():
    def __init__(self, args, id2doc, id2query):
        self.id2doc = id2doc
        self.id2query = id2query

        if args.use_bert:
            config = AutoConfig.from_pretrained(args.model_name_or_path[0], cache_dir=None)
            config.num_labels = 2

            self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path[0])
            self.model = []
            for path in args.model_name_or_path:
                model = AutoModelForSequenceClassification.from_pretrained(path, config=config)
                model.resize_token_embeddings(len(self.tokenizer))
                model = model.to(args.device)
                model.eval()
                self.model.append(model)

            self.bert_task_type = args.bert_task_type
            self.source_block_size = args.source_block_size
            self.bos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)
            self.eos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
            self.sep_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
            self.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
            self.device = args.device
            self.batch_size = args.batch_size

        if args.use_second_bert:
            config = AutoConfig.from_pretrained(args.second_model_name_or_path[0], cache_dir=None)
            config.num_labels = 2
            self.second_model = []
            for path in args.second_model_name_or_path:
                model = AutoModelForSequenceClassification.from_pretrained(path, config=config)
                model = model.to(args.device)
                model.eval()
                self.second_model.append(model)

            self.second_bert_task_type = args.second_bert_task_type
            self.second_source_block_size = args.second_source_block_size

    def bert_scorer(self, query_id, cand_doc_id, doc2index, model_num):
        query_text_tokenized = self.id2query[query_id]["text_tokenized"]

        text_list = []
        attention_mask_list = []
        index_list = []

        predict_prob = np.zeros(len(doc2index), dtype="float")

        models = self.model if model_num == "first" else self.second_model
        bert_task_type = self.bert_task_type if model_num == "first" else self.second_bert_task_type
        source_block_size = self.source_block_size if model_num == "first" else self.second_source_block_size

        if bert_task_type == "classification":
            for doc_id in cand_doc_id:
                doc_text_tokenized = self.id2doc[doc_id]["text_tokenized"]
                index = doc2index[doc_id]

                text = query_text_tokenized + [self.sep_token_id] + doc_text_tokenized
                text = text[:source_block_size-2]
                pad_size = source_block_size - 2 - len(text)

                text = [self.bos_token_id] + text + [self.eos_token_id] + [self.pad_token_id] * pad_size
                attention_mask = [1] * (source_block_size - pad_size) + [0] * pad_size

                text_list.append(text)
                attention_mask_list.append(attention_mask)
                index_list.append(index)

            for start in range(0, len(text_list), self.batch_size):
                with torch.no_grad():
                    text = text_list[start:start+self.batch_size]
                    text = torch.tensor(text, dtype=torch.long).to(self.device)

                    attention_mask = attention_mask_list[start:start+self.batch_size]
                    attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(self.device)

                    index = index_list[start:start+self.batch_size]

                    outputs = []
                    for model in models:
                        output = model(input_ids=text, attention_mask=attention_mask)[0]
                        output = output.softmax(dim=-1)[:, 1].cpu().numpy()
                        outputs.append(output)
                    predict_prob[index] = np.average(outputs, axis=0)

        else:
            print("No exist task")
            exit()

        return predict_prob

    def predict(self, query_id, cand_doc_id):
        doc2index = {doc_id:index for index, doc_id in enumerate(cand_doc_id)}
        predict_prob = np.zeros(len(cand_doc_id))

        for score, doc_id in enumerate(cand_doc_id[::-1]):
            predict_prob[doc2index[doc_id]] = score

        # BM25
        predict_doc_id_sorted = sorted([(score, cand_doc_id[index]) for index, score in enumerate(predict_prob)], key=lambda x: -x[0])
        predict_doc_id_highest = [doc_id for _, doc_id in predict_doc_id_sorted[:args.bert_num_candidate]]

        # BERT
        if args.use_bert:
            predict_prob = self.bert_scorer(query_id, predict_doc_id_highest, doc2index, "first")
            predict_doc_id_sorted = sorted([(score, cand_doc_id[index]) for index, score in enumerate(predict_prob)], key=lambda x: -x[0])
            predict_doc_id_highest = [doc_id for _, doc_id in predict_doc_id_sorted[:args.second_bert_num_candidate]]

        # Second BERT
        if args.use_second_bert:
            predict_prob += self.bert_scorer(query_id, predict_doc_id_highest, doc2index, "second") * 100

        return predict_prob

def calculate_mrr(true_dict, pred_dict, k=10):
    mrr_scores = []
    for query_id in true_dict:
        relevant_docs = {doc_id for doc_id, is_relevant in true_dict[query_id].items() if is_relevant}
        top_k_results = dict(sorted(pred_dict[query_id].items(), key=lambda item: item[1], reverse=True)[:k])
        retrieved_docs = [doc_id for doc_id, score in top_k_results.items()]

        # Find the rank of the first relevant document in the top k retrieved documents
        rank = None
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                rank = i + 1  # Rank starts from 1
                break
        
        # If no relevant document, use 0
        if rank is not None:
            mrr_scores.append(1 / rank)
        else:
            mrr_scores.append(0)
    
    return sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0

def calculate_recall(true_dict, pred_dict, k=10):
    recall_scores = []
    for query_id in true_dict:
        relevant_docs = {doc_id for doc_id, is_relevant in true_dict[query_id].items() if is_relevant}
        top_k_results = dict(sorted(pred_dict[query_id].items(), key=lambda item: item[1], reverse=True)[:k])
        retrieved_docs = [doc_id for doc_id, score in top_k_results.items()]

        # Count relevant documents in the top k retrieved documents
        count_retrieval = sum(1 for doc in retrieved_docs if doc in relevant_docs)
        score = count_retrieval / len(relevant_docs) if relevant_docs else 0
        recall_scores.append(score)

    return sum(recall_scores) / len(recall_scores) if recall_scores else 0

def calculate_myrecall(true_dict, pred_dict, k=10):
    recall_scores = []
    for query_id in true_dict:
        relevant_docs = {doc_id for doc_id, is_relevant in true_dict[query_id].items() if is_relevant}
        top_k_results = dict(sorted(pred_dict[query_id].items(), key=lambda item: item[1], reverse=True)[:k])
        retrieved_docs = [doc_id for doc_id, score in top_k_results.items()]

        # Count relevant documents in the top k retrieved documents
        count_retrieval = sum(1 for doc in retrieved_docs if doc in relevant_docs)
        score = count_retrieval / min(k, len(relevant_docs)) if relevant_docs else 0
        recall_scores.append(score)

    return sum(recall_scores) / len(recall_scores) if recall_scores else 0

def calculate_map(true_dict, pred_dict, k=10):
    ap_scores = []
    for query_id in true_dict:
        relevant_docs = {doc_id for doc_id, is_relevant in true_dict[query_id].items() if is_relevant}
        top_k_results = dict(sorted(pred_dict[query_id].items(), key=lambda item: item[1], reverse=True)[:k])
        retrieved_docs = [doc_id for doc_id, score in top_k_results.items()]

        num_relevant_retrieved = 0
        precision_sum = 0
        
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                num_relevant_retrieved += 1
                precision_at_i = num_relevant_retrieved / (i + 1)
                precision_sum += precision_at_i
        
        ap = precision_sum / len(relevant_docs) if relevant_docs else 0
        ap_scores.append(ap)


    return sum(ap_scores) / len(ap_scores) if ap_scores else 0

parser = argparse.ArgumentParser()

# Data
parser.add_argument("--id2doc_path", type=str, default=None, help="test data path")
parser.add_argument("--id2query_path", type=str, default=None, help="test data path")
parser.add_argument("--eval_query2doc_path", type=str, default=None, help="train data path")
parser.add_argument("--query_size", type=int, default=10000000000)
parser.add_argument("--faq_size", type=int, default=10000000000)
parser.add_argument("--exclude_negative_faq", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--batch_size", type=int, default=4)

# Model
parser.add_argument("--use_bert", action="store_true")
parser.add_argument("--model_name_or_path", nargs="*", type=str, default=[])
parser.add_argument("--bert_num_candidate", type=int, default=100)
parser.add_argument("--source_block_size", type=int, default=512)
parser.add_argument("--bert_task_type", type=str, default="classification")

parser.add_argument("--use_second_bert", action="store_true")
parser.add_argument("--second_model_name_or_path", nargs="*", type=str, default=[])
parser.add_argument("--second_bert_num_candidate", type=int, default=100)
parser.add_argument("--second_source_block_size", type=int, default=512)
parser.add_argument("--second_bert_task_type", type=str, default="classification")

#bm25
parser.add_argument("--use_bm25", action="store_true")

args = parser.parse_args()

if __name__ == "__main__":
    set_seed(args.seed)
    
    with open(args.id2doc_path, encoding="utf-8") as f:
        id2doc = json.load(f)
    with open(args.id2query_path, encoding="utf-8") as f:
        id2query=json.load(f)
    with open(args.eval_query2doc_path, encoding="utf-8") as f:
        eval_query2doc=json.load(f)

    all_doc_id = sorted(list(set(id2doc.keys())))
    predictor = ScorePredictor(args, id2doc, id2query)

    start_time = time.time()

    true_dict = {}
    pred_dict = {}
    for query_id, value in tqdm(eval_query2doc.items()):
        positive_doc_id = value["positive_doc_id"]
        bm25_doc_id = value["bm25_doc_id"]

        if args.use_bm25:
            cand_doc_id = bm25_doc_id
        else:
            cand_doc_id = all_doc_id

        y_true = {doc_id:int(doc_id in positive_doc_id) for doc_id in cand_doc_id}
        y_pred = \
            {doc_id:score for doc_id, score in \
                zip(cand_doc_id, predictor.predict(query_id, cand_doc_id))}

        true_dict[query_id] = y_true
        pred_dict[query_id] = y_pred
        # print(y_true)
        # print(y_pred)
        # print()
        
    end_time = time.time()

    mrr = calculate_mrr(true_dict, pred_dict)
    map_score = calculate_map(true_dict, pred_dict)

    recall1 = calculate_recall(true_dict, pred_dict, 1)
    recall3 = calculate_recall(true_dict, pred_dict, 3)
    recall5 = calculate_recall(true_dict, pred_dict, 5)
    recall10 = calculate_recall(true_dict, pred_dict, 10)
    recall20 = calculate_recall(true_dict, pred_dict, 20)
    recall50 = calculate_recall(true_dict, pred_dict, 50)
    recall100 = calculate_recall(true_dict, pred_dict, 100)
    recall200 = calculate_recall(true_dict, pred_dict, 200)

    myrecall1 = calculate_myrecall(true_dict, pred_dict, 1)
    myrecall3 = calculate_myrecall(true_dict, pred_dict, 3)
    myrecall5 = calculate_myrecall(true_dict, pred_dict, 5)
    myrecall10 = calculate_myrecall(true_dict, pred_dict, 10)
    myrecall20 = calculate_myrecall(true_dict, pred_dict, 20)
    myrecall50 = calculate_myrecall(true_dict, pred_dict, 50)
    myrecall100 = calculate_myrecall(true_dict, pred_dict, 100)
    myrecall200 = calculate_myrecall(true_dict, pred_dict, 200)

    print(f"Search time:{end_time-start_time}")
    print(f"MRR@10: {mrr:.4f}")
    print(f"MAP@10: {map_score:.4f}")

    print(f"Recall@1: {recall1:.4f}\t\tMy_recall@1: {myrecall1:.4f}")
    print(f"Recall@3: {recall3:.4f}\t\tMy_recall@3: {myrecall3:.4f}")
    print(f"Recall@5: {recall5:.4f}\t\tMy_recall@5: {myrecall5:.4f}")
    print(f"Recall@10: {recall10:.4f}\t\tMy_recall@10: {myrecall10:.4f}")
    print(f"Recall@20: {recall20:.4f}\t\tMy_recall@20: {myrecall20:.4f}")
    print(f"Recall@50: {recall50:.4f}\t\tMy_recall@50: {myrecall50:.4f}")
    print(f"Recall@100: {recall100:.4f}\t\tMy_recall@100: {myrecall100:.4f}")
    print(f"Recall@200: {recall200:.4f}\t\tMy_recall@200: {myrecall200:.4f}")