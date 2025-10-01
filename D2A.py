
import os, gc, json, math, copy, pickle, warnings
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertModel, BertTokenizer,
    get_linear_schedule_with_warmup
)

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
import random
from transformers import AutoModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(42)

import spacy, langdetect
nlp_en = spacy.load('en_core_web_sm')
nlp_zh = spacy.load('zh_core_web_sm')
# ===========================================================

warnings.filterwarnings("ignore", category=UserWarning, message="Gradients will be None")

######################################################################
# 数据处理
######################################################################

def auto_sentence_split(text):
    
    try:
        lang = langdetect.detect(text)
    except:
        lang = 'en'
    doc = nlp_zh(text) if lang.startswith('zh') else nlp_en(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def load_sentence_labeled_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)
    texts = df['sentences'].tolist()
    doc_labels = df['label'].tolist()
    sent_labels = df['sentence_labels'].tolist()
    labels = []
    for d, s in zip(doc_labels, sent_labels):
        labels.append({'doc_label': d, 'sentence_labels': s})
    return texts, labels

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_sentences=30, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_sentences = max_sentences
        self.max_length = max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label_obj = self.labels[idx]

        if isinstance(label_obj, dict) and 'sentence_labels' in label_obj:
            sentence_labels = label_obj['sentence_labels']
            label = label_obj['doc_label']
        else:
            sentence_labels = [0] * self.max_sentences
            label = int(label_obj)

        sentences = text if isinstance(text, list) else auto_sentence_split(text)
        if len(sentences) == 0:
            sentences = ["这个句子是填充"]
        original_sentences = sentences.copy()

        if len(sentences) > self.max_sentences:
            sentences = sentences[:self.max_sentences]
            sentence_labels = sentence_labels[:self.max_sentences]
            actual_length = self.max_sentences
        else:
            actual_length = len(sentences)
            pad_len = self.max_sentences - actual_length
            sentences += ["这个句子是填充"] * pad_len
            sentence_labels += [-100] * pad_len  # ignore

        encodings = self.tokenizer(
            sentences, padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt'
        )

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0  # 兜底
        if actual_length < self.max_sentences:
            encodings['input_ids'][actual_length:] = pad_id
            encodings['attention_mask'][actual_length:] = 0

        return {
            'input_ids': encodings['input_ids'],               # (N, L)
            'attention_mask': encodings['attention_mask'],     # (N, L)
            'label': torch.tensor(label, dtype=torch.long),
            'actual_length': torch.tensor(actual_length, dtype=torch.long),
            'sentence_labels': torch.tensor(sentence_labels, dtype=torch.long),
            'original_sentences': original_sentences[:actual_length]
        }

def collate_fn(batch):
    input_ids = torch.stack([b['input_ids'] for b in batch])        # (B, N, L)
    attn_mask = torch.stack([b['attention_mask'] for b in batch])   # (B, N, L)
    labels = torch.stack([b['label'] for b in batch])               # (B,)
    actual_lengths = torch.stack([b['actual_length'] for b in batch])
    sentence_labels = torch.stack([b['sentence_labels'] for b in batch])
    original_sentences = [b['original_sentences'] for b in batch]
    return {
        'input_ids': input_ids,
        'attention_mask': attn_mask,
        'labels': labels,
        'actual_lengths': actual_lengths,
        'sentence_labels': sentence_labels,
        'original_sentences': original_sentences
    }

######################################################################
# 模型
######################################################################

FAKE_LABEL = 1

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def _ensure_class_indices(labels, num_classes=2):
    """统一 labels -> [B] 类别索引"""
    if labels.dim() == 1:
        return labels.long()
    if labels.dim() == 2:
        if labels.size(1) == 1:
            return labels.squeeze(1).long()
        if labels.size(1) == num_classes:
            return labels.argmax(dim=1).long()
    return labels.view(labels.size(0)).long()

def _ensure_logits_2d(logits, batch_size, num_classes=2):
    """统一 logits -> [B, C]"""
    if logits.dim() == 2 and logits.size(0) == batch_size and logits.size(1) == num_classes:
        return logits
    if logits.dim() > 2 and logits.size(1) == num_classes:
        return logits.flatten(start_dim=2).mean(dim=2)
    if logits.dim() == 2 and logits.size(0) == num_classes and logits.size(1) == batch_size:
        return logits.transpose(0, 1)
    if logits.dim() == 1:
        if logits.numel() == batch_size * num_classes:
            return logits.view(batch_size, num_classes)
        if logits.numel() == num_classes and batch_size == 1:
            return logits.unsqueeze(0)
    return logits.view(batch_size, -1)[:, :num_classes]




# ---------- model ----------
#原始版本
class DualAttentionModel(nn.Module):
    """
    句级注意力 attn1/attn2 改为“文档条件注意力
    词语（token）分支、三路门控 gate3、各类损失（可配权重）
    """
    def __init__(self, bert_model='bert-base-chinese', num_labels=2,
                 disentangle_weight=0.1, adv_weight=0.0, diversity_weight=0.0,
                 attn_sup_weight=0.2, sentence_cls_weight=1.0, grl_lambda=1.0):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.hidden_size = self.bert.config.hidden_size
        self.num_labels = num_labels

        self.disentangle_weight = float(disentangle_weight)
        self.adv_weight         = float(adv_weight)
        self.diversity_weight   = float(diversity_weight)
        self.attn_sup_weight    = float(attn_sup_weight)
        self.sentence_cls_weight= float(sentence_cls_weight)
        self.grl_lambda         = float(grl_lambda)

        try:
            fake_lbl = FAKE_LABEL  # noqa: F821
        except NameError:
            fake_lbl = 1
        self.register_buffer('fake_label', torch.tensor(fake_lbl, dtype=torch.long))

        # === 文档条件注意力参数 ===
        h2 = self.hidden_size // 2
        self.W_h = nn.Linear(self.hidden_size, h2)
        self.W_q = nn.Linear(self.hidden_size, h2)
        self.v1  = nn.Linear(h2, 1)
        self.v2  = nn.Linear(h2, 1)

        # 文档分类支路
        self.transform2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),  nn.GELU(), nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size//2)
        )
        self.classifier2 = nn.Linear(self.hidden_size//2, num_labels)

        # 对抗分支（可选）：用 weighted1 作为输入
        self.adv_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, num_labels)
        )

        # 句子分类器（监督/加权）
        self.sentence_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, 2)
        )
        self.classifier_sentence_weighted = nn.Linear(self.hidden_size, num_labels)

        # 词语（token）注意力头
        self.attn_word = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Tanh(),
            nn.Linear(self.hidden_size // 2, 1)
        )

        # 三分支门控
        self.gate3 = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 3)
        )

        self._init_weights()

    def _init_weights(self):
        def init_linear(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for blk in [
            self.W_h, self.W_q, self.v1, self.v2,
            self.transform2, self.adv_classifier,
            self.sentence_classifier, self.classifier_sentence_weighted,
            self.attn_word, self.gate3, self.classifier2
        ]:
            blk.apply(init_linear)

    @staticmethod
    def _neg_inf_like(x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:
            return torch.tensor(-1e4, dtype=x.dtype, device=x.device)
        return torch.tensor(torch.finfo(x.dtype).min, dtype=x.dtype, device=x.device)

    def _masked_softmax_lastdim(self, scores: torch.Tensor, valid_mask: torch.Tensor):
        very_neg = -1e9 if scores.dtype == torch.float32 else -1e4
        masked = scores.masked_fill(valid_mask == 0, very_neg)
        masked = masked - masked.max(dim=-1, keepdim=True).values
        exp = torch.exp(masked) * valid_mask
        den = exp.sum(dim=-1, keepdim=True)
        return exp / (den + 1e-12)

    def forward(self, input_ids, attention_mask, actual_lengths,
                labels=None, sentence_labels=None, return_attn=False):

        B, N, L = input_ids.shape
        device = input_ids.device

        # 句级 padding 掩码：True=补齐句
        sentence_mask = torch.arange(N, device=device).unsqueeze(0) >= actual_lengths.unsqueeze(1)  # (B,N)

        input_ids      = input_ids.clone()
        attention_mask = attention_mask.clone()

        pad_id = self.bert.config.pad_token_id
        cls_id = getattr(self.bert.config, "cls_token_id", None)
        if pad_id is None: pad_id = 0
        if cls_id is None: cls_id = 101

        if sentence_mask.any():
            m3 = sentence_mask.unsqueeze(-1).expand(-1, -1, L)
            input_ids = input_ids.masked_fill(m3, pad_id)
            attention_mask = attention_mask.masked_fill(m3, 0)
            b_idx, n_idx = sentence_mask.nonzero(as_tuple=True)
            input_ids[b_idx, n_idx, 0] = cls_id
            attention_mask[b_idx, n_idx, 0] = 1

        flat_input_ids = input_ids.reshape(-1, L)
        flat_attention = attention_mask.reshape(-1, L)
        outputs = self.bert(input_ids=flat_input_ids, attention_mask=flat_attention)

        token_embeds = outputs.last_hidden_state.reshape(B, N, L, self.hidden_size)   # (B,N,L,H)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].reshape(B, N, self.hidden_size)  # (B,N,H)
        sentence_embeds = cls_embeddings

        # ===== 文档条件注意力 =====
        valid = (~sentence_mask).float().unsqueeze(-1)                          # (B,N,1)
        denom = valid.sum(dim=1).clamp_min(1.0)                                 # (B,1)
        q_doc = (sentence_embeds * valid).sum(dim=1) / denom                    # (B,H)

        h_proj = self.W_h(sentence_embeds)                                      # (B,N,H/2)
        q_proj = self.W_q(q_doc).unsqueeze(1)                                   # (B,1,H/2)

        attn_scores1 = self.v1(torch.tanh(h_proj + q_proj)).squeeze(-1)         # (B,N)
        attn_scores2 = self.v2(torch.tanh(h_proj + q_proj)).squeeze(-1)         # (B,N)

        neg_inf = self._neg_inf_like(attn_scores1)
        attn_scores1 = attn_scores1.masked_fill(sentence_mask, neg_inf)
        attn_scores2 = attn_scores2.masked_fill(sentence_mask, neg_inf)

        attn_weights1 = F.softmax(attn_scores1, dim=1)                          # (B,N)
        attn_weights2 = F.softmax(attn_scores2, dim=1)                          # (B,N)

        weighted1 = torch.sum(sentence_embeds * attn_weights1.unsqueeze(-1), dim=1)  # (B,H)
        weighted2 = torch.sum(sentence_embeds * attn_weights2.unsqueeze(-1), dim=1)  # (B,H)

        # 句子分类 + 假句权重聚合
        sentence_logits = self.sentence_classifier(sentence_embeds)             # (B,N,2)
        sentence_probs = F.softmax(sentence_logits, dim=-1)                     # (B,N,2)
        sentence_fake_probs = sentence_probs[..., 1]                            # (B,N)
        sentence_fake_probs = sentence_fake_probs.masked_fill(sentence_mask, self._neg_inf_like(sentence_fake_probs))
        sent_weight = F.softmax(sentence_fake_probs, dim=1)                     # (B,N)

        weighted_sentence_repr = torch.sum(sentence_embeds * sent_weight.unsqueeze(-1), dim=1)  # (B,H)
        logits_sentence_weighted = self.classifier_sentence_weighted(weighted_sentence_repr)

        # 词级注意力（稳定处理）
        tok_valid = (attention_mask != 0)                                       # (B,N,L)
        token_scores = self.attn_word(token_embeds).squeeze(-1)                 # (B,N,L)
        token_scores = token_scores.masked_fill(~tok_valid, self._neg_inf_like(token_scores))
        token_weights = F.softmax(token_scores, dim=-1)                         # (B,N,L)
        all_invalid = (~tok_valid).all(dim=-1, keepdim=True)                    # (B,N,1)
        token_weights = torch.where(all_invalid, torch.zeros_like(token_weights), token_weights)
        word_repr_per_sentence = torch.sum(token_weights.unsqueeze(-1) * token_embeds, dim=2)   # (B,N,H)
        word_doc_repr = torch.sum(word_repr_per_sentence * sent_weight.unsqueeze(-1), dim=1)    # (B,H)

        # 三分支门控
        gate_input3 = torch.cat([weighted2, weighted_sentence_repr, word_doc_repr], dim=-1)      # (B,3H)
        gate_logits = self.gate3(gate_input3)
        alphas = F.softmax(gate_logits, dim=-1)                                                  # (B,3)
        doc_repr = (
            alphas[:, 0:1] * weighted2 +
            alphas[:, 1:2] * weighted_sentence_repr +
            alphas[:, 2:3] * word_doc_repr
        )                                                                                        # (B,H)

        # 文档分类
        branch2 = self.transform2(doc_repr)
        logits2 = self.classifier2(branch2)                                                      # (B,C)
        sentence_preds = torch.argmax(sentence_logits, dim=-1)                                   # (B,N)

        if labels is None:
            return logits2, attn_weights1, attn_weights2, sentence_preds

        # ===== 损失 =====
        labels = _ensure_class_indices(labels, self.num_labels)
        logits2 = _ensure_logits_2d(logits2, B, self.num_labels)

        ce_doc = nn.CrossEntropyLoss()
        veracity_loss = ce_doc(logits2, labels)

        sent_ce = nn.CrossEntropyLoss(ignore_index=-100)
        if sentence_labels is not None:
            sentence_cls_loss = sent_ce(
                sentence_logits.reshape(-1, 2), sentence_labels.reshape(-1)
            )
        else:
            sentence_cls_loss = torch.tensor(0.0, device=device)

        # 对抗分支（仅对假类样本回传反向梯度）
        adv_loss = torch.tensor(0.0, device=device)
        if self.adv_weight > 0.0:
            is_fake = (labels == self.fake_label).to(device)
            adv_input = weighted1.clone()
            if is_fake.any():
                nf_idx = (~is_fake).nonzero(as_tuple=False).squeeze(-1)
                if nf_idx.numel() > 0:
                    adv_input[nf_idx] = adv_input[nf_idx].detach()
                adv_input = GradientReversal.apply(adv_input, self.grl_lambda)
            else:
                adv_input = adv_input.detach()
            adv_logits = self.adv_classifier(adv_input)
            adv_logits = _ensure_logits_2d(adv_logits, B, self.num_labels)
            adv_loss = F.cross_entropy(adv_logits, labels)

        # 解耦（句标可用时）
        disentangle_loss = torch.tensor(0.0, device=device)
        if self.disentangle_weight > 0.0 and sentence_labels is not None:
            valid_sent = (sentence_labels != -100) & (~sentence_mask)
            if valid_sent.any():
                pos_mask = (sentence_labels == 0) & valid_sent
                neg_mask = (sentence_labels == 1) & valid_sent
                pos_c = pos_mask.sum(dim=1).clamp(min=1).float()
                neg_c = neg_mask.sum(dim=1).clamp(min=1).float()
                a1_pos = (attn_weights1 * pos_mask.float()).sum(dim=1) / pos_c
                a1_neg = (attn_weights1 * neg_mask.float()).sum(dim=1) / neg_c
                a2_pos = (attn_weights2 * pos_mask.float()).sum(dim=1) / pos_c
                a2_neg = (attn_weights2 * neg_mask.float()).sum(dim=1) / neg_c
                gap1 = a1_pos - a1_neg
                gap2 = a2_neg - a2_pos
                disentangle_loss = (gap1 - gap2).pow(2).mean()

        # 多样性（避免两头塌缩）
        diversity_loss = torch.tensor(0.0, device=device)
        extra_div = torch.tensor(0.0, device=device)
        if self.diversity_weight > 0.0:
            n1 = F.normalize(weighted1, p=2, dim=1)
            n2 = F.normalize(weighted2, p=2, dim=1)
            cos = torch.sum(n1 * n2, dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            diversity_loss = -cos.mean()

        # 注意力监督（句/词）
        attn_sup_loss = torch.tensor(0.0, device=device)
        token_attn_sup_loss = torch.tensor(0.0, device=device)
        if self.attn_sup_weight > 0.0 and sentence_labels is not None:
            valid = (sentence_labels != -100) & (~sentence_mask)
            if valid.any():
                eps = 1e-12
                real_mask = (sentence_labels == 0) & valid
                fake_mask = (sentence_labels == 1) & valid
                terms = []
                if (real_mask.sum(dim=1) > 0).any():
                    a1 = (attn_weights1 * valid.float())
                    a1 = a1 / (a1.sum(dim=1, keepdim=True) + eps)
                    t1 = real_mask.float()
                    t1 = t1 / (t1.sum(dim=1, keepdim=True) + eps)
                    terms.append(F.kl_div(torch.log(a1 + eps), t1, reduction='batchmean'))
                if (fake_mask.sum(dim=1) > 0).any():
                    a2 = (attn_weights2 * valid.float())
                    a2 = a2 / (a2.sum(dim=1, keepdim=True) + eps)
                    t2 = fake_mask.float()
                    t2 = t2 / (t2.sum(dim=1, keepdim=True) + eps)
                    terms.append(F.kl_div(torch.log(a2 + eps), t2, reduction='batchmean'))
                if len(terms) > 0:
                    attn_sup_loss = sum(terms) / len(terms)

            valid_sent = (sentence_labels != -100)
            if valid_sent.any():
                eps = 1e-12
                tw = (token_weights + eps)
                ent = -(tw * torch.log(tw)).sum(dim=-1)   # (B,N)
                fake_mask = (sentence_labels == 1) & valid_sent
                real_mask = (sentence_labels == 0) & valid_sent
                terms_tok = []
                if fake_mask.any():
                    terms_tok.append(ent[fake_mask].mean())    # 假句：熵小
                if real_mask.any():
                    terms_tok.append((-ent[real_mask]).mean()) # 真句：熵大
                if len(terms_tok) > 0:
                    token_attn_sup_loss = sum(terms_tok) / len(terms_tok)

        total_loss = veracity_loss + self.sentence_cls_weight * sentence_cls_loss
        if self.adv_weight > 0.0:         total_loss += self.adv_weight * adv_loss
        if self.disentangle_weight > 0.0: total_loss += self.disentangle_weight * disentangle_loss
        if self.diversity_weight > 0.0:   total_loss += self.diversity_weight * (diversity_loss + extra_div) / 2.0
        if self.attn_sup_weight > 0.0:    total_loss += self.attn_sup_weight * (attn_sup_loss + token_attn_sup_loss) / 2.0

        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            total_loss = veracity_loss  # 兜底

        out = {
            "logits": logits2,
            "sentence_weighted_logits": logits_sentence_weighted,
            "veracity_loss": float(veracity_loss.item()),
            "sentence_cls_loss": float(sentence_cls_loss.item()),
            "disentangle_loss": float(disentangle_loss.item()),
            "diversity_loss": float((diversity_loss + extra_div).item() if self.diversity_weight>0 else diversity_loss.item()),
            "adv_loss": float(adv_loss.item()),
            "attn_supervision_loss": float((attn_sup_loss + token_attn_sup_loss).item() if self.attn_sup_weight>0 else attn_sup_loss.item())
        }
        if return_attn:
            return total_loss, out, attn_weights1, attn_weights2, sentence_preds
        else:
            return total_loss, out



######################################################################
# 训练工具（不冻结 BERT；分组 LR）
######################################################################

def enable_grad_checkpointing(model):
    if hasattr(model, "bert") and hasattr(model.bert, "gradient_checkpointing_enable"):
        model.bert.gradient_checkpointing_enable()

def set_dropout(model, p=0.3):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = p

@torch.no_grad()
def eval_accuracy_only(model, dataloader, device):
    model.eval(); model.to(device)
    total = correct = 0
    for batch in dataloader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        actual_lengths = batch['actual_lengths'].to(device)
        labels         = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, actual_lengths=actual_lengths, labels=None)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.numel()
        del input_ids, attention_mask, actual_lengths, labels, outputs, logits
        torch.cuda.empty_cache()
    return correct / total if total > 0 else 0.0

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1-self.decay)
    @torch.no_grad()
    def store(self, model):
        self.backup = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}
    @torch.no_grad()
    def copy_to(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.data.copy_(self.shadow[n].data)
    @torch.no_grad()
    def restore(self, model):
        if hasattr(self, "backup"):
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.backup:
                    p.data.copy_(self.backup[n].data)
            self.backup = {}

class EarlyStopping:
    def __init__(self, patience=2, mode='max', min_delta=1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = -float('inf') if mode=='max' else float('inf')
        self.num_bad = 0
    def step(self, metric):
        improved = (metric > self.best + self.min_delta) if self.mode=='max' else (metric < self.best - self.min_delta)
        if improved:
            self.best = metric
            self.num_bad = 0
            return True
        else:
            self.num_bad += 1
            return False
    def should_stop(self): return self.num_bad >= self.patience

def build_param_groups(model, lr_bert=2e-5, lr_heads=1e-4, weight_decay=0.01):
    bert_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if n.startswith("bert."):
            bert_params.append(p)
        else:
            head_params.append(p)
    return [
        {"params": bert_params, "lr": lr_bert, "weight_decay": weight_decay},
        {"params": head_params, "lr": lr_heads, "weight_decay": weight_decay},
    ]



def train_dual_attention_fast(
    model, train_dataset, val_dataset, device='cuda',
    num_epochs=6, batch_size=4,
    lr_bert=2e-5, lr_heads=1e-4, weight_decay=0.01,
    warmup_ratio=0.1, accumulate_steps=2,
    use_grad_ckpt=True, set_dropout_to=0.3,
    ema_decay=0.999, early_stop_patience=2,
    save_dir="fast_ckpts_0904copy", save_prefix="best_0904"
):

    os.makedirs(save_dir, exist_ok=True)
    model.to(device)

    if use_grad_ckpt:
        enable_grad_checkpointing(model)
    if set_dropout_to is not None:
        set_dropout(model, set_dropout_to)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=max(8, batch_size),
                              collate_fn=collate_fn, pin_memory=True)

    optimizer = torch.optim.AdamW(build_param_groups(model, lr_bert, lr_heads, weight_decay))

    total_update_steps = num_epochs * math.ceil(len(train_loader) / max(1, accumulate_steps))
    warmup_steps = int(warmup_ratio * max(1, total_update_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)

    scaler = torch.cuda.amp.GradScaler()
    ema = EMA(model, decay=ema_decay) if (ema_decay is not None and 0.0 < ema_decay < 1.0) else None
    early = EarlyStopping(patience=early_stop_patience, mode='max')

    best_acc = -1.0
    best_path = os.path.join(save_dir, f"{save_prefix}_model.pt")
    best_meta = os.path.join(save_dir, f"{save_prefix}_meta.json")

    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        running = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for step, batch in enumerate(pbar):
            input_ids       = batch['input_ids'].to(device, non_blocking=True)
            attention_mask  = batch['attention_mask'].to(device, non_blocking=True)
            actual_lengths  = batch['actual_lengths'].to(device, non_blocking=True)
            labels          = batch['labels'].to(device, non_blocking=True)
            sentence_labels = batch['sentence_labels'].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    actual_lengths=actual_lengths, labels=labels,
                    sentence_labels=sentence_labels
                )
                loss = outputs[0] if isinstance(outputs, tuple) else outputs

            scaler.scale(loss).backward()

            if (step + 1) % accumulate_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                if ema: ema.update(model)

            running += float(loss.item())
            pbar.set_postfix({"loss": f"{running/(step+1):.4f}"})

            del input_ids, attention_mask, actual_lengths, labels, sentence_labels, outputs, loss
            torch.cuda.empty_cache()

        if (step + 1) % accumulate_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if ema: ema.update(model)

        if ema:
            ema.store(model)                 
            ema.copy_to(model)               
            val_acc = eval_accuracy_only(model, val_loader, device)
            if early.step(val_acc):
                best_acc = val_acc
                torch.save(model.state_dict(), best_path)  
                with open(best_meta, "w") as f:
                    json.dump({"epoch": epoch+1, "val_acc": float(val_acc)}, f,
                              indent=2, ensure_ascii=False)
                print(f"  ↳ New best! (EMA) Saved to {best_path}")
            ema.restore(model)               
        else:
            val_acc = eval_accuracy_only(model, val_loader, device)
            if early.step(val_acc):
                best_acc = val_acc
                torch.save(model.state_dict(), best_path)
                with open(best_meta, "w") as f:
                    json.dump({"epoch": epoch+1, "val_acc": float(val_acc)}, f,
                              indent=2, ensure_ascii=False)
                print(f"  ↳ New best! Saved to {best_path}")

        print(f"[Epoch {epoch+1}] train_loss={running/max(1,len(train_loader)):.4f}  val_acc={val_acc:.4f}")

        if early.should_stop():
            print("Early stopped.")
            break

        gc.collect(); torch.cuda.empty_cache()

    print(f"\n[Training Finished] Best Val Acc = {best_acc:.4f}  @ {best_path}")
    return model, best_path, best_acc


@torch.no_grad()
def eval_metrics(model, dataloader, device):
    model.eval(); model.to(device)
    all_preds, all_labels = [], []
    for batch in dataloader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        actual_lengths = batch['actual_lengths'].to(device)
        labels         = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        actual_lengths=actual_lengths, labels=None)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

        del input_ids, attention_mask, actual_lengths, labels, outputs, logits, preds
        torch.cuda.empty_cache()

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec  = recall_score(all_labels,   all_preds, average='macro', zero_division=0)
    f1   = f1_score(all_labels,       all_preds, average='macro', zero_division=0)
    return acc, prec, rec, f1

@torch.no_grad()
def evaluate_sentence_classifier(model, dataloader, device='cuda', save_dir=None):
    """
    句子级分类器评估 + 注意力统计
    """
    import os, json, csv
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    model.to(device); model.eval()
    all_sent_labels, all_sent_preds = [], []
    all_sent_attn1, all_sent_attn2 = [], []
    details = []

    for batch in tqdm(dataloader, desc="Sentence-level Eval"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        actual_lengths = batch['actual_lengths'].to(device)
        sentence_labels = batch['sentence_labels'].to(device)
        sentences = batch['original_sentences']

        _, attn1, attn2, sent_preds = model(
            input_ids=input_ids, attention_mask=attention_mask, actual_lengths=actual_lengths
        )

        B = sentence_labels.size(0)
        for i in range(B):
            L = actual_lengths[i].item()
            for j in range(L):
                y = int(sentence_labels[i, j].item())
                if y == -100:
                    continue
                pred = int(sent_preds[i, j].item())
                a1 = float(attn1[i, j].item())
                a2 = float(attn2[i, j].item())
                all_sent_labels.append(y)
                all_sent_preds.append(pred)
                all_sent_attn1.append(a1)
                all_sent_attn2.append(a2)
                details.append({
                    "doc_idx": None,
                    "sent_idx": None,
                    "sentence": sentences[i][j] if j < len(sentences[i]) else "",
                    "true_label": y,
                    "pred_label": pred,
                    "attn1": a1,
                    "attn2": a2,
                    "mean_attn": (a1 + a2) / 2.0
                })

    acc = accuracy_score(all_sent_labels, all_sent_preds) if all_sent_labels else 0.0
    prec = precision_score(all_sent_labels, all_sent_preds, average='binary', pos_label=1, zero_division=0) if all_sent_labels else 0.0
    rec  = recall_score(all_sent_labels,   all_sent_preds, average='binary', pos_label=1, zero_division=0) if all_sent_labels else 0.0
    f1   = f1_score(all_sent_labels,       all_sent_preds, average='binary', pos_label=1, zero_division=0) if all_sent_labels else 0.0

    print("\n=== Sentence Classifier Performance ===")
    print(f"Accuracy: {acc:.4f}  Precision(Fake): {prec:.4f}  Recall(Fake): {rec:.4f}  F1(Fake): {f1:.4f}")
    print(classification_report(all_sent_labels, all_sent_preds, target_names=["Real", "Fake"], zero_division=0))

    def _stats(vals):
        if len(vals) == 0: return {"n": 0, "mean": 0, "std": 0, "median": 0, "p10": 0, "p90": 0, "min": 0, "max": 0}
        arr = np.array(vals, dtype=np.float64)
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    mean_attn = [(a1 + a2) / 2.0 for a1, a2 in zip(all_sent_attn1, all_sent_attn2)]
    real_idx = [k for k, y in enumerate(all_sent_labels) if y == 0]
    fake_idx = [k for k, y in enumerate(all_sent_labels) if y == 1]

    stats = {
        "attn1": {
            "Real": _stats([all_sent_attn1[k] for k in real_idx]),
            "Fake": _stats([all_sent_attn1[k] for k in fake_idx]),
        },
        "attn2": {
            "Real": _stats([all_sent_attn2[k] for k in real_idx]),
            "Fake": _stats([all_sent_attn2[k] for k in fake_idx]),
        },
        "mean_attn": {
            "Real": _stats([mean_attn[k] for k in real_idx]),
            "Fake": _stats([mean_attn[k] for k in fake_idx]),
        }
    }

    try:
        auc = roc_auc_score(all_sent_labels, mean_attn)
        print(f"AUC (mean_attn as score for Fake)= {auc:.4f}")
    except Exception:
        auc = None
        print("AUC 无法计算（可能只有单一类别）")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "sentence_eval_details.json"), "w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)

    return {
        'labels': all_sent_labels,
        'preds': all_sent_preds,
        'attn1': all_sent_attn1,
        'attn2': all_sent_attn2,
        'mean_attn': mean_attn,
        'metrics': {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc_mean_attn': auc},
        'attn_stats': stats,
        'details': details
    }

@torch.no_grad()
def visualize_attention(model, dataloader, device='cuda'):
    model.eval(); model.to(device)
    results = []
    for batch in tqdm(dataloader, desc="Visualizing attention"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        actual_lengths = batch['actual_lengths'].to(device)
        sentence_labels = batch['sentence_labels'].to(device)
        sentences = batch['original_sentences']

        _, attn1, attn2, _ = model(input_ids=input_ids, attention_mask=attention_mask, actual_lengths=actual_lengths)

        for i in range(len(sentences)):
            L = actual_lengths[i].item()
            results.append({
                "sentences": sentences[i][:L],
                "attention_weights1": attn1[i][:L].cpu().tolist(),
                "attention_weights2": attn2[i][:L].cpu().tolist(),
                "sentence_labels": sentence_labels[i][:L].cpu().tolist()
            })
    return results

def _extract_scores_labels(attention_results):
    import numpy as np
    scores, labels, details = [], [], []
    for doc_idx, doc in enumerate(attention_results):
        a1 = doc.get('attention_weights1', [])
        a2 = doc.get('attention_weights2', [])
        sents = doc.get('sentences', [])
        y = doc.get('sentence_labels', [])
        if not a1 or not a2 or not sents or not y:
            continue
        valid = [i for i, yy in enumerate(y) if yy != -100]
        for i in valid:
            m = (float(a1[i]) + float(a2[i])) / 2.0
            scores.append(m)
            labels.append(int(y[i]))
            details.append({
                'doc_idx': doc_idx, 'sent_idx': i,
                'sentence': sents[i],
                'true_label': int(y[i]),
                'score_mean_attn': m
            })
    return np.asarray(scores, dtype=float), np.asarray(labels, dtype=int), details

def find_best_threshold_on_validation(scores, labels, optimize_by='f1', grid_size=200):
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    if scores.size == 0:
        return None, None

    s_min, s_max = float(scores.min()), float(scores.max())
    if s_min == s_max:
        thr_grid = np.array([s_min], dtype=np.float64)
    else:
        base = np.linspace(s_min, s_max, num=grid_size, dtype=np.float64)
        qs = np.quantile(scores, q=np.linspace(0.01, 0.99, num=99))
        thr_grid = np.unique(np.concatenate([base, qs]))

    def eval_thr(th):
        pred = (scores >= th).astype(int)
        acc = accuracy_score(labels, pred)
        prec = precision_score(labels, pred, zero_division=0)
        rec  = recall_score(labels, pred, zero_division=0)
        f1   = f1_score(labels, pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0,1]).ravel()
        tpr = rec
        fpr = fp / max(1, (fp + tn))
        youden = tpr - fpr
        bacc = 0.5 * ((tp / max(1, tp+fn)) + (tn / max(1, tn+fp)))
        return {'thr': th, 'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
                'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'youden': youden, 'bacc': bacc}

    key = {'f1':'f1', 'youden':'youden', 'balanced_acc':'bacc'}[optimize_by.lower()]
    best = max((eval_thr(th) for th in thr_grid), key=lambda x: x[key])

    print(f"[VAL] optimize_by={optimize_by}  best_thr={best['thr']:.6f}  "
          f"acc={best['acc']:.4f} prec={best['prec']:.4f} rec={best['rec']:.4f} f1={best['f1']:.4f}")
    return float(best['thr']), best

def evaluate_with_threshold(attention_results, threshold, show_top_n=5, save_dir=None):
    import os, csv, json
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    scores, labels, details = _extract_scores_labels(attention_results)
    if scores.size == 0:
        print("Warning: no valid sentences in test attention_results.")
        return None

    preds = (scores >= threshold).astype(int)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()

    buckets = {'tp':[], 'fp':[], 'fn':[], 'tn':[]}
    for rec_i, p in zip(details, preds):
        y = rec_i['true_label']
        out = {**rec_i, 'pred_label': int(p)}
        if y==1 and p==1: buckets['tp'].append(out)
        elif y==0 and p==1: buckets['fp'].append(out)
        elif y==1 and p==0: buckets['fn'].append(out)
        else: buckets['tn'].append(out)

    print("\n" + "="*50)
    print(f"Test with fixed threshold (thr={threshold:.6f})")
    print("="*50)
    print(f"Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    print(f"Confusion Matrix -> TP:{tp} FP:{fp} FN:{fn} TN:{tn}")

    def _stats(vals):
        if len(vals)==0: return {"n":0,"mean":0,"std":0,"median":0,"p10":0,"p90":0,"min":0,"max":0}
        arr = np.asarray(vals, dtype=float)
        return {
            "n": int(arr.size), "mean": float(arr.mean()), "std": float(arr.std(ddof=0)),
            "median": float(np.median(arr)), "p10": float(np.percentile(arr,10)),
            "p90": float(np.percentile(arr,90)), "min": float(arr.min()), "max": float(arr.max())
        }
    bucket_stats = {k: _stats([r['score_mean_attn'] for r in v]) for k, v in buckets.items()}

    return {
        'threshold': float(threshold),
        'metrics': {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
                    'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'bucket_stats': bucket_stats,
        'buckets': buckets,
        'details': details
    }



######################################################################
# main: 多随机种子重复训练 & 平均结果
######################################################################
if __name__ == "__main__":
    DO_SENT_EVAL = False          # True 则执行 evaluate_sentence_classifier（句级分类器）
    DO_ATTN_THR = False           # True 则执行 mean_attn 阈值法（val 找阈后在 test 评估）

    # ====== 全局配置（不变）======
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据路径
    train_path = "./fake_dataset_v3/train_sentence_labeled.pkl"
    val_path   = "./fake_dataset_v3/val_sentence_labeled.pkl"
    test_path  = "./fake_dataset_v3/test_sentence_labeled.pkl"

    # 载入数据（一次即可）
    train_texts, train_labels = load_sentence_labeled_data(train_path)
    val_texts,   val_labels   = load_sentence_labeled_data(val_path)
    test_texts,  test_labels  = load_sentence_labeled_data(test_path)

    # 统一 Dataset（一次即可）
    max_sentences, max_length = 60, 170
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer, max_sentences=max_sentences, max_length=max_length)
    val_dataset   = NewsDataset(val_texts,   val_labels,   tokenizer, max_sentences=max_sentences, max_length=max_length)
    test_dataset  = NewsDataset(test_texts,  test_labels,  tokenizer, max_sentences=max_sentences, max_length=max_length)

    # 评测 DataLoader（一次即可）
    val_loader  = DataLoader(val_dataset,  batch_size=16, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn, pin_memory=True)

    model_kwargs = dict(
        bert_model='bert-base-chinese',
        num_labels=2,
        disentangle_weight=0.1,
        adv_weight=0.03,
        diversity_weight=0.0,
        attn_sup_weight=0.2,
        sentence_cls_weight=1.0,
        grl_lambda=1.0
    )

    train_kwargs = dict(
        num_epochs=20, batch_size=4,
        lr_bert=2e-5, lr_heads=5e-4, weight_decay=0.05,
        warmup_ratio=0.05, accumulate_steps=4,
        use_grad_ckpt=True, set_dropout_to=0.3,
        ema_decay=0.999, early_stop_patience=5,
        save_dir="fast_ckpts_0904copy"
    )  #warmup_ratio=0.06

    # ====== 10 个随机种子 ======
    SEEDS = [42,2021,2025,3,4,5,6,7,8,9]

    # 结果收集
    run_results = []   # 每个 run 的字典
    doc_metrics_mat = []   # [[acc, prec, rec, f1], ...] 用于最后取均值
    val_best_list = []     # 验证集 best acc 列表
    thr_f1_list   = []     # 如果做阈值法，这里记录 test F1

    os.makedirs("multi_seed_results", exist_ok=True)

    for seed in SEEDS:
        print("\n" + "="*70)
        print(f"===> SEED = {seed}")
        print("="*70)

        set_seed(seed)

        model = DualAttentionModel(**model_kwargs)
        save_prefix = f"best_s{seed}"
        model, best_path, best_val_acc = train_dual_attention_fast(
            model, train_dataset, val_dataset, device=device,
            save_prefix=save_prefix, **train_kwargs
        )

        best_model = DualAttentionModel(**model_kwargs)
        best_model.load_state_dict(torch.load(best_path, map_location=device))
        best_model.to(device)

        # 文档级评估（测试集）
        acc, prec, rec, f1 = eval_metrics(best_model, test_loader, device)
        print(f"\n[Seed {seed}]  Best@VAL={best_val_acc:.4f}  TEST -> acc={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}")

        # ===== 句级评估 =====
        sent_metrics = None
        if DO_SENT_EVAL:
            sent_results = evaluate_sentence_classifier(best_model, test_loader, device=device, save_dir=None)
            sent_metrics = sent_results.get('metrics', None)

        # ===== mean_attn 阈值法（先在 VAL 找阈，再 TEST 评估）=====
        thr_report = None
        if DO_ATTN_THR:
            attn_val = visualize_attention(best_model, val_loader, device=device)
            val_scores, val_labels_attn, _ = _extract_scores_labels(attn_val)
            best_thr, best_val_stats = find_best_threshold_on_validation(
                val_scores, val_labels_attn, optimize_by='f1', grid_size=200
            )

            attn_test = visualize_attention(best_model, test_loader, device=device)
            test_fixed = evaluate_with_threshold(
                attn_test, threshold=best_thr, show_top_n=5, save_dir=None
            )
            thr_report = {
                'best_thr': best_thr,
                'val_stats': best_val_stats,
                'test_metrics': test_fixed['metrics'] if test_fixed else None
            }
            if test_fixed and test_fixed.get('metrics'):
                thr_f1_list.append(test_fixed['metrics']['f1'])

        run_rec = {
            'seed': seed,
            'best_val_acc': float(best_val_acc),
            'test_doc_metrics': {'acc': float(acc), 'precision': float(prec), 'recall': float(rec), 'f1': float(f1)},
            'sent_metrics': sent_metrics,
            'attn_threshold_report': thr_report,
            'best_path': best_path
        }
        run_results.append(run_rec)
        val_best_list.append(float(best_val_acc))
        doc_metrics_mat.append([float(acc), float(prec), float(rec), float(f1)])

        with open(os.path.join("multi_seed_results", f"seed_{seed}_result.json"), "w", encoding="utf-8") as f:
            json.dump(run_rec, f, ensure_ascii=False, indent=2)

        del best_model, model
        gc.collect(); torch.cuda.empty_cache()

    # ===== 统计 10 次的均值/方差 =====
    import numpy as np
    doc_metrics_mat = np.asarray(doc_metrics_mat, dtype=np.float64)
    doc_mean = doc_metrics_mat.mean(axis=0)
    doc_std  = doc_metrics_mat.std(axis=0, ddof=0)

    avg_best_val = float(np.mean(val_best_list))
    std_best_val = float(np.std(val_best_list, ddof=0))

    summary = {
        'seeds': SEEDS,
        'val_best_acc_list': val_best_list,
        'val_best_acc_mean': avg_best_val,
        'val_best_acc_std': std_best_val,
        'test_doc_metrics_mean': {'acc': float(doc_mean[0]), 'precision': float(doc_mean[1]), 'recall': float(doc_mean[2]), 'f1': float(doc_mean[3])},
        'test_doc_metrics_std':  {'acc': float(doc_std[0]),  'precision': float(doc_std[1]),  'recall': float(doc_std[2]),  'f1': float(doc_std[3])},
        'attn_thr_test_f1_mean': (float(np.mean(thr_f1_list)) if len(thr_f1_list)>0 else None),
        'runs': run_results
    }

    print("\n" + "="*70)
    print("===> Avg over 10 runs (基于各自 VAL 最优权重在 TEST 上评估)")
    print("="*70)
    print(f"VAL best acc: mean={avg_best_val:.4f}  std={std_best_val:.4f}")
    print("TEST (doc-level) mean ± std:")
    print(f"  acc={doc_mean[0]:.4f} ± {doc_std[0]:.4f}")
    print(f"  precision={doc_mean[1]:.4f} ± {doc_std[1]:.4f}")
    print(f"  recall={doc_mean[2]:.4f} ± {doc_std[2]:.4f}")
    print(f"  f1={doc_mean[3]:.4f} ± {doc_std[3]:.4f}")
    if DO_ATTN_THR and len(thr_f1_list)>0:
        print(f"  mean_attn 阈值法 TEST F1 平均: {np.mean(thr_f1_list):.4f}")

    with open(os.path.join("multi_seed_results", "summary_10seeds.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 也存一个简洁 CSV
    try:
        import csv
        with open(os.path.join("multi_seed_results", "per_seed_test_doc_metrics.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["seed", "val_best_acc", "test_acc", "test_precision", "test_recall", "test_f1"])
            for r in run_results:
                m = r["test_doc_metrics"]
                w.writerow([r["seed"], r["best_val_acc"], m["acc"], m["precision"], m["recall"], m["f1"]])
            w.writerow([])
            w.writerow(["MEAN", avg_best_val, doc_mean[0], doc_mean[1], doc_mean[2], doc_mean[3]])
            w.writerow(["STD",  std_best_val, doc_std[0],  doc_std[1],  doc_std[2],  doc_std[3]])
    except Exception as e:
        print(f"写 CSV 失败: {e}")

    print("\nDone.")
