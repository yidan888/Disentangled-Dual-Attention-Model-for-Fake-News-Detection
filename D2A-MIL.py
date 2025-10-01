
import os, gc, json, math, pickle, warnings, random
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
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)


NO_SENT_LABELS = True  # ★★ 没有句子级标签时设为 True

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

# ----------- 分句依赖（可降级） -----------
try:
    import spacy, langdetect
    try: nlp_en = spacy.load('en_core_web_sm')
    except: nlp_en = None
    try: nlp_zh = spacy.load('zh_core_web_sm')
    except: nlp_zh = None
    HAVE_LANGDETECT = True
except:
    nlp_en = nlp_zh = None
    HAVE_LANGDETECT = False

warnings.filterwarnings("ignore", category=UserWarning, message="Gradients will be None")

######################################################################
# 数据处理
######################################################################
def _naive_split(text: str):
    seps = ['。','！','？','.','!','?','\n']
    buf, sents = '', []
    for ch in text:
        buf += ch
        if ch in seps:
            ss = buf.strip()
            if ss: sents.append(ss)
            buf = ''
    if buf.strip(): sents.append(buf.strip())
    return sents if sents else [text.strip() or "这个句子是填充"]

def auto_sentence_split(text):
    try:
        if HAVE_LANGDETECT:
            try: lang = langdetect.detect(text)
            except: lang = 'en'
            if lang.startswith('zh') and nlp_zh is not None:
                doc = nlp_zh(text)
            elif nlp_en is not None:
                doc = nlp_en(text)
            else:
                return _naive_split(text)
            return [s.text.strip() for s in doc.sents if s.text.strip()] or _naive_split(text)
        else:
            return _naive_split(text)
    except:
        return _naive_split(text)

def load_sentence_labeled_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)
    if 'sentences' in df:
        texts = df['sentences'].tolist()
    elif 'text' in df:
        texts = [auto_sentence_split(t) for t in df['text'].tolist()]
    else:
        raise ValueError("PKL 缺少 'sentences' 或 'text' 列")
    if 'label' not in df:
        raise ValueError("PKL 缺少 'label' 列")
    doc_labels = df['label'].tolist()

    if 'sentence_labels' in df and not NO_SENT_LABELS:
        sent_labels = df['sentence_labels'].tolist()
        labels = [{'doc_label': int(d), 'sentence_labels': s} for d, s in zip(doc_labels, sent_labels)]
        return texts, labels
    else:
        return texts, [int(d) for d in doc_labels]



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
            pad_id = 0  
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
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'labels': torch.stack([b['label'] for b in batch]),
        'actual_lengths': torch.stack([b['actual_length'] for b in batch]),
        'sentence_labels': torch.stack([b['sentence_labels'] for b in batch]),
        'original_sentences': [b['original_sentences'] for b in batch]
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
    if labels.dim() == 1: return labels.long()
    if labels.dim() == 2:
        if labels.size(1) == 1: return labels.squeeze(1).long()
        if labels.size(1) == num_classes: return labels.argmax(dim=1).long()
    return labels.view(labels.size(0)).long()

def _ensure_logits_2d(logits, batch_size, num_classes=2):
    if logits.dim()==2 and logits.size(0)==batch_size and logits.size(1)==num_classes: return logits
    if logits.dim()>2 and logits.size(1)==num_classes: return logits.flatten(start_dim=2).mean(dim=2)
    if logits.dim()==2 and logits.size(0)==num_classes and logits.size(1)==batch_size: return logits.transpose(0,1)
    if logits.dim()==1:
        if logits.numel()==batch_size*num_classes: return logits.view(batch_size, num_classes)
        if logits.numel()==num_classes and batch_size==1: return logits.unsqueeze(0)
    return logits.view(batch_size, -1)[:, :num_classes]

class DualAttentionModel(nn.Module):
    """
    文档条件双头句级注意力 + 词级注意力 + 三路门控
    （无句标：MIL + diversity + adversarial）
    """
    def __init__(self, bert_model='bert-base-chinese', num_labels=2,
                 adv_weight=0.0, diversity_weight=0.1, grl_lambda=1.0,
                 attn_sup_weight=0.0, sentence_cls_weight=0.0, disentangle_weight=0.0,
                 weakly_supervised=True, mil_weight=1.0):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.hidden_size = self.bert.config.hidden_size
        self.num_labels = num_labels

        self.adv_weight         = float(adv_weight)
        self.diversity_weight   = float(diversity_weight)
        self.attn_sup_weight    = float(attn_sup_weight)
        self.sentence_cls_weight= float(sentence_cls_weight)
        self.disentangle_weight = float(disentangle_weight)
        self.grl_lambda         = float(grl_lambda)

        self.weakly_supervised  = bool(weakly_supervised)
        self.mil_weight         = float(mil_weight)

        try: fake_lbl = FAKE_LABEL
        except NameError: fake_lbl = 1
        self.register_buffer('fake_label', torch.tensor(fake_lbl, dtype=torch.long))

        # 文档条件注意力参数（两头）
        h2 = self.hidden_size // 2
        self.W_h = nn.Linear(self.hidden_size, h2)
        self.W_q = nn.Linear(self.hidden_size, h2)
        self.v1  = nn.Linear(h2, 1)
        self.v2  = nn.Linear(h2, 1)

        # 文档分类支路
        self.transform2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size//2)
        )
        self.classifier2 = nn.Linear(self.hidden_size//2, num_labels)

        # 对抗分支（只用文档标签）
        self.adv_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, num_labels)
        )

        # 句子分类器（为 MIL 提供 p_fake）
        self.sentence_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, 2)
        )
        self.classifier_sentence_weighted = nn.Linear(self.hidden_size, num_labels)

        # 词级注意力
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
        ]: blk.apply(init_linear)

    @staticmethod
    def _neg_inf_like(x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:
            return torch.tensor(-1e4, dtype=x.dtype, device=x.device)
        return torch.tensor(torch.finfo(x.dtype).min, dtype=x.dtype, device=x.device)

    def forward(self, input_ids, attention_mask, actual_lengths,
                labels=None, sentence_labels=None, return_attn=False):

        B, N, L = input_ids.shape
        device = input_ids.device

        sentence_mask = torch.arange(N, device=device).unsqueeze(0) >= actual_lengths.unsqueeze(1)  # (B,N)

        input_ids      = input_ids.clone()
        attention_mask = attention_mask.clone()
        pad_id = self.bert.config.pad_token_id or 0
        cls_id = getattr(self.bert.config, "cls_token_id", 101)
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
        sentence_embeds = outputs.last_hidden_state[:, 0, :].reshape(B, N, self.hidden_size)  # (B,N,H)

        # ===== 文档条件注意力（两头） =====
        valid = (~sentence_mask).float().unsqueeze(-1)                    # (B,N,1)
        denom = valid.sum(dim=1).clamp_min(1.0)                           # (B,1)
        q_doc = (sentence_embeds * valid).sum(dim=1) / denom              # (B,H)

        h_proj = self.W_h(sentence_embeds)                                # (B,N,H/2)
        q_proj = self.W_q(q_doc).unsqueeze(1)                             # (B,1,H/2)

        attn_scores1 = self.v1(torch.tanh(h_proj + q_proj)).squeeze(-1)   # (B,N)
        attn_scores2 = self.v2(torch.tanh(h_proj + q_proj)).squeeze(-1)   # (B,N)

        neg_inf = self._neg_inf_like(attn_scores1)
        attn_scores1 = attn_scores1.masked_fill(sentence_mask, neg_inf)
        attn_scores2 = attn_scores2.masked_fill(sentence_mask, neg_inf)

        attn_weights1 = F.softmax(attn_scores1, dim=1)                    # (B,N)
        attn_weights2 = F.softmax(attn_scores2, dim=1)                    # (B,N)

        weighted1 = torch.sum(sentence_embeds * attn_weights1.unsqueeze(-1), dim=1)  # (B,H)
        weighted2 = torch.sum(sentence_embeds * attn_weights2.unsqueeze(-1), dim=1)  # (B,H)

        # 句子分类 + 假句权重聚合（p_fake）
        sentence_logits = self.sentence_classifier(sentence_embeds)       # (B,N,2)
        sentence_probs  = F.softmax(sentence_logits, dim=-1)              # (B,N,2)
        sentence_fake_probs = sentence_probs[..., 1]                      # (B,N)

        s_fake_masked = sentence_fake_probs.masked_fill(sentence_mask, self._neg_inf_like(sentence_fake_probs))
        sent_weight = F.softmax(s_fake_masked, dim=1)                     # (B,N)

        weighted_sentence_repr = torch.sum(sentence_embeds * sent_weight.unsqueeze(-1), dim=1)  # (B,H)
        logits_sentence_weighted = self.classifier_sentence_weighted(weighted_sentence_repr)

        # 词级注意力
        tok_valid = (attention_mask != 0)                                 # (B,N,L)
        token_scores = self.attn_word(token_embeds).squeeze(-1)           # (B,N,L)
        token_scores = token_scores.masked_fill(~tok_valid, self._neg_inf_like(token_scores))
        token_weights = F.softmax(token_scores, dim=-1)                   # (B,N,L)
        all_invalid = (~tok_valid).all(dim=-1, keepdim=True)
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

        if labels is None:
            sentence_preds = torch.argmax(sentence_logits, dim=-1)
            return logits2, attn_weights1, attn_weights2, sentence_preds

        # ===== 损失 =====
        labels = _ensure_class_indices(labels, self.num_labels)
        logits2 = _ensure_logits_2d(logits2, B, self.num_labels)
        veracity_loss = F.cross_entropy(logits2, labels)

        # 句标相关（默认无）
        if (sentence_labels is not None) and (not NO_SENT_LABELS):
            sent_ce = nn.CrossEntropyLoss(ignore_index=-100)
            sentence_cls_loss = sent_ce(sentence_logits.reshape(-1, 2), sentence_labels.reshape(-1))
        else:
            sentence_cls_loss = torch.tensor(0.0, device=device)

        # 多样性：严格只对两个头的表示做 -cos 分离
        diversity_loss = torch.tensor(0.0, device=device)
        if self.diversity_weight > 0.0:
            n1 = F.normalize(weighted1, p=2, dim=1)
            n2 = F.normalize(weighted2, p=2, dim=1)
            cos = torch.sum(n1 * n2, dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            diversity_loss = -cos.mean()

        # =========================
        # total loss（分模式）
        # =========================
        if (sentence_labels is None) or NO_SENT_LABELS or self.weakly_supervised:
            # 1) MIL
            sent_logit_fake = sentence_logits[..., 1]  # (B,N)
            sent_logit_real = sentence_logits[..., 0]  # (B,N)
            margin = (sent_logit_fake - sent_logit_real)               # (B,N)
            margin = margin.masked_fill(sentence_mask, self._neg_inf_like(margin))
            max_margin, _ = torch.max(margin, dim=1)                   # (B,)
            y_fake = (labels == self.fake_label).float()
            mil_loss = F.binary_cross_entropy_with_logits(max_margin, y_fake)

            total_loss = veracity_loss + self.mil_weight * mil_loss

            # 2) 多样性
            if self.diversity_weight > 0.0:
                total_loss += self.diversity_weight * diversity_loss

            # 3) 对抗分支（只在假样本回传；GRL 放在 weighted1 上）
            if self.adv_weight > 0.0:
                is_fake = (labels == self.fake_label).to(device)  # (B,)
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
                total_loss = total_loss + self.adv_weight * adv_loss

        else:
            total_loss = veracity_loss + self.sentence_cls_weight * sentence_cls_loss
            if self.diversity_weight > 0.0:
                total_loss += self.diversity_weight * diversity_loss

        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            total_loss = veracity_loss  

        out = {
            "logits": logits2,
            "sentence_weighted_logits": logits_sentence_weighted,
            "veracity_loss": float(veracity_loss.item())
        }
        if return_attn:
            sentence_preds = torch.argmax(sentence_logits, dim=-1)
            return total_loss, out, attn_weights1, attn_weights2, sentence_preds
        else:
            return total_loss, out

######################################################################
# 训练与评估工具
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
            self.best = metric; self.num_bad = 0; return True
        self.num_bad += 1; return False
    def should_stop(self): return self.num_bad >= self.patience

def build_param_groups(model, lr_bert=2e-5, lr_heads=1e-4, weight_decay=0.01):
    bert_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if n.startswith("bert."): bert_params.append(p)
        else: head_params.append(p)
    return [
        {"params": bert_params, "lr": lr_bert, "weight_decay": weight_decay},
        {"params": head_params, "lr": lr_heads, "weight_decay": weight_decay},
    ]

def train_dual_attention_fast(
    model, train_dataset, val_dataset, device='cuda',
    num_epochs=6, batch_size=4,
    lr_bert=2e-5, lr_heads=5e-5, weight_decay=0.05,
    warmup_ratio=0.06, accumulate_steps=4,
    use_grad_ckpt=True, set_dropout_to=0.2,
    ema_decay=0.999, early_stop_patience=3,
    save_dir="fast_ckpts", save_prefix="best_weaks"
):
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    if use_grad_ckpt: enable_grad_checkpointing(model)
    if set_dropout_to is not None: set_dropout(model, set_dropout_to)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=max(8, batch_size), collate_fn=collate_fn, pin_memory=True)

    optimizer = torch.optim.AdamW(build_param_groups(model, lr_bert, lr_heads, weight_decay))
    total_update_steps = num_epochs * math.ceil(len(train_loader) / max(1, accumulate_steps))
    warmup_steps = int(warmup_ratio * total_update_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)
    scaler = torch.cuda.amp.GradScaler()
    ema = EMA(model, decay=ema_decay) if (ema_decay and ema_decay < 1.0) else None
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
            input_ids      = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            actual_lengths = batch['actual_lengths'].to(device, non_blocking=True)
            labels         = batch['labels'].to(device, non_blocking=True)
            sentence_labels= batch['sentence_labels'].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, actual_lengths=actual_lengths,
                    labels=labels, sentence_labels=None if NO_SENT_LABELS else sentence_labels
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

            running += loss.item()
            pbar.set_postfix({"loss": f"{running/(step+1):.4f}"})

        # flush 尾批
        if (step + 1) % accumulate_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if ema: ema.update(model)

        # 验证（EMA更稳）
        if ema: ema.store(model); ema.copy_to(model)
        val_acc = eval_accuracy_only(model, val_loader, device)
        if ema: ema.restore(model)

        print(f"[Epoch {epoch+1}] train_loss={running/max(1,len(train_loader)):.4f}  val_acc={val_acc:.4f}")
        if early.step(val_acc):
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            with open(best_meta, "w") as f:
                json.dump({"epoch": epoch+1, "val_acc": float(val_acc)}, f, indent=2, ensure_ascii=False)
            print(f"  ↳ New best! Saved to {best_path}")
        if early.should_stop():
            print("Early stopped."); break
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
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, actual_lengths=actual_lengths, labels=None)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec  = recall_score(all_labels,   all_preds, average='macro', zero_division=0)
    f1   = f1_score(all_labels,       all_preds, average='macro', zero_division=0)
    return acc, prec, rec, f1

######################################################################
# Top-p 推理（不需阈值/句标；默认把 head2 当作“假头”）
######################################################################
@torch.no_grad()
def top_p_suspect_sentences(
    model, dataloader, device='cuda',
    p: float = 0.5, head: int = 2,
    doc_thresh: float | None = 0.5,   
    skip_non_fake: bool = True       
):
    import numpy as np
    from tqdm import tqdm

    model.eval(); model.to(device)
    outputs = []; doc_ptr = 0

    for batch in tqdm(dataloader, desc=f"Top-p (p={p:.2f}, head={head})"):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        actual_lengths = batch['actual_lengths']
        sentences_list = batch['original_sentences']

        logits2, attn1, attn2, _ = model(
            input_ids=input_ids, attention_mask=attention_mask,
            actual_lengths=actual_lengths.to(device), labels=None
        )
        p_fake = torch.softmax(logits2, dim=-1)[:, 1].detach().cpu().numpy()  # (B,)
        attn = attn2 if head == 2 else attn1

        B = attn.size(0)
        for i in range(B):
            if (doc_thresh is not None) and (p_fake[i] < doc_thresh):
                if not skip_non_fake:
                    outputs.append({
                        'doc_idx': doc_ptr, 'selected_indices': [], 'selected_sentences': [],
                        'selected_scores': [], 'cumulative_coverage': 0.0, 'all_scores': [],
                        'p_fake_doc': float(p_fake[i])
                    })
                doc_ptr += 1
                continue

            L = int(actual_lengths[i].item())
            sents = sentences_list[i][:L]
            scores = attn[i, :L].detach().cpu().numpy().astype(float)
            if scores.size == 0:
                outputs.append({
                    'doc_idx': doc_ptr, 'selected_indices': [], 'selected_sentences': [],
                    'selected_scores': [], 'cumulative_coverage': 0.0, 'all_scores': [],
                    'p_fake_doc': float(p_fake[i])
                })
                doc_ptr += 1
                continue

            order = np.argsort(-scores)
            s_sum = float(scores.sum()); cum = 0.0; picks = []
            for idx in order:
                picks.append(int(idx)); cum += float(scores[idx])
                if s_sum <= 0 or (cum / s_sum) >= p:
                    break

            picks_sorted = sorted(picks)
            outputs.append({
                'doc_idx': doc_ptr,
                'selected_indices': picks_sorted,
                'selected_sentences': [sents[j] for j in picks_sorted],
                'selected_scores': [float(scores[j]) for j in picks_sorted],
                'cumulative_coverage': float(cum / max(s_sum, 1e-12)),
                'all_scores': scores.tolist(),
                'p_fake_doc': float(p_fake[i])
            })
            doc_ptr += 1
    return outputs


@torch.no_grad()
def attn2_diagnostics(model, dataloader, device='cuda', p: float = 0.5, head: int = 2,
                      use_pred_gate: bool = False, pred_thresh: float = 0.5):
    """
    若 use_pred_gate=True，则仅统计 预测为假( p_fake >= pred_thresh ) 的文档；
    否则仍按真值(Real/Fake)对所有文档统计。
    """
    import numpy as np
    from tqdm import tqdm

    model.eval(); model.to(device)
    eps = 1e-12
    stats = {'fake': {'max_attn2': [], 'H2': [], 'k_top_p': []},
             'real': {'max_attn2': [], 'H2': [], 'k_top_p': []}}
    per_doc = []
    doc_ptr = 0

    for batch in tqdm(dataloader, desc='Attn2 diagnostics'):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        actual_lengths = batch['actual_lengths']
        labels         = batch['labels']  

        logits2, attn1, attn2, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            actual_lengths=actual_lengths.to(device),
            labels=None
        )
        p_fake = torch.softmax(logits2, dim=-1)[:, 1].detach().cpu().numpy()
        attn = attn2 if head == 2 else attn1  # (B,N)

        B = attn.size(0)
        for i in range(B):
            if use_pred_gate and (p_fake[i] < pred_thresh):
                doc_ptr += 1
                continue  

            L = int(actual_lengths[i].item())
            y = int(labels[i].item())      
            vec = attn[i, :L].detach().cpu().numpy().astype(np.float64)
            if vec.size == 0:
                doc_ptr += 1
                continue

            max_attn = float(vec.max())
            H = float(-(vec * np.log(vec + eps)).sum())
            Hn = float(H / (np.log(max(L, 1)) + eps))

            order = np.argsort(-vec)
            s_sum = float(vec.sum()); cum = 0.0; k = 0
            for j in order:
                cum += float(vec[j]); k += 1
                if s_sum <= 0 or (cum / s_sum) >= p:
                    break

            key = 'fake' if y == 1 else 'real'
            stats[key]['max_attn2'].append(max_attn)
            stats[key]['H2'].append(Hn)
            stats[key]['k_top_p'].append(k)

            per_doc.append({
                'doc_idx': doc_ptr, 'label': y, 'p_fake_pred': float(p_fake[i]),
                'max_attn2': max_attn, 'H2_norm': Hn, 'k_top_p': int(k),
                'p': p, 'head': head
            })
            doc_ptr += 1

    def _summ(v):
        if len(v) == 0:
            return {'n': 0, 'mean': 0, 'std': 0, 'p50': 0, 'p90': 0, 'min': 0, 'max': 0}
        arr = np.asarray(v, dtype=np.float64)
        return {
            'n': int(arr.size), 'mean': float(arr.mean()), 'std': float(arr.std(ddof=0)),
            'p50': float(np.median(arr)), 'p90': float(np.percentile(arr, 90)),
            'min': float(arr.min()), 'max': float(arr.max()),
        }

    summary = {
        'head': head, 'p': p, 'pred_gate': use_pred_gate, 'pred_thresh': pred_thresh,
        'max_attn2': {'real': _summ(stats['real']['max_attn2']),
                      'fake': _summ(stats['fake']['max_attn2'])},
        'H2_norm':   {'real': _summ(stats['real']['H2']),
                      'fake': _summ(stats['fake']['H2'])},
        'k_top_p':   {'real': _summ(stats['real']['k_top_p']),
                      'fake': _summ(stats['fake']['k_top_p'])},
    }

    print("\n=== Attn-head{} diagnostics (Top-p p={:.2f}, use_pred_gate={}, thr={:.2f}) ==="
          .format(head, p, use_pred_gate, pred_thresh))
    for name in ['max_attn2', 'H2_norm', 'k_top_p']:
        r = summary[name]['real']; f = summary[name]['fake']
        print(f"{name:>9s} | REAL  n={r['n']:4d} mean={r['mean']:.3f} p50={r['p50']:.3f} "
              f"|| FAKE  n={f['n']:4d} mean={f['mean']:.3f} p50={f['p50']:.3f}")
    return summary, per_doc


@torch.no_grad()
def calibrate_entropy_max_thresholds(diag_per_doc, target_fpr=0.05):
    """
    用验证集诊断结果 per_doc（见前面 attn2_diagnostics 的返回）做阈值校准。
    返回 (tau_H, tau_max)。
    """
    import numpy as np
    H_real  = [d['H2_norm']   for d in diag_per_doc if d['label'] == 0]
    M_real  = [d['max_attn2'] for d in diag_per_doc if d['label'] == 0]
    if len(H_real) == 0 or len(M_real) == 0:
        # 兜底：经验阈值
        return 0.60, 0.30
    q = float(target_fpr)
    tau_H   = float(np.quantile(np.array(H_real, dtype=np.float64), q))        
    tau_max = float(np.quantile(np.array(M_real, dtype=np.float64), 1.0 - q))  
    return tau_H, tau_max

@torch.no_grad()
def save_attn_csv(model, dataloader, device='cuda', csv_path="attn_per_sentence.csv",
                  save_text=True):
    """
    导出格式：doc_idx, sent_idx, attn1, attn2[, p_fake_doc][, sentence]
    - save_text=True 时会把句子文本也写进去。
    """
    import csv
    model.eval(); model.to(device)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        cols = ["doc_idx","sent_idx","attn1","attn2","p_fake_doc"]
        if save_text: cols.append("sentence")
        w = csv.writer(f); w.writerow(cols)

        doc_ptr = 0
        for batch in dataloader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            actual_lengths = batch['actual_lengths']
            sentences_list = batch['original_sentences']

            logits2, a1, a2, _ = model(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       actual_lengths=actual_lengths.to(device),
                                       labels=None)
            p_fake = torch.softmax(logits2, dim=-1)[:, 1].detach().cpu().numpy()

            B = a1.size(0)
            for i in range(B):
                L = int(actual_lengths[i].item())
                attn1 = a1[i, :L].detach().cpu().tolist()
                attn2 = a2[i, :L].detach().cpu().tolist()
                sents = sentences_list[i][:L]
                for j in range(L):
                    row = [doc_ptr, j, f"{attn1[j]:.6f}", f"{attn2[j]:.6f}", f"{p_fake[i]:.6f}"]
                    if save_text: row.append(sents[j].replace("\n"," "))
                    w.writerow(row)
                doc_ptr += 1
    print(f"[saved] {csv_path}")

@torch.no_grad()
def save_attn_npz(model, dataloader, device='cuda', npz_path="attn_mats.npz"):
    """
    输出：
      attn1: (D, Nmax)   # head1 注意力
      attn2: (D, Nmax)   # head2 注意力
      mask:  (D, Nmax)   # 1=有效句子, 0=padding
      p_fake: (D,)       # 文档假概率
    """
    import numpy as np
    model.eval(); model.to(device)

    # 预扫确定 D 和 Nmax
    num_docs, Nmax = 0, 0
    for batch in dataloader:
        num_docs += batch['input_ids'].size(0)
        Nmax = max(Nmax, int(batch['actual_lengths'].max().item()))

    A1 = np.zeros((num_docs, Nmax), dtype=np.float32)
    A2 = np.zeros((num_docs, Nmax), dtype=np.float32)
    M  = np.zeros((num_docs, Nmax), dtype=np.int8)
    P  = np.zeros((num_docs,),      dtype=np.float32)

    doc_ptr = 0
    for batch in dataloader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        actual_lengths = batch['actual_lengths']

        logits2, a1, a2, _ = model(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   actual_lengths=actual_lengths.to(device),
                                   labels=None)
        p_fake = torch.softmax(logits2, dim=-1)[:, 1].detach().cpu().numpy()

        B = a1.size(0)
        for i in range(B):
            L = int(actual_lengths[i].item())
            A1[doc_ptr, :L] = a1[i, :L].detach().cpu().numpy()
            A2[doc_ptr, :L] = a2[i, :L].detach().cpu().numpy()
            M[doc_ptr, :L]  = 1
            P[doc_ptr]      = p_fake[i]
            doc_ptr += 1

    np.savez(npz_path, attn1=A1, attn2=A2, mask=M, p_fake=P)
    print(f"[saved] {npz_path}")

@torch.no_grad()
def collect_sentence_scores_labels(model, dataloader, device='cuda', head:int=2):
    """
    返回：
      scores: 1D np.array，所有有效句子的分数（默认 attn2）
      labels: 1D np.array，对应句子标签（0/1）
      per_doc: list[{'doc_idx', 'scores': list, 'labels': list}]
    """
    import numpy as np
    from tqdm import tqdm

    model.eval(); model.to(device)
    scores_all, labels_all = [], []
    per_doc = []
    doc_ptr = 0

    for batch in tqdm(dataloader, desc=f"Collect (head={head})"):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        actual_lengths = batch['actual_lengths']
        sentence_labels= batch['sentence_labels']  # [B, N]
        _, attn1, attn2, _ = model(input_ids=input_ids, attention_mask=attention_mask,
                                   actual_lengths=actual_lengths.to(device), labels=None)
        attn = attn2 if head==2 else attn1
        B = attn.size(0)

        for i in range(B):
            L = int(actual_lengths[i].item())
            y  = sentence_labels[i, :L].cpu().numpy()
            valid = (y != -100)
            if valid.sum() == 0:
                doc_ptr += 1; continue
            s = attn[i, :L].detach().cpu().numpy().astype(float)
            s = s[valid]; y = y[valid]
            scores_all.extend(s.tolist())
            labels_all.extend(y.tolist())
            per_doc.append({'doc_idx': doc_ptr,
                            'scores': s.tolist(),
                            'labels': y.tolist()})
            doc_ptr += 1

    return np.asarray(scores_all, dtype=float), np.asarray(labels_all, dtype=int), per_doc

def _confusion_and_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true,  y_pred, zero_division=0)
    f1   = f1_score(y_true,     y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=[0,1])  # [[TN,FP],[FN,TP]]
    tn, fp, fn, tp = cm.ravel()
    return {'acc':acc,'prec':prec,'rec':rec,'f1':f1,'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

def find_best_threshold_val(scores_val, labels_val, optimize='f1'):
    import numpy as np
    qs = np.linspace(0.01, 0.99, 99)
    grid = np.unique(np.quantile(scores_val, qs))
    best = None
    for thr in grid:
        pred = (scores_val >= thr).astype(int)
        m = _confusion_and_metrics(labels_val, pred)
        if (best is None) or (m[optimize] > best[optimize]):
            best = {'thr': float(thr), **m}
    return best

def evaluate_threshold_on_test(scores_test, labels_test, thr):
    import numpy as np
    pred = (scores_test >= float(thr)).astype(int)
    return _confusion_and_metrics(labels_test, pred)

def evaluate_topk_on(per_doc, k:int):
    import numpy as np
    y_all, p_all = [], []
    for d in per_doc:
        s = np.asarray(d['scores'], dtype=float)
        y = np.asarray(d['labels'], dtype=int)
        if s.size == 0: continue
        order = np.argsort(-s)
        kk = min(k, s.size)
        pred = np.zeros_like(y, dtype=int)
        pred[order[:kk]] = 1
        y_all.extend(y.tolist())
        p_all.extend(pred.tolist())
    return _confusion_and_metrics(np.asarray(y_all), np.asarray(p_all))

def find_best_topk_val(per_doc_val, k_grid=(1,2,3,5,8,10)):
    best = None
    for k in k_grid:
        m = evaluate_topk_on(per_doc_val, k)
        if (best is None) or (m['f1'] > best['f1']):
            best = {'k': int(k), **m}
    return best




######################################################################
# main：10 个随机种子 → 各自 VAL 最优 → TEST 评测 → 指标求平均
######################################################################
if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ====== 数据路径（按需修改）======
    train_path = "./fake_dataset/0730/train_sentence_labeled.pkl"
    val_path   = "./fake_dataset/0730/val_sentence_labeled.pkl"
    test_path  = "./fake_dataset/0730/test_sentence_labeled.pkl"

    


    # 载入数据（一次即可）
    NO_SENT_LABELS = False
    train_texts, train_labels = load_sentence_labeled_data(train_path)
    NO_SENT_LABELS = False
    val_texts,   val_labels   = load_sentence_labeled_data(val_path)
    test_texts,  test_labels  = load_sentence_labeled_data(test_path)

    max_sents = 60
    max_len   = 170
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer, max_sentences=max_sents, max_length=max_len)
    val_dataset   = NewsDataset(val_texts,   val_labels,   tokenizer, max_sentences=max_sents, max_length=max_len)
    test_dataset  = NewsDataset(test_texts,  test_labels,  tokenizer, max_sentences=max_sents, max_length=max_len)

    val_loader  = DataLoader(val_dataset,  batch_size=16, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn, pin_memory=True)

    model_kwargs = dict(
        bert_model='bert-base-chinese', num_labels=2,
        attn_sup_weight=0.0, sentence_cls_weight=0.0, disentangle_weight=0.0,
        adv_weight=0.02, diversity_weight=0.20, grl_lambda=1.0,
        weakly_supervised=True, mil_weight=1.0
    )
    train_kwargs = dict(
        num_epochs=20, batch_size=4,
        lr_bert=2e-5, lr_heads=5e-5, weight_decay=0.05,
        warmup_ratio=0.05, accumulate_steps=4,
        use_grad_ckpt=True, set_dropout_to=0.3,
        ema_decay=0.999, early_stop_patience=5,
        save_dir="fast_ckpts"
    )

    SEEDS = [42,2021,2025,3,4,5,6,7,8,9]

    # 结果容器
    os.makedirs("multi_seed_results_weaks", exist_ok=True)
    run_results = []
    val_best_list = []
    doc_metrics_mat = []  # [[acc, prec, rec, f1], ...]

    for seed in SEEDS:
        print("\n" + "="*72)
        print(f"===> SEED = {seed}")
        print("="*72)

        set_seed(seed)

        model = DualAttentionModel(**model_kwargs)
        save_prefix = f"best_weaks_s{seed}"
        model, best_path, best_val_acc = train_dual_attention_fast(
            model, train_dataset, val_dataset, device=device,
            save_prefix=save_prefix, **train_kwargs
        )

        best_model = DualAttentionModel(**model_kwargs)
        best_model.load_state_dict(torch.load(best_path, map_location=device))
        best_model.to(device)

        acc, prec, rec, f1 = eval_metrics(best_model, test_loader, device)
        print(f"\n[Seed {seed}]  Best@VAL={best_val_acc:.4f}  TEST -> "
              f"acc={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}")

        rec_this = {
            'seed': seed,
            'best_val_acc': float(best_val_acc),
            'test_doc_metrics': {'acc': float(acc), 'precision': float(prec),
                                 'recall': float(rec), 'f1': float(f1)},
            'best_path': best_path
        }
        run_results.append(rec_this)
        val_best_list.append(float(best_val_acc))
        doc_metrics_mat.append([float(acc), float(prec), float(rec), float(f1)])

        with open(os.path.join("multi_seed_results_weaks", f"seed_{seed}_result.json"), "w", encoding="utf-8") as f:
            json.dump(rec_this, f, ensure_ascii=False, indent=2)

        del best_model, model
        gc.collect(); torch.cuda.empty_cache()

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
        'test_doc_metrics_mean': {'acc': float(doc_mean[0]), 'precision': float(doc_mean[1]),
                                  'recall': float(doc_mean[2]), 'f1': float(doc_mean[3])},
        'test_doc_metrics_std':  {'acc': float(doc_std[0]),  'precision': float(doc_std[1]),
                                  'recall': float(doc_std[2]),  'f1': float(doc_std[3])},
        'runs': run_results
    }

    print("\n" + "="*72)
    print("===> 10 seeds 平均 (各自 VAL 最优权重在 TEST 上评估)")
    print("="*72)
    print(f"VAL best acc: mean={avg_best_val:.4f}  std={std_best_val:.4f}")
    print("TEST (doc-level) mean ± std:")
    print(f"  acc={doc_mean[0]:.4f} ± {doc_std[0]:.4f}")
    print(f"  precision={doc_mean[1]:.4f} ± {doc_std[1]:.4f}")
    print(f"  recall={doc_mean[2]:.4f} ± {doc_std[2]:.4f}")
    print(f"  f1={doc_mean[3]:.4f} ± {doc_std[3]:.4f}")

    # 汇总落盘
    with open(os.path.join("multi_seed_results_weaks", "summary_10seeds.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 也输出一个 CSV
    try:
        import csv
        with open(os.path.join("multi_seed_results_weaks", "per_seed_test_doc_metrics.csv"), "w", newline="", encoding="utf-8") as f:
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
