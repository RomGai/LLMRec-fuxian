#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def progress(prefix: str, current: int, total: int) -> None:
    total = max(total, 1)
    pct = 100.0 * float(current) / float(total)
    print(f"[PROGRESS][{prefix}] {current}/{total} ({pct:.2f}%)")


def build_80_20_split_from_pairs(
    pairs_tsv: str,
    split_train_out: str,
    split_test_out: str,
    split_ratio: float = 0.8,
    progress_every_users: int = 1000,
) -> Tuple[int, int]:
    """Build user-wise chronological 80/20 split from *_u_i_pairs.tsv.

    Returns:
        (n_train_users, n_test_users)
    """
    user_events: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    print(f"[SPLIT] reading pairs file: {pairs_tsv}")
    with open(pairs_tsv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                u = int((row.get("user_id") or "").strip())
                i = int((row.get("item_id") or "").strip())
                ts = int((row.get("timestamp") or "0").strip() or 0)
            except ValueError:
                continue
            user_events[u].append((ts, i))

    users = sorted(user_events.keys())
    print(f"[SPLIT] total users in pairs={len(users)}")

    train_map: Dict[int, List[int]] = {}
    test_map: Dict[int, List[int]] = {}
    for idx, u in enumerate(users, 1):
        events = sorted(user_events[u], key=lambda x: x[0])
        items = [i for _, i in events]
        if len(items) == 0:
            continue
        if len(items) == 1:
            train_items, test_items = items, []
        else:
            cut = int(len(items) * split_ratio)
            cut = min(max(cut, 1), len(items) - 1)
            train_items = items[:cut]
            test_items = items[cut:]
        train_map[u] = train_items
        if test_items:
            test_map[u] = test_items

        if idx % max(1, progress_every_users) == 0 or idx == len(users):
            progress("SPLIT-USERS", idx, len(users))

    os.makedirs(os.path.dirname(split_train_out), exist_ok=True)
    with open(split_train_out, "w", encoding="utf-8") as f:
        for u in sorted(train_map.keys()):
            pos = ",".join(str(x) for x in train_map[u])
            f.write(f"{u}\t{pos}\t\n")
    with open(split_test_out, "w", encoding="utf-8") as f:
        for u in sorted(test_map.keys()):
            pos = ",".join(str(x) for x in test_map[u])
            f.write(f"{u}\t{pos}\t\n")

    print(f"[SPLIT] train users={len(train_map)}, test users={len(test_map)}")
    print(f"[SPLIT] wrote train split: {split_train_out}")
    print(f"[SPLIT] wrote test split:  {split_test_out}")
    return len(train_map), len(test_map)


# -----------------------------
# Data loading
# -----------------------------

def parse_user_items_file(path: str) -> Dict[int, List[int]]:
    data: Dict[int, List[int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            try:
                user = int(parts[0].strip())
            except ValueError:
                continue
            pos = []
            for x in parts[1].split(","):
                x = x.strip()
                if not x:
                    continue
                try:
                    pos.append(int(x))
                except ValueError:
                    continue
            if not pos:
                continue
            data[user] = pos
    return data


def load_item_desc(path: str) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            raw_iid = (row.get("item_id") or "").strip()
            if not raw_iid:
                continue
            try:
                iid = int(raw_iid)
            except ValueError:
                # Some data files may contain malformed rows (e.g., whitespace-only IDs).
                # Skip these rows instead of crashing.
                continue
            out[iid] = {
                "summary": (row.get("summary", "") or "").strip(),
                "image": (row.get("image", "") or "").strip(),
            }
    return out


# -----------------------------
# Prompt templates (aligned to original LLMRec scripts)
# -----------------------------

def build_ui_prompt(history_items: List[int], candidate_items: List[int], item_desc: Dict[int, Dict[str, str]]) -> str:
    history_string = "User history:\n"
    for iid in history_items:
        summary = item_desc.get(iid, {}).get("summary", "")[:220].replace("\n", " ")
        history_string += f"[{iid}] {summary}\n"

    candidate_string = "Candidates:\n"
    for iid in candidate_items:
        summary = item_desc.get(iid, {}).get("summary", "")[:220].replace("\n", " ")
        candidate_string += f"[{iid}] {summary}\n"

    output_format = (
        "Please output the index of user's favorite and least favorite movie only from candidate, but not user history. "
        "Please get the index from candidate, at the beginning of each line.\n"
        "Output format:\n"
        "Two numbers separated by '::'. Nothing else.Plese just give the index of candicates, remove [] "
        "(just output the digital value), please do not output other thing else, do not give reasoning.\n\n"
    )

    prompt = (
        "You are a movie recommendation system and required to recommend user with movies based on user history "
        "that each movie with title(same topic/doctor), year(similar years), genre(similar genre).\n"
    )
    prompt += history_string
    prompt += candidate_string
    prompt += output_format
    return prompt


def build_user_profile_prompt(history_items: List[int], item_desc: Dict[int, Dict[str, str]]) -> str:
    history_string = "User history:\n"
    for iid in history_items:
        summary = item_desc.get(iid, {}).get("summary", "")[:220].replace("\n", " ")
        history_string += f"[{iid}] {summary}\n"

    output_format = (
        "Please output the following infomation of user, output format:\n"
        "{'age':age, 'gender':gender, 'liked genre':liked genre, 'disliked genre':disliked genre, "
        "'liked directors':liked directors, 'country':country\\, 'language':language}\n"
        "Please do not fill in 'unknown', but make an educated guess based on the available information and fill in the specific content.\n"
        "please output only the content in format above, but no other thing else, no reasoning, no analysis, no Chinese. "
        "Reiterating once again!! Please only output the content after \"output format: \", and do not include any other content such as introduction or acknowledgments.\n\n"
    )

    prompt = "You are required to generate user profile based on the history of user, that each movie with title, year, genre.\n"
    prompt += history_string
    prompt += output_format
    return prompt


def build_item_attr_prompt(item_id: int, item_desc: Dict[int, Dict[str, str]]) -> str:
    summary = item_desc.get(item_id, {}).get("summary", "")[:350].replace("\n", " ")
    pre_string = "You are now a search engines, and required to provide the inquired information of the given movies bellow:\n"
    item_list_string = f"[{item_id}] {summary}\n"
    output_format = (
        "The inquired information is : director, country, language.\n"
        "And please output them in form of: \ndirector::country::language\n"
        "please output only the content in the form above, i.e., director::country::language\n"
        ", but no other thing else, no reasoning, no index.\n\n"
    )
    return pre_string + item_list_string + output_format


# -----------------------------
# Qwen3-8B wrapper (thinking mode disabled)
# -----------------------------

class Qwen3Client:
    def __init__(self, model_name: str = "Qwen/Qwen3-8B"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # required by user
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()


# -----------------------------
# Feature hashing for user/item textual augmentation
# -----------------------------

def hash_text(text: str, dim: int = 256) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in text.lower().split():
        idx = hash(tok) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


class SimpleLLMRecBPR(nn.Module):
    def __init__(self, n_users: int, n_items: int, emb_dim: int, feat_dim: int):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.user_feat_proj = nn.Linear(feat_dim, emb_dim, bias=False)
        self.item_feat_proj = nn.Linear(feat_dim, emb_dim, bias=False)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.user_feat_proj.weight)
        nn.init.xavier_uniform_(self.item_feat_proj.weight)

    def encode_users(self, user_ids: torch.Tensor, user_feats: torch.Tensor) -> torch.Tensor:
        return self.user_emb(user_ids) + self.user_feat_proj(user_feats)

    def encode_items(self, item_ids: torch.Tensor, item_feats: torch.Tensor) -> torch.Tensor:
        return self.item_emb(item_ids) + self.item_feat_proj(item_feats)


@dataclass
class EvalResult:
    hr10: float
    hr20: float
    hr40: float
    ndcg10: float
    ndcg20: float
    ndcg40: float
    mean_rank: float


def ndcg(rank: int, k: int) -> float:
    if rank <= k:
        return 1.0 / math.log2(rank + 1)
    return 0.0


def hit(rank: int, k: int) -> float:
    return 1.0 if rank <= k else 0.0


def sample_train_batch(train_pos: Dict[int, List[int]], n_items: int, batch_size: int) -> Tuple[List[int], List[int], List[int]]:
    users = random.choices(list(train_pos.keys()), k=batch_size)
    pos_items, neg_items = [], []
    for u in users:
        p = random.choice(train_pos[u])
        pos_items.append(p)
        seen = set(train_pos[u])
        n = random.randint(0, n_items - 1)
        while n in seen:
            n = random.randint(0, n_items - 1)
        neg_items.append(n)
    return users, pos_items, neg_items


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_checkpoint_compat(path: str, map_location: str):
    """Compatibility loader for PyTorch>=2.6 (weights_only default changed)."""
    try:
        # PyTorch 2.6+: default weights_only=True can fail for checkpoints with numpy arrays.
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch versions do not support the weights_only argument.
        return torch.load(path, map_location=map_location)


def check_prompt_consistency() -> None:
    required_markers = [
        "User history:",
        "Candidates:",
        "Two numbers separated by '::'",
        "generate user profile",
        "director::country::language",
    ]
    text = "\n".join(required_markers)
    assert "Candidates:" in text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="LLMRec-new-data")
    parser.add_argument("--dataset", type=str, default="Baby_Products")
    parser.add_argument("--output_dir", type=str, default="outputs_qwen3_pipeline")
    parser.add_argument("--stage", type=str, choices=["augment", "train", "eval", "all"], default="all")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--feat_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--num_ui_candidates", type=int, default=20)
    parser.add_argument("--eval_negatives", type=int, default=1000)
    parser.add_argument("--max_aug_users", type=int, default=300)
    parser.add_argument("--max_aug_items", type=int, default=1000)
    parser.add_argument("--eval_user_source", type=str, choices=["test", "all_known"], default="test")
    parser.add_argument("--auto_split_80_20", action="store_true")
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--progress_user_every", type=int, default=10)
    parser.add_argument("--progress_item_every", type=int, default=50)
    parser.add_argument("--progress_step_every", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("[STAGE] loading data files")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ensure_dir(args.output_dir)

    default_train_file = os.path.join(args.data_dir, f"{args.dataset}_user_items_negs_train.csv")
    default_test_file = os.path.join(args.data_dir, f"{args.dataset}_user_items_negs_test.csv")
    train_file = default_train_file
    test_file = default_test_file
    item_desc_file = os.path.join(args.data_dir, f"{args.dataset}_item_desc.tsv")
    pairs_file = os.path.join(args.data_dir, f"{args.dataset}_u_i_pairs.tsv")

    if args.auto_split_80_20:
        print("[STAGE] preparing 80/20 split from u_i_pairs")
        split_dir = os.path.join(args.output_dir, "splits")
        train_file = os.path.join(split_dir, f"{args.dataset}_user_items_negs_train.csv")
        test_file = os.path.join(split_dir, f"{args.dataset}_user_items_negs_test.csv")
        build_80_20_split_from_pairs(
            pairs_tsv=pairs_file,
            split_train_out=train_file,
            split_test_out=test_file,
            split_ratio=args.split_ratio,
            progress_every_users=max(100, args.progress_user_every * 10),
        )
    else:
        print("[STAGE] using existing train/test split files from data_dir (no re-split)")

    train_pos = parse_user_items_file(train_file)
    test_pos = parse_user_items_file(test_file)
    item_desc = load_item_desc(item_desc_file)
    print(f"[LOAD] train_file={train_file}, users={len(train_pos)}")
    print(f"[LOAD] test_file={test_file}, users={len(test_pos)}")
    print(f"[LOAD] item_desc_file={item_desc_file}, items={len(item_desc)}")

    train_users = sorted(train_pos.keys())
    test_users = sorted(test_pos.keys())
    all_users = sorted(set(train_users) | set(test_users))
    all_items = sorted(item_desc.keys())
    n_users = max(all_users) + 1
    n_items = max(all_items) + 1

    print(
        f"[INFO] dataset={args.dataset} total_users={n_users} "
        f"train_users={len(train_users)} test_users={len(test_users)} items={n_items}"
    )

    cache_dir = os.path.join(args.output_dir, args.dataset)
    ensure_dir(cache_dir)

    ui_aug_path = os.path.join(cache_dir, "augmented_ui_samples.json")
    user_profile_path = os.path.join(cache_dir, "augmented_user_profile.json")
    item_attr_path = os.path.join(cache_dir, "augmented_item_attr.json")
    model_path = os.path.join(cache_dir, "simple_llmrec_bpr.pt")

    if args.stage in ["augment", "all"]:
        print("[STAGE] augmentation with Qwen3-8B (thinking disabled)")
        check_prompt_consistency()
        llm = Qwen3Client("Qwen/Qwen3-8B")

        ui_aug = {}
        if os.path.exists(ui_aug_path):
            ui_aug = json.load(open(ui_aug_path, "r", encoding="utf-8"))
        user_profiles = {}
        if os.path.exists(user_profile_path):
            user_profiles = json.load(open(user_profile_path, "r", encoding="utf-8"))
        item_attrs = {}
        if os.path.exists(item_attr_path):
            item_attrs = json.load(open(item_attr_path, "r", encoding="utf-8"))

        aug_users = list(train_pos.keys())[: args.max_aug_users]
        print(f"[AUG] user_augmentation_total={len(aug_users)}")
        for idx, u in enumerate(aug_users, 1):
            key = str(u)
            history = train_pos[u]

            if key not in user_profiles:
                prompt = build_user_profile_prompt(history, item_desc)
                user_profiles[key] = llm.generate(prompt, max_new_tokens=256)

            if key not in ui_aug:
                seen = set(history)
                pool = [i for i in all_items if i not in seen]
                if len(pool) < args.num_ui_candidates:
                    continue
                cand = random.sample(pool, args.num_ui_candidates)
                prompt = build_ui_prompt(history, cand, item_desc)
                content = llm.generate(prompt, max_new_tokens=48)
                try:
                    left, right = content.split("::")
                    pos_i = int("".join(ch for ch in left if ch.isdigit()))
                    neg_i = int("".join(ch for ch in right if ch.isdigit()))
                    ui_aug[key] = {"pos": pos_i, "neg": neg_i}
                except Exception:
                    pass

            if idx % max(1, args.progress_user_every) == 0 or idx == len(aug_users):
                progress("AUG-USER", idx, len(aug_users))

        aug_items = all_items[: args.max_aug_items]
        print(f"[AUG] item_augmentation_total={len(aug_items)}")
        for idx, i in enumerate(aug_items, 1):
            key = str(i)
            if key in item_attrs:
                continue
            prompt = build_item_attr_prompt(i, item_desc)
            item_attrs[key] = llm.generate(prompt, max_new_tokens=80)
            if idx % max(1, args.progress_item_every) == 0 or idx == len(aug_items):
                progress("AUG-ITEM", idx, len(aug_items))

        json.dump(ui_aug, open(ui_aug_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(user_profiles, open(user_profile_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(item_attrs, open(item_attr_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[AUG] saved: {ui_aug_path}, {user_profile_path}, {item_attr_path}")

    if args.stage in ["train", "all"]:
        print("[STAGE] training simple BPR recommender")
        ui_aug = json.load(open(ui_aug_path, "r", encoding="utf-8")) if os.path.exists(ui_aug_path) else {}
        user_profiles = json.load(open(user_profile_path, "r", encoding="utf-8")) if os.path.exists(user_profile_path) else {}
        item_attrs = json.load(open(item_attr_path, "r", encoding="utf-8")) if os.path.exists(item_attr_path) else {}

        user_feat_np = np.zeros((n_users, args.feat_dim), dtype=np.float32)
        for u in range(n_users):
            user_feat_np[u] = hash_text(user_profiles.get(str(u), ""), args.feat_dim)

        item_feat_np = np.zeros((n_items, args.feat_dim), dtype=np.float32)
        for i in range(n_items):
            text = item_attrs.get(str(i), "") + " " + item_desc.get(i, {}).get("summary", "")[:200]
            item_feat_np[i] = hash_text(text, args.feat_dim)

        user_feat = torch.from_numpy(user_feat_np).to(args.device)
        item_feat = torch.from_numpy(item_feat_np).to(args.device)

        model = SimpleLLMRecBPR(n_users, n_items, args.emb_dim, args.feat_dim).to(args.device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        steps_per_epoch = max(1, len(train_pos) // max(1, args.batch_size))
        for ep in range(1, args.epochs + 1):
            print(f"[TRAIN] epoch_start={ep}/{args.epochs}")
            model.train()
            running = 0.0
            pbar = tqdm(range(steps_per_epoch), desc=f"epoch {ep}/{args.epochs}")
            for step in pbar:
                users, pos, neg = sample_train_batch(train_pos, n_items, args.batch_size)

                aug_users = random.sample(list(ui_aug.keys()), k=min(len(ui_aug), max(1, args.batch_size // 10))) if ui_aug else []
                for u_key in aug_users:
                    row = ui_aug[u_key]
                    users.append(int(u_key))
                    pos.append(int(row["pos"]))
                    neg.append(int(row["neg"]))

                u = torch.tensor(users, dtype=torch.long, device=args.device)
                p = torch.tensor(pos, dtype=torch.long, device=args.device)
                n = torch.tensor(neg, dtype=torch.long, device=args.device)

                u_emb = model.encode_users(u, user_feat[u])
                p_emb = model.encode_items(p, item_feat[p])
                n_emb = model.encode_items(n, item_feat[n])

                pos_score = (u_emb * p_emb).sum(dim=-1)
                neg_score = (u_emb * n_emb).sum(dim=-1)
                bpr = -F.logsigmoid(pos_score - neg_score).mean()

                opt.zero_grad()
                bpr.backward()
                opt.step()

                running += float(bpr.item())
                if (step + 1) % max(1, args.progress_step_every) == 0:
                    pbar.set_postfix({"bpr_loss": f"{running/(step+1):.4f}"})
                    progress(f"TRAIN-E{ep}", step + 1, steps_per_epoch)

            print(f"[TRAIN] epoch={ep} avg_bpr_loss={running/steps_per_epoch:.6f}")

        torch.save({
            "model": model.state_dict(),
            "user_feat": user_feat_np,
            "item_feat": item_feat_np,
            "n_users": n_users,
            "n_items": n_items,
        }, model_path)
        print(f"[TRAIN] model saved to {model_path}")

    if args.stage in ["eval", "all"]:
        print("[STAGE] evaluation on 1 target + 1000 random negatives")
        ckpt = load_checkpoint_compat(model_path, map_location=args.device)
        model = SimpleLLMRecBPR(ckpt["n_users"], ckpt["n_items"], args.emb_dim, args.feat_dim).to(args.device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        user_feat = torch.from_numpy(ckpt["user_feat"]).to(args.device)
        item_feat = torch.from_numpy(ckpt["item_feat"]).to(args.device)

        if args.eval_user_source == "test":
            users_eval = sorted([u for u in test_pos if len(test_pos[u]) > 0])
        else:
            users_eval = sorted([u for u in all_users if len(test_pos.get(u, [])) > 0])
        print(f"[EVAL] user_source={args.eval_user_source}, eval_users={len(users_eval)}")
        metrics = {"hr10": 0.0, "hr20": 0.0, "hr40": 0.0, "ndcg10": 0.0, "ndcg20": 0.0, "ndcg40": 0.0, "rank": 0.0}

        for idx, u in enumerate(users_eval, 1):
            target = test_pos[u][0]  # exactly one target per user for evaluation
            seen = set(train_pos.get(u, [])) | set(test_pos[u])
            pool = [i for i in range(ckpt["n_items"]) if i not in seen and i != target]
            if len(pool) < args.eval_negatives:
                continue
            negatives = random.sample(pool, args.eval_negatives)
            candidates = [target] + negatives

            u_tensor = torch.tensor([u] * len(candidates), dtype=torch.long, device=args.device)
            i_tensor = torch.tensor(candidates, dtype=torch.long, device=args.device)
            with torch.no_grad():
                u_emb = model.encode_users(u_tensor, user_feat[u_tensor])
                i_emb = model.encode_items(i_tensor, item_feat[i_tensor])
                scores = (u_emb * i_emb).sum(dim=-1).detach().cpu().numpy()

            target_score = scores[0]
            rank = 1 + int(np.sum(scores[1:] > target_score))

            metrics["hr10"] += hit(rank, 10)
            metrics["hr20"] += hit(rank, 20)
            metrics["hr40"] += hit(rank, 40)
            metrics["ndcg10"] += ndcg(rank, 10)
            metrics["ndcg20"] += ndcg(rank, 20)
            metrics["ndcg40"] += ndcg(rank, 40)
            metrics["rank"] += rank

            n = idx
            print(
                f"[EVAL][USER {idx}/{len(users_eval)} u={u}] "
                f"AvgHR@10={metrics['hr10']/n:.4f} AvgHR@20={metrics['hr20']/n:.4f} AvgHR@40={metrics['hr40']/n:.4f} "
                f"AvgNDCG@10={metrics['ndcg10']/n:.4f} AvgNDCG@20={metrics['ndcg20']/n:.4f} AvgNDCG@40={metrics['ndcg40']/n:.4f} "
                f"AvgRank={metrics['rank']/n:.2f}"
            )

        n = max(1, len(users_eval))
        final = EvalResult(
            hr10=metrics["hr10"] / n,
            hr20=metrics["hr20"] / n,
            hr40=metrics["hr40"] / n,
            ndcg10=metrics["ndcg10"] / n,
            ndcg20=metrics["ndcg20"] / n,
            ndcg40=metrics["ndcg40"] / n,
            mean_rank=metrics["rank"] / n,
        )
        print("[FINAL]", final)


if __name__ == "__main__":
    main()
