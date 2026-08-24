import os
import gc
import copy
import json
import random
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


# ============================================================
# CONFIGURAÇÃO
# ============================================================

DEFAULT_AUGMENTED_PATH = "train_augmented_25pct.csv"
DEFAULT_RESTRICTED_PATH = "train_restricted_25pct.csv"

MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"
SPACY_MODEL = "en_core_web_sm"

SEED = 42
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 6
PATIENCE = 2

BERT_LR = 2e-5
HEAD_LR = 5e-4
WEIGHT_DECAY = 0.01

TEST_SIZE = 0.15
VAL_SIZE = 0.15
NUM_LABELS = 3
TOP_K_SPACY = 12
SPACY_PROJECTION_DIM = 16
AUGMENTED_SYNTHETIC_SHARE = 0.25

LABEL_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

OUTPUT_DIR = "tinybert_3_scenarios_results"


# ============================================================
# ARGUMENTOS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compara três cenários: "
            "(1) TinyBERT baseline + dataset augmented 25%, "
            "(2) TinyBERT baseline + dataset restricted 25%, "
            "(3) TinyBERT + spaCy reduced TOP-12 + dataset restricted 25%."
        )
    )

    parser.add_argument(
        "--scenario",
        choices=[
            "augmented_baseline",
            "restricted_baseline",
            "restricted_spacy12",
            "all",
        ],
        default="all",
        help="Cenário a executar. Padrão: all.",
    )

    parser.add_argument(
        "--augmented-data",
        default=DEFAULT_AUGMENTED_PATH,
        help=f"CSV augmented. Padrão: {DEFAULT_AUGMENTED_PATH}",
    )

    parser.add_argument(
        "--restricted-data",
        default=DEFAULT_RESTRICTED_PATH,
        help=f"CSV restricted. Padrão: {DEFAULT_RESTRICTED_PATH}",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_SPACY,
        help=f"Número de features spaCy no cenário reduzido. Padrão: {TOP_K_SPACY}.",
    )

    parser.add_argument(
        "--synthetic-share",
        type=float,
        default=AUGMENTED_SYNTHETIC_SHARE,
        help=(
            "Proporção sintética desejada no treino do cenário augmented. "
            f"Padrão: {AUGMENTED_SYNTHETIC_SHARE * 100:.0f}%%."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Seed. Padrão: {SEED}.",
    )

    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help=f"Pasta de resultados. Padrão: {OUTPUT_DIR}",
    )

    return parser.parse_args()


# ============================================================
# REPRODUTIBILIDADE
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# FEATURES spaCy
# ============================================================

POS_TAGS = [
    "NOUN", "PROPN", "VERB", "ADJ", "ADV",
    "AUX", "NUM", "PRON", "ADP",
]

DEP_TAGS = [
    "nsubj", "nsubjpass", "obj", "dobj", "iobj",
    "ROOT", "neg", "amod", "advmod", "compound",
]

ENT_TYPES = [
    "ORG", "PERSON", "GPE", "MONEY", "PERCENT",
    "DATE", "TIME", "CARDINAL", "QUANTITY", "PRODUCT",
]


def safe_ratio(value, denominator):
    return float(value) / max(float(denominator), 1.0)


def extract_spacy_features(texts, nlp):
    rows = []
    docs = nlp.pipe(texts, batch_size=64)

    for doc in tqdm(docs, total=len(texts), desc="spaCy"):
        tokens = [token for token in doc if not token.is_space]
        alpha_tokens = [token for token in tokens if token.is_alpha]
        sents = list(doc.sents)

        n_tokens = max(len(tokens), 1)
        n_alpha = max(len(alpha_tokens), 1)

        pos_counts = Counter(token.pos_ for token in tokens)
        dep_counts = Counter(token.dep_ for token in tokens)
        ent_counts = Counter(ent.label_ for ent in doc.ents)

        avg_token_len = (
            np.mean([len(token.text) for token in alpha_tokens])
            if alpha_tokens else 0.0
        )

        avg_sentence_len = (
            np.mean([
                len([token for token in sent if not token.is_space])
                for sent in sents
            ])
            if sents else 0.0
        )

        row = {
            "log_token_count": np.log1p(len(tokens)),
            "log_sentence_count": np.log1p(len(sents)),
            "avg_token_len": avg_token_len,
            "avg_sentence_len": avg_sentence_len,
            "stop_ratio": safe_ratio(sum(t.is_stop for t in tokens), n_tokens),
            "punct_ratio": safe_ratio(sum(t.is_punct for t in tokens), n_tokens),
            "digit_ratio": safe_ratio(
                sum(t.like_num or t.is_digit for t in tokens), n_tokens
            ),
            "upper_ratio": safe_ratio(sum(t.is_upper for t in alpha_tokens), n_alpha),
            "negation_ratio": safe_ratio(
                sum(t.dep_ == "neg" for t in tokens), n_tokens
            ),
            "entity_ratio": safe_ratio(len(doc.ents), n_tokens),
        }

        for tag in POS_TAGS:
            row[f"pos_{tag}"] = safe_ratio(pos_counts[tag], n_tokens)

        for dep in DEP_TAGS:
            row[f"dep_{dep}"] = safe_ratio(dep_counts[dep], n_tokens)

        for ent_type in ENT_TYPES:
            row[f"ent_{ent_type}"] = safe_ratio(ent_counts[ent_type], n_tokens)

        rows.append(row)

    return pd.DataFrame(rows).astype(np.float32)


# ============================================================
# DATASET PYTORCH
# ============================================================

class FinancialPhraseDataset(Dataset):
    def __init__(self, texts, labels, spacy_features, tokenizer):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.spacy_features = np.asarray(spacy_features, dtype=np.float32)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        item = {key: value.squeeze(0) for key, value in encoded.items()}

        item["spacy_features"] = torch.tensor(
            self.spacy_features[idx], dtype=torch.float32
        )
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


# ============================================================
# MODELO
# ============================================================

class TinyBERTClassifier(nn.Module):
    def __init__(
        self,
        use_spacy_features=False,
        spacy_feature_dim=0,
        spacy_projection_dim=16,
        num_labels=3,
    ):
        super().__init__()

        self.use_spacy_features = use_spacy_features
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        bert_hidden = self.bert.config.hidden_size

        if use_spacy_features:
            self.spacy_projector = nn.Sequential(
                nn.Linear(spacy_feature_dim, spacy_projection_dim),
                nn.LayerNorm(spacy_projection_dim),
                nn.GELU(),
                nn.Dropout(0.15),
            )
            classifier_input = bert_hidden + spacy_projection_dim
        else:
            self.spacy_projector = None
            classifier_input = bert_hidden

        self.classifier = nn.Sequential(
            nn.Dropout(0.20),
            nn.Linear(classifier_input, 128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, num_labels),
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        spacy_features=None,
        token_type_ids=None,
    ):
        bert_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if token_type_ids is not None:
            bert_kwargs["token_type_ids"] = token_type_ids

        bert_outputs = self.bert(**bert_kwargs)
        cls_embedding = bert_outputs.last_hidden_state[:, 0, :]

        if self.use_spacy_features:
            linguistic_embedding = self.spacy_projector(spacy_features)
            combined = torch.cat([cls_embedding, linguistic_embedding], dim=1)
        else:
            combined = cls_embedding

        return self.classifier(combined)


# ============================================================
# TREINO / AVALIAÇÃO
# ============================================================

def batch_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    losses = []
    all_labels = []
    all_predictions = []

    for batch in loader:
        batch = batch_to_device(batch, device)

        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
            spacy_features=batch["spacy_features"],
        )

        labels = batch["labels"]
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        losses.append(loss.item())
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(all_labels, all_predictions)),
        "macro_f1": float(f1_score(all_labels, all_predictions, average="macro")),
        "weighted_f1": float(
            f1_score(all_labels, all_predictions, average="weighted")
        ),
        "labels": np.asarray(all_labels),
        "predictions": np.asarray(all_predictions),
    }


def make_loader(dataset, shuffle, device):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


def compute_weights(train_df, device):
    classes = np.array([0, 1, 2])
    weights_np = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["label"].to_numpy(),
    )

    print("Pesos das classes:")
    for label, weight in zip(classes, weights_np):
        print(f"  {label} ({LABEL_NAMES[label]}): {weight:.4f}")

    return torch.tensor(weights_np, dtype=torch.float32, device=device)


def build_optimizer(model):
    bert_params = list(model.bert.parameters())
    head_params = list(model.classifier.parameters())

    if model.spacy_projector is not None:
        head_params += list(model.spacy_projector.parameters())

    return torch.optim.AdamW(
        [
            {
                "params": bert_params,
                "lr": BERT_LR,
                "weight_decay": WEIGHT_DECAY,
            },
            {
                "params": head_params,
                "lr": HEAD_LR,
                "weight_decay": WEIGHT_DECAY,
            },
        ]
    )


def train_model(
    experiment_name,
    train_df,
    val_df,
    test_df,
    tokenizer,
    device,
    seed,
    output_dir,
    use_spacy_features=False,
    feature_names=None,
    feature_arrays=None,
    spacy_projection_dim=0,
):
    print("\n" + "=" * 78)
    print(f"EXPERIMENTO: {experiment_name}")
    print("=" * 78)

    print(f"Treino:    {len(train_df)}")
    print(f"Validação: {len(val_df)}")
    print(f"Teste:     {len(test_df)}")

    if "source" in train_df.columns:
        source_counts = train_df["source"].astype(str).value_counts()
        print("\nOrigem do conjunto de treino:")
        for source, count in source_counts.items():
            print(f"  {source}: {count}")

        synthetic_count = int(
            train_df["source"].astype(str).str.lower().eq("synthetic").sum()
        )
        print(
            f"  Proporção sintética no treino: "
            f"{synthetic_count / max(len(train_df), 1):.2%}"
        )

    if use_spacy_features:
        print(f"\nFeatures spaCy: {len(feature_names)}")
        print(f"Projeção spaCy: {spacy_projection_dim}D")
        for name in feature_names:
            print(f"  - {name}")
    else:
        print("\nSem features spaCy.")

    set_seed(seed)

    if feature_arrays is None:
        X_train = np.zeros((len(train_df), 0), dtype=np.float32)
        X_val = np.zeros((len(val_df), 0), dtype=np.float32)
        X_test = np.zeros((len(test_df), 0), dtype=np.float32)
        feature_names = []
    else:
        X_train, X_val, X_test = feature_arrays

    train_dataset = FinancialPhraseDataset(
        train_df["text"], train_df["label"], X_train, tokenizer
    )
    val_dataset = FinancialPhraseDataset(
        val_df["text"], val_df["label"], X_val, tokenizer
    )
    test_dataset = FinancialPhraseDataset(
        test_df["text"], test_df["label"], X_test, tokenizer
    )

    train_loader = make_loader(train_dataset, True, device)
    val_loader = make_loader(val_dataset, False, device)
    test_loader = make_loader(test_dataset, False, device)

    model = TinyBERTClassifier(
        use_spacy_features=use_spacy_features,
        spacy_feature_dim=X_train.shape[1],
        spacy_projection_dim=spacy_projection_dim,
        num_labels=NUM_LABELS,
    ).to(device)

    class_weights = compute_weights(train_df, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = build_optimizer(model)

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.10),
        num_training_steps=total_steps,
    )

    use_amp = device.type == "cuda"

    try:
        scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_state = None
    best_val_f1 = -1.0
    patience_counter = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for batch in progress:
            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                try:
                    autocast_context = torch.amp.autocast("cuda")
                except AttributeError:
                    autocast_context = torch.cuda.amp.autocast()
            else:
                autocast_context = torch.autocast(
                    device_type="cpu", enabled=False
                )

            with autocast_context:
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids"),
                    spacy_features=batch["spacy_features"],
                )
                loss = criterion(logits, batch["labels"])

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            scheduler.step()

            train_losses.append(loss.item())
            progress.set_postfix(loss=f"{np.mean(train_losses):.4f}")

        val_metrics = evaluate(model, val_loader, criterion, device)

        epoch_info = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
        }
        history.append(epoch_info)

        print(
            f"Epoch {epoch} | "
            f"train_loss={epoch_info['train_loss']:.4f} | "
            f"val_loss={epoch_info['val_loss']:.4f} | "
            f"val_acc={epoch_info['val_accuracy']:.4f} | "
            f"val_macro_f1={epoch_info['val_macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break

    if best_state is None:
        raise RuntimeError("Nenhum estado válido foi salvo.")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, criterion, device)

    report_text = classification_report(
        test_metrics["labels"],
        test_metrics["predictions"],
        labels=[0, 1, 2],
        target_names=[LABEL_NAMES[0], LABEL_NAMES[1], LABEL_NAMES[2]],
        digits=4,
        zero_division=0,
    )

    report_dict = classification_report(
        test_metrics["labels"],
        test_metrics["predictions"],
        labels=[0, 1, 2],
        target_names=[LABEL_NAMES[0], LABEL_NAMES[1], LABEL_NAMES[2]],
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        test_metrics["labels"],
        test_metrics["predictions"],
        labels=[0, 1, 2],
    )

    print("\nRESULTADO NO TESTE COMUM")
    print(report_text)
    print("Matriz de confusão:")
    print(cm)

    train_synthetic = 0
    if "source" in train_df.columns:
        train_synthetic = int(
            train_df["source"].astype(str).str.lower().eq("synthetic").sum()
        )

    result = {
        "experiment": experiment_name,
        "seed": seed,
        "train_size": int(len(train_df)),
        "validation_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "train_synthetic_count": train_synthetic,
        "train_synthetic_share": float(train_synthetic / max(len(train_df), 1)),
        "use_spacy_features": use_spacy_features,
        "spacy_feature_names": feature_names,
        "spacy_feature_dim": int(X_train.shape[1]),
        "spacy_projection_dim": int(spacy_projection_dim),
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "history": history,
    }

    model_path = os.path.join(output_dir, f"{experiment_name}.pt")
    result_path = os.path.join(output_dir, f"{experiment_name}_result.json")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": MODEL_NAME,
            "experiment": experiment_name,
            "seed": seed,
            "spacy_feature_names": feature_names,
            "spacy_feature_dim": int(X_train.shape[1]),
            "spacy_projection_dim": int(spacy_projection_dim),
            "label_names": LABEL_NAMES,
            "max_len": MAX_LEN,
        },
        model_path,
    )

    result["model_path"] = model_path

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Modelo salvo em: {model_path}")
    print(f"Resultado salvo em: {result_path}")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# ============================================================
# PREPARAÇÃO DOS DADOS
# ============================================================

def load_and_clean(path, dataset_name):
    df = pd.read_csv(path)

    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{dataset_name}: colunas ausentes: {missing}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    invalid = set(df["label"].unique()) - {0, 1, 2}
    if invalid:
        raise ValueError(f"{dataset_name}: labels inválidos: {invalid}")

    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    removed = before - len(df)

    print(f"\n{dataset_name}")
    print(f"  Carregado: {before}")
    print(f"  Duplicatas removidas: {removed}")
    print(f"  Final: {len(df)}")
    print(f"  Labels: {df['label'].value_counts().sort_index().to_dict()}")

    return df


def make_common_real_split(restricted_df, seed):
    train_val_df, test_df = train_test_split(
        restricted_df,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=restricted_df["label"],
    )

    relative_val_size = VAL_SIZE / (1.0 - TEST_SIZE)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=seed,
        stratify=train_val_df["label"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def get_synthetic_pool(augmented_df, restricted_df):
    if "source" in augmented_df.columns:
        synthetic_df = augmented_df[
            augmented_df["source"].astype(str).str.lower().eq("synthetic")
        ].copy()
    else:
        restricted_texts = set(restricted_df["text"])
        synthetic_df = augmented_df[
            ~augmented_df["text"].isin(restricted_texts)
        ].copy()

    # Segurança extra: nenhum texto real do restricted pode entrar como sintético.
    restricted_texts = set(restricted_df["text"])
    synthetic_df = synthetic_df[
        ~synthetic_df["text"].isin(restricted_texts)
    ].copy()

    synthetic_df = synthetic_df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    if len(synthetic_df) == 0:
        raise ValueError("Nenhuma amostra sintética foi encontrada no dataset augmented.")

    if "source" not in synthetic_df.columns:
        synthetic_df["source"] = "synthetic"

    return synthetic_df


def build_augmented_training(
    real_train_df,
    synthetic_df,
    seed,
    target_synthetic_share=0.25,
):
    if not (0.0 < target_synthetic_share < 1.0):
        raise ValueError("--synthetic-share deve estar entre 0 e 1.")

    real_train = real_train_df.copy()

    if "source" not in real_train.columns:
        real_train["source"] = "original"

    # Se s/(n_real+s)=p, então s = p*n_real/(1-p).
    required_synthetic = int(round(
        target_synthetic_share * len(real_train)
        / (1.0 - target_synthetic_share)
    ))

    if required_synthetic > len(synthetic_df):
        print(
            f"AVISO: seriam necessários {required_synthetic} sintéticos para "
            f"atingir {target_synthetic_share:.2%}, mas existem somente "
            f"{len(synthetic_df)} sintéticos únicos. Usando todos."
        )
        selected_synthetic = synthetic_df.copy()
    elif required_synthetic == len(synthetic_df):
        selected_synthetic = synthetic_df.copy()
    else:
        # Amostragem estratificada para manter aproximadamente a distribuição
        # das classes do conjunto sintético disponível.
        selected_synthetic, _ = train_test_split(
            synthetic_df,
            train_size=required_synthetic,
            random_state=seed,
            stratify=synthetic_df["label"],
        )
        selected_synthetic = selected_synthetic.reset_index(drop=True)

    augmented_train = pd.concat(
        [real_train, selected_synthetic],
        ignore_index=True,
    )

    augmented_train = augmented_train.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

    return augmented_train, selected_synthetic


# ============================================================
# spaCy REDUZIDO TOP-K
# ============================================================

def prepare_reduced_spacy_features(
    train_df,
    val_df,
    test_df,
    top_k,
    seed,
    output_dir,
):
    import spacy

    print("\nCarregando spaCy...")
    nlp = spacy.load(SPACY_MODEL)

    print("Extraindo spaCy do treino restricted...")
    train_features_df = extract_spacy_features(train_df["text"].tolist(), nlp)

    print("Extraindo spaCy da validação comum...")
    val_features_df = extract_spacy_features(val_df["text"].tolist(), nlp)

    print("Extraindo spaCy do teste comum...")
    test_features_df = extract_spacy_features(test_df["text"].tolist(), nlp)

    all_features = train_features_df.columns.tolist()
    top_k = max(1, min(top_k, len(all_features)))

    # IMPORTANTE: seleção exclusivamente no treino real restricted.
    mi_scores = mutual_info_classif(
        train_features_df.to_numpy(),
        train_df["label"].to_numpy(),
        random_state=seed,
    )

    ranking = pd.DataFrame(
        {
            "feature": all_features,
            "mutual_information": mi_scores,
        }
    ).sort_values("mutual_information", ascending=False).reset_index(drop=True)

    selected = ranking.head(top_k)["feature"].tolist()

    print("\nRanking spaCy por Mutual Information:")
    print(ranking.to_string(index=False))

    print(f"\nTOP {top_k} selecionadas:")
    for idx, feature in enumerate(selected, 1):
        score = ranking.loc[
            ranking["feature"] == feature,
            "mutual_information",
        ].iloc[0]
        print(f"  {idx:02d}. {feature:<25} MI={score:.6f}")

    ranking.to_csv(
        os.path.join(output_dir, "spacy_feature_importance_restricted.csv"),
        index=False,
    )

    with open(
        os.path.join(output_dir, "spacy_selected_features_restricted.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train_features_df[selected]
    ).astype(np.float32)

    X_val = scaler.transform(
        val_features_df[selected]
    ).astype(np.float32)

    X_test = scaler.transform(
        test_features_df[selected]
    ).astype(np.float32)

    return selected, (X_train, X_val, X_test)


# ============================================================
# RESUMO FINAL
# ============================================================

def build_comparison(results, output_dir):
    rows = []

    display_names = {
        "augmented_baseline": "1) Baseline - Augmented 25%",
        "restricted_baseline": "2) Baseline - Restricted 25%",
        "restricted_spacy12": "3) Restricted 25% + spaCy Reduced 12",
    }

    for scenario, result in results.items():
        report = result["classification_report"]

        rows.append(
            {
                "cenario": display_names[scenario],
                "train_size": result["train_size"],
                "synthetic_train": result["train_synthetic_count"],
                "synthetic_share_train": result["train_synthetic_share"],
                "spacy_features": result["spacy_feature_dim"],
                "accuracy": result["test_accuracy"],
                "macro_f1": result["test_macro_f1"],
                "weighted_f1": result["test_weighted_f1"],
                "f1_negative": report["negative"]["f1-score"],
                "f1_neutral": report["neutral"]["f1-score"],
                "f1_positive": report["positive"]["f1-score"],
            }
        )

    comparison = pd.DataFrame(rows)

    if "restricted_baseline" in results:
        ref = results["restricted_baseline"]
        comparison["delta_macro_f1_vs_restricted"] = (
            comparison["macro_f1"] - ref["test_macro_f1"]
        )
        comparison["delta_accuracy_vs_restricted"] = (
            comparison["accuracy"] - ref["test_accuracy"]
        )

    print("\n" + "=" * 100)
    print("COMPARAÇÃO FINAL - MESMO TESTE REAL PARA TODOS OS CENÁRIOS")
    print("=" * 100)
    print(comparison.to_string(index=False))

    comparison_path = os.path.join(output_dir, "comparison_3_scenarios.csv")
    comparison.to_csv(comparison_path, index=False)
    print(f"\nComparação salva em: {comparison_path}")

    return comparison


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("COMPARAÇÃO TinyBERT - 3 CENÁRIOS")
    print(f"Cenário solicitado: {args.scenario}")
    print(f"Seed: {args.seed}")
    print(f"Dispositivo: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("CUDA não detectada. O treinamento usará CPU.")

    print("=" * 78)

    restricted_df = load_and_clean(
        args.restricted_data,
        "RESTRICTED 25% (real)",
    )

    # O split real é criado UMA vez e reutilizado pelos três cenários.
    real_train_df, common_val_df, common_test_df = make_common_real_split(
        restricted_df,
        args.seed,
    )

    print("\nSPLIT REAL COMUM AOS TRÊS CENÁRIOS")
    print(f"  Treino real: {len(real_train_df)}")
    print(f"  Validação real: {len(common_val_df)}")
    print(f"  Teste real: {len(common_test_df)}")
    print(f"  Teste labels: {common_test_df['label'].value_counts().sort_index().to_dict()}")

    if args.scenario == "all":
        scenarios = [
            "augmented_baseline",
            "restricted_baseline",
            "restricted_spacy12",
        ]
    else:
        scenarios = [args.scenario]

    augmented_train_df = None

    if "augmented_baseline" in scenarios:
        augmented_df = load_and_clean(
            args.augmented_data,
            "AUGMENTED 25%",
        )

        synthetic_df = get_synthetic_pool(augmented_df, restricted_df)

        print("\nSintéticos únicos disponíveis para o cenário 1:")
        print(f"  Total: {len(synthetic_df)}")
        print(f"  Labels: {synthetic_df['label'].value_counts().sort_index().to_dict()}")

        augmented_train_df, selected_synthetic_df = build_augmented_training(
            real_train_df,
            synthetic_df,
            args.seed,
            target_synthetic_share=args.synthetic_share,
        )

        print("\nCenário 1 - composição efetiva do TREINO:")
        print(f"  Reais: {len(real_train_df)}")
        print(f"  Sintéticos selecionados: {len(selected_synthetic_df)}")
        print(f"  Total: {len(augmented_train_df)}")
        print(
            f"  % sintético no treino: "
            f"{len(selected_synthetic_df) / len(augmented_train_df):.2%}"
        )
        print(
            "  Os sintéticos são retirados exclusivamente do arquivo augmented "
            "e são amostrados de forma estratificada por label."
        )

    print("\nCarregando tokenizer TinyBERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    results = {}

    # --------------------------------------------------------
    # CENÁRIO 1
    # Baseline TinyBERT + treino real restricted + sintéticos
    # Validação/teste são os mesmos reais dos outros cenários.
    # --------------------------------------------------------
    if "augmented_baseline" in scenarios:
        results["augmented_baseline"] = train_model(
            experiment_name="scenario1_augmented_baseline",
            train_df=augmented_train_df,
            val_df=common_val_df,
            test_df=common_test_df,
            tokenizer=tokenizer,
            device=device,
            seed=args.seed,
            output_dir=args.output_dir,
            use_spacy_features=False,
        )

    # --------------------------------------------------------
    # CENÁRIO 2
    # Baseline TinyBERT somente com restricted real.
    # --------------------------------------------------------
    if "restricted_baseline" in scenarios:
        results["restricted_baseline"] = train_model(
            experiment_name="scenario2_restricted_baseline",
            train_df=real_train_df,
            val_df=common_val_df,
            test_df=common_test_df,
            tokenizer=tokenizer,
            device=device,
            seed=args.seed,
            output_dir=args.output_dir,
            use_spacy_features=False,
        )

    # --------------------------------------------------------
    # CENÁRIO 3
    # Mesmo restricted do cenário 2 + TOP-12 spaCy.
    # A seleção TOP-K usa somente o treino restricted.
    # --------------------------------------------------------
    if "restricted_spacy12" in scenarios:
        selected_features, feature_arrays = prepare_reduced_spacy_features(
            real_train_df,
            common_val_df,
            common_test_df,
            top_k=args.top_k,
            seed=args.seed,
            output_dir=args.output_dir,
        )

        results["restricted_spacy12"] = train_model(
            experiment_name="scenario3_restricted_spacy12",
            train_df=real_train_df,
            val_df=common_val_df,
            test_df=common_test_df,
            tokenizer=tokenizer,
            device=device,
            seed=args.seed,
            output_dir=args.output_dir,
            use_spacy_features=True,
            feature_names=selected_features,
            feature_arrays=feature_arrays,
            spacy_projection_dim=SPACY_PROJECTION_DIM,
        )

    build_comparison(results, args.output_dir)

    results_path = os.path.join(args.output_dir, "results_current_run.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Resultados completos salvos em: {results_path}")


if __name__ == "__main__":
    main()