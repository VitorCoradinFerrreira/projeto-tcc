"""
Pipeline de pesquisa: discordância humana + features linguísticas
+ modelo de sentimento sob restrição simulada de dados
Dataset: Financial PhraseBank (Malo et al., 2014)

Requisitos (instale antes de rodar):
    pip install datasets spacy scikit-learn scipy pandas numpy sentence-transformers
    python -m spacy download en_core_web_sm

Este script roda inteiramente em CPU. Precisa de internet na primeira
execução para baixar o dataset (Hugging Face) e o modelo de embeddings
(~80MB). Depois disso, tudo fica em cache local.
"""

import numpy as np
import pandas as pd
import spacy
import torch
torch.backends.mkldnn.enabled = False
from datasets import load_dataset
from scipy.stats import spearmanr
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sentence_transformers import SentenceTransformer

SEED = 42
np.random.seed(SEED)

# =====================================================================
# ETAPA 1 — Reconstruir a discordância a partir dos 4 arquivos de agreement
# =====================================================================
def carregar_e_reconstruir_discordancia():
    BASE = "hf://datasets/szlazakm/SentimentAnalysis/FinancialPhraseBank/"
    df_allagree = pd.read_parquet(BASE + "financial_phrasebank_AllAgree.parquet")
    df_75agree = pd.read_parquet(BASE + "financial_phrasebank_75Agree.parquet")
    df_66agree = pd.read_parquet(BASE + "financial_phrasebank_66Agree.parquet")
    df_base = pd.read_parquet(BASE + "financial_phrasebank_50Agree.parquet")

    # a cópia comunitária usa "sentiment" em vez de "label" — renomeia pra manter
    # o resto do script (que espera "label") funcionando sem mais mudanças
    df_base = df_base.rename(columns={"sentiment": "label"})

    set_allagree = set(df_allagree["sentence"])
    set_75 = set(df_75agree["sentence"])
    set_66 = set(df_66agree["sentence"])

    def nivel_discordancia(frase):
        if frase in set_allagree:
            return 0
        elif frase in set_75:
            return 1
        elif frase in set_66:
            return 2
        else:
            return 3

    df_base["discordancia"] = df_base["sentence"].apply(nivel_discordancia)
    return df_base


# =====================================================================
# ETAPA 2 — Simular restrição de dados (amostragem estratificada por nível)
# =====================================================================
def simular_restricao(df, n_amostras=100, seed=SEED):
    n_niveis = df["discordancia"].nunique()
    por_nivel = max(1, n_amostras // n_niveis)
    partes = []
    for _, grupo in df.groupby("discordancia"):
        partes.append(grupo.sample(min(len(grupo), por_nivel), random_state=seed))
    return pd.concat(partes).reset_index(drop=True)


# =====================================================================
# ETAPA 3 — Extrair features linguísticas interpretáveis
# =====================================================================
_nlp = spacy.load("en_core_web_sm")
HEDGES = {"may", "might", "could", "expected", "likely", "possibly", "roughly", "approximately"}
BOOSTERS = {"clearly", "significantly", "substantially", "definitely", "strongly"}

def extrair_features(frase):
    doc = _nlp(frase)
    n_tokens = len(doc)
    n_numeros = sum(1 for t in doc if t.like_num)
    n_negacoes = sum(1 for t in doc if t.dep_ == "neg")
    n_hedges = sum(1 for t in doc if t.lemma_.lower() in HEDGES)
    n_boosters = sum(1 for t in doc if t.lemma_.lower() in BOOSTERS)
    tam_medio_palavra = np.mean([len(t.text) for t in doc if t.is_alpha]) if n_tokens else 0
    return pd.Series({
        "n_tokens": n_tokens,
        "densidade_numeros": n_numeros / max(n_tokens, 1),
        "negacoes": n_negacoes,
        "hedges": n_hedges,
        "boosters": n_boosters,
        "tam_medio_palavra": tam_medio_palavra,
    })

FEATURE_COLS = ["n_tokens", "densidade_numeros", "negacoes", "hedges", "boosters", "tam_medio_palavra"]


# =====================================================================
# ETAPA 4 — Modelar a estrutura: features linguísticas -> discordância
# =====================================================================
def analisar_estrutura(df):
    X = df[FEATURE_COLS]
    y = df["discordancia"]

    print("Correlação de Spearman (feature vs. nível de discordância):")
    for col in FEATURE_COLS:
        rho, p = spearmanr(X[col], y)
        print(f"  {col:20s} rho={rho:+.2f}  p={p:.3f}")

    arvore = DecisionTreeRegressor(max_depth=3, random_state=SEED)
    arvore.fit(X, y)
    print("\nÁrvore de decisão rasa (só para interpretar, não para prever):")
    print(export_text(arvore, feature_names=FEATURE_COLS))
    return arvore


# =====================================================================
# ETAPA 5 — Embeddings leves (só inferência, sem fine-tuning)
# =====================================================================
def gerar_embeddings(frases, modelo="sentence-transformers/all-MiniLM-L6-v2"):
    modelo_st = SentenceTransformer(modelo)  # ~80MB, roda em CPU
    return modelo_st.encode(list(frases), show_progress_bar=False)


# =====================================================================
# ETAPA 6 e 7 — Modelo de sentimento sob restrição + validação leave-one-out
# =====================================================================
def treinar_e_validar_sentimento(df, embeddings):
    labels = df["label"].values
    clf = LogisticRegression(max_iter=1000)
    loo = LeaveOneOut()
    preds = cross_val_predict(clf, embeddings, labels, cv=loo)

    acc = (preds == labels).mean()
    print(f"\nAcurácia leave-one-out do modelo de sentimento: {acc:.2%}")

    df = df.copy()
    df["erro_modelo"] = (preds != labels)
    print("\nTaxa de erro do modelo por nível de discordância humana:")
    print(df.groupby("discordancia")["erro_modelo"].mean())
    print("\n(Se a taxa de erro sobe junto com a discordância, é evidência de que")
    print("a discordância reflete ambiguidade real do texto — não ruído de anotação.)")

    clf.fit(embeddings, labels)
    return clf, df


# =====================================================================
# EXECUÇÃO DO PIPELINE COMPLETO
# =====================================================================
if __name__ == "__main__":
    print("### ETAPA 1: reconstruindo discordância ###")
    df_completo = carregar_e_reconstruir_discordancia()
    print(df_completo["discordancia"].value_counts().sort_index(), "\n")

    print("### ETAPA 2: simulando restrição de dados ###")
    df_restrito = simular_restricao(df_completo, n_amostras=100)
    print(f"Amostra restrita: {len(df_restrito)} sentenças\n")

    print("### ETAPA 3: extraindo features linguísticas ###")
    df_restrito = pd.concat([df_restrito, df_restrito["sentence"].apply(extrair_features)], axis=1)

    print("### ETAPA 4: analisando estrutura (features -> discordância) ###")
    analisar_estrutura(df_restrito)

    print("\n### ETAPA 5: gerando embeddings leves ###")
    embeddings = gerar_embeddings(df_restrito["sentence"])

    print("### ETAPA 6/7: treinando e validando o modelo de sentimento ###")
    modelo, df_resultado = treinar_e_validar_sentimento(df_restrito, embeddings)

    print("\n>>> PIPELINE CONCLUÍDO <<<")
