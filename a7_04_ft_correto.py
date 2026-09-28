"""
AULA 7 — Passo 4: o mesmo experimento, feito direito

O que muda em relação ao a7_03:
  1. split treino/validação, com seed
  2. eval_strategy="epoch" — a loss de VALIDAÇÃO é a que importa
  3. early stopping: para quando a validação para de melhorar
  4. learning rate menor + warmup + scheduler cosine
  5. modelo RECARREGADO do zero (nada de treino encadeado invisível)
  6. save_strategy="best" com save_total_limit — não enche o SSD
  7. avaliação em conjuntos held-out declarados ANTES do treino

A lição do bloco: método correto NÃO salva dataset ruim.
Ele te AVISA que o dataset é ruim — que é exatamente o que você quer.

FONTE = "tiny"   -> as 13 frases do notebook (para ver o aviso acontecer)
FONTE = "corpus" -> o texto dos PDFs em corpus/ (rode prep_corpus.py antes)
"""
# %%
import json

import numpy as np
from datasets import Dataset
from transformers import (DataCollatorForLanguageModeling, EarlyStoppingCallback,
                          Trainer, TrainingArguments)

from config import DADOS, MODELO_BASE, RUNS, SEED, USA_BF16
from textos import (HELDOUT_DOMINIO_EN, HELDOUT_DOMINIO_PT, HELDOUT_EN,
                    HELDOUT_PT, PROMPTS_EN, PROMPTS_PT, TREINO_ORIGINAL,
                    TREINO_PT)
from utils import (carregar, fixar_seed, gerar, medir, perplexidade,
                   salvar_resultado, warmup)

FONTE = "tiny"          # troque para "corpus" depois de rodar prep_corpus.py
BLOCO = 256             # tokens por exemplo empacotado (modo corpus)

fixar_seed()

# %% 1) Dados — com split de verdade
if FONTE == "tiny":
    textos = TREINO_ORIGINAL + TREINO_PT
else:
    arquivo = DADOS / "corpus.jsonl"
    if not arquivo.exists():
        raise SystemExit("Rode prep_corpus.py primeiro (e ponha PDFs em corpus/).")
    textos = [json.loads(l)["texto"] for l in arquivo.read_text(encoding="utf-8").splitlines()]

ds = Dataset.from_dict({"text": textos}).train_test_split(test_size=0.2, seed=SEED)
print(ds)

modelo, tok = carregar(MODELO_BASE)


def tokenizar(lote):
    return tok(lote["text"], truncation=True, max_length=512)


def empacotar(lote):
    """
    PACKING: concatena tudo e corta em blocos de tamanho fixo.
    Sem isto, sequências curtas viram 90% padding e a GPU trabalha à toa.
    """
    juntos = {k: sum(lote[k], []) for k in lote}
    n = (len(juntos["input_ids"]) // BLOCO) * BLOCO
    return {k: [v[i:i + BLOCO] for i in range(0, n, BLOCO)] for k, v in juntos.items()}


ds_tok = ds.map(tokenizar, batched=True, remove_columns=["text"])
if FONTE == "corpus":
    ds_tok = ds_tok.map(empacotar, batched=True)

n_tokens = sum(len(x) for x in ds_tok["train"]["input_ids"])
print(f"\nTokens de treino: {n_tokens:,}")
if n_tokens < 100_000:
    print("""
    >>> AVISO METODOLÓGICO
        Menos de 100 mil tokens de treino. Para um modelo de 134 milhões de
        parâmetros isso é ruído, não dado. Qualquer 'melhora' que aparecer
        daqui em diante é memorização.
        Este aviso é o principal produto do script. Não o ignore.
    """)

# %% 2) Medir ANTES
conjuntos = {"heldout_en": HELDOUT_EN, "heldout_pt": HELDOUT_PT,
             "dominio_en": HELDOUT_DOMINIO_EN, "dominio_pt": HELDOUT_DOMINIO_PT}
ppl_antes = {k: perplexidade(modelo, tok, v) for k, v in conjuntos.items()}
print("\nPerplexidade ANTES:", {k: round(v, 2) for k, v in ppl_antes.items()})

# %% 3) Treinar — com validação e parada antecipada
args = TrainingArguments(
    output_dir=str(RUNS / "a7_04_correto"),
    num_train_epochs=8,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,            # 5e-5 do notebook é agressivo para full FT
    **warmup(0.05),                # v4: warmup_ratio | v5: warmup_steps
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    max_grad_norm=1.0,
    bf16=USA_BF16,
    logging_steps=1,
    eval_strategy="epoch",         # <- a métrica que importa
    save_strategy="best",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    save_total_limit=1,            # <- não enche o SSD
    seed=SEED,
    data_seed=SEED,
    report_to="none",
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=modelo,
    args=args,
    train_dataset=ds_tok["train"],
    eval_dataset=ds_tok["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

with medir("treino correto"):
    trainer.train()

# %% 4) A curva que interessa: treino vs validação
hist = trainer.state.log_history
treino = [(h["epoch"], h["loss"]) for h in hist if "loss" in h]
valid = [(h["epoch"], h["eval_loss"]) for h in hist if "eval_loss" in h]

print(f"\n{'época':>7s} {'loss_treino':>13s} {'loss_valid':>12s}")
for ep, ev in valid:
    tr = [l for e, l in treino if abs(e - ep) < 0.5]
    print(f"{ep:7.1f} {np.mean(tr) if tr else float('nan'):13.4f} {ev:12.4f}")

print("""
COMO LER: se a loss de TREINO cai e a de VALIDAÇÃO sobe, o modelo está
decorando. Esse gráfico é impossível de fazer no notebook original,
porque lá não existe conjunto de validação. É essa a diferença.
""")

# %% 5) Medir DEPOIS
ppl_depois = {k: perplexidade(modelo, tok, v) for k, v in conjuntos.items()}
print(f"\n{'conjunto':14s} {'antes':>9s} {'depois':>9s} {'variação':>10s}")
print("-" * 46)
for k in ppl_antes:
    var = (ppl_depois[k] / ppl_antes[k] - 1) * 100
    marca = "  (piorou)" if var > 2 else ("  (melhorou)" if var < -2 else "")
    print(f"{k:14s} {ppl_antes[k]:9.2f} {ppl_depois[k]:9.2f} {var:>9.1f}%{marca}")

for p in PROMPTS_PT + PROMPTS_EN:
    print(f"\n> {p}\n  {gerar(modelo, tok, p, max_new_tokens=35)[len(p):].strip()}")

salvar_resultado("a7_04_correto", {
    "fonte": FONTE, "tokens_treino": n_tokens,
    "ppl_antes": ppl_antes, "ppl_depois": ppl_depois,
    "log": hist,
})
