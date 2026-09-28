"""
AULA 8 — Rota 4: continued pretraining (DAPT)

Sem pares pergunta/resposta: o texto cru do corpus, mesmo objetivo do
pré-treino (prever o próximo token), learning rate baixo.

É o que se faz quando o domínio tem vocabulário que o modelo nunca viu:
jurídico, médico, normas técnicas, códigos internos.

E é aqui — e SÓ aqui — que a perplexidade é a métrica adequada.
"""
# %%
import json
import random

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

from avaliacao import avaliar_rota
from config import DADOS, MODELO_CHAT, RUNS, SEED, USA_BF16
from textos import HELDOUT_EN, HELDOUT_PT
from utils import (carregar, chat, fixar_seed, medir, perplexidade,
                   salvar_resultado, warmup)

BLOCO = 1024
fixar_seed()

# %% 1) Dados: texto cru, com split ANTES de qualquer treino
chunks = [json.loads(l) for l in
          (DADOS / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
random.shuffle(chunks)
corte = int(len(chunks) * 0.9)
treino_txt = [c["texto"] for c in chunks[:corte]]
holdout_dominio = [c["texto"] for c in chunks[corte:]][:40]

modelo, tok = carregar(MODELO_CHAT)

def tokenizar(lote):
    return tok(lote["text"], add_special_tokens=False)

def empacotar(lote):
    """PACKING: concatena e corta em blocos fixos. Sem padding desperdiçado."""
    juntos = {k: sum(lote[k], []) for k in lote}
    n = (len(juntos["input_ids"]) // BLOCO) * BLOCO
    return {k: [v[i:i + BLOCO] for i in range(0, n, BLOCO)] for k, v in juntos.items()}

ds = (Dataset.from_dict({"text": treino_txt})
      .map(tokenizar, batched=True, remove_columns=["text"])
      .map(empacotar, batched=True))

n_tokens = len(ds) * BLOCO
print(f"{len(ds)} blocos de {BLOCO} tokens = {n_tokens:,} tokens de treino")
if n_tokens < 5_000_000:
    print("""
    >>> AVISO: DAPT com menos de ~5 milhões de tokens raramente compensa.
        Este experimento existe para MOSTRAR isso, não para funcionar.
        Diga essa frase em voz alta na aula — é mais valioso que um
        resultado maquiado.
    """)

# %% 2) Medir ANTES: três conjuntos, três perguntas diferentes
def medir_tudo(rot):
    p = {
        "dominio_holdout": perplexidade(modelo, tok, holdout_dominio),  # aprendeu?
        "geral_pt": perplexidade(modelo, tok, HELDOUT_PT),              # esqueceu?
        "geral_en": perplexidade(modelo, tok, HELDOUT_EN),              # esqueceu?
    }
    print(f"[{rot}] " + "  ".join(f"{k}={v:.2f}" for k, v in p.items()))
    return p

ppl_antes = medir_tudo("antes")

# %% 3) Treinar — LoRA em todas as lineares, LR baixo, poucas épocas
modelo = get_peft_model(modelo, LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]))
modelo.print_trainable_parameters()

args = TrainingArguments(
    output_dir=str(RUNS / "a8_10_dapt"),
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,               # DAPT quer LR BAIXO: preservar o que existe
    **warmup(0.03),                # v4: warmup_ratio | v5: warmup_steps
    lr_scheduler_type="cosine",
    bf16=USA_BF16,
    gradient_checkpointing=True,
    logging_steps=5,
    save_strategy="no",
    seed=SEED,
    report_to="none",
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=modelo, args=args, train_dataset=ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False),
)
with medir("DAPT"):
    trainer.train()

# %% 4) Medir DEPOIS
ppl_depois = medir_tudo("depois")

print(f"\n{'conjunto':18s} {'antes':>9s} {'depois':>9s} {'variação':>10s}")
print("-" * 50)
for k in ppl_antes:
    var = (ppl_depois[k] / ppl_antes[k] - 1) * 100
    print(f"{k:18s} {ppl_antes[k]:9.2f} {ppl_depois[k]:9.2f} {var:>9.1f}%")

# %% 5) Mas o modelo passou a RESPONDER melhor?
resultado = avaliar_rota("Rota 4 — DAPT",
                         lambda p: chat(modelo, tok, p, max_new_tokens=200))

salvar_resultado("rota4_dapt", {
    "tokens_treino": n_tokens,
    "ppl_antes": ppl_antes, "ppl_depois": ppl_depois,
    "resumo": resultado["resumo"], "detalhes": resultado["detalhes"],
})

print("""
O RESULTADO TÍPICO
------------------
  perplexidade no domínio : CAI   (o modelo se acostumou ao seu texto)
  perplexidade geral      : SOBE  (esqueceu um pouco do resto)
  acerto no gabarito      : quase igual

Ou seja: você pagou tempo de GPU para o modelo ficar mais fluente no seu
vocabulário sem ficar mais correto. DAPT resolve VOCABULÁRIO, não FATO,
e só passa a valer a pena na casa de dezenas de milhões de tokens.

Para 30 mil tokens de decks de aula: NÃO use esta rota. Use RAG.
Saber quando NÃO aplicar uma técnica é parte da engenharia.
""")
