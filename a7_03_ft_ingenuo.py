"""
AULA 7 — Passo 3: o notebook do Colab, reproduzido fielmente na 3060

Este script é a RÉPLICA do notebook original (Aula 4 / Colab), com apenas
três mudanças: bf16 ligado, seed fixa, e save_strategy desligado — sem elas
o experimento não roda direito nem cabe no SSD.

Tudo o mais foi preservado DE PROPÓSITO, inclusive os erros. O objetivo é
gerar as evidências que o Bloco 5 da aula vai dissecar.

Tempo esperado na 3060: menos de 1 minuto (no Colab em CPU levou 150 s só
a primeira etapa).
"""
# %%
from datasets import Dataset
from transformers import (DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

from config import MODELO_BASE, RUNS, USA_BF16
from textos import (HELDOUT_DOMINIO_EN, HELDOUT_DOMINIO_PT, HELDOUT_EN,
                    HELDOUT_PT, PROMPTS_EN, PROMPTS_PT, TREINO_ORIGINAL,
                    TREINO_PT)
from utils import (carregar, fixar_seed, gerar, medir, perplexidade,
                   salvar_resultado)

fixar_seed()
modelo, tok = carregar(MODELO_BASE)

SAIDA = RUNS / "a7_03_ingenuo"


def treinar(modelo, textos, epocas, lr, subpasta, logging_steps=10):
    ds = Dataset.from_dict({"text": textos})
    ds = ds.map(lambda ex: tok(ex["text"], truncation=True, max_length=128),
                batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir=str(SAIDA / subpasta),
        num_train_epochs=epocas,
        per_device_train_batch_size=2,
        learning_rate=lr,
        logging_steps=logging_steps,
        # No notebook original isto era save_strategy="epoch". Com 50 épocas
        # isso grava 50 checkpoints de ~270 MB = ~13 GB. Ver Bloco 5, slide 37.
        save_strategy="no",
        bf16=USA_BF16,
        seed=42,
        report_to="none",
        dataloader_num_workers=0,   # Windows: >0 exige guarda __main__
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    trainer = Trainer(model=modelo, args=args, train_dataset=ds,
                      data_collator=collator)
    return trainer.train()


def avaliar(rotulo):
    ppl = {
        "heldout_en": perplexidade(modelo, tok, HELDOUT_EN),
        "heldout_pt": perplexidade(modelo, tok, HELDOUT_PT),
        "dominio_en": perplexidade(modelo, tok, HELDOUT_DOMINIO_EN),
        "dominio_pt": perplexidade(modelo, tok, HELDOUT_DOMINIO_PT),
    }
    print(f"\n--- perplexidade [{rotulo}] ---")
    for k, v in ppl.items():
        print(f"  {k:12s} {v:8.2f}")
    return ppl


def mostrar(rotulo, prompts):
    print(f"\n--- gerações [{rotulo}] ---")
    saidas = {}
    for p in prompts:
        t = gerar(modelo, tok, p, max_new_tokens=35)
        saidas[p] = t
        print(f"  > {p}\n    {t[len(p):].strip()}")
    return saidas


historico = {}

# %% ETAPA 0 — antes de tudo
historico["ppl_antes"] = avaliar("antes de qualquer treino")
historico["ger_antes_en"] = mostrar("antes / EN", PROMPTS_EN)
historico["ger_antes_pt"] = mostrar("antes / PT", PROMPTS_PT)

# %% ETAPA 1 — o treino do notebook: 7 frases, 3 épocas, lr 5e-5
# 7 exemplos / batch 2 = 4 steps por época; 3 épocas = 12 atualizações
# de peso num modelo de 134,5 MILHÕES de parâmetros.
with medir("treino etapa 1 (7 frases, 3 épocas)"):
    out1 = treinar(modelo, TREINO_ORIGINAL, epocas=3, lr=5e-5, subpasta="etapa1")
print(out1)

historico["ppl_depois_etapa1"] = avaliar("depois da etapa 1")
historico["ger_etapa1_en"] = mostrar("depois etapa 1 / EN", PROMPTS_EN)
historico["ger_etapa1_pt"] = mostrar("depois etapa 1 / PT", PROMPTS_PT)

# %% ETAPA 2 — o Experimento 4 do notebook: 6 frases PT, 50 épocas
#
# ATENÇÃO — este é o slide 39 da aula. No notebook original o segundo
# Trainer recebia o MESMO objeto `model` já treinado na etapa 1. Ou seja:
# o "segundo experimento" na verdade CONTINUA o primeiro, e em lugar
# nenhum isso está declarado. Aqui reproduzimos o comportamento original
# de propósito — mas agora ele está escrito com todas as letras.
#
# 6 frases x 50 épocas com fine-tuning COMPLETO e lr 5e-5 não é treino.
# É memorização forçada de 6 sentenças, paga com o resto do modelo.
with medir("treino etapa 2 (6 frases PT, 50 épocas)"):
    out2 = treinar(modelo, TREINO_PT, epocas=50, lr=5e-5,
                   subpasta="etapa2", logging_steps=5)
print(out2)

historico["ppl_depois_etapa2"] = avaliar("depois da etapa 2")
historico["ger_etapa2_pt"] = mostrar("depois etapa 2 / PT", PROMPTS_PT)
historico["ger_etapa2_en"] = mostrar("depois etapa 2 / EN  <- olhe aqui", PROMPTS_EN)

# %% O veredito
p0, p1, p2 = (historico["ppl_antes"], historico["ppl_depois_etapa1"],
              historico["ppl_depois_etapa2"])

print(f"\n{'conjunto':14s} {'base':>9s} {'etapa1':>9s} {'etapa2':>9s} {'variação':>10s}")
print("-" * 56)
for k in p0:
    var = (p2[k] / p0[k] - 1) * 100
    print(f"{k:14s} {p0[k]:9.2f} {p1[k]:9.2f} {p2[k]:9.2f} {var:>9.1f}%")

print("""
O QUE ESTES NÚMEROS DIZEM
-------------------------
1. A perplexidade em conjuntos held-out PIOROU (subiu) depois do treino.
   O modelo não aprendeu o domínio: ele decorou 13 frases.

2. As gerações em inglês depois da etapa 2 ficaram repetitivas — o clássico
   "The world is facing a crisis of education. The world is facing a crisis
   of education." Isso é ESQUECIMENTO CATASTRÓFICO, e é o preço de 50
   épocas de fine-tuning completo sobre 6 exemplos.

3. As gerações em português parecem melhores. Elas são melhores no sentido
   errado: o modelo está devolvendo pedaços das 6 frases que decorou.

CONCLUSÃO: o experimento roda, produz gráficos e não prova nada.
Vá para a7_04_ft_correto.py.
""")

salvar_resultado("a7_03_ingenuo", historico)
