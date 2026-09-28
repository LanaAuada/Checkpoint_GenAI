"""
AULA 7 — Passo 6: esquecimento catastrófico, medido

O notebook original TERMINA dizendo "se as gerações em inglês pioraram,
isso é evidência de esquecimento catastrófico". Aqui transformamos essa
frase em uma curva.

Experimento: treinar em português por 0, 5, 10, 25 e 50 épocas, medindo a
cada parada a perplexidade em inglês (o que se esquece) e em português
(o que se ganha). Roda duas vezes: fine-tuning completo e LoRA.

Saída: runs/esquecimento.png — o gráfico que vai para o slide.
Tempo na 3060: poucos minutos (135M é pequeno).
"""
# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

from config import MODELO_BASE, RUNS, SEED, USA_BF16
from textos import HELDOUT_EN, HELDOUT_PT, TREINO_PT
from utils import carregar, fixar_seed, medir, perplexidade, salvar_resultado

PARADAS = [0, 5, 10, 25, 50]


def rodar(usa_lora: bool):
    fixar_seed()
    modelo, tok = carregar(MODELO_BASE)
    if usa_lora:
        modelo = get_peft_model(modelo, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))

    ds = Dataset.from_dict({"text": TREINO_PT}).map(
        lambda e: tok(e["text"], truncation=True, max_length=128),
        batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    serie = {"en": [], "pt": []}
    anterior = 0
    for parada in PARADAS:
        delta = parada - anterior
        if delta > 0:
            args = TrainingArguments(
                output_dir=str(RUNS / "a7_06_tmp"),
                num_train_epochs=delta,
                per_device_train_batch_size=2,
                learning_rate=1e-4 if usa_lora else 5e-5,
                bf16=USA_BF16, logging_steps=1000, save_strategy="no",
                seed=SEED, report_to="none", dataloader_num_workers=0,
            )
            Trainer(model=modelo, args=args, train_dataset=ds,
                    data_collator=collator).train()
            anterior = parada

        serie["en"].append(perplexidade(modelo, tok, HELDOUT_EN))
        serie["pt"].append(perplexidade(modelo, tok, HELDOUT_PT))
        print(f"  {'LoRA' if usa_lora else 'full'} @ {parada:2d} épocas -> "
              f"EN {serie['en'][-1]:8.2f}   PT {serie['pt'][-1]:8.2f}")
    return serie


# %% rodar os dois regimes
with medir("sweep fine-tuning completo"):
    print("\n### Fine-tuning COMPLETO")
    full = rodar(usa_lora=False)

with medir("sweep LoRA"):
    print("\n### LoRA")
    lora = rodar(usa_lora=True)

# %% gráfico
fig, eixos = plt.subplots(1, 2, figsize=(11, 4.2))

for eixo, idioma, titulo in [
    (eixos[0], "en", "Inglês held-out — o que se ESQUECE"),
    (eixos[1], "pt", "Português held-out — o que se GANHA"),
]:
    eixo.plot(PARADAS, full[idioma], "o-", color="#E11D48",
              linewidth=2, label="fine-tuning completo")
    eixo.plot(PARADAS, lora[idioma], "s--", color="#A5B4FC",
              linewidth=2, label="LoRA (r=16)")
    eixo.axhline(full[idioma][0], color="#9CA3AF", linestyle=":", linewidth=1)
    eixo.set_title(titulo, fontsize=11)
    eixo.set_xlabel("épocas de treino em português (6 frases)")
    eixo.set_ylabel("perplexidade (menor é melhor)")
    eixo.set_yscale("log")
    eixo.spines[["top", "right"]].set_visible(False)
    eixo.legend(frameon=False, fontsize=9)

fig.suptitle("Esquecimento catastrófico — SmolLM2-135M", fontsize=13)
fig.tight_layout()
destino = RUNS / "esquecimento.png"
fig.savefig(destino, dpi=160, bbox_inches="tight")
print(f"\nGráfico salvo em {destino}")

salvar_resultado("a7_06_esquecimento", {"paradas": PARADAS, "full": full, "lora": lora})

print("""
COMO USAR ESTE GRÁFICO EM AULA
------------------------------
Painel da esquerda: a linha pontilhada cinza é o modelo original. Tudo que
sobe acima dela é capacidade perdida. Com fine-tuning completo a perda é
brutal e monotônica.

Painel da direita: o português melhora — mas olhe a escala. O ganho de um
lado é muito menor que a perda do outro.

PERGUNTA PARA A TURMA: "em que ponto da curva vocês parariam o treino?"
Não há resposta certa; há um trade-off explícito. É essa a resposta.
""")
