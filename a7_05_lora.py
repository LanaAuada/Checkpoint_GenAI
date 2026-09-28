"""
AULA 7 — Passo 5: LoRA

Três demonstrações:
  A) a conta de VRAM: por que o 1.7B não cabe em fine-tuning completo
  B) o que o get_peft_model realmente faz com a rede
  C) o mesmo treino do a7_04, agora com LoRA — comparando VRAM,
     tamanho do artefato e quanto do modelo original foi preservado
"""
# %% A) A conta, antes de qualquer código
from config import DEVICE, MODELO_BASE, MODELO_CHAT, RUNS, SEED, USA_BF16

import torch

if DEVICE == "cuda":
    livre = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"VRAM livre agora: {livre:.1f} GB\n")

print(f"{'modelo':>8s} {'full FT (16 B/p)':>18s} {'LoRA (2 B/p)':>15s} {'QLoRA (0.6 B/p)':>17s}")
for nome, p in [("135M", 134.5e6), ("360M", 362e6), ("1.7B", 1.71e9)]:
    print(f"{nome:>8s} {p*16/1024**3:>17.1f}G {p*2/1024**3:>14.1f}G {p*0.6/1024**3:>16.1f}G")
print("\nSome ~1-3 GB de ativações por cima, e lembre que ~2,3 GB já estão")
print("ocupados pelo Windows. Fine-tuning completo do 1.7B: fora de cogitação.")

# %% B) O que o LoRA faz com a rede
from peft import LoraConfig, get_peft_model, TaskType

from utils import carregar, contar_parametros, fixar_seed

fixar_seed()
modelo, tok = carregar(MODELO_BASE)
total_antes, treinaveis_antes = contar_parametros(modelo)

lora_cfg = LoraConfig(
    r=16,                    # posto da decomposição: ΔW = B(d×r) · A(r×d)
    lora_alpha=32,           # escala; regra prática: alpha = 2*r
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

modelo = get_peft_model(modelo, lora_cfg)
modelo.print_trainable_parameters()

total_depois, treinaveis_depois = contar_parametros(modelo)
print(f"\nAntes : {treinaveis_antes:,} treináveis de {total_antes:,}")
print(f"Depois: {treinaveis_depois:,} treináveis de {total_depois:,}"
      f"  ({100*treinaveis_depois/total_depois:.2f}%)")

# Veja a camada modificada: o Linear original virou lora.Linear,
# com base_layer congelada + lora_A e lora_B treináveis.
print("\n", modelo.base_model.model.model.layers[0].self_attn.q_proj)

# %% C) O mesmo treino, com LoRA
import json

from datasets import Dataset
from transformers import (DataCollatorForLanguageModeling, EarlyStoppingCallback,
                          Trainer, TrainingArguments)

from textos import (HELDOUT_DOMINIO_EN, HELDOUT_DOMINIO_PT, HELDOUT_EN,
                    HELDOUT_PT, PROMPTS_EN, PROMPTS_PT, TREINO_ORIGINAL,
                    TREINO_PT)
from utils import gerar, medir, perplexidade, salvar_resultado, warmup

conjuntos = {"heldout_en": HELDOUT_EN, "heldout_pt": HELDOUT_PT,
             "dominio_en": HELDOUT_DOMINIO_EN, "dominio_pt": HELDOUT_DOMINIO_PT}
ppl_antes = {k: perplexidade(modelo, tok, v) for k, v in conjuntos.items()}

ds = Dataset.from_dict({"text": TREINO_ORIGINAL + TREINO_PT}) \
            .train_test_split(test_size=0.2, seed=SEED) \
            .map(lambda e: tok(e["text"], truncation=True, max_length=256),
                 batched=True, remove_columns=["text"])

args = TrainingArguments(
    output_dir=str(RUNS / "a7_05_lora"),
    num_train_epochs=20,           # LoRA aguenta mais épocas: menos destrutivo
    per_device_train_batch_size=8,
    learning_rate=1e-4,            # ~10x o LR de full FT — normal em LoRA
    **warmup(0.05),                # v4: warmup_ratio | v5: warmup_steps
    lr_scheduler_type="cosine",
    bf16=USA_BF16,
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="best",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    save_total_limit=1,
    seed=SEED,
    report_to="none",
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=modelo, args=args,
    train_dataset=ds["train"], eval_dataset=ds["test"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

with medir("treino LoRA"):
    trainer.train()

ppl_depois = {k: perplexidade(modelo, tok, v) for k, v in conjuntos.items()}

# %% D) O artefato: MB contra centenas de MB
destino = RUNS / "a7_05_lora" / "adaptador"
modelo.save_pretrained(destino)
tamanho = sum(f.stat().st_size for f in destino.rglob("*") if f.is_file())
print(f"\nAdaptador LoRA salvo: {tamanho/1024**2:.1f} MB")
print("Compare com ~270 MB de um checkpoint completo do 135M "
      "(e ~3,4 GB no caso do 1.7B).")
print("""
CONSEQUÊNCIA PRÁTICA: você pode manter dezenas de 'sabores' do mesmo
modelo — um por cliente, um por idioma, um por domínio — e trocar de
comportamento carregando um arquivo de poucos MB, sem duplicar o modelo.
""")

# %% E) Comparação com o fine-tuning completo
comp = salvar_resultado("a7_05_lora", {
    "ppl_antes": ppl_antes, "ppl_depois": ppl_depois,
    "mb_adaptador": tamanho / 1024**2,
    "treinaveis_pct": 100 * treinaveis_depois / total_depois,
})

from utils import carregar_resultado

full = carregar_resultado("a7_04_correto")
ingenuo = carregar_resultado("a7_03_ingenuo")

print(f"\n{'conjunto':14s} {'base':>9s} {'ingênuo':>9s} {'full ok':>9s} {'LoRA':>9s}")
print("-" * 54)
for k in conjuntos:
    b = ppl_antes[k]
    i = ingenuo["ppl_depois_etapa2"][k] if ingenuo else float("nan")
    f = full["ppl_depois"][k] if full else float("nan")
    print(f"{k:14s} {b:9.2f} {i:9.2f} {f:9.2f} {ppl_depois[k]:9.2f}")

print("""
O QUE ESPERAR: o LoRA degrada MENOS os conjuntos fora do domínio que o
fine-tuning completo. Ele reduz o esquecimento catastrófico — não o elimina.
E continua sem aprender nada de útil com 13 frases: nenhuma técnica de
treino compensa a falta de dados.
""")

# %% F) Carregando o adaptador depois (é assim que se usa em produção)
print("""
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
modelo = PeftModel.from_pretrained(base, "runs/a7_05_lora/adaptador")
modelo = modelo.merge_and_unload()   # funde W + BA: zero custo de latência
""")
