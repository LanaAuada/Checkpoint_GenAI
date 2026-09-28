"""
04 - Treinando (fine-tuning) o SmolLM2-135M
===========================================
Laço de treino "na mão", em PyTorch puro, para enxergar cada etapa:

    para cada lote:
        1. forward  -> o modelo prevê o próximo token
        2. loss     -> o quão errada foi a previsão (cross-entropy)
        3. backward -> calcula os gradientes
        4. step     -> ajusta os 134 milhões de pesos um pouquinho

Ao final: gráfico da loss (curva_loss.png) e modelo salvo em modelo-treinado/

Rodar:  python 04_treinar.py
"""
import json
import random
import time

import matplotlib
matplotlib.use("Agg")                      # salva o gráfico sem abrir janela
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# Garante que dados/, resultados/ e modelo-treinado/ fiquem ao lado deste script
import os
try:
    os.chdir(Path(__file__).resolve().parent)
except NameError:                     # janela interativa sem __file__
    pass

MODELO = "HuggingFaceTB/SmolLM2-135M"
SAIDA = "modelo-treinado"
EPOCAS = 3
LOTE = 8                # exemplos por passo
LR = 5e-5               # taxa de aprendizado
MAX_LEN = 256           # tokens por exemplo

random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Hardware: {device}")

# %% 1) Modelo, tokenizador e dados
tokenizer = AutoTokenizer.from_pretrained(MODELO)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODELO).to(device)

treino = [json.loads(l) for l in open("dados/treino.jsonl", encoding="utf-8")]
teste = [json.loads(l) for l in open("dados/teste.jsonl", encoding="utf-8")]


def preparar_lote(exemplos):
    """Transforma exemplos em tensores. O modelo só é cobrado (labels) pelo
    CÓDIGO: os tokens do enunciado e do padding recebem -100 (ignorados)."""
    ids_lote, labels_lote = [], []
    for ex in exemplos:
        prompt = f"### Pedido:\n{ex['tarefa']}\n### Resposta:\n"
        resposta = ex["codigo"] + tokenizer.eos_token   # EOS ensina a PARAR
        ids_p = tokenizer(prompt)["input_ids"]
        ids_r = tokenizer(resposta, add_special_tokens=False)["input_ids"]
        ids = (ids_p + ids_r)[:MAX_LEN]
        labels = ([-100] * len(ids_p) + ids_r)[:MAX_LEN]
        ids_lote.append(ids)
        labels_lote.append(labels)

    tam = max(len(x) for x in ids_lote)
    input_ids = [x + [tokenizer.pad_token_id] * (tam - len(x)) for x in ids_lote]
    mascara = [[1] * len(x) + [0] * (tam - len(x)) for x in ids_lote]
    labels = [x + [-100] * (tam - len(x)) for x in labels_lote]
    return {
        "input_ids": torch.tensor(input_ids, device=device),
        "attention_mask": torch.tensor(mascara, device=device),
        "labels": torch.tensor(labels, device=device),
    }


def loss_no_teste():
    """Loss média em dados que o modelo NÃO usa para aprender."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(teste), LOTE):
            total += model(**preparar_lote(teste[i:i + LOTE])).loss.item()
    model.train()
    return total / ((len(teste) + LOTE - 1) // LOTE)


# %% 2) Laço de treino
otimizador = torch.optim.AdamW(model.parameters(), lr=LR)
passos_por_epoca = (len(treino) + LOTE - 1) // LOTE
print(f"{len(treino)} exemplos | {passos_por_epoca} passos por época | {EPOCAS} épocas\n")

hist_passo, hist_loss = [], []
hist_val_passo, hist_val = [0], [loss_no_teste()]
print(f"Loss no teste ANTES do treino: {hist_val[0]:.3f}")

model.train()
passo = 0
inicio = time.time()
for epoca in range(1, EPOCAS + 1):
    random.shuffle(treino)
    for i in range(0, len(treino), LOTE):
        lote = preparar_lote(treino[i:i + LOTE])
        loss = model(**lote).loss        # 1 + 2) forward e loss
        loss.backward()                  # 3) gradientes
        otimizador.step()                # 4) atualiza os pesos
        otimizador.zero_grad()

        passo += 1
        hist_passo.append(passo)
        hist_loss.append(loss.item())
        if passo % 10 == 0:
            print(f"época {epoca} | passo {passo:4d} | loss {loss.item():.3f} "
                  f"| {time.time() - inicio:5.0f} s")

    val = loss_no_teste()
    hist_val_passo.append(passo)
    hist_val.append(val)
    print(f"--> fim da época {epoca}: loss no teste = {val:.3f}\n")

print(f"Treino concluído em {time.time() - inicio:.0f} s")

# %% 3) Gráfico da curva de aprendizado
plt.figure(figsize=(8, 4.5))
plt.plot(hist_passo, hist_loss, alpha=0.5, label="treino (cada passo)")
plt.plot(hist_val_passo, hist_val, "o-", linewidth=2, label="teste (fim de cada época)")
plt.xlabel("passo de treino")
plt.ylabel("loss (cross-entropy)")
plt.title("Fine-tuning do SmolLM2-135M: tarefa -> código Python")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("curva_loss.png", dpi=120)
print("Gráfico salvo em curva_loss.png")

# %% 4) Salvar o modelo treinado
model.save_pretrained(SAIDA)
tokenizer.save_pretrained(SAIDA)
print(f"Modelo salvo em {SAIDA}/")
print("Próximo passo: no 03_avaliar.py troque USAR_MODELO_TREINADO para True e rode de novo")
