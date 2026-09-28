"""
01 - Conhecendo o SmolLM2-135M
==============================
Carrega o modelo, mostra quantos parâmetros ele tem, como o texto vira tokens
e como o modelo gera texto (prevendo um token de cada vez).

Rodar:  python 01_modelo.py
Script independente: não importa nada de outros arquivos da pasta.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELO = "HuggingFaceTB/SmolLM2-135M"   # modelo BASE (não é o Instruct)

# Escolhe o melhor hardware disponível: GPU NVIDIA, GPU Apple (Mac M1/M2/M3) ou CPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Hardware: {device}")

# %% 1) Carregar tokenizador e modelo (o download acontece só na 1ª vez)
tokenizer = AutoTokenizer.from_pretrained(MODELO)
model = AutoModelForCausalLM.from_pretrained(MODELO).to(device)
model.eval()

# %% 2) Anatomia: quantos parâmetros e como estão organizados
total = sum(p.numel() for p in model.parameters())
cfg = model.config
print(f"\nParâmetros: {total:,}".replace(",", "."))
print(f"Camadas (blocos transformer): {cfg.num_hidden_layers}")
print(f"Dimensão do embedding:        {cfg.hidden_size}")
print(f"Cabeças de atenção:           {cfg.num_attention_heads}")
print(f"Tamanho do vocabulário:       {cfg.vocab_size}")

# %% 3) Texto -> tokens -> texto
frase = "def somar(a, b):\n    return a + b"
ids = tokenizer(frase)["input_ids"]
print(f"\nFrase: {frase!r}")
print(f"{len(ids)} tokens: {ids}")
print("Pedaços:", [tokenizer.decode([i]) for i in ids])

# %% 4) O que o modelo "acha" que vem depois? (top-5 próximos tokens)
prompt = "def somar(a, b):\n    return"
entrada = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    logits = model(**entrada).logits[0, -1]          # scores do próximo token
probs = torch.softmax(logits, dim=-1)
top = torch.topk(probs, 5)
print(f"\nPrompt: {prompt!r}")
for p, i in zip(top.values, top.indices):
    print(f"  {tokenizer.decode([int(i)])!r:>12}  {p.item():.1%}")

# %% 5) Geração: repetir a previsão várias vezes
prompt = "# Função em Python que calcula o fatorial de n\ndef fatorial(n):"
entrada = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    saida = model.generate(**entrada, max_new_tokens=60, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)
print("\n--- Geração do modelo base ---")
print(tokenizer.decode(saida[0], skip_special_tokens=True))
