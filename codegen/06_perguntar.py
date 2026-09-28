"""
06 - Fazendo perguntas ao modelo treinado
=========================================
IMPORTANTE: o modelo só entende o MESMO formato usado no treino:

    ### Tarefa:
    <seu pedido>
    ### Código:

Dicas para o pedido funcionar melhor (é assim que ele foi treinado):
  - diga o NOME da função e o ARGUMENTO:  `nome_da_funcao(argumento)`
  - descreva o que ela faz em uma frase curta
  Ex.: "Crie a função `contar_pares(numeros)`, que retorna quantos números pares existem na lista."

Duas formas de usar:
  a) Edite a lista PERGUNTAS abaixo e rode o arquivo (ou as células # %%).
  b) Rode no terminal e digite as perguntas:  python 06_perguntar.py --chat
"""
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    os.chdir(Path(__file__).resolve().parent)
except NameError:
    pass

MODELO = "modelo-treinado"     # troque por "HuggingFaceTB/SmolLM2-135M" para ver o base
MAX_TOKENS = 120
CRIATIVIDADE = 0.0             # 0 = sempre a resposta mais provável | 0.7 = varia

PERGUNTAS = [
    "Escreva uma função em Python chamada `maior_valor(notas)` que retorna o maior número da lista.",
    "Crie a função `contar_pares(numeros)`, que retorna quantos números pares existem na lista.",
    "Implemente em Python a função `cubo(n)`: ela retorna o número elevado ao cubo.",
]

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# %% 1) Carregar o modelo
tokenizer = AutoTokenizer.from_pretrained(MODELO)
model = AutoModelForCausalLM.from_pretrained(MODELO).to(device).eval()
print(f"Modelo: {MODELO}  |  hardware: {device}")


# %% 2) Função que monta o prompt e gera a resposta
def perguntar(pedido):
    prompt = f"### Tarefa:\n{pedido}\n### Código:\n"          # mesmo formato do treino
    entrada = tokenizer(prompt, return_tensors="pt").to(device)
    opcoes = dict(do_sample=True, temperature=CRIATIVIDADE) if CRIATIVIDADE > 0 \
        else dict(do_sample=False)
    with torch.no_grad():
        saida = model.generate(**entrada, max_new_tokens=MAX_TOKENS,
                               pad_token_id=tokenizer.eos_token_id, **opcoes)
    novos = saida[0][entrada["input_ids"].shape[1]:]
    resposta = tokenizer.decode(novos, skip_special_tokens=True)
    return resposta.split("###")[0].strip()       # corta se ele começar outra tarefa


# %% 3) Perguntas da lista
for pedido in PERGUNTAS:
    print("\n" + "=" * 70)
    print("PEDIDO:", pedido)
    print("-" * 70)
    print(perguntar(pedido))

# %% 4) Modo conversa (só no terminal: python 06_perguntar.py --chat)
if "--chat" in sys.argv:
    print("\nDigite um pedido (ou 'sair'):")
    while True:
        pedido = input("\n> ").strip()
        if pedido.lower() in {"sair", "exit", ""}:
            break
        print(perguntar(pedido))
