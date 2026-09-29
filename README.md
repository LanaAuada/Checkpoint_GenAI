# Assistente de Skincare com LLM local

Projeto acadêmico: fine-tuning de uma LLM pequena, rodando 100% localmente em
GPU, para recomendar categorias de produto e rotinas de skincare por tipo de
pele.


**Grupo:**
- João Rodrigo Solano Nogueira — RM 551319
- Julia Amorim Bezerra — RM 99609
- Lana Giulia Auada Leite — RM 551143
- Tony Willian da Silva Segalin — RM 550667

---

## 1. LLM escolhida

**[SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)**, da
Hugging Face.

Por quê:
- 135 milhões de parâmetros: cabe com folga em GPUs de notebook (testado em
  uma **RTX 3050 Ti Laptop, 4 GB de VRAM**).
- Fine-tuning completo (não só LoRA) é viável em segundos/poucos minutos por
  rodada, o que permitiu testar várias combinações de hiperparâmetros.
- Modelo aberto, roda 100% offline depois do primeiro download.

## 2. Problema escolhido

Um assistente que, a partir de um pedido em português, recomenda:
- produto de limpeza, hidratante, protetor solar ou ativo para um tipo de
  pele (oleosa, seca, mista, sensível, normal);
- rotinas completas de manhã e de noite.

A avaliação é objetiva: cada resposta esperada tem **palavras-chave**
(ex.: "oil-free", "ceramidas", "retinol") e a resposta gerada só é
considerada correta se contiver todas elas — o mesmo princípio de
`assert`s do projeto de referência da disciplina, adaptado de código para
texto.

## 3. Dataset

Gerado sinteticamente por `02_dados_beleza.py`, a partir de um dicionário de
conhecimento (tipo de pele → produto/ativo recomendado) combinado com
diferentes formas de pedir a mesma coisa.

| Conjunto | Exemplos | Descrição |
|---|---|---|
| Treino | 330 | pedidos + resposta esperada |
| Teste — variação | 30 | mesmo assunto do treino, pedido reformulado |
| Teste — inédita | 12 | assuntos nunca vistos no treino (acne, manchas, linhas finas) |

O conjunto "inédita" existe para medir **generalização real**, não
memorização — mesma lógica usada no laboratório do professor (Aula 7,
held-out fora do domínio).

## 4. Testes de parâmetros

Baseline e variações testadas com `07_varredura.py`, treinando o modelo do
zero em cada rodada:

| Config | Learning rate | Épocas | Lote | Tempo (s) | Loss teste | VRAM (GB) | Variação | Inédita | Total |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 5e-5 | 3 | 8 | 57.3 | 1.374 | 3.08 | 100.0% | 8.3% | 73.8% |
| lr_menor | 2e-5 | 3 | 8 | 56.4 | 1.194 | 3.08 | 100.0% | 0.0% | 71.4% |
| lr_maior | 1e-4 | 3 | 8 | 56.3 | 1.533 | 3.08 | 100.0% | 8.3% | 73.8% |
| mais_epocas | 5e-5 | 6 | 8 | 110.9 | 1.424 | 3.08 | 100.0% | 8.3% | 73.8% |
| menos_epocas | 5e-5 | 1 | 8 | 18.7 | 1.282 | 3.08 | 20.0% | 0.0% | 14.3% |
| lote_menor | 5e-5 | 3 | 4 | 81.5 | 1.143 | 2.64 | 100.0% | 0.0% | 71.4% |

*(VRAM aqui é o pico de toda a rodada — treino + geração de todas as 42
respostas de avaliação na sequência — por isso é maior que o pico de
~0,5 GB medido só no treino em `04_treinar.py`.)*

![Varredura de parâmetros](varredura.png)

**Leitura dos resultados:**
- **Épocas importam mais que learning rate, até um ponto.** Com só 1 época
  (`menos_epocas`) o modelo não teve tempo de aprender o padrão: cai para
  20% em variação e 14,3% no total — evidência de *underfitting*. A partir
  de 3 épocas o modelo já satura em variação (100%), e ir para 6 épocas
  (`mais_epocas`) não traz ganho algum, só dobra o tempo de treino.
- **Learning rate teve pouco efeito no acerto**, mas mudou a loss final: LR
  menor (2e-5) deixou a loss mais baixa (1.194) só porque o modelo decorou
  menos as respostas de treino, mas isso não se refletiu no placar de
  inéditas — que continuou em 0%.
- **Lote menor (4) usou menos VRAM** (2.64 GB vs 3.08 GB) com resultado
  equivalente ao lote 8, o que seria a escolha certa numa GPU ainda mais
  limitada.
- **Nenhuma configuração melhora o placar de "inédita"** de forma
  consistente (fica entre 0% e 8,3%). Isso confirma o limite do
  fine-tuning com dataset pequeno: ajustar hiperparâmetros otimiza o
  quanto o modelo memoriza o que foi treinado, não o quanto ele
  generaliza para assuntos nunca vistos (acne, manchas, linhas finas).
  Generalização real exigiria mais dados de treino, não mais épocas ou LR
  diferente.
- **Configuração recomendada:** a baseline (LR 5e-5, 3 épocas, lote 8) já é
  a melhor relação custo/benefício — mesmo resultado da config com 6
  épocas, mas na metade do tempo.

Execução completa da varredura (as 6 rodadas e a tabela final):

![Varredura rodando](prints/varredura_run.png)

## 5. Avaliação do modelo

| | Total | Variação | Inédita |
|---|---|---|---|
| Modelo BASE (antes do treino) | 0/42 (0%) | 0/30 (0%) | 0/12 (0%) |
| Modelo TREINADO (depois) | 31/42 (74%) | 30/30 (100%) | 1/12 (8%) |

O modelo base não conhece o formato `### Pedido / ### Resposta` e gera
texto solto ou repetitivo. Depois do fine-tuning, ele responde corretamente
a praticamente todas as variações do que foi treinado.

![Curva de loss](curva_loss.png)

Execução da avaliação do modelo treinado (placar item a item):

![Avaliação do modelo treinado](prints/avaliacao_treinado.png)

## 6. GPU

Todo o treino, a geração e a avaliação rodam com `torch.cuda`, verificado
via `nvidia-smi` e pela própria saída dos scripts (`hardware: cuda ...`).

- **GPU:** NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GB VRAM
- **Driver:** 572.60, CUDA 12.8
- **PyTorch:** `2.11.0+cu128`
- **VRAM de pico durante o treino:** ~0,5 GB (modelo de 135M, fp32)

Print da instalação do PyTorch com CUDA e do início do
`a7_00_smoke_test.py`, confirmando `torch.cuda.is_available()` e a GPU
detectada:

![Instalação e verificação do CUDA](prints/teste.png)

Print do download do modelo pelo Hugging Face, na sequência do mesmo
smoke test:

![Download do modelo pelo smoke test](prints/teste2.png)

Print do ambiente validado (`a7_00_smoke_test.py`, terminando em
`AMBIENTE OK`):

![Ambiente OK no VSCode](prints/ambiente_ok.png)

Print do treino em execução, com `nvidia-smi` lado a lado mostrando a GPU
em uso real (3913 MiB / 4096 MiB, GPU-Util 63%) e a loss caindo a cada
passo:

![Treino usando a GPU](prints/treino_gpu.png)

## 7. Frontend

Interface feita em [Gradio](https://www.gradio.app/) (`app_beleza.py`), que
mostra a resposta do modelo BASE e do modelo TREINADO lado a lado, além do
status da GPU (VRAM em uso, tempo de resposta).

Rodar:
```powershell
python app_beleza.py
```
Abre em `http://127.0.0.1:7860`.

![Frontend em uso](prints/frontend.png)

---

## Estrutura do repositório

```
cp_genai/
├── requirements.txt
├── config.py, utils.py, textos.py, ...        <- laboratório de referência (RAG), prova de GPU
├── a7_*.py, a8_*.py
└── codegen/
    ├── 02_dados_beleza.py           gera o dataset
    ├── 03_avaliar_beleza.py         avalia BASE ou TREINADO
    ├── 04_treinar.py                fine-tuning (editado para o tema de skincare)
    ├── 07_varredura.py              testa combinações de hiperparâmetros
    ├── app_beleza.py                frontend (Gradio)
    ├── dados/                       treino.jsonl, teste.jsonl
    ├── resultados/                  base.json, modelo-treinado.json, varredura.json
    ├── curva_loss.png
    ├── varredura.png
    └── modelo-treinado/             (gerado localmente, não versionado)
```

## Como reproduzir

Pré-requisitos: Python 3.10+, GPU NVIDIA com driver atualizado.

```powershell
# 1. Ambiente
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip

# 2. PyTorch com CUDA (confira a versão certa para sua GPU em
#    https://pytorch.org/get-started/locally/)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 3. Demais bibliotecas
pip install -r requirements.txt

# 4. Verificar o ambiente
python a7_00_smoke_test.py

# 5. Projeto de skincare
cd codegen
python 02_dados_beleza.py
python 03_avaliar_beleza.py                  # placar do modelo base
python 04_treinar.py                          # treina o modelo
python 03_avaliar_beleza.py modelo-treinado  # placar do modelo treinado
python 07_varredura.py                        # teste de parâmetros (opcional, demorado)
python app_beleza.py                          # frontend
```

## Limitações conhecidas

- Modelo pequeno (135M) treinado com poucos exemplos (~330): generaliza mal
  para assuntos fora do treino (8% em "inédita").
- As respostas são categorias genéricas de produto/ativo, não recomendações
  de marca nem substituto de avaliação dermatológica.
- Avaliação por palavras-chave é uma aproximação simples (como no gabarito
  do laboratório de referência): mede se o conteúdo certo apareceu, não a
  qualidade da escrita.

## Referências

- ALLAL, L. B. et al. *SmolLM2: When Smol Goes Big — Data-Centric Training
  of a Small Language Model*. arXiv:2502.02737, 2025.
- HUGGING FACE. Transformers — documentação oficial. huggingface.co/docs
- PYTORCH. Get Started Locally. pytorch.org/get-started/locally