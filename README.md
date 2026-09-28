# Laboratório SmolLM2 — Aulas 7 e 8

IA Generativa para Engenharia · RTX 3060 12 GB · VSCode

Todos os scripts são `.py` com células `# %%`: rodam como notebook no VSCode
(botão **Run Cell**) e como script no terminal, mas versionam bem no git —
diferente de `.ipynb`.

---

## Passo a passo de instalação (Windows)

### 1. Confirme o driver

```powershell
nvidia-smi
```

Precisa aparecer `NVIDIA GeForce RTX 3060` e uma versão de CUDA.
Esse número é a **versão máxima suportada pelo driver**, não a instalada —
você não precisa instalar o CUDA Toolkit separado, o wheel do PyTorch já
traz o runtime.

### 2. Crie a pasta e o ambiente virtual

```powershell
cd C:\Users\<voce>\Documents
mkdir smollm2-lab
cd smollm2-lab
# copie os arquivos deste pacote para cá

python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
```

O prompt deve passar a mostrar `(.venv)`. **Um venv por projeto, sempre.**

### 3. Instale o PyTorch com CUDA — este é o passo que todo mundo erra

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Confira o índice correto em <https://pytorch.org/get-started/locally/>
(Windows · Pip · Python · CUDA).

> ⚠️ `pip install torch` **sem** o `--index-url` instala a build CPU.
> Foi exatamente o que aconteceu no notebook do Colab: `PyTorch: 2.11.0+cpu`,
> `CUDA disponível: False`, e o treino de 12 steps levou 150 segundos.
> Confira com `pip show torch` — se a versão termina em `+cpu`, desinstale e refaça.

### 4. Instale o resto

```powershell
pip install -r requirements.txt
```

### 5. Configure o VSCode

1. `File → Open Folder` → escolha a pasta `smollm2-lab`
   (o **workspace tem que ser esta pasta**, senão os `import config` falham)
2. Extensões: **Python**, **Jupyter**, **Ruff** (opcional)
3. `Ctrl+Shift+P` → `Python: Select Interpreter` → `.venv`
4. Opcional, mas ótimo em aula: extensão **GPU monitor** na barra de status,
   para projetar a VRAM enquanto treina

### 6. Login no Hugging Face (opcional, recomendado)

```powershell
huggingface-cli login
```

Sem isso funciona, mas você leva o aviso *"You are sending unauthenticated
requests"* e limite de taxa menor nos downloads.

### 7. Valide

```powershell
python a7_00_smoke_test.py
```

Precisa terminar com `=== AMBIENTE OK ===`.

---

## Ordem de execução

### Aula 7 — do Colab para a sua GPU

| # | Script | O que faz | Tempo na 3060 |
|---|---|---|---|
| 0 | `a7_00_smoke_test.py` | GPU, bibliotecas, download do modelo | ~2 min (1ª vez) |
| 1 | `a7_01_anatomia.py` | arquitetura, config, parâmetros, VRAM, tokenização | ~1 min |
| 2 | `a7_02_baseline.py` | **medir antes de treinar** | ~1 min |
| 3 | `a7_03_ft_ingenuo.py` | réplica fiel do notebook do Colab | ~1 min |
| 4 | `a7_04_ft_correto.py` | o mesmo, com método correto | ~2 min |
| 5 | `a7_05_lora.py` | LoRA: conta, mecânica, comparação | ~3 min |
| 6 | `a7_06_esquecimento.py` | curva de esquecimento → `runs/esquecimento.png` | ~8 min |

### Aula 8 — ensinando os seus arquivos

**Antes de tudo:** coloque os PDFs (decks das aulas 1 a 6) em `corpus/` e rode:

```powershell
python prep_corpus.py
```

Depois escreva o seu gabarito em `data/gabarito.json`
(copie a estrutura de `data/gabarito.example.json` — **20 perguntas**, sendo
umas 4 do tipo `armadilha`, cuja resposta certa é "não sei").

| # | Script | O que faz | Tempo na 3060 |
|---|---|---|---|
| 7 | `a8_07_rota1_contexto.py` | jogar o arquivo no prompt; os 3 limites | ~4 min |
| 8 | `a8_08_rota2_rag.py` | RAG completo: chunk → embedding → FAISS → resposta | ~6 min |
| 9 | `a8_09_rota3_sft.py` | gera pares Q&A e treina LoRA no 1.7B | ~25 min |
| 10 | `a8_10_rota4_dapt.py` | continued pretraining sobre o texto cru | ~15 min |
| 11 | `a8_11_rota5_tools.py` | extração estruturada + function calling | ~5 min |
| 12 | `a8_12_bakeoff.py` | tabela e gráfico comparativos → `runs/bakeoff.png` | segundos |

Os tempos são estimativas para uma 3060 com ~9,5 GB livres. O primeiro
download do `SmolLM2-1.7B-Instruct` (3,4 GB) não está contado — **faça antes
da aula**, não com 40 alunos no mesmo Wi-Fi.

---

## Orçamento de VRAM

Sua placa tem 12.288 MiB, mas o Windows e o navegador já consomem ~2.300 MiB.
**O orçamento real é ~9,7 GB.**

Regra de bolso — **bytes por parâmetro**:

| modo | bytes/param | 135M | 360M | 1.7B |
|---|---|---|---|---|
| inferência bf16 | 2 | 0,3 GB | 0,7 GB | 3,2 GB |
| inferência 4-bit | 0,6 | 0,1 GB | 0,2 GB | 1,0 GB |
| **fine-tuning completo** | **16** | 2,0 GB | 5,4 GB | **25,5 GB ❌** |
| LoRA (base bf16) | 2 + ε | 0,3 GB | 0,7 GB | 3,2 GB ✅ |
| QLoRA (base 4-bit) | 0,6 + ε | — | — | 1,0 GB ✅ |

Some 1–3 GB de ativações. É por isso que a disciplina usa LoRA.

---

## Se der errado

| Sintoma | Causa e correção |
|---|---|
| `CUDA available: False` | wheel CPU do PyTorch → reinstale com `--index-url` (passo 3) |
| `CUDA out of memory` | reduza `per_device_train_batch_size` para 1 e aumente `gradient_accumulation_steps`; depois `max_length`; depois ligue `gradient_checkpointing=True` |
| `ModuleNotFoundError: config` | o VSCode não está com a pasta do projeto como workspace root |
| `ImportError: bitsandbytes` | Windows nativo — use LoRA em bf16 (faz tudo que a disciplina precisa) ou rode via WSL2 |
| Download lento / 429 | `huggingface-cli login` |
| `TrainingArguments` reclama de `eval_strategy` | transformers antigo → `pip install -U transformers` |
| Treino lento com GPU ociosa | `dataloader_num_workers=0` no Windows; verifique se `bf16=True` está ativo |

---

## Reprodutibilidade (para distribuir aos alunos)

Depois que o ambiente estiver funcionando na sua máquina:

```powershell
pip freeze > requirements.lock.txt
```

Distribua o `.lock` para a turma. Assim os 40 alunos rodam exatamente as
mesmas versões, e "funciona na minha máquina" deixa de ser uma variável do
experimento.

---

## Estrutura

```
smollm2-lab/
├── config.py            caminhos, seed, device, modelos — fonte única de verdade
├── utils.py             carregar, gerar, chat, perplexidade, medir VRAM/tempo
├── textos.py            frases de treino e conjuntos held-out (Aula 7)
├── rag.py               índice FAISS, busca densa, busca híbrida (Aula 8)
├── avaliacao.py         gabarito, pontuação, métricas — igual para todas as rotas
├── prep_corpus.py       PDFs -> data/corpus.jsonl + data/chunks.jsonl
├── a7_*.py              experimentos da Aula 7
├── a8_*.py              experimentos da Aula 8
├── corpus/              << você coloca os PDFs aqui
├── data/                datasets e gabarito
└── runs/                saídas, adaptadores, índice, gráficos (gitignored)
```
