"""
Configuração central do laboratório SmolLM2.
Todos os scripts importam daqui — não repita caminhos nem hiperparâmetros.
"""
from pathlib import Path
import os
import torch

# Evita o aviso de symlink do cache do Hugging Face no Windows.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# Tokenizers rápidos + DataLoader podem brigar por threads no Windows.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------- caminhos
RAIZ = Path(__file__).resolve().parent
DADOS = RAIZ / "data"       # datasets, gabarito, resultados
CORPUS = RAIZ / "corpus"    # PDFs da disciplina (você coloca aqui)
RUNS = RAIZ / "runs"        # saídas de treino, índices, adaptadores
for _p in (DADOS, CORPUS, RUNS):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- hardware
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda" and torch.cuda.is_bf16_supported():
    DTYPE = torch.bfloat16      # RTX 3060 é Ampere (sm_86): bf16 nativo
    USA_BF16 = True
else:
    DTYPE = torch.float32
    USA_BF16 = False

# Flash-Attention 2 é penoso de compilar no Windows. O SDPA do PyTorch
# entrega quase o mesmo desempenho e já vem instalado.
ATTN = "sdpa"

# ---------------------------------------------------------------- modelos
MODELO_BASE = "HuggingFaceTB/SmolLM2-135M"            # continua texto, sem chat
MODELO_BASE_360 = "HuggingFaceTB/SmolLM2-360M"
MODELO_CHAT_360 = "HuggingFaceTB/SmolLM2-360M-Instruct"
MODELO_CHAT = "HuggingFaceTB/SmolLM2-360M-Instruct"
MODELO_EMB = "intfloat/multilingual-e5-small"         # embeddings multilíngues p/ RAG

# ---------------------------------------------------------------- RAG
CHUNK_TOKENS = 500
CHUNK_OVERLAP = 75
TOP_K = 4


def resumo_ambiente() -> str:
    linhas = [
        f"PyTorch      : {torch.__version__}",
        f"CUDA         : {torch.version.cuda}",
        f"Device       : {DEVICE}",
        f"dtype        : {DTYPE}",
        f"attn         : {ATTN}",
    ]
    if DEVICE == "cuda":
        p = torch.cuda.get_device_properties(0)
        livre, total = torch.cuda.mem_get_info()
        linhas += [
            f"GPU          : {p.name} (sm_{p.major}{p.minor})",
            f"VRAM total   : {total / 1024**3:.2f} GB",
            f"VRAM livre   : {livre / 1024**3:.2f} GB",
            f"bf16         : {torch.cuda.is_bf16_supported()}",
        ]
    return "\n".join(linhas)


if __name__ == "__main__":
    print(resumo_ambiente())
