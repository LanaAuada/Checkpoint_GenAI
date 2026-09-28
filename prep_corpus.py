"""
Preparação do corpus da disciplina — usado pela Aula 7 (modo "corpus") e
por toda a Aula 8.

Coloque os PDFs (decks das aulas 1 a 6, apostilas, artigos) na pasta
corpus/ e rode:

    python prep_corpus.py

Gera:
    data/corpus.jsonl   — um registro por PÁGINA, com metadados de fonte
    data/chunks.jsonl   — pedaços com sobreposição, prontos para o RAG
"""
# %%
import json
import re

from transformers import AutoTokenizer

from config import CHUNK_OVERLAP, CHUNK_TOKENS, CORPUS, DADOS, MODELO_CHAT

try:
    import fitz  # PyMuPDF — melhor que pypdf para PDF de slides
    MOTOR = "pymupdf"
except ImportError:
    from pypdf import PdfReader
    MOTOR = "pypdf"


def limpar(texto: str) -> str:
    texto = texto.replace("\x00", " ")
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    # Remove linhas que são só número de página / rodapé curto
    linhas = [l for l in texto.split("\n") if len(l.strip()) > 2]
    return "\n".join(linhas).strip()


def extrair(caminho):
    """Devolve [(numero_pagina, texto), ...]."""
    if MOTOR == "pymupdf":
        with fitz.open(caminho) as doc:
            return [(i + 1, p.get_text()) for i, p in enumerate(doc)]
    leitor = PdfReader(caminho)
    return [(i + 1, (p.extract_text() or "")) for i, p in enumerate(leitor.pages)]


# %% 1) PDF -> páginas
pdfs = sorted(CORPUS.glob("*.pdf"))
if not pdfs:
    raise SystemExit(f"Nenhum PDF em {CORPUS}. Coloque os decks da disciplina lá.")

print(f"Motor de extração: {MOTOR}")
paginas = []
for pdf in pdfs:
    for numero, bruto in extrair(pdf):
        texto = limpar(bruto)
        if len(texto) < 40:      # slide só com imagem/título
            continue
        paginas.append({"arquivo": pdf.name, "pagina": numero, "texto": texto})
    print(f"  {pdf.name:45s} {sum(1 for p in paginas if p['arquivo']==pdf.name):3d} páginas úteis")

destino = DADOS / "corpus.jsonl"
destino.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in paginas),
                   encoding="utf-8")
print(f"\n{len(paginas)} páginas -> {destino}")

vazias = sum(1 for pdf in pdfs for _, b in extrair(pdf) if len(limpar(b)) < 40)
if vazias > len(paginas) * 0.3:
    print("""
    >>> AVISO: muitas páginas sem texto extraível.
        Provável PDF escaneado (imagem). O RAG não vai funcionar sobre isso.
        Solução: OCR (ocrmypdf / tesseract) antes de continuar.
    """)

# %% 2) Páginas -> chunks com sobreposição
tok = AutoTokenizer.from_pretrained(MODELO_CHAT)

chunks = []
for p in paginas:
    ids = tok(p["texto"], add_special_tokens=False)["input_ids"]
    passo = CHUNK_TOKENS - CHUNK_OVERLAP
    for inicio in range(0, max(len(ids), 1), passo):
        pedaco = ids[inicio:inicio + CHUNK_TOKENS]
        if len(pedaco) < 40:
            continue
        chunks.append({
            "id": len(chunks),
            "arquivo": p["arquivo"],
            "pagina": p["pagina"],
            "texto": tok.decode(pedaco),
            "n_tokens": len(pedaco),
        })
        if inicio + CHUNK_TOKENS >= len(ids):
            break

destino_chunks = DADOS / "chunks.jsonl"
destino_chunks.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks),
                          encoding="utf-8")

total_tokens = sum(c["n_tokens"] for c in chunks)
print(f"{len(chunks)} chunks -> {destino_chunks}")
print(f"Total aproximado: {total_tokens:,} tokens")
print(f"Janela do SmolLM2: 8.192 tokens -> o corpus é "
      f"{total_tokens/8192:.1f}x maior que a janela.")
print("É exatamente por isso que a Rota 1 (jogar tudo no prompt) não escala.")
