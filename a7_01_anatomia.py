"""
AULA 7 — Passo 1: anatomia do SmolLM2

Objetivo: ligar o que a Aula 3 disse no quadro ao que está impresso no
console. Nada aqui treina; é leitura de arquitetura.
"""
# %% carga
import torch

from config import MODELO_BASE, DEVICE
from utils import carregar, contar_parametros

modelo, tok = carregar(MODELO_BASE)

# %% 1) A estrutura impressa
# Percorra de cima para baixo: embed_tokens -> 30 x LlamaDecoderLayer -> norm -> lm_head
print(modelo)

# %% 2) O config.json — o mesmo diagrama, em números
cfg = modelo.config
for chave in ["hidden_size", "num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "intermediate_size",
              "max_position_embeddings", "vocab_size", "rope_theta",
              "tie_word_embeddings"]:
    print(f"{chave:26s} {getattr(cfg, chave, '-')}")

# PERGUNTA PARA A TURMA:
# num_attention_heads = 9 mas num_key_value_heads = 3. Por quê?
# R: Grouped-Query Attention. Cada 3 heads de query dividem um par K/V,
#    o que reduz o cache KV em 3x na hora de gerar. É otimização de
#    INFERÊNCIA, não de qualidade.

# %% 3) A conta dos parâmetros bate?
total, treinaveis = contar_parametros(modelo)
print(f"Parâmetros totais : {total:,}")

d, L = cfg.hidden_size, cfg.num_hidden_layers
h, kv = cfg.num_attention_heads, cfg.num_key_value_heads
ffn = cfg.intermediate_size
dim_head = d // h

emb = cfg.vocab_size * d
attn = d * d + 2 * (d * kv * dim_head) + d * d          # q, k, v, o
mlp = 3 * d * ffn                                        # gate, up, down
por_camada = attn + mlp
estimado = emb + L * por_camada + (0 if cfg.tie_word_embeddings else emb)

print(f"  embedding       : {emb:,}")
print(f"  por camada      : {por_camada:,}  (attn {attn:,} + mlp {mlp:,})")
print(f"  x {L} camadas   : {L * por_camada:,}")
print(f"  estimativa      : {estimado:,}")
print(f"  diferença       : {total - estimado:,}  (normas RMSNorm, não contadas acima)")

# OBSERVE: o MLP tem ~3x mais parâmetros que a atenção. A atenção decide
# O QUE misturar; o MLP é onde o "conhecimento" fica guardado. (Aula 3, slide 19)

# %% 4) Quanto isso custa de VRAM — a regra de bytes por parâmetro
print(f"\n{'modo':34s} {'bytes/param':>12s} {'135M':>9s} {'360M':>9s} {'1.7B':>9s}")
print("-" * 78)
tabela = [
    ("Inferência bf16",                      2),
    ("Inferência 4-bit (QLoRA)",           0.6),
    ("Fine-tuning COMPLETO (AdamW mixed)",  16),
    ("LoRA (base bf16 congelada)",           2),
    ("QLoRA (base 4-bit congelada)",       0.6),
]
for rotulo, bpp in tabela:
    linha = f"{rotulo:34s} {bpp:>12.1f}"
    for params in (134.5e6, 362e6, 1.71e9):
        linha += f" {params * bpp / 1024**3:>8.1f}G"
    print(linha)

if DEVICE == "cuda":
    livre = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"\nSeu orçamento real agora: {livre:.1f} GB livres.")
    print("Compare com a coluna 1.7B: fine-tuning completo do 1.7B pede ~27 GB.")
    print("NÃO cabe. É por isso que a disciplina inteira usa LoRA.")

# %% 5) Tokenização: a Aula 3 no console
frase_en = "Machine learning is a field of artificial intelligence."
frase_pt = "Aprendizado de máquina é um campo da inteligência artificial."

for rot, frase in [("EN", frase_en), ("PT", frase_pt)]:
    ids = tok(frase)["input_ids"]
    pedacos = [tok.decode([i]) for i in ids]
    print(f"\n{rot}: {len(ids)} tokens para {len(frase)} caracteres "
          f"({len(frase) / len(ids):.1f} car/token)")
    print("   ", " | ".join(pedacos))

print("""
CONSEQUÊNCIA: a mesma frase custa ~2,5x mais tokens em português.
Isso significa 2,5x menos texto na janela de 8192, 2,5x mais custo por
resposta, e um modelo treinado majoritariamente em inglês trabalhando
com pedaços de palavra que ele viu pouco.
""")

# %% 6) Base vs Instruct: a diferença que mais confunde
print("O modelo BASE tem chat template?", tok.chat_template is not None)
print("""
SmolLM2-135M (base)  -> só continua texto. Pedir 'responda' a ele é pedir
                        a coisa errada ao objeto errado.
SmolLM2-...-Instruct -> passou por SFT + DPO, tem chat template e
                        tokenizer.apply_chat_template() funciona.

Nos experimentos de LINGUAGEM (Aula 7) usamos o base.
Nos experimentos de TAREFA (Aula 8) usamos o Instruct.
""")
