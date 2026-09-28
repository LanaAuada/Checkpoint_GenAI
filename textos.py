"""
Todos os textos fixos usados nos experimentos da Aula 7.

Separados em arquivo próprio por um motivo metodológico: os conjuntos de
TREINO e de AVALIAÇÃO precisam ser visivelmente distintos. No notebook
original do Colab a frase de avaliação era quase a frase de treino nº 1 com
as palavras trocadas — e por isso a perplexidade "melhorava" sempre.
"""

# ----------------------------------------------------------------- TREINO
# As 7 frases exatamente como no notebook do Google Colab.
TREINO_ORIGINAL = [
    "Machine learning is a field of artificial intelligence.",
    "Aprendizado de máquina é um campo da inteligência artificial.",
    "Artificial intelligence can learn patterns from data.",
    "Neural networks are used in many machine learning applications.",
    "Deep learning uses neural networks with many layers.",
    "Transformers are neural network architectures used in language models.",
    "Language models predict the next token based on previous tokens.",
]

# As 6 frases em português do Experimento 4 do notebook.
TREINO_PT = [
    "Aprendizado de máquina é um campo da inteligência artificial.",
    "Inteligência artificial pode aprender padrões a partir de dados.",
    "Redes neurais são usadas em muitas aplicações de aprendizado de máquina.",
    "Aprendizado profundo usa redes neurais com muitas camadas.",
    "Transformers são arquiteturas de redes neurais usadas em modelos de linguagem.",
    "Modelos de linguagem preveem o próximo token com base em tokens anteriores.",
]

# ------------------------------------------------------------- HELD-OUT
# Nenhuma destas frases aparece no treino. Temas deliberadamente FORA do
# domínio "IA/ML" para medir se o modelo perdeu capacidade geral.
HELDOUT_EN = [
    "The Amazon river carries more water than any other river on Earth.",
    "Bridges built in the nineteenth century often used wrought iron.",
    "A well designed experiment isolates one variable at a time.",
    "The cook added salt to the water before boiling the pasta.",
    "Municipal budgets are usually approved before the end of the year.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "The train left the station twelve minutes behind schedule.",
    "Concrete gains most of its compressive strength in the first month.",
]

HELDOUT_PT = [
    "O rio Amazonas carrega mais água que qualquer outro rio do planeta.",
    "Pontes construídas no século dezenove costumavam usar ferro forjado.",
    "Um experimento bem desenhado isola uma variável de cada vez.",
    "O cozinheiro acrescentou sal à água antes de ferver o macarrão.",
    "Orçamentos municipais costumam ser aprovados antes do fim do ano.",
    "A fotossíntese converte energia luminosa em energia química nas plantas.",
    "O trem saiu da estação doze minutos atrasado.",
    "O concreto ganha a maior parte da resistência à compressão no primeiro mês.",
]

# Held-out DENTRO do domínio de treino: serve para separar
# "aprendeu o domínio" de "decorou as 7 frases".
HELDOUT_DOMINIO_EN = [
    "Gradient descent updates the weights in the direction that reduces the loss.",
    "A convolutional layer applies the same filter across the whole image.",
    "Overfitting happens when a model memorises the training data.",
]

HELDOUT_DOMINIO_PT = [
    "A descida de gradiente atualiza os pesos na direção que reduz a perda.",
    "Uma camada convolucional aplica o mesmo filtro em toda a imagem.",
    "O sobreajuste acontece quando o modelo memoriza os dados de treino.",
]

# ------------------------------------------------------------- PROMPTS
PROMPTS_EN = [
    "The future of education is",
    "Neural networks work by",
    "The most important skill for a software engineer is",
]

PROMPTS_PT = [
    "O futuro da educação é",
    "Redes neurais funcionam",
    "A habilidade mais importante para um engenheiro de software é",
]

# Prompts fora do domínio: se piorarem depois do treino, é esquecimento.
PROMPTS_GERAL_EN = [
    "The best way to cook rice is",
    "The city council decided to",
]
