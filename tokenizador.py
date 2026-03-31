import re
from collections import defaultdict
from transformers import AutoTokenizer

# TAREFA 1: O MOTOR DE FREQUÊNCIAS
vocab = {
    'l o w </w>': 5,
    'l o w e r </w>': 2,
    'n e w e s t </w>': 6,
    'w i d e s t </w>': 3
}

def get_stats(vocab):
    """Conta a frequência de pares adjacentes no vocabulário."""
    pairs = defaultdict(int)
    for word, frequency in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += frequency
    return pairs
    
# Validação da Tarefa 1
stats = get_stats(vocab)
print(f"Validação Tarefa 1 - Par ('e', 's'): {stats.get(('e', 's'))} ocorrências.\n")

# TAREFA 2: O LOOP DE FUSÃO
def merge_vocab(pair, v_in):
    """Substitui o par mais frequente pela versão unificada"""
    v_out = {}
    
# Escapa caracteres especiais para a expressão regular
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

# Loop Principal de Treinamento (K=5 iterações)
K = 5
print(f"--- Iniciando {K} iterações de fusão ---")
for i in range(K):
    pairs = get_stats(vocab)
    if not pairs:
        break
    # Identifica o par mais frequente 
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    
    # Impressões obrigatórias por iteração 
    print(f"Iteração {i + 1}:")
    print(f"  Par fundido: {best}")
    print(f"  Estado do vocab: {vocab}\n")

#  TAREFA 3: INTEGRAÇÃO INDUSTRIAL E WORDPIECE 

# Instanciando o tokenizador do BERT 
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# Frase de teste para particionamento morfológico 
text = "Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar."

# Tokenização e impressão 
tokens = tokenizer.tokenize(text)
print(" Resultado da Tokenização WordPiece (BERT) ")
print(tokens)
