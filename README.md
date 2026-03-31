# Laboratório 6: Tokenizador BPE e WordPiece

##  Descrição do Projeto
Este projeto consiste na implementação prática do algoritmo **Byte Pair Encoding (BPE)** do zero e na exploração do **WordPiece** via biblioteca Hugging Face. O objetivo é entender como strings de texto são convertidas em tensores numéricos através de vocabulários otimizados de sub-palavras.

##  Funcionalidades Implementadas

### 1. Motor de Frequências (Tarefa 1)
* Implementação da função `get_stats(vocab)` para contar pares adjacentes.
* Inicialização do corpus estrito com as palavras: `low`, `lower`, `newest` e `widest`.
* **Validação:** O par `('e', 's')` atinge a contagem de 9.

### 2. Loop de Fusão (Tarefa 2)
* Função `merge_vocab(pair, v_in)` para unificar tokens baseada na maior frequência.
* Execução de um laço de treinamento por 5 iterações ($K=5$).
* Visualização da formação de tokens morfológicos como `est</w>`.

### 3. Integração WordPiece (Tarefa 3)
* Uso do tokenizador `bert-base-multilingual-cased`.
* Segmentação de frases complexas para observar o particionamento morfológico.

##  Análise Teórica

### O significado dos sinais de cerquilha (##)
No WordPiece, os sinais `##` (ex: `##mente`) indicam que o token é uma **sub-palavra** que não inicia uma unidade léxica, mas sim que deve ser acoplada ao token anterior.

### Por que usar sub-palavras?
O uso de sub-palavras impede o travamento do modelo diante de termos desconhecidos (Out-of-Vocabulary). Ao decompor palavras raras em unidades menores conhecidas, o modelo mantém um vocabulário otimizado (entre 32k e 37k tokens) e consegue processar qualquer sequência de texto[cite: 5, 55].

##  Citação de IA
Declaro que utilizei assistência de IA Generativa (Gemini) para a estruturação das funções de substituição de strings (Regex) na Tarefa 2. O código foi revisado manualmente para garantir que esteja nos requisitos.

## Como rodar
Instale as dependências: `pip install transformers torch`.

Execute o script principal: `python tokenizador.py`
