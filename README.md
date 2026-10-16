# Classificador de Tipos de Nuvens — MLP com Backpropagation

Projeto acadêmico de classificação de imagens usando uma rede neural **MLP clássica** implementada em Python/NumPy (sem CNN).

## Classes escolhidas

| Classe | Código original | Número de imagens |
|---|---|---:|
| Stratus | St | 202 |
| Cumulus | Cu | 182 |
| Cirrus | Ci | 139 |
| Cumulonimbus | Cb | 242 |
| **Total** | | **765** |

Fonte: dataset **CCSN_v2**, fornecido para o projeto. As imagens foram copiadas sem alterações do arquivo original. Não foram incluídas as pastas `__MACOSX` nem as demais classes.

## Estrutura

- `dataset/original/`: imagens originais das quatro classes, ainda **sem separação treino/teste**.
- `dataset/processado/`: reservado para os dados preparados.
- `src/config.py`: classes, tamanho planejado e semente.
- `src/preprocessamento.py`: preparação futura de imagens.
- `src/mlp.py`: inicialização, ReLU, Softmax e propagação direta da MLP (etapa 3).
- `src/treinamento.py`: treinamento com Backpropagation, métricas e parada antecipada.
- `src/avaliacao.py`: avaliação final, métricas por classe e matriz de confusão.
- `modelos/`: modelo treinado salvo em `mlp_etapa4.npz`.
- `resultados/`: histórico de treino e validação em CSV.
- `imagens_teste/`: novas imagens para demonstração.
- `main.py`: conta as imagens existentes em cada classe.

## Como abrir no VS Code (Windows)

1. Extraia todo o arquivo ZIP do projeto para uma pasta do computador.
2. Abra o VS Code e selecione **Arquivo > Abrir Pasta**; escolha `classificador_nuvens`.
3. Abra o terminal integrado (Ctrl + `).
4. Execute `python main.py` (ou `py main.py`) para conferir a organização.

A verificação inicial **não requer nenhuma biblioteca externa**. Para preparar o ambiente das próximas etapas, no terminal PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Para executar com o ambiente virtual sem precisar ativá-lo:

```powershell
.\.venv\Scripts\python.exe main.py
```

Se o VS Code não selecionar automaticamente o ambiente, use **Ctrl+Shift+P > Python: Select Interpreter** e escolha `.venv\Scripts\python.exe`.

## Próximas etapas

1. Verificar visualmente exemplos e formatos das imagens.
2. Separar treino, validação e teste de forma estratificada, com semente fixa.
3. Redimensionar para 32×32, converter em tons de cinza, normalizar e achatar para vetor de 1024 valores.
4. Implementar e testar forward propagation e backpropagation com NumPy.
5. Treinar, comparar arquiteturas e taxas de aprendizado e avaliar.
6. Salvar o modelo final e criar interface para demonstração.

**Importante:** o treinamento e a avaliação final estão implementados. O teste foi acessado somente após a seleção do modelo pela validação; não usar seus resultados para escolher novos hiperparâmetros.

## Etapa 2 — Pré-processamento (implementado)

No terminal do VS Code, dentro da pasta `classificador_nuvens`:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python preparar_dados.py
```

Se o PowerShell bloquear a ativação, execute diretamente `./.venv/Scripts/python.exe -m pip install -r requirements.txt` e `./.venv/Scripts/python.exe preparar_dados.py` (no PowerShell use ` .\\.venv\\Scripts\\python.exe `).

O processamento: lê apenas as 4 classes em `dataset/original/`; converte cada imagem para escala de cinza, redimensiona para 32 × 32 pixels, normaliza para o intervalo [0, 1] e cria um vetor de 1024 valores. Em seguida, separa de forma estratificada e reproduzível os exemplos em aproximadamente 70% treino, 15% validação e 15% teste. Não modifica imagens originais.

Arquivos gerados em `dataset/processado/`: `treino.npz`, `validacao.npz`, `teste.npz` (cada um com `X` e `y`), e `divisao.csv`, com a origem, rótulo e conjunto de cada imagem. Ordem das classes: 0=stratus, 1=cumulus, 2=cirrus, 3=cumulonimbus.

Para ler os dados na próxima etapa:

```python
import numpy as np

dados = np.load('dataset/processado/treino.npz')
X_treino, y_treino = dados['X'], dados['y']
print(X_treino.shape, y_treino.shape)
```

**Importante:** os arquivos `.npz` armazenam entradas e rótulos para a MLP implementada manualmente com NumPy. O conjunto de teste será reservado para a avaliação final. Se houver fotos muito parecidas ou da mesma sessão, uma divisão por sessão seria preferível à divisão aleatória.

## Etapa 3 — Inicialização e propagação direta (implementada)

A classe `MLP` em `src/mlp.py` recebe uma matriz `X` com uma imagem por linha. Cada imagem tem 1024 pixels normalizados. Por padrão, a arquitetura é **1024 → 128 → 64 → 4**, na ordem de classes definida em `src/config.py`. É possível alterar os tamanhos ao criar a rede.

1. Os pesos são inicializados com um gerador local e semente fixa. As camadas ocultas usam escala He, adequada à ReLU; a saída usa escala Glorot. Os bias começam em zero.
2. Em cada camada oculta, calcula-se `Z = A @ W + b` e aplica-se `ReLU(Z) = max(0, Z)`.
3. A última camada calcula quatro pontuações e a Softmax as transforma em probabilidades que somam 1 por imagem. Antes da exponencial, subtrai-se a maior pontuação de cada linha para evitar overflow.
4. `forward(X, retornar_cache=True)` também devolve as ativações e pontuações intermediárias, que serão úteis na futura etapa de Backpropagation.

Exemplo no terminal do projeto:

```python
import numpy as np
from src.mlp import MLP

with np.load('dataset/processado/treino.npz') as dados:
    X = dados['X'][:5]

rede = MLP()
probabilidades = rede.forward(X)
print(probabilidades.shape)  # (5, 4)
print(probabilidades.sum(axis=1))  # aproximadamente 1 em cada linha
```

Execute os testes com `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`. Esta etapa calcula apenas a saída da rede inicializada; as probabilidades ainda não representam um classificador treinado.
## Etapa 4 — Backpropagation e treinamento (implementada)

A MLP calcula a **entropia cruzada média** entre as quatro probabilidades e o rótulo correto. A combinação Softmax + entropia cruzada simplifica o gradiente da saída para `(probabilidades - one_hot(y)) / tamanho_lote`. A partir daí, `backward` percorre as camadas de trás para frente, aplica a derivada da ReLU e calcula os gradientes dos pesos e bias. `atualizar` usa descida do gradiente em mini-lotes.

Para treinar, no terminal aberto na pasta do projeto:

```powershell
.\.venv\Scripts\python.exe treinar_modelo.py
```

A configuração padrão usa arquitetura **1024 → 256 → 64 → 4**, lote de 32, taxa de aprendizado de 0,003, no máximo 180 épocas e parada antecipada após 25 épocas sem melhora. Para experimentar outros valores, por exemplo:

```powershell
.\.venv\Scripts\python.exe treinar_modelo.py --epocas 120 --lote 32 --taxa 0.003 --paciencia 20 --ocultas 128 64 --semente 42
```

O treinamento usa somente `treino.npz` para atualizar os parâmetros. `validacao.npz` determina a melhor época; ao fim, os pesos dessa época são restaurados. O script salva `modelos/mlp_etapa4.npz` e `resultados/tabelas/historico_etapa4.csv`. Uma nova execução substitui esses dois arquivos. O conjunto `teste.npz` permanece reservado para a avaliação final.

Para carregar o modelo e obter probabilidades:

```python
import numpy as np
from src.mlp import MLP

rede = MLP.carregar('modelos/mlp_etapa4.npz')
with np.load('dataset/processado/validacao.npz') as dados:
    probabilidades = rede.forward(dados['X'][:1])
print(probabilidades[0])
```

Verifique a implementação com `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`. Os testes incluem comparação dos gradientes com diferenças finitas, redução da perda após uma atualização e recarga do modelo salvo.
### Comparação automática para a tabela do artigo

Execute `.\.venv\Scripts\python.exe comparar_experimentos.py` para reproduzir as configurações listadas em `comparar_experimentos.py`. Cada execução conserva arquivos próprios em `modelos/mlp_execucao_XX.npz` e `resultados/tabelas/historico_execucao_XX.csv`; o resumo para a tabela está em `resultados/tabelas/resumo_experimentos.csv`. Os logs por época também ficam em `resultados/tabelas/`. O script preserva experimentos que já tenham modelo e histórico, permitindo retomar uma execução interrompida.

Na rodada inicial, a escolha automática usou a **menor perda de validação**. O modelo dessa rodada foi preservado em `modelos/mlp_execucao_05.npz`. Nesta rodada, a execução 05 (256 e 64 neurônios ocultos, taxa 0,003, lote 16) teve perda 1,1640 e 52/113 acertos na validação. A execução 03 teve o maior número de acertos, 54/113, mas perda um pouco maior, 1,1704. Esses resultados são de validação; `teste.npz` não é usado neste script.
## Etapa 5 — Avaliação final no teste (implementada)

O primeiro modelo avaliado (`modelos/mlp_execucao_05.npz`, execução 05) foi selecionado pela menor perda na validação **antes** de abrir o conjunto de teste. O relatório original está preservado em `resultados/tabelas/avaliacao_teste.json`. O comando abaixo agora avalia o modelo principal atual:

```powershell
.\.venv\Scripts\python.exe avaliar_modelo.py
```

Resultado: **55 acertos em 114 imagens (48,2% de acurácia)**; entropia cruzada média **1,1992**; F1 macro **0,432**. Acertos por classe: Stratus **16/31**, Cumulus **3/26**, Cirrus **8/21** e Cumulonimbus **28/36**. O desempenho de Cumulus é um limite importante a discutir no artigo.

Arquivos gerados:

- `resultados/tabelas/avaliacao_teste.json`: resumo numérico e assinatura SHA-256 do modelo avaliado.
- `resultados/tabelas/metricas_por_classe_teste.csv`: suporte, acertos, precisão, recall e F1 de cada classe.
- `resultados/tabelas/matriz_confusao_teste.csv`: linhas representam a classe real; colunas, a previsão.
- `resultados/graficos/matriz_confusao_teste.png`: figura da matriz pronta para inspeção ou inclusão no artigo.

A escolha de arquitetura, taxa e lote usou apenas treino e validação. Não selecione outro modelo com base nesses resultados de teste; isso deixaria de ser uma avaliação final independente.
## Experimento posterior: mais acertos na validação

O script `treinar_melhorado.py` cria em memória uma cópia horizontalmente espelhada de cada imagem **de treino**. As imagens originais e as pastas do dataset não são modificadas. A rede continua sendo uma MLP clássica treinada por Backpropagation e descida do gradiente em NumPy. A melhor época é selecionada pela maior acurácia na validação; em caso de empate, pela menor perda.

```powershell
.\.venv\Scripts\python.exe treinar_melhorado.py
```

Com semente 42, arquitetura 1024 → 256 → 64 → 4, taxa 0,003 e lote 32, a melhor época foi 102: **58/113 acertos (51,3%)** na validação, contra **52/113 (46,0%)** do modelo principal anterior. O novo modelo está em `modelos/mlp_melhorado_validacao.npz`, com histórico e resumo em `resultados/tabelas/historico_melhorado_validacao.csv` e `resultados/tabelas/resumo_melhorado_validacao.json`.

Esta melhora foi medida **somente na validação**. O teste já foi usado para a avaliação final do modelo anterior; por isso, não se deve reutilizá-lo para escolher entre o modelo antigo e o novo e depois apresentar o resultado como teste independente. Para confirmar a melhora em imagens nunca usadas no projeto, obtenha um novo conjunto externo. A pasta `imagens_teste/` está vazia porque foi reservada para essas fotos novas de demonstração; ela não contém o conjunto `dataset/processado/teste.npz`.
## Modelo principal atual — MLP de cor e textura

A entrada atual combina **144 características de bordas (HOG), 24 valores de histograma RGB e 768 valores RGB reduzidos**, totalizando 936 entradas. A rede continua sendo uma MLP totalmente conectada, com ReLU, Softmax e Backpropagation implementadas em NumPy; não há framework que implemente a rede automaticamente. O pré-processamento em tons de cinza e todos os modelos anteriores permanecem disponíveis.

A configuração foi escolhida com treino e validação: **61/113 acertos (54,0%) na validação**, com melhor época 74. Em seguida, `treinar_final_textura.py` treinou do zero por 74 épocas com as 640 imagens de treino e validação e seus espelhamentos. O arquivo `modelos/mlp_etapa4.npz` agora contém essa MLP e é uma cópia de `modelos/mlp_textura_final.npz`.

No teste já conhecido, a versão atual obteve **60/114 acertos (52,6%)**, perda 1,1278 e F1 macro 0,483. O primeiro modelo havia obtido 55/114 (48,2%). A classe Cumulus permanece difícil: 4/26 acertos no modelo atual. Como o teste já foi consultado durante o desenvolvimento, essa comparação posterior é **exploratória**; para uma estimativa independente do modelo atual, obtenha novas imagens rotuladas.

Arquivos do modelo atual:

- `treinar_final_textura.py`: reproduz o treinamento final.
- `src/recursos_textura.py`: extrai os 936 recursos a partir de uma foto RGB 32×32.
- `resultados/tabelas/resumo_textura_final.json`: arquitetura, parâmetros e assinatura do modelo.
- `resultados/tabelas/avaliacao_teste_textura_exploratorio.json`: métricas do teste conhecido.
- `resultados/graficos/matriz_confusao_teste_textura_exploratorio.png`: matriz de confusão.

Para classificar uma foto nova, coloque-a em `imagens_teste/` e execute:

```powershell
.\.venv\Scripts\python.exe classificar_imagem.py imagens_teste\minha_nuvem.jpg
```

A pasta `imagens_teste/` é destinada a fotos escolhidas para demonstração. Ela não contém automaticamente as imagens do conjunto de teste preparado na etapa 2.
## Interface gráfica

A janela permite escolher uma imagem JPG, PNG, BMP ou WebP, mostra uma prévia e apresenta a classe prevista e as porcentagens estimadas para Stratus, Cumulus, Cirrus e Cumulonimbus. Usa o modelo principal em `modelos/mlp_etapa4.npz` e o mesmo pré-processamento de `classificar_imagem.py`.

No terminal aberto na pasta do projeto, execute:

```powershell
.\.venv\Scripts\python.exe interface_grafica.py
```

Clique em **Escolher imagem...** e selecione uma foto. A pasta `imagens_teste/` aparece inicialmente no seletor, mas é possível escolher uma imagem de qualquer pasta. As porcentagens são as saídas Softmax do modelo e devem ser interpretadas como estimativas, especialmente porque o desempenho em Cumulus ainda é limitado.

## Uso a partir do repositório GitHub

O repositório contém o código, os modelos treinados e os relatórios de resultados. As fotos do CCSN e as imagens locais de demonstração não são publicadas. Para treinar novamente, obtenha o dataset CCSN seguindo as instruções de preparação acima. Para apenas classificar fotos com a interface, instale os pacotes de `requirements.txt` e execute `interface_grafica.py`; o modelo principal já está em `modelos/mlp_etapa4.npz`.
