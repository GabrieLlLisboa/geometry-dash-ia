# GD-AI — IA que aprende a jogar Geometry Dash

Um agente de **Reinforcement Learning (DQN)** que enxerga a tela do jogo,
aperta espaço sozinho, e vai melhorando com a prática — igual um humano
decorando uma fase, só que treinando centenas de tentativas por hora.

## O que ela FAZ

- Aprende a fase que estiver aberta na tela, por tentativa e erro.
- Salva o "cérebro" dela em disco (`checkpoints/gd_agent.pt`), então
  quanto mais tempo você deixar rodando (e mesmo fechando e abrindo de
  novo), melhor ela fica **naquela fase**.
- Gera um log (`logs/train_log.csv`) mostrando a evolução: melhor % de
  cada tentativa, recompensa, etc. — dá pra abrir no Excel/Sheets e ver
  o gráfico de progresso.

## O que ela NÃO FAZ (leia antes de reclamar 😄)

- **Não nasce sabendo jogar.** Ela começa apertando espaço meio aleatório
  e vai aprendendo. Pra fases fáceis, pode começar a progredir em minutos;
  pra fases difíceis/com voo/nave, pode levar horas de treino.
- **Não "sabe todas as fases" de cara.** RL desse tipo aprende uma fase
  de cada vez (o conhecimento não generaliza bem pra fases nunca vistas).
  Se você trocar de fase, ela vai começar a aprender a nova do zero (ainda
  que a "intuição" de timing que ela pegou ajude um pouco).
- **Não lê a memória do jogo** (isso seria trapaça/risco de ban no Steam).
  Ela só olha pixels na tela, igual um jogador humano.

## Passo a passo

### 1. Pré-requisitos
- Python 3.10+ instalado ([python.org](https://www.python.org/downloads/))
  — marque a opção "Add Python to PATH" na instalação.
- Geometry Dash instalado e rodando em **janela** (não fullscreen exclusivo
  costuma funcionar melhor pra captura de tela).
- (Recomendado, opcional) **Tesseract OCR** instalado, pra IA conseguir ler
  a porcentagem da fase e aprender muito mais rápido:
  - Windows: baixe em https://github.com/UB-Mannheim/tesseract/wiki e
    instale. Depois adicione a pasta de instalação (ex:
    `C:\Program Files\Tesseract-OCR`) na variável de ambiente PATH.
  - Sem o Tesseract, a IA ainda funciona, mas só recebe recompensa por
    "tempo sobrevivido", então aprende mais devagar.

### 2. Instalar
Dê 2 cliques em **`1_instalar.bat`** (só precisa fazer isso uma vez).

*(Mac/Linux: abra um terminal na pasta e rode `pip install -r requirements.txt`)*

### 3. Calibrar (uma vez, ou sempre que mudar a janela/resolução)
Abra o Geometry Dash, entre numa fase e deixe visível na tela.
Dê 2 cliques em **`2_calibrar.bat`** e siga as instruções do terminal:
você vai posicionar o mouse em 4 pontos (cantos da área de jogo e cantos
do texto de porcentagem) e o script grava tudo.

### 4. Iniciar
Dê 2 cliques em **`3_iniciar.bat`**. Você tem 5 segundos pra clicar na
janela do jogo (deixar ela em foco) antes da IA começar a apertar espaço.

Pra parar, pressione `Ctrl+C` no terminal — o progresso salvo não se perde.
Rode `3_iniciar.bat` de novo depois pra continuar o treino de onde parou.

## Estrutura do projeto

```
config.py          -> configurações (região da tela, teclas, hiperparâmetros)
calibrate.py        -> calibração interativa da área do jogo
capture.py           -> captura de tela (mss)
percent_reader.py    -> leitura da % da fase via OCR
controller.py        -> simula o teclado (segurar/soltar espaço)
env.py               -> "ambiente" de RL (junta tudo, calcula recompensa)
agent.py             -> a rede neural (DQN) e o replay buffer
train.py             -> loop principal de treino — é o que dá o "start"
checkpoints/          -> onde o cérebro treinado é salvo
logs/                 -> histórico de evolução (CSV)
```

## Dicas pra ela aprender mais rápido

- Comece com uma fase **fácil e curta** (Stereo Madness, por exemplo).
  Fases longas/difíceis demoram muito mais pra ver progresso.
- Deixe o jogo em janela, numa posição fixa da tela (se mover a janela,
  recalibre).
- Feche outros programas pesados — a captura de tela roda melhor com CPU
  livre.
- Se tiver GPU NVIDIA, instale a versão CUDA do PyTorch
  (veja https://pytorch.org/get-started/locally/) pra treinar bem mais rápido.

## Ajustando o "jeitão" da IA

Em `config.py` dá pra mexer em:
- `step_delay`: quão rápido ela reage (menor = mais preciso, mais pesado).
- `frame_stack`: quantos frames ela vê de uma vez (mais = melhor noção de
  velocidade, mas mais lento pra treinar).

Em `agent.py`, a classe `DQNAgent` tem `epsilon_decay_steps` (quanto tempo
ela leva pra parar de "chutar" e passar a confiar no que aprendeu) e `gamma`
(quanto ela valoriza recompensas futuras vs imediatas).
