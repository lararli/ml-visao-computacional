# visaocomp

Visão computacional com CNNs e Transformers (Infnet). Quatro atividades em um repositório:

| Atividade | Tema | Fonte de dados |
|---|---|---|
| A1 | Vision Transformer: piscinas e painéis solares em casas de Austin | NAIP, lotes, prédios, OSM e licenças |
| A2 | CLIP: ranking e busca semântica | Kaggle `groffo/ads16-dataset` |
| A3 | CNN pré-treinada congelada | Kaggle `pavansanagapati/images-dataset` |
| A4 | Raio-X, GAN e plano de melhoria | Kaggle `sachinkumar413/covid-pneumonia-normal-chest-xray-images` |

## O pipeline

Uma atividade = um dataset = um pipeline: bronze (o dado como veio, com sha256) -> silver (pronto
para treinar) -> modelos (treino e avaliação) -> gold (o resultado pronto para consumo). Tudo fica no
`data.nosync/catalog.duckdb`, e todo modelo é registrado em `gold.model_run` (atividade, dataset,
modelo, versão com o hash do checkpoint, métrica e run).

| Atividade | Dataset (`make pipeline ONLY=`) | Modelos | Gold | Notebook |
|---|---|---|---|---|
| A1 | `austin_houses` | ViT do zero, ViT-S pré-treinado, ResNet-18 | `gold.austin_house_attribute`, `gold.austin_house_attribute_evidence` | `A1_vision_transformers.ipynb` |
| A2 | `ads16` | CLIP ViT-B/32 (sem treino) | `gold.ads16_ad_object` | `A2_clip_ads16.ipynb` |
| A3 | `images` | EfficientNet-B0 congelada | `gold.images_prediction` | `A3_cnn_kaggle.ipynb` |
| A4.1 | `chest_xray` | GAN condicional (sem e com mitigação), ResNet-18 (sem e com sintético) | `gold.xray_prediction` | `A4_estudo_caso_raio_x.ipynb` |
| A4.2 | só escrita | — | — | — (só no relatório) |

O código de cada atividade está em `src/visaocomp/a1` a `a4` (e `vit/`, o ViT feito à mão); o
pipeline que liga tudo é `src/visaocomp/pipeline.py`. O relatório PDF abre cada atividade com a
estrutura do PD: problema, decisões técnicas, resultados e análise crítica.

## Como reproduzir

```
git clone https://github.com/lararli/ml-visao-computacional.git
cd ml-visao-computacional
make all
```

`make all` faz `setup`, `data` e `pipeline`: do download às tabelas gold, com todos os treinos e avaliações. O relatório PDF e as apresentações de negócio vão prontos no ZIP da entrega.
Não precisa de conta nem de token do Kaggle: o download usa o endpoint público.

Requisitos: `git`, `make` e [`uv`](https://docs.astral.sh/uv/). O `uv` instala o Python 3.12 sozinho.

## Resultados no repositório

`outputs/` está no git com todos os resultados de cada atividade (métricas, previsões, figuras e
manifestos), para conferir sem rodar nada. Só os pesos dos modelos (`*.pt`, cerca de 200 MB) ficam
fora; o sha256 de cada um está ao lado, em `<peso>.pt.sha256`. Num clone novo, `make all` baixa os
dados, reaproveita esses resultados e publica a gold; `make pipeline FORCE=1` treina tudo de novo.

## Alvos do Makefile

| Alvo | Faz |
|---|---|
| `make setup` | cria `.venv.nosync` e instala o `uv.lock` |
| `make lint` | `ruff check` e `ruff format --check` |
| `make test` | testes unitários, sem rede e sem os dados |
| `make smoke` | a ingestão inteira com `.zip` minúsculos, em CPU |
| `make ci` | `lint` + `test` + `smoke` |
| `make run-a3` | A3: treina a cabeça da EfficientNet-B0 congelada, avalia no teste e grava `outputs/<run_id>/a3/` |
| `make run-a1-scratch` | A1: ViT do zero (piscina e solar), métricas no teste e na amostra conferida, mapas de atenção e `reports/sections/a1.md` |
| `make run-a1-pretrained` | A1: ViT-S/16 pré-treinado e ResNet-18 nos mesmos recortes, tabela comparativa, "não dá" com motivo, `gold.austin_house_attribute` e a parte do card 06 em `a1.md` |
| `make run-a2` | A2: embeddings do CLIP no DuckDB, ranking de objetos, busca por texto e `reports/sections/a2.md` |
| `make run-a4` | A4.1: dedup, caso 840/240/120, GAN sem e com mitigação, recall com e sem sintético e `reports/sections/a4.md` |
| `make data` | baixa, verifica o sha256, extrai e grava `outputs/<run_id>/manifest.json`; depois monta a A1 |
| `make pipeline` | por dataset: treino e avaliação (só o que falta em `outputs/`) e a gold no `catalog.duckdb`; `ONLY=<dataset>` roda um só |
| `make gold` | só publica a gold dos 4 datasets a partir de `outputs/` |
| `make query` | roda as consultas de exemplo nas tabelas gold (uma pergunta de negócio por dataset) |
| `make sql Q="SELECT ..."` | roda qualquer consulta na gold, só para leitura |
| `make clean` | apaga `outputs/`, `interim/`, `processed/` e caches; nunca `data.nosync/raw/` |

Prova (depois de `make data`): `make ci && uv run pytest -q tests/test_dados_kaggle.py`.

## A3: CNN pré-treinada congelada

```
make data       # uma vez
make run-a3     # cerca de 2 minutos no Apple M5 (MPS)
make ci && uv run pytest -q tests/test_rubrica_a3.py -k 'R01 or R02 or R03 or R04 or R05'
```

- O que faz: EfficientNet-B0 (`IMAGENET1K_V1`) com o corpo congelado e uma cabeça nova com o número de
  classes achado em `data/<classe>/`. Um treino, 10 épocas, Adam, lr 1e-3. Todos os parâmetros estão em
  `configs/a3.yaml`.
- Dados: a varredura (`ImageScanner`) abre cada arquivo, pula o ilegível, tira os idênticos pelo sha256
  e ignora a cópia aninhada `data/data/` que o dataset traz. As contagens vão para o manifesto.
- Split: estratificado 70/15/15, seed 42. O teste é usado uma vez, com o melhor checkpoint pela validação.
- Resultados em `outputs/<run_id>/a3/`: `metrics.json` (accuracy global e por classe, matriz de confusão,
  baseline, erros), `curves.png`, `confusion.png` e `best_head.pt`. O `manifest.json` fica em
  `outputs/<run_id>/`.
- Texto: `make run-a3` também gera `reports/sections/a3.md` (R03, R04 e R05) a partir do `metrics.json`.
  Não edite à mão: o texto está em `src/visaocomp/a3/templates/a3_section.md`.
- Memória: o corpo roda sem gradiente, então o treino com lote de 32 cabe com folga. Se faltar memória
  na GPU, reduza `batch_size` no `configs/a3.yaml`.
- Notebook: `notebooks/A3_cnn_kaggle.ipynb` chama as mesmas classes (requisitos no topo).
- Limite: a MPS não é determinística bit a bit. Com a mesma seed, dois treinos deram accuracy de teste
  de 98,1% e 98,5% (diferença de uma imagem em 264).

## Onde ficam as coisas

- `data.nosync/raw/<fonte>/`: o dado como veio (nunca editado). Os `.zip` ficam em `raw/_archives/`.
- `data.nosync/catalog.duckdb`: schemas `bronze`, `silver` e `gold`, criados vazios pelo `make data`.
- `outputs/<run_id>/manifest.json`: fonte, versão, sha256, contagens e commit de cada execução.
- `configs/data.yaml`: as 3 fontes, a versão do Kaggle e o sha256 do `.zip`.
- `src/visaocomp/`: `data/sources.py` (`DataSource`, `KaggleSource`), `data/ingest.py`, `catalog.py`, `run.py` (`RunDir`).

A pasta de dados muda com a variável `VISAOCOMP_DATA_DIR` (útil com `git worktree`, para vários
checkouts lerem os mesmos dados).

## A1: dados de Austin

`scripts/build_a1.py` (chamado pelo `make data`) monta, só com fontes públicas e sem conta, um
recorte de foto aérea por casa, com endereço, id do lote e rótulos de piscina e solar. O config é
`configs/a1.yaml` (área, URLs, regras de palavra, limiares e seeds).

1. **Consultas** (`src/visaocomp/a1/sources.py`): NAIP 2022 (STAC do Planetary Computer, uma
   leitura por janela de cada quadrícula), lotes e endereços (ArcGIS da prefeitura, paginado),
   prédios (Overture no S3, baixado uma vez), piscinas do OSM (Overpass com data fixa) e licenças
   POOL e SOLAR residenciais até 2021 (Socrata). Cada consulta grava a resposta crua em
   `data.nosync/raw/a1/<fonte>/<chave>/` e um `_done.json` com a consulta, a data, a contagem e o
   sha256 de cada arquivo. Rodar de novo não consulta a rede ("consulta já gravada"). Resposta de
   erro ou de limite (HTTP 504, `{"error": ...}`, Overpass com erro) nunca é gravada: o cliente
   espera e repete; no Overpass, tenta o espelho do config.
2. **Silver** (`src/visaocomp/a1/build.py`): lote (`PID_10`); endereço (só `ADDRESS_TYPE = 1`) no
   lote que o contém; casa = o maior prédio do lote, com o endereço mais perto. Lote sem prédio, sem
   endereço, com mais de 4 endereços (multifamiliar) ou com prédio fora de 40 a 1.500 m² não vira
   casa. Licença sem lote na camada fica fora e é contada.
3. **Recorte**: 96 x 96 px do NAIP, centrado na casa, em `data.nosync/processed/chips/<chip_id>.png`
   (RGB) e `processed/chips_nir/<chip_id>.png` (infravermelho, para o NDVI). Recorte que já existe é
   reaproveitado.
4. **Rótulo** por recorte e atributo: 1 com evidência (OSM ou licença), 0 sem nenhuma, vazio com
   motivo: `arvore` (NDVI: vegetação em mais de 50% do telhado, para solar, ou de 75% do quintal,
   para piscina) e `fora_do_recorte` (a piscina do OSM está no lote, mas fora da foto).
5. **Split** por células de 1 km (70/15/15, seed 42): uma célula inteira vai para um só lado.
6. **Catálogo**: `bronze.*_raw` e `silver.parcel`, `address`, `house`, `permit_event`,
   `osm_pool`, `chip`, `label` no `catalog.duckdb`, por MERGE pela chave (rodar de novo não duplica).
7. **Manifesto**: `outputs/<run_id>-a1_data/manifest.json` com cada consulta e as contagens por
   etapa, por atributo e rótulo, por motivo e por split.
8. **Conferência**: a amostra estratificada de 200 recortes do teste fica em
   `processed/a1_review_sample.csv`. A conferência inicial marca cada um em `notas/conferencia/conferencia_inicial.csv` e
   `scripts/review_a1.py` gera `notas/conferencia/revisao.html` para a Lara preencher o `lara.csv`.
   Quando o `conferencia_inicial.csv` existe, o `make data` copia a marcação para `silver.label.first_check`.

Prova (depois de `make data`): `make ci && uv run pytest -q tests/test_dados_austin.py`.

## A1: ViT do zero

```
make run-a1-scratch                             # treina, avalia, mapas de atenção e a1.md
python scripts/run_a1_scratch.py --report-only  # só regenera a parte do ViT do zero no a1.md
python scripts/run_a1_scratch.py --epochs 5     # um treino curto (troca as épocas do config)
```

Precisa do `make data` (o `catalog.duckdb` e os recortes da montagem dos dados). O config é
`configs/a1_scratch.yaml`.

1. **Módulos feitos à mão** (`src/visaocomp/vit/`): `ScaledDotProductAttention`, `AttentionHead`
   (Q, K e V próprios), `MultiHeadAttention` (cabeças concatenadas e W_O), `TransformerEncoderBlock`
   (pré-LayerNorm, FFN de duas camadas com GELU, residual), `PatchEmbedding` (convolução com stride
   do patch) e `VisionTransformer` (CLS e posição aprendíveis, 2 logits: piscina e solar).
2. **Dados** (`src/visaocomp/a1/chips.py`): lê `silver.chip`, `silver.label`, `silver.house` e
   `silver.osm_pool` pela chave `chip_id`, com o catálogo aberto só para leitura. Confere o contrato
   (colunas, splits, rótulo em 0, 1 ou vazio com motivo, recorte legível de 96 x 96) e para cedo se
   algo não bate. O split é o da montagem dos dados.
3. **Treino** (`src/visaocomp/a1/train.py`, `MultiLabelTrainer`): BCE por atributo, rótulo vazio
   fora da perda e da métrica, peso nos positivos; AdamW com aquecimento e cosseno; giros de 90
   graus, espelho e brilho leve; a melhor época pelo PR-AUC médio da validação.
4. **Avaliação**: limiar por atributo com o melhor F1 da validação; precision, recall, F1 e PR-AUC
   no teste inteiro (rótulo do catálogo) e nos 200 recortes conferidos a olho
   (`silver.label.first_check`, cópia do `conferencia_inicial.csv`).
5. **Mapas de atenção** (`AttentionMapper`): a linha do CLS na última camada, por cabeça, sobre o
   recorte (piscina, solar e o erro mais confiante), e a razão entre a atenção que cai no telhado ou
   na piscina do OSM e a área deles, em todo o teste.
6. **Saída**: `outputs/<run_id>-a1_scratch/` com `manifest.json`, `a1_scratch/metrics.json`,
   `best.pt`, `curves.png` e `attention_*.png`; as figuras são copiadas para `reports/figures/a1/`
   e o texto vai para `reports/sections/a1.md` (entre os marcadores `a1_scratch`, para a parte
   escrever a parte dele no mesmo arquivo). O texto de análise fica em
   `reports/a1_scratch_narrativa.yaml`, com os números por `{{marcador}}`.

Prova: `make ci && uv run pytest -q tests/test_rubrica_a1.py -k 'R06 or R07 or R08 or R09 or R11 or R12'`.

Tempo: 28 min na GPU do Mac (Apple M5, MPS), 75 épocas; cerca de 4 GB de RAM.

## A1: ViT pré-treinado, comparação, "não dá" e gold

```
make run-a1-pretrained                            # treina os dois modelos e faz o resto
python scripts/run_a1_pretrained.py --post-only   # só comparação, "não dá", gold e a1.md
```

1. `configs/a1_pretrained.yaml`: ViT-S/16 do timm e ResNet-18 do torchvision, com a cabeça trocada
   por 2 saídas, 6 épocas e lr baixo. O resto (recortes, split, perda, augmentation, métricas) é o do
   ViT do zero.
2. `AbstainPolicy` (`src/visaocomp/a1/abstain.py`): sim, não ou "não dá", com o motivo na ordem sem
   prédio, árvore, não visível de cima, incerto. A faixa de incerteza sai da validação.
3. `GoldWriter` (`src/visaocomp/a1/gold.py`): `gold.austin_house_attribute` (lote x atributo x ano x versão
   do modelo) e `gold.house_attribute_evidence` (lote x atributo, com `agreement`), por MERGE.
4. Resultados em `outputs/<run_id>-a1_pretrained/a1_pretrained/`: `metrics.json`, `predictions.csv`,
   `comparison.csv`, `decisions.csv`, `abstain.json` e os checkpoints.

Prova: `make ci && uv run pytest -q tests/test_abstain.py tests/test_gold.py tests/test_rubrica_a1.py`.

Tempo: cerca de 20 min na GPU do Mac (Apple M5, MPS): 14 min o ViT-S e o resto a ResNet-18.

## Relatório e apresentações

O relatório técnico único (`lara_lopez_deep-learning-and-vision_computer-vision.pdf`) e um PDF de
negócio por entrega (`apresentacoes/`) vão prontos na raiz do ZIP. Eles são montados a partir de
`reports/sections/` e de `outputs/`; o código de montagem fica fora desta entrega.

## Como o `make data` funciona

1. Para cada fonte, confere se o `.zip` e a pasta extraída existem e se o sha256 bate com o config.
   Se bate, escreve "já existe e passa no checksum" e não baixa nada.
2. Senão, baixa em streaming para `*.zip.part`. Se a conexão cai ou o tamanho não bate, o parcial é
   descartado e há até 3 tentativas. O próximo `make data` refaz.
3. Se o sha256 baixado difere do config, para com "a fonte mudou" e não sobrescreve nada.
4. O endpoint aceita `?datasetVersionNumber=`, então a versão (1, a atual das 3 fontes) fica fixada.
   O sha256 do `.zip` é gravado no `configs/data.yaml` no primeiro download e conferido nos seguintes.

## Tempo e memória

Medidos no Apple M5 (arm64), com conexão doméstica. Os números do `make data` estão na seção abaixo
e valem para a versão desta entrega.

| Etapa | Tempo | Memória |
|---|---|---|
| `make setup` | cerca de 10 s com cache do `uv` | baixa |
| `make data` (primeira vez, 3 GB) | ver "Medidas" | baixa (streaming em blocos de 1 MB) |
| `make data` (de novo) | segundos, só confere os checksums | baixa |
| `make ci` | poucos segundos | baixa |

Espaço em disco: cerca de 3 GB de `.zip` mais cerca de 3 GB extraídos em `data.nosync/`.

## Medidas

Apple M5, 2026-10-12:

| Etapa | Medido |
|---|---|
| `uv sync` (cache frio, 163 pacotes, com torch) | 9 s |
| `make data`, primeira vez (2,8 GB de `.zip`, 12.263 arquivos) | 1 min 55 s |
| `make data`, de novo (só confere os checksums) | 5 s |
| `make ci` | cerca de 3 s |
| `uv run pytest -q tests/test_dados_kaggle.py` | cerca de 5 s |
| `make run-a3` (10 épocas, MPS, lote 32) | cerca de 2 min |
| A1, primeira vez (6 consultas, 555 MB em `raw/a1/`, 7.884 recortes) | NAIP 504 s, Overture 165 s, Overpass 67 s (em paralelo: cerca de 8,5 min); recortes 46 s |
| A1, de novo (nada consultado, recortes reaproveitados) | cerca de 1 min |

O sha256 de uma fonte foi baixado duas vezes e deu o mesmo valor, então o checksum é estável.

## A2: CLIP no ADS-16

```
make run-a2                              # embeddings no DuckDB, ranking, busca, relatório
python scripts/run_a2.py --report-only   # só regenera reports/sections/a2.md
python scripts/run_a2.py --calibrate     # sorteia pares novos para conferir o limiar a olho
```

- **Amostra:** 600 imagens, seed 42. O ADS-16 tem só 301 anúncios (20 pastas `Ads/<n>/`, não 16 como
  diz o PD); o PD pede 500 ou mais, então a amostra tem todos os anúncios e 299 fotos pessoais do
  corpus, estratificadas por polaridade. Mapa pasta para categoria: cabeçalho de
  `Corpus/U0001/U0001-RT.csv` (`outputs/<run_id>/a2/category_map.json`).
- **DuckDB:** `silver.ads16_embedding` em `catalog.duckdb` (chave `image_path` + `model`, vetor
  `FLOAT[512]` normalizado). Rodar de novo escreve `embeddings: 0 novos` e não chama o modelo.
  O arquivo é compartilhado: cada operação abre, escreve e fecha, e espera se estiver travado.
- **Limiar:** `configs/a2_threshold_check.json` guarda 50 pares conferidos a olho (sem o score
  visível). O limiar sai desses rótulos e da distribuição das similaridades, em `threshold.json`.
- **Texto da análise:** `reports/a2_narrativa.yaml` (números entram por `{{marcador}}`).
  `reports/sections/a2.md` é gerado, não editado à mão.
- **Prova:** `make ci && uv run pytest -q tests/test_rubrica_a2.py -k 'R17 or R18 or R19 or R20'`.
- **Licença do ADS-16:** só uso científico; as imagens não podem ser publicadas (as figuras ficam
  em `outputs/`; os pesos `.pt` ficam fora do git).

Tempo no Apple M5 (MPS), lote de 32: 600 embeddings em cerca de 10 s; `make run-a2` completo cerca de
30 s com os modelos em cache (CLIP e `bert-base-cased`, cerca de 1 GB juntos para baixar na primeira vez).
## A4: raio-X, GAN e recall

`make run-a4` (depois de `make data`) roda `scripts/run_a4.py`, com o config em `configs/a4.yaml`:

1. Remove cópias por hash perceptual (pHash) antes de qualquer split; o cache dos hashes fica em
   `data.nosync/interim/a4/`.
2. Separa o teste real (150 por classe) e sorteia o caso do PD (840/240/120, 20% para validação).
3. Treina a GAN condicional duas vezes (sem e com mitigação), mede o FID e gera 600 COVID com a melhor.
4. Treina a ResNet-18 sem e com as sintéticas, em 3 seeds, e mede o recall da COVID no teste real.
5. Grava tudo em `outputs/<run_id>/` (com `manifest.json`) e gera `reports/sections/a4.md` e as
   figuras em `reports/figures/a4/`.

Se já existe uma execução completa com o mesmo config, o alvo só regera o `a4.md` (segundos).
`--force` treina de novo numa pasta nova; `--resume <run_id>` continua uma execução interrompida.

Medido no Apple M5 com a GPU dividida com outros treinos (2026-10-12): 31 min no total (GAN sem
mitigação 10,5 min, GAN com mitigação 11 min, 6 treinos do classificador 9 min), 1,5 GB de RAM.
O notebook `notebooks/A4_estudo_caso_raio_x.ipynb` só lê `outputs/` e roda em segundos.

Prova: `make ci && uv run pytest -q tests/test_rubrica_a4.py -k 'R21 or R22 or R23 or R24 or R26'`
e `uv run pytest -q tests/test_a4.py` (o teste de vazamento por hash precisa de `make run-a4`).
