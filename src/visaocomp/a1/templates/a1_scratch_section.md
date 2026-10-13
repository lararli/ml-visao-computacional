{{start}}
## ViT do zero

Gerado por `make run-a1-scratch` a partir de `outputs/{{run_id}}/a1_scratch/metrics.json`. Não
edite à mão: o texto de análise fica em `reports/a1_scratch_narrativa.yaml` e os números vêm do run.
A tarefa é multirrótulo: cada recorte de 96 x 96 px (cerca de 58 m de lado) tem duas saídas, piscina
e solar, cada uma um "sim ou não" independente.

### Arquitetura feita à mão (R06, R08, R11)

| Módulo | Arquivo | O que faz |
|---|---|---|
| `ScaledDotProductAttention` | `src/visaocomp/vit/attention.py` | softmax(Q K^T / raiz(d_k)) V; devolve a saída e os pesos |
| `AttentionHead` | `src/visaocomp/vit/attention.py` | uma cabeça com as suas próprias projeções Q, K e V |
| `MultiHeadAttention` | `src/visaocomp/vit/attention.py` | {{heads}} cabeças independentes, concatenadas, e a projeção de saída W_O |
| `TransformerEncoderBlock` | `src/visaocomp/vit/encoder.py` | x + MHA(LayerNorm(x)); x + FFN(LayerNorm(x)); FFN = Linear, GELU, Linear |
| `PatchEmbedding` | `src/visaocomp/vit/model.py` | convolução {{patch}} x {{patch}} com stride {{patch}}: cada patch vira um vetor de {{dim}} |
| `VisionTransformer` | `src/visaocomp/vit/model.py` | CLS aprendível + posição aprendível + {{depth}} blocos + LayerNorm + cabeça de {{num_outputs}} logits |

Imagem {{image_size}} x {{image_size}} com patch {{patch}}: {{grid}} x {{grid}} patches, mais o CLS, dá {{tokens}} tokens de dimensão {{dim}}.
São {{params}} parâmetros, todos treinados do zero (nenhum peso pré-treinado).

### Positional encoding (R09)

{{r09_texto}}

### Treino do zero (R12)

| Item | Valor |
|---|---|
| Split | o da montagem dos dados (células de 1 km): treino {{n_train}}, validação {{n_val}}, teste {{n_test_split}} |
| Perda | BCE por atributo, rótulo vazio fora, peso nos positivos {{pos_weight}} |
| Otimizador | AdamW, lr {{lr}}, weight decay {{weight_decay}}, {{warmup_epochs}} épocas de aquecimento e cosseno |
| Lote e épocas | {{batch_size}} e {{epochs}} |
| Augmentation | giros de 90 graus, espelho e brilho leve (foto aérea não tem "cima") |
| Melhor época (PR-AUC médio na validação) | {{best_epoch}} ({{best_val_pr_auc_mean}}) |
| Limiar por atributo (máximo F1 na validação) | {{thresholds}} |
| Dispositivo e tempo de treino | {{device}}, {{train_min}} min |

![curvas](../figures/a1/curves.png)

### Resultados no teste

O teste é usado uma vez. A "amostra conferida" são os {{n_reviewed}} recortes do teste que foram
marcados a olho (`conferencia_inicial.csv`, copiado para `silver.label.first_check`): "sim" = 1, "não" = 0, e
"não dá" fica fora. A última coluna é o PR-AUC de um palpite ao acaso, que é igual à fração de
positivos: é o baseline.

{{metrics_table}}

{{resultados_texto}}

### Mapas de atenção (R07, R12)

Cada mapa é a linha do token CLS na última camada: quanto o CLS, que decide a classe, olha para cada
um dos {{grid}} x {{grid}} patches. O mapa é ampliado para 96 x 96 e posto sobre o recorte. A figura mostra a média das {{heads}} cabeças e
cada cabeça; a marcada "(usada)" é a que mais olha para a região do atributo (piscina: cabeça
{{head_pool}}; solar: cabeça {{head_solar}}).

{{figures}}

Para não depender só de três exemplos, a tabela mede, em todo o teste, quanto da atenção do CLS cai
no telhado (o contorno da casa na montagem dos dados) e na piscina (os polígonos do OSM dentro do recorte),
dividido pela área que a região ocupa. Razão 1 é o acaso; 2 quer dizer que a região recebe o dobro
da atenção que a área dela explicaria.

{{regions_table}}

{{regioes_texto}}

### Limites

{{limites_texto}}
{{end}}
