{{start}}
## ViT pré-treinado, CNN e a decisão com motivo

Gerado por `make run-a1-pretrained` a partir de `outputs/{{run_id}}/a1_pretrained/`. Não edite à
mão: o texto de análise fica em `reports/a1_pretrained_narrativa.yaml` e os números vêm do run.
Mesmos recortes, mesmo split (células de 1 km), mesma perda, mesma augmentation e mesmas métricas do
ViT do zero. Muda só o modelo e o treino curto.

### Fine-tuning do ViT pré-treinado (R13)

| Item | ViT-S/16 pré-treinado | ResNet-18 pré-treinada |
|---|---|---|
| Pesos | timm `vit_small_patch16_224` (ImageNet-21k, ajustado no 1k) | torchvision `IMAGENET1K_V1` |
| Cabeça | trocada: `head` = Linear({{vit_pretrained_head_in}}, 2) | trocada: `fc` = Linear({{resnet18_head_in}}, 2) |
| Entrada | recorte 96 x 96 ampliado para 224 x 224 e normalizado com a média do ImageNet, dentro do modelo | igual |
| Treino | {{vit_pretrained_epochs}} épocas, lr {{vit_pretrained_lr}}, corpo inteiro ajustado | {{resnet18_epochs}} épocas, lr {{resnet18_lr}} |
| Versão gravada | `{{vit_pretrained_version}}` | `{{resnet18_version}}` |

{{r13_texto}}

### Tabela comparativa (R13, R16)

{{comparison_table}}

{{r16_texto}}

### Sim, não ou "não dá para dizer"

A regra roda no modelo escolhido ({{chosen_label}}). A ordem dos motivos é fixa: sem prédio, árvore
(NDVI do telhado para solar e do quintal para piscina, os mesmos limites da montagem dos dados), não visível de
cima (cerca, revestimento, janela e porta, que a POC cortou), e incerto. A faixa de incerteza é
escolhida na validação: abaixo dela o "não" acerta pelo menos 95%, acima dela o "sim" acerta pelo
menos 60%, sempre contra o rótulo do catálogo.

{{abstain_table}}

{{abstencao_texto}}

### A camada gold

O resultado volta para o trabalho em duas tabelas do `catalog.duckdb`, gravadas por MERGE (rodar de
novo não duplica; um modelo novo grava linhas novas e não apaga as antigas):

- `gold.austin_house_attribute`: lote x atributo x ano da foto x versão do modelo, com endereço,
  coordenada, decisão, probabilidade, motivo, recorte e run. Nesta POC: {{gold_rows}} linhas (as
  casas do teste).
- `gold.austin_house_attribute_evidence`: lote x atributo, cruza a decisão da imagem com a licença e o
  OpenStreetMap e grava `agreement` (concorda, imagem sem licença, licença sem imagem, não dá).

```sql
-- casas com piscina vista na foto e sem licença nem OSM: lista para o time comercial
SELECT a.parcel_id, a.address, a.probability
FROM gold.austin_house_attribute a
JOIN gold.austin_house_attribute_evidence e USING (parcel_id, attribute)
WHERE a.attribute = 'pool' AND a.model_version = e.model_version
  AND e.agreement = 'imagem sem licença'
ORDER BY a.probability DESC;
```

### BERT e ViT: o que cada pré-treino maximiza (R10)

{{r10_texto}}

### DeiT e Swin: o que resolvem que o ViT original não resolve (R14)

{{r14_texto}}

### Quando o ViT supera a CNN, e quando a CNN é melhor (R15)

{{r15_texto}}
{{end}}
