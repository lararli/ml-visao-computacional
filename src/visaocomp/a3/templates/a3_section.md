# A3: classificador com CNN pré-treinada

Seção gerada por `scripts/run_a3.py` a partir de `outputs/$run_id/a3/metrics.json`. Todos os números vêm desse arquivo. Não edite à mão: rode `make run-a3`.

Classes do dataset: $classes_csv.

## Resultado do treino (R01 e R02)

Carregamos a EfficientNet-B0 do torchvision com os pesos `IMAGENET1K_V1`, congelamos o corpo (nenhum parâmetro dele tem gradiente) e trocamos o classification head por uma camada linear nova de $head_in entradas e $head_out saídas, uma por classe. Dos $params_total de parâmetros do modelo, só $params_trainable são treinados. É feature extraction: um treino só, $epochs épocas, Adam com lr $lr, lote de $batch_size, em `$device`, $train_seconds segundos de treino.

Os dados: $files_found imagens lidas, $files_duplicate idênticas removidas (mesmo sha256) e $files_unreadable ilegíveis puladas, sobrando $files_kept. O split é estratificado, seed $seed: $n_train para treino, $n_val para validação e $n_test para teste. A melhor época pela validação foi a $best_epoch (accuracy de validação $best_val_acc). O teste foi usado uma vez, com esse checkpoint.

**Accuracy global no teste: $acc_global.** Accuracy balanceada (média das classes): $balanced_acc. F1 macro: $macro_f1. O baseline que prevê sempre a classe mais comum do treino ($baseline_class) acerta $baseline_acc no teste.

$per_class_table

As classes mais difíceis foram $hard_classes. Trocas mais frequentes: $confusions.

Curvas de loss e accuracy por época: `outputs/$run_id/a3/curves.png`. Matriz de confusão: `outputs/$run_id/a3/confusion.png`.

## Estratégias de augmentation propostas (R03)

O treino acima não usa augmentation: só resize para o lado menor, recorte central e a normalização do ImageNet (média e desvio ficam dentro do modelo). O PD pede propor, não testar. As propostas abaixo vêm do que as imagens têm: bike e cars são cenas de rua em 640x480 (o objeto ocupa uma parte pequena da foto), cats, dogs e horses são fotos de animais em ambientes variados, flowers são recortes de 128x128 pixels (ampliados para 224), e human são pessoas de corpo inteiro, muitas vestidas para montaria. Começaríamos pelas classes com mais erro: $hard_classes.

### Flip horizontal
- Por que ajuda: nenhum objeto do dataset tem um lado fixo. Um gato, um cavalo ou um carro espelhado continua sendo da mesma classe, e o flip dobra as poses vistas sem nenhuma imagem nova.
- Pode distorcer: cars, porque placas e letreiros ficam escritos ao contrário, o que nunca ocorre numa foto real; bike, porque a corrente e os pedais passam para o lado oposto. A forma geral não muda, então o risco é pequeno.

### Rotação leve (até cerca de 15 graus)
- Por que ajuda: fotos feitas na mão saem inclinadas, principalmente de cats, dogs e bike. A rotação leve ensina o modelo a ignorar isso.
- Pode distorcer: horses e human, porque aparecem em pé e com o horizonte reto; rotações grandes criam poses que quase não existem nos dados. flowers, porque a imagem tem só 128x128 pixels e os cantos vazios criados pela rotação ocupam uma parte grande dela.

### Recorte aleatório com resize (RandomResizedCrop)
- Por que ajuda: obriga o modelo a reconhecer a classe por partes do objeto (a cabeça do cavalo, a roda da bicicleta), e não por uma única posição.
- Pode distorcer: bike e cars, porque o objeto é pequeno numa cena de rua e um recorte pode deixar só a rua, com o rótulo da classe errado. human, porque muitas dessas pessoas aparecem ao lado de cavalos, e um recorte que tira o corpo inteiro tira também a pista que separa human de horses. flowers, porque recortar uma imagem de 128x128 e ampliar perde ainda mais detalhe.

### Color jitter (brilho, contraste e saturação)
- Por que ajuda: a luz muda muito entre as fotos, e várias fotos de bike são escuras. Variar brilho e contraste torna o modelo menos dependente da iluminação.
- Pode distorcer: flowers, porque a cor (rosa, laranja, roxo) é a pista principal da classe e uma mudança de matiz cria flores de cor irreal; usar só brilho e contraste nessa classe. horses, porque a cor da pelagem varia dentro da classe e uma mudança forte de matiz cria cavalos de cor impossível.

### Conversão aleatória para grayscale (probabilidade baixa)
- Por que ajuda: força o modelo a usar forma e textura, e não só cor. Serve bem a cars e bike, onde a cor do objeto não define a classe.
- Pode distorcer: flowers, porque tira a cor, que é a pista principal; e cats, dogs e horses, porque parte da diferença entre fotos está na pelagem. Por isso a probabilidade deveria ser baixa (por exemplo 0,1).

### Variação de escala (resize aleatório com crop e treino multi-scale)
- Por que ajuda: os tamanhos originais vão de 128x128 (flowers) a 640x480 (bike e cars). O modelo vê o objeto em escalas diferentes, como acontece em uso real.
- Pode distorcer: flowers, porque já são ampliadas de 128 para 224 pixels e ampliar mais deixa a imagem borrada; e bike e cars, porque reduzir demais deixa o objeto, que já é pequeno, com poucos pixels.

## Feature extraction ou fine-tuning (R04)

**Feature extraction** congela o corpo e treina só a cabeça. **Fine-tuning** também atualiza parte ou todo o corpo, com passo de atualização menor nas camadas já treinadas. A escolha depende de dois fatores: o tamanho do dataset e o quanto o domínio se parece com aquele em que o modelo foi pré-treinado (o ImageNet).

- Dataset pequeno e domínio parecido: feature extraction. É o nosso caso. Temos $n_train imagens de treino e $n_classes classes, de objetos comuns (gatos, cachorros, carros e bicicletas têm muitas classes no ImageNet). Os parâmetros do corpo já extraem bem esses objetos, e treinar só $params_trainable parâmetros com $n_train imagens quase não tem risco de overfitting. O resultado confirma: $acc_global no teste, contra $baseline_acc do baseline.
- Dataset pequeno e domínio diferente: feature extraction das camadas intermediárias, ou fine-tuning só das últimas camadas com passo bem pequeno. Atualizar o corpo inteiro com poucas imagens decora o treino.
- Dataset grande e domínio parecido: fine-tuning traz ganho pequeno e custa mais; vale se a diferença de accuracy compensar o tempo.
- Dataset grande e domínio diferente: fine-tuning completo. É o caso de radiografias de tórax (A4) e de imagens aéreas (A1), onde o ImageNet não tem nada parecido e as características do corpo congelado servem pouco.

Na aula 4, o professor mostrou o mesmo trade-off com um ViT: com o corpo congelado, cerca de 91% de accuracy contra 96% do fine-tuning completo, com cerca de 20 segundos a menos de treino. Para o nosso dataset, o fine-tuning só seria o próximo passo se o erro que sobrou ($confusions) se concentrasse em classes que o corpo congelado não separa.

## Escolha do modelo pré-treinado (R05)

Escolhemos a EfficientNet-B0 (`IMAGENET1K_V1`) por três razões.

1. **Capacidade do Colab T4.** A T4 tem 16 GB de memória de vídeo. A EfficientNet-B0 tem $params_total de parâmetros com a nossa cabeça de $n_classes classes ($eff_params com a cabeça original de 1.000 classes do ImageNet) e $eff_gflops GFLOPs por imagem, contra $res_params e $res_gflops da ResNet-50 (valores do torchvision). Com o corpo congelado, o treino não guarda as ativações para o backward: só a cabeça é treinada, então o consumo se parece com o da inferência. Aqui o treino inteiro levou $train_seconds segundos com lote de $batch_size, num computador local.
2. **Número de classes.** São $n_classes classes. A cabeça é uma camada linear de $head_in por $head_out, com $params_trainable parâmetros. Para tão poucas classes não faz falta um corpo grande: o que importa é a qualidade das características, e a EfficientNet-B0 chega a $eff_top1% de accuracy top-1 no ImageNet, contra $res_top1% da ResNet-50 nos pesos `IMAGENET1K_V1`.
3. **Domínio.** Os objetos do dataset são comuns no ImageNet, então os pesos pré-treinados servem sem adaptação do corpo. Para um domínio distante (A4, radiografias) a escolha seria outra: um modelo maior e fine-tuning.

A cabeça é pequena e o corpo é leve, então o mesmo código roda na T4 do Colab e no Apple M5 (MPS).
