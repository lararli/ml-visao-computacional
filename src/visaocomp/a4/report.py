"""Gera reports/sections/a4.md a partir de outputs/<run_id>/ (nenhum número digitado)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats as sstats

from visaocomp.a4.config import A4Config

FIGURES = {
    "gan_comparison.png": "gan_comparison.png",
    "synthetic_grid.png": "synthetic_grid.png",
    "recall/recall.png": "recall.png",
    "recall/falsos_negativos.png": "falsos_negativos.png",
}


def find_complete_run(outputs_root: Path, config_sha256: str | None = None) -> Path | None:
    """Pasta da execução completa mais recente da A4 (com o mesmo config, se pedido)."""
    for manifest in sorted(Path(outputs_root).glob("*/manifest.json"), reverse=True):
        doc = json.loads(manifest.read_text(encoding="utf-8"))
        if doc.get("stage") != "a4" or doc.get("status") != "completo":
            continue
        if config_sha256 and doc.get("config_sha256") != config_sha256:
            continue
        return manifest.parent
    return None


def num(x: float, nd: int = 2) -> str:
    """Número com vírgula decimal."""
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x: float, nd: int = 1) -> str:
    """Fração como porcentagem com vírgula."""
    return f"{100 * x:.{nd}f}%".replace(".", ",")


def yes_no(flag: bool) -> str:
    """Sim ou não."""
    return "sim" if flag else "não"


class A4Report:
    """Monta a seção da A4 do relatório lendo os arquivos de resultado de uma execução."""

    def __init__(self, run_path: Path, cfg: A4Config) -> None:
        self.run_path = Path(run_path)
        self.cfg = cfg
        self.fig_prefix = "../figures/a4"

    def load(self) -> dict[str, Any]:
        """Lê manifesto, dedup, métricas da GAN e do recall."""

        def read(name: str) -> Any:
            return json.loads((self.run_path / name).read_text(encoding="utf-8"))

        return {
            "manifest": read("manifest.json"),
            "dedup": read("dedup.json"),
            "gan": read("gan_metrics.json"),
            "summary": read("recall/recall_summary.json"),
            "recall": pd.read_csv(self.run_path / "recall" / "recall.csv"),
            "preds": pd.read_csv(self.run_path / "recall" / "test_predictions.csv"),
        }

    # ---------- partes ----------

    def _header(self, d: dict[str, Any]) -> list[str]:
        m = d["manifest"]
        return [
            "# A4.1: detecção de COVID-19 em raio-X de tórax",
            "",
            f"> Gerado por `scripts/run_a4.py` (`make run-a4`) a partir de `outputs/{m['run_id']}/`"
            f" (commit `{m['git']['commit'][:7]}`"
            + (", com alterações ainda não commitadas" if m["git"].get("dirty") else "")
            + f", dispositivo `{m['device']}`). Não edite à mão:"
            " todo número abaixo é lido dos arquivos de resultado.",
            "",
            "## Problema",
            "",
            "Um grupo treinou um classificador de raio-X em três classes (Normal, Pneumonia,"
            " COVID-19) e reportou 93% de accuracy de treino e 61% de validação; o time clínico"
            ' disse que o modelo "ignora casos positivos". Esta atividade faz três coisas:'
            " aponta os problemas técnicos do projeto, gera raio-X de COVID com uma GAN"
            " condicional para atacar a falta de exemplos, e mede, num teste só com imagens"
            " reais, se o recall da COVID melhora com as imagens geradas.",
            "",
        ]

    def _data(self, d: dict[str, Any]) -> list[str]:
        dd = d["dedup"]
        counts = d["manifest"]["counts"]["caso"]
        src = d["manifest"]["sources"][0] if d["manifest"]["sources"] else {}
        lines = [
            "## Dados: o caso remontado",
            "",
            f"Fonte: Kaggle `{src.get('source', '?')}`, versão {src.get('version', '?')},"
            f" sha256 do .zip `{str(src.get('sha256', ''))[:12]}…`, {src.get('n_files', '?')}"
            " arquivos.",
            "",
            f"1. **Cópias removidas antes de qualquer split.** O hash perceptual (pHash de"
            f" {dd['hash_size'] ** 2} bits) achou {dd['grupos_com_copias']} grupos de cópias:"
            f" {dd['copias_removidas']} imagens repetidas saíram, e"
            f" {dd['removidas_por_conflito_de_rotulo']} imagens de"
            f" {dd['grupos_com_rotulos_diferentes']} grupos com rótulos diferentes foram"
            f" descartadas (não dá para saber qual rótulo é o certo). De {dd['n_in']} imagens"
            f" sobraram {dd['n_out']} ({pct(1 - dd['n_out'] / dd['n_in'])} eram cópias ou"
            " conflitos). Duas imagens contam como cópia quando a distância de Hamming entre"
            f" os hashes é no máximo {dd['max_distance']}.",
            "2. **Teste real separado primeiro**: 150 imagens por classe, nunca vistas pela GAN"
            " nem pelo classificador. O teste é balanceado; o recall de cada classe não depende"
            " da proporção do teste, e a precisão da COVID é recalculada na prevalência do caso.",
            f"3. **O caso do PD sorteado do resto** (seed {self.cfg.case.seed}), com"
            f" {pct(self.cfg.case.val_fraction, 0)} estratificados para validação.",
            "",
            "| Parte | " + " | ".join(self.cfg.classes) + " | Total |",
            "|---|" + "---|" * (len(self.cfg.classes) + 1),
        ]
        names = {"train": "treino", "val": "validação", "test": "teste real"}
        for split in ("train", "val", "test"):
            row = [counts[split][c] for c in self.cfg.classes]
            lines.append(f"| {names[split]} | " + " | ".join(map(str, row)) + f" | {sum(row)} |")
        lines.append("")
        return lines

    def _problems(self, d: dict[str, Any]) -> list[str]:
        tc = self.cfg.case.train_counts
        total = sum(tc.values())
        n_covid = tc[self.cfg.positive_class]
        majority = max(tc.values()) / total
        n_val = round(total * 0.2)
        hyper = sstats.hypergeom(total, n_covid, n_val)
        lo, hi = int(hyper.ppf(0.025)), int(hyper.ppf(0.975))
        dd = d["dedup"]
        dup_rate = 1 - dd["n_out"] / dd["n_in"]
        return [
            "## Problemas do projeto anterior",
            "",
            "Cada problema traz o que estava errado, por que isso distorce o resultado e o"
            " impacto clínico esperado. Os números vêm do enunciado (PD) ou deste experimento.",
            "",
            "### 1. Accuracy global como única métrica",
            "",
            f"Com {majority * 100:.0f}% das imagens na classe Normal ({max(tc.values())} de"
            f' {total}), um modelo que responde sempre "Normal" tem {pct(majority, 0)} de'
            " accuracy, mais que os 61% reportados na validação, e recall zero para COVID. A"
            " accuracy mistura as classes e esconde justamente o erro que importa. Nour & Tariq"
            " (2023) acharam recall abaixo de 60% para COVID em modelos com accuracy alta.",
            "",
            "**Impacto clínico:** o hospital recebe um número que parece aceitável enquanto os"
            " casos positivos passam como negativos. É o que o time clínico relatou: o modelo"
            ' "ignora casos positivos".',
            "",
            "### 2. Desbalanceamento não tratado",
            "",
            f"COVID é {pct(n_covid / total, 0)} do conjunto ({n_covid} de {total}). Sem class"
            " weights, reamostragem ou augmentation dirigida, a perda é dominada pela classe"
            " Normal e o modelo aprende que errar COVID custa pouco.",
            "",
            "**Impacto clínico:** falsos negativos de COVID. Um paciente infectado classificado"
            " como normal pode ser liberado sem isolamento nem tratamento, e transmitir o vírus.",
            "",
            "### 3. Divisão 80/20 sem estratificação e sem teste separado",
            "",
            f"Sem estratificar, a validação de {n_val} imagens tem em média"
            f" {hyper.mean():.0f} COVID, mas o sorteio pode dar de {lo} a {hi} (intervalo de"
            " 95%, distribuição hipergeométrica). O recall da COVID passa a depender da sorte do"
            " sorteio. E a mesma validação serviu para acompanhar o treino e para reportar o"
            " resultado: não há teste intocado, então o número final é otimista.",
            "",
            "**Impacto clínico:** a sensibilidade prometida não se repete no uso real. Uma"
            " decisão de implantar tomada com esse número expõe pacientes a um modelo pior do"
            " que o anunciado.",
            "",
            "### 4. ResNet-18 sem pré-treino em 1.200 imagens: overfitting",
            "",
            "Treinar 11 milhões de parâmetros do zero com 1.200 imagens leva à memorização: 93%"
            " de accuracy no treino contra 61% na validação, uma distância de 32 pontos. O"
            " modelo aprendeu detalhes das imagens de treino, não padrões de doença. Pesos"
            " pré-treinados (ImageNet ou, melhor, raio-X) dão características gerais prontas.",
            "",
            "**Impacto clínico:** o desempenho cai em imagens novas, de outro aparelho ou de"
            " outro hospital. O modelo fica instável justamente fora do ambiente em que foi"
            " testado.",
            "",
            "### 5. Nenhuma augmentation",
            "",
            "Sem rotação, escala, brilho ou contraste, a rede só vê cada raio-X de um jeito. Ela"
            " fica sensível a posicionamento do paciente, exposição e equipamento, e a classe"
            f" rara continua com só {n_covid} exemplos.",
            "",
            "**Impacto clínico:** um raio-X portátil (leito, AP) ou com exposição diferente pode"
            " ser classificado errado. Esses são justamente os pacientes mais graves.",
            "",
            "### 6. Otimização sem critério: SGD, lr fixo de 0,01, 15 épocas",
            "",
            "Learning rate fixo e alto, sem agenda, sem early stopping e sem escolher a época"
            " pela validação: o resultado é o da época 15, seja ela boa ou não. Não há curva de"
            " treino nem repetição com outra seed para saber se o número é estável.",
            "",
            "**Impacto clínico:** o modelo não é reprodutível. Retreinar pode dar outro"
            " resultado, e o hospital não tem como confiar que a versão implantada é a medida.",
            "",
            "### 7. Cópias no dataset e risco de vazamento",
            "",
            f"Neste dataset público, {pct(dup_rate)} das imagens eram cópias ou quase-cópias"
            f" ({dd['copias_removidas']} repetidas e {dd['removidas_por_conflito_de_rotulo']}"
            " com rótulos em conflito). Sem dedup, a mesma radiografia cai no treino e na"
            " validação, e a validação mede memória, não generalização. O projeto não fala de"
            " dedup nem de separar por paciente.",
            "",
            "**Impacto clínico:** desempenho superestimado. Imagens com rótulos em conflito"
            " também ensinam o modelo errado.",
            "",
            "### 8. Resultado único, sem intervalo e sem ponto de operação",
            "",
            "Um único treino e uma única accuracy, sem intervalo de confiança, matriz de"
            " confusão ou limiar escolhido para triagem. Em triagem, o limiar deve ser ajustado"
            " para alta sensibilidade, aceitando mais falsos positivos.",
            "",
            '**Impacto clínico:** "promissor" sem intervalo pode ser ruído. E sem ponto de'
            " operação definido, não se sabe quantos positivos o sistema deixa passar por cada"
            " 100 casos.",
            "",
        ]

    def _gan(self, d: dict[str, Any]) -> list[str]:
        g = self.cfg.gan
        gan = d["gan"]
        lines = [
            "## GAN condicional",
            "",
            f"DCGAN condicional em {g.image_size} x {g.image_size}, em tons de cinza, treinada"
            " só com o treino do caso"
            f" ({gan['fid_reference']['n_real']} imagens, as três classes)."
            f" O gerador recebe ruído z ({g.nz} números) concatenado a um embedding da classe e"
            " sobe a resolução com convoluções transpostas até 64 x 64, com `tanh` no fim. O"
            " discriminador recebe a imagem e um mapa da classe (embedding do rótulo do tamanho"
            " da imagem) como segundo canal, e devolve um logit (real ou falsa). Sem max pooling;"
            " pesos iniciados como no artigo da DCGAN (aula 07).",
            "",
            "**Loop adversarial**, por lote:",
            "",
            '1. Passo do discriminador: imagens reais com o rótulo certo têm alvo "real",'
            ' imagens geradas (com rótulos sorteados) têm alvo "falsa". Só os pesos do'
            " discriminador mudam; a imagem gerada entra desligada do grafo do gerador.",
            "2. Passo do gerador: as mesmas imagens geradas passam pelo discriminador com alvo"
            ' "real" (perda não saturante, aula 07). Só os pesos do gerador mudam.',
            "",
            f"Adam com lr {num(g.lr, 4)} e betas ({num(g.betas[0], 1)}; {num(g.betas[1], 3)}),"
            f" lote {g.batch_size},"
            f" {g.epochs} épocas. Os testes `test_a4.py` conferem que cada passo muda só a sua"
            " rede e que o loop alterna D e G.",
            "",
            "| Variante | Passos de D | Passos de G | Tempo (s) |",
            "|---|---|---|---|",
        ]
        for name, r in gan["runs"].items():
            lines.append(
                f"| {name} | {r['n_d_steps']} | {r['n_g_steps']} | {num(r['seconds'], 0)} |"
            )
        lines += [
            "",
            "Amostras com o mesmo z ao longo do treino (uma linha por classe:"
            f" {', '.join(self.cfg.classes)}):",
            "",
        ]
        for name, r in gan["runs"].items():
            grids = r["sample_grids"]
            for gname in (grids[0], grids[len(grids) // 2], grids[-1]):
                lines.append(f"![{name}, {gname}]({self.fig_prefix}/{name}_{gname})")
            lines.append("")
        return lines

    def _instability(self, d: dict[str, Any]) -> list[str]:
        gan = d["gan"]
        runs = gan["runs"]
        names = list(runs)
        lines = [
            "## Instabilidade e mitigação",
            "",
            "Treinamos a mesma GAN duas vezes: sem mitigação e com"
            f" {', '.join(runs[names[-1]]['strategies'])}. As estratégias vêm da aula 07: rótulo"
            " 0,9 em vez de 1 (o discriminador fica menos confiante), spectral norm no"
            " discriminador (limita o quanto ele muda por passo) e replay buffer (o"
            ' discriminador revê imagens falsas antigas e o gerador não consegue "pular" de um'
            " modo para outro).",
            "",
            f"Como medimos: FID entre {gan['fid_reference']['n_fake']} imagens geradas e as"
            f" {gan['fid_reference']['n_real']} reais do treino, na mesma resolução"
            f" ({gan['fid_reference']['resolucao']} px, Inception-v3 do ImageNet, aula 08),"
            f" a cada {self.cfg.gan.fid_every} épocas. O FID se divide no termo da média (o"
            " gerado está deslocado do real?) e no termo da covariância (a variedade bate?)."
            " Para colapso de modo, medimos também a diversidade da COVID gerada: distância"
            " média entre pares de imagens geradas dividida pela mesma distância entre as reais"
            " (1 = mesma variedade; abaixo de 0,5 tratamos como colapso).",
            "",
            "| Métrica | " + " | ".join(names) + " |",
            "|---|" + "---|" * len(names),
        ]

        def row(label: str, fn: Any) -> str:
            return f"| {label} | " + " | ".join(fn(runs[n]) for n in names) + " |"

        lines += [
            row("FID final", lambda r: num(r["fid_final"], 1)),
            row("Melhor FID (época)", lambda r: f"{num(r['fid_best'], 1)} ({r['fid_best_epoch']})"),
            row("Termo da média no melhor", lambda r: num(r["fid_best_terms"]["mean_term"], 1)),
            row(
                "Termo da covariância no melhor", lambda r: num(r["fid_best_terms"]["cov_term"], 1)
            ),
            row(
                "FID só da COVID (reais x geradas)",
                lambda r: (
                    f"{num(r['fid_covid_best'], 1)} (n = {r['fid_covid_n'][0]} x"
                    f" {r['fid_covid_n'][1]})"
                ),
            ),
            row(
                "Diversidade da COVID gerada", lambda r: num(r["diagnostico"]["diversidade_covid"])
            ),
            row("D(x) real, fim do treino", lambda r: num(r["diagnostico"]["d_real_final"])),
            row("D(G(z)) falsa, fim do treino", lambda r: num(r["diagnostico"]["d_fake_final"])),
            row(
                "Discriminador dominou", lambda r: yes_no(r["diagnostico"]["discriminador_dominou"])
            ),
            row(
                "FID subiu mais de 10% depois do melhor",
                lambda r: yes_no(r["diagnostico"]["fid_subiu_depois_do_melhor"]),
            ),
            row("Colapso de modo", lambda r: yes_no(r["diagnostico"]["colapso_de_modo"])),
            "",
            f"![Curvas das duas variantes]({self.fig_prefix}/gan_comparison.png)",
            "",
            "**Diagnóstico.**",
            "",
        ]
        for n in names:
            dg = runs[n]["diagnostico"]
            signs = []
            if dg["discriminador_dominou"]:
                signs.append(
                    f"o discriminador dominou (D(x) = {num(dg['d_real_final'])} e D(G(z)) ="
                    f" {num(dg['d_fake_final'])} no fim): o gerador recebe gradiente fraco, sinal"
                    " de divergência"
                )
            if dg["fid_subiu_depois_do_melhor"]:
                signs.append(
                    f"o FID piorou depois da época {runs[n]['fid_best_epoch']}"
                    f" ({num(runs[n]['fid_best'], 1)} para {num(runs[n]['fid_final'], 1)}): o"
                    " treino não se sustentou"
                )
            if dg["colapso_de_modo"]:
                signs.append(
                    "colapso de modo: a COVID gerada tem diversidade"
                    f" {num(dg['diversidade_covid'])} da real"
                )
            if not dg["discriminador_dominou"]:
                signs.append(
                    f"o discriminador não dominou (D(x) = {num(dg['d_real_final'])} e D(G(z)) ="
                    f" {num(dg['d_fake_final'])} no fim): o gerador continua recebendo gradiente"
                )
            if not dg["colapso_de_modo"] and not dg["fid_subiu_depois_do_melhor"]:
                signs.append(
                    f"sem sinal de colapso de modo (diversidade {num(dg['diversidade_covid'])})"
                )
            lines.append(f"- **{n}:** " + "; ".join(signs) + ".")
        comp = gan.get("comparacao", {})
        if comp:
            a, b = comp["fid_best"]
            fa, fb = comp["fid_final"]
            if comp["melhora_fid"]:
                verdict = (
                    f"A mitigação melhorou o FID: o melhor FID foi de {num(a, 1)} para {num(b, 1)}"
                    f" ({pct((b - a) / a)}), e o FID final de {num(fa, 1)} para {num(fb, 1)}."
                )
            else:
                verdict = (
                    f"A mitigação não melhorou o FID: o melhor FID foi de {num(a, 1)} para"
                    f" {num(b, 1)}, e o FID final de {num(fa, 1)} para {num(fb, 1)}."
                )
            da, db = (runs[n]["diagnostico"]["diversidade_covid"] for n in names[:2])
            ca, cb = (runs[n]["diagnostico"]["colapso_de_modo"] for n in names[:2])
            if cb:
                verdict += (
                    " A mitigação estabilizou o jogo entre as redes, mas não resolveu o colapso de"
                    f" modo: a diversidade da COVID gerada foi de {num(da)} para {num(db)}, e as"
                    " imagens de uma mesma classe saem quase iguais mesmo com z diferente."
                )
            elif ca:
                verdict += (
                    f" A mitigação também tirou a GAN do colapso de modo: a diversidade foi de"
                    f" {num(da)} para {num(db)}."
                )
            lines += [
                "",
                f"**Evidência.** {verdict} Variante usada para gerar as COVID"
                f" sintéticas: `{gan['chosen_for_synthetic']}` (melhor checkpoint por FID).",
            ]
        lines += [
            "",
            "Limites do FID aqui: a Inception foi treinada em fotos do ImageNet, não em raio-X,"
            " e o FID só da COVID usa poucas reais, então é enviesado para cima. Serve para"
            " comparar as duas variantes medidas do mesmo jeito, não como nota absoluta"
            " (aula 08).",
            "",
            f"![COVID sintéticas usadas no treino]({self.fig_prefix}/synthetic_grid.png)",
            "",
        ]
        return lines

    def _recall(self, d: dict[str, Any]) -> list[str]:
        s = d["summary"]
        df = d["recall"]
        c = self.cfg.classifier
        n_syn = int(df.loc[df.condition == "com_sintetico", "n_synthetic"].iloc[0])
        n_test_pos = int(df["n_test_covid"].iloc[0])
        tc = self.cfg.case.train_counts
        total_case = sum(tc.values())
        preds = d["preds"]
        pos = self.cfg.positive_index
        fn = preds[(preds["true"] == pos) & (preds["pred"] != pos)]
        fn_normal = int((fn["pred"] == self.cfg.classes.index("NORMAL")).sum()) if len(fn) else 0
        sem_mean = s["conditions"]["sem_sintetico"]["recall_covid"]["mean"]
        lines = [
            "## Recall da COVID com e sem sintético",
            "",
            f"Mesmo classificador nas duas condições: ResNet-18 pré-treinada no ImageNet, com"
            f" class weights, augmentation leve (rotação, escala, translação, brilho e"
            f" contraste), AdamW (lr {num(c.lr, 4)}), {c.epochs} épocas, imagem"
            f" {c.image_size} x {c.image_size}, época escolhida pelo maior recall médio na"
            f' validação. "Com sintético" soma {n_syn} COVID geradas (64 x 64, redimensionadas)'
            f" ao treino. As duas condições são avaliadas no mesmo teste real ({n_test_pos} COVID),"
            f" em {len(c.seeds)} seeds.",
            "",
            "| Condição | Recall COVID: média | IC 95% | Mín. a máx. | Por seed |"
            " Recall médio (3 classes) | Precisão da COVID na prevalência do caso |",
            "|---|---|---|---|---|---|---|",
        ]
        for cond, st in s["conditions"].items():
            r = st["recall_covid"]
            per_seed = ", ".join(
                num(v) for v in df[df.condition == cond].sort_values("seed")["recall_covid"]
            )
            lines.append(
                f"| {cond} | {num(r['mean'], 3)} | {num(r['ci_low'], 3)} a {num(r['ci_high'], 3)}"
                f" | {num(r['min'], 3)} a {num(r['max'], 3)} | {per_seed} |"
                f" {num(st['balanced_accuracy']['mean'], 3)} |"
                f" {num(st['ppv_covid_prevalencia_caso']['mean'], 3)} |"
            )
        pdiff = s.get("paired_difference")
        if pdiff:
            text = {
                "melhora": "o sintético melhorou o recall da COVID: o intervalo da diferença fica"
                " todo acima de zero.",
                "piora": "o sintético piorou o recall da COVID: o intervalo da diferença fica todo"
                " abaixo de zero.",
                "inconclusivo": "não dá para dizer que o sintético mudou o recall da COVID: o"
                " intervalo da diferença inclui o zero. Com 3 seeds, uma diferença desse tamanho"
                " pode ser ruído.",
            }[pdiff["verdict"]]
            lines += [
                "",
                f"Diferença pareada por seed (com menos sem): média {num(pdiff['mean'], 3)},"
                f" IC 95% de {num(pdiff['ci_low'], 3)} a {num(pdiff['ci_high'], 3)}"
                f" (por seed: {', '.join(num(v, 3) for v in pdiff['per_seed'])}). Veredito:"
                f" {text}",
            ]
        lines += [
            "",
            "O intervalo usa a distribuição t com n = 3 seeds; ele mede a variação do treino,"
            " não a incerteza da amostra de teste. A precisão na prevalência do caso recalcula a"
            " precisão como se o teste tivesse "
            + ", ".join(f"{pct(v / total_case, 0)} {k}" for k, v in tc.items())
            + ".",
            "",
            f"![Recall por seed e curvas de validação]({self.fig_prefix}/recall.png)",
            "",
            f"O classificador sem sintético já acerta {pct(sem_mean)} das COVID reais: sobra pouco"
            " espaço para o sintético ajudar (efeito teto). Isso contrasta com o projeto anterior,"
            ' que "ignora casos positivos": com pré-treino, class weights, augmentation e dedup'
            " juntos o recall já é alto (este experimento não separa qual deles pesa mais).",
            "",
            f"Análise de erros: somando as {len(df)} execuções, houve {len(fn)} COVID não"
            f" reconhecidas, e {fn_normal} delas foram chamadas de Normal, não de Pneumonia. São"
            " possivelmente casos com pouco achado visível, em que a triagem por imagem é mais"
            " fraca. Abaixo, as da primeira seed.",
            "",
            f"![Falsos negativos]({self.fig_prefix}/falsos_negativos.png)",
            "",
        ]
        return lines

    def _plan(self, d: dict[str, Any]) -> list[str]:
        s = d["summary"]
        sem = s["conditions"]["sem_sintetico"]["recall_covid"]
        com = s["conditions"]["com_sintetico"]["recall_covid"]
        verdict = s.get("paired_difference", {}).get("verdict", "inconclusivo")
        gan = d["gan"]
        syn_text = {
            "melhora": "Neste experimento o sintético melhorou o recall, então ele entra no"
            " pipeline, com as três travas.",
            "piora": "Neste experimento o sintético piorou o recall, então ele não entra no"
            " pipeline até uma GAN melhor passar pelas três travas.",
            "inconclusivo": "Neste experimento o ganho não se separou do ruído, então o sintético"
            " fica como opção, não como padrão, até passar pelas três travas.",
        }[verdict]
        best_fid = min(r["fid_best"] for r in gan["runs"].values())
        chosen = gan["runs"][gan["chosen_for_synthetic"]]["diagnostico"]
        if chosen["colapso_de_modo"]:
            syn_text = (
                "Neste experimento a trava (1) falhou: a GAN escolhida colapsou (diversidade"
                f" {num(chosen['diversidade_covid'])}), então as COVID sintéticas são quase a mesma"
                f" imagem repetida (veredito do recall: {verdict}). O sintético não entra no"
                " pipeline até uma GAN passar pelas três travas."
            )
        return [
            "## Plano de melhoria integrado",
            "",
            "Cada problema da lista acima tem uma ação. O plano se organiza em dados, modelo,"
            " métrica, augmentation sintética e critério de adoção clínica.",
            "",
            "| Problema | Ação |",
            "|---|---|",
            "| 1. Accuracy como única métrica | recall da COVID como métrica principal,"
            " matriz de confusão e recall por classe |",
            "| 2. Desbalanceamento | class weights (já aplicado aqui) e sintético condicionado |",
            "| 3. Split sem estratificação e sem teste | teste real separado primeiro;"
            " estratificação; o teste é usado uma vez |",
            "| 4. Sem pré-treino, overfitting | ResNet-18 pré-treinada, fine-tuning, weight"
            " decay |",
            "| 5. Sem augmentation | augmentation leve e plausível para raio-X |",
            "| 6. Otimização sem critério | AdamW com agenda, escolha da época na validação,"
            " várias seeds |",
            "| 7. Cópias e vazamento | dedup por pHash antes do split; split por paciente quando"
            " houver o identificador |",
            "| 8. Resultado sem intervalo nem limiar | média e IC em seeds; limiar escolhido"
            " para sensibilidade |",
            "",
            "### Dados e validação",
            "",
            "Dedup por hash perceptual antes de qualquer split (aqui removeu"
            f" {pct(1 - d['dedup']['n_out'] / d['dedup']['n_in'])} das imagens). Teste real"
            " separado primeiro e nunca usado para escolher nada. Quando houver identificador de"
            " paciente, separar por paciente. Validação externa com imagens de outro hospital"
            " antes de qualquer uso.",
            "",
            "### Modelo",
            "",
            "Partir de pesos pré-treinados (ImageNet, ou melhor, um modelo pré-treinado em raio-X"
            " de tórax) e fazer fine-tuning com AdamW, agenda de learning rate, weight decay e"
            " escolha da época pela validação, como aqui. Class weights na perda. Repetir em"
            " várias seeds e reportar média e intervalo. Como próximo passo, comparar com uma"
            " arquitetura maior (DenseNet-121, comum em raio-X) com o mesmo protocolo, e"
            " calibrar as probabilidades (temperature scaling) para que o limiar signifique algo.",
            "",
            "### Métrica",
            "",
            "Recall (sensibilidade) da COVID como métrica principal, porque o custo clínico é o"
            " falso negativo. Junto: recall por classe, matriz de confusão, recall médio das três"
            " classes e precisão da COVID na prevalência esperada no hospital (não na do teste)."
            " Todo número com intervalo. Escolher o limiar de decisão na validação para atingir a"
            " sensibilidade alvo, e reportar a especificidade que sobra.",
            "",
            "### Augmentation sintética",
            "",
            "GAN condicional treinada só com o treino, com mitigação de instabilidade (label"
            " smoothing, spectral norm, replay buffer) e escolha do checkpoint pelo FID (aqui o"
            f" melhor FID foi {num(best_fid, 1)}). Três travas antes de usar: (1) FID e"
            " diversidade medidos e sem colapso de modo; (2) o teste fica 100% real; (3) o ganho"
            " no recall da COVID precisa ter intervalo acima de zero em várias seeds."
            f" {syn_text} Próximos passos: atacar o colapso com minibatch discrimination (o"
            " discriminador compara as imagens do lote entre si) e learning rates diferentes"
            " para D e G; gerar em resolução maior (128 ou 256 px) para não"
            ' ensinar ao classificador que "imagem borrada = COVID", revisar uma amostra das'
            " geradas com um radiologista e testar um modelo de difusão com o mesmo protocolo.",
            "",
            "### Critério de adoção clínica",
            "",
            "O modelo só vai para uso assistido (segunda leitura, nunca decisão sozinho) se, num"
            " teste externo real com a prevalência do hospital: (1) o recall da COVID for pelo"
            " menos 0,90 com o limite inferior do IC 95% acima de 0,85; (2) nenhuma classe tiver"
            " recall abaixo de 0,80; (3) a precisão da COVID na prevalência real for informada à"
            " equipe; (4) o resultado se mantiver num período silencioso (o modelo roda sem"
            " influenciar a conduta, comparado com o laudo). Depois, monitorar o recall por mês e"
            " retreinar quando cair. Hoje o recall médio da COVID no teste real é"
            f" {num(sem['mean'], 3)} sem sintético e {num(com['mean'], 3)} com sintético; o"
            " critério (1) "
            + (
                "é atingido no teste interno, o que ainda não dispensa o teste externo."
                if max(sem["ci_low"], com["ci_low"]) >= 0.85
                and max(sem["mean"], com["mean"]) >= 0.9
                else "ainda não é atingido."
            ),
            "",
        ]

    def _limits(self, d: dict[str, Any]) -> list[str]:
        n_pos = int(d["recall"]["n_test_covid"].iloc[0])
        return [
            "## Limites",
            "",
            "- Uma execução de cada GAN (uma seed): a comparação sem e com mitigação não tem"
            " intervalo.",
            "- O sintético é 64 x 64 redimensionado para 224: o classificador pode associar"
            " imagem borrada a COVID. O teste real mede o efeito final, mas não separa as causas.",
            f"- {len(self.cfg.classifier.seeds)} seeds dão um intervalo largo; o teste tem"
            f" {n_pos} COVID, então cada imagem vale {num(1 / n_pos, 4)} de recall.",
            "- O dataset junta fontes diferentes por classe; o modelo pode aprender a fonte, não"
            " a doença. Só um teste externo responde isso.",
            "",
        ]

    def render(self) -> str:
        """Texto completo da seção."""
        d = self.load()
        parts = [
            self._header(d),
            self._data(d),
            self._problems(d),
            self._gan(d),
            self._instability(d),
            self._recall(d),
            self._plan(d),
            self._limits(d),
        ]
        return "\n".join(line for part in parts for line in part).rstrip() + "\n"

    def copy_figures(self, figures_dir: Path) -> list[Path]:
        """Copia as figuras citadas para reports/figures/a4/."""
        figures_dir.mkdir(parents=True, exist_ok=True)
        out = []
        for src, dst in FIGURES.items():
            if (self.run_path / src).exists():
                out.append(Path(shutil.copy2(self.run_path / src, figures_dir / dst)))
        gan = json.loads((self.run_path / "gan_metrics.json").read_text(encoding="utf-8"))
        for name, r in gan["runs"].items():
            grids = r["sample_grids"]
            for gname in {grids[0], grids[len(grids) // 2], grids[-1]}:
                src = self.run_path / "gan" / name / gname
                out.append(Path(shutil.copy2(src, figures_dir / f"{name}_{gname}")))
        return out

    def write(self, path: Path, figures_dir: Path | None = None) -> Path:
        """Grava a seção em `path` e copia as figuras."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if figures_dir is not None:
            self.copy_figures(figures_dir)
            self.fig_prefix = Path(os.path.relpath(figures_dir, path.parent)).as_posix()
        path.write_text(self.render(), encoding="utf-8")
        return path
