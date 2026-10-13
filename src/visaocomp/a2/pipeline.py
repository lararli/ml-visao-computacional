"""A2Pipeline: amostra, embeddings no DuckDB, ranking de objetos, busca, alinhamento e R20."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from visaocomp.a2.alignment import AlignmentProbe
from visaocomp.a2.config import A2Config
from visaocomp.a2.embedder import ClipEmbedder, Embedder
from visaocomp.a2.figures import FigureBuilder
from visaocomp.a2.ranker import ObjectRanker, ThresholdCalibrator
from visaocomp.a2.sampler import Ads16Sampler
from visaocomp.a2.search import TextSearch
from visaocomp.a2.store import EmbeddingStore
from visaocomp.a2.textinput import TextInputComparison
from visaocomp.config import load_data_config
from visaocomp.run import RunDir

log = logging.getLogger(__name__)

STAGE = "a2"


class ThresholdCheckMissingError(RuntimeError):
    """Falta o arquivo com os pares conferidos a olho (ou há pares sem rótulo)."""


def load_threshold_check(path: Path, minimum: int) -> dict[str, Any]:
    """Lê os pares rotulados (present: true/false) e confere que há pelo menos `minimum`."""
    if not Path(path).exists():
        raise ThresholdCheckMissingError(
            f"esperava {path} com {minimum} pares conferidos; rode `run_a2.py --calibrate` "
            "e rotule os pares (present: true/false)"
        )
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs = doc.get("pairs", [])
    unlabeled = [p for p in pairs if not isinstance(p.get("present"), bool)]
    if unlabeled or len(pairs) < minimum:
        raise ThresholdCheckMissingError(
            f"esperava {minimum} pares com present true/false em {path}; "
            f"vieram {len(pairs)}, {len(unlabeled)} sem rótulo"
        )
    return doc


class A2Pipeline:
    """Orquestra a atividade 2. Dependências entram pelo construtor (config, dados, embedder)."""

    def __init__(
        self,
        cfg: A2Config,
        data_dir: Path,
        outputs_root: Path,
        embedder: Embedder,
        store: EmbeddingStore | None = None,
        threshold_check: Path | None = None,
        text_comparison: Callable[[], dict[str, Any]] | None = None,
        ads16_root: Path | None = None,
        data_cfg_path: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.data_dir = Path(data_dir)
        self.outputs_root = Path(outputs_root)
        self.embedder = embedder
        self.store = store or EmbeddingStore(self.data_dir / "catalog.duckdb", cfg.embedding_table)
        self.threshold_check = Path(threshold_check) if threshold_check else None
        self.text_comparison = text_comparison
        self.sampler = Ads16Sampler(ads16_root or self.data_dir / "raw" / "ads16")
        self.data_cfg_path = data_cfg_path

    # ---- etapas ---------------------------------------------------------------------------

    def embed_sample(self, sample: pd.DataFrame) -> dict[str, int]:
        """Calcula só o que falta no DuckDB. Rodar de novo dá 0 novos."""
        model = self.cfg.model
        paths = list(sample.image_path)
        have = self.store.existing(paths, model)
        todo = sample[~sample.image_path.isin(have)].reset_index(drop=True)
        new = unreadable = 0
        if len(todo):
            vectors, ok, errors = self.embedder.embed_images([Path(p) for p in todo.abs_path])
            unreadable = len(errors)
            for err in errors:
                log.warning("imagem ilegível, pulada: %s", err)
            if ok:
                new = self.store.add(todo.iloc[ok].reset_index(drop=True), vectors, model)
        counts = {"new": new, "reused": len(have), "unreadable": unreadable}
        log.info(
            "embeddings: %d novos, %d reaproveitados, %d ilegíveis",
            new,
            len(have),
            unreadable,
        )
        return counts

    def _context(self) -> dict[str, Any]:
        """Amostra + embeddings + similaridades com as descrições."""
        contract = self.sampler.validate()
        sample = self.sampler.sample(self.cfg.sample_size, self.cfg.seed)
        counts = self.embed_sample(sample)
        meta, image_vecs = self.store.load(list(sample.image_path), self.cfg.model)
        meta = meta.merge(
            sample[["image_path", "abs_path", "stratum"]], on="image_path", how="left"
        )
        groups = [self.cfg.prompts(d) for d in self.cfg.descriptions]
        if hasattr(self.embedder, "embed_prompt_groups"):
            text_vecs = self.embedder.embed_prompt_groups(groups)
        else:  # embedder simples: média feita aqui
            from visaocomp.a2.embedder import l2_normalize

            text_vecs = l2_normalize(
                np.stack([self.embedder.embed_texts(g).mean(axis=0) for g in groups])
            )
        sims = ObjectRanker.similarities(image_vecs, text_vecs)
        return {
            "contract": contract,
            "sample": sample,
            "counts": counts,
            "meta": meta,
            "image_vecs": image_vecs,
            "text_vecs": text_vecs,
            "sims": sims,
            "resolve": {r.image_path: Path(r.abs_path) for r in meta.itertuples()},
        }

    def calibrate(self, out_dir: Path) -> Path:
        """Sorteia os pares para conferir a olho e grava folhas sem o score (avaliação cega)."""
        ctx = self._context()
        cal = ThresholdCalibrator(self.cfg.seed, self.cfg.n_threshold_check)
        pairs = cal.candidate_pairs(
            ctx["sims"], list(ctx["meta"].image_path), self.cfg.descriptions
        )
        order = np.random.default_rng(self.cfg.seed + 1).permutation(len(pairs))
        pairs = [
            {"id": n + 1, **pairs[i], "present": None, "note": ""} for n, i in enumerate(order)
        ]
        out_dir.mkdir(parents=True, exist_ok=True)
        figs = FigureBuilder(ctx["resolve"])
        per_sheet = 10
        for s in range(0, len(pairs), per_sheet):
            cells = [
                (p["image_path"], f"#{p['id']}: {p['description']}")
                for p in pairs[s : s + per_sheet]
            ]
            figs.sheet(cells, out_dir / f"threshold_sheet_{s // per_sheet + 1}.png", ncols=5)
        target = out_dir / "threshold_candidates.json"
        target.write_text(
            json.dumps(
                {
                    "rater": "preencher",
                    "criterion": "present = a descrição aparece de forma reconhecível na imagem",
                    "model": self.cfg.model,
                    "seed": self.cfg.seed,
                    "pairs": pairs,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        log.info("calibração: %d pares em %s", len(pairs), out_dir)
        return target

    def _threshold(self, ctx: dict[str, Any]) -> dict[str, Any]:
        if self.threshold_check is None:
            raise ThresholdCheckMissingError("sem arquivo de conferência do limiar")
        doc = load_threshold_check(self.threshold_check, self.cfg.n_threshold_check)
        index = {p: i for i, p in enumerate(ctx["meta"].image_path)}
        col = {d: j for j, d in enumerate(self.cfg.descriptions)}
        scores, present, used = [], [], []
        for p in doc["pairs"]:
            if p["image_path"] not in index or p["description"] not in col:
                continue
            scores.append(float(ctx["sims"][index[p["image_path"]], col[p["description"]]]))
            present.append(bool(p["present"]))
            used.append(p)
        if len(scores) < self.cfg.n_threshold_check:
            raise ThresholdCheckMissingError(
                f"só {len(scores)} dos pares conferidos estão na amostra; esperava "
                f"{self.cfg.n_threshold_check}"
            )
        chosen = ThresholdCalibrator.choose(scores, present)
        chosen["distribution"] = ThresholdCalibrator.distribution(ctx["sims"])
        chosen["rater"] = doc.get("rater", "")
        chosen["criterion"] = doc.get("criterion", "")
        chosen["pairs"] = [{**p, "score": s} for p, s in zip(used, scores, strict=True)]
        return chosen

    # ---- execução completa ------------------------------------------------------------------

    def run(self) -> RunDir:
        """Executa tudo e grava outputs/<run_id>/ (a2/*, metrics.json, manifest.json)."""
        t0 = time.time()
        run = RunDir(self.outputs_root, STAGE)
        out = run.path / "a2"
        out.mkdir()
        log.info("A2: início, run %s", run.run_id)
        ctx = self._context()
        meta, sims = ctx["meta"], ctx["sims"]
        figs = FigureBuilder(ctx["resolve"])

        # mapa pasta -> categoria e a amostra (lista no manifesto)
        category_map = self.sampler.category_map()
        (out / "category_map.json").write_text(
            json.dumps(
                {
                    "folder_to_category": category_map,
                    "source": "cabeçalho de Corpus/U0001/U0001-RT.csv (Cat0..Cat19, na ordem das "
                    "pastas 1..20); confere com a Tabela 1 do artigo em Documents/",
                    "ads_per_folder": ctx["contract"]["ads_per_folder"],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        ctx["sample"].drop(columns=["abs_path"]).to_csv(out / "sample.csv", index=False)

        # limiar
        thr = self._threshold(ctx)
        (out / "threshold.json").write_text(
            json.dumps(thr, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        ranker = ObjectRanker(self.cfg.descriptions, thr["threshold"])

        # R18: ranking e figura dos 5 mais frequentes
        ranking = ranker.rank(sims, meta["source"])
        ranking.to_csv(out / "ranking.csv", index=False)
        # score de cada anúncio para cada descrição: vira gold.ads16_ad_object no pipeline
        scores = pd.DataFrame(sims, columns=list(self.cfg.descriptions))
        scores.insert(0, "image_path", list(meta["image_path"]))
        scores.insert(1, "source", list(meta["source"]))
        scores.to_csv(out / "scores.csv", index=False)
        top5 = list(ranking.description[:5])
        rows, examples = [], {}
        for d in top5:
            ex = ranker.examples(sims, d, self.cfg.examples_per_object)
            examples[d] = [
                {"image_path": meta.image_path[i], "score": s, "source": meta.source[i]}
                for i, s in ex
            ]
            rows.append([(meta.image_path[i], f"{s:.3f}") for i, s in ex])
        labels = [
            f"{d}\n({ranking.set_index('description').frequency[d]:.0%} das imagens)" for d in top5
        ]
        figs.grid(rows, out / "top5.png", row_labels=labels, title="5 objetos mais frequentes")
        (out / "top5_examples.json").write_text(
            json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # R19: busca por texto, cosseno dentro do DuckDB
        search = TextSearch(self.embedder, self.store, list(meta.image_path), self.cfg.top_k)
        queries = search.run_all(self.cfg.queries)
        (out / "queries.json").write_text(
            json.dumps(queries, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        for q in queries:
            figs.grid(
                [[(r["image_path"], f"#{r['rank']} {r['score']:.3f}") for r in q["results"]]],
                out / "queries" / f"{q['id']}.png",
                title=f"{q['id']}: {q['text']}",
            )

        # R17: alinhamento medido
        alignment = AlignmentProbe(self.embedder).run(
            ctx["image_vecs"], meta, category_map, ctx["text_vecs"], self.cfg.seed
        )
        (out / "alignment.json").write_text(
            json.dumps(alignment, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # R20: os dois tokenizadores
        if self.text_comparison is not None:
            (out / "tokenizers.json").write_text(
                json.dumps(self.text_comparison(), indent=2, ensure_ascii=False), encoding="utf-8"
            )

        seconds = round(time.time() - t0, 1)
        metrics = {
            "n_sample": int(len(ctx["sample"])),
            "n_embedded": int(len(meta)),
            "embeddings_new": ctx["counts"]["new"],
            "embeddings_reused": ctx["counts"]["reused"],
            "unreadable": ctx["counts"]["unreadable"],
            "threshold": thr["threshold"],
            "threshold_accuracy": thr["accuracy"],
            "n_descriptions": len(self.cfg.descriptions),
            "top5": top5,
            "n_queries": len(queries),
            "zero_shot_top1": alignment["zero_shot_top1"],
            "seconds": seconds,
        }
        run.write_json("metrics.json", metrics)
        data_cfg = load_data_config(self.data_cfg_path) if self.data_cfg_path else None
        sources = []
        if data_cfg is not None:
            src = data_cfg.source("ads16")
            sources.append(
                {
                    "name": src.name,
                    "source": f"{src.owner}/{src.dataset}",
                    "version": src.version,
                    "sha256": src.sha256,
                }
            )
        run.write_manifest(
            sources,
            counts={
                "ads": int((ctx["sample"].source == "ad").sum()),
                "personal_photos": int((ctx["sample"].source == "pessoal").sum()),
                **ctx["counts"],
            },
            extra={
                "model": self.cfg.model,
                "seed": self.cfg.seed,
                "embedding_table": self.cfg.embedding_table,
                "sample_image_paths": list(ctx["sample"].image_path),
                "contract": ctx["contract"],
            },
        )
        log.info("A2: fim em %.1fs; %s", seconds, run.path)
        return run


def build_pipeline(
    cfg: A2Config,
    data_dir: Path,
    outputs_root: Path,
    threshold_check: Path | None,
    data_cfg_path: Path | None = None,
    with_text_comparison: bool = True,
) -> A2Pipeline:
    """Monta o pipeline real: CLIP no melhor dispositivo e a comparação com o BERT."""
    embedder = ClipEmbedder(cfg.model, batch_size=cfg.batch_size)
    comparison = None
    if with_text_comparison:
        demo = cfg.tokenizer_demo
        runner = TextInputComparison(embedder, demo["bert_model"])
        comparison = lambda: runner.run(demo["sentences"])  # noqa: E731
    return A2Pipeline(
        cfg,
        data_dir,
        outputs_root,
        embedder,
        threshold_check=threshold_check,
        text_comparison=comparison,
        data_cfg_path=data_cfg_path,
    )
