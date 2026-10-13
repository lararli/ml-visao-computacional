"""Ponto de entrada do `make query` e do `make sql`: consulta as tabelas gold do catalog.duckdb.

`make query` roda as consultas de exemplo (uma por pergunta de negócio, nos 4 datasets) e mostra o
resultado. `make sql Q="SELECT ..."` roda qualquer consulta, só para leitura.
"""

from __future__ import annotations

import argparse
import sys

import duckdb
import pandas as pd

from visaocomp.paths import data_dir

CONSULTAS = {
    "Todos os modelos, com a versão e a métrica principal (gold.model_run)": """
        SELECT activity AS atividade, dataset, model AS modelo, model_version AS versao,
               metric AS metrica, round(value, 3) AS valor
        FROM gold.model_run ORDER BY activity, model""",
    "A1: decisões por atributo e motivo do 'não dá' (gold.austin_house_attribute)": """
        SELECT attribute AS atributo, decision AS decisao, coalesce(reason, '') AS motivo,
               count(*) AS casas
        FROM gold.austin_house_attribute GROUP BY ALL ORDER BY atributo, casas DESC""",
    "A1: casas com piscina na foto e sem licença nem OSM, das mais certas para as menos certas": """
        SELECT a.parcel_id AS lote, a.address AS endereco, round(a.probability, 2) AS probabilidade
        FROM gold.austin_house_attribute a
        JOIN gold.austin_house_attribute_evidence e USING (parcel_id, attribute, model_version)
        WHERE a.attribute = 'pool' AND e.agreement = 'imagem sem licença'
        ORDER BY a.probability DESC LIMIT 10""",
    "A1: concordância entre a foto e os registros (gold.austin_house_attribute_evidence)": """
        SELECT attribute AS atributo, agreement AS concordancia, count(*) AS casas
        FROM gold.austin_house_attribute_evidence GROUP BY ALL ORDER BY atributo, casas DESC""",
    "A2: objetos mais presentes nos anúncios (gold.ads16_ad_object)": """
        SELECT description AS objeto, count(*) FILTER (WHERE present) AS imagens_com_o_objeto,
               round(avg(score), 3) AS score_medio
        FROM gold.ads16_ad_object GROUP BY description
        ORDER BY imagens_com_o_objeto DESC LIMIT 8""",
    "A3: acerto por classe e os erros (gold.images_prediction)": """
        SELECT true_class AS classe, count(*) AS imagens, round(avg(correct::INT), 3) AS acerto,
               string_agg(CASE WHEN NOT correct THEN pred_class END, ', ') AS previsto_nos_erros
        FROM gold.images_prediction GROUP BY true_class ORDER BY acerto""",
    "A4.1: recall da COVID por condição e semente, no teste real (gold.xray_prediction)": """
        SELECT condition AS condicao, seed AS semente,
               round(avg((pred_class = 'COVID')::INT), 3) AS recall_covid, count(*) AS raio_x_covid
        FROM gold.xray_prediction WHERE true_class = 'COVID' GROUP BY ALL ORDER BY ALL""",
}


def run(sql: str) -> pd.DataFrame:
    con = duckdb.connect(str(data_dir() / "catalog.duckdb"), read_only=True)
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sql", default="", help="uma consulta qualquer (só leitura)")
    args = ap.parse_args(argv)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 48)
    if args.sql:
        print(run(args.sql).to_string(index=False))
        return 0
    for title, sql in CONSULTAS.items():
        print(f"\n== {title}\n")
        print(run(sql).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
