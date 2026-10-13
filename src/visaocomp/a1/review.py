"""Conferência a olho da A1: página HTML com os 200 recortes da amostra e a conferência inicial,
para a Lara revisar e baixar o lara.csv."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import pandas as pd

FIRST_CHECK_COLUMNS = [
    "chip_id", "address", "parcel_id", "house_id", "label_pool", "label_solar", "reason_pool",
    "reason_solar", "inicial_pool", "inicial_pool_reason", "inicial_solar", "inicial_solar_reason",
    "observacao",
]  # fmt: skip
MARKS = ("sim", "não", "não dá")


def _label_text(value: object, reason: object) -> str:
    if pd.isna(value) or value == "":
        return f"vazio ({reason})" if isinstance(reason, str) and reason else "vazio"
    return {"1": "1", "0": "0", "1.0": "1", "0.0": "0"}.get(str(value), str(value))


class ReviewPage:
    """Monta revisao.html: imagem, id, endereço, rótulo automático e a conferência inicial."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)

    @staticmethod
    def validate(inicial: pd.DataFrame) -> None:
        """Confere colunas e valores da conferência inicial."""
        missing = [c for c in FIRST_CHECK_COLUMNS if c not in inicial.columns]
        if missing:
            raise ValueError(f"conferencia_inicial.csv sem as colunas {missing}")
        for col in ("inicial_pool", "inicial_solar"):
            bad = set(inicial[col]) - set(MARKS)
            if bad:
                raise ValueError(f"conferencia_inicial.csv: valores inválidos em {col}: {bad}")
        if inicial["chip_id"].duplicated().any():
            raise ValueError("conferencia_inicial.csv: chip_id repetido")

    def _img(self, rel: str) -> str:
        data = (self.data_dir / rel).read_bytes()
        return "data:image/png;base64," + base64.b64encode(data).decode("ascii")

    def render(self, sample: pd.DataFrame, inicial: pd.DataFrame) -> str:
        """HTML completo (imagens embutidas, sem dependência externa)."""
        self.validate(inicial)
        df = inicial.merge(sample[["chip_id", "path", "stratum"]], on="chip_id", how="left")
        cards = []
        for r in df.itertuples(index=False):
            opts = "".join(f"<option>{m}</option>" for m in ("", *MARKS))
            cards.append(
                f"""<figure class="card" data-chip="{html.escape(r.chip_id)}">
<img src="{self._img(r.path)}" alt="recorte {html.escape(r.chip_id)}" width="192" height="192">
<figcaption>
<div class="id">{html.escape(r.chip_id)} · lote {html.escape(str(r.parcel_id))}</div>
<div class="addr">{html.escape(str(r.address))}</div>
<table>
<tr><th></th><th>rótulo</th><th>conferência inicial</th><th>Lara</th></tr>
<tr><td>piscina</td><td>{html.escape(_label_text(r.label_pool, r.reason_pool))}</td>
<td title="{html.escape(str(r.inicial_pool_reason or ""))}">{html.escape(r.inicial_pool)}</td>
<td><select data-attr="pool">{opts}</select></td></tr>
<tr><td>solar</td><td>{html.escape(_label_text(r.label_solar, r.reason_solar))}</td>
<td title="{html.escape(str(r.inicial_solar_reason or ""))}">{html.escape(r.inicial_solar)}</td>
<td><select data-attr="solar">{opts}</select></td></tr>
</table>
<div class="why">{html.escape(self._why(r))}</div>
</figcaption></figure>"""
            )
        agree = self.agreement(inicial)
        return PAGE.format(
            n=len(df),
            summary=html.escape(json.dumps(agree, ensure_ascii=False)),
            cards="\n".join(cards),
        )

    @staticmethod
    def _why(r: object) -> str:
        parts = []
        for name, col in (("piscina", "inicial_pool_reason"), ("solar", "inicial_solar_reason")):
            v = getattr(r, col)
            if isinstance(v, str) and v:
                parts.append(f"{name}: {v}")
        obs = r.observacao
        if isinstance(obs, str) and obs:
            parts.append(f"obs.: {obs}")
        return " · ".join(parts)

    @staticmethod
    def agreement(inicial: pd.DataFrame) -> dict[str, dict[str, int]]:
        """Cruza o rótulo automático (1, 0, vazio) com a conferência inicial, por atributo."""
        out: dict[str, dict[str, int]] = {}
        for attr in ("pool", "solar"):
            lab = inicial[f"label_{attr}"].map(
                lambda v: "vazio" if pd.isna(v) or v == "" else str(int(float(v)))
            )
            pair = lab + " x " + inicial[f"inicial_{attr}"]
            out[attr] = {k: int(v) for k, v in pair.value_counts().sort_index().items()}
        return out

    def write(self, sample_csv: Path, first_csv: Path, out_html: Path) -> Path:
        """Lê a amostra e o conferencia_inicial.csv e grava revisao.html."""
        sample = pd.read_csv(sample_csv, dtype=str)
        inicial = pd.read_csv(first_csv, dtype=str, keep_default_na=False)
        inicial = inicial.replace({"": None}).fillna({"inicial_pool": "", "inicial_solar": ""})
        out_html.write_text(self.render(sample, inicial), encoding="utf-8")
        return out_html


PAGE = """<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Conferência A1</title>
<style>
:root {{ --bg:#fafafa; --fg:#1d1d1f; --muted:#666; --card:#fff; --line:#ddd; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#161617; --fg:#eee; --muted:#aaa; --card:#222; --line:#444; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.4 system-ui, sans-serif;
  margin:0; padding:16px; }}
h1 {{ font-size:20px; margin:0 0 8px; }}
p {{ max-width:70ch; color:var(--muted); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(220px, 1fr)); gap:12px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; margin:0;
  padding:8px; }}
.card img {{ width:100%; height:auto; image-rendering:pixelated; border-radius:4px; }}
.id {{ font-family:ui-monospace, monospace; font-size:11px; color:var(--muted); }}
.addr {{ font-weight:600; margin:2px 0 4px; }}
table {{ width:100%; border-collapse:collapse; font-size:12px; }}
th, td {{ text-align:left; padding:2px 4px; border-top:1px solid var(--line); }}
.why {{ font-size:11px; color:var(--muted); margin-top:4px; }}
button {{ font:inherit; padding:6px 12px; margin:8px 0 16px; }}
</style></head><body>
<h1>Conferência A1: {n} recortes do teste</h1>
<p>Cada recorte tem 96 x 96 pixels do NAIP 2022 (cerca de 58 m), centrado na casa. "rótulo" é o
automático (1 = há licença ou piscina no OpenStreetMap; 0 = nenhuma evidência; vazio = a foto não
permite responder, com o motivo).
"conferência inicial" é a marcação feita a olho (passe o mouse para ver o
motivo). Marque a sua coluna e clique em "Baixar lara.csv"; salve o arquivo em
notas/conferencia/.</p>
<p>Concordância rótulo x conferência inicial: <code>{summary}</code></p>
<button id="dl">Baixar lara.csv</button>
<div class="grid">
{cards}
</div>
<script>
document.getElementById('dl').addEventListener('click', () => {{
  const rows = [['chip_id','lara_pool','lara_solar']];
  document.querySelectorAll('.card').forEach(c => {{
    const v = a => c.querySelector('select[data-attr="' + a + '"]').value;
    rows.push([c.dataset.chip, v('pool'), v('solar')]);
  }});
  const csv = rows.map(r => r.map(x => '"' + String(x).replace(/"/g, '""') + '"').join(','))
    .join('\\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv], {{type: 'text/csv'}}));
  a.download = 'lara.csv';
  a.click();
}});
</script>
</body></html>
"""
