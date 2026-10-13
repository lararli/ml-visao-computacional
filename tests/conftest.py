"""Ajudantes de teste: um servidor de download falso, sem rede."""

from __future__ import annotations

import io
import zipfile

import pytest
import requests

from visaocomp.config import SourceConfig


def make_zip(files: dict[str, bytes]) -> bytes:
    """Monta um .zip em memória."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


class FakeResponse:
    """Resposta HTTP em streaming; pode cortar a conexão no meio."""

    def __init__(self, body: bytes, cut_at: int | None = None, lie_length: bool = False) -> None:
        self.body = body
        self.cut_at = cut_at
        self.headers = {"Content-Length": str(len(body) + (10 if lie_length else 0))}

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        data = self.body if self.cut_at is None else self.body[: self.cut_at]
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]
        if self.cut_at is not None:
            raise requests.ConnectionError("conexão cortada")

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class FakeSession:
    """Sessão falsa: devolve o corpo do `body` e conta as chamadas."""

    def __init__(self, body: bytes | None = None, **resp_kwargs: object) -> None:
        self.body = body
        self.resp_kwargs = resp_kwargs
        self.calls: list[str] = []

    def get(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append(url)
        if self.body is None:
            raise AssertionError("não devia usar a rede")
        return FakeResponse(self.body, **self.resp_kwargs)  # type: ignore[arg-type]


@pytest.fixture
def zip_bytes() -> bytes:
    """Um .zip pequeno com 3 arquivos."""
    return make_zip({"a/um.txt": b"um", "a/dois.txt": b"dois", "tres.txt": b"tres"})


@pytest.fixture
def source_cfg() -> SourceConfig:
    """Uma fonte de teste, sem sha256 gravado."""
    return SourceConfig(
        name="fonte",
        activity="T",
        kind="kaggle",
        owner="dono",
        dataset="dados",
        version=1,
        sha256=None,
        dest="raw/fonte",
    )


def _png(path, color, size=(40, 30), mode="RGB") -> None:
    """Grava uma imagem de uma cor só (ou, para `bmp`, no formato do arquivo)."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color).save(path)


@pytest.fixture
def image_folder(tmp_path):
    """Pasta com 3 classes de cores bem diferentes, uma cópia aninhada `data/`,
    um .bmp corrompido e um arquivo duplicado. Tem 8 + 8 + 4 imagens boas e únicas."""
    root = tmp_path / "imgs"
    colors = {"azul": (0, 0, 255), "verde": (0, 255, 0), "rara": (255, 0, 0)}
    counts = {"azul": 8, "verde": 8, "rara": 4}
    for name, n in counts.items():
        for i in range(n):
            r, g, b = colors[name]
            shade = (min(r + i * 3, 255), min(g + i * 3, 255), min(b + i * 3, 255))
            _png(root / name / f"{name}_{i}.png", shade, mode="RGB")
    (root / "azul" / "quebrada.bmp").write_bytes(b"BM isto nao e um bmp valido")
    (root / "verde" / "copia.png").write_bytes((root / "verde" / "verde_0.png").read_bytes())
    _png(root / "data" / "azul" / "azul_0.png", (0, 0, 255))  # cópia aninhada: deve ser ignorada
    return root
