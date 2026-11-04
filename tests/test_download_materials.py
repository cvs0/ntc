"""ambientCG download: missing maps, a bad zip, and a finished manifest."""

import io
import zipfile
from pathlib import Path

import pytest

from bench import download_materials as dl


def _zip(names: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in names:
            zf.writestr(name, b"jpg")
    return buf.getvalue()


def _good() -> bytes:
    return _zip(["Foo_Color.jpg", "Foo_NormalGL.jpg", "Foo_Roughness.jpg"])


def test_ambient_names_the_material_and_the_missing_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dl, "_get", lambda url, dest: dest.write_bytes(_zip(["Foo_Color.jpg", "Foo_NormalGL.jpg"])))
    with pytest.raises(RuntimeError, match="Foo: missing map Roughness"):
        dl.ambient(tmp_path, "Foo", size="2K")


def test_ambient_skips_a_directory_that_already_has_a_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    folder = tmp_path / "Foo_2K"
    folder.mkdir()
    (folder / "manifest.json").write_text("{}")

    def boom(url: str, dest: Path) -> None:
        raise AssertionError("downloaded")

    monkeypatch.setattr(dl, "_get", boom)
    assert dl.ambient(tmp_path, "Foo", size="2K") == folder
    assert not (folder / "Foo.zip").exists()


def test_ambient_retries_a_zip_that_does_not_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls = {"n": 0}

    def fake_get(url: str, dest: Path) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            dest.write_bytes(b"not a zip" + b"x" * 2000)
        else:
            dest.write_bytes(_good())

    monkeypatch.setattr(dl, "_get", fake_get)
    folder = dl.ambient(tmp_path, "Foo", size="2K")
    assert calls["n"] == 2
    manifest = (folder / "manifest.json").read_text()
    assert "Foo_Roughness.jpg" in manifest
    assert "Foo_Color.jpg" in manifest
