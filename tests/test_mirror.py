"""Machine-local config + true-mirror backup."""
import importlib


def test_config_set_mirror(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))      # isolate ~/.config/neitz
    monkeypatch.delenv("EPHYSDATAIO_MIRROR", raising=False)
    from neitz.dataio import config as cfg
    importlib.reload(cfg)
    assert cfg.mirror_dir() is None
    cfg.set_mirror("/some/drive/ephysdataio")
    assert str(cfg.mirror_dir()) == "/some/drive/ephysdataio"
    assert cfg.config_path().exists()
    assert cfg.auto_mirror() is True               # default


def test_mirror_true_copy_and_delete(tmp_path):
    from neitz.dataio.mirror import mirror_store
    src = tmp_path / "store"
    dst = tmp_path / "mirror"
    (src / "2026-06-02" / "c01" / "raw").mkdir(parents=True)
    (src / "2026-06-02" / "c01" / "manifest.json").write_text("{}")
    (src / "2026-06-02" / "c01" / "raw" / "a.abf").write_bytes(b"AAA")

    mirror_store(src, dst)
    assert (dst / "2026-06-02" / "c01" / "raw" / "a.abf").read_bytes() == b"AAA"
    assert (dst / "2026-06-02" / "c01" / "manifest.json").exists()

    # change source: remove a file, add another -> true mirror reflects both
    (src / "2026-06-02" / "c01" / "raw" / "a.abf").unlink()
    (src / "2026-06-02" / "c01" / "raw" / "b.abf").write_bytes(b"BBB")
    mirror_store(src, dst)
    assert not (dst / "2026-06-02" / "c01" / "raw" / "a.abf").exists()   # deleted in mirror
    assert (dst / "2026-06-02" / "c01" / "raw" / "b.abf").read_bytes() == b"BBB"


def test_mirror_additive_keeps_extras(tmp_path):
    from neitz.dataio.mirror import mirror_store
    src = tmp_path / "s"; dst = tmp_path / "m"
    src.mkdir(); (src / "x.txt").write_text("x")
    dst.mkdir(); (dst / "old.txt").write_text("keep")
    mirror_store(src, dst, delete=False)
    assert (dst / "x.txt").exists() and (dst / "old.txt").exists()       # extra kept
