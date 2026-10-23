from __future__ import annotations

from pathlib import Path

import pytest

from autoclip.fetch import ARCHIVE_NAME, build_options, expand_urls, fetch, source_name


def test_expand_urls_reads_txt_and_dedupes(tmp_path):
    f = tmp_path / "links.txt"
    f.write_text(
        "# video khách\nhttps://fb.com/a/videos/1\n\nhttps://fb.com/a/videos/2\n",
        encoding="utf-8-sig",
    )
    assert expand_urls([str(f), "https://fb.com/a/videos/1", "https://fb.com/a/videos/3"]) == [
        "https://fb.com/a/videos/1",
        "https://fb.com/a/videos/2",
        "https://fb.com/a/videos/3",
    ]


@pytest.mark.parametrize(
    ("url", "name"),
    [
        ("https://www.facebook.com/GiangMinhVN", "GiangMinhVN"),
        ("https://www.facebook.com/GiangMinhVN/videos", "GiangMinhVN"),
        ("https://m.facebook.com/GiangMinhVN/reels/?x=1", "GiangMinhVN"),
        ("https://www.facebook.com/watch/?v=123", "www.facebook.com"),
        ("https://example.com/a.mp4", "example.com"),
    ],
)
def test_source_name(url, name):
    assert source_name(url) == name


def test_build_options(tmp_path):
    opts = build_options(tmp_path, limit=5, cookies_from_browser="edge", sleep_sec=2)
    assert opts["download_archive"] == str(tmp_path / ARCHIVE_NAME)
    assert opts["merge_output_format"] == "mp4"
    assert opts["writeinfojson"] is True
    assert opts["playlistend"] == 5
    assert opts["cookiesfrombrowser"] == ("edge",)
    assert opts["sleep_interval"] == 2
    assert "cookiefile" not in opts


class FakeYDL:
    """Giả lập yt-dlp: mỗi link 'tải' một id, ghi archive + file mp4 như yt-dlp thật."""

    def __init__(self, opts):
        self.opts = opts
        self.dest = Path(opts["outtmpl"]).parent
        self.archive = Path(opts["download_archive"])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def download(self, urls):
        for url in urls:
            vid = url.rsplit("=", 1)[-1]
            lines = self.archive.read_text().splitlines() if self.archive.exists() else []
            if f"facebook {vid}" in lines:
                continue
            (self.dest / f"{vid}.mp4").write_bytes(b"x")
            self.archive.write_text("\n".join([*lines, f"facebook {vid}"]) + "\n")
        return 0


def test_fetch_downloads_then_skips(tmp_path):
    urls = ["https://www.facebook.com/GiangMinhVN/videos?v=111"]
    first = fetch(urls, tmp_path, ydl_factory=FakeYDL)
    assert first.dest == tmp_path / "GiangMinhVN"
    assert [p.name for p in first.downloaded] == ["111.mp4"]
    assert first.skipped == 0 and not first.errors

    second = fetch(urls, tmp_path, ydl_factory=FakeYDL)
    assert second.downloaded == []
    assert second.skipped == 1


def test_fetch_records_errors(tmp_path):
    class Boom(FakeYDL):
        def download(self, urls):
            raise RuntimeError("blocked")

    r = fetch(["https://www.facebook.com/GiangMinhVN"], tmp_path, ydl_factory=Boom)
    assert r.downloaded == []
    assert r.errors and "blocked" in r.errors[0]
