"""CLI `autoclip`: ingest / capcut-report / extract / spike / concat-tools."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

from autoclip.config import get_settings

app = typer.Typer(help="AutoClip — tự sinh video ngắn từ project CapCut mẫu.", no_args_is_help=True)


@app.callback()
def _main(verbose: bool = typer.Option(False, "-v", "--verbose", help="Log DEBUG")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("alembic").setLevel(logging.WARNING)


@app.command()
def ingest(
    folder: Path = typer.Argument(..., exists=True, file_okay=False, help="Thư mục clip"),
    whisper: bool = typer.Option(False, "--whisper/--no-whisper", help="Chép lời bằng Whisper"),
) -> None:
    """Ingest clip: probe, cắt cảnh, tag, phash, thumbnail → DB (bỏ qua clip đã có)."""
    from autoclip.db import SqlClipRepository, make_engine
    from autoclip.db.migrate import upgrade
    from autoclip.ingest import IngestOptions, ingest_folder

    settings = get_settings()
    engine = make_engine()
    upgrade(engine)
    options = IngestOptions.from_settings(settings)
    options.whisper_enabled = whisper
    report = ingest_folder(folder, SqlClipRepository(engine), options)
    typer.echo(
        f"Thấy {report.clips_seen} clip: thêm {report.clips_added}, bỏ qua {report.clips_skipped}, "
        f"{report.segments_added} segment (loại trùng {report.segments_deduped}), "
        f"{len(report.errors)} lỗi, {report.elapsed_sec:.1f}s"
    )
    for err in report.errors:
        typer.echo(f"  lỗi: {err.path}: {err.message}", err=True)
    if report.errors:
        raise typer.Exit(1)


@app.command("capcut-report")
def capcut_report(
    samples: Path = typer.Argument(Path("samples"), help="Thư mục chứa các draft CapCut"),
    output: Path = typer.Option(None, "-o", help="File markdown (mặc định data/capcut_report.md)"),
) -> None:
    """Báo cáo mọi draft CapCut: đọc được / mã hoá / hỏng, số track, segment, text, effect."""
    from autoclip.recipes.extract import write_report

    path = write_report(samples, output or get_settings().data_dir / "capcut_report.md")
    typer.echo(f"Đã ghi báo cáo: {path}")


@app.command()
def extract(
    draft_dir: Path = typer.Argument(None, help="Một thư mục draft CapCut"),
    output: Path = typer.Option(None, "-o", help="File recipe.json (mặc định recipes/<tên>.json)"),
    all_in: Path = typer.Option(None, "--all", help="Trích mọi draft đọc được trong thư mục này"),
) -> None:
    """Trích recipe.json từ draft CapCut."""
    from autoclip.recipes import extract as ex

    if all_in is not None:
        argv = ["--all", str(all_in)]
    elif draft_dir is not None:
        argv = [str(draft_dir)] + (["-o", str(output)] if output else [])
    else:
        raise typer.BadParameter("cần <draft_dir> hoặc --all <thư mục>")
    raise typer.Exit(ex.main(argv))


@app.command()
def spike(
    clips: list[Path] = typer.Argument(None, help="Clip để nối (mặc định data/testclips/*.mp4)"),
    title: str = typer.Option("AutoClip spike", help="Title ở đầu video"),
) -> None:
    """Dựng 1080x1920 từ các clip + title + captions qua Concat headless, export MP4."""
    from autoclip.spike import run_spike

    clips = clips or sorted((get_settings().data_dir / "testclips").glob("*.mp4"))[:3]
    if not clips:
        raise typer.BadParameter("không có clip: truyền đường dẫn hoặc tạo data/testclips/*.mp4")
    r = run_spike(clips, title=title)
    typer.echo(f"MP4:     {r.output}")
    typer.echo(f"Video:   {r.video_sec:.2f}s từ {len(clips)} clip")
    typer.echo(f"Dựng:    {r.build_sec:.2f}s")
    typer.echo(f"Render:  {r.render_sec:.2f}s  ({r.render_ratio:.2f} giây render / giây video)")


@app.command()
def render(
    recipe: Path = typer.Option(..., "--recipe", "-r", exists=True, help="File recipe.json"),
    seed: int = typer.Option(None, help="Seed (mặc định ngẫu nhiên); cùng seed → cùng plan"),
    count: int = typer.Option(1, help="Số video (seed, seed+1, ...)"),
    llm: bool = typer.Option(False, "--llm/--no-llm", help="Dùng LLM planner (LLM_MODEL)"),
) -> None:
    """Dựng video từ recipe + segment trong DB qua Concat headless (không ghi job/QA)."""
    import random

    from autoclip.db import SqlClipRepository, make_engine
    from autoclip.render import build_plan, load_recipe
    from autoclip.render import render as do_render

    spec = load_recipe(recipe)
    repo = SqlClipRepository(make_engine())
    base = seed if seed is not None else random.randrange(1_000_000)
    for n in range(count):
        plan = build_plan(spec, repo, base + n, use_llm=llm)
        r = do_render(spec, plan)
        typer.echo(
            f"[seed {plan.seed}] {r.output} | {len(plan.items)} cảnh, {r.video_sec:.1f}s video, "
            f"{r.commands} lệnh, render {r.render_sec:.1f}s ({r.render_sec / r.video_sec:.2f} s/s)"
        )


@app.command()
def batch(
    recipe: Path = typer.Option(..., "--recipe", "-r", exists=True, help="File recipe.json"),
    count: int = typer.Option(20, help="Số video cần sinh"),
    seed: int = typer.Option(None, help="Seed đầu (mặc định ngẫu nhiên)"),
    llm: bool = typer.Option(True, "--llm/--no-llm", help="LLM planner; tắt = theo tag"),
) -> None:
    """Sinh N video khác nhau: planner → render → QA, không lặp segment, ghi job vào DB."""
    from autoclip.batch import run_batch
    from autoclip.render import load_recipe

    spec = load_recipe(recipe)

    def show(item) -> None:
        if item.status == "done":
            p = item.plan
            headline = next((t.text for t in p.texts if t.kind != "captions"), "")
            typer.echo(
                f"  ✓ job {item.job_id} seed {item.seed} [{p.params.get('planner')}] "
                f"{item.video_sec:.1f}s QA {item.qa.score:.2f} | {headline}"
            )
            typer.echo(f"    {item.output}")
        else:
            typer.echo(f"  ✗ job {item.job_id} seed {item.seed}: {item.error}")

    r = run_batch(spec, count, seed=seed, use_llm=llm, progress=show)
    done = r.done
    seg_sets = [{i.segment_id for i in d.plan.items} for d in done]
    overlap = sum(len(a & b) for k, a in enumerate(seg_sets) for b in seg_sets[k + 1 :])
    total = sum(d.render_sec for d in done)
    video = sum(d.video_sec for d in done)
    typer.echo(
        f"Xong {len(done)}/{r.requested} video, {len(r.items) - len(done)} job lỗi, "
        f"segment trùng giữa các video: {overlap}, "
        f"render {total:.0f}s cho {video:.0f}s video"
    )
    if r.stopped_reason:
        typer.echo(f"Dừng sớm: {r.stopped_reason}", err=True)
    if len(done) < r.requested:
        raise typer.Exit(1)


@app.command()
def fetch(
    urls: list[str] = typer.Argument(..., help="Link video, hoặc trang Videos/Reels của page"),
    dest: Path = typer.Option(None, help="Thư mục gốc (mặc định data/source); con: <tên page>"),
    page: str = typer.Option(None, help="Tên thư mục con (link /reel/ không chứa tên page)"),
    limit: int = typer.Option(None, help="Tải tối đa N video mỗi trang danh sách"),
    cookies_from_browser: str = typer.Option(
        None, help="Lấy cookie từ trình duyệt (chrome/edge/firefox) khi video cần đăng nhập"
    ),
    cookies: Path = typer.Option(None, help="File cookies.txt (Netscape) thay cho trình duyệt"),
    sleep: float = typer.Option(3.0, help="Nghỉ giữa các video (giây)"),
    then_ingest: bool = typer.Option(True, "--ingest/--no-ingest", help="Ingest ngay sau khi tải"),
    whisper: bool = typer.Option(False, "--whisper/--no-whisper", help="Whisper khi ingest"),
) -> None:
    """Tải video nguồn của khách bằng yt-dlp (không tải trùng), rồi ingest.

    Mỗi đối số là một link video/reel, hoặc một file .txt chứa mỗi dòng một link.
    yt-dlp không đọc được danh sách video của cả page Facebook.
    """
    from autoclip.fetch import expand_urls
    from autoclip.fetch import fetch as do_fetch

    urls = expand_urls(urls)
    if not urls:
        raise typer.BadParameter("không có link nào")
    r = do_fetch(
        urls,
        dest or get_settings().data_dir / "source",
        limit=limit,
        cookies_from_browser=cookies_from_browser,
        cookies_file=cookies,
        sleep_sec=sleep,
        name=page,
    )
    typer.echo(f"Tải mới {len(r.downloaded)} video vào {r.dest} (đã có từ trước: {r.skipped})")
    for err in r.errors:
        typer.echo(f"  lỗi: {err}", err=True)
    if then_ingest and (r.downloaded or r.skipped):
        ingest(folder=r.dest, whisper=whisper)
    if r.errors and not r.downloaded:
        raise typer.Exit(1)


@app.command("concat-tools")
def concat_tools() -> None:
    """Chạy concat headless, lưu tools/list ra data/concat_tools.json."""
    from autoclip.concat_client import SyncConcatClient

    with SyncConcatClient() as cc:
        typer.echo(f"{len(cc.tools)} tool → {get_settings().concat_tools_path}")


def main() -> None:
    """Entry point: console Windows mặc định cp1252, phải đổi sang UTF-8 trước khi in help."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    app()


if __name__ == "__main__":
    main()
