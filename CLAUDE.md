# AutoClip — hệ thống tự tạo video ngắn từ project mẫu

## Bối cảnh
- Khách cung cấp: ~10 project CapCut mẫu, kho clip source, CapCut Pro.
- Mục tiêu: tự động sinh hàng loạt video ngắn (TikTok/Reels/Shorts) theo "kiểu" của project mẫu, ĐA DẠNG và thu hút người xem (tránh nội dung lặp/hàng loạt bị nền tảng bóp reach hoặc cắt kiếm tiền).
- Engine dựng/render: **ReiCut** — bản fork của Concat (https://github.com/jub0t/Concat), editor video mã nguồn mở viết bằng Rust, có MCP server và JSON-RPC API, chạy được headless (`concat --mcp headless`).
- Thời gian: ~4 tuần. Máy dev: Windows.

## Cấu trúc thư mục làm việc
```
D:\work\
├── reicut\     # fork Concat (AGPL-3.0) — SỬA ÍT NHẤT CÓ THỂ
└── autoclip\   # repo này (private) — toàn bộ logic nằm ở đây
```

## RANH GIỚI GIẤY PHÉP — BẮT BUỘC TUÂN THỦ
Concat là AGPL-3.0 kèm "Concat Plugin Exception". Code của autoclip chỉ được giữ kín nếu là *Independent Module*:
1. KHÔNG copy, vendor hay chép lại bất kỳ đoạn code nào của Concat vào repo này.
2. KHÔNG link (static/dynamic) tới các crate nội bộ của Concat. Chỉ giao tiếp qua: MCP (stdio), JSON-RPC socket, hoặc định dạng file/project có tài liệu.
3. Mọi thay đổi trong `reicut/` là AGPL — chỉ sửa ở đó khi thật cần (đổi tên/icon, panel UI gọi HTTP sang autoclip). Mỗi file mới trong reicut phải có header SPDX như CONTRIBUTING.md yêu cầu.
4. Không dùng tên/logo "Concat" cho sản phẩm; tên sản phẩm là ReiCut.
Nếu một yêu cầu buộc phải vi phạm các điều trên → DỪNG và hỏi tôi.

## Kiến trúc
```
autoclip/
├── autoclip/
│   ├── concat_client/  # client MCP/JSON-RPC tới concat headless (spawn tiến trình, gọi tool)
│   ├── db/             # SQLAlchemy models + Alembic migrations, SQLite (WAL mode)
│   ├── ingest/         # ffprobe, PySceneDetect, Whisper, vision tagging, phash dedupe, thumbnail
│   ├── recipes/        # parser CapCut draft JSON -> recipe.json; compiler recipe -> lệnh Concat
│   ├── planner/        # LLM chọn clip cho slot, viết hook/caption/title
│   ├── diversity/      # random biến thể theo seed, chống lặp segment
│   ├── qa/             # preview_frame -> chấm điểm, loại bản lỗi
│   ├── worker/         # hàng đợi job (bảng DB), pool tiến trình concat, retry/timeout
│   ├── api/            # FastAPI (localhost:8787) cho panel ReiCut + dashboard
│   └── cli.py          # Typer CLI: ingest / recipe / render / batch / serve
├── recipes/            # recipe.json đã trích xuất (commit vào git)
├── samples/            # project CapCut mẫu (KHÔNG commit, .gitignore)
├── data/               # db, cache, output (KHÔNG commit)
├── tests/
└── pyproject.toml
```

### Lưu ý kỹ thuật về Concat
- Mỗi tiến trình Concat KHOÁ thư mục project đang mở (`concat.lock`). → Mỗi job = 1 thư mục project riêng; không mở cùng project ở 2 tiến trình.
- Headless chỉ ghi file dưới thư mục home; đặt `CONCAT_MCP_ROOTS` để cho phép thư mục khác (Windows ngăn cách bằng `;`).
- Edit chỉ nằm trong RAM cho tới `project_save`. Thời gian tính bằng giây; vị trí keyframe là tỉ lệ 0–1 của clip; kích thước/offset là tỉ lệ khung hình.
- Tool MCP chính: project_create/open/save/close, project_set_video, media_import, media_probe, clips_add_sequence, clip_add, clip_trim, clips_split, clip_transform, clip_speed, title_add, captions_add, keyframe_set, effects_list, effect_layer_add, template_list, template_instantiate, edit_apply (batch), export_start, export_wait, preview_frame, timeline_read.
- Luôn gọi `tools/list` để lấy schema thật thay vì đoán tham số. Đọc `reicut/docs/MCP.md` và `reicut/src/README.md` khi cần.
- Đường dẫn binary: đặt qua biến môi trường `CONCAT_BIN` (mặc định `D:\work\reicut\src\target\quick\concat.exe`).

### Lưu ý về project CapCut
- Draft nằm ở `%LOCALAPPDATA%\CapCut\User Data\Projects\com.lveditor.draft\<tên>\`; file chính `draft_content.json` (hoặc `draft_info.json`).
- File có thể bị mã hoá ở bản mới. Parser phải phát hiện: không bắt đầu bằng `{` → báo "encrypted", không crash.
- Thời gian trong draft tính bằng micro giây; text style là JSON lồng trong chuỗi.
- Effect/sticker/font độc quyền của CapCut không có bản tương đương → map gần đúng sang effect của Concat, ghi lại phần không map được.

## Schema DB (khởi điểm)
- `clips`: id, path, sha1, duration, width, height, fps, has_audio, created_at
- `segments`: id, clip_id, start, end, transcript, tags (JSON), hook_score, phash, embedding (blob, nullable)
- `recipes`: id, name, source_project, aspect, duration_target, spec (JSON)
- `recipe_variants`: id, recipe_id, params (JSON)
- `jobs`: id, recipe_id, variant_id, seed, status (queued/running/qa/done/failed), error, attempts, created_at, updated_at
- `renders`: id, job_id, output_path, qa_score, qa_notes, duration, size_bytes
- `usage_log`: segment_id, render_id, used_at  (chống lặp)
- `metrics`: render_id, platform, views, avg_watch, likes, captured_at (nhập tay/CSV)

## Định dạng recipe.json (đề xuất, được phép cải tiến)
```json
{
  "name": "hook-3cut-caption",
  "aspect": "9:16",
  "duration_target": [20, 35],
  "music": {"mode": "library", "beat_sync": true},
  "slots": [
    {"role": "hook", "dur": [1.5, 2.5], "want_tags": ["action", "face"], "speed": [1.0, 1.2]},
    {"role": "body", "repeat": [3, 6], "dur": [1.5, 3.0], "transition": ["cut", "zoom", "whip"]},
    {"role": "outro", "dur": [1.5, 2.5]}
  ],
  "texts": [
    {"slot": "hook", "kind": "headline", "style": "bold-stroke", "pos": [0.5, 0.2]},
    {"kind": "captions", "source": "whisper", "style": "word-pop"}
  ],
  "variation": {"palette": ["#FFD400", "#FFFFFF", "#00E5FF"], "fonts": ["..."], "shuffle_body": true}
}
```

## Các giai đoạn & tiêu chí hoàn thành
Làm TUẦN TỰ, mỗi giai đoạn xong thì dừng lại báo cáo và chờ tôi duyệt.

**P0 — Spike (2–3 ngày)**
- `scripts/spike.py`: spawn concat headless qua MCP → tạo project 1080x1920 → import 3 clip → nối → thêm title + captions → export MP4 → in thời gian render.
- `scripts/check_capcut.py <draft_dir>`: báo cáo draft có mã hoá không; nếu không, in tóm tắt số track/segment/text/effect.
- ✅ Done khi: có MP4 xuất ra và bảng thời gian render; báo cáo 10 project mẫu.

**P1 — DB + ingest (tuần 1)**
- Models + migration; `autoclip ingest <folder>`: probe, scene detect, Whisper, tag, phash, thumbnail. Idempotent (sha1).
- ✅ Done khi: ingest 50 clip không lỗi, chạy lại không tạo trùng, có test.

**P2 — Recipe (tuần 1–2)**
- Parser CapCut → recipe.json; compiler recipe + danh sách segment → batch lệnh Concat (`edit_apply`).
- ✅ Done khi: dựng lại 1 project mẫu, tôi đánh giá đạt ≥80% "cảm giác" bản gốc.

**P3 — Planner + đa dạng + QA (tuần 2–3)**
- LLM qua API (biến môi trường `OPENAI_API_KEY` hoặc `ANTHROPIC_API_KEY`, chọn provider qua config; KHÔNG hardcode key). Fallback không-LLM (chọn theo tag + template text) khi không có key.
- Seed tái lập được; chống lặp segment trong N render gần nhất (config).
- QA: lấy 3–5 frame qua preview_frame, kiểm tra cơ bản (khung đen, chữ tràn khỏi khung, độ dài) + chấm điểm vision tuỳ chọn.
- ✅ Done khi: `autoclip batch --recipe X --count 20` ra 20 video khác nhau rõ rệt, không segment nào lặp trong batch.

**P4 — Worker (tuần 3)**
- Hàng đợi job trên DB, N worker song song (config), timeout, retry, log có cấu trúc, dọn thư mục project tạm.
- ✅ Done khi: chạy 100 job qua đêm không treo, job lỗi được ghi lý do.

**P5 — API + panel ReiCut + bàn giao (tuần 4)**
- FastAPI: tạo batch, xem hàng đợi, xem/tải render. Dashboard đơn giản (HTMX).
- Trong `reicut/`: panel "AutoClip" (Slint) CHỈ gọi HTTP localhost:8787 — không đưa logic vào đây.
- README cài đặt + vận hành.

## Quy ước code
- Python 3.11+, quản lý bằng `uv` (hoặc venv + pip nếu uv không có). Type hints, ruff, pytest.
- Config qua `.env` + pydantic-settings. Không commit secret, không commit file media.
- Log bằng `logging` có cấu trúc; mọi lệnh gửi tới Concat được log ở mức DEBUG.
- Viết test cho parser, compiler, diversity (dùng fixture JSON nhỏ, không dùng video thật trong unit test).
- Mỗi thay đổi lớn: commit nhỏ, message rõ ràng.

## Khi không chắc
Đọc tài liệu trong `reicut/docs/` và gọi `tools/list` trước khi đoán. Nếu quyết định ảnh hưởng tới giấy phép, chi phí API, hoặc định dạng dữ liệu lâu dài → hỏi tôi.

## Làm việc theo nhóm (1 lead + 2 subagent)
Phiên Claude Code chính là **lead**. Hai subagent định nghĩa trong `.claude/agents/`:

| Ai | Sở hữu (chỉ người này được sửa) |
|---|---|
| **lead** | `pyproject.toml`, `autoclip/contracts.py`, `autoclip/config.py`, `autoclip/db/**`, `autoclip/concat_client/**`, `autoclip/planner/**`, `autoclip/diversity/**`, `autoclip/worker/**`, `autoclip/api/**`, `autoclip/cli.py`, `scripts/**`, `tests/` ngoài 2 mục dưới, `CLAUDE.md` |
| **media-ingest** | `autoclip/ingest/**`, `autoclip/qa/**`, `tests/ingest/**`, `tests/qa/**`, `tests/fixtures/media/**` |
| **recipe-builder** | `autoclip/recipes/**`, `recipes/**`, `tests/recipes/**`, `tests/fixtures/capcut/**` |

Quy tắc:
- `autoclip/contracts.py` là hợp đồng chung (dataclass/pydantic + Protocol). Chỉ lead sửa. Subagent cần đổi → ghi vào "Yêu cầu gửi lead".
- Lead viết contracts + skeleton TRƯỚC, rồi mới giao việc song song.
- Không ai sửa file ngoài phạm vi của mình. Lead tích hợp và chạy toàn bộ test sau mỗi đợt.
