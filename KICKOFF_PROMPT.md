# Prompt tổng cho Claude Code (lead + 2 subagent)

Dán toàn bộ khối dưới đây vào Claude Code, mở tại `D:\work\autoclip` bằng:
`claude --add-dir D:\work\reicut`

---

```
Bạn là LEAD của dự án AutoClip. Đọc kỹ CLAUDE.md (đặc biệt mục "RANH GIỚI GIẤY PHÉP" và "Làm việc theo nhóm"), rồi đọc D:\work\reicut\docs\MCP.md và D:\work\reicut\src\README.md.

Bạn có 2 subagent trong .claude/agents/: `media-ingest` và `recipe-builder`. Mỗi agent chỉ sửa file trong phạm vi của mình. Bạn giữ contracts, DB, client Concat, tích hợp và mọi quyết định.

Mục tiêu đợt này: hoàn thành P0 và P1, và phần trích xuất của P2. Làm theo đúng 4 bước dưới, không nhảy bước.

== BƯỚC 1 — Lead làm một mình (nền móng) ==
1. Khởi tạo project bằng uv: pyproject.toml (python>=3.11; deps: mcp, pydantic, pydantic-settings, sqlalchemy, alembic, typer, fastapi, uvicorn, scenedetect[opencv], faster-whisper, imagehash, pillow, numpy; dev: pytest, ruff), .gitignore (data/, samples/, .env, *.mp4, *.mov), .env.example (CONCAT_BIN, CONCAT_MCP_ROOTS, OPENAI_API_KEY, ANTHROPIC_API_KEY, DATA_DIR).
2. autoclip/config.py: Settings bằng pydantic-settings.
3. autoclip/contracts.py — hợp đồng chung, CHỈ chứa kiểu dữ liệu và Protocol, không logic:
   - ClipInfo, SegmentInfo (start, end, transcript, tags, hook_score, phash, thumb_path), IngestReport
   - ClipRepository (Protocol): get_clip_by_sha1, add_clip, add_segments, list_segments(filter)
   - DraftStatus (enum ok/encrypted/corrupt/missing)
   - RecipeSpec (khớp định dạng recipe.json trong CLAUDE.md + trường `unmapped`)
   - RenderPlan (recipe name, seed, danh sách (slot_role, segment_id, clip_path, src_start, src_end, speed), texts đã điền nội dung)
   - ConcatCommand (tool: str, args: dict)
   - QAResult (score, passed, notes)
4. autoclip/db/: models SQLAlchemy theo schema trong CLAUDE.md, SQLite WAL, Alembic migration đầu tiên, và một SqlClipRepository hiện thực ClipRepository.
5. autoclip/concat_client/: spawn `CONCAT_BIN --mcp headless` qua stdio bằng thư viện mcp; hàm list_tools(), call_tool(name, args), preview_frame(t); lúc khởi động gọi tools/list và lưu schema ra data/concat_tools.json. Log mọi lệnh ở mức DEBUG.
6. Tạo skeleton rỗng (chỉ __init__.py + docstring) cho autoclip/ingest, autoclip/qa, autoclip/recipes và các thư mục tests tương ứng.
7. Chạy `uv sync`, `uv run ruff check`, `uv run pytest` (được phép 0 test). Commit "chore: scaffold + contracts".

== BƯỚC 2 — Giao việc SONG SONG cho 2 subagent ==
Gọi CẢ HAI subagent trong cùng một lượt để chúng chạy song song:

→ media-ingest: "Đọc CLAUDE.md và autoclip/contracts.py. Hiện thực toàn bộ nhiệm vụ trong định nghĩa agent của bạn (ingest + qa). Chỉ dùng ClipRepository từ contracts để ghi DB. Xong thì báo cáo đủ 4 mục."

→ recipe-builder: "Đọc CLAUDE.md, autoclip/contracts.py và data/concat_tools.json. Hiện thực capcut_reader, extract, report và compiler như định nghĩa agent của bạn. Chạy report trên samples/ để sinh data/capcut_report.md, và extract recipe cho mọi draft đọc được vào recipes/. Xong thì báo cáo đủ 4 mục."

Trong lúc chờ, lead làm phần của mình:
   - scripts/spike.py: tạo project 1080x1920 → import 3 clip (argv) → nối → title + captions → export MP4 vào data/out/ → in thời gian render. Chạy thử với clip trong samples/ hoặc clip test tự sinh bằng ffmpeg.
   - autoclip/cli.py (Typer): lệnh `ingest <folder>`, `capcut-report`, `extract <draft_dir>`; gọi code của 2 agent qua import.

== BƯỚC 3 — Lead tích hợp ==
1. Đọc báo cáo của 2 agent. Xử lý "Yêu cầu gửi lead": nếu hợp lý thì cập nhật contracts/deps/DB rồi báo agent sửa theo (gọi lại agent, không tự sửa code trong phạm vi của agent, trừ lỗi import nhỏ).
2. Chạy toàn bộ: `uv run ruff check` và `uv run pytest`. Lỗi thuộc phạm vi agent nào → giao lại cho agent đó kèm log lỗi.
3. Chạy thật: `autoclip ingest <thư mục clip mẫu>` hai lần (lần 2 phải không tạo bản ghi trùng); `autoclip capcut-report`; `python scripts/spike.py ...`.
4. Thử nối end-to-end bản thô: lấy 1 recipe trong recipes/, chọn segment ngẫu nhiên từ DB thành RenderPlan, chạy compiler → gửi lệnh qua concat_client → export. Chưa cần đẹp, chỉ cần ra MP4.

== BƯỚC 4 — DỪNG và báo cáo cho tôi ==
Báo cáo ngắn gọn gồm:
- Những gì đã xong theo từng giai đoạn P0/P1/P2, ai làm.
- Thời gian render của spike và của bản end-to-end (giây / giây video).
- Tóm tắt data/capcut_report.md: bao nhiêu draft đọc được / bị mã hoá; danh sách recipe đã trích xuất; các effect không map được.
- Kết quả lint + test.
- Rủi ro / câu hỏi cần tôi quyết định.
Sau đó CHỜ tôi duyệt, không tự chuyển sang P3.

Nguyên tắc chung:
- Trước khi viết code ở bước 1, nêu kế hoạch ngắn và các giả định của bạn.
- Không đoán tham số Concat: đọc data/concat_tools.json.
- Không copy code từ Concat, pyJianYingDraft, VectCutAPI.
- Gặp việc ảnh hưởng tới giấy phép, chi phí API, hoặc định dạng dữ liệu lâu dài → hỏi tôi trước.
```

---

## Prompt cho các đợt sau (ngắn)

**Đợt 2 — P2 hoàn thiện + P3:**
```
Đợt 1 đã duyệt. Đợt 2: (1) recipe-builder hoàn thiện compiler + effect_map để dựng lại project mẫu "<tên>" sát nhất có thể; (2) media-ingest bổ sung QA nối với ConcatClient.preview_frame; (3) lead làm planner (LLM + fallback không-LLM) và diversity (seed, chống lặp segment), rồi lệnh `autoclip batch --recipe X --count 20`. Giao 2 agent song song như đợt 1, tích hợp, chạy batch 20 video, rồi DỪNG báo cáo.
```

**Đợt 3 — P4 + P5:**
```
Đợt 2 đã duyệt. Đợt 3: lead làm worker (hàng đợi DB, N tiến trình concat, timeout/retry, dọn thư mục tạm) và FastAPI + dashboard HTMX. media-ingest viết test tải cho QA trên 100 render. recipe-builder trích xuất và kiểm tra tất cả recipe còn lại. Cuối đợt: chạy 100 job, viết README cài đặt/vận hành, DỪNG báo cáo.
```

**Panel trong ReiCut (làm riêng, cuối dự án):**
```
Làm việc trong D:\work\reicut. Thêm panel "AutoClip" vào UI Slint: chọn recipe, số lượng, nút Tạo batch, danh sách job, xem trước render. Panel CHỈ gọi HTTP tới http://127.0.0.1:8787 (API của autoclip), không chứa logic sinh video. Mọi file mới có header SPDX AGPL như CONTRIBUTING.md. Sửa ít file nhất có thể để dễ rebase upstream. Chạy cargo fmt --check, cargo clippy, cargo test trước khi báo xong.
```
