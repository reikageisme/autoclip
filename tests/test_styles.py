from __future__ import annotations

import pytest

from autoclip.styles import StyleForm, Styles, estimate_videos, form_to_spec, slugify, spec_to_form


def test_form_roundtrip_and_fixes_reversed_ranges():
    form = StyleForm(
        label="Kể chuyện",
        duration_min=40,
        duration_max=20,
        body_min=6,
        body_max=3,
        transitions=["zoom", "bogus"],
        colors=["#ff0000", "red"],
    )
    assert (form.duration_min, form.duration_max) == (20, 40)
    assert (form.body_min, form.body_max) == (3, 6)
    assert form.transitions == ["zoom"] and form.colors == ["#ff0000"]
    spec = form_to_spec(form, "my-ke-chuyen")
    assert spec.label == "Kể chuyện" and [s.role for s in spec.slots] == ["hook", "body", "outro"]
    back = spec_to_form(spec)
    assert back.model_dump() == form.model_dump()


def test_slugify_vietnamese():
    assert slugify("Giới thiệu dịch vụ Đặc biệt!") == "gioi-thieu-dich-vu-dac-biet"


def test_styles_save_update_delete(tmp_path):
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    (builtin / "base.json").write_text(
        form_to_spec(StyleForm(label="Base"), "base").model_dump_json(), encoding="utf-8"
    )
    st = Styles(tmp_path / "custom", builtin)
    f1 = st.save(StyleForm(label="Của tôi"))
    assert f1 == "my-cua-toi.json" and not st.is_builtin(f1)
    assert st.save(StyleForm(label="Của tôi")) == "my-cua-toi-2.json"
    assert st.save(StyleForm(label="Đổi tên", duration_min=12), f1) == f1  # cập nhật tại chỗ
    assert st.load(f1).label == "Đổi tên"
    copy = st.save(StyleForm(label="Base"), "base.json")  # kiểu có sẵn → bản sao
    assert copy.startswith("my-base") and st.is_builtin("base.json")
    assert not st.delete("base.json")
    assert st.delete(f1) and st.load(f1) is None
    assert st.path("../secrets.json") is None and st.path("x.txt") is None
    assert [f for f, *_ in st.list()] == ["base.json", "my-base.json", "my-cua-toi-2.json"]


@pytest.mark.parametrize(
    ("avail", "avg", "low", "high"), [(33, 3.0, 3, 5), (400, 3.0, 40, 70), (0, None, 0, 0)]
)
def test_estimate_videos(avail, avg, low, high):
    spec = form_to_spec(StyleForm(label="x", duration_min=15, body_min=4, body_max=6), "x")
    assert low <= estimate_videos(spec, avail, avg) <= high
