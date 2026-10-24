from __future__ import annotations

from autoclip.planner.vi_text import headline_candidates, is_vietnamese_word, vi_ratio


def test_vietnamese_syllables_pass() -> None:
    for w in ["nghiêng", "khuya", "quá", "gì", "Đẹp", "trước,", "hàng", "oanh", "ừ", "20s", "50k"]:
        assert is_vietnamese_word(w), w


def test_whisper_guesses_fail() -> None:
    for w in ["Chrogelo", "And", "have", "Mix", "shop"]:
        assert not is_vietnamese_word(w), w


def test_vi_ratio_on_real_render_text() -> None:
    assert vi_ratio("Cái này rất đẹp Cái này") == 1.0
    assert vi_ratio("And I have to") < 0.8
    assert vi_ratio("nhiều Chrogelo") < 0.8
    assert vi_ratio("") == 0.0


def test_headline_needs_whole_clean_clause() -> None:
    texts = [
        "hàn rất là nhiều mọi",  # cắt giữa câu (không viết hoa đầu) → bỏ
        "nhiều Chrogelo. Hàng về rất nhiều mẫu mới.",
        "And I have to go.",
        "Cái này cái này cái này.",  # lặp từ → bỏ
    ]
    assert headline_candidates(texts) == ["Hàng về rất nhiều mẫu mới"]


def test_headline_rejects_clause_cut_mid_sentence() -> None:
    # cuối đoạn cảnh, không dấu câu / kết thúc bằng hư từ → câu dở
    assert headline_candidates(["Hàn rất là nhiều mọi người sẽ"]) == []
    assert headline_candidates(["Mạng này kìa rất, đẹp lắm."]) == []
    assert headline_candidates(["Công nhân cũng đón rất là vất vả,"]) == [
        "Công nhân cũng đón rất là vất vả"
    ]
