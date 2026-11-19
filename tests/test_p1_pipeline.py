"""P1 契约测试：等字数改词、重唱时长对齐、轻混响混音。

不依赖真实 YingMusic 模型：重唱子进程以外的纯函数与混音链路用合成音频覆盖。
"""

import numpy as np
import pytest
import soundfile as sf

from core.errors import MixFailed, RewriteFailed, SingFailed
from core.mix import apply_light_reverb, render_mix
from core.rewrite import count_syllables, rewrite_line
from core.schema import SR, CharUnit, Phrase, SongDoc
from core.sing import fit_duration


# ---------- rewrite：等字数硬约束 ----------

def test_count_syllables_ignores_punctuation():
    assert count_syllables("干啥子嘛，动起来！") == 7
    assert count_syllables("  ") == 0


def test_mock_rewrite_keeps_exact_length():
    # 演示素材真实改词点：慌 -> 干，整句等长
    out = rewrite_line("慌啥子嘛", 4, mock=True)
    assert out == "干啥子嘛"
    assert count_syllables(out) == 4


def test_mock_rewrite_unchanged_when_no_match():
    out = rewrite_line("一二三四", 4, mock=True)
    assert out == "一二三四"  # 无匹配词时原样返回，字数仍然一致


def test_llm_wrong_length_retries_then_fails(monkeypatch):
    monkeypatch.setenv("NEWVERSE_LLM_BASE_URL", "http://llm.local/v1")
    monkeypatch.setenv("NEWVERSE_LLM_API_KEY", "k")
    monkeypatch.setenv("NEWVERSE_LLM_MODEL", "m")
    # 候选始终多一个字，禁止删补，必须失败
    monkeypatch.setattr("core.rewrite._call_llm", lambda *a, **k: "多了一个字哦")  # 6 字
    with pytest.raises(RewriteFailed, match="字数不达标"):
        rewrite_line("原句五个字", 5, max_retry=2)  # 原句 5 字


def test_llm_accepts_equal_length_candidate(monkeypatch):
    monkeypatch.setenv("NEWVERSE_LLM_BASE_URL", "http://llm.local/v1")
    monkeypatch.setenv("NEWVERSE_LLM_API_KEY", "k")
    monkeypatch.setenv("NEWVERSE_LLM_MODEL", "m")
    monkeypatch.setattr("core.rewrite._call_llm", lambda *a, **k: "新词四字")
    assert rewrite_line("原句四字", 4) == "新词四字"


# ---------- pipeline.regenerate_all：全曲改词与回写 ----------

def test_regenerate_all_rewrites_every_phrase(monkeypatch, tmp_path):
    from pipeline import regenerate_all

    phrases = [
        Phrase(idx=0, text="黄沙慢慢漫来", start=0.0, end=1.0,
               chars=[CharUnit(idx=i, ch=ch, start=i * 0.1, end=i * 0.1 + 0.1)
                      for i, ch in enumerate("黄沙慢慢漫来")]),
        Phrase(idx=1, text="风啊静静吹", start=1.0, end=2.0,
               chars=[CharUnit(idx=i, ch=ch, start=1.0 + i * 0.1, end=1.0 + i * 0.1 + 0.1)
                      for i, ch in enumerate("风啊静静吹")]),
    ]
    doc = SongDoc(src_path="x.mp3", vocal_path="v.wav", accomp_path="a.wav", phrases=phrases)

    def fake_sing_lines(doc, items, work_dir, **kwargs):
        return {p.idx: str(tmp_path / f"new_{p.idx}.wav") for p, _ in items}

    monkeypatch.setattr("pipeline.sing_lines", fake_sing_lines)
    result = regenerate_all(doc, "", str(tmp_path), mock=True)

    assert result["total"] == 2
    assert result["text_changed"] == 2  # 两句都命中词表
    assert all(p.edited and p.new_vocal_path and p.new_text for p in doc.phrases)
    assert doc.phrases[0].new_text == "干啥快快漫来"
    assert (tmp_path / "song_doc.json").exists()


# ---------- sing：时长对齐 ----------

def test_fit_duration_exact_length_passthrough():
    audio = np.ones(1000, dtype=np.float32)
    out = fit_duration(audio, SR, 1000)
    np.testing.assert_array_equal(out, audio)


def test_fit_duration_stretches_to_target():
    target = SR * 2  # 2 秒
    for n in (SR, SR * 3):  # 偏短 / 偏长各测一次
        out = fit_duration(np.zeros(n, dtype=np.float32), SR, target)
        assert out.shape[0] == target


def test_fit_duration_rejects_extreme_mismatch():
    with pytest.raises(SingFailed, match="时长偏差过大"):
        fit_duration(np.zeros(SR, dtype=np.float32), SR, SR * 10)


def test_aligned_vocal_matches_splice_region_rounding():
    """采样口径必须与 splice 一致：round(end*SR)-round(start*SR)。

    刻意构造使 round(duration*SR) 与分别 round 相差 1 采样的边界。
    """
    from core.audio_io import splice

    start, end = 4.6 / SR, 13.4 / SR
    target = round(end * SR) - round(start * SR)
    assert target != round((end - start) * SR)  # 确认该组数据确实能触发差异
    raw = np.zeros(16, dtype=np.float32)
    aligned = fit_duration(raw, SR, target)
    base = np.zeros(SR, dtype=np.float32)
    # 口径一致时 splice 不再抛「替换段长度不匹配」
    splice(base, aligned, SR, start, end)


def test_fit_duration_rejects_stereo():
    with pytest.raises(SingFailed):
        fit_duration(np.zeros((2, 100), dtype=np.float32), SR, 100)


# ---------- mix：轻混响与整曲合成 ----------

def _make_doc(tmp_path, edited: bool) -> SongDoc:
    sr = SR
    vocal = (np.random.default_rng(1).standard_normal(sr) * 0.1).astype(np.float32)
    accomp = np.stack([vocal * 0.5, vocal * 0.5], axis=0).astype(np.float32)
    vocal_path = str(tmp_path / "vocals.wav")
    accomp_path = str(tmp_path / "no_vocals.wav")
    sf.write(vocal_path, vocal, sr)
    sf.write(accomp_path, accomp.T, sr)

    chars = [CharUnit(idx=i, ch=ch, start=i * 0.1, end=i * 0.1 + 0.1)
             for i, ch in enumerate("测试歌词")]
    phrase = Phrase(idx=0, text="测试歌词", start=0.0, end=0.4, chars=chars)
    if edited:
        new_vocal = np.zeros(round(0.4 * sr), dtype=np.float32)
        new_path = str(tmp_path / "new.wav")
        sf.write(new_path, new_vocal, sr)
        phrase.edited = True
        phrase.new_text = "新词四字"
        phrase.new_vocal_path = new_path
    return SongDoc(src_path="x.mp3", vocal_path=vocal_path,
                   accomp_path=accomp_path, phrases=[phrase])


def test_reverb_preserves_length_and_bounds():
    audio = (np.random.default_rng(2).standard_normal(SR) * 0.2).astype(np.float32)
    out = apply_light_reverb(audio, SR)
    assert out.shape == audio.shape
    assert np.max(np.abs(out)) <= 1.0


def test_render_mix_stereo_output(tmp_path):
    doc = _make_doc(tmp_path, edited=True)
    out_path = str(tmp_path / "mixed.wav")
    render_mix(doc, out_path)
    data, sr = sf.read(out_path, always_2d=True)
    assert sr == SR
    assert data.shape[1] == 2  # 立体声
    assert np.max(np.abs(data)) <= 1.0


def test_render_mix_missing_new_vocal_raises(tmp_path):
    doc = _make_doc(tmp_path, edited=True)
    doc.phrases[0].new_vocal_path = str(tmp_path / "lost.wav")
    with pytest.raises(MixFailed):
        render_mix(doc, str(tmp_path / "x.wav"))


def test_render_without_edits_still_renders(tmp_path):
    doc = _make_doc(tmp_path, edited=False)
    out_path = str(tmp_path / "plain.wav")
    render_mix(doc, out_path)
    assert sf.info(out_path).samplerate == SR
