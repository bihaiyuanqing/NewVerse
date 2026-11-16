"""人声分离（文档第六节（二））。

- Demucs two-stems 思路：只取人声与伴奏两条
- 直接调用 demucs 的 Python API 并自行落盘，绕开 demucs CLI
  按「模型名/文件名」建两级子目录的解析问题
- 分离后做人声 RMS 检查，低于 -45dB 抛 SeparationFailed
- MPS 个别算子不兼容时自动回退 CPU 重试
"""

import os

import torch

from core.audio_io import rms_db, write_audio
from core.device import demucs_device
from core.errors import SeparationFailed, call_with_timeout
from core.schema import SR

# 人声能量下限（dBFS），硬约束
VOCAL_RMS_FLOOR_DB = -45.0
DEFAULT_TIMEOUT = 600


def _run_model(model, wav: torch.Tensor, device: torch.device) -> torch.Tensor:
    from demucs.apply import apply_model

    return apply_model(model, wav.unsqueeze(0), device=device, progress=False)[0]


def separate(
    src_path: str,
    work_dir: str,
    model_name: str = "htdemucs_ft",
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    """分离人声与伴奏。

    返回 {"vocal_path": ..., "accomp_path": ..., "rms_db": ...}。
    16G 显存场景按文档把 model_name 换成 "htdemucs"。
    """

    if not os.path.exists(src_path):
        raise SeparationFailed(f"音频文件不存在：{src_path}")
    os.makedirs(work_dir, exist_ok=True)

    from demucs.audio import AudioFile, convert_audio
    from demucs.pretrained import get_model

    try:
        model = get_model(model_name)
    except Exception as exc:  # noqa: BLE001
        raise SeparationFailed(f"分离模型 {model_name} 加载失败：{exc}") from exc

    device = demucs_device()
    model.to(device)
    model.eval()

    try:
        wav = (
            AudioFile(src_path)
            .read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)
        )
        wav = convert_audio(wav, model.samplerate, SR, model.audio_channels)
        wav = wav.to(device)
    except Exception as exc:  # noqa: BLE001
        raise SeparationFailed(f"音频读取失败：{exc}") from exc

    try:
        sources = call_with_timeout(
            _run_model,
            timeout=timeout,
            timeout_msg=f"人声分离超时（>{timeout}s），请换更短的素材或改用 CPU 后重试",
            model=model,
            wav=wav,
            device=device,
        )
    except Exception as exc:
        # MPS 算子兼容问题：回退 CPU 重试一次
        if device.type == "mps":
            model.cpu()
            wav = wav.cpu()
            sources = call_with_timeout(
                _run_model,
                timeout=timeout,
                timeout_msg=f"人声分离超时（>{timeout}s）",
                model=model,
                wav=wav,
                device=torch.device("cpu"),
            )
        else:
            raise

    source_names = model.sources
    vocal_idx = source_names.index("vocals")
    with torch.no_grad():
        vocal_st = sources[vocal_idx]  # (channels, samples)
        accomp_st = sum(s for i, s in enumerate(sources) if i != vocal_idx)

    vocal_np = vocal_st.detach().cpu().numpy().astype("float32")
    accomp_np = accomp_st.detach().cpu().numpy().astype("float32")

    # 人声轨后续给模型一律单声道；伴奏保留立体声供最终混音
    if vocal_np.shape[0] > 1:
        import librosa

        vocal_mono = librosa.to_mono(vocal_np).astype("float32")
    else:
        vocal_mono = vocal_np[0]

    vocal_rms = rms_db(vocal_mono)
    if vocal_rms < VOCAL_RMS_FLOOR_DB:
        raise SeparationFailed(
            f"人声轨能量过低（RMS={vocal_rms:.1f}dB），这首歌可能不适合改词，请更换素材"
        )

    vocal_path = os.path.join(work_dir, "vocals.wav")
    accomp_path = os.path.join(work_dir, "no_vocals.wav")
    write_audio(vocal_path, vocal_mono, SR)
    write_audio(accomp_path, accomp_np, SR)

    return {
        "vocal_path": vocal_path,
        "accomp_path": accomp_path,
        "rms_db": vocal_rms,
        "device": device.type,
    }
