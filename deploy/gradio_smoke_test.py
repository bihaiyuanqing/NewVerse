"""Gradio 联调冒烟脚本：通过 HTTP API 调用本机 app_local 的 /synthesize。

验证链路：上传音频 → Gradio 队列 → YingMusic 模型改词合成 → 取回产物。
仅用于本机 CPU 联调，输出落在 output/gradio_smoke/。
"""

import json
import os
import time

import requests

BASE = "http://127.0.0.1:7860"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "third_party", "YingMusic-Singer-Plus"))
WAV = os.path.join(REPO, "examples", "hf_space", "lyric_edit", "SingEdit_ZH_01.wav")
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "gradio_smoke"))
os.makedirs(OUT_DIR, exist_ok=True)

# 1) 上传音频（ref 与 melody 同一段，对应 lyric edit 用法）
with open(WAV, "rb") as f:
    resp = requests.post(f"{BASE}/gradio_api/upload", files={"files": (os.path.basename(WAV), f, "audio/wav")})
resp.raise_for_status()
server_path = resp.json()[0]
print("[smoke] uploaded ->", server_path, flush=True)

# Gradio 6 起文件入参必须是 FileData 结构，裸路径字符串会被 pydantic 拒绝
file_data = {
    "path": server_path,
    "orig_name": os.path.basename(WAV),
    "meta": {"_type": "gradio.FileData"},
}

payload = {
    "data": [
        file_data,                                    # ref_audio
        file_data,                                    # melody_audio
        "天青色等烟雨|而我在等你|炊烟袅袅升起",      # ref_text
        "阳光中赏花香|花瓣在飘落|山间幽静致远",      # target_text
        False,                                        # separate_vocals_flag（示例已是纯人声）
        False,                                        # mix_accompaniment_flag
        0.5,                                          # sil_len_to_end
        0.5,                                          # t_shift
        16,                                           # nfe_step（CPU 联调降步提速）
        3.0,                                          # cfg_strength
        666,                                          # seed
    ]
}

# 2) 入队
t0 = time.time()
r = requests.post(f"{BASE}/gradio_api/call/synthesize", json=payload)
r.raise_for_status()
event_id = r.json()["event_id"]
print("[smoke] event_id:", event_id, flush=True)

# 3) 轮询 SSE 结果（Gradio 6 会先持续推 event: heartbeat / data: null，
#    只有 event: complete 后的 data 才是产物，event: error 后的 data 是错误详情）
result_url = None
pending_event = None
with requests.get(f"{BASE}/gradio_api/call/synthesize/{event_id}", stream=True, timeout=1800) as stream:
    for raw in stream.iter_lines():
        if not raw:
            continue
        line = raw.decode()
        if line.startswith("event:"):
            pending_event = line.split(":", 1)[1].strip()
            continue
        if not line.startswith("data:"):
            continue
        payload_text = line[5:].strip()
        if pending_event == "heartbeat":
            pending_event = None
            continue
        print(f"[smoke][sse] {pending_event}: {payload_text[:200]}", flush=True)
        if pending_event == "error":
            raise RuntimeError(f"合成返回 error: {payload_text}")
        if pending_event == "complete":
            data = json.loads(payload_text)
            item = data[0] if isinstance(data, list) else data
            # FileData 同时返回服务器本地 path 与可下载 url，下载必须用 url
            result_url = item.get("url") if isinstance(item, dict) else item
            break
        pending_event = None

if not result_url:
    raise RuntimeError("SSE 流结束但未拿到 complete 产物")

print(f"[smoke] result ref: {result_url}, 耗时 {time.time()-t0:.1f}s", flush=True)

# 4) 下载产物
out_wav = os.path.join(OUT_DIR, "singedit_zh01_rewrite_cpu.wav")
dl = requests.get(f"{BASE}{result_url}" if result_url.startswith("/") else result_url)
dl.raise_for_status()
with open(out_wav, "wb") as f:
    f.write(dl.content)
print(f"[smoke] saved {out_wav} ({len(dl.content)/1024:.0f} KB)", flush=True)
