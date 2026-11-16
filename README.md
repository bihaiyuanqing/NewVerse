# NewVerse · 音乐改词 Demo

按《NewVerse · 音乐改词 Demo 技术实现文档 v1.0》实现。链路：
Demucs 人声分离 → WhisperX 字级对齐 + torchcrepe 音高 → LLM 等字数改词 →
YingMusic-Singer-Plus 单句重唱 → 50ms 交叉淡化替换 → 轻混响混音。

## 设备策略

自动探测，优先级 `NEWVERSE_DEVICE` 环境变量 > CUDA > MPS > CPU：

| 组件 | NVIDIA GPU 机 | 本机（Apple Silicon） |
|---|---|---|
| Demucs | cuda（16G 显存换 htdemucs） | mps，算子不兼容自动回退 cpu |
| WhisperX | cuda / float16 | cpu / int8（CTranslate2 不支持 MPS） |
| torchcrepe | cuda | mps，失败回退 cpu |
| YingMusic-Singer-Plus | cuda | cpu（仅联调，速度不达标） |

## 本机环境（macOS arm64）

```bash
# 需原生 arm64 Python（Rosetta 版 x86_64 Python 无法使用 MPS）
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
```

已跑通组合锁定在 `requirements-lock.txt`（torch 2.15.0 / demucs 4.1.0 /
whisperx 3.4.5 / torchcrepe 0.0.24，Python 3.11.12 arm64）。

## 目录结构

```
core/schema.py    统一中间表示 CharUnit/Phrase/SongDoc（先写，契约校验在内）
core/device.py    设备自动探测与组件能力映射
core/errors.py    中文业务异常 + 模型调用超时包装
core/audio_io.py  读写/切片/50ms 交叉淡化 splice/RMS
core/separate.py  Demucs 分离 + 人声能量检查
core/analyze.py   WhisperX 字级对齐 + 断句 + 整轨 F0
pipeline.py       对外门面 prepare/regenerate_line/render
cli.py            命令行入口
tests/            P0.4 契约测试
deploy/           GPU 离线部署脚本
```

## 阶段进展

### P0 · 链路打通（进行中）

| 任务 | 状态 | 验证方式 / 结果 |
|---|---|---|
| P0.1 环境与分离 | 代码完成，待素材实测 | `cli.py prepare` 产出 vocals.wav / no_vocals.wav，待人工试听 |
| P0.2 SongDoc 落盘 | 代码完成，待素材实测 | `validate_song_doc` 校验时间单调、不重叠、无零时长 |
| P0.3 打印每句字数 | 代码完成，待素材实测 | CLI 输出「第 N 句 \| X 字 \| 歌词」 |
| P0.4 契约测试 | **通过** | `pytest tests/test_contracts.py` → 13 passed |

实测回填区（拿到演示素材后填写，禁止编造）：

- 分离耗时 / 显存占用：待实测
- WhisperX 对齐准确率（抽查 3 句）：待实测
- torchcrepe 全曲 F0 耗时：待实测
- 3 分钟歌曲全流程耗时（验收线 ≤5 分钟，本机 CPU/MPS 预计超标，仅 GPU 机作数）

### P1 · 改词与单句重唱（未开始）

- rewrite：LLM 服务与密钥待定（先 Mock 后接入）
- sing：对接 YingMusic-Singer-Plus `infer.py`，实际入参为
  `--ref_audio/--melody_audio/--ref_text/--target_text`，输出 44.1kHz 立体声需 downmix
- 止损规则按文档第七节执行

### P2 · Gradio 界面（未开始）

## 已知限制（交付时需附）

不支持和声多的歌、不支持英文、单句 12 字以内、不承诺音色克隆。
YingMusic-Singer-Plus 的 VAE 权重为 Stability AI Community License，商用受限。
