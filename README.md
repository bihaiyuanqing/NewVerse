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

已跑通组合锁定在 `requirements-lock.txt`（torch 2.6.0 / torchaudio 2.6.0 /
demucs 4.1.0 / whisperx 3.4.5 / torchcrepe 0.0.24，Python 3.11.12 arm64）。
注意 torch 2.15 会导致 torchaudio.AudioMetaData 缺失，且其默认 weights_only=True
会破坏 pyannote VAD 加载，故锁定 2.6.0（与 YingMusic 生态对齐）。

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
deploy/           GPU 机部署三步（01 Mac 备料传输 / 02 驱动 / 03 Python 环境）
```

## GPU 机实测（Debian 10 / 8×L20 46G / 内网隔离）

驱动与环境按 `deploy/` 三步脚本搭建（内网镜像无 .run，采用 focal R535 deb
解包编译；Python 用 standalone 3.10；依赖走 bytedpypi；模型缓存从本机 rsync）。
同一首 3 分 06 秒素材、单卡 CUDA_VISIBLE_DEVICES=0：

| 指标 | 本机 M2 Pro（MPS+CPU） | GPU 机单卡 L20（cuda/float16） |
|---|---|---|
| 分离耗时 | 56-62s | 17.6s |
| 解析耗时 | 83-87s | 15.2s |
| 全流程 prepare | 140-149s | **约 33s（提速 4.4 倍）** |
| 峰值显存 | 149MB（MPS allocated） | 2601MB（46G 卡占用 5.6%） |
| 产物 | 30 句，契约通过 | 31 句，契约通过，均无跨间奏句 |

两套设备产出句数差 1 属 float16/int8 数值差异导致的边界抖动，契约指标
（句长 ≤12、时间单调不重叠、无零时长、间奏隔离）两端全部通过。

## 阶段进展

### P0 · 链路打通（已完成，本机 + GPU 双端实测通过）

| 任务 | 状态 | 验证方式 / 结果 |
|---|---|---|
| P0.1 环境与分离 | **实测通过** | 产出 vocals.wav（单声道）/ no_vocals.wav（立体声），人声 RMS -14.9 dBFS |
| P0.2 SongDoc 落盘 | **实测通过** | `validate_song_doc` 校验时间单调、不重叠、无零时长，30 句全部通过 |
| P0.3 打印每句字数 | **实测通过** | CLI 输出「第 N 句 \| X 字 \| 歌词」，句长 4-12 字，抽查见下表 |
| P0.4 契约测试 | **通过** | `pytest tests/` → 16 passed（含间奏伪时间戳钳制、跨静音碎片合并用例） |

实测回填区（素材 `assets/demo/huangshazima.mp3`，3 分 06 秒，44.1kHz 立体声，
设备 MacBook Pro M2 Pro / MPS，WhisperX 走 CPU int8）：

- 分离耗时约 56-62s（Demucs htdemucs_ft，MPS），产出人声 RMS -14.9 dBFS（健康）
- 解析耗时约 83-87s（WhisperX small CPU 对齐 + torchcrepe full MPS 分块提 F0）
- 全流程 prepare 约 140-149s（验收线 ≤5 分钟，本机 MPS+CPU 已达标；GPU 机待驱动）
- MPS 峰值 allocated 149MB（torch 当前已分配快照，非进程驻留峰值）
- F0 清音帧占比约 28%，和声跳变占比 0.10%（无和声风险）
- 断句实测：能量门限 -28dB + 最小静音 0.15s，单字时长钳制上限 2.0s，
  碎片合并间隙上限 1.0s；30 句全部满足单句 ≤12 字，契约校验通过
- WhisperX 识别人工抽查：见下方抽查记录

WhisperX small 识别抽查（注意 `huangshazima.lrc` 是改词目标稿《干啥子嘛》，
与音频原曲《慌啥子嘛》文本不同，不能作为 ASR 基准；以下按原曲可辨识歌词人工
评估。P0.3 只约束字数与断句，文本误差属 small 模型精度问题，留待 P1 换 large
模型或人工校对）：

| 句 | 识别结果 | 评估 |
|---|---|---|
| 第 15 句 | 一切都是最好的安排（9 字，2.36s） | 整句完整切出，文本正确 |
| 第 6 句 | 这一刻就忘了你是谁（9 字，4.64s） | 整句完整切出，文本正确 |
| 第 18 句 | 风啊静静吹（5 字，2.74s） | 断句正确，「静/轻」近音字识别误差 |
| 间奏 91.2-107.9s | 两侧句子独立成句，无跨间奏粘合 | 16.6s 间奏隔离正确 |

已知 small 模型典型误差：近音字（静/轻、整/枕）、连读字（街拍/节拍）。
断句与字数契约不受影响，P1 改词前需人工校对或升级 large 模型。

### P1 · 改词与单句重唱（未开始）

- rewrite：LLM 服务与密钥待定（先 Mock 后接入）
- sing：对接 YingMusic-Singer-Plus `infer.py`，实际入参为
  `--ref_audio/--melody_audio/--ref_text/--target_text`，输出 44.1kHz 立体声需 downmix
- 止损规则按文档第七节执行

### P2 · Gradio 界面（本机联调已打通，2026-11-25）

直接复用上游 [YingMusic-Singer-Plus](https://github.com/ASLP-lab/YingMusic-Singer-Plus)
官方 `app_local.py`，不另写界面。第三方仓库克隆到 `third_party/`（已 gitignore），
与 P0 主环境隔离：

```bash
git clone --depth 1 https://github.com/ASLP-lab/YingMusic-Singer-Plus.git \
    third_party/YingMusic-Singer-Plus
uv venv --python 3.11 .venv-yingmusic
uv pip install --python .venv-yingmusic/bin/python -r deploy/yingmusic-gradio-requirements.txt
# 权重：MelBandRoformer 871MB 落 ckpts/，主模型 2.8GB 落 HF 缓存
(cd third_party/YingMusic-Singer-Plus && ../../.venv-yingmusic/bin/python initialization.py --task infer)
.venv-yingmusic/bin/python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('ASLP-lab/YingMusic-Singer-Plus', allow_patterns=['config.json','model.safetensors'])"
# 启动（启动器内置 espeak 惰性桩，第三方仓库零侵入）
.venv-yingmusic/bin/python deploy/run_yingmusic_gradio.py   # http://127.0.0.1:7860
```

本机实测（M2 Pro / CPU）：6.5s 中文示例改词合成约 22s，输出 44.1kHz 立体声，
时长与输入一致；API 级冒烟脚本见 `deploy/gradio_smoke_test.py`，
产物在 `output/gradio_smoke/`。两个本机适配点：

- 本机 Homebrew 仅 x86_64 libespeak-ng，arm64 Python 无法加载，启动器复用
  `sing_adapter.py` 的 EspeakBackend 惰性桩（中文 g2p 走 pypinyin+BERT ONNX，不依赖 espeak）
- Gradio 6 的 call API 文件入参需 FileData 结构（含 `meta._type`），裸路径会被 pydantic 拒绝

## 已知限制（交付时需附）

不支持和声多的歌、不支持英文、单句 12 字以内、不承诺音色克隆。
YingMusic-Singer-Plus 的 VAE 权重为 Stability AI Community License，商用受限。
