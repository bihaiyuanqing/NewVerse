"""本机 Gradio 联调启动器（macOS arm64）。

为什么需要本启动器：
1. app_local.py 位于第三方仓库 third_party/YingMusic-Singer-Plus，且依赖相对路径
   （ckpts/、examples/），必须以该仓库根为 cwd 启动；
2. 本机 Homebrew 只有 x86_64 版 libespeak-ng，arm64 Python 无法 ctypes 加载，
   而 f5_tts/g2p/utils/g2p.py 在模块导入期会为 6 个语种急切构造 EspeakBackend。
   中文 g2p 走 pypinyin + 内置 BERT ONNX，不依赖 espeak，这里复用 sing_adapter.py
   已验证的惰性桩：构造不失败，仅误走非中文音素化时显式报错。

用法（在 NewVerse 仓库根）：
    .venv-yingmusic/bin/python deploy/run_yingmusic_gradio.py
"""

import os
import runpy
import sys

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "third_party", "YingMusic-Singer-Plus")
)

# —— espeak-ng 惰性桩（与 deploy/sing_adapter.py 保持同一方案）——
import phonemizer.backend as _phonemizer_backend

if not _phonemizer_backend.EspeakBackend.is_available():
    class _EspeakStub:
        def __init__(self, *args, **kwargs):
            self.language = kwargs.get("language", args[0] if args else "unknown")

        def phonemize(self, *args, **kwargs):
            raise RuntimeError(
                f"当前环境未安装可用的 arm64 espeak-ng，语种 {self.language} 的音素化不可用；"
                "本机联调仅支持中文歌词"
            )

    _phonemizer_backend.EspeakBackend = _EspeakStub
    print("[launcher] espeak-ng 不可用，已启用惰性桩（仅中文歌词可音素化）", flush=True)

# —— 切换到 YingMusic 仓库根并启动官方 Gradio 应用 ——
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)
print(f"[launcher] cwd={REPO_ROOT}", flush=True)
runpy.run_path(os.path.join(REPO_ROOT, "app_local.py"), run_name="__main__")
