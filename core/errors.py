"""统一异常与超时工具。

硬约束（技术实现文档第三节）：所有模型调用必须带超时，
异常统一转成中文错误信息，不得把裸 stack trace 抛给界面。
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Callable


class NewVerseError(Exception):
    """所有业务异常的基类，message 必须是可直接展示的中文。"""


class SeparationFailed(NewVerseError):
    """人声分离失败或人声能量过低。"""


class AlignmentFailed(NewVerseError):
    """转写或字级对齐失败。"""


class RewriteFailed(NewVerseError):
    """LLM 改词失败或字数始终不达标。"""


class SingFailed(NewVerseError):
    """重唱合成失败。"""


class SpliceError(NewVerseError):
    """音频切片/拼接参数非法。"""


class MixFailed(NewVerseError):
    """混音失败。"""


def call_with_timeout(func: Callable, timeout: float, timeout_msg: str, **kwargs):
    """带超时地调用模型函数。

    超时后向调用方抛出中文异常。注意：Python 无法强杀线程，
    底层推理可能仍在后台结束，但流水线不会继续无限等待。
    """

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, **kwargs)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeout:
            raise NewVerseError(timeout_msg)
        except NewVerseError:
            raise
        except Exception as exc:  # noqa: BLE001 - 边界层统一转中文
            raise NewVerseError(f"模型调用失败：{exc}") from exc
