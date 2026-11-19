"""等字数改词。

硬约束（技术实现文档第三、四节）：
- 新词字数必须与原句完全一致，禁止用代码删字、补字凑字数
- LLM 未配置时走内置 Mock（等长同义替换），仅用于链路联调与界面演示
- 所有对外错误统一抛 RewriteFailed 中文异常
"""

import json
import os
import re
import urllib.request

from core.errors import RewriteFailed

# Mock 等长同义词表：键值必须等长，替换前后总字数不变。
# 含演示素材《慌啥子嘛》→《干啥子嘛》的真实改词点。
# 顺序约束：多字词必须排在含同字的单字词之前，避免单字替换破坏多字词；
# 故不收录「醉→累」这类会破坏「沉醉」的单字项。
MOCK_SYNONYMS = {
    # —— 多字词（优先替换）——
    "黄沙": "干啥",
    "慢慢": "快快",
    "月亮": "羽球",
    "烦恼": "忧愁",
    "疲惫": "劳累",
    "节奏": "旋律",
    "自在": "轻快",
    "好运": "福气",
    "星光": "月光",
    "入睡": "出场",
    "静静": "轻轻",
    "清清": "轻轻",
    "清醒": "沉醉",
    "人生": "岁月",
    "远方": "前路",
    "放开": "挥拍",
    "安排": "节拍",
    "街拍": "节拍",
    # —— 单字词 ——
    "慌": "干",
    "山": "海",
    "风": "云",
    "刻": "秒",
    "远": "长",
    "都": "皆",
    "背": "扛",
    "影": "梦",
}

# 不计入字数的字符（空白与中英文标点）
_NON_SYLLABLE = re.compile(r"[\s，。！？、；：,.!?;:~…\-—\"'（）()「」]")


def count_syllables(text: str) -> int:
    """统计有效字数：剔除空白与标点后的字符数。"""

    return len(_NON_SYLLABLE.sub("", text or ""))


def _mock_rewrite(text: str) -> str:
    """确定性等长替换。全部替换项等长，天然保证字数一致。"""

    result = text
    for src, dst in MOCK_SYNONYMS.items():
        if len(src) != len(dst):  # 防御：词表本身必须等长
            raise RewriteFailed(f"Mock 词表配置错误：「{src}」与「{dst}」长度不一致")
        result = result.replace(src, dst)
    return result


def _call_llm(text: str, theme: str, n_syl: int, timeout: float) -> str:
    """调用 OpenAI 兼容的 chat completions 接口，返回候选新词。"""

    base_url = os.environ.get("NEWVERSE_LLM_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("NEWVERSE_LLM_API_KEY", "")
    model = os.environ.get("NEWVERSE_LLM_MODEL", "")
    if not (base_url and api_key and model):
        raise RewriteFailed("LLM 服务未配置（缺少 NEWVERSE_LLM_BASE_URL/API_KEY/MODEL）")

    prompt = (
        f"你是歌词改写专家。请把下面这句歌词围绕主题「{theme or '通用'}」改写，"
        f"要求：1）新歌词必须恰好 {n_syl} 个汉字，与原句字数完全一致；"
        "2）只输出新歌词本身，不要标点、不要解释、不要引号；"
        "3）保持口语化、押韵、适合演唱。\n"
        f"原句：{text}"
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # noqa: BLE001 - 边界统一转中文
        raise RewriteFailed(f"LLM 服务调用失败：{exc}") from exc


def rewrite_line(
    text: str,
    n_syl: int,
    theme: str = "",
    *,
    mock: bool = False,
    max_retry: int = 3,
    timeout: float = 30.0,
) -> str:
    """改写一句歌词，返回与原句严格等字数的新词。

    mock=True 或未配置 LLM 环境变量时走内置 Mock；
    LLM 候选字数不达标时整句重试，超过次数抛 RewriteFailed，绝不删补凑字。
    """

    if count_syllables(text) != n_syl:
        raise RewriteFailed(f"原句字数校验失败：文本 {count_syllables(text)} 字 ≠ 契约 {n_syl} 字")

    use_mock = mock or not (
        os.environ.get("NEWVERSE_LLM_BASE_URL")
        and os.environ.get("NEWVERSE_LLM_API_KEY")
        and os.environ.get("NEWVERSE_LLM_MODEL")
    )
    if use_mock:
        new_text = _mock_rewrite(text)
        if count_syllables(new_text) != n_syl:
            raise RewriteFailed("Mock 改词字数异常，请检查同义词表")
        return new_text

    last_candidate = ""
    for attempt in range(1, max_retry + 1):
        candidate = _call_llm(text, theme, n_syl, timeout)
        last_candidate = candidate
        if count_syllables(candidate) == n_syl:
            return candidate
    raise RewriteFailed(
        f"LLM 连续 {max_retry} 次改词字数不达标（要求 {n_syl} 字，"
        f"最后候选 {count_syllables(last_candidate)} 字：{last_candidate}），已放弃，不做删补凑字"
    )
