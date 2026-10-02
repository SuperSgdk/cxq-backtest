"""检查 Git 已跟踪的公开发布文件；疑似凭据只报告文件和行号。

这个检查是文件白名单与模式检查，不能替代数据授权确认或人工审查。
默认检查工作区里已跟踪文件的内容；--staged 检查暂存区内容。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 1_048_576
CSV_FIXTURES = {
    "examples/market_handcalc.csv",
    "examples/signals_handcalc.csv",
}
TOP_FILES = {
    ".gitignore", ".dockerignore", ".gitattributes",
    "README.md", "LICENSE", "NOTICE", "Dockerfile",
    "pyproject.toml", "requirements.txt", "run_tests.py", "app.py", "启动面板.cmd",
}
DIR_SUFFIXES = {
    "cxq_bt": {".py"},
    "tests": {".py"},
    "docs": {".md"},
    "examples": {".md", ".py"},
    "scripts": {".py"},
}

# 将敏感标记分开构造，避免检查脚本自身形成真实的凭据样例。
SECRET_PATTERNS = {
    "私钥标记": re.compile(
        re.escape("-----BEGIN ") + r"(?:RSA |EC |DSA |OPENSSH )?"
        + re.escape("PRIVATE" + " KEY-----")
    ),
    "常见平台凭据": re.compile(
        r"(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{20,}"
        r"|github_pat_[A-Za-z0-9_]{20,}"
        r"|sk-(?:proj-)?[A-Za-z0-9_-]{20,}"
        r"|AKIA[0-9A-Z]{16})(?![A-Za-z0-9])"
    ),
    "URL 内的鉴权信息": re.compile(
        r"https?://[^\s/:@]+:[^\s/@]+@"
        r"|[?&](?:token|api_key|access_token|password)=[^&\s]+",
        re.IGNORECASE,
    ),
    "个人电脑路径": re.compile(
        r"[A-Za-z]:" + r"[\\/]+Users[\\/]+[^\\/\s]+"
        + r"|/" + r"Users/[^/\s]+|/" + r"home/[^/\s]+",
        re.IGNORECASE,
    ),
    "邮箱地址": re.compile(
        r"(?<![\w.-])[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    ),
}
ASSIGNMENT = re.compile(
    r"""(?ix)
    \b(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?key|
        password|passwd|tushare[_-]?token|webhook[_-]?url)
    \s*['"]?\s*[:=]\s*['"]([^'"\r\n]+)['"]
    """
)
PLACEHOLDERS = {
    "", "none", "null", "example", "changeme", "change_me",
    "your_api_key", "your_token", "your_password",
}
PRIVATE_REFERENCES = (
    "chenxiaoqun_" + "strategy",
    "cxq_" + "ml_research",
)


def git(*args: str) -> bytes:
    """Git 错误只显示概括，避免输出私有远端或配置。"""
    proc = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        capture_output=True,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError("无法读取 Git 发布清单，请在公开版目录初始化并暂存发布文件。")
    return proc.stdout


def allowed_path(name: str) -> bool:
    """目录与扩展双白名单；CSV 只允许明确的人工样例位置。"""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        return False
    if path.suffix.lower() == ".csv":
        return name in CSV_FIXTURES
    if name in TOP_FILES:
        return True
    if len(path.parts) < 2:
        return False
    if path.parts[0] == ".github":
        return len(path.parts) == 3 and path.parts[1] == "workflows" and path.suffix in {".yml", ".yaml"}
    if path.parts[:2] == ("tests", "fixtures"):
        return path.suffix in {".py", ".md"}
    return path.suffix in DIR_SUFFIXES.get(path.parts[0], set())


def content_findings(text: str) -> list[tuple[int, str]]:
    """只返回位置和类别，不返回匹配到的值。"""
    findings: set[tuple[int, str]] = set()
    for kind, pattern in SECRET_PATTERNS.items():
        for match in pattern.finditer(text):
            # 作者可自行选择 noreply 地址；普通邮箱仍需显式移除或审查。
            if kind == "邮箱地址" and match.group().endswith("users.noreply.github.com"):
                continue
            findings.add((text.count("\n", 0, match.start()) + 1, kind))
    for match in ASSIGNMENT.finditer(text):
        value = match.group(1).strip().lower()
        if value in PLACEHOLDERS or value.startswith(("$" + "{", "<", "os.", "getenv(")):
            continue
        findings.add((text.count("\n", 0, match.start()) + 1, "疑似明文凭据赋值"))
    for marker in PRIVATE_REFERENCES:
        for match in re.finditer(re.escape(marker), text):
            findings.add((text.count("\n", 0, match.start()) + 1, "原私有研究引用"))
    return sorted(findings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="读取暂存区内容而非工作区内容")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        names = [n.decode("utf-8") for n in git("ls-files", "-z").split(b"\0") if n]
    except (RuntimeError, UnicodeDecodeError) as error:
        print(f"检查未完成：{error}")
        return 2
    if not names:
        print("检查未完成：Git 发布清单为空，请先暂存准备公开的文件。")
        return 2

    errors: list[tuple[str, int, str]] = []
    total_bytes = 0
    for name in sorted(set(names)):
        if not allowed_path(name):
            errors.append((name, 0, "不在发布目录或扩展白名单"))
            continue
        path = ROOT / name
        if path.is_symlink():
            errors.append((name, 0, "符号链接须人工审查"))
            continue
        try:
            raw = git("show", ":" + name) if args.staged else path.read_bytes()
        except (OSError, RuntimeError):
            errors.append((name, 0, "无法读取发布文件"))
            continue
        total_bytes += len(raw)
        if len(raw) > MAX_FILE_BYTES:
            errors.append((name, 0, "超过单文件体积上限"))
            continue
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            errors.append((name, 0, "不是允许的 UTF-8 文本发布物"))
            continue
        if "\0" in text:
            errors.append((name, 0, "含二进制内容"))
            continue
        errors.extend((name, line, kind) for line, kind in content_findings(text))

    if errors:
        print(f"公开文件检查未通过：{len(errors)} 项。仅显示路径、行号和类型。")
        for name, line, kind in errors:
            print(f"{name}:{line or '-'} [{kind}]")
        return 1
    print(f"公开文件检查通过：{len(names)} 个已跟踪文件，合计 {total_bytes} 字节。")
    print("未发现上述模式匹配，不等于绝对无泄露；数据分发权仍须单独确认。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
