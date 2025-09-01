#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
from pathlib import Path
import re
from typing import List
from collections import deque

from openai import OpenAI


# ---------- Chunker ----------
class MarkdownChunker:
    """Markdownテキストのチャンク分割を担当するクラス"""

    def __init__(self, max_chunk_size: int = 5000, min_chunk_size: int = 500):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def split_markdown(self, markdown_text: str) -> List[str]:
        """Markdownテキストを翻訳用チャンクに分割"""
        if len(markdown_text) <= self.max_chunk_size:
            return [markdown_text]
        chunks = re.split(r"(?=^#{1,6} .*)", markdown_text, flags=re.MULTILINE)
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        return self._merge_small_chunks(chunks)

    def _merge_small_chunks(self, small_chunks: List[str]) -> List[str]:
        if len(small_chunks) <= 1:
            return small_chunks
        chunks: deque[str] = deque(small_chunks)
        merged = []
        while chunks:
            chunk = chunks.popleft()
            if len(chunk) > self.min_chunk_size:
                merged.append(chunk)
            elif chunks:
                chunks[0] = chunk + "\n\n" + chunks[0]
            else:
                merged[-1] += "\n\n" + chunk
        return merged


# ---------- System prompt ----------
TRANSLATION_SYSTEM_PROMPT = """あなたは英語から日本語への専門翻訳者です。以下のルールに従って翻訳してください：

- 自然で読みやすい日本語に翻訳する
- Markdownの構文（見出し、箇条書き、強調、リンクなど）は保持する
- コードブロック（```）内のコードは翻訳しない（ただし、コメントは翻訳する）
- Markdownの見出し1とPythonコードのコメントの区別に注意してください
- 数式（$...$、\\[...\\]）は翻訳しない
- 画像リンク（![]()）は翻訳しない

翻訳結果のみを出力してください。"""


# ---------- Translator ----------
class MarkdownTranslator:
    """Markdown翻訳を担当するクラス"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1-mini",
        max_chunk_size: int = 5000,
        min_chunk_size: int = 500,
        verbose: bool = False,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI APIキーが未設定です。--api-key または環境変数 OPENAI_API_KEY を指定してください。"
            )
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.chunker = MarkdownChunker(max_chunk_size, min_chunk_size)
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(msg, file=sys.stderr)

    def translate_chunk(self, chunk: str, i: int, total: int) -> str:
        """単一チャンクを翻訳（失敗時は原文を返す）"""
        try:
            self._log(f"🌐 翻訳中 ({i + 1}/{total}) - {len(chunk)}文字...")
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
                    {"role": "user", "content": chunk},
                ],
                temperature=0,
            )
            out = resp.choices[0].message.content.strip()
            usage = getattr(resp, "usage", None)
            if usage:
                self._log(
                    f"  ✅ 入力: {usage.prompt_tokens}toks, 出力: {usage.completion_tokens}toks"
                )
            return out
        except Exception as e:
            self._log(f"  ❌ エラー: {e}（このチャンクは原文を返します）")
            return chunk

    def translate_text(self, text: str, delay: float = 1.0) -> str:
        chunks = self.chunker.split_markdown(text)
        start = time.time()
        out_chunks: List[str] = []
        for i, c in enumerate(chunks):
            out_chunks.append(self.translate_chunk(c, i, len(chunks)))
            if i < len(chunks) - 1 and delay > 0:
                time.sleep(delay)  # レート制限対策
        self._log(f"🎉 完了: {len(chunks)}チャンク, {time.time() - start:.1f}s")
        return "\n\n".join(out_chunks)

    def translate_file(self, path: str, delay: float = 1.0) -> str:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"入力ファイルが見つかりません: {path}")
        self._log(f"📖 読み込み: {path}")
        return self.translate_text(p.read_text(encoding="utf-8"), delay=delay)


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Markdown（英→日）翻訳CLI（STDOUTに翻訳後Markdownを出力）"
    )
    ap.add_argument("path", help="入力Markdownのパス。'-'でSTDINから読み込み")
    ap.add_argument(
        "--api-key", help="OpenAI APIキー（未指定時は環境変数 OPENAI_API_KEY ）"
    )
    ap.add_argument(
        "--model", default="gpt-4.1-mini", help="使用モデル（既定: gpt-4.1-mini）"
    )
    ap.add_argument(
        "--max-chunk-size", type=int, default=5000, help="チャンク最大文字数"
    )
    ap.add_argument(
        "--min-chunk-size", type=int, default=1000, help="チャンク最小文字数"
    )
    ap.add_argument(
        "--delay",
        type=float,
        default=0,
        help="API呼び出し間の待機秒（レート制限対策）",
    )
    ap.add_argument("--output", "-o", help="出力パス。未指定or '-' ならSTDOUT")
    ap.add_argument("--verbose", action="store_true", help="進捗ログをSTDERRに出力")
    return ap.parse_args()


def main():
    args = parse_args()

    translator = MarkdownTranslator(
        api_key=args.api_key,
        model=args.model,
        max_chunk_size=args.max_chunk_size,
        min_chunk_size=args.min_chunk_size,
        verbose=args.verbose,
    )

    # 入力
    if args.path == "-":
        src = sys.stdin.read()
        translated = translator.translate_text(src, delay=args.delay)
    else:
        translated = translator.translate_file(args.path, delay=args.delay)

    # 出力
    if args.output and args.output != "-":
        Path(args.output).write_text(translated, encoding="utf-8")
    else:
        sys.stdout.write(translated)


if __name__ == "__main__":
    main()
