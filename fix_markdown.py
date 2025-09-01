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


class MarkdownChunker:
    """Markdownテキストのチャンク分割を担当するクラス"""

    def __init__(self, max_chunk_size: int = 5000, min_chunk_size: int = 500):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def split_markdown(self, markdown_text: str) -> List[str]:
        """Markdownテキストを翻訳用チャンクに分割"""
        if len(markdown_text) <= self.max_chunk_size:
            return [markdown_text]

        # ヘッダーで分割
        chunks = re.split(r"(?=^#{1,6} .*)", markdown_text, flags=re.MULTILINE)
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

        # 小チャンク結合
        chunks = self._merge_small_chunks(chunks)
        return chunks

    def _merge_small_chunks(self, small_chunks: List[str]) -> List[str]:
        if len(small_chunks) <= 1:
            return small_chunks

        chunks: deque[str] = deque(small_chunks)
        merged_chunks = []
        while chunks:
            chunk = chunks.popleft()
            if len(chunk) > self.min_chunk_size:
                merged_chunks.append(chunk)
            elif chunks:
                chunks[0] = chunk + "\n\n" + chunks[0]
            else:
                merged_chunks[-1] += "\n\n" + chunk
        return merged_chunks


REPAIR_SYSTEM_PROMPT = """あなたは、PDFなどから自動変換されたMarkdown文書を修復する専門家です。
入力されるMarkdownには、論文や技術書に含まれる内容（コード、数式、図、表、段落など）が含まれます。
以下の指示に従い、Markdownを再構築してください。

- コード（プログラム、擬似コード、コマンド出力）は適切な```ブロックで囲ってください。可能であれば言語を指定してください（例：```python、```bash、```text）。
- 数式や数値表現（例：$x^2 + y^2 = r^2$、\\begin{equation}など）はMarkdownまたはLaTeX形式で正しく表示されるよう保全してください。
- 誤って見出し（例：`# コメント`）として処理された行は、見出しではなく本文やコメントとして扱ってください。
- 不適切な改行や箇条書きの崩れ、インデントの誤りを修正してください。
- 図や表（テーブル）と思われる部分はMarkdown記法で表現できるように整形してください。
- 画像リンク（![]()記法）はコードブロック内に含めないでください。画像リンクの内容（パスやURL）は一切変更しないでください。
- 内容を要約・改変しないでください。**構造の修正のみに留めてください。**

出力は**Markdown形式のみ**で返してください。説明や補足コメントは不要です。"""


class MarkdownFixer:
    """Markdown構文修復を担当するクラス"""

    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-4.1-mini",
        max_chunk_size: int = 5000,
        min_chunk_size: int = 500,
        verbose: bool = False,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI APIキーが設定されていません。--api-key か環境変数 OPENAI_API_KEY を指定してください。"
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.chunker = MarkdownChunker(max_chunk_size, min_chunk_size)
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(msg, file=sys.stderr)

    def fix_chunk(self, chunk: str, chunk_index: int, total_chunks: int) -> str:
        """単一チャンクを修復"""
        try:
            self._log(
                f"🔧 修復中 ({chunk_index + 1}/{total_chunks}) - {len(chunk)} 文字..."
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
                    {"role": "user", "content": chunk},
                ],
                temperature=0,
            )
            fixed = response.choices[0].message.content.strip()
            usage = getattr(response, "usage", None)
            if usage:
                self._log(
                    f"  ✅ 入力: {usage.prompt_tokens}toks, 出力: {usage.completion_tokens}toks"
                )
            return fixed
        except Exception as e:
            self._log(f"  ❌ エラー: {e}")
            return chunk

    def fix_text(self, markdown_text: str, delay: float = 1.0) -> str:
        """Markdown文字列を修復して返す"""
        chunks = self.chunker.split_markdown(markdown_text)
        fixed_chunks = []
        start_time = time.time()
        for i, chunk in enumerate(chunks):
            fixed_chunks.append(self.fix_chunk(chunk, i, len(chunks)))
            if i < len(chunks) - 1 and delay > 0:
                time.sleep(delay)
        fixed_markdown = "\n\n".join(fixed_chunks)
        self._log(f"✅ 完了: {len(chunks)} チャンク, {time.time() - start_time:.1f} 秒")
        return fixed_markdown

    def fix_file(self, input_path: str, delay: float = 1.0) -> str:
        """ファイルを読み込み修復後Markdownを返す"""
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"入力ファイルが見つかりません: {input_path}")
        self._log(f"📖 読み込み: {input_path}")
        markdown_text = input_file.read_text(encoding="utf-8")
        return self.fix_text(markdown_text, delay=delay)


def parse_args():
    p = argparse.ArgumentParser(
        description="PDF等から変換されたMarkdownを構造修復して出力するCLI"
    )
    p.add_argument(
        "path", help="入力Markdownファイルのパス。'-' を指定するとSTDINから読み込み"
    )
    p.add_argument(
        "--api-key", help="OpenAI APIキー（未指定時は環境変数 OPENAI_API_KEY を使用）"
    )
    p.add_argument(
        "--model", default="gpt-4.1-mini", help="使用モデル（既定: gpt-4.1-mini）"
    )
    p.add_argument(
        "--max-chunk-size", type=int, default=5000, help="チャンク最大文字数"
    )
    p.add_argument(
        "--min-chunk-size", type=int, default=1000, help="チャンク最小文字数"
    )
    p.add_argument(
        "--delay",
        type=float,
        default=0,
        help="API呼び出し間の待機秒（レート制限対策）",
    )
    p.add_argument(
        "--output", "-o", help="出力先パス。未指定ならSTDOUT。'-'でSTDOUTを明示"
    )
    p.add_argument(
        "--verbose", action="store_true", help="進捗ログを表示（STDERRに出力）"
    )
    return p.parse_args()


def main():
    args = parse_args()

    # 入力読み込み
    if args.path == "-":
        src_text = sys.stdin.read()
        use_stdin = True
    else:
        use_stdin = False

    fixer = MarkdownFixer(
        api_key=args.api_key,
        model=args.model,
        max_chunk_size=args.max_chunk_size,
        min_chunk_size=args.min_chunk_size,
        verbose=args.verbose,
    )

    if use_stdin:
        fixed_md = fixer.fix_text(src_text, delay=args.delay)
    else:
        fixed_md = fixer.fix_file(args.path, delay=args.delay)

    # 出力
    if args.output and args.output != "-":
        Path(args.output).write_text(fixed_md, encoding="utf-8")
    else:
        # STDOUTへ
        sys.stdout.write(fixed_md)


if __name__ == "__main__":
    main()
