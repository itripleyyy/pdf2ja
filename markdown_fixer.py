import os
import time
from pathlib import Path
from typing import List
from openai import OpenAI

# 修復用システムプロンプト
REPAIR_SYSTEM_PROMPT = """あなたは、PDFなどから自動変換されたMarkdown文書を修復する専門家です。
入力されるMarkdownには、論文や技術書に含まれる内容（コード、数式、図、表、段落など）が含まれます。
以下の指示に従い、Markdownを再構築してください。

- コード（プログラム、擬似コード、コマンド出力）は適切な```ブロックで囲ってください。可能であれば言語を指定してください（例：```python、```bash、```text）。
- 数式や数値表現（例：$x^2 + y^2 = r^2$、\begin{equation}など）はMarkdownまたはLaTeX形式で正しく表示されるよう保全してください。
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
        model: str = "gpt-4o-mini",
        max_chunk_size: int = 5000,
        min_chunk_size: int = 500,
    ):
        """
        Markdown修復システムの初期化

        Args:
            api_key: OpenAI APIキー（Noneの場合は環境変数から取得）
            model: 使用するOpenAIモデル
            max_chunk_size: チャンク最大文字数
            min_chunk_size: チャンク最小文字数
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI APIキーが設定されていません。環境変数OPENAI_API_KEYまたは引数で指定してください。"
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model

        # チャンク分割を担当するクラス
        self.chunker = MarkdownChunker(max_chunk_size, min_chunk_size)

    def fix_chunk(self, chunk: str, chunk_index: int, total_chunks: int) -> str:
        """単一チャンクを修復"""
        try:
            print(f"🔧 修復中 ({chunk_index + 1}/{total_chunks}) - {len(chunk)}文字...")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
                    {"role": "user", "content": chunk},
                ],
                temperature=0,
                max_tokens=None,  # gpt-4o-miniは自動で最大まで使用
            )

            fixed = response.choices[0].message.content.strip()

            # APIの使用量情報を表示（デバッグ用）
            usage = response.usage
            print(
                f"  ✅ 完了 - 入力: {usage.prompt_tokens}tokens, 出力: {usage.completion_tokens}tokens"
            )

            return fixed

        except Exception as e:
            print(f"  ❌ エラー: {str(e)}")
            # エラーの場合は元のチャンクを返す（修復失敗でも処理続行）
            return chunk

    def fix_file(
        self, input_path: str, output_path: str = None, delay: float = 1.0
    ) -> str:
        """
        Markdownファイルを修復

        Args:
            input_path: 入力ファイルパス
            output_path: 出力ファイルパス（Noneの場合は自動生成）
            delay: API呼び出し間の待機時間（秒）

        Returns:
            修復後のMarkdownテキスト
        """
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"入力ファイルが見つかりません: {input_path}")

        # 出力パスの自動生成
        if output_path is None:
            output_path = (
                input_file.parent / f"{input_file.stem}_fixed{input_file.suffix}"
            )

        print(f"📖 ファイル読み込み: {input_path}")
        markdown_text = input_file.read_text(encoding="utf-8")
        print(f"📊 ファイルサイズ: {len(markdown_text):,}文字")

        # チャンク分割（MarkdownChunkerに委譲）
        chunks = self.chunker.split_markdown(markdown_text)

        if len(chunks) == 1:
            print("📄 分割不要（単一チャンク）")

        # 各チャンクを修復
        fixed_chunks = []
        start_time = time.time()

        for i, chunk in enumerate(chunks):
            fixed_chunk = self.fix_chunk(chunk, i, len(chunks))
            fixed_chunks.append(fixed_chunk)

            # レート制限対策（最後のチャンク以外で待機）
            if i < len(chunks) - 1 and delay > 0:
                time.sleep(delay)

        # 修復結果を結合
        fixed_markdown = "\n\n".join(fixed_chunks)

        # ファイル保存
        output_file = Path(output_path)
        output_file.write_text(fixed_markdown, encoding="utf-8")

        # 結果サマリー
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 50)
        print("🎉 修復完了!")
        print(f"📁 出力ファイル: {output_path}")
        print(f"📊 チャンク数: {len(chunks)}")
        print(f"⏱️  所要時間: {elapsed_time:.1f}秒")
        print(f"📈 修復後サイズ: {len(fixed_markdown):,}文字")
        print("=" * 50)

        return fixed_markdown
