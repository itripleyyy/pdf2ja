import os
import time
import re
from pathlib import Path
from typing import List
from collections import deque
from openai import OpenAI

# 翻訳用システムプロンプト
TRANSLATION_SYSTEM_PROMPT = """あなたは英語から日本語への専門翻訳者です。以下のルールに従って翻訳してください：

- 自然で読みやすい日本語に翻訳する
- Markdownの構文（見出し、箇条書き、強調、リンクなど）は保持する
- Markdown構文が壊れている場合は正しく修正する
- コードブロック（```）内のコードは翻訳しない（ただし、コメントは翻訳する）
- Markdownの見出し1とPythonコードのコメントの区別に注意してください
- 数式（$...$、$...$）は翻訳しない
- 画像リンク（![]()）は翻訳しない

翻訳結果のみを出力してください。"""

class MarkdownChunker:
    """Markdownテキストのチャンク分割を担当するクラス"""
    
    def __init__(self, max_chunk_size: int = 5000, min_chunk_size: int = 500):
        """
        チャンカーの初期化
        
        Args:
            max_chunk_size: チャンク最大文字数
            min_chunk_size: チャンク最小文字数
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def split_markdown(self, markdown_text: str) -> List[str]:
        """Markdownテキストを翻訳用チャンクに分割"""
        if len(markdown_text) <= self.max_chunk_size:
            return [markdown_text]
        
        print("📝 Markdownヘッダーで分割中...")

        # re.split を使ってヘッダー行ごとに分割して、それぞれに元のヘッダーを付け直す
        chunks = re.split(r'(?=^#{1,6} .*)', markdown_text, flags=re.MULTILINE)

        # 空文字列が混ざる可能性があるのでフィルタする
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        print(f"📊 ヘッダー分割後: {len(chunks)}チャンク")
        
        # 小さすぎるチャンクを結合
        chunks = self._merge_small_chunks(chunks)
        
        print(f"✅ 分割完了: 総チャンク数 {len(chunks)}個")
        return chunks
    

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """小さすぎるチャンクを結合（リファクタリング版）"""
        # チャンクが1つしかない場合はそのまま返す
        if len(chunks) <= 1:
            return chunks
        
        n_chunks = len(chunks)
        chunks = deque(chunks)

        merged_chunks = []
        while chunks:
            chunk = chunks.popleft()
            if len(chunk) > self.min_chunk_size:
                merged_chunks.append(chunk)
            elif chunks:
                chunks[0] = chunk + "\n\n" + chunks[0]  # 次のチャンクと結合
            else:
                merged_chunks[-1] += "\n\n" + chunk  # 最後のチャンクと結合
                
        print(f"✅ 総結合回数: {n_chunks - len(merged_chunks)}回")    
        return merged_chunks


class MarkdownTranslator:
    """Markdown翻訳を担当するクラス"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4.1-mini", 
                 max_chunk_size: int = 5000, min_chunk_size: int = 500):
        """
        Markdown翻訳システムの初期化
        
        Args:
            api_key: OpenAI APIキー（Noneの場合は環境変数から取得）
            model: 使用するOpenAIモデル
            max_chunk_size: チャンク最大文字数
            min_chunk_size: チャンク最小文字数
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI APIキーが設定されていません。環境変数OPENAI_API_KEYまたは引数で指定してください。")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        
        # チャンク分割を担当するクラス
        self.chunker = MarkdownChunker(max_chunk_size, min_chunk_size)

    def translate_chunk(self, chunk: str, chunk_index: int, total_chunks: int) -> str:
        """単一チャンクを翻訳"""
        try:
            print(f"🌐 翻訳中 ({chunk_index + 1}/{total_chunks}) - {len(chunk)}文字...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
                    {"role": "user", "content": chunk}
                ],
                temperature=0,
                max_tokens=None  # gpt-4.1-miniは自動で最大まで使用
            )
            
            translated = response.choices[0].message.content.strip()
            
            # APIの使用量情報を表示（デバッグ用）
            usage = response.usage
            print(f"  ✅ 完了 - 入力: {usage.prompt_tokens}tokens, 出力: {usage.completion_tokens}tokens")
            
            return translated
            
        except Exception as e:
            print(f"  ❌ エラー: {str(e)}")
            # エラーの場合は元のチャンクを返す（翻訳失敗でも処理続行）
            return chunk

    def translate_file(self, input_path: str, output_path: str = None, delay: float = 1.0) -> str:
        """
        Markdownファイルを翻訳
        
        Args:
            input_path: 入力ファイルパス
            output_path: 出力ファイルパス（Noneの場合は自動生成）
            delay: API呼び出し間の待機時間（秒）
            
        Returns:
            翻訳後のMarkdownテキスト
        """
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"入力ファイルが見つかりません: {input_path}")
        
        # 出力パスの自動生成
        if output_path is None:
            output_path = input_file.parent / f"{input_file.stem}_ja{input_file.suffix}"
        
        print(f"📖 ファイル読み込み: {input_path}")
        markdown_text = input_file.read_text(encoding="utf-8")
        print(f"📊 ファイルサイズ: {len(markdown_text):,}文字")
        
        # チャンク分割（MarkdownChunkerに委譲）
        chunks = self.chunker.split_markdown(markdown_text)
        
        if len(chunks) == 1:
            print("📄 分割不要（単一チャンク）")
        
        # 各チャンクを翻訳
        translated_chunks = []
        start_time = time.time()
        
        for i, chunk in enumerate(chunks):
            translated_chunk = self.translate_chunk(chunk, i, len(chunks))
            translated_chunks.append(translated_chunk)
            
            # レート制限対策（最後のチャンク以外で待機）
            if i < len(chunks) - 1 and delay > 0:
                time.sleep(delay)
        
        # 翻訳結果を結合
        translated_markdown = "\n\n".join(translated_chunks)
        
        # ファイル保存
        output_file = Path(output_path)
        output_file.write_text(translated_markdown, encoding="utf-8")
        
        # 結果サマリー
        elapsed_time = time.time() - start_time
        print("\n" + "="*50)
        print("🎉 翻訳完了!")
        print(f"📁 出力ファイル: {output_path}")
        print(f"📊 チャンク数: {len(chunks)}")
        print(f"⏱️  所要時間: {elapsed_time:.1f}秒")
        print(f"📈 翻訳後サイズ: {len(translated_markdown):,}文字")
        print("="*50)
        
        return translated_markdown