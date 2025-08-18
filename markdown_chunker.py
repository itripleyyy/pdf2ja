import re
from typing import List
from collections import deque


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
        chunks = re.split(r"(?=^#{1,6} .*)", markdown_text, flags=re.MULTILINE)

        # 空文字列が混ざる可能性があるのでフィルタする
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

        print(f"📊 ヘッダー分割後: {len(chunks)}チャンク")

        # 小さすぎるチャンクを結合
        chunks = self._merge_small_chunks(chunks)

        print(f"✅ 分割完了: 総チャンク数 {len(chunks)}個")
        return chunks

    def _merge_small_chunks(self, small_chunks: List[str]) -> List[str]:
        """小さすぎるチャンクを結合（リファクタリング版）"""
        # チャンクが1つしかない場合はそのまま返す
        if len(small_chunks) <= 1:
            return small_chunks

        n_chunks = len(small_chunks)
        chunks: deque[str] = deque(small_chunks)

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
