import argparse

from pdf_converter import convert_pdf_to_markdown_with_llm
from markdown_translator import MarkdownTranslator

# コマンドライン引数のパーサーを設定
# ここではPDFファイルのパスとAPIキーを受け取る
parser = argparse.ArgumentParser(description="PDF to Markdown Converter with LLM Translation")
parser.add_argument(
    "pdf_path",
    type=str,
    help="Path to the PDF file to be converted."
)
parser.add_argument(
    "api_key",
    type=str,
    help="API key for the OpenAI service."
)

def main():
    args = parser.parse_args()

    # PDFファイルをMarkdownに変換
    md_filepath = convert_pdf_to_markdown_with_llm(
        pdf_path=args.pdf_path,
        api_key=args.api_key,
        model="gpt-4o-mini"
    )

    # Markdownファイルを翻訳
    translator = MarkdownTranslator(api_key=args.api_key)
    translator.translate_file(md_filepath)

if __name__ == "__main__":
    main()