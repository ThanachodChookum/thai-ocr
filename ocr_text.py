import sys
import time
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "OpenThai2.0-OCR-Text"


def pages_with_count(path, on_page_start=None):
    if path.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        try:
            total = len(pdf)
            for i in range(total):
                if on_page_start:
                    on_page_start(i + 1, total)
                yield i + 1, total, pdf[i].render(scale=200 / 72).to_pil()
        finally:
            pdf.close()
    else:
        if on_page_start:
            on_page_start(1, 1)
        yield 1, 1, str(path)


def page_count(path):
    if path.suffix.lower() != ".pdf":
        return 1

    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        return len(pdf)
    finally:
        pdf.close()


def pages(path):
    for _, _, page in pages_with_count(path):
        yield page


def load_ocr():
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"Model folder was not found: {MODEL_DIR}")

    sys.path.insert(0, str(MODEL_DIR))
    from runtime.line_engine import LineOCR

    return LineOCR(str(MODEL_DIR), threads=2)


def recognize_file(path, ocr, report):
    texts = []
    for number, page in enumerate(pages(path), start=1):
        start = time.perf_counter()
        result = ocr.read(page)
        seconds = time.perf_counter() - start
        report(f"{path.name}: page {number}, {len(result['lines'])} lines ({seconds:.2f} s)")
        texts.append(result["text"])

    text = "\n\n".join(texts)
    output_path = path.with_name(path.name + ".txt")
    output_path.write_text(text, encoding="utf-8")
    return text, output_path


def run_cli(paths):
    ocr = load_ocr()
    for name in paths:
        path = Path(name)
        text, _ = recognize_file(path, ocr, lambda message: print(message, file=sys.stderr))
        print(text)


def main():
    if len(sys.argv) > 1:
        run_cli(sys.argv[1:])
    else:
        from web_app import main as launch_web

        launch_web()


if __name__ == "__main__":
    main()
