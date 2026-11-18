import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, unquote, urlsplit

from ocr_text import MODEL_DIR, load_ocr, page_count, pages_with_count

WEB_DIR = Path(__file__).resolve().parent / "web"
MARKDOWN_MODEL_DIR = Path(__file__).resolve().parent / "OpenThai2.0-OCR-Markdown"
HOST = "127.0.0.1"
PORT = 8000
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".pdf"}

_models = {"text": None, "markdown": None}
_ocr_lock = threading.Lock()


def load_markdown_parser():
    if not MARKDOWN_MODEL_DIR.is_dir():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์โมเดล Markdown: {MARKDOWN_MODEL_DIR}")
    if not (MARKDOWN_MODEL_DIR / "model.safetensors").is_file():
        raise FileNotFoundError(f"ไม่พบไฟล์น้ำหนักโมเดล: {MARKDOWN_MODEL_DIR / 'model.safetensors'}")

    import sys

    sys.path.insert(0, str(MARKDOWN_MODEL_DIR))
    try:
        import torch
        from transformers import AutoModel, AutoProcessor, GenerationConfig
        from nemotron_parse.inference import DocumentParser, TransformersBackend
    except ImportError as exc:
        raise RuntimeError(
            "โมเดล Markdown ต้องติดตั้ง PyTorch, Transformers และ dependencies "
            "ตาม pyproject.toml ของ OpenThai2.0-OCR-Markdown ก่อน"
        ) from exc

    if not torch.cuda.is_available():
        cuda_build = torch.version.cuda or "ไม่มี CUDA"
        raise RuntimeError(
            "โมเดล Markdown ตั้งค่าให้ใช้ GPU NVIDIA เท่านั้น แต่ PyTorch ที่กำลังใช้ "
            f"ไม่พบ CUDA GPU (PyTorch CUDA build: {cuda_build}) "
            "ติดตั้ง PyTorch รุ่นที่รองรับ CUDA และตรวจสอบไดรเวอร์ NVIDIA ก่อน"
        )

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    gpu_name = torch.cuda.get_device_name(device)
    model = AutoModel.from_pretrained(
        MARKDOWN_MODEL_DIR,
        trust_remote_code=True,
        dtype=dtype,
        local_files_only=True,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(
        MARKDOWN_MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )
    generation_config = GenerationConfig.from_pretrained(
        MARKDOWN_MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )
    print(f"[web] Markdown OCR loaded on {gpu_name} ({dtype}, {device})")
    return DocumentParser(
        backend=TransformersBackend(model, processor, generation_config)
    )


def json_response(handler, status, payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def start_stream(handler):
    handler.send_response(200)
    handler.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
    handler.send_header("Transfer-Encoding", "chunked")
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()


def stream_event(handler, payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"
    handler.wfile.write(f"{len(body):X}\r\n".encode("ascii"))
    handler.wfile.write(body)
    handler.wfile.write(b"\r\n")
    handler.wfile.flush()


def end_stream(handler):
    handler.wfile.write(b"0\r\n\r\n")
    handler.wfile.flush()


class OCRRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        if urlsplit(self.path).path != "/":
            self.send_error(404, "Not found")
            return

        page = WEB_DIR / "index.html"
        body = page.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        request = urlsplit(self.path)
        if request.path != "/api/ocr":
            json_response(self, 404, {"error": "ไม่พบ API ที่ร้องขอ"})
            return

        model_name = self.headers.get("X-OCR-Model", "text").strip().lower()
        if model_name not in _models:
            json_response(self, 400, {"error": "กรุณาเลือกโมเดล Text หรือ Markdown"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            json_response(self, 400, {"error": "ขนาดไฟล์ไม่ถูกต้อง"})
            return
        if length <= 0:
            json_response(self, 400, {"error": "กรุณาเลือกไฟล์ที่ไม่ว่าง"})
            return
        if length > MAX_UPLOAD_BYTES:
            json_response(self, 413, {"error": "ไฟล์มีขนาดเกิน 25 MB"})
            return

        query = parse_qs(request.query)
        raw_name = unquote(query.get("filename", ["upload"])[0])
        filename = PureWindowsPath(raw_name.replace("/", "\\")).name
        extension = Path(filename).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            json_response(self, 415, {"error": "รองรับไฟล์ภาพและ PDF เท่านั้น"})
            return

        content = self.rfile.read(length)
        if len(content) != length:
            json_response(self, 400, {"error": "รับข้อมูลไฟล์ไม่ครบ"})
            return

        streaming = False
        try:
            with TemporaryDirectory(prefix="openthai-ocr-") as temp_dir:
                file_path = Path(temp_dir) / f"upload{extension}"
                file_path.write_bytes(content)
                started = time.perf_counter()
                total_pages = page_count(file_path)

                start_stream(self)
                streaming = True
                stream_event(self, {
                    "type": "status",
                    "message": "กำลังโหลดโมเดล OCR (ครั้งแรกอาจใช้เวลาสักครู่)",
                    "total": total_pages,
                })
                line_count = 0

                with _ocr_lock:
                    if _models[model_name] is None:
                        if model_name == "text":
                            if not MODEL_DIR.is_dir():
                                raise FileNotFoundError(f"ไม่พบโฟลเดอร์โมเดล: {MODEL_DIR}")
                            _models[model_name] = load_ocr()
                        else:
                            _models[model_name] = load_markdown_parser()
                    model = _models[model_name]
                    stream_event(self, {
                        "type": "status",
                        "message": f"โหลดโมเดลแล้ว กำลังเริ่มอ่าน {total_pages} หน้า",
                        "total": total_pages,
                    })

                    def report_page_start(page_number, page_total):
                        stream_event(self, {
                            "type": "status",
                            "page": page_number,
                            "total": page_total,
                            "message": f"กำลังอ่านหน้า {page_number} / {page_total}",
                        })

                    page_iterator = pages_with_count(file_path, report_page_start)
                    try:
                        for page_number, page_total, image in page_iterator:
                            if model_name == "text":
                                result = model.read(image)
                                page_text = result["text"]
                                page_lines = len(result["lines"])
                                line_count += page_lines
                            else:
                                result = model.parse(image, mode="double")
                                page_text = result["markdown"]
                                page_lines = None
                            stream_event(self, {
                                "type": "page",
                                "page": page_number,
                                "total": page_total,
                                "text": page_text,
                                "lines": page_lines,
                            })
                    finally:
                        page_iterator.close()

                stream_event(self, {
                    "type": "complete",
                    "filename": filename,
                    "pages": total_pages,
                    "lines": line_count if model_name == "text" else None,
                    "model": model_name,
                    "seconds": round(time.perf_counter() - started, 2),
                })
                end_stream(self)
        except Exception as exc:
            if streaming:
                try:
                    stream_event(self, {"type": "error", "error": f"อ่านไฟล์ไม่สำเร็จ: {exc}"})
                    end_stream(self)
                except OSError:
                    self.close_connection = True
            else:
                json_response(self, 422, {"error": f"อ่านไฟล์ไม่สำเร็จ: {exc}"})

    def log_message(self, format, *args):
        print(f"[web] {self.address_string()} - {format % args}")


def main():
    if not (WEB_DIR / "index.html").is_file():
        raise FileNotFoundError(f"ไม่พบหน้าเว็บ: {WEB_DIR / 'index.html'}")

    server = ThreadingHTTPServer((HOST, PORT), OCRRequestHandler)
    print(f"OpenThai OCR พร้อมใช้งานที่ http://{HOST}:{PORT}")
    print("กด Ctrl+C เพื่อปิดเว็บเซิร์ฟเวอร์")
    webbrowser.open(f"http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nกำลังปิดเว็บเซิร์ฟเวอร์...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
