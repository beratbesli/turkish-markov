"""Dependency-free local web interface for the Turkish Markov application."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import webbrowser
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from .database import MarkovDatabase
from .exceptions import MarkovError
from .indexer import BuildConfig, BuildReport, build_database
from .markov import MarkovGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = PROJECT_ROOT / "turkish.db"


HTML = r"""<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Türkçe Markov Metin Üretici</title>
  <style>
    :root{--bg:#eef3f8;--card:#fff;--ink:#14202b;--muted:#607080;--blue:#2267d8;--line:#d8e1ea;--ok:#16784a;--bad:#b42318}
    *{box-sizing:border-box} body{margin:0;background:linear-gradient(135deg,#e8f0fa,#f5f7fa);color:var(--ink);font:15px/1.5 system-ui,"Segoe UI",sans-serif}
    .wrap{max-width:940px;margin:36px auto;padding:0 18px}.head{display:flex;align-items:flex-start;justify-content:space-between;gap:20px;margin-bottom:22px}
    h1{font-size:29px;margin:0 0 5px}.sub{color:var(--muted);margin:0}.card{background:var(--card);border:1px solid var(--line);border-radius:16px;box-shadow:0 12px 35px #29476416;overflow:hidden}
    .tabs{display:flex;border-bottom:1px solid var(--line);background:#f7f9fc}.tab{border:0;background:none;padding:16px 22px;font-weight:700;color:var(--muted);cursor:pointer}
    .tab.active{color:var(--blue);box-shadow:inset 0 -3px var(--blue)}.panel{display:none;padding:24px}.panel.active{display:block}
    label{display:block;font-weight:650;margin:0 0 6px}.field{margin-bottom:17px}.row{display:grid;grid-template-columns:1fr 1fr;gap:16px}
    input,select,textarea{width:100%;border:1px solid #bcc9d5;border-radius:9px;padding:11px 12px;background:#fff;color:var(--ink);font:inherit}
    input:focus,select:focus,textarea:focus{outline:3px solid #2d76e526;border-color:var(--blue)} textarea{min-height:240px;resize:vertical;line-height:1.65}
    button{border:0;border-radius:9px;padding:11px 17px;font:700 14px inherit;cursor:pointer}.primary{background:var(--blue);color:#fff}.secondary{background:#e8eef5;color:#25384a}.danger{background:#fff0ee;color:var(--bad)}
    button:disabled{opacity:.55;cursor:not-allowed}.actions{display:flex;gap:9px;flex-wrap:wrap;margin:4px 0 17px}.status{min-height:24px;color:var(--muted);font-weight:600}.status.ok{color:var(--ok)}.status.bad{color:var(--bad)}
    .progress{height:12px;border-radius:99px;background:#e6edf4;overflow:hidden;margin:14px 0 8px}.bar{height:100%;width:0;background:linear-gradient(90deg,#2267d8,#35a5e8);transition:width .25s}
    .hint{color:var(--muted);font-size:13px;margin-top:6px}.close{white-space:nowrap}@media(max-width:650px){.row{grid-template-columns:1fr}.head{display:block}.close{margin-top:14px}.panel{padding:18px}}
  </style>
</head>
<body><main class="wrap">
  <header class="head"><div><h1>Türkçe Markov Metin Üretici</h1><p class="sub">Hazır modelinden yeni Türkçe metinler üret.</p></div><button class="danger close" onclick="closeApp()">Uygulamayı kapat</button></header>
  <section class="card">
    <nav class="tabs"><button class="tab active" data-id="generate">Metin Üret</button><button class="tab" data-id="build">Veritabanı Oluştur</button></nav>
    <div class="panel active" id="generate">
      <div class="field"><label>Veritabanı dosyası</label><input id="gdb" value="__DEFAULT_DB__"><div class="hint">Hazırladığın turkish.db otomatik seçildi.</div></div>
      <div class="row">
        <div class="field"><label>Üretim türü</label><select id="gmode"><option value="word">Kelime / cümle</option><option value="char">Yeni, uydurma kelime</option></select></div>
        <div class="field"><label>Başlangıç sözü</label><input id="prompt" value="bir" placeholder="Örnek: İstanbul"></div>
      </div>
      <div class="row">
        <div class="field"><label>Uzunluk</label><input id="length" type="number" min="1" max="10000" value="50"></div>
        <div class="field"><label>Yaratıcılık (sıcaklık)</label><input id="temperature" type="number" min="0" max="5" step="0.1" value="0.6"><div class="hint">0.6 daha düzenli, 1.1 daha yaratıcıdır.</div></div>
      </div>
      <div class="actions"><button id="generateBtn" class="primary" onclick="generateText()">Metin Üret</button><button class="secondary" onclick="copyResult()">Sonucu Kopyala</button><button class="secondary" onclick="document.querySelector('#result').value=''">Temizle</button></div>
      <div id="gstatus" class="status">Hazır</div>
      <textarea id="result" placeholder="Üretilen metin burada görünecek..."></textarea>
    </div>
    <div class="panel" id="build">
      <div class="field"><label>Metin klasörü veya .txt dosyası</label><input id="inputPath" value="__PROJECT_ROOT__"></div>
      <div class="field"><label>Yeni veritabanının kaydedileceği yer</label><input id="bdb" value="__NEW_DB__"></div>
      <div class="row">
        <div class="field"><label>N-gram derecesi</label><input id="order" type="number" min="1" max="10" value="2"></div>
        <div class="field"><label>Model türü</label><select id="bmode"><option value="word">Kelime</option><option value="char">Karakter</option><option value="both">İkisi birden</option></select></div>
      </div>
      <div class="row">
        <div class="field"><label>İşlemci sayısı</label><input id="workers" type="number" min="0" max="64" value="0"><div class="hint">0 bırakırsan otomatik seçilir.</div></div>
        <div class="field"><label><input id="overwrite" type="checkbox" style="width:auto;margin-right:8px">Var olan DB'nin üzerine yaz</label></div>
      </div>
      <div class="actions"><button id="buildBtn" class="primary" onclick="buildDb()">Veritabanını Oluştur</button></div>
      <div class="progress"><div id="bar" class="bar"></div></div><div id="bstatus" class="status">Hazır</div>
      <p class="hint">Büyük veri setlerinde bu işlem uzun sürer. Tamamlanana kadar bu pencereyi açık bırak.</p>
    </div>
  </section>
</main>
<script>
  let busy=false;
  document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{document.querySelectorAll('.tab,.panel').forEach(x=>x.classList.remove('active'));t.classList.add('active');document.getElementById(t.dataset.id).classList.add('active')});
  async function post(url,data){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});const j=await r.json();if(!r.ok)throw Error(j.error||'İşlem başarısız');return j}
  function setBusy(v){busy=v;document.querySelector('#generateBtn').disabled=v;document.querySelector('#buildBtn').disabled=v}
  function status(id,text,kind=''){const e=document.getElementById(id);e.textContent=text;e.className='status '+kind}
  async function generateText(){
    if(busy)return;setBusy(true);status('gstatus','Metin üretiliyor...');
    try{await post('/api/generate',{db:document.querySelector('#gdb').value,mode:document.querySelector('#gmode').value,prompt:document.querySelector('#prompt').value,length:Number(document.querySelector('#length').value),temperature:Number(document.querySelector('#temperature').value)});poll()}
    catch(e){setBusy(false);status('gstatus',e.message,'bad')}
  }
  async function buildDb(){
    if(busy)return;if(!confirm('Veritabanı oluşturma işlemi başlatılsın mı?'))return;setBusy(true);document.querySelector('#bar').style.width='0%';status('bstatus','Dosyalar hazırlanıyor...');
    try{await post('/api/build',{input:document.querySelector('#inputPath').value,db:document.querySelector('#bdb').value,order:Number(document.querySelector('#order').value),mode:document.querySelector('#bmode').value,workers:Number(document.querySelector('#workers').value),overwrite:document.querySelector('#overwrite').checked});poll()}
    catch(e){setBusy(false);status('bstatus',e.message,'bad')}
  }
  async function poll(){
    try{const r=await fetch('/api/status');const s=await r.json();
      if(s.kind==='generate'){status('gstatus',s.message,s.state==='error'?'bad':s.state==='done'?'ok':'');if(s.result!==undefined)document.querySelector('#result').value=s.result}
      if(s.kind==='build'){status('bstatus',s.message,s.state==='error'?'bad':s.state==='done'?'ok':'');document.querySelector('#bar').style.width=(s.progress||0)+'%';if(s.db&&s.state==='done')document.querySelector('#gdb').value=s.db}
      if(s.state==='running')setTimeout(poll,500);else setBusy(false)
    }catch(e){setBusy(false);status('gstatus','Bağlantı hatası: '+e.message,'bad')}
  }
  async function copyResult(){const text=document.querySelector('#result').value;if(!text)return;await navigator.clipboard.writeText(text);status('gstatus','Metin panoya kopyalandı','ok')}
  async function closeApp(){if(busy&&!confirm('Bir işlem sürüyor. Yine de kapatılsın mı?'))return;await fetch('/api/shutdown',{method:'POST'});document.body.innerHTML='<main class="wrap"><h1>Uygulama kapatıldı.</h1><p>Bu sekmeyi kapatabilirsin.</p></main>'}
</script></body></html>"""


class AppState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.data: dict[str, Any] = {
            "kind": "generate",
            "state": "idle",
            "message": "Hazır",
            "progress": 0.0,
        }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.data)

    def update(self, **values: Any) -> None:
        with self.lock:
            self.data.update(values)

    def start(self, kind: str, message: str) -> bool:
        with self.lock:
            if self.data.get("state") == "running":
                return False
            self.data = {
                "kind": kind,
                "state": "running",
                "message": message,
                "progress": 0.0,
            }
            return True


STATE = AppState()


class _DatabaseCache:
    """Keep SQLite's page cache warm between web-interface generations."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.path: Path | None = None
        self.database: MarkovDatabase | None = None

    def generate(
        self, path: Path, mode: str, prompt: str, length: int, temperature: float
    ) -> str:
        resolved = path.expanduser().resolve()
        with self.lock:
            if self.database is None or self.path != resolved:
                if self.database is not None:
                    self.database.close()
                self.database = MarkovDatabase(resolved, check_same_thread=False)
                self.path = resolved
            return MarkovGenerator(self.database).generate(
                mode, prompt, length=length, temperature=temperature
            )

    def invalidate(self, path: Path | None = None) -> None:
        with self.lock:
            if self.database is None:
                return
            if path is None or self.path == path.expanduser().resolve():
                self.database.close()
                self.database = None
                self.path = None


DATABASE_CACHE = _DatabaseCache()


def _background(kind: str, operation: Callable[[], dict[str, Any]]) -> None:
    def run() -> None:
        try:
            result = operation()
        except (MarkovError, OSError, sqlite3.Error, ValueError) as exc:
            STATE.update(state="error", message=str(exc))
        except Exception as exc:  # pragma: no cover - final UI safety net
            STATE.update(state="error", message=f"Beklenmeyen hata: {exc}")
        else:
            STATE.update(state="done", **result)

    threading.Thread(target=run, daemon=True, name=f"markov-{kind}").start()


def _generate(data: dict[str, Any]) -> None:
    if not STATE.start("generate", "Metin üretiliyor..."):
        raise ValueError("Başka bir işlem devam ediyor.")
    database_path = Path(str(data.get("db", "")).strip())
    mode = str(data.get("mode", "word"))
    prompt = str(data.get("prompt", ""))
    length = int(data.get("length", 0))
    temperature = float(data.get("temperature", 0.8))
    if mode not in {"word", "char"}:
        raise ValueError("Geçersiz üretim türü.")
    if length < 1:
        raise ValueError("Uzunluk en az 1 olmalıdır.")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("Sıcaklık 0 veya daha büyük olmalıdır.")

    def operation() -> dict[str, Any]:
        result = DATABASE_CACHE.generate(
            database_path, mode, prompt, length, temperature
        )
        return {"message": "Metin hazır", "result": result}

    _background("generate", operation)


def _build(data: dict[str, Any]) -> None:
    if not STATE.start("build", "Dosyalar hazırlanıyor..."):
        raise ValueError("Başka bir işlem devam ediyor.")
    config = BuildConfig(
        input_path=Path(str(data.get("input", "")).strip()),
        database_path=Path(str(data.get("db", "")).strip()),
        order=int(data.get("order", 2)),
        mode=str(data.get("mode", "word")),
        workers=int(data.get("workers", 0)),
        encoding_errors="replace",
        overwrite=bool(data.get("overwrite", False)),
    )
    config.validate()

    def progress(phase: str, current: int, total: int, message: str) -> None:
        ratio = 100.0 if total <= 0 else min(100.0, current * 100.0 / total)
        labels = {"index": "Veriler işleniyor", "merge": "Birleştiriliyor", "done": "Tamamlandı"}
        STATE.update(progress=ratio, message=f"{labels.get(phase, phase)}: %{ratio:.1f} — {message}")

    def operation() -> dict[str, Any]:
        report: BuildReport = build_database(config, progress=progress)
        DATABASE_CACHE.invalidate(report.database_path)
        rows = ", ".join(f"{key}: {value:,}" for key, value in report.row_counts.items())
        return {
            "progress": 100.0,
            "message": f"Veritabanı hazır — geçiş satırları: {rows}",
            "db": str(report.database_path),
        }

    _background("build", operation)


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "TurkishMarkov/1.0"

    def log_message(self, _format: str, *args: object) -> None:
        return

    def _json(self, value: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 64 * 1024:
            raise ValueError("İstek çok büyük.")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Geçersiz istek.")
        return value

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            page = (
                HTML.replace("__DEFAULT_DB__", escape(str(DEFAULT_DATABASE), quote=True))
                .replace("__PROJECT_ROOT__", escape(str(PROJECT_ROOT), quote=True))
                .replace("__NEW_DB__", escape(str(PROJECT_ROOT / "yeni_model.db"), quote=True))
            ).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(page)
        elif self.path == "/api/status":
            self._json(STATE.snapshot())
        else:
            self._json({"error": "Bulunamadı"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/api/generate":
                _generate(self._body())
                self._json({"ok": True}, HTTPStatus.ACCEPTED)
            elif self.path == "/api/build":
                _build(self._body())
                self._json({"ok": True}, HTTPStatus.ACCEPTED)
            elif self.path == "/api/shutdown":
                self._json({"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self._json({"error": "Bulunamadı"}, HTTPStatus.NOT_FOUND)
        except (ValueError, TypeError, json.JSONDecodeError, MarkovError) as exc:
            if STATE.snapshot().get("state") == "running":
                STATE.update(state="error", message=str(exc))
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), RequestHandler)
    url = f"http://127.0.0.1:{server.server_port}/"
    threading.Timer(0.35, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        DATABASE_CACHE.invalidate()
        server.server_close()


if __name__ == "__main__":
    main()
