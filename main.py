# ========================================================================
# Source notebook code cell 2
# ========================================================================

import sys, platform, subprocess, shutil
print("Python   :", sys.version.split()[0], "|", platform.platform())
print("Interpreter:", sys.executable)

# Check torch in a throw-away process so this notebook does NOT grab VRAM for a CUDA context.
probe = "import torch;print(torch.__version__, '| CUDA build:', torch.version.cuda, '| GPU visible:', torch.cuda.is_available())"
r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
print("PyTorch  :", (r.stdout.strip() or "NOT INSTALLED -> run setup_windows.bat") + (r.stderr.strip()[-200:] if r.returncode else ""))

smi = shutil.which("nvidia-smi")
print("nvidia-smi:", smi or "NOT FOUND -> install the NVIDIA driver (GeForce Experience / NVIDIA app)")
if smi:
    print(subprocess.run([smi, "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv"],
                         capture_output=True, text=True).stdout)
for mod in ("transformers", "bitsandbytes", "fitz", "cv2", "gradio", "pandas", "sklearn"):
    try:
        __import__(mod); print(f"  ok  {mod}")
    except Exception as ex:
        print(f"  MISSING {mod}: {ex}")


# ========================================================================
# Source notebook code cell 4
# ========================================================================

import os, sys, json, re, time, gc, html, sqlite3, subprocess, hashlib, datetime, traceback
from pathlib import Path

# =================  EDIT THIS ONLY IF YOU WANT A DIFFERENT FOLDER  =================
# Must be the SAME folder where setup_windows.bat created the 'venv'. Needs ~15 GB free.
BASE_DIR = Path(r"D:\AnswerEvalAI") if os.path.isdir("D:\\") else Path(r"C:\AnswerEvalAI")
# ====================================================================================

# Keep every download inside BASE_DIR (nothing hidden in C:\Users\...) and keep telemetry off.
os.environ["HF_HOME"] = str(BASE_DIR / "hf_cache")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

MODEL_DIR  = BASE_DIR / "models"
TOOLS_DIR  = BASE_DIR / "tools" / "llama.cpp"
DATA_DIR   = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
REPORT_DIR = DATA_DIR / "reports"
WORK_DIR   = DATA_DIR / "work"
DB_PATH    = DATA_DIR / "evaluations.db"
for p in (MODEL_DIR / "ocr", MODEL_DIR / "llm", TOOLS_DIR, UPLOAD_DIR, REPORT_DIR, WORK_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ---- OCR model (vision-language; reads handwriting + printed text) ----
OCR_REPO = "Qwen/Qwen2.5-VL-3B-Instruct"
OCR_DIR  = MODEL_DIR / "ocr" / "Qwen2.5-VL-3B-Instruct"
OCR_MAX_VISUAL_TOKENS = 900      # lower (e.g. 600) if you ever hit CUDA out-of-memory during OCR

# ---- LLM presets (GGUF, 4-bit). 'layers' = transformer block count, used for GPU offload maths ----
LLM_PRESETS = {
    "Fast (Qwen2.5-3B, fully on GPU)": dict(
        repo="bartowski/Qwen2.5-3B-Instruct-GGUF", file="Qwen2.5-3B-Instruct-Q4_K_M.gguf", layers=36, n_ctx=6144),
    "Accurate (Qwen2.5-7B, GPU+CPU split)": dict(
        repo="bartowski/Qwen2.5-7B-Instruct-GGUF", file="Qwen2.5-7B-Instruct-Q4_K_M.gguf", layers=28, n_ctx=6144),
}
DEFAULT_LLM = list(LLM_PRESETS)[0]
VRAM_RESERVE_MB = 800            # kept free for KV-cache + compute buffers of the LLM
MARK_STEP = 0.5                  # marks are rounded to the nearest 0.5
LLM_BACKEND = "auto"             # "auto" = llama-server.exe if present, else llama-cpp-python ; or force "server" / "python"

def gpu_mem():
    # Uses nvidia-smi, so this notebook never creates a CUDA context (no VRAM wasted).
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"], text=True, timeout=10).strip().splitlines()[0]
        n, t, u, f = [x.strip() for x in out.split(",")]
        return dict(name=n, total=int(t), used=int(u), free=int(f))
    except Exception:
        return None

def find_llama_server():
    if TOOLS_DIR.exists():
        for p in TOOLS_DIR.rglob("llama-server.exe"):
            return p
    return None

print("Storage root :", BASE_DIR)
print("GPU          :", gpu_mem() or "no NVIDIA GPU detected (will fall back to CPU - slow)")
print("llama-server :", find_llama_server() or "not downloaded yet (next step does it)")


# ========================================================================
# Source notebook code cell 6
# ========================================================================

import urllib.request, zipfile
from huggingface_hub import snapshot_download, hf_hub_download

def download_ocr():
    if (OCR_DIR / "config.json").exists():
        return f"OCR model already present: {OCR_DIR}"
    snapshot_download(repo_id=OCR_REPO, local_dir=str(OCR_DIR),
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
    return f"OCR model downloaded to {OCR_DIR}"

def llm_path(key):
    return MODEL_DIR / "llm" / LLM_PRESETS[key]["file"]

def download_llm(key):
    p = LLM_PRESETS[key]
    if llm_path(key).exists():
        return f"LLM already present: {llm_path(key)}"
    hf_hub_download(repo_id=p["repo"], filename=p["file"], local_dir=str(MODEL_DIR / "llm"))
    return f"LLM downloaded to {llm_path(key)}"

def download_llama_server():
    # Official prebuilt Windows CUDA binaries of llama.cpp (no compiler needed); picks CUDA 12.x if offered, else 13.x.
    exe = find_llama_server()
    if exe:
        return f"llama-server already present: {exe}"
    req = urllib.request.Request("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30",
                                 headers={"User-Agent": "answer-eval"})
    releases = json.load(urllib.request.urlopen(req, timeout=60))
    def _ver(n): return float(re.search(r"cuda-(\d+\.\d+)", n).group(1))
    pat = r"llama-.*-bin-win-cuda-1[23]\.\d+-x64\.zip"
    for rel in releases:                       # newest release that really contains Windows CUDA zips
        assets = {a["name"]: a["browser_download_url"] for a in rel.get("assets", [])}
        main = sorted((n for n in assets if re.fullmatch(pat, n)), key=_ver)
        if main:
            break
    else:
        raise RuntimeError("No release with a Windows CUDA zip found. Download 'llama-*-bin-win-cuda-*-x64.zip' and "
                           f"'cudart-llama-bin-win-cuda-*-x64.zip' manually from https://github.com/ggml-org/llama.cpp/releases and unzip BOTH into {TOOLS_DIR}")
    pick = main[0]          # lowest CUDA version offered (12.x needs driver 551+, 13.x needs 580+)
    rt = [n for n in assets if n.startswith("cudart-llama-bin-win-") and n.endswith("-x64.zip") and _ver(n) == _ver(pick)]
    for n in [pick] + rt[:1]:
        dst = TOOLS_DIR / n
        urllib.request.urlretrieve(assets[n], dst)
        with zipfile.ZipFile(dst) as z:
            z.extractall(TOOLS_DIR)
        dst.unlink()
    exe = find_llama_server()
    if not exe:
        raise RuntimeError("llama-server.exe not found after extraction.")
    return f"llama-server installed: {exe}"

# Which LLMs to fetch now? (7B is optional, ~4.7 GB)
DOWNLOAD_NOW = [DEFAULT_LLM]          # add list(LLM_PRESETS)[1] to also fetch the 7B model
print(download_ocr())
for k in DOWNLOAD_NOW:
    print(download_llm(k))
try:
    print(download_llama_server())
except Exception as ex:
    print("llama-server download failed ->", ex)


# ========================================================================
# Source notebook code cell 8
# ========================================================================

import cv2, numpy as np, fitz            # fitz = PyMuPDF
from PIL import Image, ImageOps

def to_path(f):
    return getattr(f, "name", f) if f is not None else None

def load_pages(path, dpi=200):
    # Returns list of PIL images for an image file or each page of a PDF.
    path = str(path)
    if Path(path).suffix.lower() == ".pdf":
        pages = []
        with fitz.open(path) as doc:
            for pg in doc:
                pix = pg.get_pixmap(dpi=dpi)
                pages.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
        return pages
    return [ImageOps.exif_transpose(Image.open(path)).convert("RGB")]

def _estimate_skew(gray):
    small = cv2.resize(gray, None, fx=0.35, fy=0.35, interpolation=cv2.INTER_AREA)
    bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h, w = bw.shape
    best, best_score = 0.0, -1.0
    for ang in np.arange(-6, 6.1, 0.5):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        rot = cv2.warpAffine(bw, M, (w, h), flags=cv2.INTER_NEAREST)
        score = np.var(rot.sum(axis=1))
        if score > best_score:
            best, best_score = ang, score
    return best

def preprocess(img, max_side=1800):
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ang = _estimate_skew(gray)
    if abs(ang) >= 0.5:
        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
    gray = cv2.fastNlMeansDenoising(gray, None, h=7)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return Image.fromarray(gray).convert("RGB")

def pdf_text_if_digital(path, min_chars=60):
    # Digital PDFs already contain text -> no OCR needed.
    with fitz.open(str(path)) as doc:
        txt = "\n".join(pg.get_text() for pg in doc)
    return txt if len(txt.strip()) >= min_chars else None
print("Pre-processing ready.")


# ========================================================================
# Source notebook code cell 10
# ========================================================================

OCR_WORKER = r"""
import sys, json, gc
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

PROMPTS = {
 "answer": "This is a page from a student's handwritten answer script. Transcribe ALL the text exactly as written, in reading order, line by line. Keep question numbers and labels (Q1, 2(a), Ans 3 ...) and any Name / Roll No header. Write maths in plain text or LaTeX. Do not correct, translate, summarise or add anything. Output only the transcription.",
 "printed": "Transcribe all the text on this page exactly, in reading order, preserving question numbers, marks in brackets and tables as plain text. Output only the transcription.",
}

def main():
    model_dir, jobs_path, out_path, vis_tokens = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    jobs = json.load(open(jobs_path, encoding="utf-8"))
    cuda = torch.cuda.is_available()
    kw = {}
    if cuda:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.float16,
                                                       bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": 0}
        kw["torch_dtype"] = torch.float16
    else:
        kw["torch_dtype"] = torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir, **kw).eval()
    proc = AutoProcessor.from_pretrained(model_dir, min_pixels=256 * 28 * 28, max_pixels=vis_tokens * 28 * 28)

    def transcribe(img, kind, scale):
        if scale < 1:
            img = img.resize((int(img.width * scale), int(img.height * scale)))
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPTS[kind]}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=1536, do_sample=False, repetition_penalty=1.05)
        return proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

    results = {}
    for i, j in enumerate(jobs):
        img = Image.open(j["path"]).convert("RGB")
        text = ""
        for scale in (1.0, 0.7, 0.5):
            try:
                text = transcribe(img, j["kind"], scale)
                break
            except torch.cuda.OutOfMemoryError:
                gc.collect(); torch.cuda.empty_cache()
        results[j["id"]] = text
        json.dump(results, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
        print("PROGRESS %d/%d" % (i + 1, len(jobs)), flush=True)

main()
"""
OCR_WORKER_PATH = BASE_DIR / "ocr_worker.py"
OCR_WORKER_PATH.write_text(OCR_WORKER, encoding="utf-8")

def run_ocr(jobs, progress_cb=None):
    # jobs: [{"id":..., "path":..., "kind": "answer"|"printed"}]  ->  {id: text}
    if not jobs:
        return {}
    run_dir = WORK_DIR / f"ocr_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    jp, op = run_dir / "jobs.json", run_dir / "out.json"
    jp.write_text(json.dumps(jobs), encoding="utf-8")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen([sys.executable, str(OCR_WORKER_PATH), str(OCR_DIR), str(jp), str(op),
                             str(OCR_MAX_VISUAL_TOKENS)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, encoding="utf-8")
    tail = []
    for line in proc.stdout:
        tail.append(line.rstrip()); tail = tail[-25:]
        if line.startswith("PROGRESS") and progress_cb:
            a, b = line.split()[1].split("/")
            progress_cb(int(a) / int(b))
    proc.wait()
    if proc.returncode != 0 or not op.exists():
        raise RuntimeError("OCR process failed:\n" + "\n".join(tail))
    return json.loads(op.read_text(encoding="utf-8"))   # GPU is free again at this point
print("OCR worker written to", OCR_WORKER_PATH)


# ========================================================================
# Source notebook code cell 12
# ========================================================================

import urllib.request, urllib.error, socket, atexit
try:
    from llama_cpp import Llama
except Exception:
    Llama = None

def parse_json(text):
    if not text:
        return None
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    s, e = t.find("{"), t.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(t[s:e + 1])
        except Exception:
            return None
    return None

_NOPROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # never route localhost via a proxy

def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p

class LocalLLM:
    def __init__(self):
        self.model = self.proc = self.port = None
        self.backend, self.key, self.n_ctx, self.ngl = None, None, 0, 0

    def load(self, key):
        self.unload()
        time.sleep(1.5)                                   # let Windows release the OCR process' VRAM
        p = LLM_PRESETS[key]; path = llm_path(key)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run the download cell first.")
        size_mb = path.stat().st_size / 2**20
        g = gpu_mem()
        if g is None:
            ngl = 0
        else:
            per_layer = size_mb / p["layers"]
            fit = int((g["free"] - VRAM_RESERVE_MB) // per_layer)
            ngl = -1 if fit >= p["layers"] + 1 else max(0, fit)
        self.n_ctx, self.ngl, self.key = p["n_ctx"], ngl, key
        threads = max(1, (os.cpu_count() or 8) // 2)
        exe = find_llama_server()
        if LLM_BACKEND == "server" or (LLM_BACKEND == "auto" and exe):
            if not exe:
                raise FileNotFoundError("llama-server.exe missing - run the download cell (Step 2).")
            self.backend, self.port = "llama-server", _free_port()
            log = open(WORK_DIR / "llama_server.log", "w")
            cmd = [str(exe), "-m", str(path), "-c", str(self.n_ctx), "-ngl", str(999 if ngl == -1 else ngl),
                   "-np", "1", "-t", str(threads), "--host", "127.0.0.1", "--port", str(self.port)]
            self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(exe.parent),
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            t0 = time.time()
            while True:
                if self.proc.poll() is not None:
                    tail = (WORK_DIR / "llama_server.log").read_text(errors="ignore")[-1500:]
                    raise RuntimeError("llama-server exited early:\n" + tail)
                try:
                    if _NOPROXY.open(f"http://127.0.0.1:{self.port}/health", timeout=3).status == 200:
                        break
                except Exception:
                    pass
                if time.time() - t0 > 300:
                    self.unload(); raise TimeoutError("llama-server did not become ready in 5 min (see llama_server.log).")
                time.sleep(1)
        else:
            if Llama is None:
                raise RuntimeError("Neither llama-server.exe nor llama-cpp-python is available. Run Step 2.")
            self.backend = "llama-cpp-python"
            self.model = Llama(model_path=str(path), n_gpu_layers=ngl, n_ctx=self.n_ctx, n_batch=512,
                               n_threads=threads, flash_attn=True, verbose=False)
        return (f"LLM '{key}' | backend: {self.backend} | GPU layers: {'all' if ngl == -1 else ngl}/{p['layers']} | "
                f"ctx {self.n_ctx} | free VRAM before load: {g['free'] if g else 'n/a'} MB")

    def unload(self):
        if self.proc is not None:
            try:
                self.proc.terminate(); self.proc.wait(timeout=15)
            except Exception:
                try: self.proc.kill()
                except Exception: pass
            self.proc = None
        if self.model is not None:
            try: self.model.close()
            except Exception: pass
            self.model = None
        gc.collect()

    def clip(self, text, reserve_tokens=1500):
        limit = max(500, (self.n_ctx - reserve_tokens) * 3)
        return text if len(text) <= limit else text[:limit] + "\n...[truncated]"

    def _complete(self, system, user, max_tokens, temperature):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.backend == "llama-cpp-python":
            out = self.model.create_chat_completion(messages=msgs, temperature=temperature, top_p=0.9, max_tokens=max_tokens,
                                                    response_format={"type": "json_object"})
            return out["choices"][0]["message"]["content"]
        for use_fmt in (True, False):
            body = {"messages": msgs, "temperature": temperature, "top_p": 0.9, "max_tokens": max_tokens, "stream": False}
            if use_fmt:
                body["response_format"] = {"type": "json_object"}
            req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            try:
                return json.loads(_NOPROXY.open(req, timeout=900).read())["choices"][0]["message"]["content"]
            except urllib.error.HTTPError:
                if not use_fmt:
                    raise

    def chat_json(self, system, user, max_tokens=900, retries=1):
        for attempt in range(retries + 1):
            data = parse_json(self._complete(system, user, max_tokens, 0.1 if attempt == 0 else 0.3))
            if data is not None:
                return data
        return None

LLM = LocalLLM()
atexit.register(LLM.unload)
print("LLM manager ready.")


# ========================================================================
# Source notebook code cell 14
# ========================================================================

PARSE_SYS = (
 "You convert an exam question paper (plus an optional marking scheme) into structured JSON. Return ONLY JSON of the form "
 "{\"questions\":[{\"id\":\"1\",\"text\":\"...\",\"max_marks\":5,\"rubric\":[{\"criterion\":\"...\",\"marks\":2}],\"model_answer\":\"...\"}]}. "
 "Rules: keep the printed numbering ('1', '2a', '3(b)' -> '3b'); a sub-part that carries its own marks is its own question; "
 "if a marking scheme is supplied copy its criteria and marks; otherwise write 2-5 concise rubric criteria whose marks sum exactly to max_marks "
 "(e.g. conceptual correctness, key points, method/steps, final answer) and a short model_answer; max_marks=null if the marks are not stated; "
 "never invent questions; ignore general instructions and headers.")

def _norm_id(x):
    s = re.sub(r"[^0-9a-z]", "", str(x).lower())
    return re.sub(r"^(?:question|ques|que|answer|ans|q)(?=\d)", "", s)       # "Q2(a)" -> "2a"

def parse_question_paper(qp_text, rubric_text, default_marks=5):
    chunks, cur = [], ""
    for line in qp_text.splitlines():
        if len(cur) + len(line) > 3500 and cur:
            chunks.append(cur); cur = ""
        cur += line + "\n"
    if cur.strip():
        chunks.append(cur)
    questions, seen = [], set()
    for ch in chunks:
        user = "QUESTION PAPER:\n" + LLM.clip(ch, 2500)
        if rubric_text:
            user += "\n\nMARKING SCHEME (authoritative - copy its criteria and marks for the matching questions):\n" + rubric_text[:3000]
        data = LLM.chat_json(PARSE_SYS, user, max_tokens=1800) or {}
        for q in data.get("questions", []):
            qid = _norm_id(q.get("id", ""))
            if not qid or qid in seen or not str(q.get("text", "")).strip():
                continue
            seen.add(qid)
            mm = q.get("max_marks")
            try:
                mm = float(mm) if mm not in (None, "") else float(default_marks)
            except Exception:
                mm = float(default_marks)
            rub = []
            for r in q.get("rubric") or []:
                try:
                    rub.append({"criterion": str(r["criterion"]), "marks": float(r["marks"])})
                except Exception:
                    pass
            tot = sum(r["marks"] for r in rub)
            if not rub or tot <= 0:
                rub = [{"criterion": "Overall correctness, completeness and relevance", "marks": mm}]
            elif abs(tot - mm) > 1e-6:
                rub = [{"criterion": r["criterion"], "marks": round(r["marks"] * mm / tot, 2)} for r in rub]
            questions.append({"id": qid, "text": str(q["text"]).strip(), "max_marks": mm,
                              "rubric": rub, "model_answer": str(q.get("model_answer", "")).strip()})
    if not questions:
        raise ValueError("No questions could be parsed. Check the question paper, or paste the questions JSON manually in the UI.")
    return questions

# ---------------- answer segmentation ----------------
def _marker_regex(qid):
    m = re.match(r"^(\d+)([a-z]|[ivx]+)?$", qid)
    if not m:
        return re.compile(r"(?im)^[ \t]*" + re.escape(qid) + r"[ \t]*[\.\):\-]")
    n, sub = m.group(1), m.group(2)
    prefix = r"(?:(?:q(?:uestion)?|ans(?:wer)?)[ \t]*\.?[ \t]*(?:no\.?)?[ \t]*)"
    if sub:    # a sub-part letter must be explicit: 2(a), 2 (a), 2a, 2.a  - never the first letter of a word ("Q2. Block")
        core = r"\(?" + n + r"\)?(?:[ \t]*\(" + sub + r"\)|[\.\-]?" + sub + r"(?![a-z0-9]))"
        return re.compile(r"(?im)^[ \t]*(?:" + prefix + r")?" + core + r"[ \t]*[\.\):\-]?(?=[ \t\r\n]|$)")
    core = r"\(?" + n + r"\)?"
    return re.compile(r"(?im)^[ \t]*(?:" + prefix + core + r"[ \t]*[\.\):\-]?|" + core + r"[ \t]*[\.\):\-])(?!\d)")

def regex_segment(text, qids):
    found, pos = [], 0
    for qid in qids:
        m = _marker_regex(qid).search(text, pos)
        if m:
            found.append((qid, m.start(), m.end()))
            pos = m.end()
    ans = {q: "" for q in qids}
    for i, (qid, s, e) in enumerate(found):
        end = found[i + 1][1] if i + 1 < len(found) else len(text)
        ans[qid] = text[e:end].strip()
    return ans, len(found)

SEG_SYS = ("You split an OCR transcript of a student's answer script into answers per question. Return ONLY JSON "
           "{\"answers\":{\"<question id>\":\"<student's text for that question, copied verbatim>\"}}. "
           "Use the question list to decide which text answers which question (students may answer out of order or skip numbers). "
           "Use \"\" for unanswered questions. Never rewrite, fix or add text.")

def segment_answers(text, questions, student_text_limit=None):
    qids = [q["id"] for q in questions]
    ans, n_found = regex_segment(text, qids)
    method = "regex"
    if n_found < max(1, int(0.6 * len(qids))):          # markers unreliable -> ask the LLM
        qlist = "\n".join(f"{q['id']}: {q['text'][:200]}" for q in questions)
        data = LLM.chat_json(SEG_SYS, f"QUESTIONS:\n{qlist}\n\nTRANSCRIPT:\n{LLM.clip(text, 2500)}", max_tokens=1800) or {}
        llm_ans = {_norm_id(k): str(v) for k, v in (data.get("answers") or {}).items()}
        if any(v.strip() for v in llm_ans.values()):
            ans = {q: llm_ans.get(q, "") for q in qids}
            method = "llm"
    return ans, method

def extract_metadata(text):
    name = re.search(r"(?im)\bname\s*[:\-]\s*([A-Za-z][A-Za-z .'-]{1,40}?)(?=\s{2,}|\s+(?:roll|reg|id)\b|$)", text)
    roll = re.search(r"(?im)\b(?:roll|reg(?:istration)?|enrol\w*|student)\s*(?:no\.?|number|id)?\s*[:\-]\s*([A-Za-z0-9/\-]{3,20})", text)
    return (name.group(1).strip() if name else ""), (roll.group(1).strip() if roll else "")
# ---------- rule-based question + marking-scheme parsing (fast, exact; the LLM is only a fallback) ----------
_MARKS_PATS = [re.compile(p, re.I) for p in (
    r"[\(\[]\s*(\d+(?:\.\d+)?)\s*(?:marks?|mks?)\s*[\)\]]",
    r"\[\s*(\d+(?:\.\d+)?)\s*\]\s*$",
    r"(?:^|[\s,\-])(\d+(?:\.\d+)?)\s*marks?\b")]
def _find_marks(txt):
    for p in _MARKS_PATS:
        found = list(p.finditer(txt))
        if found:
            return float(found[-1].group(1))
    return None

def _strip_marks(txt):
    txt = _MARKS_PATS[0].sub("", txt)
    return re.sub(r"\s+", " ", txt).strip()

_QSTART = re.compile(r"(?im)^[ \t]*(?:q(?:uestion)?\.?[ \t]*)?(\d{1,2})[ \t]*[\.\):\-][ \t]*(?=\S)")

def regex_parse_questions(text, default_marks=5):
    starts, expect = [], None
    for m in _QSTART.finditer(text):
        n = int(m.group(1))
        if (expect is None and n == 1) or (expect is not None and n == expect):
            starts.append((n, m.start(), m.end())); expect = n + 1
    if not starts:
        return None
    qs = []
    for i, (n, s0, e0) in enumerate(starts):
        body = text[e0:(starts[i + 1][1] if i + 1 < len(starts) else len(text))].strip()
        subs = list(re.finditer(r"(?im)^[ \t]*\(([a-z])\)[ \t]*", body))
        letters = [m.group(1) for m in subs]
        parts = []
        if len(subs) >= 2 and letters == [chr(97 + k) for k in range(len(letters))]:
            stem = body[:subs[0].start()].strip()
            for k, m in enumerate(subs):
                ptxt = body[m.end():(subs[k + 1].start() if k + 1 < len(subs) else len(body))].strip()
                parts.append((m.group(1), (stem + " " + ptxt).strip(), _find_marks(ptxt)))
        if parts and all(p[2] is not None for p in parts):          # sub-parts that carry their own marks
            for letter, ptxt, mk in parts:
                qs.append({"id": f"{n}{letter}", "text": _strip_marks(ptxt), "max_marks": mk, "rubric": [], "model_answer": ""})
        else:
            mk = _find_marks(body)
            qs.append({"id": str(n), "text": _strip_marks(body), "max_marks": mk if mk is not None else float(default_marks),
                       "rubric": [], "model_answer": ""})
    return qs

def declared_total(text):
    m = re.search(r"(?i)(?:max(?:imum)?\.?|total)\s*marks?\s*[:=\-]?\s*(\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else None

_HDR = re.compile(r"^\s*(?:q(?:uestion)?\.?\s*)?(\d{1,2})\s*(?:[\.\-]?\s*\(?([a-z])\)?(?![a-z]))?\s*[\.\):\-]?\s*(?:\(?\s*(\d+(?:\.\d+)?)\s*marks?\s*\)?)?\s*[:\-]?\s*$", re.I)
_BUL = r"(?:[-•*–·]+|\d+[\.\)]|\(?[a-z]\))?"
_CRIT1 = re.compile(r"^\s*" + _BUL + r"\s*(.+?)\s*[:=]\s*(\d+(?:\.\d+)?)\s*(?:marks?|m)?\s*$", re.I)
_CRIT2 = re.compile(r"^\s*" + _BUL + r"\s*(.+?)\s*\(\s*(\d+(?:\.\d+)?)\s*(?:marks?|m)\s*\)\s*$", re.I)

def regex_parse_scheme(text, qids):
    out, cur = {}, None
    for line in text.splitlines():
        if not line.strip():
            continue
        h = _HDR.match(line)
        if h:
            qid = h.group(1) + (h.group(2) or "").lower()
            if qid in qids:
                cur = qid; out.setdefault(cur, []); continue
        c = _CRIT1.match(line) or _CRIT2.match(line)
        if c and cur and not re.match(r"(?i)^(total|max|maximum|marks?)\b", c.group(1).strip()):
            out[cur].append({"criterion": c.group(1).strip(" -•*"), "marks": float(c.group(2))})
    return {k: v for k, v in out.items() if v}

SCHEME_SYS = ("You extract rubric criteria for ONE exam question from a teacher's marking scheme. Return ONLY JSON "
              "{\"rubric\":[{\"criterion\":\"...\",\"marks\":number}]}. Copy the criteria and marks from the marking scheme wording; "
              "never invent criteria. If the scheme has no entry for this question return {\"rubric\":[]}.")
GEN_SYS = ("You design marking rubrics. Return ONLY JSON {\"rubric\":[{\"criterion\":\"...\",\"marks\":number}],\"model_answer\":\"...\"} "
           "for the given exam question. Write 2-4 specific, checkable criteria (key concept, key points, method/steps, final answer with unit - as relevant) "
           "whose marks sum EXACTLY to the question's marks, and a concise correct model answer.")

def _clean_rubric(lst):
    out = []
    for r in lst or []:
        try:
            out.append({"criterion": str(r["criterion"]).strip(), "marks": float(r["marks"])})
        except Exception:
            pass
    return [r for r in out if r["criterion"] and r["marks"] > 0]

def build_rubrics(questions, rub_text):
    warns, how = [], {}
    scheme = regex_parse_scheme(rub_text, {q["id"] for q in questions}) if rub_text.strip() else {}
    for q in questions:
        rub = scheme.get(q["id"]); h = "scheme"
        if not rub and rub_text.strip():
            d = LLM.chat_json(SCHEME_SYS, f"QUESTION {q['id']} ({q['max_marks']} marks): {q['text']}\n\nMARKING SCHEME:\n{LLM.clip(rub_text, 2500)}", max_tokens=500) or {}
            rub = _clean_rubric(d.get("rubric")); h = "scheme"
        if not rub:
            d = LLM.chat_json(GEN_SYS, f"QUESTION {q['id']} ({q['max_marks']} marks): {q['text']}", max_tokens=700) or {}
            rub = _clean_rubric(d.get("rubric")); q["model_answer"] = str(d.get("model_answer", "")).strip(); h = "auto"
        if not rub:
            rub = [{"criterion": "Overall correctness, completeness and relevance", "marks": q["max_marks"]}]; h = "auto"
        tot = sum(r["marks"] for r in rub)
        if abs(tot - q["max_marks"]) > 0.01:
            warns.append(f"Q{q['id']}: rubric marks add up to {tot:g} but the question carries {q['max_marks']:g} - rubric was rescaled; please check.")
            rub = [{"criterion": r["criterion"], "marks": round(r["marks"] * q["max_marks"] / tot, 2)} for r in rub]
        q["rubric"], how[q["id"]] = rub, h
    auto = [i for i, h in how.items() if h == "auto"]
    if not rub_text.strip():
        label = "AUTO-GENERATED by the AI from the question paper (no marking scheme supplied) - please verify"
    elif auto:
        label = "Teacher-provided marking scheme; AUTO-GENERATED by the AI for Q" + ", Q".join(auto) + " (not found in the scheme) - please verify"
    else:
        label = "Teacher-provided marking scheme"
    return warns, label

def parse_all(qp_text, rub_text, default_marks=5):
    warns = []
    qs = regex_parse_questions(qp_text, default_marks)
    if not qs:
        qs = parse_question_paper(qp_text, rub_text, default_marks)
        warns.append("Question-paper layout was not recognised by the rules, so the AI split it into questions - please check the list below.")
    w2, label = build_rubrics(qs, rub_text)
    declared, total = declared_total(qp_text), sum(q["max_marks"] for q in qs)
    if declared and abs(declared - total) > 0.01:
        warns.append(f"Question marks add up to {total:g} but the paper says Max Marks {declared:g} - check the question list / marks.")
    return qs, label, warns + w2

def clean_answer_text(text):
    keep = []
    for ln in text.splitlines():
        if re.match(r"(?i)^\s*(name|roll|reg(?:istration)?\b|date|class|subject)\b.*$", ln):
            continue
        if re.search(r"(?i)\bcontinued\b|\bcont'?d\b", ln) and len(ln) < 60:
            continue
        keep.append(ln)
    return re.sub(r"\(\s*\d+(?:\.\d+)?\s*marks?\s*\)", "", "\n".join(keep), flags=re.I)
print("Parsing + segmentation ready.")


# ========================================================================
# Source notebook code cell 16
# ========================================================================


CRIT_SYS = (
 "You are a strict but fair examiner checking ONE rubric criterion against a student's answer. The answer is an OCR transcript of handwriting "
 "(ignore spelling/OCR noise). Decide the marks this criterion deserves: full marks if it is fully satisfied, partial marks if partly satisfied, 0 if absent or wrong. "
 "Judge by meaning, not keywords. Accept valid alternative methods and wording. For maths, give credit for correct steps and for consistent follow-through, and require the "
 "correct final value/unit only if the criterion asks for it. Quote the words of the student's answer that justify your decision (or \"none\"). "
 "Return ONLY JSON: {\"evidence\":\"quote from the student's answer or none\",\"awarded\":number,\"reason\":\"one short sentence\"}.")

SUM_SYS = (
 "You write the justification for marks that were ALREADY decided per criterion. Return ONLY JSON "
 "{\"rationale\":\"2-4 sentences explaining the marks, fully consistent with the per-criterion results (never contradict them)\","
 "\"feedback\":\"2-3 sentences addressed to the student: what was good, what was missing or wrong, and how to improve\"}.")

def _round_step(x, step=MARK_STEP):
    return round(x / step) * step

def judge_criterion(q, crit, answer):
    ref = f"\n\nREFERENCE ANSWER (for guidance only):\n{q['model_answer']}" if q.get("model_answer") else ""
    user = (f"QUESTION ({q['max_marks']} marks):\n{q['text']}\n\nCRITERION TO CHECK (max {crit['marks']} marks):\n{crit['criterion']}{ref}\n\n"
            f"STUDENT ANSWER (OCR transcript):\n{LLM.clip(answer, 2500)}")
    d = LLM.chat_json(CRIT_SYS, user, max_tokens=250) or {}
    try:
        aw = min(max(float(d["awarded"]), 0.0), crit["marks"])
    except Exception:
        return None, d
    return aw, d

def evaluate_answer(q, answer):
    answer = (answer or "").strip()
    item = {"qid": q["id"], "question": q["text"], "max_marks": q["max_marks"], "answer": answer,
            "ai_marks": 0.0, "teacher_marks": None, "rationale": "", "feedback": "", "criteria": [], "needs_review": False}
    if len(answer) < 3:
        item.update(rationale="No answer detected for this question (check that the question number is written clearly in the margin).",
                    feedback="This question appears to be unanswered. Attempt every question; even partial working earns marks.",
                    needs_review=True)
        return item
    crit_out, total, failed = [], 0.0, False
    for r in q["rubric"]:
        aw, d = judge_criterion(q, r, answer)
        if aw is None:
            failed, aw = True, 0.0
        total += aw
        crit_out.append({"criterion": r["criterion"], "max": r["marks"], "awarded": round(aw, 2),
                         "comment": str(d.get("reason", "")).strip(), "evidence": str(d.get("evidence", "")).strip()})
    item["criteria"] = crit_out
    item["ai_marks"] = min(q["max_marks"], max(0.0, _round_step(total)))
    lines = "\n".join(f"- {c['criterion']}: {c['awarded']}/{c['max']} ({c['comment']})" for c in crit_out)
    s = LLM.chat_json(SUM_SYS, f"QUESTION: {q['text']}\n\nSTUDENT ANSWER:\n{LLM.clip(answer, 2000)}\n\nMARKS: {item['ai_marks']}/{q['max_marks']}\nCRITERIA RESULTS:\n{lines}", max_tokens=400) or {}
    item["rationale"] = str(s.get("rationale", "")).strip()
    item["feedback"] = str(s.get("feedback", "")).strip()
    item["needs_review"] = failed or len(answer) < 15 or not item["rationale"] or (item["ai_marks"] == 0 and len(answer) >= 40)
    return item

OVERALL_SYS = ("You write a short overall feedback note (3-4 sentences) for a student, based on their question-wise results. "
               "Mention strengths, the main weaknesses and one concrete study suggestion. Use ONLY the information given. Return ONLY JSON {\"overall_feedback\":\"...\"}.")

def overall_feedback(items):
    lines = "\n".join(f"Q{i['qid']}: {i['ai_marks']}/{i['max_marks']} - {i['rationale'][:160]}" for i in items)
    data = LLM.chat_json(OVERALL_SYS, lines, max_tokens=300) or {}
    return str(data.get("overall_feedback", "")).strip()
print("Evaluator ready (per-criterion judging).")


# ========================================================================
# Source notebook code cell 18
# ========================================================================

def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

with db() as con:
    con.execute("CREATE TABLE IF NOT EXISTS exams(id TEXT PRIMARY KEY, title TEXT, questions_json TEXT, created TEXT)")
    con.execute("""CREATE TABLE IF NOT EXISTS evals(id INTEGER PRIMARY KEY AUTOINCREMENT, exam_id TEXT, student TEXT,
                   roll TEXT, created TEXT, data_json TEXT)""")

def db_save_exam(eid, title, questions):
    with db() as con:
        con.execute("INSERT OR REPLACE INTO exams VALUES (?,?,?,?)",
                    (eid, title, json.dumps(questions), datetime.datetime.now().isoformat(timespec="seconds")))

def db_save_eval(exam_id, student, roll, rec):
    with db() as con:
        cur = con.execute("INSERT INTO evals(exam_id,student,roll,created,data_json) VALUES (?,?,?,?,?)",
                          (exam_id, student, roll, datetime.datetime.now().isoformat(timespec="seconds"), json.dumps(rec)))
        return cur.lastrowid

def db_update_eval(eid, rec):
    with db() as con:
        con.execute("UPDATE evals SET data_json=? WHERE id=?", (json.dumps(rec), eid))

def db_get_eval(eid):
    with db() as con:
        r = con.execute("SELECT * FROM evals WHERE id=?", (eid,)).fetchone()
    rec = json.loads(r["data_json"]); rec.update(id=r["id"], exam_id=r["exam_id"], student=r["student"], roll=r["roll"])
    return rec

def db_list_evals():
    with db() as con:
        rows = con.execute("SELECT e.id,e.roll,e.student,x.title FROM evals e LEFT JOIN exams x ON x.id=e.exam_id ORDER BY e.id DESC").fetchall()
    return [f"{r['id']} | {r['roll']} | {r['student']} | {r['title']}" for r in rows]

def final_marks(it):
    return it["teacher_marks"] if it.get("teacher_marks") is not None else it["ai_marks"]

def totals(rec):
    return sum(final_marks(i) for i in rec["items"]), sum(i["max_marks"] for i in rec["items"])

def build_html(rec):
    tot, mx = totals(rec)
    e = html.escape
    warn_html = "".join(f"<p style='color:#b00;font-size:11px'><b>Check:</b> {html.escape(w)}</p>" for w in rec.get('warnings', []))
    rows = ""
    for it in rec["items"]:
        crit = "".join(f"<br/>&bull; {e(c['criterion'])}: {c['awarded']}/{c['max']}" for c in it.get("criteria", []))
        tm = "" if it.get("teacher_marks") is None else f" (AI: {it['ai_marks']})"
        rows += (f"<tr><td>{e(str(it['qid']))}</td><td>{e(it['question'])}</td><td>{e(it['answer']).replace(chr(10), '<br/>')}</td>"
                 f"<td>{final_marks(it)} / {it['max_marks']}{tm}</td><td>{e(it['rationale'])}<br/><i>Rubric breakdown:</i>{crit}</td><td>{e(it['feedback'])}</td></tr>")
    return (f"<html><body style='font-family:sans-serif'><h2>Answer Script Evaluation Report</h2>"
            f"<p><b>Student:</b> {e(rec['student'])} &nbsp; <b>Roll/ID:</b> {e(rec['roll'])} &nbsp; <b>Exam:</b> {e(rec.get('exam_title',''))}</p>"
            f"<h3>Total: {tot} / {mx}</h3><p style='font-size:11px'><b>Marking scheme:</b> {e(rec.get('rubric_source','n/a'))}</p>{warn_html}"
            f"<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse;font-size:12px'>"
            f"<tr><th>Q</th><th>Question</th><th>Extracted answer</th><th>Marks</th><th>Evaluation rationale</th><th>Feedback</th></tr>{rows}</table>"
            f"<h3>Overall feedback</h3><p>{e(rec.get('overall_feedback',''))}</p>"
            f"<p style='font-size:10px;color:#666'>Generated locally by an AI system. Marks may be modified by the teacher.</p></body></html>")

def html_to_pdf(html_str, out):
    story = fitz.Story(html=html_str)
    writer = fitz.DocumentWriter(str(out))
    mediabox = fitz.paper_rect("a4-l")           # landscape for the wide table
    where = mediabox + (30, 30, -30, -30)
    more = 1
    while more:
        dev = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()

def write_reports(eid):
    rec = db_get_eval(eid)
    with db() as con:
        t = con.execute("SELECT title FROM exams WHERE id=?", (rec["exam_id"],)).fetchone()
    rec["exam_title"] = t["title"] if t else ""
    h = build_html(rec)
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", f"{rec['roll'] or 'NA'}_{rec['student'] or 'student'}_{eid}")
    files = []
    hp = REPORT_DIR / f"{safe}.html"; hp.write_text(h, encoding="utf-8"); files.append(str(hp))
    try:
        pp = REPORT_DIR / f"{safe}.pdf"; html_to_pdf(h, pp); files.append(str(pp))
    except Exception as ex:
        print("PDF export skipped:", ex)
    import pandas as pd
    df = pd.DataFrame([{"roll": rec["roll"], "student": rec["student"], "qid": i["qid"], "max_marks": i["max_marks"],
                        "ai_marks": i["ai_marks"], "final_marks": final_marks(i), "rationale": i["rationale"],
                        "feedback": i["feedback"]} for i in rec["items"]])
    cp = REPORT_DIR / f"{safe}_marksheet.csv"; df.to_csv(cp, index=False); files.append(str(cp))
    return h, files
print("Database:", DB_PATH)


# ========================================================================
# Source notebook code cell 20
# ========================================================================

RUBRIC_MODES = ["Use my marking scheme; if none is given, auto-generate one from the question paper",
                "Marking scheme is mandatory (stop if none is given)"]

def _stage_file(path, tag, run_dir, jobs):
    # Returns text if the file is digital; otherwise queues its pages for OCR and returns None.
    ext = Path(path).suffix.lower()
    if ext in (".txt", ".md"):
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    if ext == ".pdf":
        t = pdf_text_if_digital(path)
        if t:
            return t
    for k, pg in enumerate(load_pages(path)):
        pth = run_dir / f"{tag}_{k:03d}.png"
        preprocess(pg).save(pth)
        jobs.append({"id": f"{tag}_{k:03d}", "path": str(pth), "kind": "printed"})
    return None

def _ocr_text(ocr, tag):
    return "\n".join(ocr[k] for k in sorted(ocr) if k.startswith(tag + "_"))

def _prepare_qp(qp_file, rubric_file, rubric_text, run_dir, jobs):
    qp_text = _stage_file(to_path(qp_file), "qp", run_dir, jobs) if qp_file is not None else ""
    rub_text, rub_pending = (rubric_text or "").strip(), False
    if rubric_file is not None:
        t = _stage_file(to_path(rubric_file), "rub", run_dir, jobs)
        if t is None: rub_pending = True
        else: rub_text = (rub_text + "\n" + t).strip()
    return qp_text, rub_text, rub_pending

def normalize_questions(qs, default_marks=5):
    out = []
    for q in qs:
        qid = _norm_id(q.get("id", ""))
        if not qid or not str(q.get("text", "")).strip():
            continue
        try: mm = float(q.get("max_marks"))
        except Exception: mm = float(default_marks)
        rub = []
        for r in q.get("rubric") or []:
            try: rub.append({"criterion": str(r["criterion"]), "marks": float(r["marks"])})
            except Exception: pass
        tot = sum(r["marks"] for r in rub)
        if not rub or tot <= 0:
            rub = [{"criterion": "Overall correctness, completeness and relevance", "marks": mm}]
        elif abs(tot - mm) > 1e-6:
            rub = [{"criterion": r["criterion"], "marks": round(r["marks"] * mm / tot, 2)} for r in rub]
        out.append({"id": qid, "text": str(q["text"]).strip(), "max_marks": mm, "rubric": rub,
                    "model_answer": str(q.get("model_answer", "")).strip()})
    return out

def build_questions(qp_text, rub_text, mode, default_marks, override_json):
    if (override_json or "").strip():
        data = json.loads(override_json)
        qs = normalize_questions(data["questions"] if isinstance(data, dict) else data, default_marks)
        if not qs:
            raise ValueError("The questions/rubric JSON has no valid questions.")
        d = data if isinstance(data, dict) else {}
        return qs, d.get("rubric_source", "Teacher-edited rubric"), d.get("warnings", [])
    if not (qp_text or "").strip():
        raise ValueError("No text could be read from the question paper.")
    if not rub_text.strip() and mode.startswith("Marking scheme is mandatory"):
        raise ValueError("No marking scheme supplied (upload one or paste it), or switch the option to 'auto-generate'.")
    return parse_all(qp_text, rub_text, default_marks)

def parse_only(qp_file, rubric_file, rubric_text, mode, default_marks, llm_key, progress=lambda f, m="": None):
    run_dir = WORK_DIR / f"parse_{int(time.time())}"; run_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    progress(0.05, "Reading question paper / marking scheme...")
    qp_text, rub_text, rub_pending = _prepare_qp(qp_file, rubric_file, rubric_text, run_dir, jobs)
    ocr = run_ocr(jobs, lambda f: progress(0.1 + 0.4 * f, "OCR of scanned paper/scheme...")) if jobs else {}
    if qp_text is None: qp_text = _ocr_text(ocr, "qp")
    if rub_pending: rub_text = (rub_text + "\n" + _ocr_text(ocr, "rub")).strip()
    progress(0.55, "Loading LLM...")
    LLM.load(llm_key)
    try:
        progress(0.7, "Parsing questions + rubric...")
        qs, src, warns = build_questions(qp_text, rub_text, mode, default_marks, "")
    finally:
        LLM.unload()
    return {"rubric_source": src, "warnings": warns, "questions": qs}

def run_pipeline(qp_file, rubric_file, rubric_text, rubric_mode, answer_files, student, roll, llm_key,
                 default_marks=5, override_questions_json="", progress=lambda f, m="": None):
    t0 = time.time()
    run_dir = WORK_DIR / f"run_{int(t0)}"; run_dir.mkdir(parents=True, exist_ok=True)
    jobs, have_override = [], bool((override_questions_json or "").strip())
    progress(0.02, "Pre-processing images...")
    qp_text, rub_text, rub_pending = ("", "", False) if have_override else _prepare_qp(qp_file, rubric_file, rubric_text, run_dir, jobs)
    for fi, f in enumerate(answer_files or []):
        for k, pg in enumerate(load_pages(to_path(f))):
            pth = run_dir / f"ans_{fi:02d}{k:03d}.png"
            preprocess(pg).save(pth)
            jobs.append({"id": f"ans_{fi:02d}{k:03d}", "path": str(pth), "kind": "answer"})
    if not any(j["id"].startswith("ans_") for j in jobs):
        raise ValueError("Please upload at least one answer-sheet image/PDF.")

    # ---------- Stage A: OCR in its own process (GPU is fully released when it exits) ----------
    progress(0.08, f"OCR on {len(jobs)} page(s) - GPU is used by the OCR model...")
    ocr = run_ocr(jobs, lambda f: progress(0.08 + 0.42 * f, "OCR running..."))
    gc.collect()
    ans_text = _ocr_text(ocr, "ans")
    if qp_text is None: qp_text = _ocr_text(ocr, "qp")
    if rub_pending: rub_text = (rub_text + "\n" + _ocr_text(ocr, "rub")).strip()

    # ---------- Stage B: LLM with (almost) the whole GPU ----------
    progress(0.52, "Starting LLM on the freed GPU...")
    info = LLM.load(llm_key); print(info)
    try:
        progress(0.58, "Preparing question paper + marking scheme...")
        questions, rubric_source, warns = build_questions(qp_text, rub_text, rubric_mode, default_marks, override_questions_json or "")
        exam_id = hashlib.sha1(json.dumps(questions, sort_keys=True).encode()).hexdigest()[:8]
        title = Path(to_path(qp_file)).stem if qp_file is not None else "manual-questions"
        db_save_exam(exam_id, title, questions)

        n1, r1 = extract_metadata(ans_text)
        student, roll = (student or "").strip() or n1, (roll or "").strip() or r1
        progress(0.65, "Segmenting answers & matching them to questions...")
        answers, method = segment_answers(clean_answer_text(ans_text), questions)

        items = []
        for i, q in enumerate(questions):
            progress(0.68 + 0.27 * i / len(questions), f"Evaluating Q{q['id']} ({i+1}/{len(questions)})...")
            items.append(evaluate_answer(q, answers.get(q["id"], "")))
        progress(0.96, "Writing overall feedback...")
        overall = overall_feedback(items)
    finally:
        LLM.unload()                                  # GPU is free again

    rec = {"items": items, "overall_feedback": overall, "segmentation": method,
           "rubric_source": rubric_source, "warnings": warns, "full_ocr_text": ans_text}
    eid = db_save_eval(exam_id, student, roll, rec)
    h, files = write_reports(eid)
    progress(1.0, "Done")
    return dict(eval_id=eid, student=student, roll=roll, rec=db_get_eval(eid), html=h, files=files,
                questions=questions, rubric_source=rubric_source, warnings=warns, seconds=round(time.time() - t0, 1), llm_info=info)
print("Pipeline ready.")


# ========================================================================
# Source notebook code cell 22
# ========================================================================

import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

def all_ai_rows():
    rows = []
    with db() as con:
        for r in con.execute("SELECT id,roll,exam_id,data_json FROM evals").fetchall():
            for it in json.loads(r["data_json"])["items"]:
                rows.append({"eval_id": r["id"], "roll": str(r["roll"]).strip(), "exam_id": r["exam_id"], "qid": _norm_id(it["qid"]),
                             "max_marks": it["max_marks"], "ai_marks": it["ai_marks"], "final_marks": final_marks(it)})
    return pd.DataFrame(rows)

def _metrics(h, a, mx, label):
    h, a, mx = np.asarray(h, float), np.asarray(a, float), np.asarray(mx, float)
    n = len(h)
    out = {"level": label, "n": n, "MAE": round(float(np.mean(np.abs(h - a))), 3),
           "RMSE": round(float(np.sqrt(np.mean((h - a) ** 2))), 3),
           "MAE % of max": round(float(np.mean(np.abs(h - a) / mx) * 100), 2),
           "exact agreement %": round(float(np.mean(np.isclose(h, a)) * 100), 1),
           "within 0.5 mark %": round(float(np.mean(np.abs(h - a) <= 0.5 + 1e-9) * 100), 1),
           "within 1 mark %": round(float(np.mean(np.abs(h - a) <= 1 + 1e-9) * 100), 1)}
    try:
        out["Pearson r"] = round(float(pearsonr(h, a)[0]), 3) if n > 2 else np.nan
        out["Spearman rho"] = round(float(spearmanr(h, a)[0]), 3) if n > 2 else np.nan
    except Exception:
        out["Pearson r"] = out["Spearman rho"] = np.nan
    try:
        hi, ai = np.rint(h * 2).astype(int), np.rint(a * 2).astype(int)
        out["QWK"] = round(float(cohen_kappa_score(hi, ai, weights="quadratic",
                          labels=list(range(0, int(mx.max() * 2) + 1)))), 3)
    except Exception:
        out["QWK"] = np.nan
    out["mean bias (AI-human)"] = round(float(np.mean(a - h)), 3)
    return out

def agreement(csv_path):
    hum = pd.read_csv(csv_path)
    hum.columns = [c.strip().lower() for c in hum.columns]
    hum["roll"] = hum["roll"].astype(str).str.strip()
    hum["qid"] = hum["qid"].map(_norm_id)
    hum = hum.dropna(subset=["human_marks"])
    ai = all_ai_rows()
    if ai.empty:
        raise ValueError("No AI evaluations stored yet.")
    ai = ai.sort_values("eval_id").drop_duplicates(["roll", "qid"], keep="last")   # use the latest AI run per student/question
    m = hum.merge(ai, on=["roll", "qid"], how="inner")
    if m.empty:
        raise ValueError("No (roll, qid) pairs matched between the human CSV and AI results.")
    rows = [_metrics(m["human_marks"], m["ai_marks"], m["max_marks"], "question-wise")]
    t = m.groupby("roll").agg(h=("human_marks", "sum"), a=("ai_marks", "sum"), mx=("max_marks", "sum")).reset_index()
    if len(t) >= 2:
        rows.append(_metrics(t["h"], t["a"], t["mx"], "total per student"))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(m["human_marks"], m["ai_marks"], alpha=.6); lim = [0, m["max_marks"].max()]
    ax[0].plot(lim, lim, "r--"); ax[0].set_xlabel("Human marks"); ax[0].set_ylabel("AI marks"); ax[0].set_title("Question-wise")
    d = m["ai_marks"] - m["human_marks"]
    ax[1].hist(d, bins=np.arange(-5.25, 5.5, 0.5), edgecolor="k"); ax[1].set_xlabel("AI - human (marks)"); ax[1].set_title("Error distribution")
    fig.tight_layout()
    return pd.DataFrame(rows), fig, m

def human_template():
    ai = all_ai_rows()
    if ai.empty:
        raise ValueError("Evaluate some scripts first.")
    t = ai.sort_values("eval_id").drop_duplicates(["roll", "qid"], keep="last")[["roll", "qid", "max_marks"]].assign(human_marks="")
    p = REPORT_DIR / "human_marks_template.csv"; t.to_csv(p, index=False); return str(p)
print("Agreement metrics ready.")


# ========================================================================
# Source notebook code cell 24
# ========================================================================

import gradio as gr

def ui_models(action, llm_key):
    try:
        if action == "ocr":   return download_ocr()
        if action == "llm":   return download_llm(llm_key)
        if action == "llama": return download_llama_server()
        if action == "gpu":   return json.dumps({"gpu": gpu_mem() or "No NVIDIA GPU found", "llama_server": str(find_llama_server())}, indent=2)
    except Exception as ex:
        return f"Error: {ex}"

def rubric_preview_md(res):
    lines = [f"**Marking scheme source:** {res['rubric_source']}", ""] + [f"⚠️ **{w}**\n" for w in res.get("warnings", [])] + ["| Q | Marks | Question | Rubric criteria |", "|---|---|---|---|"]
    for q in res["questions"]:
        crit = "<br>".join(f"{r['criterion']} — **{r['marks']}**" for r in q["rubric"])
        lines.append(f"| {q['id']} | {q['max_marks']} | {q['text'][:90].replace('|','/')} | {crit.replace('|','/')} |")
    return "\n".join(lines)

def ui_parse(qp, rub, rub_text, rub_mode, dm, llm_key, progress=gr.Progress()):
    try:
        if qp is None:
            return "Upload the question paper first.", "", ""
        res = parse_only(qp, rub, rub_text, rub_mode, dm or 5, llm_key, lambda f, m="": progress(f, desc=m))
        return ("Parsed ✔ — check the table, edit the JSON if needed, then press ② Evaluate.",
                rubric_preview_md(res), json.dumps(res, indent=2, ensure_ascii=False))
    except Exception as ex:
        return f"❌ {ex}\n\n{traceback.format_exc()[-1200:]}", "", ""

def ui_evaluate(qp, rub, rub_text, rub_mode, answers, name, roll, llm_key, default_marks, override_json, progress=gr.Progress()):
    try:
        if qp is None and not (override_json or "").strip():
            return "Upload a question paper (or press ① first).", "", None, "", None
        res = run_pipeline(qp, rub, rub_text, rub_mode, answers, name, roll, llm_key, default_marks or 5, override_json or "",
                           lambda f, m="": progress(f, desc=m))
        rec = res["rec"]; tot, mx = totals(rec)
        df = pd.DataFrame([{"Q": i["qid"], "Max": i["max_marks"], "AI marks": i["ai_marks"], "Extracted answer": i["answer"],
                            "Rationale": i["rationale"], "Feedback": i["feedback"], "Needs review": "yes" if i["needs_review"] else ""}
                           for i in rec["items"]])
        summ = (f"### {res['student'] or '(name not found)'} · {res['roll'] or '(roll not found)'} — **{tot} / {mx}**\n\n"
                f"Evaluation id **{res['eval_id']}** · marking scheme: *{res['rubric_source']}* · segmentation: `{rec['segmentation']}` · {res['seconds']} s\n\n"
                + "".join(f"⚠️ **{w}**\n\n" for w in res.get("warnings", [])) + f"`{res['llm_info']}`\n\n**Overall feedback:** {rec['overall_feedback']}")
        return "Done ✔", summ, df, res["html"], res["files"]
    except Exception as ex:
        return f"❌ {ex}\n\n{traceback.format_exc()[-1500:]}", "", None, "", None

REVIEW_COLS = ["Q", "Max", "AI marks", "Teacher marks", "Answer", "Rationale", "Feedback"]
def review_refresh():
    return gr.update(choices=db_list_evals())

def review_load(label):
    if not label: return None, ""
    rec = db_get_eval(int(label.split("|")[0])); tot, mx = totals(rec)
    df = pd.DataFrame([{"Q": i["qid"], "Max": i["max_marks"], "AI marks": i["ai_marks"],
                        "Teacher marks": "" if i.get("teacher_marks") is None else i["teacher_marks"],
                        "Answer": i["answer"], "Rationale": i["rationale"], "Feedback": i["feedback"]} for i in rec["items"]], columns=REVIEW_COLS)
    return df, f"Current total: **{tot} / {mx}**  ·  marking scheme: *{rec.get('rubric_source', 'n/a')}*"

def review_save(label, df):
    if not label or df is None: return "Select an evaluation first.", None
    eid = int(label.split("|")[0]); rec = db_get_eval(eid)
    for _, r in pd.DataFrame(df).iterrows():
        for it in rec["items"]:
            if str(it["qid"]) == str(r["Q"]):
                tm = str(r["Teacher marks"]).strip()
                it["teacher_marks"] = None if tm in ("", "nan", "None") else min(max(float(tm), 0.0), it["max_marks"])
                it["feedback"] = str(r["Feedback"])
    db_update_eval(eid, {k: rec.get(k, "") for k in ("items", "overall_feedback", "segmentation", "rubric_source", "warnings", "full_ocr_text")})
    _, files = write_reports(eid); tot, mx = totals(db_get_eval(eid))
    return f"Saved ✔  New total: {tot} / {mx}", files

def ui_template():
    try: return human_template()
    except Exception as ex: raise gr.Error(str(ex))

def ui_agreement(csv):
    try:
        t, fig, m = agreement(to_path(csv))
        return t, fig, m
    except Exception as ex:
        raise gr.Error(str(ex))

FT = [".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg"]
with gr.Blocks(title="Local Answer Script Evaluator") as demo:
    gr.Markdown("# 🎓 Automated Answer Script Evaluation — runs 100 % locally\nNo data leaves this computer.")
    with gr.Tab("1 · Models & GPU"):
        llm_dd0 = gr.Dropdown(list(LLM_PRESETS), value=DEFAULT_LLM, label="LLM preset")
        with gr.Row():
            b1 = gr.Button("Check/download OCR model"); b2 = gr.Button("Check/download selected LLM")
            b4 = gr.Button("Check/download llama.cpp"); b3 = gr.Button("Show GPU memory")
        mout = gr.Textbox(label="Status", lines=8)
        b1.click(lambda: ui_models("ocr", None), None, mout); b2.click(lambda k: ui_models("llm", k), llm_dd0, mout)
        b4.click(lambda: ui_models("llama", None), None, mout); b3.click(lambda: ui_models("gpu", None), None, mout)
        gr.Markdown(f"Everything is stored in: `{BASE_DIR}`")
    with gr.Tab("2 · Evaluate"):
        gr.Markdown("### A · Question paper & marking scheme")
        with gr.Row():
            qp = gr.File(label="Question paper (PDF / TXT / photo)", file_types=FT)
            rub = gr.File(label="Marking scheme / rubric file (optional: PDF / TXT / photo)", file_types=FT)
            rub_text = gr.Textbox(label="…or paste the marking scheme here", lines=8,
                                  placeholder="Q1 (4 marks)\n- States Newton's 2nd law: 2\n- Gives F = ma with symbols: 2\nQ2 (6 marks)\n- ...")
        with gr.Row():
            rub_mode = gr.Radio(RUBRIC_MODES, value=RUBRIC_MODES[0], label="If no marking scheme is provided")
            llm_dd = gr.Dropdown(list(LLM_PRESETS), value=DEFAULT_LLM, label="LLM for parsing + evaluation")
            dm = gr.Number(value=5, label="Default marks if a question shows none")
        parse_btn = gr.Button("① Parse question paper + marking scheme (preview & edit)", variant="secondary")
        pst = gr.Textbox(label="Status", lines=2); pprev = gr.Markdown()
        ov = gr.Code(language="json", label="Parsed questions + rubric (editable). Whatever is here is used by ②. Keep it for the next student of the same exam.", value="")
        gr.Markdown("### B · Student & answer script")
        with gr.Row():
            ans = gr.File(label="Student answer script (photos / scans / PDF — all pages)", file_count="multiple",
                          file_types=[".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".pdf"])
            with gr.Column():
                nm = gr.Textbox(label="Student name (optional - auto-read from sheet if empty)")
                rl = gr.Textbox(label="Roll number / ID")
        go = gr.Button("② Evaluate answer script", variant="primary")
        st = gr.Textbox(label="Status"); sm = gr.Markdown()
        tbl = gr.Dataframe(label="Question-wise result", wrap=True)
        rep = gr.HTML(label="Report preview"); fl = gr.File(label="Download report (HTML / PDF / CSV)", file_count="multiple")
        parse_btn.click(ui_parse, [qp, rub, rub_text, rub_mode, dm, llm_dd], [pst, pprev, ov])
        go.click(ui_evaluate, [qp, rub, rub_text, rub_mode, ans, nm, rl, llm_dd, dm, ov], [st, sm, tbl, rep, fl])
    with gr.Tab("3 · Teacher review"):
        with gr.Row():
            sel = gr.Dropdown(choices=db_list_evals(), label="Evaluation"); rf = gr.Button("↻ Refresh"); ld = gr.Button("Load")
        tot_md = gr.Markdown()
        rdf = gr.Dataframe(headers=REVIEW_COLS, interactive=True, wrap=True,
                           label="Edit 'Teacher marks' and 'Feedback' — only these two columns are saved.")
        sv = gr.Button("💾 Save changes & regenerate report", variant="primary"); smsg = gr.Markdown(); sfiles = gr.File(file_count="multiple")
        rf.click(review_refresh, None, sel); ld.click(review_load, sel, [rdf, tot_md]); sel.change(review_load, sel, [rdf, tot_md])
        sv.click(review_save, [sel, rdf], [smsg, sfiles])
    with gr.Tab("4 · AI vs Human agreement"):
        gr.Markdown("1) Create the template, 2) fill `human_marks` from the faculty evaluation, 3) upload it here.")
        tb = gr.Button("Create human-marks CSV template"); tf = gr.File(label="Template")
        hc = gr.File(label="Faculty marks CSV (roll, qid, human_marks)", file_types=[".csv"]); ab = gr.Button("Compute agreement", variant="primary")
        at = gr.Dataframe(label="Agreement metrics"); ap = gr.Plot(); am = gr.Dataframe(label="Matched rows")
        tb.click(ui_template, None, tf); ab.click(ui_agreement, hc, [at, ap, am])

demo.queue().launch(server_name="127.0.0.1", inbrowser=True, share=False)
