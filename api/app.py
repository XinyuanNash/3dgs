"""FastAPI 入口 —— uvicorn 启动目标。

启动:
    cd /data4/huxinyuan/3dgs/api
    /usr/bin/python3 -m uvicorn app:app --host 0.0.0.0 --port 8000

环境变量:
    HOST_UID, HOST_GID:容器内 UID 桥接(默认 1022, 1023)
    GPU_INDEX:COLMAP + 训练使用的 GPU(默认 1)
    DOCKER_CONTAINER:目标容器名(默认 3dgs_xy)
    DISK_FREE_MIN_GB:磁盘预检阈值(默认 5)
"""
from __future__ import annotations
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from pipeline3dgs.db import Db
from pipeline3dgs.runner import AsyncPipelineRunner
from pipeline3dgs.routes import init as init_routes, router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("pipeline3dgs.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动/关闭钩子。

    启动:
      1. 连接 SQLite
      2. 把任何非终态任务标为 FAILED(防悬挂)
      3. 初始化 AsyncPipelineRunner + 注入 routes

    关闭:
      1. 关闭 DB
    """
    db = Db()
    await db.connect()

    reset_count = await db.reset_inflight_to_failed()
    if reset_count > 0:
        log.warning("reset %d in-flight jobs to FAILED on startup", reset_count)

    runner = AsyncPipelineRunner(db)
    init_routes(runner, db)
    app.state.db = db
    app.state.runner = runner

    log.info("pipeline3dgs service ready (db=%s)", db.path)
    try:
        yield
    finally:
        await db.close()
        log.info("pipeline3dgs service shutdown complete")


app = FastAPI(
    title="3DGS Pipeline Service",
    description="端到端 3D Gaussian Splatting 训练流水线编排(图片/视频 → PLY)",
    version="0.1.0",
    lifespan=lifespan,
    # 关闭默认 /docs / redoc,改为自定义(支持文件夹上传)
    docs_url=None,
    redoc_url=None,
)

# ---- CORS 跨域中间件 ----
# 让任意 IP/任意端口的浏览器(或 curl / fetch)都能访问 API。
# uvicorn --host 0.0.0.0 已经允许网络层访问;
# 但浏览器对跨域 fetch 会拦截响应 —— 需要 CORS 头。
# allow_origins=["*"] 表示任何 origin 都允许;
# allow_credentials=False 时 "*" 是合法的(否则需列出具体 origin)。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# 静态文件:浏览器上传页面
app.mount("/static", StaticFiles(directory="scripts", html=True), name="static")


# ---- Static no-cache 中间件 ----
# 静态文件 (HTML/JS/CSS) 改动后,浏览器可能因 ETag 缓存继续返回旧版本,
# 导致用户看不到新功能 (例如刚加的 frame_interval 输入框)。
# 给 /static/* 响应强制加 Cache-Control: no-cache + must-revalidate,
# 浏览器每次都跟服务器校验 (Last-Modified),但仍是 304 快速路径,
# 不影响性能,只保证"刷一下就能看到新版"。
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class StaticNoCacheMiddleware(BaseHTTPMiddleware):
    """给 HTML 页面响应加 no-cache 头(防止浏览器缓存老版本导致新功能看不到)。

    覆盖路径:
      - /static/*  : StaticFiles mount (备用拖拽页)
      - /upload    : 内嵌 HTMLResponse 上传页(主入口,推荐)
      - /docs      : Swagger UI(开发中可能改 schema)
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/static/") or path in ("/upload", "/docs", "/"):
            if path == "/" or path.endswith(".html") or path.startswith("/static/") or path in ("/upload", "/docs"):
                # 仅对 HTML 响应加 no-cache;其他 (CSS/JS/JSON/图片) 走正常缓存
                ct = response.headers.get("content-type", "")
                if "text/html" in ct or path in ("/upload", "/docs"):
                    response.headers["Cache-Control"] = "no-cache, must-revalidate"
                    response.headers["Pragma"] = "no-cache"
                    response.headers["Expires"] = "0"
        return response


app.add_middleware(StaticNoCacheMiddleware)


# ---- 自定义 Swagger UI ----
# 策略:进入 /docs 时,如果浏览器没有 special bypass 参数 ?nopush,
# 直接跳转到 /upload (独立上传页)。Swagger UI 5 的 try-it-out
# 即使 <input type="file" multiple> 也只读 files[0] —— 多文件上传必须用 /upload。
# 用户如果真的想看 API 文档,加 ?nopush 留在 Swagger UI。
CUSTOM_SWAGGER_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>3DGS Pipeline API</title>
<script>
  // 即时 redirect 到 /upload,除非 ?nopush 或已经是从 /upload 点过来的
  if (!location.search.includes('nopush') && document.referrer.indexOf('/upload') === -1) {
    location.replace('/upload' + location.search);
  }
</script>
<link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
<link rel="icon" type="image/png" href="https://fastapi.tiangolo.com/img/favicon.png">
<style>
  body { margin: 0; padding: 0; }
  .folder-hint {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border-left: 4px solid #f59e0b;
    padding: 8px 12px;
    margin: 8px 0;
    border-radius: 4px;
    font-size: 13px;
    font-family: -apple-system, "Segoe UI", sans-serif;
  }
  .top-banner {
    background: #f59e0b; color: #1a1a1a; padding: 16px;
    text-align: center; font-family: -apple-system, sans-serif;
    font-size: 15px; font-weight: 600;
  }
  .top-banner a { color: #1a1a1a; text-decoration: underline; }
</style>
</head>
<body>
<div class="top-banner">
  ⚠️ POST /api/v1/jobs 在 Swagger UI 里有已知 bug (只读 files[0])。
  请用 <a href="/upload">/upload</a> 上传(支持文件夹 + 多文件 + zip)。
  <a href="/docs?nopush">[继续在 Swagger UI 看文档]</a>
</div>
<div id="swagger-ui"></div>
<script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>
window.onload = function() {
  const ui = SwaggerUIBundle({
    url: "/openapi.json",
    dom_id: "#swagger-ui",
    deepLinking: true,
    presets: [SwaggerUIBundle.presets.apis],
    plugins: [SwaggerUIBundle.plugins.DownloadUrl],
    layout: "BaseLayout",
  });

  // 关键:给所有 <input type="file"> 加上 webkitdirectory 让其支持选文件夹
  // (Chrome / Edge / Safari 支持;Firefox 暂不支持)
  // 警告:Swagger UI 5 即使 input 是 multiple,内部也只读 files[0] —— 多文件必须用 /upload
  const addFolderSupport = () => {
    document.querySelectorAll('input[type="file"]').forEach(input => {
      if (!input.hasAttribute('webkitdirectory')) {
        input.setAttribute('webkitdirectory', '');
        input.setAttribute('directory', '');
        if (!input.hasAttribute('multiple')) {
          input.setAttribute('multiple', '');
        }
      }
    });
    // 在 create_job 操作块上添加提示横幅
    const createJobOp = Array.from(document.querySelectorAll('.opblock')).find(
      el => el.querySelector('[data-tag]')
        && el.textContent.includes('/api/v1/jobs')
        && el.querySelector('.opblock-summary-method')
        && el.querySelector('.opblock-summary-method').textContent.trim() === 'post'
    );
    if (createJobOp && !createJobOp.querySelector('.folder-hint')) {
      const hint = document.createElement('div');
      hint.className = 'folder-hint';
      hint.innerHTML = '📁 <strong>多文件请用独立上传页</strong>: Swagger UI 一次只能发 1 个文件(已知 bug)。请打开 <a href="/upload" target="_blank" style="color:#fbbf24;text-decoration:underline">/upload</a> —— 支持文件夹选择 + 多文件 + zip/tar.gz,提交后自动轮询状态 + 提供 PLY 下载链接。';
      const body = createJobOp.querySelector('.opblock-body') || createJobOp;
      body.insertBefore(hint, body.firstChild);
    }
  };

  // Swagger UI 异步渲染,需要 MutationObserver 持续监听
  setTimeout(addFolderSupport, 300);
  setTimeout(addFolderSupport, 1000);
  new MutationObserver(addFolderSupport).observe(document.body, {childList: true, subtree: true});
};
</script>
</body>
</html>
"""


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    return HTMLResponse(CUSTOM_SWAGGER_HTML)


@app.get("/redoc", include_in_schema=False)
async def custom_redoc():
    """简化版 ReDoc(也指向自定义 Swagger)。"""
    return HTMLResponse(CUSTOM_SWAGGER_HTML.replace("3DGS Pipeline API", "3DGS Pipeline API (ReDOC unavailable, view /docs instead)"))


# ---- 独立的上传页面:完全绕开 Swagger UI 5 的多文件 bug ----
# Swagger UI 5 即使 <input type="file" multiple>,内部也只读 files[0]
# 所以我们提供一个独立的 form,直接 POST 到 /api/v1/jobs
UPLOAD_PAGE_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>3DGS Upload</title>
<style>
  body { font-family: -apple-system, "Segoe UI", sans-serif; max-width: 720px; margin: 40px auto; padding: 0 20px; color: #e6e6e6; background: #1a1a1a; }
  h1 { color: #fbbf24; }
  .card { background: #2a2a2a; border-radius: 8px; padding: 24px; margin: 16px 0; border-left: 4px solid #f59e0b; }
  label { display: block; margin: 12px 0 4px; font-weight: 600; }
  input[type="text"], input[type="number"] { width: 100%; padding: 8px; border-radius: 4px; border: 1px solid #444; background: #1a1a1a; color: #e6e6e6; box-sizing: border-box; }
  input[type="file"] { margin: 8px 0; color: #e6e6e6; }
  button { background: #f59e0b; color: #1a1a1a; padding: 10px 24px; border: none; border-radius: 4px; font-weight: 600; cursor: pointer; font-size: 15px; }
  button:hover { background: #fbbf24; }
  button:disabled { background: #666; cursor: not-allowed; }
  #status { padding: 12px; border-radius: 4px; margin: 16px 0; display: none; }
  #status.ok { background: #065f46; display: block; }
  #status.err { background: #7f1d1d; display: block; }
  #status.run { background: #1e3a8a; display: block; }
  a { color: #fbbf24; }
  .hint { color: #888; font-size: 13px; margin-top: 4px; }
  code { background: #111; padding: 2px 6px; border-radius: 3px; color: #fbbf24; }
</style>
</head>
<body>
<h1>📁 上传数据集</h1>
<div class="card">
  <p>支持:① 整个图片文件夹(webkitdirectory) ② 多个图片文件 ③ 单个 zip / tar.gz 压缩包 ④ 视频文件</p>
  <details style="margin:8px 0;color:#888;font-size:13px">
    <summary style="cursor:pointer;color:#fbbf24">📋 上传格式详情(点击展开)</summary>
    <ul style="margin:8px 0;padding-left:24px">
      <li><b>图片</b>:png / jpg / jpeg,字母序编号为 <code>images_0.png</code> 等</li>
      <li><b>压缩包</b>:.zip / .tar.gz / .tgz / .tar(解压后递归找图,10 GB / 1 万文件上限防 zip bomb)</li>
      <li><b>视频</b>:mp4 / mov / avi / mkv / webm(单个文件);服务端用 <code>taketheframe</code> 默认每 13 帧取 1 帧</li>
    </ul>
    <p style="margin:4px 0;color:#fbbf24">⚠️ 上传采用流式写入(1 MB chunks)—— 几百 MB 的视频不会一次性读进内存,也不会撑爆进程。</p>
    <p style="margin:4px 0">磁盘剩余建议 &gt; 5 GB(单次 30K 训练 + COLMAP 中间产物约 2-3 GB)。</p>
  </details>
  <form id="uploadForm">
    <label>数据集名称(可选,会出现在任务文件夹名里)</label>
    <input type="text" name="name" placeholder="my_scene">
    <p class="hint">填写后,任务文件夹名为 <code>YYYYMMDDHHMMSS-&lt;name&gt;-&lt;4hex&gt;</code>(如 <code>20260904163242-my_scene-a1b2</code>);留空则用随机 8 位 hex。建议用易识别名(<code>campus</code>、<code>9_1_2</code>、<code>biandianzhan7</code>);中文 / 空格 / 路径分隔符会被清洗为 <code>_</code>。</p>
    <label>迭代次数(默认 30000)</label>
    <input type="number" name="iterations" value="30000" min="100" max="100000">

    <label>截帧间隔(仅视频生效,image_folder 忽略;范围 1-1000,默认 13 ≈ 25fps 下 2 帧/秒)</label>
    <input type="number" name="frame_interval" value="13" min="1" max="1000" step="1" title="每隔 N 帧取 1 帧;调大 → 帧少训得快但 SfM 精度降;调小 → 帧多训得慢但精度高">

    <label>下采样倍数(图片 / 视频抽帧后,在 COLMAP 前生效;1× = 不下采样)</label>
    <select name="downsample_factor" style="width:100%;padding:8px;border-radius:4px;border:1px solid #444;background:#1a1a1a;color:#e6e6e6;font-size:14px">
      <option value="1" selected>1×(不下采样,默认)</option>
      <option value="2">2×(长宽各减半,像素数 1/4,显著加速)</option>
      <option value="4">4×(像素数 1/16,适合大场景预览)</option>
      <option value="8">8×(像素数 1/64,最快,适合试跑)</option>
    </select>
    <p class="hint">⚠️ 8K 视频抽帧后默认已是 2K 量级,推荐 1×~2×;若 SfM 跑得很慢或 OOM,可升到 4× / 8×。EXIF(含 GPS)会被保留。</p>

    <label>图片文件夹 / 多个文件 / 单个压缩包 / 视频</label>
    <input type="file" name="files" webkitdirectory directory multiple>
    <p class="hint">Chrome / Edge / Safari:直接选文件夹;Firefox:请手动选多个文件(也支持 zip);zip/tar.gz 也支持任何浏览器。视频:只能选单个文件(选多个会被忽略,只取第一个)。</p>

    <button type="submit" id="submitBtn">上传并训练</button>
  </form>
</div>
<div id="status"></div>
<div id="result"></div>
<div class="card">
  <p>API 文档: <a href="/docs">/docs</a></p>
  <p>健康检查: <a href="/api/v1/health">/api/v1/health</a></p>
</div>
<script>
const form = document.getElementById('uploadForm');
const submitBtn = document.getElementById('submitBtn');
const statusEl = document.getElementById('status');
const resultEl = document.getElementById('result');

function showStatus(msg, cls) {
  statusEl.textContent = msg;
  statusEl.className = cls;
}

async function pollJob(jobId) {
  for (let i = 0; i < 600; i++) {  // 最多 30 分钟(每 3s)
    await new Promise(r => setTimeout(r, 3000));
    try {
      const r = await fetch('/api/v1/jobs/' + jobId);
      if (!r.ok) continue;
      const d = await r.json();
      const stage = d.current_stage || '?';
      const psnr = d.final_psnr ? ' PSNR=' + d.final_psnr.toFixed(2) : '';
      showStatus(`状态: ${d.status} | 阶段: ${stage}${psnr}`, d.status === 'failed' ? 'err' : 'run');
      if (d.status === 'succeeded') {
        showStatus(`✓ 完成! PSNR = ${d.final_psnr?.toFixed(2) || '?'} dB @ iter ${d.final_psnr_iter || '?'}`, 'ok');
        resultEl.innerHTML = `<p><a href="/api/v1/jobs/${jobId}/ply" download="${jobId}.ply">⬇ 下载 PLY 模型</a> (${d.artifacts?.ply || ''})</p>`;
        return;
      } else if (d.status === 'failed') {
        showStatus(`✗ 失败: ${d.error_message || ''}`, 'err');
        return;
      } else if (d.status === 'cancelled') {
        showStatus('已取消', 'err');
        return;
      }
    } catch (e) {}
  }
  showStatus('超时(>30 分钟),请查看 /api/v1/jobs/' + jobId, 'err');
}

form.onsubmit = async (e) => {
  e.preventDefault();
  submitBtn.disabled = true;
  showStatus('上传中...', 'run');
  resultEl.innerHTML = '';

  const fd = new FormData();
  const nameInput = form.querySelector('[name=name]').value;
  const iterInput = form.querySelector('[name=iterations]').value;
  const fiInput = form.querySelector('[name=frame_interval]').value;
  if (nameInput) fd.append('name', nameInput);
  fd.append('iterations', iterInput);
  fd.append('frame_interval', fiInput);
  const dsInput = form.querySelector('[name=downsample_factor]');
  fd.append('downsample_factor', dsInput ? dsInput.value : '1');

  const fileInput = form.querySelector('[name=files]');
  const files = fileInput.files;
  if (!files || files.length === 0) {
    showStatus('请选择文件 / 文件夹 / 压缩包', 'err');
    submitBtn.disabled = false;
    return;
  }
  for (let i = 0; i < files.length; i++) {
    fd.append('files', files[i], files[i].webkitRelativePath || files[i].name);
  }
  showStatus(`上传 ${files.length} 个文件...`, 'run');

  try {
    const r = await fetch('/api/v1/jobs', { method: 'POST', body: fd });
    if (!r.ok) {
      const err = await r.text();
      showStatus('上传失败: ' + r.status + ' ' + err, 'err');
      submitBtn.disabled = false;
      return;
    }
    const d = await r.json();
    showStatus(`✓ 已创建 job ${d.id},开始轮询状态...`, 'run');
    pollJob(d.id);
  } catch (e) {
    showStatus('网络错误: ' + e.message, 'err');
  } finally {
    submitBtn.disabled = false;
  }
};
</script>
</body>
</html>
"""


@app.get("/upload", include_in_schema=False)
async def upload_page():
    """独立的上传页面 —— 绕开 Swagger UI 5 的多文件 bug。"""
    return HTMLResponse(UPLOAD_PAGE_HTML)


@app.get("/")
async def root():
    return {
        "service": "3dgs-pipeline",
        "version": "0.1.0",
        "docs": "/docs",
        "upload_page": "/upload",   # 推荐:独立上传页(支持文件夹)
        "upload_alt": "/static/index.html",  # 备用:旧版拖拽页
        "api_base": "/api/v1",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,
    )