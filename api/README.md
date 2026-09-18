# 3DGS Pipeline Service

端到端 3D Gaussian Splatting 训练流水线编排服务。

把"上传图片/视频 → 训练 → 下载 `.ply` 模型"的完整流程封装为 HTTP API。

## 架构

- FastAPI 服务运行在宿主机(端口 8000)
- 通过 `docker exec 3dgs_xy …` 在容器内执行每个阶段
- 3DGS 训练代码 (`/home/3dgs-work/`) 和工具 (`/home/Pretools/`) 都已在容器内
- SQLite 持久化任务状态(`/data4/huxinyuan/3dgs/api/db.sqlite`)
- 每个任务一个时间戳文件夹,所有文件仅在其内部产生/删除

```
POST /api/v1/jobs   (multipart 上传)  →  job_id
GET  /api/v1/jobs/{id}                →  状态 + PSNR
GET  /api/v1/jobs/{id}/ply            →  最终 PLY 下载
GET  /api/v1/jobs/{id}/log?stage=...  →  阶段日志尾部
DELETE /api/v1/jobs/{id}              →  删除任务(默认连文件夹)
POST /api/v1/jobs/{id}/cancel         →  取消任务
GET  /api/v1/health                   →  健康检查
GET  /                                →  入口
GET  /docs                            →  Swagger UI
```

## 安装

服务需要以下 Python 包:**fastapi、uvicorn[standard]、python-multipart、aiosqlite、pydantic**。

### 当前状态(2026-08-31)

**离线**:当前宿主环境无法访问 PyPI / pypi.org(DNS 解析失败)。
所有依赖都需要在有网络时安装。

### 安装命令(有网络时执行)

```bash
cd /data4/huxinyuan/3dgs/api
/usr/bin/python3 -m pip install --user -r requirements.txt
# 或者使用 3dgs conda 环境(推荐,Python 3.10.20):
/data4/huxinyuan/3dgs/miniconda3/envs/3dgs/bin/python -m pip install -r requirements.txt
```

### 国内镜像(推荐,默认 pypi.org 在国内常连不通)

**常用 pip 镜像地址**:

| 镜像 | URL |
|---|---|
| 清华 | https://pypi.tuna.tsinghua.edu.cn/simple |
| 阿里云 | https://mirrors.aliyun.com/pypi/simple/ |
| 中科大 | https://pypi.mirrors.ustc.edu.cn/simple/ |
| 豆瓣 | https://pypi.doubanio.com/simple/ |
| 腾讯云 | https://mirrors.cloud.tencent.com/pypi/simple/ |
| 华为云 | https://repo.huaweicloud.com/repository/pypi/simple/ |

**方式 1:命令行 `-i`**
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple fastapi 'uvicorn[standard]' aiosqlite pydantic python-multipart python-dotenv
```

**方式 2:`~/.pip/pip.conf` 用户级(全局生效,推荐)**
```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple/
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 60

[install]
user = true
```

**方式 3:环境变量(临时)**
```bash
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/ pip install -r requirements.txt
```

**方式 4:`requirements.txt` 内嵌镜像** —— 把镜像地址写入文件,便于团队共享:
```
-i https://pypi.tuna.tsinghua.edu.cn/simple/
fastapi==0.115.0
uvicorn[standard]==0.32.0
python-multipart==0.0.12
aiosqlite==0.20.0
pydantic==2.9.2
python-dotenv==1.0.1
```

### DNS 不通时(DNS 解析失败,但 HTTP 还能走 IP)

如果 `pypi.org` 解析失败但能拿到镜像 IP,临时绕过 DNS:
```bash
echo "223.5.5.5 pypi.org" | sudo tee -a /etc/hosts   # 不推荐
# 更好的做法:让网络管理员把 DNS 改成 223.5.5.5(阿里 DNS)或 119.29.29.29(DNSPod)
```

### 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `HOST_UID` | 1022 | docker exec 内的 UID(匹配宿主机用户) |
| `HOST_GID` | 1023 | docker exec 内的 GID |
| `GPU_INDEX` | 1 | COLMAP + 训练使用的 GPU 编号 |
| `DOCKER_CONTAINER` | 3dgs_xy | 目标容器名 |
| `DISK_FREE_MIN_GB` | 5 | 磁盘预检阈值 |

## 启动

```bash
cd /data4/huxinyuan/3dgs/api
# 用 3dgs conda 环境启动(推荐)
PY=/data4/huxinyuan/3dgs/miniconda3/envs/3dgs/bin/python
$PY -m uvicorn app:app --host 0.0.0.0 --port 8000

# 后台运行
nohup $PY -m uvicorn app:app --host 0.0.0.0 --port 8000 \
    > /data4/huxinyuan/3dgs/api/server.log 2>&1 &
```

## 上传方式

API 支持三种上传方式(自动检测):

### 1. 多个图片文件(multipart 多文件)
```bash
curl -X POST http://localhost:8000/api/v1/jobs \
    -F "name=my_dataset" \
    -F "iterations=30000" \
    -F "files=@IMG_0001.jpg" \
    -F "files=@IMG_0002.jpg" \
    -F "files=@IMG_0003.jpg" \
    ...
```

### 2. 整个文件夹压缩上传(推荐)
把整个文件夹打成 zip/tar.gz 一次上传,服务端自动解压:
```bash
# 打包
zip -r my_dataset.zip my_dataset/
# 或:tar czf my_dataset.tar.gz my_dataset/

# 上传(单文件)
curl -X POST http://localhost:8000/api/v1/jobs \
    -F "name=my_dataset" \
    -F "iterations=30000" \
    -F "files=@my_dataset.zip"
```

支持的压缩格式:
- `.zip`(标准 zip)
- `.tar.gz` / `.tgz`(gzip 压缩 tar)
- `.tar`(未压缩 tar)

自动跳过:`__MACOSX/`、`.DS_Store`、`._*`(macOS 元数据)

解压限制(防 zip bomb):
- 解压后总大小 ≤ 10 GB
- 文件数 ≤ 10000

### 3. 视频文件(单文件)
```bash
curl -X POST http://localhost:8000/api/v1/jobs \
    -F "name=my_video" \
    -F "iterations=30000" \
    -F "files=@my_video.mp4"
```

支持的视频格式: `.mp4`、`.mov`、`.avi`、`.mkv`、`.webm`

每 ~10 帧抽取一帧(可调 `-f` 参数,见 `stages/frame_extract.py`)

### 4. 命令行一行上传(无需手动 zip)

`scripts/upload.py` 自动检测输入,打包 zip 后上传,轮询到完成并下载 PLY:

```bash
# 上传图片文件夹
python /data4/huxinyuan/3dgs/api/scripts/upload.py /path/to/my_images/

# 上传视频
python /data4/huxinyuan/3dgs/api/scripts/upload.py /path/to/video.mp4

# 指定名称 + 迭代次数
python /data4/huxinyuan/3dgs/api/scripts/upload.py ./my_images/ --name scene01 --iterations 30000

# 仅下载已有 job 的 PLY
python /data4/huxinyuan/3dgs/api/scripts/upload.py --fetch-only 20260831153000-abcd1234 --out ./model.ply
```

### 5. 浏览器拖拽上传(零代码)

打开 `scripts/index.html`(也可由 FastAPI 静态托管),拖整个文件夹即可:

```
直接双击文件: file:///data4/huxinyuan/3dgs/api/scripts/index.html
或服务起来后: http://localhost:8000/static/index.html
```

特性:
- `<input webkitdirectory>` 一键选择整个目录(Chrome / Edge / Safari)
- 拖拽支持
- 提交后实时轮询状态
- 完成后自动提供下载链接

### 6. Swagger UI 内直接上传文件夹

打开 `http://localhost:8005/docs`,点开 `POST /api/v1/jobs` 操作块,文件选择框**已自动启用 webkitdirectory**(Chrome / Edge / Safari):可以直接选整个文件夹,或单选 zip / tar.gz。Firefox 用户需先手动压缩成 zip 后上传。

技术细节: `app.py` 通过 `docs_url=None` 关闭默认 Swagger UI,然后自定义 `/docs` 路由,注入 MutationObserver 给动态渲染的 `<input type="file">` 加 `webkitdirectory` + `directory` + `multiple` 属性,并在 create_job 操作块上添加橙色提示横幅。

## 烟雾测试

```bash
# 1. 准备 ~20 张测试图片
mkdir -p /tmp/smoke_input
cp /data4/huxinyuan/3dgs/3dgs-work/datasets/campus/images/IMG_*.JPG /tmp/smoke_input/ | head -20

# 或者打 zip 上传:
# zip -r /tmp/smoke_input.zip /tmp/smoke_input/

# 2. 运行测试脚本
bash /data4/huxinyuan/3dgs/api/scripts/smoke.sh
```

脚本会:
1. 健康检查
2. 上传图片 → 创建 job
3. 轮询直到 SUCCEEDED(默认 60 分钟超时)
4. 下载 PLY 到 `/tmp/smoke_<job_id>.ply`
5. 验证 PLY 格式

## 流水线顺序

```
inbound → normalize → [frame_extract] → colmap feature_extractor → exhaustive_matcher
       → mapper → undistort → imgs2poses → train.py (ship-path) → PLY
```

`image_folder` 输入跳过 `frame_extract`;`video` 输入包含全部 9 阶段。

## Ship-path CLI(verbatim,在 `stages/train.py` 中)

```bash
cd <job_dir> && \
CUDA_VISIBLE_DEVICES=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/miniconda3/envs/3dgs/bin/python /home/3dgs-work/train.py \
    -s <job_dir>/distort_free \
    -m <job_dir>/model \
    --iterations 30000 \
    --eval \
    --test_iterations 7000 18000 30000 \
    --save_iterations 30000 \
    --checkpoint_iterations 30000 \
    --densification_strategy igs_plus \
    --igs_plus_max_cap 1200000 \
    --igs_plus_scale_reg_weight 0.01 \
    --use_bilateral_grid \
    --use_ppisp \
    --ppisp_start_iter 26000 \
    --admm_start_iter 26000 --admm_end_iter 29000 \
    --disable_viewer \
    --bilateral_grid_start_iter 26000 \
    2>&1 | tee <job_dir>/logs/stage_08_train.log
```

## 文件结构

```
api/
├── app.py                    # FastAPI 入口
├── requirements.txt
├── README.md
├── pipeline3dgs/
│   ├── config.py             # 路径、容器、UID 配置
│   ├── utils.py              # docker_exec 辅助
│   ├── db.py                 # SQLite 异步封装
│   ├── models.py             # Pydantic 模型
│   ├── routes.py             # 所有 /api/v1 路由
│   ├── runner.py             # 流水线编排
│   └── stages/
│       ├── inbound.py        # 阶段 0:接收 + 分类
│       ├── normalize.py      # 阶段 1:重命名 images_0,1,2,...
│       ├── frame_extract.py  # 阶段 2:视频抽帧
│       ├── colmap_sfm.py     # 阶段 3-5:COLMAP
│       ├── undistort.py      # 阶段 6:畸变校正
│       ├── imgs2poses.py     # 阶段 7:生成 LLFF poses
│       ├── train.py          # 阶段 8:ship-path 训练
│       └── metrics.py        # PSNR 正则提取
└── scripts/
    └── smoke.sh              # 烟雾测试
```

## 单任务文件夹布局

```
/data4/huxinyuan/3dgs/jobs/<job_id>/
├── manifest.json
├── input.bin                  # 视频输入
│   └── input/<原文件名>       # 图片输入
├── images/images_0,1,...      # 标准化后的图片
├── database.db                # COLMAP
├── sparse/0/                  # COLMAP mapper 输出
├── distort_free/              # undistort + LLFF poses
│   ├── images/, sparse/0/, poses_bounds.npy
├── model/                     # train.py 输出
│   └── point_cloud/iteration_30000/point_cloud.ply   # 最终 PLY
└── logs/stage_00_inbound.log  # 每阶段一份日志
```

## 故障排查

| 现象 | 可能原因 |
|---|---|
| 502 / 503 health check fails | docker daemon 不可用或容器未运行 |
| 任务卡在 inbound | 多文件上传失败,检查 `Content-Type: multipart/form-data` |
| 任务卡在 colmap_feat | images/ 为空 / GPU 不可用 |
| 任务卡在 undistort | sparse/0/ 不存在,mapper 失败 |
| 任务卡在 imgs2poses | PYTHONPATH 未生效(检查容器内 `/home/Pretools`) |
| 训练 OOM | GPU 显存不足;降低 `--igs_plus_max_cap` 或取消 `--igs_plus_scale_reg_weight` |
| PLY 不存在 | 训练阶段失败;查 `logs/stage_08_train.log` |
| 文件属主变 root | `-u 1022:1023` 没生效;检查 `HOST_UID/HOST_GID` 环境变量 |