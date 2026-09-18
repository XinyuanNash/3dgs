"""Pydantic 请求/响应模型。"""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    INBOUND = "inbound"
    NORMALIZING = "normalizing"
    FRAME_EXTRACTING = "frame_extracting"
    DOWNSAMPLING = "downsampling"
    COLMAP_FEAT = "colmap_feat"
    COLMAP_MATCH = "colmap_match"
    COLMAP_MAP = "colmap_map"
    ALIGN_GPS = "align_gps"
    UNDISTORT = "undistort"
    IMGS2POSES = "imgs2poses"
    TRAINING = "training"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobCreated(BaseModel):
    id: str
    status: JobStatus
    downsample_factor: int = 1


class JobSummary(BaseModel):
    id: str
    name: Optional[str] = None
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    iterations: int
    final_psnr: Optional[float] = None
    error: Optional[str] = None
    input_kind: str
    input_filename: str
    image_count: Optional[int] = None
    downsample_factor: Optional[int] = None  # 1=未下采样;2/4/8=下采样倍数


class StageRunInfo(BaseModel):
    stage: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    exit_code: Optional[int] = None
    log_path: str


class JobStatusResponse(JobSummary):
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    stage_runs: list[StageRunInfo] = Field(default_factory=list)


class JobListResponse(BaseModel):
    items: list[JobSummary]
    total: int


class LogChunkResponse(BaseModel):
    stage: str
    lines: list[str]
    total_lines: int


class HealthResponse(BaseModel):
    ok: bool
    container_reachable: bool
    gpu_free_mb: Optional[int] = None
    jobs_count: int