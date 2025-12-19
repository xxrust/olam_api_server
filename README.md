# OLAM 数据分析（API 服务）

本仓库为 OLAM 数据分析系统的后端 API（Flask），负责从 PostgreSQL 读取数据并向前端（桌面端/网页端）提供查询、分析与导出接口。

## 运行环境

- Python 3.10+（建议使用虚拟环境）
- PostgreSQL

## 快速开始

### 1) 安装依赖

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) 配置环境变量

将 `.env.example` 复制为 `.env` 并按实际修改：

```bash
DB_NAME=olam_demo
DB_USER=postgres
DB_PASSWORD=changeme
DB_HOST=localhost
DB_PORT=5432
```

注意：`.env` 包含敏感信息，请勿提交到仓库（已在 `.gitignore` 中忽略）。

### 3) 准备数据库

API 会读取 `public` schema 下的数据表。最常用的表包括：

- `olam`
- `grid`
- `last_round_f`

这些表结构可参考 `ServerForTcpTest` 仓库的 Diesel migrations（或使用你们提供的数据库备份/初始化脚本）。

### 4) 启动服务

```bash
python app.py
```

默认监听：`http://127.0.0.1:5000`

## 健康检查

- `GET /api/ping`：确认服务已启动
- `GET /api/test-db`：测试数据库连接与表读取

## 主要接口（概览）

以代码 `app.py` 中的路由为准，常用接口包括：

- 数据库信息：`GET /api/db-info`、`GET /api/test-db`
- 批次/设备/操作员：`GET /api/batches`、`GET /api/devices`、`GET /api/operators`
- 网格与时间轴：`GET /api/grid`、`GET /api/grid/options`、`GET /api/timeline/day`
- 分析类接口：`/api/analysis/*`（如初始方差、修盘效果、频率范围、操作员/设备影响等）
- 数据导出：`GET /api/olam/export`、`GET /api/last_round_f/export`
- AI 对话（可选）：`POST /api/ai/chat`

## AI 接口说明（可选）

`/api/ai/chat` 会按请求参数调用外部模型服务（例如自建/第三方推理服务）。请将 API Key 等敏感信息仅保存在调用方（前端/调用脚本）的安全配置中，不要写进仓库。

## 常见问题

- 连接数据库失败：检查 `.env`、数据库是否可达、用户权限与 `DB_PORT`
- 表不存在：先导入数据库备份或执行 `ServerForTcpTest` migrations 初始化表结构

