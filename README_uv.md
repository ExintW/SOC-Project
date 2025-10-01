# SOC项目 - uv依赖管理

## 项目依赖管理从Pipenv迁移到uv

### 为什么使用uv？
- **更快的安装速度**：uv比pipenv快10-100倍
- **更好的依赖解析**：更智能的依赖冲突解决
- **现代化工具**：基于Rust开发，性能优异
- **兼容性**：完全兼容pip和pipenv

### 安装uv
```bash
# 全局安装uv
pip install uv

# 或者使用官方安装脚本
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 项目设置

#### 1. 初始化项目
```bash
uv init
```

#### 2. 添加依赖
```bash
# 添加单个包
uv add numpy

# 添加多个包
uv add pandas matplotlib scipy

# 添加开发依赖
uv add --dev pytest black
```

#### 3. 安装所有依赖
```bash
uv sync
```

### 运行项目

#### 运行Python脚本
```bash
# 运行主模型
uv run python htgy/D_Prediction_Model/htgy_SOC_model_with_river_basin.py

# 运行可视化脚本
uv run python htgy/A_Data_visualization/Plot_Total_C.py

# 运行数据转换
uv run python htgy/A_Data_visualization/parquet_to_csv.py
```

#### 激活虚拟环境
```bash
# 激活虚拟环境
uv shell

# 在激活的环境中运行
python htgy/D_Prediction_Model/htgy_SOC_model_with_river_basin.py
```

### 依赖管理

#### 查看当前依赖
```bash
uv tree
```

#### 更新依赖
```bash
# 更新所有依赖到最新版本
uv add --upgrade

# 更新特定包
uv add --upgrade numpy pandas
```

#### 移除依赖
```bash
uv remove numpy
```

### 项目文件

- `pyproject.toml` - 项目配置和依赖定义
- `uv.lock` - 锁定的依赖版本（自动生成）
- `.venv/` - 虚拟环境目录

### 从Pipenv迁移

#### 旧文件（可以删除）
- `Pipfile` - Pipenv配置文件
- `Pipfile.lock` - Pipenv锁定文件

#### 新文件
- `pyproject.toml` - uv项目配置
- `uv.lock` - uv锁定文件

### 团队协作

#### 新成员设置
```bash
# 克隆项目
git clone <repository-url>
cd SOC-Project

# 安装依赖
uv sync

# 运行项目
uv run python htgy/D_Prediction_Model/htgy_SOC_model_with_river_basin.py
```

#### 更新依赖
```bash
# 拉取最新代码
git pull

# 同步依赖
uv sync
```

### 常用命令对比

| 功能 | Pipenv | uv |
|------|--------|----|
| 安装依赖 | `pipenv install` | `uv sync` |
| 运行脚本 | `pipenv run python script.py` | `uv run python script.py` |
| 激活环境 | `pipenv shell` | `uv shell` |
| 添加包 | `pipenv install package` | `uv add package` |
| 移除包 | `pipenv uninstall package` | `uv remove package` |

### 性能优势

- **安装速度**：uv比pipenv快10-100倍
- **依赖解析**：更快的依赖冲突解决
- **缓存管理**：更智能的缓存策略
- **并行处理**：支持并行下载和安装

### 注意事项

1. **虚拟环境**：uv使用`.venv/`目录，与Pipenv的虚拟环境不同
2. **锁定文件**：`uv.lock`文件应该提交到版本控制
3. **Python版本**：确保使用Python 3.10或更高版本
4. **依赖兼容性**：所有原有依赖都已迁移，功能完全兼容
