# ============================================================
# MADDPG 多智能体路径规划实验
# 构建: docker build -t maddpg:latest .
# 运行: docker run -it --gpus all -v /host/data:/data maddpg:latest bash
# ============================================================

FROM registry.baidubce.com/paddlepaddle/paddle:2.5.1-gpu-cuda11.7-cudnn8.4-trt8.4

LABEL maintainer="ruihang"
LABEL description="MADDPG multi-agent path planning experiments"

ENV TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1

# 系统依赖 (shapely/scikit-image 需要)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgeos-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 先安装 Python 依赖（利用 Docker 层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- PARL 兼容性补丁 ----
# PyPI 版 parl 2.2.1 的两处 import 在 PaddlePaddle 2.5 上会报错，此处原地修补
RUN PARL_DIR=$(python -c "import parl, os; print(os.path.dirname(parl.__file__))") && \
    # 1) parl/__init__.py: remote 模块导入容错
    sed -i 's/from parl.remote import remote_class, connect/try:\n        from parl.remote import remote_class, connect\n    except (ImportError, ModuleNotFoundError):\n        pass/' \
        "$PARL_DIR/__init__.py" && \
    # 2) parl/utils/utils.py: paddle.fluid 已移除
    sed -i 's/from paddle import fluid/pass  # fluid removed in paddle>=2.5/' \
        "$PARL_DIR/utils/utils.py"

# 复制 maddpg 包
COPY . /workspace/maddpg

# 让 maddpg 可被 import（将 /workspace 加入 PYTHONPATH）
ENV PYTHONPATH=/workspace:${PYTHONPATH}

# 默认数据挂载点
VOLUME /data

HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "import paddle; import parl; from maddpg.envs import ParkEnv; print('OK')"

CMD ["/bin/bash"]
