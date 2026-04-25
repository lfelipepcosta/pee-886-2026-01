FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    joblib \
    matplotlib \
    seaborn \
    tqdm \
    gdown \
    optuna \
    SQLAlchemy \
    xgboost \
    optuna-integration[xgboost] \
    geopandas \
    shapely \
    rasterio \
    pyarrow \
    pennylane \
    pennylane-lightning \
    loguru \
    rich-argparse

VOLUME ["/app/data"]

CMD ["/bin/bash"]