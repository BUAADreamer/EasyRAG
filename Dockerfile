# Use the official Python base image
FROM python:3.10.14-slim

RUN mkdir -p /app
COPY requirements.txt /app

WORKDIR /app

RUN pip install --upgrade pip -i https://pypi.mirrors.ustc.edu.cn/simple/
# RUN pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple uv && \
# uv pip sync --python $(which python) --no-cache -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple requirements.txt
RUN pip install --no-cache -i https://pypi.mirrors.ustc.edu.cn/simple/ -r requirements.txt

# expose /data and /model as volumes
VOLUME [ "/data", "/models", "/app" ]

CMD ["python", "main.py"]
