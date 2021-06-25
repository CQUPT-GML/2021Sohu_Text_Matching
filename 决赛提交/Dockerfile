FROM tensorflow/tensorflow:1.14.0-gpu-py3-jupyter

ENV DEBIAN_FRONTEND=noninteractive

# 设置环境变量，使其支持中文
ENV LANG "en_US.UTF-8"

COPY . /app

WORKDIR /app

# 可以在本地构建镜像时指定源
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

ENTRYPOINT ["python","/app/new_get_test_output.py"]