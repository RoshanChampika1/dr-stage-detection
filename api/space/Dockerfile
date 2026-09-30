# Hugging Face Space (Docker, free CPU) for the DR stage detection API.
# The Space only holds this Dockerfile, README.md and the model file best.pt;
# the code is cloned from the public GitHub repository at build time, so the
# deployed service always runs the same code as the repository.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face runs containers as user 1000.
RUN useradd -m -u 1000 user
WORKDIR /home/user

ARG REPO=https://github.com/RoshanChampika1/dr-stage-detection
# Changes whenever the main branch changes, so a rebuild picks up new code
# instead of reusing a cached clone.
ADD https://github.com/RoshanChampika1/dr-stage-detection/commits/main.atom /tmp/main-commits.atom
RUN git clone --depth 1 ${REPO}.git app

RUN pip install --no-cache-dir -r app/api/requirements.txt

COPY --chown=user best.pt /home/user/app/api/model/best.pt
RUN chown -R user /home/user/app
USER user
WORKDIR /home/user/app

ENV MODEL_PATH=/home/user/app/api/model/best.pt \
    PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
