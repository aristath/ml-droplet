#!/usr/bin/env bash
set -euo pipefail

MODEL_ID_1="MoritzLaurer/bge-m3-zeroshot-v2.0"
MODEL_ID_2="MoritzLaurer/ModernBERT-base-zeroshot-v2.0"
MODEL_ID_3="MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33"
MODEL_ID_4="MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33"
APP_DIR="/opt/ml-droplet"

echo "==> Adding 2GB swap"
if [ ! -f /swapfile ]; then
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    echo "Swap enabled"
else
    echo "Swap already exists"
fi

echo "==> Installing system packages"
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv

echo "==> Setting up venv"
python3 -m venv /opt/ml-venv
source /opt/ml-venv/bin/activate

echo "==> Installing Python dependencies"
pip install --upgrade pip
pip install -r "$APP_DIR/requirements.txt"

echo "==> Pre-downloading model files"
python3 -c "
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

for model_id in ['$MODEL_ID_1', '$MODEL_ID_2', '$MODEL_ID_3', '$MODEL_ID_4']:
    print(f'Downloading {model_id}...')
    AutoTokenizer.from_pretrained(model_id)
    ORTModelForSequenceClassification.from_pretrained(model_id, subfolder='onnx')
    print(f'  {model_id} cached')
print('All models downloaded')
"

echo "==> Creating systemd service"
cat > /etc/systemd/system/ml-classifier.service <<EOF
[Unit]
Description=Zero-Shot Classifier API
After=network.target

[Service]
Type=simple
WorkingDirectory=$APP_DIR
ExecStart=/opt/ml-venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ml-classifier
systemctl restart ml-classifier

echo "==> Done. Service running on port 8000"
