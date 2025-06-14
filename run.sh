#!/bin/bash
source venv_rl_webots/bin/activate
export WEBOTS_HEADLESS=1

export PORT=10000 && PORT=10000 xvfb-run -a webots --mode=fast --no-rendering --batch --minimize worlds/rl_project_tesla.wbt &

export PORT=10001 && PORT=10001 xvfb-run -a webots --mode=fast --no-rendering --batch --minimize worlds/rl_project_tesla.wbt &

export PORT=10002 && PORT=10002 xvfb-run -a webots --mode=fast --no-rendering --batch --minimize worlds/rl_project_tesla.wbt &

export PORT=10003 && PORT=10003 xvfb-run -a webots --mode=fast --no-rendering --batch --minimize worlds/rl_project_tesla.wbt &

export PORT=10004 && PORT=10004 xvfb-run -a webots --mode=fast --no-rendering --batch --minimize worlds/rl_project_tesla.wbt &

export PORT=10005 && PORT=10005 xvfb-run -a webots --mode=fast --no-rendering --batch --minimize worlds/rl_project_tesla.wbt &

export PORT=10006 && PORT=10006 xvfb-run -a webots --mode=fast --no-rendering --batch --minimize worlds/rl_project_tesla.wbt &

export PORT=10007 && PORT=10007 xvfb-run -a webots --mode=fast --no-rendering --batch --minimize worlds/rl_project_tesla.wbt &

# Aspetta che tutte le istanze siano avviate
sleep 60
sudo lsof -i -P -n | grep LISTEN

# Avvia lo script di training
python3 train.py

