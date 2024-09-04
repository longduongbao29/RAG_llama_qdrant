#!/bin/bash
# Khởi động UI service
python3 -m uvicorn ui:app --host 0.0.0.0 --port 1234 &

# Khởi động API service
python3 -m uvicorn main:app --host 0.0.0.0 --port 1235 &

# Đợi các process hoàn thành
wait -n
