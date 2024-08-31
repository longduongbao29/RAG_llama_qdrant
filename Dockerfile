FROM python:3.9-slim

# Cài đặt các gói cần thiết
RUN apt-get update && apt-get install -y gcc build-essential dos2unix

# Đặt thư mục làm việc
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt
# RUN pip install torch

# Sao chép file mã nguồn và script
COPY . /app
COPY start.sh /app/start.sh

# Chuyển đổi start.sh sang định dạng UNIX và cấp quyền thực thi
RUN dos2unix /app/start.sh
RUN chmod +x /app/start.sh

# Đặt entrypoint để chạy script
ENTRYPOINT ["/app/start.sh"]
