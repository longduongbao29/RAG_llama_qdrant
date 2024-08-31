import logging
import sys

# Định nghĩa mức độ mới
CUSTOM_LEVEL = 60
logging.addLevelName(CUSTOM_LEVEL, "OUTPUT")


# Hàm để ghi log với mức độ tùy chỉnh
def output(self, message, *args, **kwargs):
    if self.isEnabledFor(CUSTOM_LEVEL):
        self._log(CUSTOM_LEVEL, message, args, **kwargs)


logging.Logger.output = output

logger = logging.getLogger()
logger.setLevel(CUSTOM_LEVEL)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

# stdout_handler = logging.StreamHandler(sys.stdout)
# stdout_handler.setLevel(logging.DEBUG)
# stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler("logs/logs.log", encoding="utf-8")
file_handler.setLevel(CUSTOM_LEVEL)
file_handler.setFormatter(formatter)
# Xóa tất cả handlers đã có (nếu có)
if logger.hasHandlers():
    logger.handlers.clear()


logger.addHandler(file_handler)
# logger.addHandler(stdout_handler)
