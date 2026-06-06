import os
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API", "https://models.sjtu.edu.cn/api/v1")
API_SECRET = os.getenv("API_SECRET", "")
API_MODEL = os.getenv("API_MODEL", "deepseek-reasoner")
