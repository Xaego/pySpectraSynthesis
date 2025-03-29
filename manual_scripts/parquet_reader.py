# 83 sec for ods, probably due archive structure of ods format
# 7 sec xlsx

import pandas as pd
from loguru import logger
from pathlib import Path
import time

start = time.time()

# Определяем корневую папку проекта
project_root = Path(__file__).resolve().parents[1]
logger.debug(f"project_root: {project_root}")

# Пути к файлам
parquet_path = project_root / "data" / "light_sources.parquet"
ods_path = project_root / "data" / "light_sources.ods"
xlsx_path = project_root / "data" / "light_sources.xlsx"
logger.debug(f"ods_path: {parquet_path}")
logger.debug(f"xlsx_path: {xlsx_path}")

# Чтение Parquet
df = pd.read_parquet(parquet_path)
logger.debug(df)

# Сохранение в ODS
# df.to_excel(ods_path, engine="odf", index=False)
df.to_excel(xlsx_path, index=False)

end = time.time()
logger.success(f"Done in {end - start:.2f} seconds")
