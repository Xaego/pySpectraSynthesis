from pathlib import Path
import polars as pl
from loguru import logger

# project dir
script_dir = Path(__file__).resolve().parent
light_source_file = script_dir.parent / "data" / "light_sources.xlsx"

logger.debug(f"light_source_file {light_source_file}")

# pl read excel
df = pl.read_excel(light_source_file)

logger.debug(f"df.shape {df.shape}")

# save to parquet
parquet_path = light_source_file.with_suffix('.parquet')
df.write_parquet(light_source_file)

logger.success(f"saved: parquet file {parquet_path}")
logger.success(f"done")