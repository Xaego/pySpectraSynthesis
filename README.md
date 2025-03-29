# Spectrum Analyzer

A Streamlit web application for analyzing optical spectra of emitters, filters, and detectors.

## Features

- Visualize and analyze various light emitter spectra (LEDs, Lamps, Lasers)
- Apply optical filters with customizable parameters:
  - Band-pass filters with adjustable center wavelength and bandwidth
  - Long-pass filters with customizable cutoff wavelength
  - Short-pass filters with customizable cutoff wavelength
- Apply detector response curves to see how detectors "see" the filtered light
- Calculate and display key spectral statistics:
  - Peak wavelength
  - Full Width at Half Maximum (FWHM)
  - Relative intensity
- Support for both library filters and custom filter creation
- Interactive visualizations using Plotly

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create and activate a virtual environment (optional but recommended):
```bash
# Using venv
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run main.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the sidebar to:
   - Upload your own data in Parquet format or use the included sample data
   - Select emitters to analyze
   - Add and configure filters
   - Enable detector response if needed

4. The main panel will display:
   - Emitter spectra
   - Filter and detector transmission/sensitivity curves
   - Resulting filtered spectrum for each emitter
   - Spectral statistics in the expandable section

## Data Format

The application expects data in Parquet format with specific column naming conventions:

- For emitters:
  - Identifier columns: "company" and "device_id"
  - Type column: "type" (e.g., "LED", "Lamp", "Laser")
  - Wavelength column: "wave_nm"
  - Intensity column: "int_au"

- For filters:
  - Identifier columns: "company" and "device_id"
  - Type column: "type" (should contain keywords like "long", "short", "band")
  - Wavelength column: "wave_nm"
  - Transmission column: "int_au"

- For detectors:
  - Identifier columns: "company" and "device_id"
  - Type column: "type"
  - Wavelength column: "wave_nm"
  - Sensitivity column: "int_au"

## Configuration

The application uses a configuration file (config.toml) to customize behavior:

- Data paths
- Spectrum wavelength range and step size
- Default filter parameters
- Data structure definition and keywords for column identification

## Utility Scripts

The project includes utility scripts for working with data files:

- `parquet_reader.py`: Converts Parquet files to Excel format
- `ods_to_parquet.py`: Converts Excel files to Parquet format

## Requirements

- Python 3.7+
- Streamlit
- Polars
- NumPy
- Plotly
- Loguru
- Pandas

## Project Structure

```
.
├── data/
│   ├── light_sources.parquet   # Sample data
│   ├── light_sources.xlsx      # Source data (Excel format)
│   └── light_sources.ods       # Alternative source data (ODS format)
├── manual_scripts/
│   ├── ods_to_parquet.py       # Utility to convert Excel to Parquet
│   └── parquet_reader.py       # Utility to read Parquet data
├── config.toml                 # Application configuration
├── main.py                     # Main Streamlit application
├── README.md                   # This documentation
└── requirements.txt            # Python dependencies
```

## License

MIT

## Contributing

[Include contribution guidelines here]