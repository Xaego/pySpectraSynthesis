# Spectrum Analyzer

A Streamlit web application for analyzing emission spectra and filters.

## Features

- Display emitter spectrum from local data files
- Model filter spectra with specified parameters
- Calculate and display the resulting spectrum
- Analyze characteristics of the resulting spectrum

## Installation

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install streamlit polars numpy plotly
   ```

## File Structure

```
web_apps/
├── venv/
├── data/
│   └── light_sources/
│       ├── light_sources.ods
│       ├── light_sources.parquet
│       └── light_sources.xlsx
├── manual_scripts/
│   ├── ods_to_parquet.py
│   ├── parquet_reader.py
│   └── main.py
└── README.md
```

## Usage

Run the application using Streamlit:

```
streamlit run main.py
```

## Data Format

The application uses light source spectrum data stored in Parquet format. The data should contain columns for wavelength and intensity values.

## Examples

The application includes demonstration data if no file is loaded:
- Demo Green (550nm)
- Demo Blue (470nm)

## Development

The application is built with:
- Streamlit - Web interface
- Polars - Data manipulation
- NumPy - Numerical operations
- Plotly - Interactive data visualization

## License

[Your License Information]

## Contributors

[Your Name]