import streamlit as st
import polars as pl
import numpy as np
import plotly.graph_objects as go
import os
import io

# Page configuration
st.set_page_config(
    page_title="Spectrum Analyzer",
    page_icon="📊",
    layout="wide"
)

# Application title
st.title("Emitter and Filter Spectrum Analyzer")

# Function to load emitter data from Parquet file
def load_emitters_data():
    """
    Loads emitter data from a Parquet file

    Returns:
        dict: Dictionary with emitter data
    """
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Possible file paths
    possible_paths = [
        os.path.join("data", "light_sources", "light_sources.parquet"),  # Relative to working directory
        os.path.join(script_dir, "data", "light_sources", "light_sources.parquet"),  # Relative to script
        os.path.join(script_dir, "..", "data", "light_sources", "light_sources.parquet"),  # One level up
        os.path.join(os.getcwd(), "data", "light_sources", "light_sources.parquet"),  # Explicitly from current working directory
    ]

    # Allow user to upload a file manually if not found automatically
    uploaded_file = st.sidebar.file_uploader("Or upload a file with emitter data",
                                             type=["parquet"])

    # Dictionary to store emitter data
    emitters = {}
    file_path_used = None
    error_messages = []

    # First check the uploaded file, if present
    if uploaded_file is not None:
        try:
            # Read the bytes directly using a BytesIO object
            file_bytes = uploaded_file.getvalue()

            # Read the parquet file with polars
            df = pl.read_parquet(io.BytesIO(file_bytes))
            file_path_used = "Uploaded file"
            st.sidebar.success("File successfully loaded!")

            # Process the dataframe
            emitters = process_dataframe(df)

        except Exception as e:
            error_messages.append(f"Error reading uploaded file: {str(e)}")

    # If no uploaded file or error occurred, try the paths
    if not emitters:
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    # Use polars to read the file
                    df = pl.read_parquet(path)
                    file_path_used = path

                    # Process the dataframe
                    emitters = process_dataframe(df)

                    break
            except Exception as e:
                error_messages.append(f"Error reading file {path}: {str(e)}")

    # If data loaded successfully
    if emitters:
        st.sidebar.success(f"Data successfully loaded from: {file_path_used}")
        return emitters

    # If data could not be loaded, show error and details
    error_details = "\n".join(error_messages)
    st.error(f"Failed to load data from file. Try uploading a file manually through the sidebar.")
    with st.expander("Error details"):
        st.code(error_details)
        st.markdown("**Current working directory:** " + os.getcwd())
        st.markdown("**Try the following:**")
        st.markdown("""
        1. Make sure the file exists and has the correct extension (`.parquet`)
        2. Check the project structure: file should be in `data/light_sources/light_sources.parquet`
        3. Upload the file directly through the upload interface in the sidebar
        4. Make sure the required packages are installed: `pip install polars pyarrow`
        """)

    return {}

# Helper function to process dataframe with data
def process_dataframe(df):
    """
    Processes a DataFrame with spectrum data and extracts necessary columns

    Args:
        df (pl.DataFrame): DataFrame with spectrum data

    Returns:
        dict: Dictionary with processed data
    """
    emitters = {}

    # Check if we need to group by company/device_id
    if "company" in df.columns and "device_id" in df.columns:
        # Get unique combinations of company and device_id
        unique_devices = df.select(["company", "device_id"]).unique()

        # Process each unique device
        for row in unique_devices.iter_rows(named=True):
            company = row["company"]
            device_id = row["device_id"]

            # Filter data for this specific device
            data_df = df.filter(
                (pl.col("company") == company) &
                (pl.col("device_id") == device_id)
            )

            # Find wavelength and intensity columns - use specific column names from the screenshot
            wavelength_col = "wave_nm" if "wave_nm" in data_df.columns else None
            intensity_col = "int_au" if "int_au" in data_df.columns else None

            # If specific columns not found, try generic detection
            if wavelength_col is None or intensity_col is None:
                for col in data_df.columns:
                    col_lower = str(col).lower()
                    if wavelength_col is None and ("wavelength" in col_lower or "длина волны" in col_lower or "нм" in col_lower or "nm" in col_lower):
                        wavelength_col = col
                    if intensity_col is None and ("intensity" in col_lower or "интенсивность" in col_lower or "relative" in col_lower):
                        intensity_col = col

            # If we found the necessary columns
            if wavelength_col is not None and intensity_col is not None:
                # Convert to numpy arrays for processing
                wavelengths = data_df[wavelength_col].to_numpy()
                intensities = data_df[intensity_col].to_numpy()

                # Remove NaN values
                valid_indices = ~np.isnan(wavelengths) & ~np.isnan(intensities)
                clean_wavelengths = wavelengths[valid_indices]
                clean_intensities = intensities[valid_indices]

                # If we have data, add to dictionary
                if len(clean_wavelengths) > 0 and len(clean_intensities) > 0:
                    emitter_name = f"{company} {device_id}"
                    emitters[emitter_name] = {
                        'wavelengths': clean_wavelengths.tolist(),
                        'intensities': clean_intensities.tolist()
                    }
    else:
        # Assume simple structure with wavelength and intensity columns
        wavelength_col = None
        intensity_col = None

        for col in df.columns:
            col_lower = col.lower()
            if "wavelength" in col_lower or "wave_nm" in col_lower or "nm" in col_lower:
                wavelength_col = col
            if "intensity" in col_lower or "int_au" in col_lower or "relative" in col_lower:
                intensity_col = col

        # If we found the necessary columns
        if wavelength_col is not None and intensity_col is not None:
            # Convert to numpy arrays for processing
            wavelengths = df[wavelength_col].to_numpy()
            intensities = df[intensity_col].to_numpy()

            # Remove NaN values
            valid_indices = ~np.isnan(wavelengths) & ~np.isnan(intensities)
            clean_wavelengths = wavelengths[valid_indices]
            clean_intensities = intensities[valid_indices]

            # If we have data, add to dictionary
            if len(clean_wavelengths) > 0 and len(clean_intensities) > 0:
                emitters["Default Emitter"] = {
                    'wavelengths': clean_wavelengths.tolist(),
                    'intensities': clean_intensities.tolist()
                }

    return emitters

# Function to create a model filter spectrum
def create_filter_spectrum(center, width, min_wavelength=350, max_wavelength=800, step=1):
    """
    Creates a model bandpass filter spectrum

    Args:
        center (float): Central wavelength of the filter (nm)
        width (float): Bandwidth FWHM (nm)
        min_wavelength (float): Minimum wavelength (nm)
        max_wavelength (float): Maximum wavelength (nm)
        step (float): Wavelength step (nm)

    Returns:
        pl.DataFrame: DataFrame with wavelengths and transmission coefficient
    """
    wavelengths = np.arange(min_wavelength, max_wavelength, step)

    # Create a bandpass filter using a super-gaussian function
    # for sharper band edges
    n = 10  # Exponent (affects slope steepness)
    transmission = np.exp(-0.5 * ((wavelengths - center) / (width/2))**(2*n))

    return pl.DataFrame({
        'wavelength': wavelengths,
        'transmission': transmission
    })

# Function to calculate the resulting spectrum
def calculate_resulting_spectrum(emitter_data, filter_dfs):
    """
    Calculates the resulting spectrum after applying filters to the emitter.

    Args:
        emitter_data (dict): Emitter spectrum data
        filter_dfs (list): List of filter DataFrames

    Returns:
        pl.DataFrame: DataFrame with the resulting spectrum
    """
    if not emitter_data or len(filter_dfs) == 0:
        return None

    # Create DataFrame for the emitter
    emitter_df = pl.DataFrame({
        'wavelength': emitter_data['wavelengths'],
        'intensity': emitter_data['intensities']
    })

    # Create a copy for the result
    wavelengths = emitter_df['wavelength'].to_numpy()
    intensities = emitter_df['intensity'].to_numpy()
    resulting_intensities = intensities.copy()

    # Apply each filter
    for filter_df in filter_dfs:
        if filter_df is not None:
            # Interpolate filter data to the emitter wavelength grid
            filter_wavelengths = filter_df['wavelength'].to_numpy()
            filter_transmissions = filter_df['transmission'].to_numpy()

            interp_transmission = np.interp(
                wavelengths,
                filter_wavelengths,
                filter_transmissions,
                left=0, right=0
            )
            # Multiply intensity by the transmission coefficient
            resulting_intensities *= interp_transmission

    # Create the result DataFrame
    result = pl.DataFrame({
        'wavelength': wavelengths,
        'intensity': intensities,
        'resulting_intensity': resulting_intensities
    })

    return result

# Create sidebar for parameter input
with st.sidebar:
    st.header("Parameters")

# Load emitter data
emitters_data = load_emitters_data()
emitter_names = list(emitters_data.keys())

# If no data loaded, show demo data
if not emitter_names:
    st.warning("Using demonstration data, as the file could not be loaded. Upload a file through the sidebar to work with real data.")

    # Create demonstration data
    demo_wavelengths = np.arange(350, 800, 1)

    # Create two demo emitters with different spectra
    demo_green = np.exp(-((demo_wavelengths - 550)**2) / (2 * 30**2))
    demo_blue = np.exp(-((demo_wavelengths - 470)**2) / (2 * 25**2))

    emitters_data = {
        "Demo: Green (550nm)": {
            'wavelengths': demo_wavelengths.tolist(),
            'intensities': demo_green.tolist()
        },
        "Demo: Blue (470nm)": {
            'wavelengths': demo_wavelengths.tolist(),
            'intensities': demo_blue.tolist()
        }
    }
    emitter_names = list(emitters_data.keys())

# Continue sidebar setup
with st.sidebar:
    # Select emitter from loaded data
    selected_emitter = st.selectbox("Emitter", emitter_names)

    # Section for adding filters
    st.subheader("Filters")

    # Number of filters
    num_filters = st.number_input("Number of filters", min_value=1, max_value=10, value=1)

    # Create a list to store filter parameters
    filter_params = []

    # Add input fields for each filter
    for i in range(num_filters):
        st.markdown(f"**Filter {i+1}**")
        filter_center = st.number_input(f"Central wavelength (nm)",
                                        min_value=350.0, max_value=800.0,
                                        value=550.0, key=f"center_{i}")
        filter_width = st.number_input(f"Width FWHM (nm)",
                                       min_value=1.0, max_value=200.0,
                                       value=40.0, key=f"width_{i}")
        filter_params.append({
            'center': filter_center,
            'width': filter_width
        })

# Get data for the selected emitter
selected_emitter_data = emitters_data.get(selected_emitter)

# Create filter spectra based on the input parameters
filter_spectra = []
for params in filter_params:
    filter_spectrum = create_filter_spectrum(params['center'], params['width'])
    filter_spectra.append(filter_spectrum)

# Calculate the resulting spectrum
resulting_spectrum = calculate_resulting_spectrum(selected_emitter_data, filter_spectra)

# Create containers for graphs
col1, col2 = st.columns(2)

# Emitter spectrum graph
with col1:
    st.subheader(f"Emitter spectrum: {selected_emitter}")
    if selected_emitter_data:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=selected_emitter_data['wavelengths'],
            y=selected_emitter_data['intensities'],
            mode='lines',
            name='Emitter'
        ))
        fig.update_layout(
            xaxis_title='Wavelength (nm)',
            yaxis_title='Relative intensity',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# Filter spectra graph
with col2:
    st.subheader("Filter spectra")
    if filter_spectra:
        fig = go.Figure()
        for i, filter_spectrum in enumerate(filter_spectra):
            params = filter_params[i]
            fig.add_trace(go.Scatter(
                x=filter_spectrum['wavelength'].to_numpy(),
                y=filter_spectrum['transmission'].to_numpy(),
                mode='lines',
                name=f'Filter {i+1} ({params["center"]} nm, {params["width"]} nm)'
            ))
        fig.update_layout(
            xaxis_title='Wavelength (nm)',
            yaxis_title='Transmission coefficient',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# Resulting spectrum graph
st.subheader("Resulting spectrum")
if resulting_spectrum is not None:
    wavelengths = resulting_spectrum['wavelength'].to_numpy()
    intensities = resulting_spectrum['intensity'].to_numpy()
    result_intensities = resulting_spectrum['resulting_intensity'].to_numpy()

    fig = go.Figure()
    # Original emitter spectrum (with low opacity)
    fig.add_trace(go.Scatter(
        x=wavelengths,
        y=intensities,
        mode='lines',
        name='Original',
        line=dict(color='gray', width=1, dash='dash'),
        opacity=0.5
    ))
    # Resulting spectrum
    fig.add_trace(go.Scatter(
        x=wavelengths,
        y=result_intensities,
        mode='lines',
        name='Resulting',
        line=dict(color='blue', width=2),
    ))
    fig.update_layout(
        xaxis_title='Wavelength (nm)',
        yaxis_title='Relative intensity',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# Information about results
if resulting_spectrum is not None:
    with st.expander("Resulting spectrum statistics"):
        # Find peak wavelength
        result_intensities = resulting_spectrum['resulting_intensity'].to_numpy()
        wavelengths = resulting_spectrum['wavelength'].to_numpy()

        peak_idx = np.argmax(result_intensities)
        peak_wavelength = wavelengths[peak_idx]
        peak_intensity = result_intensities[peak_idx]

        # Calculate FWHM (full width at half maximum)
        half_max = peak_intensity / 2
        above_half_max = wavelengths[result_intensities >= half_max]
        if len(above_half_max) > 1:
            min_wl = above_half_max.min()
            max_wl = above_half_max.max()
            fwhm = max_wl - min_wl
        else:
            fwhm = "Could not calculate"

        # Calculate integral intensity
        intensities = resulting_spectrum['intensity'].to_numpy()
        integral_intensity = np.trapz(
            result_intensities,
            wavelengths
        )

        # Display statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Peak wavelength", f"{peak_wavelength:.1f} nm")
        with col2:
            st.metric("FWHM", f"{fwhm if isinstance(fwhm, str) else fwhm:.1f} nm")
        with col3:
            # Normalize integral intensity relative to original
            original_integral = np.trapz(
                intensities,
                wavelengths
            )
            relative_intensity = (integral_intensity / original_integral) * 100
            st.metric("Relative intensity", f"{relative_intensity:.1f}%")

# Additional information
with st.expander("Project information"):
    st.markdown("""
    ### About the project
    This application allows you to analyze emitter spectra and filters.

    **Functionality:**
    - Display emitter spectrum from local data file
    - Model filter spectra with specified parameters
    - Calculate and display the resulting spectrum
    - Analyze characteristics of the resulting spectrum

    **Project structure:**
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
    """)