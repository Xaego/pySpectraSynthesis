import streamlit as st
import polars as pl
import numpy as np
import plotly.graph_objects as go
import os
import io
import json

# Load configuration from config.json
def load_config():
    """
    Loads configuration from config.json

    Returns:
        dict: Configuration dictionary
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_paths = [
        os.path.join(script_dir, "config.json"),
        os.path.join(script_dir, "config.toml"),  # Added .toml extension
        os.path.join(os.getcwd(), "config.json"),
        os.path.join(os.getcwd(), "config.toml"),  # Added .toml extension
        "config.json",
        "config.toml"  # Added .toml extension
    ]

    for path in config_paths:
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    config = json.load(f)
                return config
        except Exception as e:
            pass

    # Default configuration if file not found
    return {
        "data_paths": {
            "parquet_files": [
                "data/light_sources.parquet"
            ]
        },
        "spectrum_settings": {
            "min_wavelength": 350,
            "max_wavelength": 1000,
            "step": 1
        },
        "filter_defaults": {
            "center": 550.0,
            "center_step": 50.0,
            "width": 40.0,
            "width_step": 5.0
        }
    }

# Page configuration
st.set_page_config(
    page_title="Spectrum Analyzer",
    page_icon="📊",
    layout="wide"
)

# Initialize session state for persistent selections
if 'selected_emitter_index' not in st.session_state:
    st.session_state.selected_emitter_index = 0

# Add session state for emitter data to prevent recalculation
if 'emitters_data' not in st.session_state:
    st.session_state.emitters_data = None

if 'filters_data' not in st.session_state:
    st.session_state.filters_data = None

if 'emitter_graph' not in st.session_state:
    st.session_state.emitter_graph = None

# Application title
st.title("Emitter and Filter Spectrum Analyzer")

# Load config
config = load_config()

# Function to load emitter and filter data from Parquet file
def load_data():
    """
    Loads emitter and filter data from a Parquet file

    Returns:
        tuple: Dictionary with emitter data, Dictionary with filter data
    """
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Possible file paths
    config = load_config()
    possible_paths = []

    # Add paths from config
    for path in config["data_paths"]["parquet_files"]:
        possible_paths.append(path)
        possible_paths.append(os.path.join(script_dir, path))
        possible_paths.append(os.path.join(script_dir, "..", path))
        possible_paths.append(os.path.join(os.getcwd(), path))

    # Add specific paths based on the file structure from the screenshot
    possible_paths.append(os.path.join(script_dir, "data", "light_sources.parquet"))
    possible_paths.append(os.path.join(os.getcwd(), "data", "light_sources.parquet"))
    # Add the direct path shown in the image
    possible_paths.append("data/light_sources.parquet")

    # Allow user to upload a file manually if not found automatically
    uploaded_file = st.sidebar.file_uploader("Or upload a file with emitter data",
                                             type=["parquet"])

    # Dictionary to store emitter data
    emitters = {}
    filters = {}
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

            # Process the dataframe for emitters and filters
            emitters = process_emitter_dataframe(df)
            filters = process_filter_dataframe(df)

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

                    # Process the dataframe for emitters and filters
                    emitters = process_emitter_dataframe(df)
                    filters = process_filter_dataframe(df)

                    break
            except Exception as e:
                error_messages.append(f"Error reading file {path}: {str(e)}")

    # If data loaded successfully
    if emitters:
        st.sidebar.success(f"Data successfully loaded from: {file_path_used}")
        return emitters, filters

    # If data could not be loaded, show error and details
    error_details = "\n".join(error_messages)
    st.error(f"Failed to load data from file. Try uploading a file manually through the sidebar.")
    with st.expander("Error details"):
        st.code(error_details)
        st.markdown("**Current working directory:** " + os.getcwd())
        st.markdown("**Try the following:**")
        st.markdown("""
        1. Make sure the file exists and has the correct extension (`.parquet`)
        2. Check the project structure: file should be in `data/light_sources.parquet`
        3. Upload the file directly through the upload interface in the sidebar
        4. Make sure the required packages are installed: `pip install polars pyarrow`
        """)

    return {}, {}

# Helper function to process dataframe for emitter data
def process_emitter_dataframe(df):
    """
    Processes a DataFrame and extracts emitter data

    Args:
        df (pl.DataFrame): DataFrame with spectrum data

    Returns:
        dict: Dictionary with processed emitter data
    """
    emitters = {}

    # Check if we need to group by company/device_id
    if "company" in df.columns and "device_id" in df.columns:
        # Check if 'type' column exists
        has_type_column = "type" in df.columns

        # Get unique combinations of company and device_id
        if has_type_column:
            unique_devices = df.select(["company", "device_id", "type"]).unique()
        else:
            unique_devices = df.select(["company", "device_id"]).unique()

        # Process each unique device
        for row in unique_devices.iter_rows(named=True):
            company = row["company"]
            device_id = row["device_id"]

            # Get the type if available, otherwise set to "Unknown"
            emitter_type = row["type"] if has_type_column else "Unknown"

            # Filter data for this specific device
            filter_conditions = [
                (pl.col("company") == company) &
                (pl.col("device_id") == device_id)
            ]

            data_df = df.filter(
                filter_conditions[0]
            )

            # Find wavelength and intensity columns
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
                        'intensities': clean_intensities.tolist(),
                        'type': emitter_type
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
                    'intensities': clean_intensities.tolist(),
                    'type': "Unknown"
                }

    return emitters

# Helper function to process dataframe for filter data
def process_filter_dataframe(df):
    """
    Processes a DataFrame and extracts filter data

    Args:
        df (pl.DataFrame): DataFrame with spectrum data

    Returns:
        dict: Dictionary with processed filter data
    """
    filters = {
        "long_pass": {},
        "short_pass": {},
        "band_pass": {}
    }

    # Check if filter data exists in the DataFrame
    # This function can be customized based on your specific file structure
    # For demonstration, we'll try to find columns related to filters

    # Method 1: If filters have their own rows with a type column
    if "filter_type" in df.columns and "filter_name" in df.columns:
        # Get all filter rows
        filter_df = df.filter(pl.col("filter_type").is_not_null())

        # Process each filter type
        for filter_type in ["long_pass", "short_pass", "band_pass"]:
            type_df = filter_df.filter(pl.col("filter_type") == filter_type)

            for row in type_df.iter_rows(named=True):
                filter_name = row["filter_name"]

                # Find wavelength and transmission columns
                wavelength_col = "wavelength" if "wavelength" in type_df.columns else "wave_nm"
                transmission_col = "transmission" if "transmission" in type_df.columns else "trans"

                if wavelength_col in type_df.columns and transmission_col in type_df.columns:
                    filters[filter_type][filter_name] = {
                        'wavelengths': row[wavelength_col],
                        'transmission': row[transmission_col]
                    }

    # If no filter data found, create some demo filters
    if all(len(filters[key]) == 0 for key in filters):
        # Use the spectrum settings from config for demo filters
        config = load_config()
        min_wl = config["spectrum_settings"]["min_wavelength"]
        max_wl = config["spectrum_settings"]["max_wavelength"]
        step = config["spectrum_settings"]["step"]

        wavelengths = np.arange(min_wl, max_wl, step)

        # Create demo long pass filters (transmits longer wavelengths)
        for cutoff in [450, 550, 650]:
            transmission = 1 / (1 + np.exp(-(wavelengths - cutoff) / 5))
            filters["long_pass"][f"LP{cutoff}"] = {
                'wavelengths': wavelengths.tolist(),
                'transmission': transmission.tolist()
            }

        # Create demo short pass filters (transmits shorter wavelengths)
        for cutoff in [450, 550, 650]:
            transmission = 1 / (1 + np.exp((wavelengths - cutoff) / 5))
            filters["short_pass"][f"SP{cutoff}"] = {
                'wavelengths': wavelengths.tolist(),
                'transmission': transmission.tolist()
            }

        # Create demo band pass filters
        for center in [450, 550, 650]:
            for width in [20, 40, 60]:
                transmission = np.exp(-0.5 * ((wavelengths - center) / (width/2))**(2*10))
                filters["band_pass"][f"BP{center}±{width//2}"] = {
                    'wavelengths': wavelengths.tolist(),
                    'transmission': transmission.tolist()
                }

    return filters

# Function to create a model filter spectrum
def create_filter_spectrum(center, width, config):
    """
    Creates a model bandpass filter spectrum

    Args:
        center (float): Central wavelength of the filter (nm)
        width (float): Bandwidth FWHM (nm)
        config (dict): Configuration dictionary

    Returns:
        pl.DataFrame: DataFrame with wavelengths and transmission coefficient
    """
    min_wavelength = config["spectrum_settings"]["min_wavelength"]
    max_wavelength = config["spectrum_settings"]["max_wavelength"]
    step = config["spectrum_settings"]["step"]

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

# Load emitter and filter data - This should be done only once if not already in session state
if st.session_state.emitters_data is None or st.session_state.filters_data is None:
    st.session_state.emitters_data, st.session_state.filters_data = load_data()

emitters_data = st.session_state.emitters_data
filters_data = st.session_state.filters_data

# If no data loaded, show demo data
if not emitters_data:
    st.warning("Using demonstration data, as the file could not be loaded. Upload a file through the sidebar to work with real data.")

    # Create demonstration data
    demo_wavelengths = np.arange(350, 800, 1)

    # Create demo emitters with different spectra
    demo_green = np.exp(-((demo_wavelengths - 550)**2) / (2 * 30**2))
    demo_blue = np.exp(-((demo_wavelengths - 470)**2) / (2 * 25**2))
    demo_red = np.exp(-((demo_wavelengths - 630)**2) / (2 * 20**2))

    emitters_data = {
        "Demo: Green LED (550nm)": {
            'wavelengths': demo_wavelengths.tolist(),
            'intensities': demo_green.tolist(),
            'type': "LED"
        },
        "Demo: Blue Laser (470nm)": {
            'wavelengths': demo_wavelengths.tolist(),
            'intensities': demo_blue.tolist(),
            'type': "Laser"
        },
        "Demo: Red Lamp (630nm)": {
            'wavelengths': demo_wavelengths.tolist(),
            'intensities': demo_red.tolist(),
            'type': "Lamp"
        }
    }
    st.session_state.emitters_data = emitters_data

    # If no filter data, create demo filters in the same format as process_filter_dataframe
    if not filters_data:
        st.session_state.filters_data = process_filter_dataframe(pl.DataFrame())
        filters_data = st.session_state.filters_data

# Get all available emitter types
emitter_types = list(set(data['type'] for data in emitters_data.values()))
emitter_types.sort()
# Add "All" option at the beginning
emitter_types = ["All"] + emitter_types

# Create sidebar for parameter input
with st.sidebar:
    st.header("Parameters")

    # Filter emitters by type
    selected_type = st.selectbox(
        "Filter by Emitter Type",
        options=emitter_types,
        index=0,
        key="emitter_type_filter"
    )

    # Filter emitter list based on selected type
    if selected_type == "All":
        filtered_emitters = emitters_data
    else:
        filtered_emitters = {name: data for name, data in emitters_data.items()
                             if data['type'] == selected_type}

    # Get filtered emitter names
    emitter_names = list(filtered_emitters.keys())

    # If no emitters match the filter, display a warning
    if not emitter_names:
        st.warning(f"No emitters found with type: {selected_type}")
        # Reset to "All" if no emitters match
        selected_type = "All"
        filtered_emitters = emitters_data
        emitter_names = list(filtered_emitters.keys())

    # Create the selectbox widget with a key and default index from session state
    selected_emitter_index = st.selectbox(
        "Emitter",
        options=range(len(emitter_names)),
        format_func=lambda i: emitter_names[i],
        key="emitter_selector",
        index=min(st.session_state.selected_emitter_index, len(emitter_names)-1)
        if emitter_names else 0
    )

    # Save the selected index to session state for persistence
    st.session_state.selected_emitter_index = selected_emitter_index

    # Get the selected emitter name
    selected_emitter = emitter_names[selected_emitter_index] if emitter_names else None

    # Section for adding filters
    st.subheader("Filters")

    # Filter type selection
    filter_type_selection = st.radio(
        "Filter Type",
        ["Custom", "From Library"],
        key="filter_type_selection"
    )

    # Create a list to store filter parameters or selections
    filter_params = []
    selected_filters = []

    if filter_type_selection == "Custom":
        # Number of custom filters
        num_filters = st.number_input("Number of filters", min_value=1, max_value=10, value=1, key="num_filters")

        # Add input fields for each custom filter
        for i in range(num_filters):
            st.markdown(f"**Filter {i+1}**")

            # Use config defaults
            default_center = config["filter_defaults"]["center"]
            default_width = config["filter_defaults"]["width"]
            center_step = config["filter_defaults"]["center_step"]
            width_step = config["filter_defaults"]["width_step"]

            min_wl = config["spectrum_settings"]["min_wavelength"]
            max_wl = config["spectrum_settings"]["max_wavelength"]

            # Select filter type
            filter_type = st.selectbox(
                "Filter Type",
                ["band_pass", "long_pass", "short_pass"],
                key=f"custom_filter_{i}_type"
            )

            if filter_type == "band_pass":
                filter_center = st.number_input(
                    f"Central wavelength (nm)",
                    min_value=float(min_wl),
                    max_value=float(max_wl),
                    value=float(default_center),
                    step=float(center_step),
                    key=f"filter_{i}_center"
                )

                filter_width = st.number_input(
                    f"Width FWHM (nm)",
                    min_value=1.0,
                    max_value=float(max_wl - min_wl),
                    value=float(default_width),
                    step=float(width_step),
                    key=f"filter_{i}_width"
                )

                filter_params.append({
                    'type': filter_type,
                    'center': filter_center,
                    'width': filter_width
                })
            else:  # long_pass or short_pass
                filter_cutoff = st.number_input(
                    f"Cutoff wavelength (nm)",
                    min_value=float(min_wl),
                    max_value=float(max_wl),
                    value=float(default_center),
                    step=float(center_step),
                    key=f"filter_{i}_cutoff"
                )

                filter_params.append({
                    'type': filter_type,
                    'cutoff': filter_cutoff
                })
    else:  # "From Library"
        # Number of library filters
        num_filters = st.number_input("Number of filters", min_value=1, max_value=5, value=1, key="num_lib_filters")

        # Add selection fields for each library filter
        for i in range(num_filters):
            st.markdown(f"**Filter {i+1}**")

            # Select filter type
            filter_type = st.selectbox(
                "Filter Type",
                list(filters_data.keys()),
                key=f"lib_filter_{i}_type"
            )

            # Get filter names for the selected type
            filter_names = list(filters_data[filter_type].keys())

            if filter_names:
                # Select filter from the type
                filter_name = st.selectbox(
                    "Filter",
                    filter_names,
                    key=f"lib_filter_{i}_name"
                )

                selected_filters.append({
                    'type': filter_type,
                    'name': filter_name
                })
            else:
                st.warning(f"No filters available for type: {filter_type}")

# Get data for the selected emitter
selected_emitter_data = filtered_emitters.get(selected_emitter) if selected_emitter else None

# Create filter spectra based on the input parameters or selections
filter_spectra = []

if filter_type_selection == "Custom":
    for params in filter_params:
        if params['type'] == "band_pass":
            # Create a bandpass filter
            filter_spectrum = create_filter_spectrum(params['center'], params['width'], config)
            filter_spectra.append(filter_spectrum)
        elif params['type'] == "long_pass":
            # Create a long pass filter (transmits longer wavelengths)
            min_wl = config["spectrum_settings"]["min_wavelength"]
            max_wl = config["spectrum_settings"]["max_wavelength"]
            step = config["spectrum_settings"]["step"]

            wavelengths = np.arange(min_wl, max_wl, step)
            cutoff = params['cutoff']

            # Create transmission curve (sigmoid function)
            transmission = 1 / (1 + np.exp(-(wavelengths - cutoff) / 5))

            filter_spectrum = pl.DataFrame({
                'wavelength': wavelengths,
                'transmission': transmission
            })
            filter_spectra.append(filter_spectrum)
        elif params['type'] == "short_pass":
            # Create a short pass filter (transmits shorter wavelengths)
            min_wl = config["spectrum_settings"]["min_wavelength"]
            max_wl = config["spectrum_settings"]["max_wavelength"]
            step = config["spectrum_settings"]["step"]

            wavelengths = np.arange(min_wl, max_wl, step)
            cutoff = params['cutoff']

            # Create transmission curve (inverse sigmoid function)
            transmission = 1 / (1 + np.exp((wavelengths - cutoff) / 5))

            filter_spectrum = pl.DataFrame({
                'wavelength': wavelengths,
                'transmission': transmission
            })
            filter_spectra.append(filter_spectrum)
else:  # "From Library"
    for filter_selection in selected_filters:
        filter_type = filter_selection['type']
        filter_name = filter_selection['name']

        if filter_type in filters_data and filter_name in filters_data[filter_type]:
            filter_data = filters_data[filter_type][filter_name]

            filter_spectrum = pl.DataFrame({
                'wavelength': filter_data['wavelengths'],
                'transmission': filter_data['transmission']
            })
            filter_spectra.append(filter_spectrum)

# Calculate the resulting spectrum
resulting_spectrum = calculate_resulting_spectrum(selected_emitter_data, filter_spectra) if selected_emitter_data else None

# Create containers for graphs
col1, col2 = st.columns(2)

# Generate emitter spectrum figure only if it's not in session state or if emitter changed
emitter_key = f"{selected_emitter}_graph" if selected_emitter else None
if emitter_key and emitter_key not in st.session_state and selected_emitter_data:
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
    st.session_state[emitter_key] = fig

# Emitter spectrum graph - Using the cached figure to prevent re-rendering
with col1:
    if selected_emitter:
        st.subheader(f"Emitter spectrum: {selected_emitter}")
        emitter_type = selected_emitter_data.get('type', 'Unknown')
        st.caption(f"Type: {emitter_type}")

        if emitter_key in st.session_state:
            # Use the cached figure
            st.plotly_chart(st.session_state[emitter_key], use_container_width=True)

# Filter spectra graph
with col2:
    st.subheader("Filter spectra")
    if filter_spectra:
        fig = go.Figure()

        if filter_type_selection == "Custom":
            for i, filter_spectrum in enumerate(filter_spectra):
                params = filter_params[i]
                if params['type'] == "band_pass":
                    filter_name = f"Band pass ({params['center']} nm, ±{params['width']/2} nm)"
                elif params['type'] == "long_pass":
                    filter_name = f"Long pass (cutoff {params['cutoff']} nm)"
                elif params['type'] == "short_pass":
                    filter_name = f"Short pass (cutoff {params['cutoff']} nm)"
                else:
                    filter_name = f"Filter {i+1}"

                fig.add_trace(go.Scatter(
                    x=filter_spectrum['wavelength'].to_numpy(),
                    y=filter_spectrum['transmission'].to_numpy(),
                    mode='lines',
                    name=filter_name
                ))
        else:  # "From Library"
            for i, filter_spectrum in enumerate(filter_spectra):
                if i < len(selected_filters):
                    filter_selection = selected_filters[i]
                    filter_name = f"{filter_selection['type']} - {filter_selection['name']}"
                else:
                    filter_name = f"Filter {i+1}"

                fig.add_trace(go.Scatter(
                    x=filter_spectrum['wavelength'].to_numpy(),
                    y=filter_spectrum['transmission'].to_numpy(),
                    mode='lines',
                    name=filter_name
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
    - Filter emitters by type (Lamp, Laser, LED)
    - Model filter spectra with specified parameters
    - Select filters from a library of pre-defined filters
    - Calculate and display the resulting spectrum
    - Analyze characteristics of the resulting spectrum

    **Project structure:**
    ```
    web_apps/
    ├── venv/
    ├── data/
    │   └── light_sources.parquet
    ├── manual_scripts/
    │   ├── ods_to_parquet.py
    │   ├── parquet_reader.py
    │   └── main.py
    ├── config.toml
    └── README.md
    ```
    """)