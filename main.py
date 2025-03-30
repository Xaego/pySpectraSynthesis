import streamlit as st
import polars as pl
import numpy as np
import plotly.graph_objects as go
import os
import io
import json

# -----------------------------------------------------------------------------
# CONFIGURATION LOADING
# -----------------------------------------------------------------------------
def load_config():
    """
    Loads configuration from config.json or config.toml.
    Returns:
        dict: The configuration dictionary loaded from a file.
    Raises:
        FileNotFoundError: If no configuration file is found.
    """
    import os, json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_paths = [
        os.path.join(script_dir, "config.json"),
        os.path.join(script_dir, "config.toml"),
        os.path.join(os.getcwd(), "config.json"),
        os.path.join(os.getcwd(), "config.toml")
    ]
    for path in config_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    try:
                        return json.load(f)
                    except json.JSONDecodeError:
                        import toml  # Assumes toml is installed
                        f.seek(0)
                        return toml.load(f)
            except Exception as e:
                st.error(f"Error reading config file {path}: {e}")
                raise e
    raise FileNotFoundError("Configuration file not found. Please provide config.json or config.toml.")

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def find_column_by_keywords(df, keywords):
    """
    Finds a DataFrame column whose name contains one of the keywords.
    Args:
        df (pl.DataFrame): DataFrame to search.
        keywords (list): List of keywords.
    Returns:
        str or None: Matching column name or None if not found.
    """
    for col in df.columns:
        col_lower = str(col).lower()
        for keyword in keywords:
            if str(keyword).lower() in col_lower:
                return col
    return None

# -----------------------------------------------------------------------------
# DATA PROCESSING FUNCTIONS
# -----------------------------------------------------------------------------
def process_emitter_dataframe(df, config):
    """
    Extracts emitter data from the DataFrame.
    Args:
        df (pl.DataFrame): Spectrum DataFrame.
        config (dict): Configuration dictionary.
    Returns:
        dict: Processed emitter data.
    """
    emitters = {}
    emitter_config = config.get("data_structure", {}).get("emitters", {})
    identifier_columns = emitter_config.get("identifier_columns", ["company", "device_id"])
    type_column = emitter_config.get("type_column", "type")
    allowed_types = [t.lower() for t in emitter_config.get("type_values", [])]
    wavelength_keywords = emitter_config.get("wavelength_keywords", ["wave_nm", "wavelength"])
    intensity_keywords = emitter_config.get("intensity_keywords", ["int_au", "intensity"])

    has_identifiers = all(col in df.columns for col in identifier_columns)
    has_type = type_column in df.columns
    wavelength_col = find_column_by_keywords(df, wavelength_keywords)
    intensity_col = find_column_by_keywords(df, intensity_keywords)

    if has_identifiers and wavelength_col and intensity_col:
        unique_ids = df.select(identifier_columns + ([type_column] if has_type else [])).unique()
        for row in unique_ids.iter_rows(named=True):
            if has_type:
                type_val = str(row.get(type_column, "")).lower()
                if allowed_types and type_val not in allowed_types:
                    continue
            condition = pl.lit(True)
            for col in identifier_columns:
                condition &= (pl.col(col) == row[col])
            if has_type:
                condition &= (pl.col(type_column) == row[type_column])
            device_df = df.filter(condition)
            wavelengths = device_df[wavelength_col].to_numpy()
            intensities = device_df[intensity_col].to_numpy()
            valid = ~np.isnan(wavelengths) & ~np.isnan(intensities)
            clean_wavelengths = wavelengths[valid]
            clean_intensities = intensities[valid]
            if clean_wavelengths.size and clean_intensities.size:
                name = " ".join(str(row[col]) for col in identifier_columns)
                emitters[name] = {
                    'wavelengths': clean_wavelengths.tolist(),
                    'intensities': clean_intensities.tolist(),
                    'type': row.get(type_column, "Unknown")
                }
    else:
        st.error("Emitter identifiers or required columns missing.")
    return emitters

def process_filter_dataframe(df, config):
    """
    Extracts filter data from the DataFrame using configuration keywords.
    Args:
        df (pl.DataFrame): Spectrum DataFrame.
        config (dict): Configuration dictionary.
    Returns:
        dict: Processed filter data organized by filter category.
    """
    # Include safety_glasses in the filters dictionary
    filters_dict = {"long_pass": {}, "short_pass": {}, "band_pass": {}, "safety_glasses": {}}
    filter_config = config.get("data_structure", {}).get("filters", {})
    identifier_columns = filter_config.get("identifier_columns", ["company", "device_id"])
    type_column = filter_config.get("type_column", "type")
    type_keywords = filter_config.get("type_keywords", {
        "long_pass": ["long", "lp", "long pass"],
        "short_pass": ["short", "sp", "short pass"],
        "band_pass": ["band", "bp", "band pass"],
        "safety_glasses": ["safety glasses"]
    })
    wavelength_keywords = filter_config.get("wavelength_keywords", ["wave_nm", "wavelength"])
    transmission_keywords = filter_config.get("transmission_keywords", ["int_au", "transmission"])

    if not all(col in df.columns for col in identifier_columns + [type_column]):
        st.error("Filter identifiers or type column missing in data.")
        return filters_dict

    wavelength_col = find_column_by_keywords(df, wavelength_keywords)
    transmission_col = find_column_by_keywords(df, transmission_keywords)
    if not (wavelength_col and transmission_col):
        st.error("Required wavelength or transmission columns for filters not found.")
        return filters_dict

    unique_rows = df.select(identifier_columns + [type_column]).unique()
    for row in unique_rows.iter_rows(named=True):
        type_val = str(row.get(type_column, "")).lower()
        filter_category = None
        for category, keywords in type_keywords.items():
            if any(keyword.lower() in type_val for keyword in keywords):
                filter_category = category
                break
        if filter_category is None:
            continue

        filter_name = " ".join(str(row[col]) for col in identifier_columns)
        condition = pl.lit(True)
        for col in identifier_columns:
            condition &= (pl.col(col) == row[col])
        condition &= (pl.col(type_column) == row[type_column])
        filter_rows = df.filter(condition)

        sample_val = filter_rows[transmission_col][0]
        if isinstance(sample_val, (list, np.ndarray)):
            wavelengths = filter_rows[0][wavelength_col]
            transmissions = filter_rows[0][transmission_col]
        else:
            wavelengths = filter_rows[wavelength_col].to_numpy().tolist()
            transmissions = filter_rows[transmission_col].to_numpy().tolist()
        if wavelengths and transmissions:
            filters_dict[filter_category][filter_name] = {
                'wavelengths': wavelengths,
                'transmission': transmissions
            }
    return filters_dict

def process_detector_dataframe(df, config):
    """
    Extracts detector data from the DataFrame using configuration keywords.
    Args:
        df (pl.DataFrame): Spectrum DataFrame.
        config (dict): Configuration dictionary.
    Returns:
        dict: Processed detector data.
    """
    detectors = {}
    detector_config = config.get("data_structure", {}).get("detectors", {})
    identifier_columns = detector_config.get("identifier_columns", ["company", "device_id"])
    type_column = detector_config.get("type_column", "type")
    type_keywords = detector_config.get("type_keywords", {"human_eye": ["eye", "si"]})
    wavelength_keywords = detector_config.get("wavelength_keywords", ["wave_nm", "wavelength"])
    sensitivity_keywords = detector_config.get("sensitivity_keywords", ["int_au", "sensitivity"])

    if not all(col in df.columns for col in identifier_columns + [type_column]):
        st.error("Detector identifiers or type column missing in data.")
        return detectors

    wavelength_col = find_column_by_keywords(df, wavelength_keywords)
    sensitivity_col = find_column_by_keywords(df, sensitivity_keywords)
    if not (wavelength_col and sensitivity_col):
        st.error("Required wavelength or sensitivity columns for detectors not found.")
        return detectors

    unique_rows = df.select(identifier_columns + [type_column]).unique()
    for row in unique_rows.iter_rows(named=True):
        type_val = str(row.get(type_column, "")).lower()
        detector_category = None
        for category, keywords in type_keywords.items():
            if any(keyword.lower() in type_val for keyword in keywords):
                detector_category = category
                break
        if detector_category is None:
            continue

        detector_name = " ".join(str(row[col]) for col in identifier_columns)
        condition = pl.lit(True)
        for col in identifier_columns:
            condition &= (pl.col(col) == row[col])
        condition &= (pl.col(type_column) == row[type_column])
        detector_rows = df.filter(condition)
        if len(detector_rows) == 0:
            continue

        sample_val = detector_rows[sensitivity_col][0]
        if isinstance(sample_val, (list, np.ndarray)):
            wavelengths = detector_rows[0][wavelength_col]
            sensitivities = detector_rows[0][sensitivity_col]
        else:
            wavelengths = detector_rows[wavelength_col].to_numpy().tolist()
            sensitivities = detector_rows[sensitivity_col].to_numpy().tolist()
        if wavelengths and sensitivities:
            detectors[detector_name] = {
                'wavelengths': wavelengths,
                'sensitivity': sensitivities,
                'type': detector_category
            }
    return detectors

# -----------------------------------------------------------------------------
# STREAMLIT APP SETUP & DATA LOADING
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Spectrum Analyzer", page_icon="📊", layout="wide")

# Initialize session state variables
if 'emitters_data' not in st.session_state:
    st.session_state.emitters_data = None
if 'filters_data' not in st.session_state:
    st.session_state.filters_data = None
if 'detectors_data' not in st.session_state:
    st.session_state.detectors_data = None

st.title("Emitter, Filter, and Detector Spectrum Analyzer")
config = load_config()

def load_data():
    """
    Loads and processes spectrum data from a Parquet file.
    Returns:
        tuple: (emitters, filters, detectors)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = []
    for path in config["data_paths"]["parquet_files"]:
        possible_paths.extend([
            path,
            os.path.join(script_dir, path),
            os.path.join(script_dir, "..", path),
            os.path.join(os.getcwd(), path)
        ])

    uploaded_file = st.sidebar.file_uploader("Or upload a file with emitter data", type=["parquet"])
    df = None
    file_path_used = None
    error_messages = []
    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.getvalue()
            df = pl.read_parquet(io.BytesIO(file_bytes))
            file_path_used = "Uploaded file"
            st.sidebar.success("File successfully loaded!")
        except Exception as e:
            error_messages.append(f"Error reading uploaded file: {str(e)}")
    if df is None:
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    df = pl.read_parquet(path)
                    file_path_used = path
                    break
            except Exception as e:
                error_messages.append(f"Error reading file {path}: {str(e)}")
    if df is not None:
        st.sidebar.success(f"Data successfully loaded from: {file_path_used}")
        # Store the raw dataframe for viewer
        st.session_state.raw_df = df
        emitters = process_emitter_dataframe(df, config)
        filters = process_filter_dataframe(df, config)
        detectors = process_detector_dataframe(df, config)
        if not emitters:
            with st.expander("Debug: DataFrame structure"):
                st.write("DataFrame columns:", df.columns)
                st.write("First 5 rows:", df.head(5))
        return emitters, filters, detectors
    else:
        st.error("Failed to load data. Please upload a valid parquet file.")
        with st.expander("Error details"):
            st.code("\n".join(error_messages))
            st.markdown("**Current working directory:** " + os.getcwd())
        return {}, {}, {}

if st.session_state.emitters_data is None or st.session_state.filters_data is None or st.session_state.detectors_data is None:
    st.session_state.emitters_data, st.session_state.filters_data, st.session_state.detectors_data = load_data()

emitters_data = st.session_state.emitters_data
filters_data = st.session_state.filters_data
detectors_data = st.session_state.detectors_data

# -----------------------------------------------------------------------------
# SIDEBAR: PARAMETER & DEVICE SELECTION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Parameters")

    # Wavelength Range Controller
    st.subheader("Wavelength Range")
    min_wl = config["spectrum_settings"]["min_wavelength"]
    max_wl = config["spectrum_settings"]["max_wavelength"]
    wl_range = st.slider(
        "Display Range (nm)",
        min_value=int(min_wl),
        max_value=int(max_wl),
        value=(int(min_wl), int(max_wl)),
        step=10,
        key="wavelength_range"
    )
    display_min_wl, display_max_wl = wl_range

    # --- Emitter Selection with Multiselect ---
    st.subheader("Emitter Selection")
    emitter_types = list(set(data['type'] for data in emitters_data.values())) if emitters_data else []
    emitter_types.sort()
    emitter_types = ["All"] + emitter_types
    selected_type = st.selectbox("Filter by Emitter Type", options=emitter_types, index=0, key="emitter_type_filter") if emitter_types else "All"
    if selected_type == "All":
        filtered_emitters = emitters_data
    else:
        filtered_emitters = {name: data for name, data in emitters_data.items() if data['type'] == selected_type}
    emitter_names = list(filtered_emitters.keys())
    if not emitter_names:
        st.warning(f"No emitters found with type: {selected_type}")
        filtered_emitters = emitters_data
        emitter_names = list(filtered_emitters.keys())
    # Use default emitters from config if available
    default_emitters = config.get("ui", {}).get("default_emitters", emitter_names)
    default_emitters = [e for e in default_emitters if e in emitter_names] or emitter_names
    selected_emitters = st.multiselect("Select Emitters", options=emitter_names, default=default_emitters, key="emitter_selector")

    # --- Filter Selection ---
    st.subheader("Filter Selection")
    filter_type_selection = st.radio(
        "Filter Type",
        options=["Custom", "Library"],
        key="filter_type"
    )

    # Custom Filter Parameters
    filter_params = []
    if filter_type_selection == "Custom":
        # Default values from config
        default_center = config["filter_defaults"]["center"]
        center_step = config["filter_defaults"]["center_step"]
        default_width = config["filter_defaults"]["width"]
        width_step = config["filter_defaults"]["width_step"]

        # Number of custom filters selector
        num_filters = st.number_input("Number of Custom Filters", min_value=0, max_value=5, value=1, step=1)

        for i in range(num_filters):
            with st.expander(f"Filter {i+1}"):
                filter_type = st.selectbox(
                    "Filter Type",
                    options=["band_pass", "long_pass", "short_pass"],
                    key=f"filter_type_{i}"
                )

                if filter_type == "band_pass":
                    center = st.slider(
                        "Center Wavelength (nm)",
                        max_value=float(max_wl),
                        value=float(default_center),
                        step=float(center_step),
                        key=f"center_{i}"
                    )
                    width = st.slider(
                        "Width (nm)",
                        min_value=10.0,
                        max_value=200.0,
                        value=default_width,
                        step=width_step,
                        key=f"width_{i}"
                    )
                    filter_params.append({
                        'type': filter_type,
                        'center': center,
                        'width': width
                    })
                elif filter_type in ["long_pass", "short_pass"]:
                    cutoff = st.slider(
                        "Cutoff Wavelength (nm)",
                        min_value=int(min_wl),
                        max_value=int(max_wl),
                        value=int(default_center),
                        step=int(center_step),
                        key=f"cutoff_{i}"
                    )
                    filter_params.append({
                        'type': filter_type,
                        'cutoff': cutoff
                    })

    # Library Filter Selection
    selected_filters = []
    if filter_type_selection == "Library":
        # Create a list of all available filters with their categories
        filter_options = []
        for category, filters in filters_data.items():
            for filter_name in filters.keys():
                filter_options.append({
                    'type': category,
                    'name': filter_name,
                    'display': f"{category} - {filter_name}"
                })

        # Allow selecting multiple filters from the library
        if filter_options:
            display_options = [f["display"] for f in filter_options]
            selected_displays = st.multiselect(
                "Select Filters from Library",
                options=display_options,
                key="lib_filter_selector"
            )

            # Map selected display options back to the filter data
            for display in selected_displays:
                for filter_option in filter_options:
                    if filter_option["display"] == display:
                        selected_filters.append(filter_option)
                        break
        else:
            st.warning("No filters available in the library.")

    # --- Detector Selection ---
    st.subheader("Detector Selection")
    detector_enabled = st.checkbox("Enable Detector", value=False)

    selected_detector = None
    if detector_enabled:
        detector_names = list(detectors_data.keys())
        if detector_names:
            selected_detector = st.selectbox(
                "Select Detector",
                options=detector_names,
                key="detector_selector"
            )
        else:
            st.warning("No detectors available in the data.")

# -----------------------------------------------------------------------------
# SPECTRUM CREATION & RESULT CALCULATION
# -----------------------------------------------------------------------------
def create_filter_spectrum(center, width, config):
    """
    Creates a bandpass filter spectrum using a super-gaussian function.
    Args:
        center (float): Central wavelength (nm).
        width (float): Bandwidth FWHM (nm).
        config (dict): Configuration dictionary.
    Returns:
        pl.DataFrame: DataFrame with wavelength and transmission.
    """
    min_wl = config["spectrum_settings"]["min_wavelength"]
    max_wl = config["spectrum_settings"]["max_wavelength"]
    step = config["spectrum_settings"]["step"]
    wavelengths = np.arange(min_wl, max_wl, step)
    # Use configurable exponent for super-gaussian function (default=10)
    n = config.get("spectrum_settings", {}).get("filter_exponent", 10)
    transmission = np.exp(-0.5 * ((wavelengths - center) / (width/2))**(2*n))
    return pl.DataFrame({'wavelength': wavelengths, 'transmission': transmission})

def calculate_resulting_spectrum(emitter_data, filter_dfs, detector_df=None, from_library=None):
    """
    Calculates the resulting spectrum by applying filter transmissions and, if enabled, detector response.
    Args:
        emitter_data (dict): Emitter spectrum data.
        filter_dfs (list): List of filter DataFrames.
        detector_df (pl.DataFrame, optional): Detector response DataFrame.
        from_library (list, optional): List of booleans indicating if each filter is from library (in OD) or custom (in transmission).
    Returns:
        pl.DataFrame: Resulting spectrum.
    """
    if not emitter_data or len(filter_dfs) == 0:
        return None

    # If from_library is not provided, assume all filters are in transmission format
    if from_library is None:
        from_library = [False] * len(filter_dfs)

    emitter_df = pl.DataFrame({'wavelength': emitter_data['wavelengths'], 'intensity': emitter_data['intensities']})
    wavelengths = emitter_df['wavelength'].to_numpy()
    intensities = emitter_df['intensity'].to_numpy()
    resulting_intensities = intensities.copy()

    for i, filter_df in enumerate(filter_dfs):
        if filter_df is not None:
            filter_wavelengths = filter_df['wavelength'].to_numpy()
            filter_values = filter_df['transmission'].to_numpy()

            # If filter is from library, it's in optical density format and needs conversion to transmission
            # If it's a custom filter, it's already in transmission format (0-1)
            is_library = from_library[i] if i < len(from_library) else False

            if is_library:
                # Convert optical density to transmission: T = 10^(-OD)
                # Ensure we handle very high OD values gracefully by clipping
                # OD > 10 would mean T < 1e-10, which is effectively zero transmission
                filter_values = np.where(filter_values > 10, 0, 10 ** (-filter_values))

            interp_transmission = np.interp(wavelengths, filter_wavelengths, filter_values, left=0, right=0)
            resulting_intensities *= interp_transmission

    detector_weighted = None
    if detector_df is not None:
        detector_wavelengths = detector_df['wavelength'].to_numpy()
        detector_sensitivities = detector_df['sensitivity'].to_numpy()
        interp_sensitivity = np.interp(wavelengths, detector_wavelengths, detector_sensitivities, left=0, right=0)
        detector_weighted = resulting_intensities * interp_sensitivity

    result_dict = {'wavelength': wavelengths, 'intensity': intensities, 'resulting_intensity': resulting_intensities}
    if detector_weighted is not None:
        result_dict['detector_weighted'] = detector_weighted

    return pl.DataFrame(result_dict)

# Create filter spectra from user selections
filter_spectra = []
from_library = []  # Track which filters are from library (and in OD format)

if filter_type_selection == "Custom":
    for params in filter_params:
        if params['type'] == "band_pass":
            fs = create_filter_spectrum(params['center'], params['width'], config)
            filter_spectra.append(fs)
            from_library.append(False)  # Custom filters are in transmission format
        elif params['type'] == "long_pass":
            min_wl = config["spectrum_settings"]["min_wavelength"]
            max_wl = config["spectrum_settings"]["max_wavelength"]
            step = config["spectrum_settings"]["step"]
            wavelengths = np.arange(min_wl, max_wl, step)
            cutoff = params['cutoff']
            # Use configurable logistic steepness (default=5)
            steepness = config.get("spectrum_settings", {}).get("logistic_steepness", 5)
            transmission = 1 / (1 + np.exp(-(wavelengths - cutoff) / steepness))
            fs = pl.DataFrame({'wavelength': wavelengths, 'transmission': transmission})
            filter_spectra.append(fs)
            from_library.append(False)  # Custom filters are in transmission format
        elif params['type'] == "short_pass":
            min_wl = config["spectrum_settings"]["min_wavelength"]
            max_wl = config["spectrum_settings"]["max_wavelength"]
            step = config["spectrum_settings"]["step"]
            wavelengths = np.arange(min_wl, max_wl, step)
            cutoff = params['cutoff']
            steepness = config.get("spectrum_settings", {}).get("logistic_steepness", 5)
            transmission = 1 / (1 + np.exp((wavelengths - cutoff) / steepness))
            fs = pl.DataFrame({'wavelength': wavelengths, 'transmission': transmission})
            filter_spectra.append(fs)
            from_library.append(False)  # Custom filters are in transmission format
else:
    for selection in selected_filters:
        ftype = selection['type']
        fname = selection['name']
        if ftype in filters_data and fname in filters_data[ftype]:
            fdata = filters_data[ftype][fname]
            fs = pl.DataFrame({'wavelength': fdata['wavelengths'], 'transmission': fdata['transmission']})
            filter_spectra.append(fs)
            from_library.append(True)  # Library filters are in optical density format

detector_df = None
if detector_enabled and selected_detector and selected_detector in detectors_data:
    ddata = detectors_data[selected_detector]
    detector_df = pl.DataFrame({'wavelength': ddata['wavelengths'], 'sensitivity': ddata['sensitivity']})

# -----------------------------------------------------------------------------
# PRELIMINARY GRAPHS: Emitter Spectrum & Filter/Detector Spectra
# -----------------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    if selected_emitters:
        st.subheader("Emitter Spectra")
        fig_em = go.Figure()
        # Plot each selected emitter individually
        for name in selected_emitters:
            data = filtered_emitters[name]
            fig_em.add_trace(go.Scatter(
                x=data['wavelengths'],
                y=data['intensities'],
                mode='lines',
                name=name
            ))
        # Apply the wavelength range from the sidebar
        fig_em.update_layout(
            xaxis_title='Wavelength (nm)',
            yaxis_title='Intensity',
            height=400,
            xaxis=dict(range=[display_min_wl, display_max_wl])
        )
        st.plotly_chart(fig_em, use_container_width=True)

with col2:
    st.subheader("Filter & Detector Spectra")
    if filter_spectra or detector_df is not None:
        fig_fd = go.Figure()
        # Plot filter traces on primary y-axis (as Optical Density)
        if filter_type_selection == "Custom":
            for i, fs in enumerate(filter_spectra):
                params = filter_params[i]
                if params['type'] == "band_pass":
                    label = f"Band pass ({params['center']} nm, ±{params['width']/2} nm)"
                elif params['type'] == "long_pass":
                    label = f"Long pass (cutoff {params['cutoff']} nm)"
                elif params['type'] == "short_pass":
                    label = f"Short pass (cutoff {params['cutoff']} nm)"
                else:
                    label = f"Filter {i+1}"
                # Convert transmission to Optical Density: OD = -log10(T)
                transmission = fs['transmission'].to_numpy()
                optical_density = np.where(transmission > 0, -np.log10(transmission), np.nan)
                fig_fd.add_trace(go.Scatter(
                    x=fs['wavelength'].to_numpy(),
                    y=optical_density,
                    mode='lines',
                    name=label
                ))
        else:
            for i, fs in enumerate(filter_spectra):
                if i < len(selected_filters):
                    sel = selected_filters[i]
                    label = f"{sel['type']} - {sel['name']}"
                else:
                    label = f"Filter {i+1}"
                # Assume library filter data is already in Optical Density
                fig_fd.add_trace(go.Scatter(
                    x=fs['wavelength'].to_numpy(),
                    y=fs['transmission'].to_numpy(),
                    mode='lines',
                    name=label
                ))
        # Plot detector trace on secondary y-axis (Detector Sensitivity)
        if detector_df is not None:
            fig_fd.add_trace(go.Scatter(
                x=detector_df['wavelength'].to_numpy(),
                y=detector_df['sensitivity'].to_numpy(),
                mode='lines',
                name=selected_detector,
                line=dict(dash='dash'),
                yaxis="y2"
            ))
        # Apply the wavelength range from the sidebar
        fig_fd.update_layout(
            xaxis_title='Wavelength (nm)',
            yaxis=dict(title='Optical Density'),
            yaxis2=dict(title='Detector Sensitivity', overlaying='y', side='right'),
            height=400,
            xaxis=dict(range=[display_min_wl, display_max_wl])
        )
        st.plotly_chart(fig_fd, use_container_width=True)

# -----------------------------------------------------------------------------
# RESULTING SPECTRUM: Individual spectra for each emitter
# -----------------------------------------------------------------------------
if selected_emitters:
    st.subheader("Resulting Spectrum")
    fig_res = go.Figure()
    for name in selected_emitters:
        emitter_data = filtered_emitters[name]
        # Calculate resulting spectrum for each emitter separately
        rs = calculate_resulting_spectrum(emitter_data, filter_spectra, detector_df, from_library)
        # Original spectrum - dashed line
        fig_res.add_trace(go.Scatter(
            x=emitter_data['wavelengths'],
            y=emitter_data['intensities'],
            mode='lines',
            name=f"{name} Original",
            line=dict(dash='dash')
        ))
        # Final (filtered) spectrum - solid line
        if rs is not None:
            fig_res.add_trace(go.Scatter(
                x=rs['wavelength'].to_numpy(),
                y=rs['resulting_intensity'].to_numpy(),
                mode='lines',
                name=f"{name} Filtered",
                line=dict(dash='solid')
            ))
            # If detector is enabled, add its response (dashdot)
            if 'detector_weighted' in rs.columns:
                fig_res.add_trace(go.Scatter(
                    x=rs['wavelength'].to_numpy(),
                    y=rs['detector_weighted'].to_numpy(),
                    mode='lines',
                    name=f"{name} Detector Response",
                    line=dict(dash='dashdot')
                ))
    # Apply the wavelength range from the sidebar
    fig_res.update_layout(
        xaxis_title='Wavelength (nm)',
        yaxis_title='Intensity',
        height=500,
        xaxis=dict(range=[display_min_wl, display_max_wl])
    )
    st.plotly_chart(fig_res, use_container_width=True)
else:
    st.info("Please select at least one emitter and one filter to display the resulting spectrum.")

# -----------------------------------------------------------------------------
# STATISTICS
# -----------------------------------------------------------------------------
with st.expander("Resulting Spectrum Statistics"):
    # Calculate statistics for each emitter separately
    for name in selected_emitters:
        emitter_data = filtered_emitters[name]
        rs = calculate_resulting_spectrum(emitter_data, filter_spectra, detector_df, from_library)
        if rs is not None:
            wavelengths = rs['wavelength'].to_numpy()
            result_intensities = rs['resulting_intensity'].to_numpy()
            peak_idx = np.argmax(result_intensities)
            peak_wavelength = wavelengths[peak_idx]
            peak_intensity = result_intensities[peak_idx]
            half_max = peak_intensity / 2
            above_half = wavelengths[result_intensities >= half_max]
            fwhm = (above_half.max() - above_half.min()) if above_half.size > 1 else "N/A"
            # Use np.trapezoid instead of np.trapz
            integral = np.trapezoid(result_intensities, wavelengths)
            original_integral = np.trapezoid(np.array(emitter_data['intensities']), np.array(emitter_data['wavelengths']))
            rel_intensity = (integral / original_integral) * 100 if original_integral else 0
            st.markdown(f"**Emitter: {name}**")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Peak wavelength", f"{peak_wavelength:.1f} nm")
            col_b.metric("FWHM", f"{fwhm if isinstance(fwhm, str) else f'{fwhm:.1f}'} nm")
            col_c.metric("Relative Intensity", f"{rel_intensity:.1f}%")

with st.expander("Project Information"):
    st.markdown("""
    ### About the Project
    This application allows analysis of emitter spectra, filters, and detectors.

    **Features:**
    - Display emitter spectra from a data file.
    - Model custom filter spectra or select from library filters.
    - Apply detector responses.
    - Calculate and display the resulting spectrum for each emitter separately.
    - Preliminary graphs (Emitter Spectra and Filter/Detector Spectra) are retained.

    **Project Structure:**
    ```
    web_apps/
    ├── data/
    │   └── devices.parquet
    ├── config.toml
    └── main.py
    ```
    """)

# -----------------------------------------------------------------------------
# PARQUET FILE VIEWER
# -----------------------------------------------------------------------------
with st.expander("Parquet File Viewer"):
    if "raw_df" in st.session_state and st.session_state.raw_df is not None:
        st.dataframe(st.session_state.raw_df.head(100))
    else:
        st.write("No parquet data available.")
