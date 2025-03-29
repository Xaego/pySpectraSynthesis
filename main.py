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
        dict: Configuration dictionary.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_paths = [
        os.path.join(script_dir, "config.json"),
        os.path.join(script_dir, "config.toml"),
        os.path.join(os.getcwd(), "config.json"),
        os.path.join(os.getcwd(), "config.toml"),
        "config.json",
        "config.toml"
    ]
    for path in config_paths:
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    try:
                        config = json.load(f)
                    except json.JSONDecodeError:
                        import toml  # Assumes toml is installed
                        f.seek(0)
                        config = toml.load(f)
                return config
        except Exception:
            continue
    # Fallback default configuration
    return {
        "data_paths": {"parquet_files": ["data/light_sources.parquet"]},
        "spectrum_settings": {"min_wavelength": 350, "max_wavelength": 1000, "step": 1},
        "filter_defaults": {"center": 550.0, "center_step": 50.0, "width": 40.0, "width_step": 5.0},
        "detector_defaults": {"peak": 555.0, "sensitivity": 1.0},
        "data_structure": {
            "emitters": {
                "identifier_columns": ["company", "device_id"],
                "type_column": "type",
                "type_values": ["Lamp", "LED", "Laser"],
                "wavelength_keywords": ["wave_nm", "wavelength"],
                "intensity_keywords": ["int_au", "intensity"]
            },
            "filters": {
                "identifier_columns": ["company", "device_id"],
                "type_column": "type",
                "type_keywords": {
                    "long_pass": ["long", "lp", "long pass"],
                    "short_pass": ["short", "sp", "short pass"],
                    "band_pass": ["band", "bp", "band pass"]
                },
                "wavelength_keywords": ["wave_nm", "wavelength"],
                "transmission_keywords": ["int_au", "transmission"]
            },
            "detectors": {
                "identifier_columns": ["company", "device_id"],
                "type_column": "type",
                "type_keywords": {"human_eye": ["eye", "si"]},
                "wavelength_keywords": ["wave_nm", "wavelength"],
                "sensitivity_keywords": ["int_au", "sensitivity"]
            }
        }
    }

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
    filters_dict = {"long_pass": {}, "short_pass": {}, "band_pass": {}}
    filter_config = config.get("data_structure", {}).get("filters", {})
    identifier_columns = filter_config.get("identifier_columns", ["company", "device_id"])
    type_column = filter_config.get("type_column", "type")
    type_keywords = filter_config.get("type_keywords", {
        "long_pass": ["long", "lp", "long pass"],
        "short_pass": ["short", "sp", "short pass"],
        "band_pass": ["band", "bp", "band pass"]
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
    possible_paths.append(os.path.join(script_dir, "data", "light_sources.parquet"))
    possible_paths.append(os.path.join(os.getcwd(), "data", "light_sources.parquet"))
    possible_paths.append("data/light_sources.parquet")

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
    # --- Emitter Selection with Multiselect ---
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
    # Multiselect for emitter names (default: all)
    selected_emitters = st.multiselect("Select Emitters", options=emitter_names, default=emitter_names, key="emitter_selector")

    # --- Filter Selection ---
    st.subheader("Filters")
    has_library_filters = any(len(filters_data.get(ft, {})) > 0 for ft in ["long_pass", "short_pass", "band_pass"])
    filter_options = ["Custom"] + (["From Library"] if has_library_filters else [])
    filter_type_selection = st.radio("Filter Type", filter_options, key="filter_type_selection")
    filter_params = []  # for custom filters
    selected_filters = []  # for library filters
    if filter_type_selection == "Custom":
        num_filters = st.number_input("Number of filters", min_value=0, max_value=10, value=1, key="num_filters")
        for i in range(num_filters):
            st.markdown(f"**Filter {i+1}**")
            default_center = config["filter_defaults"]["center"]
            default_width = config["filter_defaults"]["width"]
            center_step = config["filter_defaults"]["center_step"]
            width_step = config["filter_defaults"]["width_step"]
            min_wl = config["spectrum_settings"]["min_wavelength"]
            max_wl = config["spectrum_settings"]["max_wavelength"]
            ftype = st.selectbox("Filter Type", ["band_pass", "long_pass", "short_pass"], key=f"custom_filter_{i}_type")
            if ftype == "band_pass":
                filter_center = st.number_input("Central wavelength (nm)", min_value=float(min_wl), max_value=float(max_wl),
                                                value=float(default_center), step=float(center_step), key=f"filter_{i}_center")
                filter_width = st.number_input("Width FWHM (nm)", min_value=1.0, max_value=float(max_wl - min_wl),
                                               value=float(default_width), step=float(width_step), key=f"filter_{i}_width")
                filter_params.append({'type': ftype, 'center': filter_center, 'width': filter_width})
            else:
                filter_cutoff = st.number_input("Cutoff wavelength (nm)", min_value=float(min_wl), max_value=float(max_wl),
                                                value=float(default_center), step=float(center_step), key=f"filter_{i}_cutoff")
                filter_params.append({'type': ftype, 'cutoff': filter_cutoff})
    else:  # "From Library"
        num_filters = st.number_input("Number of filters", min_value=0, max_value=5, value=1, key="num_lib_filters")
        for i in range(num_filters):
            st.markdown(f"**Filter {i+1}**")
            available_filter_types = [ft for ft in ["long_pass", "short_pass", "band_pass"]
                                      if len(filters_data.get(ft, {})) > 0]
            ftype = st.selectbox("Filter Type", available_filter_types, key=f"lib_filter_{i}_type") if available_filter_types else None
            if ftype:
                filter_names = list(filters_data[ftype].keys())
                if filter_names:
                    fname = st.selectbox("Filter", filter_names, key=f"lib_filter_{i}_name")
                    selected_filters.append({'type': ftype, 'name': fname})
                else:
                    st.warning(f"No filters available for type: {ftype}")
    # --- Detector Selection ---
    st.subheader("Detector")
    detector_enabled = st.checkbox("Enable Detector", value=False, key="detector_enabled")
    selected_detector = None
    if detector_enabled:
        if not detectors_data:
            st.warning("No detector data found in the file. Please upload a file with detector data.")
        else:
            detector_names = list(detectors_data.keys())
            if detector_names:
                selected_detector_name = st.selectbox("Select Detector", detector_names, key="selected_detector")
                if selected_detector_name in detectors_data:
                    detector_type = detectors_data[selected_detector_name].get('type', 'Unknown')
                    st.info(f"Detector type: {detector_type}")
                selected_detector = selected_detector_name
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
    n = 10  # Super-Gaussian exponent for sharper edges
    transmission = np.exp(-0.5 * ((wavelengths - center) / (width/2))**(2*n))
    return pl.DataFrame({'wavelength': wavelengths, 'transmission': transmission})

def calculate_resulting_spectrum(emitter_data, filter_dfs, detector_df=None):
    """
    Calculates the resulting spectrum by applying filter transmissions and, if enabled, detector response.
    Args:
        emitter_data (dict): Emitter spectrum data.
        filter_dfs (list): List of filter DataFrames.
        detector_df (pl.DataFrame, optional): Detector response DataFrame.
    Returns:
        pl.DataFrame: Resulting spectrum.
    """
    if not emitter_data or len(filter_dfs) == 0:
        return None
    emitter_df = pl.DataFrame({'wavelength': emitter_data['wavelengths'], 'intensity': emitter_data['intensities']})
    wavelengths = emitter_df['wavelength'].to_numpy()
    intensities = emitter_df['intensity'].to_numpy()
    resulting_intensities = intensities.copy()
    for filter_df in filter_dfs:
        if filter_df is not None:
            filter_wavelengths = filter_df['wavelength'].to_numpy()
            filter_transmissions = filter_df['transmission'].to_numpy()
            interp_transmission = np.interp(wavelengths, filter_wavelengths, filter_transmissions, left=0, right=0)
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
if filter_type_selection == "Custom":
    for params in filter_params:
        if params['type'] == "band_pass":
            fs = create_filter_spectrum(params['center'], params['width'], config)
            filter_spectra.append(fs)
        elif params['type'] == "long_pass":
            min_wl = config["spectrum_settings"]["min_wavelength"]
            max_wl = config["spectrum_settings"]["max_wavelength"]
            step = config["spectrum_settings"]["step"]
            wavelengths = np.arange(min_wl, max_wl, step)
            cutoff = params['cutoff']
            transmission = 1 / (1 + np.exp(-(wavelengths - cutoff) / 5))
            fs = pl.DataFrame({'wavelength': wavelengths, 'transmission': transmission})
            filter_spectra.append(fs)
        elif params['type'] == "short_pass":
            min_wl = config["spectrum_settings"]["min_wavelength"]
            max_wl = config["spectrum_settings"]["max_wavelength"]
            step = config["spectrum_settings"]["step"]
            wavelengths = np.arange(min_wl, max_wl, step)
            cutoff = params['cutoff']
            transmission = 1 / (1 + np.exp((wavelengths - cutoff) / 5))
            fs = pl.DataFrame({'wavelength': wavelengths, 'transmission': transmission})
            filter_spectra.append(fs)
else:
    for selection in selected_filters:
        ftype = selection['type']
        fname = selection['name']
        if ftype in filters_data and fname in filters_data[ftype]:
            fdata = filters_data[ftype][fname]
            fs = pl.DataFrame({'wavelength': fdata['wavelengths'], 'transmission': fdata['transmission']})
            filter_spectra.append(fs)

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
        fig_em.update_layout(xaxis_title='Wavelength (nm)', yaxis_title='Intensity', height=400)
        st.plotly_chart(fig_em, use_container_width=True)
with col2:
    st.subheader("Filter & Detector Spectra")
    if filter_spectra or detector_df is not None:
        fig_fd = go.Figure()
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
                fig_fd.add_trace(go.Scatter(
                    x=fs['wavelength'].to_numpy(),
                    y=fs['transmission'].to_numpy(),
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
                fig_fd.add_trace(go.Scatter(
                    x=fs['wavelength'].to_numpy(),
                    y=fs['transmission'].to_numpy(),
                    mode='lines',
                    name=label
                ))
        if detector_df is not None:
            fig_fd.add_trace(go.Scatter(
                x=detector_df['wavelength'].to_numpy(),
                y=detector_df['sensitivity'].to_numpy(),
                mode='lines',
                name=selected_detector,
                line=dict(dash='dash')
            ))
        fig_fd.update_layout(xaxis_title='Wavelength (nm)', yaxis_title='Transmission / Sensitivity', height=400)
        st.plotly_chart(fig_fd, use_container_width=True)

# -----------------------------------------------------------------------------
# RESULTING SPECTRUM: Отдельные спектры для каждого эмиттера
# -----------------------------------------------------------------------------
if selected_emitters:
    st.subheader("Resulting Spectrum")
    fig_res = go.Figure()
    for name in selected_emitters:
        emitter_data = filtered_emitters[name]
        # Вычисляем результирующий спектр для каждого эмиттера отдельно
        rs = calculate_resulting_spectrum(emitter_data, filter_spectra, detector_df)
        # Оригинальный спектр – пунктир
        fig_res.add_trace(go.Scatter(
            x=emitter_data['wavelengths'],
            y=emitter_data['intensities'],
            mode='lines',
            name=f"{name} Original",
            line=dict(dash='dash')
        ))
        # Финальный (отфильтрованный) спектр – сплошная линия
        if rs is not None:
            fig_res.add_trace(go.Scatter(
                x=rs['wavelength'].to_numpy(),
                y=rs['resulting_intensity'].to_numpy(),
                mode='lines',
                name=f"{name} Filtered",
                line=dict(dash='solid')
            ))
            # Если включён детектор, добавляем его отклик (например, dashdot)
            if 'detector_weighted' in rs.columns:
                fig_res.add_trace(go.Scatter(
                    x=rs['wavelength'].to_numpy(),
                    y=rs['detector_weighted'].to_numpy(),
                    mode='lines',
                    name=f"{name} Detector Response",
                    line=dict(dash='dashdot')
                ))
    fig_res.update_layout(xaxis_title='Wavelength (nm)', yaxis_title='Intensity', height=500)
    st.plotly_chart(fig_res, use_container_width=True)
else:
    st.info("Please select at least one emitter and one filter to display the resulting spectrum.")

# -----------------------------------------------------------------------------
# STATISTICS
# -----------------------------------------------------------------------------
with st.expander("Resulting Spectrum Statistics"):
    # Вычисляем статистику для каждого эмиттера отдельно
    for name in selected_emitters:
        emitter_data = filtered_emitters[name]
        rs = calculate_resulting_spectrum(emitter_data, filter_spectra, detector_df)
        if rs is not None:
            wavelengths = rs['wavelength'].to_numpy()
            result_intensities = rs['resulting_intensity'].to_numpy()
            peak_idx = np.argmax(result_intensities)
            peak_wavelength = wavelengths[peak_idx]
            peak_intensity = result_intensities[peak_idx]
            half_max = peak_intensity / 2
            above_half = wavelengths[result_intensities >= half_max]
            fwhm = (above_half.max() - above_half.min()) if above_half.size > 1 else "N/A"
            # Используем np.trapezoid вместо np.trapz
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
    │   └── light_sources.parquet
    ├── config.toml
    └── main.py
    ```
    """)
