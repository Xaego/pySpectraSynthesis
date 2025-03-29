import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import io

# Настройка страницы
st.set_page_config(
    page_title="Анализатор спектров",
    page_icon="📊",
    layout="wide"
)

# Заголовок приложения
st.title("Анализатор спектров излучателей и фильтров")

# Функция для загрузки данных излучателей из Excel файла
def load_emitters_data():
    """
    Загружает данные об излучателях из Excel-файла (ODS/XLS/XLSX)

    Returns:
        dict: Словарь с данными излучателей
    """
    # Получаем текущий каталог скрипта
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Варианты путей к файлу
    possible_paths = [
        os.path.join("data", "light_sources", "light_sources.ods"),  # Относительно рабочего каталога
        os.path.join(script_dir, "data", "light_sources", "light_sources.ods"),  # Относительно скрипта
        os.path.join(script_dir, "..", "data", "light_sources", "light_sources.ods"),  # Один уровень вверх
        os.path.join(os.getcwd(), "data", "light_sources", "light_sources.ods"),  # Явно из текущего рабочего каталога
        # Альтернативные расширения файлов
        os.path.join("data", "light_sources", "Light_sources.xlsx"),
        os.path.join("data", "light_sources", "Light_sources.xls"),
        os.path.join(script_dir, "data", "light_sources", "Light_sources.xlsx"),
        os.path.join(script_dir, "data", "light_sources", "Light_sources.xls"),
    ]

    # Позволяем пользователю загрузить файл вручную, если автоматически не находим
    uploaded_file = st.sidebar.file_uploader("Или загрузите файл с данными излучателей",
                                             type=["ods", "xlsx", "xls"])

    # Словарь для хранения данных излучателей
    emitters = {}
    file_path_used = None
    error_messages = []

    # Сначала проверяем загруженный файл, если он есть
    if uploaded_file is not None:
        try:
            # Для загруженного файла используем pandas
            df_dict = pd.read_excel(uploaded_file, sheet_name=None)
            file_path_used = "Загруженный файл"
            st.sidebar.success("Файл успешно загружен!")

            # Обрабатываем каждый лист
            for sheet_name, df in df_dict.items():
                emitters.update(process_dataframe(sheet_name, df))

        except Exception as e:
            error_messages.append(f"Ошибка при чтении загруженного файла: {str(e)}")

    # Если загруженного файла нет или при его чтении возникла ошибка, пробуем пути
    if not emitters:
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    # Используем pandas для чтения файла
                    df_dict = pd.read_excel(path, sheet_name=None)
                    file_path_used = path

                    # Обрабатываем каждый лист
                    for sheet_name, df in df_dict.items():
                        emitters.update(process_dataframe(sheet_name, df))

                    break
            except Exception as e:
                error_messages.append(f"Ошибка при чтении файла {path}: {str(e)}")

    # Если данные загружены успешно
    if emitters:
        st.sidebar.success(f"Данные успешно загружены из: {file_path_used}")
        return emitters

    # Если данные не удалось загрузить, выводим ошибку и подробности
    error_details = "\n".join(error_messages)
    st.error(f"Не удалось загрузить данные из файла. Попробуйте загрузить файл вручную через боковую панель.")
    with st.expander("Подробности ошибки"):
        st.code(error_details)
        st.markdown("**Текущий рабочий каталог:** " + os.getcwd())
        st.markdown("**Попробуйте следующее:**")
        st.markdown("""
        1. Убедитесь, что файл существует и имеет правильное расширение (`.ods`, `.xlsx` или `.xls`)
        2. Проверьте структуру проекта: файл должен находиться в `data/light_sources/light_sources.ods`
        3. Загрузите файл напрямую через интерфейс загрузки в боковой панели
        4. Убедитесь, что установлены пакеты для работы с Excel: `pip install pandas openpyxl odfpy`
        """)

    return {}

# Вспомогательная функция для обработки DataFrame с данными
def process_dataframe(sheet_name, df):
    """
    Обрабатывает DataFrame с данными спектра и извлекает нужные столбцы

    Args:
        sheet_name (str): Имя листа/спектра
        df (pd.DataFrame): DataFrame с данными

    Returns:
        dict: Словарь с обработанными данными
    """
    emitters = {}

    # Проверяем наличие нужных столбцов
    wavelength_col = None
    intensity_col = None

    for col in df.columns:
        col_lower = str(col).lower()
        if 'wavelength' in col_lower or 'длина волны' in col_lower or 'нм' in col_lower or 'nm' in col_lower:
            wavelength_col = col
        if 'intensity' in col_lower or 'интенсивность' in col_lower or 'relative' in col_lower:
            intensity_col = col

    # Если нашли нужные столбцы
    if wavelength_col is not None and intensity_col is not None:
        # Извлекаем данные, игнорируя NaN значения
        wavelengths = []
        intensities = []

        for _, row in df.iterrows():
            try:
                wavelength = float(row[wavelength_col])
                intensity = float(row[intensity_col])
                if not (np.isnan(wavelength) or np.isnan(intensity)):
                    wavelengths.append(wavelength)
                    intensities.append(intensity)
            except (ValueError, TypeError):
                # Пропускаем ошибочные строки
                pass

        # Если есть данные, добавляем в словарь
        if wavelengths and intensities:
            emitters[sheet_name] = {
                'wavelengths': wavelengths,
                'intensities': intensities
            }

    return emitters

# Функция для создания модельного спектра фильтра
def create_filter_spectrum(center, width, min_wavelength=350, max_wavelength=800, step=1):
    """
    Создает модельный спектр полосового фильтра

    Args:
        center (float): Центральная длина волны фильтра (нм)
        width (float): Ширина полосы пропускания FWHM (нм)
        min_wavelength (float): Минимальная длина волны (нм)
        max_wavelength (float): Максимальная длина волны (нм)
        step (float): Шаг по длине волны (нм)

    Returns:
        pd.DataFrame: DataFrame с длинами волн и коэффициентом пропускания
    """
    wavelengths = np.arange(min_wavelength, max_wavelength, step)

    # Создаем полосовой фильтр, использую функцию супергаусса
    # для более резких краев полосы пропускания
    n = 10  # Показатель степени (влияет на крутизну склонов)
    transmission = np.exp(-0.5 * ((wavelengths - center) / (width/2))**(2*n))

    return pd.DataFrame({
        'wavelength': wavelengths,
        'transmission': transmission
    })

# Функция для расчета результирующего спектра
def calculate_resulting_spectrum(emitter_data, filter_dfs):
    """
    Рассчитывает результирующий спектр после применения фильтров к излучателю.

    Args:
        emitter_data (dict): Данные спектра излучателя
        filter_dfs (list): Список DataFrame'ов фильтров

    Returns:
        pd.DataFrame: DataFrame с результирующим спектром
    """
    if not emitter_data or len(filter_dfs) == 0:
        return None

    # Создаем DataFrame для излучателя
    emitter_df = pd.DataFrame({
        'wavelength': emitter_data['wavelengths'],
        'intensity': emitter_data['intensities']
    })

    result = emitter_df.copy()
    result['resulting_intensity'] = result['intensity']

    # Применяем каждый фильтр
    for filter_df in filter_dfs:
        if filter_df is not None:
            # Интерполяция данных фильтра на сетку длин волн излучателя
            interp_transmission = np.interp(
                result['wavelength'],
                filter_df['wavelength'],
                filter_df['transmission'],
                left=0, right=0
            )
            # Умножаем интенсивность на коэффициент пропускания
            result['resulting_intensity'] *= interp_transmission

    return result

# Создаем боковую панель для ввода параметров
with st.sidebar:
    st.header("Параметры")

# Загружаем данные излучателей
emitters_data = load_emitters_data()
emitter_names = list(emitters_data.keys())

# Если данные не загружены, показываем демо данные
if not emitter_names:
    st.warning("Используются демонстрационные данные, так как не удалось загрузить файл. Загрузите файл через боковую панель для работы с реальными данными.")

    # Создаем демонстрационные данные
    demo_wavelengths = np.arange(350, 800, 1)

    # Создаем два демо-излучателя с разными спектрами
    demo_green = np.exp(-((demo_wavelengths - 550)**2) / (2 * 30**2))
    demo_blue = np.exp(-((demo_wavelengths - 470)**2) / (2 * 25**2))

    emitters_data = {
        "Демо: Зеленый (550нм)": {
            'wavelengths': demo_wavelengths.tolist(),
            'intensities': demo_green.tolist()
        },
        "Демо: Синий (470нм)": {
            'wavelengths': demo_wavelengths.tolist(),
            'intensities': demo_blue.tolist()
        }
    }
    emitter_names = list(emitters_data.keys())

# Продолжаем настройку боковой панели
with st.sidebar:
    # Выбор излучателя из загруженных данных
    selected_emitter = st.selectbox("Излучатель", emitter_names)

    # Секция для добавления фильтров
    st.subheader("Фильтры")

    # Количество фильтров
    num_filters = st.number_input("Количество фильтров", min_value=1, max_value=10, value=1)

    # Создаем список для хранения параметров фильтров
    filter_params = []

    # Добавляем поля ввода для каждого фильтра
    for i in range(num_filters):
        st.markdown(f"**Фильтр {i+1}**")
        filter_center = st.number_input(f"Центральная длина волны (нм)",
                                        min_value=350.0, max_value=800.0,
                                        value=550.0, key=f"center_{i}")
        filter_width = st.number_input(f"Ширина FWHM (нм)",
                                       min_value=1.0, max_value=200.0,
                                       value=40.0, key=f"width_{i}")
        filter_params.append({
            'center': filter_center,
            'width': filter_width
        })

# Получаем данные выбранного излучателя
selected_emitter_data = emitters_data.get(selected_emitter)

# Создаем спектры фильтров на основе введенных параметров
filter_spectra = []
for params in filter_params:
    filter_spectrum = create_filter_spectrum(params['center'], params['width'])
    filter_spectra.append(filter_spectrum)

# Рассчитываем результирующий спектр
resulting_spectrum = calculate_resulting_spectrum(selected_emitter_data, filter_spectra)

# Создаем контейнеры для графиков
col1, col2 = st.columns(2)

# График спектра излучателя
with col1:
    st.subheader(f"Спектр излучателя: {selected_emitter}")
    if selected_emitter_data:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=selected_emitter_data['wavelengths'],
            y=selected_emitter_data['intensities'],
            mode='lines',
            name='Излучатель'
        ))
        fig.update_layout(
            xaxis_title='Длина волны (нм)',
            yaxis_title='Относительная интенсивность',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# График спектров фильтров
with col2:
    st.subheader("Спектры фильтров")
    if filter_spectra:
        fig = go.Figure()
        for i, filter_spectrum in enumerate(filter_spectra):
            params = filter_params[i]
            fig.add_trace(go.Scatter(
                x=filter_spectrum['wavelength'],
                y=filter_spectrum['transmission'],
                mode='lines',
                name=f'Фильтр {i+1} ({params["center"]} нм, {params["width"]} нм)'
            ))
        fig.update_layout(
            xaxis_title='Длина волны (нм)',
            yaxis_title='Коэффициент пропускания',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# График результирующего спектра
st.subheader("Результирующий спектр")
if resulting_spectrum is not None:
    fig = go.Figure()
    # Исходный спектр излучателя (с низкой непрозрачностью)
    fig.add_trace(go.Scatter(
        x=resulting_spectrum['wavelength'],
        y=resulting_spectrum['intensity'],
        mode='lines',
        name='Исходный',
        line=dict(color='gray', width=1, dash='dash'),
        opacity=0.5
    ))
    # Результирующий спектр
    fig.add_trace(go.Scatter(
        x=resulting_spectrum['wavelength'],
        y=resulting_spectrum['resulting_intensity'],
        mode='lines',
        name='Результирующий',
        line=dict(color='blue', width=2),
    ))
    fig.update_layout(
        xaxis_title='Длина волны (нм)',
        yaxis_title='Относительная интенсивность',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# Информация о результатах
if resulting_spectrum is not None:
    with st.expander("Статистика результирующего спектра"):
        # Находим пиковую длину волны
        peak_idx = resulting_spectrum['resulting_intensity'].idxmax()
        peak_wavelength = resulting_spectrum.loc[peak_idx, 'wavelength']
        peak_intensity = resulting_spectrum.loc[peak_idx, 'resulting_intensity']

        # Вычисляем FWHM (полная ширина на половине максимума)
        half_max = peak_intensity / 2
        above_half_max = resulting_spectrum[resulting_spectrum['resulting_intensity'] >= half_max]
        if not above_half_max.empty and len(above_half_max) > 1:
            min_wl = above_half_max['wavelength'].min()
            max_wl = above_half_max['wavelength'].max()
            fwhm = max_wl - min_wl
        else:
            fwhm = "Не удалось рассчитать"

        # Вычисляем интегральную интенсивность
        integral_intensity = np.trapz(
            resulting_spectrum['resulting_intensity'],
            resulting_spectrum['wavelength']
        )

        # Отображаем статистику
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Пиковая длина волны", f"{peak_wavelength:.1f} нм")
        with col2:
            st.metric("FWHM", f"{fwhm if isinstance(fwhm, str) else fwhm:.1f} нм")
        with col3:
            # Нормируем интегральную интенсивность относительно исходной
            original_integral = np.trapz(
                resulting_spectrum['intensity'],
                resulting_spectrum['wavelength']
            )
            relative_intensity = (integral_intensity / original_integral) * 100
            st.metric("Относительная интенсивность", f"{relative_intensity:.1f}%")

# Дополнительная информация
with st.expander("Информация о проекте"):
    st.markdown("""
    ### О проекте
    Это приложение позволяет анализировать спектры излучателей и фильтров.

    **Функциональность:**
    - Отображение спектра излучателя из локального файла данных
    - Моделирование спектров фильтров с заданными параметрами
    - Расчет и отображение результирующего спектра
    - Анализ характеристик результирующего спектра

    **Структура проекта:**
    ```
    web_apps/
    ├── venv/
    ├── data/
    │   └── light_sources/
    │       └── light_sources.ods
    └── main.py
    ```
    """)