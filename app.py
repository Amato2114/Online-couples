import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import re
import warnings
warnings.filterwarnings('ignore')

# ====================== НАСТРОЙКИ СТРАНИЦЫ ======================
st.set_page_config(page_title="LSTM Валютный прогноз (RUB)", layout="wide")
st.title("📈 Прогноз курса рубля к иностранным валютам (LSTM + ЦБ РФ)")

st.markdown("""
Это приложение прогнозирует курс **рубля к доллару США (USD/RUB)** с помощью LSTM-нейросети.  
📌 **Все прогнозы показывают, сколько рублей нужно для покупки 1 доллара.**  
Дополнительные признаки: курс евро (EUR/RUB) и ключевая ставка ЦБ РФ (из файла `keyrate.txt` в папке проекта).
""")

# ====================== КОНСТАНТЫ ======================
KEYRATE_FILE = "keyrate.txt"  # имя файла с ключевой ставкой в папке проекта

# ====================== ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ ======================
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'history' not in st.session_state:
    st.session_state.history = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'last_train_date' not in st.session_state:
    st.session_state.last_train_date = None
if 'data_shape' not in st.session_state:
    st.session_state.data_shape = None
if 'pred_days' not in st.session_state:
    st.session_state.pred_days = 30
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'has_keyrate' not in st.session_state:
    st.session_state.has_keyrate = False  # флаг, использовалась ли ставка при обучении

# ====================== ФУНКЦИИ ЗАГРУЗКИ ДАННЫХ ======================
def fetch_cbr_rates_single(date_from, date_to, currency_code, fill_missing=True):
    """Загружает курс одной валюты с сайта ЦБ РФ."""
    date_from_obj = datetime.strptime(date_from, "%Y-%m-%d")
    date_to_obj = datetime.strptime(date_to, "%Y-%m-%d")
    date_from_str = date_from_obj.strftime("%d/%m/%Y")
    date_to_str = date_to_obj.strftime("%d/%m/%Y")

    url = "http://www.cbr.ru/scripts/XML_dynamic.asp"
    params = {
        'date_req1': date_from_str,
        'date_req2': date_to_str,
        'VAL_NM_RQ': currency_code
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.encoding = 'windows-1251'
        response.raise_for_status()
        root = ET.fromstring(response.text)
        dates, rates = [], []
        for record in root.findall('Record'):
            date_str = record.get('Date')
            rate_value = record.find('Value').text
            nominal = int(record.find('Nominal').text)
            date_obj = datetime.strptime(date_str, "%d.%m.%Y")
            rate_float = float(rate_value.replace(',', '.')) / nominal
            dates.append(date_obj)
            rates.append(rate_float)
        df = pd.DataFrame({'rate': rates}, index=dates)
        df.sort_index(inplace=True)

        if fill_missing:
            full_idx = pd.date_range(start=date_from_obj, end=date_to_obj, freq='D')
            df = df.reindex(full_idx)
            df['rate'] = df['rate'].fillna(method='ffill').fillna(method='bfill')
        return df['rate']
    except Exception as e:
        st.warning(f"⚠️ Не удалось загрузить курс {currency_code}: {e}")
        return None

def fetch_cbr_rates_multi(date_from, date_to, currency_dict, fill_missing=True):
    """Загружает курсы нескольких валют, возвращает DataFrame с колонками."""
    result = pd.DataFrame()
    for name, code in currency_dict.items():
        series = fetch_cbr_rates_single(date_from, date_to, code, fill_missing)
        if series is not None:
            result[name] = series
    return result

def load_keyrate_from_txt(filepath, date_from, date_to):
    """
    Загружает ключевую ставку из текстового файла, где строки имеют формат:
    ДД.ММ.ГГГГ<разделитель>ЗНАЧЕНИЕ (с запятой)
    Разделитель может быть табуляцией, пробелом или другим.
    Автоматически пропускает служебные строки в начале.
    Возвращает Series с именем 'keyrate', проиндексированную по датам.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='windows-1251') as f:
            lines = f.readlines()

    date_pattern = re.compile(r'\b(\d{2}\.\d{2}\.\d{4})\b')
    data = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = date_pattern.search(line)
        if match:
            date_str = match.group(1)
            rest = line[match.end():].strip()
            value_match = re.search(r'([\d,]+)', rest)
            if value_match:
                value_str = value_match.group(1).replace(',', '.')
                try:
                    date = datetime.strptime(date_str, '%d.%m.%Y')
                    value = float(value_str)
                    data.append((date, value))
                except:
                    continue

    if not data:
        st.warning("⚠️ В файле не найдено строк с датами и значениями.")
        return None

    df = pd.DataFrame(data, columns=['date', 'keyrate'])
    df = df.set_index('date').sort_index()
    df = df[~df.index.duplicated(keep='first')]  # удаляем дубликаты дат

    start = datetime.strptime(date_from, "%Y-%m-%d")
    end = datetime.strptime(date_to, "%Y-%m-%d")
    df = df[(df.index >= start) & (df.index <= end)]

    if df.empty:
        st.warning(f"⚠️ Файл не содержит данных за период {date_from} – {date_to}.")
        return None

    full_idx = pd.date_range(start=start, end=end, freq='D')
    df = df.reindex(full_idx)
    df['keyrate'] = df['keyrate'].fillna(method='ffill').fillna(method='bfill')
    return df['keyrate']

def add_technical_indicators(series):
    """Добавляет SMA_7, SMA_30, RSI, Volatility для одного ряда (например, USD)."""
    df = pd.DataFrame({'price': series})
    df['SMA_7'] = df['price'].rolling(window=7).mean()
    df['SMA_30'] = df['price'].rolling(window=30).mean()
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Volatility'] = df['price'].rolling(window=20).std()
    return df

def create_sequences(data, seq_length, pred_days):
    X, y = [], []
    for i in range(len(data) - seq_length - pred_days + 1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+pred_days, 0])  # предсказываем цену (первый столбец)
    return np.array(X), np.array(y)

def build_model(architecture, input_shape, pred_days, units1, units2, dropout):
    """Создаёт модель в зависимости от выбранной архитектуры."""
    if architecture == "LSTM (классическая)":
        model = keras.Sequential([
            layers.LSTM(units1, return_sequences=True, input_shape=input_shape),
            layers.Dropout(dropout),
            layers.LSTM(units2, return_sequences=False),
            layers.Dropout(dropout),
            layers.Dense(64, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(pred_days)
        ])

    elif architecture == "LSTM + Attention":
        inputs = keras.Input(shape=input_shape)
        x = layers.LSTM(units1, return_sequences=True)(inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LSTM(units2, return_sequences=True)(x)
        x = layers.Dropout(dropout)(x)
        attention = layers.Attention()([x, x])
        x = layers.Concatenate()([x, attention])
        x = GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        outputs = layers.Dense(pred_days)(x)
        model = keras.Model(inputs, outputs)

    elif architecture == "Transformer":
        num_heads = 4
        ff_dim = 128
        inputs = keras.Input(shape=input_shape)
        x = layers.Dense(units1, activation='relu')(inputs)
        positions = tf.range(start=0, limit=input_shape[0], delta=1)
        pos_embedding = layers.Embedding(input_dim=input_shape[0], output_dim=units1)(positions)
        x = x + pos_embedding

        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=units1)(x, x)
        attn_output = layers.Dropout(dropout)(attn_output)
        out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)

        ffn = layers.Dense(ff_dim, activation='relu')(out1)
        ffn = layers.Dense(units1)(ffn)
        ffn = layers.Dropout(dropout)(ffn)
        out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn)

        x = GlobalAveragePooling1D()(out2)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        outputs = layers.Dense(pred_days)(x)
        model = keras.Model(inputs, outputs)

    return model

# ====================== БОКОВАЯ ПАНЕЛЬ ======================
st.sidebar.header("⚙️ Параметры")

# Параметры валюты
st.sidebar.subheader("Валюта")
currency_dict = {
    'USD': 'R01235',
    'EUR': 'R01239',
}
currency_display = st.sidebar.selectbox("Целевая валюта", list(currency_dict.keys()), index=0, help="Выберите валюту для прогноза (USD или EUR).")
currency_code = currency_dict[currency_display]

# Параметры данных
st.sidebar.subheader("Данные")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Начало", datetime(2015, 1, 1), help="Начальная дата для загрузки исторических данных.")
with col2:
    end_date = st.date_input("Конец", datetime(2024, 1, 1), help="Конечная дата для загрузки исторических данных.")

# Параметры модели
st.sidebar.subheader("Модель")
sequence_length = st.sidebar.number_input("Длина последовательности (дни)", min_value=30, max_value=120, value=60, help="Сколько прошлых дней используем для прогноза.")
pred_days = st.sidebar.number_input("Горизонт прогноза (дни)", min_value=7, max_value=60, value=30, help="На сколько дней вперёд предсказываем.")
train_split = st.sidebar.slider("Доля обучения", 0.6, 0.9, 0.8, help="Доля данных для обучения, остальное для теста.")

# Архитектура
architecture = st.sidebar.selectbox("Архитектура",
                                    ["LSTM (классическая)", "LSTM + Attention", "Transformer"],
                                    index=0, help="Выберите тип нейросети.")

# Параметры LSTM/Transformer
with st.sidebar.expander("🔧 Детальные параметры"):
    lstm_units1 = st.number_input("Слой 1 нейронов", min_value=32, max_value=256, value=128, step=32, help="Количество нейронов в первом LSTM слое (или размер проекции для Transformer).")
    lstm_units2 = st.number_input("Слой 2 нейронов", min_value=32, max_value=256, value=64, step=32, help="Количество нейронов во втором LSTM слое (или промежуточный размер).")
    dropout = st.slider("Dropout", 0.0, 0.5, 0.2, help="Регуляризация, отключает случайные нейроны для борьбы с переобучением.")
    epochs = st.number_input("Эпохи", min_value=10, max_value=200, value=50, step=10, help="Количество проходов по данным.")
    batch_size = st.number_input("Batch size", min_value=16, max_value=128, value=32, step=16, help="Размер мини-выборки.")
    learning_rate = st.number_input("Learning rate", min_value=0.0001, max_value=0.01, value=0.001, format="%.4f", help="Скорость обучения.")

# Информация о ключевой ставке
st.sidebar.subheader("📁 Ключевая ставка")
keyrate_file_exists = os.path.exists(KEYRATE_FILE)
if keyrate_file_exists:
    st.sidebar.success(f"✅ Найден файл {KEYRATE_FILE}. Он будет использован автоматически.")
else:
    st.sidebar.warning(f"⚠️ Файл {KEYRATE_FILE} не найден. Обучение будет без ключевой ставки.")

# ====================== ОСНОВНЫЕ ВКЛАДКИ ======================
tab1, tab2, tab3 = st.tabs(["🚀 Обучение", "🔮 Прогноз", "ℹ️ О модели"])

# ====================== ВКЛАДКА 1: ОБУЧЕНИЕ ======================
with tab1:
    st.header("Обучение модели")

    if st.button("🚀 Начать обучение", type="primary"):
        with st.spinner("Загрузка данных..."):
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            # Загружаем USD и EUR
            df_cur = fetch_cbr_rates_multi(start_str, end_str,
                                           {'USD': 'R01235', 'EUR': 'R01239'}, fill_missing=True)
            if df_cur is None or df_cur.empty or 'USD' not in df_cur.columns:
                st.error("❌ Не удалось загрузить курс USD. Проверьте подключение к интернету.")
                st.stop()

            data = df_cur.copy()
            has_keyrate = False

            # Загружаем ключевую ставку из локального файла, если он существует
            if keyrate_file_exists:
                keyrate = load_keyrate_from_txt(KEYRATE_FILE, start_str, end_str)
                if keyrate is not None:
                    data = data.join(keyrate, how='left')
                    has_keyrate = True
                    st.info("✅ Ключевая ставка загружена из локального файла.")
                else:
                    st.warning("⚠️ Не удалось загрузить ключевую ставку из файла (возможно, нет данных за период). Обучение будет без неё.")
            else:
                st.warning("⚠️ Ключевая ставка не загружена. Обучение будет без неё.")

            data = data.dropna()
            if len(data) < sequence_length + pred_days:
                st.error(f"❌ Недостаточно данных: {len(data)} записей, нужно {sequence_length + pred_days}.")
                st.stop()

            # Технические индикаторы
            tech = add_technical_indicators(data['USD'])
            data = pd.concat([data, tech[['SMA_7', 'SMA_30', 'RSI', 'Volatility']]], axis=1)
            data = data.dropna()
            if len(data) < sequence_length + pred_days:
                st.error(f"❌ После индикаторов осталось {len(data)} записей, нужно {sequence_length + pred_days}.")
                st.stop()

            st.success(f"✅ Данные загружены: {len(data)} записей. Признаки: {list(data.columns)}")
            st.session_state.last_train_date = data.index[-1]
            st.session_state.feature_names = data.columns.tolist()
            st.session_state.has_keyrate = has_keyrate

            # Визуализация данных
            st.subheader("📊 Визуализация данных")
            cols_to_plot = ['USD'] + (['EUR'] if 'EUR' in data.columns else []) + (['keyrate'] if has_keyrate else [])
            st.line_chart(data[cols_to_plot].head(500))

            # Корреляция
            st.subheader("🔗 Корреляция признаков с USD")
            corr = data.corr()['USD'].sort_values(ascending=False)
            st.write(corr)

        # Масштабирование
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        # Последовательности
        X, y = create_sequences(scaled_data, sequence_length, pred_days)
        split_idx = int(len(X) * train_split)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        st.session_state.data_shape = f"Обучающая: {X_train.shape}, Тестовая: {X_test.shape}"
        st.info(st.session_state.data_shape)

        # Модель
        model = build_model(architecture, (X.shape[1], X.shape[2]), pred_days, lstm_units1, lstm_units2, dropout)

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.00001)
        ]

        # Обучение с прогрессом
        progress_bar = st.progress(0)
        status_text = st.empty()

        class StreamlitCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                percent = (epoch + 1) / epochs
                progress_bar.progress(percent)
                status_text.text(f"Эпоха {epoch+1}/{epochs} – loss: {logs['loss']:.4f}, val_loss: {logs['val_loss']:.4f}")

        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            callbacks=callbacks + [StreamlitCallback()],
            verbose=0
        )

        progress_bar.empty()
        status_text.empty()
        st.success("✅ Обучение завершено!")

        # Оценка
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_test_first = y_test[:, 0].reshape(-1, 1)
        y_pred_first = y_pred_scaled[:, 0].reshape(-1, 1)

        dummy_test = np.zeros((len(y_test_first), data.shape[1]))
        dummy_test[:, 0] = y_test_first[:, 0]
        dummy_pred = np.zeros((len(y_pred_first), data.shape[1]))
        dummy_pred[:, 0] = y_pred_first[:, 0]

        y_test_orig = scaler.inverse_transform(dummy_test)[:, 0]
        y_pred_orig = scaler.inverse_transform(dummy_pred)[:, 0]

        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
        r2 = r2_score(y_test_orig, y_pred_orig)
        mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100

        st.session_state.metrics = {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}
        st.session_state.history = history

        # Метрики с пояснениями
        st.subheader("📉 Метрики качества")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MAE", f"{mae:.2f} руб.", help="Средняя абсолютная ошибка в рублях.")
        with col2:
            st.metric("RMSE", f"{rmse:.2f} руб.", help="Корень из среднеквадратичной ошибки.")
        with col3:
            st.metric("R²", f"{r2:.3f}", help="Коэффициент детерминации (чем ближе к 1, тем лучше).")
        with col4:
            st.metric("MAPE", f"{mape:.1f}%", help="Средняя абсолютная процентная ошибка.")

        # Графики обучения
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history['loss'], label='Train Loss')
        ax1.plot(history.history['val_loss'], label='Val Loss')
        ax1.set_title('Динамика потерь')
        ax1.set_xlabel('Эпоха')
        ax1.set_ylabel('MSE')
        ax1.legend()
        ax2.plot(history.history['mae'], label='Train MAE')
        ax2.plot(history.history['val_mae'], label='Val MAE')
        ax2.set_title('Динамика MAE')
        ax2.set_xlabel('Эпоха')
        ax2.set_ylabel('MAE (руб)')
        ax2.legend()
        st.pyplot(fig)

        # Сохраняем модель в session_state и во временные файлы
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.model_trained = True
        st.session_state.pred_days = pred_days

        model.save('temp_model.h5')
        joblib.dump(scaler, 'temp_scaler.pkl')
        joblib.dump(st.session_state.feature_names, 'feature_names.pkl')
        st.success("✅ Модель сохранена. Теперь можно строить прогнозы во вкладке 'Прогноз'.")

        # Кнопка для скачивания модели
        with open('temp_model.h5', 'rb') as f:
            st.download_button("📥 Скачать модель", f, file_name="lstm_currency_model.h5")

# ====================== ВКЛАДКА 2: ПРОГНОЗ ======================
with tab2:
    st.header("Прогноз USD/RUB")

    if not st.session_state.model_trained:
        st.warning("Сначала обучите модель на вкладке 'Обучение'.")
        st.stop()

    show_inverse = st.checkbox("Показать обратный курс (RUB/USD)", value=False)

    # Загрузка свежих данных для прогноза
    days_to_load = st.number_input("Загрузить исторических дней (минимум 120)", min_value=120, max_value=500, value=180, step=10)

    if st.button("🔮 Получить прогноз"):
        with st.spinner("Загружаем свежие данные..."):
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days_to_load)
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            # Загружаем USD и EUR
            df_cur = fetch_cbr_rates_multi(start_str, end_str, {'USD': 'R01235', 'EUR': 'R01239'}, fill_missing=True)
            if df_cur is None or df_cur.empty:
                st.error("Не удалось загрузить курсы валют.")
                st.stop()

            data = df_cur.copy()

            # Если модель использует ключевую ставку, загружаем из файла
            if st.session_state.has_keyrate:
                if keyrate_file_exists:
                    keyrate = load_keyrate_from_txt(KEYRATE_FILE, start_str, end_str)
                    if keyrate is not None:
                        data = data.join(keyrate, how='left')
                    else:
                        st.error("Не удалось загрузить ключевую ставку из файла.")
                        st.stop()
                else:
                    st.error("Для прогноза нужна ключевая ставка, но файл не найден.")
                    st.stop()

            data = data.dropna()
            if len(data) < 60:
                st.error(f"Недостаточно данных: {len(data)} записей. Нужно минимум 60.")
                st.stop()

            # Технические индикаторы
            tech = add_technical_indicators(data['USD'])
            data = pd.concat([data, tech[['SMA_7', 'SMA_30', 'RSI', 'Volatility']]], axis=1)
            data = data.dropna()
            if len(data) < 60:
                st.error(f"После индикаторов осталось {len(data)} строк, нужно минимум 60.")
                st.stop()

            # Выбираем последние 60 строк и нужные колонки
            last_60 = data.iloc[-60:].copy()
            feature_names = st.session_state.feature_names
            missing = set(feature_names) - set(last_60.columns)
            if missing:
                st.error(f"Отсутствуют колонки: {missing}")
                st.stop()
            last_60 = last_60[feature_names]

            # Масштабирование
            try:
                scaled = st.session_state.scaler.transform(last_60)
            except Exception as e:
                st.error(f"Ошибка масштабирования: {e}")
                st.stop()

            X = scaled.reshape(1, 60, len(feature_names))

            # Предсказание
            pred_scaled = st.session_state.model.predict(X, verbose=0)[0]
            dummy = np.zeros((st.session_state.pred_days, len(feature_names)))
            dummy[:, 0] = pred_scaled
            pred_prices = st.session_state.scaler.inverse_transform(dummy)[:, 0]

            last_date = last_60.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=st.session_state.pred_days)

            # График
            st.subheader("Прогноз USD/RUB")
            fig, ax = plt.subplots(figsize=(14, 6))
            plot_days = min(200, len(data))
            ax.plot(data.index[-plot_days:], data['USD'].values[-plot_days:], label='История', color='blue')
            ax.plot(future_dates, pred_prices, label='Прогноз', color='red', marker='o')
            ax.set_title(f'Прогноз на {st.session_state.pred_days} дней')
            ax.set_xlabel('Дата')
            ax.set_ylabel('Рубли за доллар')
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Таблица
            forecast_df = pd.DataFrame({
                'День': range(1, st.session_state.pred_days+1),
                'Дата': future_dates.strftime('%d.%m.%Y'),
                'Прогноз (RUB)': pred_prices.round(4)
            })
            st.dataframe(forecast_df, width='stretch')

            if show_inverse:
                inverse_prices = 1.0 / pred_prices
                st.subheader("Обратный курс (RUB/USD)")
                fig2, ax2 = plt.subplots(figsize=(14, 4))
                ax2.plot(future_dates, inverse_prices, color='green', marker='x')
                ax2.set_title(f'Прогноз обратного курса на {st.session_state.pred_days} дней')
                ax2.set_xlabel('Дата')
                ax2.set_ylabel('USD за 1 RUB')
                ax2.grid(True)
                plt.xticks(rotation=45)
                st.pyplot(fig2)

                inverse_df = pd.DataFrame({
                    'День': range(1, st.session_state.pred_days+1),
                    'Дата': future_dates.strftime('%d.%m.%Y'),
                    'Прогноз (USD за 1 RUB)': inverse_prices.round(6)
                })
                st.dataframe(inverse_df, width='stretch')

            st.info(f"Последний известный курс: {last_60['USD'].iloc[-1]:.4f} RUB")

# ====================== ВКЛАДКА 3: О МОДЕЛИ ======================
with tab3:
    st.header("ℹ️ О модели")
    st.markdown("""
    **Архитектура:** LSTM, LSTM+Attention или Transformer.
    **Признаки:**
    - Курсы USD/RUB и EUR/RUB (загружаются с сайта ЦБ РФ)
    - Ключевая ставка ЦБ РФ (из файла `keyrate.txt` в папке проекта)
    - Технические индикаторы на основе USD/RUB:
        - SMA_7, SMA_30 (скользящие средние)
        - RSI (индекс относительной силы)
        - Волатильность (20-дневное стандартное отклонение)
    **Горизонт прогноза:** настраивается (по умолчанию 30 дней).
    **Метрики:** MAE, RMSE, R², MAPE.
    """)
    if st.session_state.metrics:
        st.subheader("Последние метрики обученной модели")
        st.json(st.session_state.metrics)