import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="NYC Taxi Analytics", layout="wide", page_icon="🚕")

# ===== LOAD DATA =====
@st.cache_data
def load_data():
    df = pd.read_parquet(r"data/gold/yellow_tripdata_gold.parquet")
    return df

@st.cache_data
def load_silver():
    df = pd.read_parquet(r"data/silver/yellow_tripdata_silver.parquet",
                         columns=['trip_duration', 'trip_distance', 'pickup_hour',
                                  'pickup_day', 'pickup_month', 'is_rush_hour',
                                  'PULocationID', 'DOLocationID', 'passenger_count'])
    return df.sample(100000, random_state=42)

st.title("🚕 NYC Taxi Trip Analytics Dashboard")
st.markdown("**Big Data Analytics | SDG 9 | Sains Data ITERA 2026**")
st.markdown("---")

with st.spinner("Loading data..."):
    df_gold = load_data()
    df_silver = load_silver()

# ===== SIDEBAR =====
st.sidebar.title("⚙️ Filter")
hour_range = st.sidebar.slider("Jam Pickup", 0, 23, (0, 23))
day_option = st.sidebar.multiselect(
    "Hari (0=Senin, 6=Minggu)",
    options=list(range(7)),
    default=list(range(7))
)

df_filtered = df_silver[
    (df_silver['pickup_hour'] >= hour_range[0]) &
    (df_silver['pickup_hour'] <= hour_range[1]) &
    (df_silver['pickup_day'].isin(day_option))
]

# ===== METRICS =====
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Trip", f"{len(df_filtered):,}")
col2.metric("Rata-rata Durasi", f"{df_filtered['trip_duration'].mean():.1f} menit")
col3.metric("Rata-rata Jarak", f"{df_filtered['trip_distance'].mean():.2f} mil")
col4.metric("Rata-rata Penumpang", f"{df_filtered['passenger_count'].mean():.1f}")

st.markdown("---")

# ===== EDA =====
st.subheader("📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(
        df_filtered,
        x='trip_duration', nbins=50,
        title='Distribusi Durasi Perjalanan (menit)',
        color_discrete_sequence=['steelblue']
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    trip_per_hour = df_filtered.groupby('pickup_hour')['trip_duration'].count().reset_index()
    trip_per_hour.columns = ['Jam', 'Jumlah Trip']
    fig2 = px.bar(trip_per_hour, x='Jam', y='Jumlah Trip',
                  title='Jumlah Trip per Jam',
                  color_discrete_sequence=['orange'])
    st.plotly_chart(fig2, use_container_width=True)

# ===== POLA WAKTU =====
st.markdown("---")
st.subheader("🕐 Pola Waktu Perjalanan")

col1, col2 = st.columns(2)

with col1:
    avg_per_hour = df_filtered.groupby('pickup_hour')['trip_duration'].mean().reset_index()
    avg_per_hour.columns = ['Jam', 'Rata-rata Durasi']
    fig3 = px.line(avg_per_hour, x='Jam', y='Rata-rata Durasi',
                   title='Rata-rata Durasi per Jam (Rush Hour Analysis)',
                   markers=True, color_discrete_sequence=['red'])
    fig3.add_vrect(x0=7, x1=9, fillcolor="yellow", opacity=0.2, annotation_text="Rush Hour Pagi")
    fig3.add_vrect(x0=17, x1=19, fillcolor="yellow", opacity=0.2, annotation_text="Rush Hour Sore")
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    df_filtered2 = df_filtered.copy()
    df_filtered2['day_type'] = df_filtered2['pickup_day'].apply(
        lambda x: 'Weekend' if x >= 5 else 'Weekday'
    )
    weekday_data = df_filtered2.groupby(['pickup_hour', 'day_type'])['trip_duration'].mean().reset_index()
    fig4 = px.line(weekday_data, x='pickup_hour', y='trip_duration',
                   color='day_type', markers=True,
                   title='Weekday vs Weekend - Rata-rata Durasi per Jam',
                   labels={'pickup_hour': 'Jam', 'trip_duration': 'Durasi (menit)'})
    st.plotly_chart(fig4, use_container_width=True)

# ===== KEPADATAN LOKASI =====
st.markdown("---")
st.subheader("🗺️ Kepadatan Trip per Lokasi (Top 20 Zone)")

col1, col2 = st.columns(2)

with col1:
    top_pickup = df_filtered['PULocationID'].value_counts().head(20).reset_index()
    top_pickup.columns = ['Zone ID', 'Jumlah Trip']
    fig5 = px.bar(top_pickup, x='Zone ID', y='Jumlah Trip',
                  title='Top 20 Pickup Location',
                  color='Jumlah Trip', color_continuous_scale='Blues')
    st.plotly_chart(fig5, use_container_width=True)

with col2:
    top_dropoff = df_filtered['DOLocationID'].value_counts().head(20).reset_index()
    top_dropoff.columns = ['Zone ID', 'Jumlah Trip']
    fig6 = px.bar(top_dropoff, x='Zone ID', y='Jumlah Trip',
                  title='Top 20 Dropoff Location',
                  color='Jumlah Trip', color_continuous_scale='Oranges')
    st.plotly_chart(fig6, use_container_width=True)

# ===== PERBANDINGAN MODEL =====
st.markdown("---")
st.subheader("🤖 Perbandingan Model Machine Learning")

model_data = {
    'Model': ['Baseline', 'Linear Regression', 'XGBoost'],
    'MAE': [10.5943, 6.0239, 4.0218],
    'RMSE': [15.5475, 9.5643, 6.9145],
    'R²': [0, 0.6216, 0.8022]
}
df_model = pd.DataFrame(model_data)

col1, col2 = st.columns(2)

with col1:
    st.dataframe(df_model.style.highlight_min(subset=['MAE', 'RMSE'], color='lightgreen')
                              .highlight_max(subset=['R²'], color='lightgreen'),
                 use_container_width=True)

with col2:
    fig7 = go.Figure()
    fig7.add_trace(go.Bar(name='MAE', x=df_model['Model'], y=df_model['MAE'], marker_color='steelblue'))
    fig7.add_trace(go.Bar(name='RMSE', x=df_model['Model'], y=df_model['RMSE'], marker_color='orange'))
    fig7.update_layout(barmode='group', title='MAE & RMSE per Model')
    st.plotly_chart(fig7, use_container_width=True)

# ===== PREDIKSI =====
st.markdown("---")
st.subheader("🔮 Prediksi Durasi Perjalanan")

@st.cache_resource
def train_model():
    features = ['trip_distance', 'passenger_count', 'pickup_hour',
                'pickup_day', 'pickup_month', 'is_rush_hour',
                'PULocationID', 'DOLocationID']
    X = df_gold[features]
    y = df_gold['trip_duration']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

with st.spinner("Training model untuk prediksi..."):
    model = train_model()

col1, col2, col3 = st.columns(3)
with col1:
    distance = st.number_input("Jarak Perjalanan (mil)", min_value=0.1, max_value=50.0, value=2.5)
    passenger = st.number_input("Jumlah Penumpang", min_value=1, max_value=6, value=1)
    hour = st.slider("Jam Pickup", 0, 23, 8)
with col2:
    day = st.selectbox("Hari", options=list(range(7)),
                       format_func=lambda x: ['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu'][x])
    month = st.selectbox("Bulan", options=[4], format_func=lambda x: 'April')
    pu_location = st.number_input("Pickup Zone ID", min_value=1, max_value=265, value=161)
with col3:
    do_location = st.number_input("Dropoff Zone ID", min_value=1, max_value=265, value=236)
    is_rush = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
    st.info(f"Rush Hour: {'✅ Ya' if is_rush else '❌ Tidak'}")

    if st.button("🔮 Prediksi!", use_container_width=True):
        input_data = np.array([[distance, passenger, hour, day, month, is_rush, pu_location, do_location]])
        pred = model.predict(input_data)[0]
        st.success(f"⏱️ Estimasi Durasi: **{pred:.1f} menit**")

st.markdown("---")
st.caption("Kelompok 16 | Azzahra Putri Kamilah, Nayla Salsabila Fathianisa, Fiodora Alyysa Juandi, Wan Nashwa A.Y | Sains Data ITERA 2026")