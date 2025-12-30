import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Import modul custom
from shopee_scraper import ShopeeScraper
from sentiment_analyzer import SentimentAnalyzer
from klasifikasi_topik import TopicClassifier

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Shopee Intelligence Dashboard", page_icon="🛍️", layout="wide")

# --- LOAD RESOURCES (CACHED) ---
@st.cache_resource
def get_analyzer():
    return SentimentAnalyzer()

@st.cache_resource
def get_classifier():
    return TopicClassifier()

# --- FUNGSI LOGIN ---
def check_login(username, password):
    users = st.secrets["users"]
    if username in users and users[username]["password"] == password:
        return users[username]
    return None

# --- STATE MANAGEMENT ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'data' not in st.session_state:
    st.session_state.data = None

# ==========================================
# HALAMAN LOGIN
# ==========================================
if not st.session_state.logged_in:
    st.title("🔐 Login Dashboard Analisis")
    col1, col2 = st.columns([1, 2])
    with col1:
        with st.form("login_form"):
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            submit = st.form_submit_button("Masuk")
            
            if submit:
                user_info = check_login(user, pwd)
                if user_info:
                    st.session_state.logged_in = True
                    st.session_state.username = user
                    st.session_state.shop_id = user_info['shop_id']
                    st.rerun()
                else:
                    st.error("Username atau Password salah!")
    st.stop()

# ==========================================
# DASHBOARD UTAMA
# ==========================================
# Sidebar
with st.sidebar:
    st.title(f"👤 {st.session_state.username}")
    st.write(f"Target Shop ID: {st.session_state.shop_id}")
    
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()
    
    st.divider()
    st.info("Aplikasi ini berjalan secara real-time tanpa database SQL.")

st.title("🚀 Shopee Review Intelligence")
st.markdown("Scraping, Analisis Sentimen (AI), dan Klasifikasi Topik dalam satu dashboard.")

# Tabs
tab1, tab2, tab3 = st.tabs(["📥 1. Scraping Data", "🧠 2. Analisis AI", "📊 3. Visualisasi"])

# --- TAB 1: SCRAPING ---
with tab1:
    st.header("Ambil Data Terbaru")
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        shop_id_input = st.number_input("Shop ID", value=st.session_state.shop_id)
    with col_input2:
        limit_input = st.slider("Jumlah Data Maksimal", 20, 500, 100)
    
    if st.button("Mulai Scraping", type="primary"):
        scraper = ShopeeScraper(shop_id=shop_id_input, limit=limit_input)
        status_text = st.empty()
        
        # Jalankan scraping
        df = scraper.scrape(progress_callback=status_text.text)
        
        if not df.empty:
            st.session_state.data = df
            st.success(f"Berhasil mengambil {len(df)} ulasan!")
            st.dataframe(df.head())
        else:
            st.error("Gagal mengambil data. Cek koneksi atau Shop ID.")

# --- TAB 2: ANALISIS ---
with tab2:
    st.header("Analisis Sentimen & Topik")
    
    if st.session_state.data is None:
        st.warning("⚠️ Silakan ambil data dulu di Tab 1")
    else:
        df = st.session_state.data
        col_act1, col_act2 = st.columns(2)
        
        with col_act1:
            st.metric("Data Tersedia", f"{len(df)} baris")
        
        # Tombol Jalankan Analisis
        if st.button("Jalankan Analisis AI"):
            analyzer = get_analyzer()
            classifier = get_classifier()
            
            # Progress bar
            bar = st.progress(0)
            status = st.empty()
            
            sentiments = []
            scores = []
            
            # Loop analisis sentimen
            total = len(df)
            for i, text in enumerate(df['Review']):
                s, sc, meta = analyzer.predict(text)
                sentiments.append(s)
                scores.append(sc)
                bar.progress((i + 1) / total)
                status.text(f"Menganalisis review ke-{i+1}...")
            
            df['Sentimen'] = sentiments
            df['Confidence'] = scores
            
            # Klasifikasi Topik
            status.text("Mengklasifikasikan topik...")
            df = classifier.process_dataframe(df, col_review='Review')
            
            st.session_state.data = df
            status.text("Selesai!")
            bar.empty()
            st.success("Analisis Selesai!")
        
        # Tampilkan Hasil jika sudah ada kolom Sentimen
        if 'Sentimen' in df.columns:
            st.dataframe(df)
            
            # Download Buttons
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "hasil_analisis.csv", "text/csv")

# --- TAB 3: VISUALISASI ---
with tab3:
    if st.session_state.data is None or 'Sentimen' not in st.session_state.data.columns:
        st.info("Lakukan Scraping dan Analisis terlebih dahulu.")
    else:
        df = st.session_state.data
        
        # KPI
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Review", len(df))
        c2.metric("Rating Rata-rata", f"{df['Rating'].mean():.2f}⭐")
        pos_pct = (len(df[df['Sentimen']=='Positif']) / len(df)) * 100
        c3.metric("Sentimen Positif", f"{pos_pct:.1f}%")
        
        st.divider()
        
        # Grafik 1 & 2
        g1, g2 = st.columns(2)
        
        with g1:
            st.subheader("Distribusi Sentimen")
            fig_pie = px.pie(df, names='Sentimen', color='Sentimen', 
                             color_discrete_map={'Positif':'green', 'Netral':'grey', 'Negatif':'red'})
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with g2:
            st.subheader("Analisis Topik Masalah")
            # Filter hanya yang negatif/netral untuk melihat masalah
            df_neg = df[df['Sentimen'] != 'Positif']
            if not df_neg.empty:
                counts = df_neg['Topik'].value_counts().reset_index()
                counts.columns = ['Topik', 'Jumlah']
                fig_bar = px.bar(counts, x='Topik', y='Jumlah', color='Topik', title="Topik Keluhan Pelanggan")
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.write("Tidak ada ulasan negatif untuk dianalisis topiknya.")

        # WordCloud
        st.subheader("Word Cloud")
        text_all = " ".join(df['Review'].astype(str))
        wc = WordCloud(width=800, height=300, background_color='white').generate(text_all)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
