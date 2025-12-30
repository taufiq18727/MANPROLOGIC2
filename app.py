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
