import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Isolation Forest",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .prediction-box-normal {
            background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
            padding: 30px; border-radius: 15px; color: white;
            text-align: center; box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        }
        .prediction-box-anomaly {
            background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
            padding: 30px; border-radius: 15px; color: white;
            text-align: center; box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        }
        .cluster-info {
            background-color: rgba(255,255,255,0.95);
            padding: 20px; border-radius: 10px;
            margin: 10px 0; border-left: 5px solid #27ae60;
        }
        .anomaly-info {
            background-color: #fdecea;
            padding: 20px; border-radius: 10px;
            margin: 10px 0; border-left: 5px solid #c0392b;
        }
        .algo-badge {
            background: #27ae60; color: white; padding: 5px 15px;
            border-radius: 20px; font-size: 13px; font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

CLUSTER_INFO = {
    0: {"name": "High Value Customers",  "color": "#FF6B6B",
        "description": "Young customers with high spending score and good income",
        "characteristics": ["Age: Young (25-40)", "Income: High (40-80k)", "Spending: High (70-100)"]},
    1: {"name": "Potential Target",       "color": "#4ECDC4",
        "description": "Middle-aged customers with moderate to high spending",
        "characteristics": ["Age: Middle-aged (35-50)", "Income: Moderate (30-70k)", "Spending: Moderate-High"]},
    2: {"name": "Average Customers",      "color": "#45B7D1",
        "description": "Young customers with low to moderate spending",
        "characteristics": ["Age: Young (20-50)", "Income: Low (20-50k)", "Spending: Low-Moderate"]},
    3: {"name": "Loyal Customers",        "color": "#FFA07A",
        "description": "Older customers with variable spending patterns",
        "characteristics": ["Age: Older (40-70)", "Income: Variable", "Spending: Variable"]},
    4: {"name": "Budget Conscious",       "color": "#98D8C8",
        "description": "Customers with high income but low spending",
        "characteristics": ["Age: Varied", "Income: High (50-150k)", "Spending: Low (10-50)"]},
}

@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

@st.cache_resource
def run_isolation_forest(contamination, n_estimators, n_clusters):
    df = load_data()
    features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    iso = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                          random_state=42)
    iso_labels = iso.fit_predict(scaled)   # 1 = normal, -1 = anomaly
    scores     = iso.score_samples(scaled) # lower = more anomalous

    # Cluster the normal points only
    normal_idx = np.where(iso_labels == 1)[0]
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = np.full(len(scaled), -1)
    cluster_labels[normal_idx] = km.fit_predict(scaled[normal_idx])

    return iso, scaler, iso_labels, scores, cluster_labels, scaled

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("<span class='algo-badge'>Isolation Forest</span>", unsafe_allow_html=True)
st.title("🌲 Mall Customer Segmentation — Isolation Forest")
st.markdown("Detect **anomalous customers** (outliers) by isolating rare observations, then cluster normal ones.")
st.markdown("---")

df = load_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Isolation Forest Settings")
    contamination = st.slider("Contamination (expected outlier %)", 0.01, 0.30, 0.05, 0.01)
    n_estimators  = st.slider("Number of Trees", 50, 300, 100, 10)
    n_clusters    = st.slider("Clusters (for normal customers)", 2, 8, 5)
    st.markdown("---")
    st.markdown("**How Isolation Forest works:**")
    st.markdown("""
- Randomly split features and values  
- Anomalies → isolated in fewer splits  
- Average path length = anomaly score  
- Low score = more anomalous  
    """)

iso, scaler, iso_labels, scores, cluster_labels, scaled = run_isolation_forest(
    contamination, n_estimators, n_clusters)

n_anomalies = int(np.sum(iso_labels == -1))
n_normal    = int(np.sum(iso_labels == 1))

# ── Metrics Banner ─────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Customers",  len(df))
m2.metric("Normal Customers", n_normal)
m3.metric("Anomalies Found",  n_anomalies)
m4.metric("Anomaly Rate",     f"{n_anomalies/len(df)*100:.1f}%")
st.markdown("---")

# ── Input + Stats ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")
with col1:
    st.markdown("### 📊 Customer Information")
    age            = st.slider("👤 Age",                     int(df['Age'].min()),               int(df['Age'].max()),               30)
    annual_income  = st.slider("💰 Annual Income (k$)",     int(df['Annual Income (k$)'].min()), int(df['Annual Income (k$)'].max()), 50)
    spending_score = st.slider("🎯 Spending Score (1-100)",  1, 100, 50)
    ca, cb, cc = st.columns(3)
    ca.metric("Age",      f"{age} yrs")
    cb.metric("Income",   f"${annual_income}k")
    cc.metric("Spending", f"{spending_score}/100")

with col2:
    st.markdown("### 📈 Dataset Statistics")
    cx, cy, cz = st.columns(3)
    cx.metric("Total Customers", len(df))
    cy.metric("Avg Age",    f"{df['Age'].mean():.1f} yrs")
    cz.metric("Avg Income", f"${df['Annual Income (k$)'].mean():.1f}k")
    st.markdown("---")
    cp, cq, cr = st.columns(3)
    cp.metric("Min Score",     f"{scores.min():.3f}")
    cq.metric("Max Score",     f"{scores.max():.3f}")
    cr.metric("Mean Score",    f"{scores.mean():.3f}")

st.markdown("---")

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🚀 Predict — Normal or Anomaly?", use_container_width=True):
    inp        = np.array([[age, annual_income, spending_score]])
    scaled_inp = scaler.transform(inp)

    iso_pred = int(iso.predict(scaled_inp)[0])   # 1 or -1
    iso_score = float(iso.score_samples(scaled_inp)[0])

    if iso_pred == -1:
        # ANOMALY
        st.markdown(f"""
            <div class="prediction-box-anomaly">
                <h2 style='margin:0;font-size:26px;'>⚠️ Anomaly Detected!</h2>
                <h1 style='margin:10px 0;font-size:52px;'>OUTLIER</h1>
                <h3 style='font-size:20px;'>This customer shows unusual behaviour</h3>
                <p style='font-size:16px;'>Anomaly Score: <strong>{iso_score:.4f}</strong></p>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
            <div class="anomaly-info">
                <p><strong>What does this mean?</strong></p>
                <p>• This customer's profile is statistically rare in the dataset</p>
                <p>• Could be a high-value VIP, a fraudulent entry, or a data error</p>
                <p>• Recommend manual review and personalised engagement</p>
                <p>• Lower anomaly score = more anomalous</p>
            </div>""", unsafe_allow_html=True)

        # Score percentile
        pct = float(np.mean(scores < iso_score) * 100)
        st.metric("Score Percentile (vs dataset)", f"{pct:.1f}th percentile")
        st.warning("**Recommendation:** 🔍 Investigate individually — may be VIP, fraud, or data issue")
    else:
        # NORMAL — assign cluster
        dists   = np.linalg.norm(scaled - scaled_inp, axis=1)
        nearest = int(np.argmin(dists))
        cluster = int(cluster_labels[nearest])
        if cluster == -1:
            cluster = 0  # fallback

        safe_cluster = cluster if cluster in CLUSTER_INFO else cluster % len(CLUSTER_INFO)
        details = CLUSTER_INFO[safe_cluster]

        st.markdown(f"""
            <div class="prediction-box-normal">
                <h2 style='margin:0;font-size:26px;'>✅ Normal Customer — Cluster {cluster}</h2>
                <h1 style='margin:10px 0;font-size:52px;'>{details['name']}</h1>
                <p style='font-size:16px;'>Anomaly Score: <strong>{iso_score:.4f}</strong> (Normal range)</p>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        ci1, ci2 = st.columns(2, gap="large")
        with ci1:
            st.markdown("### 📋 Cluster Description")
            st.markdown(f"<div class='cluster-info'><p><strong>{details['description']}</strong></p></div>", unsafe_allow_html=True)
            st.markdown("### 🎯 Key Characteristics")
            for c in details['characteristics']:
                st.markdown(f"• {c}")
        with ci2:
            st.markdown("### 💡 Cluster Insights")
            mask = cluster_labels == cluster
            sub  = df[mask]
            st.markdown(f"""
                <div class='cluster-info'>
                    <p><strong>Size:</strong> {mask.sum()} customers ({mask.sum()/len(df)*100:.1f}%)</p>
                    <p><strong>Avg Age:</strong> {sub['Age'].mean():.1f} yrs</p>
                    <p><strong>Avg Income:</strong> ${sub['Annual Income (k$)'].mean():.1f}k</p>
                    <p><strong>Avg Spending:</strong> {sub['Spending Score (1-100)'].mean():.1f}/100</p>
                    <p><strong>Anomaly Score:</strong> {iso_score:.4f}</p>
                </div>""", unsafe_allow_html=True)

        recommendations = {
            0: "🎯 Premium targeting strategy — upsell premium products",
            1: "📈 Growth opportunity — seasonal promotions and loyalty programs",
            2: "🎁 Budget offerings — value packs and discount campaigns",
            3: "🤝 Relationship building — personalized communication",
            4: "💎 Exclusive products — despite lower spending",
        }
        st.info(f"**Recommendation:** {recommendations.get(safe_cluster, '—')}")

    # ── Visualisations ─────────────────────────────────────────────────────────
    st.markdown("### 📊 Visualizations")
    vc1, vc2 = st.columns(2, gap="large")

    iso_color = ['Anomaly' if l == -1 else 'Normal' for l in iso_labels]
    with vc1:
        fig = px.scatter_3d(
            x=df['Age'], y=df['Annual Income (k$)'], z=df['Spending Score (1-100)'],
            color=iso_color,
            color_discrete_map={'Normal': '#27ae60', 'Anomaly': '#e74c3c'},
            labels={'x': 'Age', 'y': 'Annual Income (k$)', 'z': 'Spending Score'},
            title='3D: Normal vs Anomaly Customers',
        )
        fig.add_scatter3d(
            x=[age], y=[annual_income], z=[spending_score],
            mode='markers', marker=dict(size=14, color='yellow', symbol='diamond'),
            name='Your Input'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with vc2:
        # Anomaly score distribution
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=scores[iso_labels == 1], name='Normal',
            marker_color='#27ae60', opacity=0.7, nbinsx=20
        ))
        fig_hist.add_trace(go.Histogram(
            x=scores[iso_labels == -1], name='Anomaly',
            marker_color='#e74c3c', opacity=0.7, nbinsx=20
        ))
        fig_hist.add_vline(x=iso_score, line_dash="dash", line_color="blue",
                           annotation_text=f"Your Score: {iso_score:.3f}")
        fig_hist.update_layout(
            title='Anomaly Score Distribution',
            xaxis_title='Anomaly Score', yaxis_title='Count',
            barmode='overlay', height=500
        )
        st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align:center;color:gray;'>🌲 Isolation Forest | Mall Customer Anomaly Detection</div>", unsafe_allow_html=True)