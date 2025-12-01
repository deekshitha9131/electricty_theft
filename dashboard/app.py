import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px

# -------------------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="Electricity Theft Detection Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# COMPACT THEME
# -------------------------------------------------------------
st.markdown("""
<style>
/* Dark sidebar */
[data-testid="stSidebar"] {
    background-color: #3B4A5C !important;
}

[data-testid="stSidebar"] > div {
    background-color: #3B4A5C !important;
}

/* Sidebar text white */
[data-testid="stSidebar"] * {
    color: white !important;
}

[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h3 {
    color: white !important;
}

/* Form elements black text */
[data-testid="stSidebar"] .stSelectbox * {
    color: black !important;
}

[data-testid="stSidebar"] .stFileUploader * {
    color: black !important;
}

/* Compact header */
.main-header {
    padding: 0.5rem 0;
    margin-bottom: 1rem;
}

/* KPI cards */
[data-testid="metric-container"] {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# COMPACT HEADER
# -------------------------------------------------------------
st.markdown('<div class="main-header">', unsafe_allow_html=True)
col1, col2 = st.columns([4, 1])
with col1:
    st.title("⚡ Electricity Theft Detection Dashboard")
with col2:
    if 'uploaded_file' in locals() and uploaded_file:
        st.metric("Model", "Autoencoder")
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR WITH NAVIGATION
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='color: white !important; margin-bottom: 1rem;'>⚡ Dashboard</h2>", unsafe_allow_html=True)
    
    # Upload Data
    st.markdown("<h4 style='color: white !important;'>📁 Upload Data</h4>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose CSV", type=["csv"])
    
    # Model Selection
    st.markdown("<h4 style='color: white !important;'>⚙️ Select Model</h4>", unsafe_allow_html=True)
    model_type = st.selectbox("", ["Autoencoder", "Isolation Forest", "SVDD"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Navigation
        st.markdown("<h4 style='color: white !important;'>📊 Visualizations</h4>", unsafe_allow_html=True)
        selected_page = st.radio("", [
            "🏠 Home", 
            "📈 Consumption Timeline", 
            "🚨 Anomaly Scores", 
            "🔍 Explainability", 
            "🏆 Customer Ranking"
        ])
        
        # Customer Selection
        if "customer_id" in df.columns:
            st.markdown("<h4 style='color: white !important;'>👤 Customer</h4>", unsafe_allow_html=True)
            customers = ["All"] + sorted(df["customer_id"].unique().tolist())
            selected_customer = st.selectbox("", customers)

# -------------------------------------------------------------
# MAIN CONTENT
# -------------------------------------------------------------
if uploaded_file is None:
    st.info("📋 **Expected Data Format:** CSV with customer_id, timestamp, consumption columns")
    st.stop()

# Data validation
if "customer_id" not in df.columns:
    st.error("❌ Missing 'customer_id' column")
    st.stop()

# KPI CARDS ROW
col1, col2, col3, col4 = st.columns(4)
total_customers = df['customer_id'].nunique()
total_records = len(df)
suspicious_customers = max(1, int(total_customers * 0.15))  # Simulate 15% suspicious

col1.metric("Total Records", f"{total_records:,}")
col2.metric("Total Customers", f"{total_customers:,}")
col3.metric("Suspicious Customers", suspicious_customers)
col4.metric("Last Updated", "2 mins ago")

st.markdown("---")

# -------------------------------------------------------------
# PAGE CONTENT BASED ON NAVIGATION
# -------------------------------------------------------------
if 'selected_page' not in locals():
    selected_page = "🏠 Home"

if selected_page == "🏠 Home":
    st.markdown("## 🏠 Dashboard Overview")
    
    # Quick stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Dataset Summary")
        if "consumption" in df.columns:
            avg_consumption = df['consumption'].mean()
            max_consumption = df['consumption'].max()
            st.write(f"• Average consumption: **{avg_consumption:.1f} kWh**")
            st.write(f"• Peak consumption: **{max_consumption:.1f} kWh**")
            st.write(f"• Data points: **{len(df):,}**")
    
    with col2:
        st.markdown("### 🤖 Model Status")
        st.write(f"• Active model: **{model_type}**")
        st.write("• Status: **🟢 Online**")
        st.write("• Last training: **Yesterday**")

elif selected_page == "📈 Consumption Timeline":
    st.markdown("## 📈 Consumption Timeline")
    
    if selected_customer != "All":
        cust_df = df[df["customer_id"] == selected_customer].copy()
        
        if "timestamp" in cust_df.columns and "consumption" in cust_df.columns:
            cust_df["timestamp"] = pd.to_datetime(cust_df["timestamp"])
            cust_df = cust_df.sort_values("timestamp")
            
            # Consumption plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cust_df["timestamp"],
                y=cust_df["consumption"],
                mode="lines",
                name="Consumption",
                line=dict(color="#2C3E50", width=2)
            ))
            
            # Add anomaly markers (simulate some)
            anomaly_indices = np.random.choice(len(cust_df), size=max(1, len(cust_df)//20), replace=False)
            anomaly_data = cust_df.iloc[anomaly_indices]
            
            fig.add_trace(go.Scatter(
                x=anomaly_data["timestamp"],
                y=anomaly_data["consumption"],
                mode="markers",
                name="Detected Anomalies",
                marker=dict(color="red", size=8, symbol="x")
            ))
            
            fig.update_layout(
                title=f"Customer {selected_customer} - Consumption with Anomalies",
                xaxis_title="Date",
                yaxis_title="Consumption (kWh)",
                height=500,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Stats
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Average", f"{cust_df['consumption'].mean():.1f} kWh")
            col2.metric("Peak", f"{cust_df['consumption'].max():.1f} kWh")
            col3.metric("Anomalies", len(anomaly_indices))
            col4.metric("Risk Level", "🟡 Medium")
        else:
            st.error("Missing timestamp or consumption columns")
    else:
        st.info("Select a specific customer to view timeline")

elif selected_page == "🚨 Anomaly Scores":
    st.markdown("## 🚨 Anomaly Score Analysis")
    
    if selected_customer != "All":
        cust_df = df[df["customer_id"] == selected_customer].copy()
        
        if "timestamp" in cust_df.columns:
            cust_df["timestamp"] = pd.to_datetime(cust_df["timestamp"])
            cust_df = cust_df.sort_values("timestamp")
            
            # Generate synthetic anomaly scores
            np.random.seed(42)
            anomaly_scores = np.random.exponential(0.1, len(cust_df))
            threshold = 0.3
            
            # Anomaly score plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cust_df["timestamp"],
                y=anomaly_scores,
                mode="lines",
                name="Anomaly Score",
                line=dict(color="#E74C3C", width=2)
            ))
            
            fig.add_hline(y=threshold, line_dash="dash", line_color="red", 
                         annotation_text="Threshold")
            
            # Highlight anomalies
            anomalies = anomaly_scores > threshold
            if np.any(anomalies):
                fig.add_trace(go.Scatter(
                    x=cust_df["timestamp"][anomalies],
                    y=anomaly_scores[anomalies],
                    mode="markers",
                    name="Anomalies",
                    marker=dict(color="red", size=8)
                ))
            
            fig.update_layout(
                title="Anomaly Scores Over Time",
                xaxis_title="Date",
                yaxis_title="Anomaly Score",
                height=500,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly stats
            anomaly_count = np.sum(anomalies)
            col1, col2, col3 = st.columns(3)
            col1.metric("Threshold", f"{threshold:.2f}")
            col2.metric("Anomalies Detected", anomaly_count)
            col3.metric("Anomaly Rate", f"{anomaly_count/len(cust_df)*100:.1f}%")
    else:
        st.info("Select a specific customer to view anomaly scores")

elif selected_page == "🔍 Explainability":
    st.markdown("## 🔍 Model Explainability")
    
    if selected_customer != "All":
        st.markdown(f"### Analysis for Customer {selected_customer}")
        
        # Feature importance (simulated)
        features = ["consumption", "hour", "day_of_week", "rolling_mean", "rolling_std"]
        importance = [0.45, 0.25, 0.15, 0.10, 0.05]
        
        fig = px.bar(
            x=importance,
            y=features,
            orientation='h',
            title="Feature Importance for Anomaly Detection",
            labels={'x': 'Importance Score', 'y': 'Features'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation text
        st.markdown("### 🧠 Why was this flagged as anomalous?")
        st.write("• **High consumption variance**: 35% above normal pattern")
        st.write("• **Unusual timing**: Peak usage during off-hours")
        st.write("• **Pattern deviation**: Consumption doesn't match historical behavior")
    else:
        st.info("Select a specific customer to view explainability analysis")

elif selected_page == "🏆 Customer Ranking":
    st.markdown("## 🏆 Customer Risk Ranking")
    
    # Generate risk scores
    def compute_risk_scores(df):
        scores = []
        for customer in df['customer_id'].unique():
            cust_data = df[df['customer_id'] == customer]
            if 'consumption' in cust_data.columns and len(cust_data) > 5:
                consumption = cust_data['consumption']
                mean_cons = consumption.mean()
                std_cons = consumption.std()
                cv = std_cons / mean_cons if mean_cons > 0 else 0
                
                # Risk score with some randomness
                np.random.seed(int(str(customer)[-1]) if str(customer)[-1].isdigit() else 42)
                risk_score = min(100, cv * 30 + np.random.uniform(10, 40))
                
                # Risk level
                if risk_score >= 70:
                    risk_level = "🔴 Critical"
                elif risk_score >= 50:
                    risk_level = "🟠 High"
                elif risk_score >= 30:
                    risk_level = "🟡 Medium"
                else:
                    risk_level = "🟢 Low"
                
                scores.append({
                    'Customer ID': customer,
                    'Risk Score': round(risk_score, 1),
                    'Risk Level': risk_level,
                    'Avg Consumption': round(mean_cons, 1),
                    'Variability': round(cv, 3)
                })
        return pd.DataFrame(scores)
    
    ranking_df = compute_risk_scores(df)
    
    if len(ranking_df) > 0:
        ranking_df = ranking_df.sort_values('Risk Score', ascending=False)
        
        # Top 5 alerts
        st.markdown("### 🚨 Top 5 High-Risk Customers")
        top5 = ranking_df.head(5)
        
        cols = st.columns(5)
        for i, (_, row) in enumerate(top5.iterrows()):
            with cols[i]:
                st.metric(
                    f"#{i+1}",
                    f"ID: {row['Customer ID']}",
                    f"Score: {row['Risk Score']}"
                )
        
        st.markdown("### 📊 Complete Risk Assessment")
        
        # Color coding function
        def color_rows(row):
            if "Critical" in str(row['Risk Level']):
                return ['background-color: #ffe6e6'] * len(row)
            elif "High" in str(row['Risk Level']):
                return ['background-color: #fff2e6'] * len(row)
            elif "Medium" in str(row['Risk Level']):
                return ['background-color: #fffbe6'] * len(row)
            else:
                return ['background-color: #e6ffe6'] * len(row)
        
        styled_df = ranking_df.style.apply(color_rows, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Export button
        csv = ranking_df.to_csv(index=False)
        st.download_button("📥 Download Risk Report", csv, "risk_report.csv", "text/csv")
    else:
        st.error("No data available for ranking")