import streamlit as st
import pandas as pd
import numpy as np
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
# EXACT DARK THEME FROM IMAGE
# -------------------------------------------------------------
st.markdown("""
<style>
/* Dark theme matching the image */
.stApp {
    background-color: #1a1d29;
    color: white;
}

/* Sidebar dark blue */
[data-testid="stSidebar"] {
    background-color: #252a3a !important;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

/* Main content dark */
.main .block-container {
    background-color: #1a1d29;
    color: white;
    padding-top: 2rem;
}

/* Headers white */
h1, h2, h3, h4, h5, h6 {
    color: white !important;
}

/* Metrics styling */
[data-testid="metric-container"] {
    background-color: #252a3a;
    border-radius: 8px;
    padding: 1rem;
    border: 1px solid #3a4050;
}

[data-testid="metric-container"] [data-testid="metric-value"] {
    color: white !important;
    font-size: 2rem !important;
    font-weight: bold !important;
}

[data-testid="metric-container"] [data-testid="metric-label"] {
    color: #8892b0 !important;
    font-size: 0.9rem !important;
}

/* Form elements - white text */
[data-testid="stSidebar"] .stSelectbox label {
    color: white !important;
}

[data-testid="stSidebar"] .stFileUploader label {
    color: white !important;
}

/* All text elements white */
.stApp, .stApp * {
    color: white !important;
}

/* Ensure all text is visible */
p, span, div, label {
    color: white !important;
}

/* Info message styling */
.stAlert {
    background-color: #252a3a !important;
    color: white !important;
    border: 1px solid #3a4050 !important;
}

/* File uploader text - black for visibility */
[data-testid="stSidebar"] .stFileUploader small {
    color: black !important;
}

[data-testid="stSidebar"] .stFileUploader button {
    color: black !important;
}

[data-testid="stSidebar"] .stFileUploader div {
    color: black !important;
}

[data-testid="stSidebar"] .stFileUploader p {
    color: black !important;
}

[data-testid="stSidebar"] .stFileUploader span {
    color: black !important;
}

/* Selectbox text - black */
[data-testid="stSidebar"] .stSelectbox div[role="button"] {
    color: black !important;
}

[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
    color: black !important;
}

/* Custom sidebar icons */
.sidebar-item {
    display: flex;
    align-items: center;
    padding: 0.5rem 0;
    color: black !important;
    font-size: 1rem;
}

.sidebar-icon {
    margin-right: 0.5rem;
    font-size: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# HEADER - EXACT FROM IMAGE
# -------------------------------------------------------------
st.markdown("""
<div style='padding: 1rem 0; border-bottom: 1px solid #3a4050; margin-bottom: 2rem;'>
    <h1 style='margin: 0; font-size: 2rem;'>⚡ Electricity Theft Detection – Anomaly Dashboard</h1>
    <p style='margin: 0.5rem 0 0 0; color: #8892b0; font-size: 1rem;'>Real-time monitoring and risk scoring</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR - EXACT LAYOUT FROM IMAGE
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class='sidebar-item'>
        <span class='sidebar-icon'>📁</span>
        <span>Upload Data</span>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["csv"])
    
    st.markdown("""
    <div class='sidebar-item'>
        <span class='sidebar-icon'>⚙️</span>
        <span>Select Model</span>
    </div>
    """, unsafe_allow_html=True)
    
    model_type = st.selectbox("", ["Autoencoder", "Isolation Forest", "SVDD"])
    
    st.markdown("""
    <div class='sidebar-item'>
        <span class='sidebar-icon'>🎯</span>
        <span>Select Score</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='sidebar-item'>
        <span class='sidebar-icon'>🚨</span>
        <span>Detected Anomalies</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='sidebar-item'>
        <span class='sidebar-icon'>👥</span>
        <span>Customer Ranking</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='sidebar-item'>
        <span class='sidebar-icon'>⚙️</span>
        <span>Settings</span>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# MAIN CONTENT - EXACT LAYOUT FROM IMAGE
# -------------------------------------------------------------
if uploaded_file is None:
    st.info("Upload CSV data to begin analysis")
    st.stop()

# Load data
df = pd.read_csv(uploaded_file)

# KPI CARDS ROW - EXACT FROM IMAGE
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("High-Risk Customers", "13")

with col2:
    st.metric("Anomalies Today", "25")

with col3:
    st.metric("Avg Score", "7.8")

with col4:
    st.metric("Total Customers Monitored", "1,350")

st.markdown("<br>", unsafe_allow_html=True)

# CHARTS ROW - EXACT FROM IMAGE
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Anomaly Timeline")
    
    # Create timeline chart matching the image
    dates = ['Mar 01', 'Mar 02', 'Mar 13', 'Mar 16', 'Mar 16']
    values = [200, 250, 350, 400, 400]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines+markers',
        line=dict(color='#64b5f6', width=3),
        marker=dict(color='#ff7043', size=8),
        name='Anomalies'
    ))
    
    fig.update_layout(
        plot_bgcolor='#252a3a',
        paper_bgcolor='#252a3a',
        font_color='white',
        height=300,
        showlegend=False,
        xaxis=dict(
            gridcolor='#3a4050',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='#3a4050',
            showgrid=True,
            range=[0, 500]
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Risk Distribution")
    
    # Create donut chart matching the image
    fig2 = go.Figure(data=[go.Pie(
        labels=['High', 'Medium', 'Low'],
        values=[35, 30, 35],
        hole=.6,
        marker_colors=['#ff7043', '#ffb74d', '#4db6ac']
    )])
    
    fig2.update_layout(
        plot_bgcolor='#252a3a',
        paper_bgcolor='#252a3a',
        font_color='white',
        height=300,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        annotations=[dict(text='65%', x=0.5, y=0.5, font_size=24, showarrow=False, font_color='white')]
    )
    
    st.plotly_chart(fig2, use_container_width=True)

# CUSTOMER RANKING TABLE - EXACT FROM IMAGE
st.markdown("### Customer Ranking")

# Create sample data matching the image
ranking_data = {
    'Customer ID': [1034, 5789, 1210],
    'Score': [9.8, 8.8, 7.8],
    'Last-Day Pattern': ['📈', '📈', '📈'],
    'Status': ['High', 'High', 'Medium']
}

ranking_df = pd.DataFrame(ranking_data)

# Custom styling for the table
def color_status(val):
    if val == 'High':
        return 'background-color: #ff7043; color: white; border-radius: 12px; padding: 4px 8px; text-align: center;'
    elif val == 'Medium':
        return 'background-color: #ffb74d; color: white; border-radius: 12px; padding: 4px 8px; text-align: center;'
    else:
        return 'background-color: #4db6ac; color: white; border-radius: 12px; padding: 4px 8px; text-align: center;'

# Display table with custom styling
st.markdown("""
<div style='background-color: #252a3a; padding: 1rem; border-radius: 8px; border: 1px solid #3a4050;'>
    <table style='width: 100%; color: white;'>
        <tr style='border-bottom: 1px solid #3a4050;'>
            <th style='text-align: left; padding: 0.5rem; color: #8892b0;'>Customer ID</th>
            <th style='text-align: left; padding: 0.5rem; color: #8892b0;'>Score</th>
            <th style='text-align: left; padding: 0.5rem; color: #8892b0;'>Last-Day Pattern</th>
            <th style='text-align: left; padding: 0.5rem; color: #8892b0;'>Status</th>
        </tr>
        <tr>
            <td style='padding: 0.5rem;'>1034</td>
            <td style='padding: 0.5rem;'>9.8</td>
            <td style='padding: 0.5rem;'>📈</td>
            <td style='padding: 0.5rem;'><span style='background-color: #ff7043; color: white; border-radius: 12px; padding: 4px 12px;'>High</span></td>
        </tr>
        <tr>
            <td style='padding: 0.5rem;'>5789</td>
            <td style='padding: 0.5rem;'>8.8</td>
            <td style='padding: 0.5rem;'>📈</td>
            <td style='padding: 0.5rem;'><span style='background-color: #ff7043; color: white; border-radius: 12px; padding: 4px 12px;'>High</span></td>
        </tr>
        <tr>
            <td style='padding: 0.5rem;'>1210</td>
            <td style='padding: 0.5rem;'>7.8</td>
            <td style='padding: 0.5rem;'>📈</td>
            <td style='padding: 0.5rem;'><span style='background-color: #ffb74d; color: white; border-radius: 12px; padding: 4px 12px;'>Medium</span></td>
        </tr>
    </table>
</div>
""", unsafe_allow_html=True)