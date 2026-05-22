import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import os
import warnings

warnings.filterwarnings('ignore')

# 获取脚本所在目录，用于加载模型
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ========== 页面配置 ==========
st.set_page_config(
    page_title="创伤伤员伤情分类预测系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== 自定义CSS样式 ==========
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.3rem;
    }
    .mild { background-color: #d4edda; color: #155724; border: 2px solid #28a745; }
    .moderate { background-color: #fff3cd; color: #856404; border: 2px solid #ffc107; }
    .severe { background-color: #f8d7da; color: #721c24; border: 2px solid #dc3545; }
    .critical { background-color: #f5c6cb; color: #721c24; border: 2px solid #bd2130; }
    .metric-label { font-size: 0.9rem; color: #666; }
    .metric-value { font-size: 1.5rem; font-weight: bold; }
    .section-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f4e79;
        border-left: 4px solid #1f4e79;
        padding-left: 10px;
        margin: 1rem 0 0.5rem 0;
    }
    .stButton>button {
        background-color: #1f4e79;
        color: white;
        font-size: 1.1rem;
        font-weight: bold;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #163a5c;
    }
</style>
""", unsafe_allow_html=True)

# ========== 模型加载 ==========
@st.cache_resource
def load_models():
    try:
        model_path = os.path.join(BASE_DIR, "best_fcnn_model.joblib")
        scaler_path = os.path.join(BASE_DIR, "scaler.joblib")
        model = load(model_path)
        scaler = load(scaler_path)
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"模型文件未找到！{e}\n请确保 'best_fcnn_model.joblib' 和 'scaler.joblib' 在脚本所在目录下。")
        return None, None

# ========== 特征定义 ==========
FEATURE_DETAILS = {
    'AGEyears': {
        'display': '年龄（岁）',
        'type': 'number',
        'min': 1.0, 'max': 120.0, 'default': 45.0, 'step': 1.0
    },
    'SEX': {
        'display': '性别',
        'type': 'select',
        'options': [1, 2],
        'labels': {1: '男', 2: '女'},
        'default': 1
    },
    'HEIGHT': {
        'display': '身高（cm）',
        'type': 'number',
        'min': 100.0, 'max': 220.0, 'default': 170.0, 'step': 1.0
    },
    'WEIGHT': {
        'display': '体重（kg）',
        'type': 'number',
        'min': 30.0, 'max': 250.0, 'default': 70.0, 'step': 0.5
    },
    'TEMPERATURE': {
        'display': '体温（℃）',
        'type': 'number',
        'min': 30.0, 'max': 43.0, 'default': 36.5, 'step': 0.1
    },
    'EMSSBP': {
        'display': '收缩压（mmHg）',
        'type': 'number',
        'min': 40.0, 'max': 250.0, 'default': 120.0, 'step': 1.0
    },
    'EMSPULSERATE': {
        'display': '心率（次/分）',
        'type': 'number',
        'min': 20.0, 'max': 200.0, 'default': 80.0, 'step': 1.0
    },
    'EMSRESPIRATORYRATE': {
        'display': '呼吸频率（次/分）',
        'type': 'number',
        'min': 1.0, 'max': 60.0, 'default': 18.0, 'step': 1.0
    },
    'EMSPULSEOXIMETRY': {
        'display': '血氧饱和度（%）',
        'type': 'number',
        'min': 70.0, 'max': 100.0, 'default': 98.0, 'step': 1.0
    },
    'EMSTOTALGCS': {
        'display': 'GCS评分',
        'type': 'number',
        'min': 3.0, 'max': 15.0, 'default': 15.0, 'step': 1.0
    }
}

FEATURE_ORDER = ['AGEyears', 'SEX', 'HEIGHT', 'WEIGHT', 'TEMPERATURE',
                 'EMSSBP', 'EMSPULSERATE', 'EMSRESPIRATORYRATE',
                 'EMSPULSEOXIMETRY', 'EMSTOTALGCS']

CLASS_NAMES = ["轻伤", "中等伤", "重伤", "危重伤"]
CLASS_COLORS = ["#28a745", "#ffc107", "#fd7e14", "#dc3545"]
CLASS_CSS = ["mild", "moderate", "severe", "critical"]

# ========== 创建输入表单 ==========
def create_input_form():
    input_data = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-title">基本信息</div>', unsafe_allow_html=True)
        input_data['AGEyears'] = st.number_input(
            FEATURE_DETAILS['AGEyears']['display'],
            min_value=FEATURE_DETAILS['AGEyears']['min'],
            max_value=FEATURE_DETAILS['AGEyears']['max'],
            value=FEATURE_DETAILS['AGEyears']['default'],
            step=FEATURE_DETAILS['AGEyears']['step']
        )
        input_data['SEX'] = st.selectbox(
            FEATURE_DETAILS['SEX']['display'],
            FEATURE_DETAILS['SEX']['options'],
            format_func=lambda x: FEATURE_DETAILS['SEX']['labels'][x]
        )
        input_data['HEIGHT'] = st.number_input(
            FEATURE_DETAILS['HEIGHT']['display'],
            min_value=FEATURE_DETAILS['HEIGHT']['min'],
            max_value=FEATURE_DETAILS['HEIGHT']['max'],
            value=FEATURE_DETAILS['HEIGHT']['default'],
            step=FEATURE_DETAILS['HEIGHT']['step']
        )
        input_data['WEIGHT'] = st.number_input(
            FEATURE_DETAILS['WEIGHT']['display'],
            min_value=FEATURE_DETAILS['WEIGHT']['min'],
            max_value=FEATURE_DETAILS['WEIGHT']['max'],
            value=FEATURE_DETAILS['WEIGHT']['default'],
            step=FEATURE_DETAILS['WEIGHT']['step']
        )
        input_data['TEMPERATURE'] = st.number_input(
            FEATURE_DETAILS['TEMPERATURE']['display'],
            min_value=FEATURE_DETAILS['TEMPERATURE']['min'],
            max_value=FEATURE_DETAILS['TEMPERATURE']['max'],
            value=FEATURE_DETAILS['TEMPERATURE']['default'],
            step=FEATURE_DETAILS['TEMPERATURE']['step']
        )
    
    with col2:
        st.markdown('<div class="section-title">生命体征</div>', unsafe_allow_html=True)
        input_data['EMSSBP'] = st.number_input(
            FEATURE_DETAILS['EMSSBP']['display'],
            min_value=FEATURE_DETAILS['EMSSBP']['min'],
            max_value=FEATURE_DETAILS['EMSSBP']['max'],
            value=FEATURE_DETAILS['EMSSBP']['default'],
            step=FEATURE_DETAILS['EMSSBP']['step']
        )
        input_data['EMSPULSERATE'] = st.number_input(
            FEATURE_DETAILS['EMSPULSERATE']['display'],
            min_value=FEATURE_DETAILS['EMSPULSERATE']['min'],
            max_value=FEATURE_DETAILS['EMSPULSERATE']['max'],
            value=FEATURE_DETAILS['EMSPULSERATE']['default'],
            step=FEATURE_DETAILS['EMSPULSERATE']['step']
        )
        input_data['EMSRESPIRATORYRATE'] = st.number_input(
            FEATURE_DETAILS['EMSRESPIRATORYRATE']['display'],
            min_value=FEATURE_DETAILS['EMSRESPIRATORYRATE']['min'],
            max_value=FEATURE_DETAILS['EMSRESPIRATORYRATE']['max'],
            value=FEATURE_DETAILS['EMSRESPIRATORYRATE']['default'],
            step=FEATURE_DETAILS['EMSRESPIRATORYRATE']['step']
        )
        input_data['EMSPULSEOXIMETRY'] = st.number_input(
            FEATURE_DETAILS['EMSPULSEOXIMETRY']['display'],
            min_value=FEATURE_DETAILS['EMSPULSEOXIMETRY']['min'],
            max_value=FEATURE_DETAILS['EMSPULSEOXIMETRY']['max'],
            value=FEATURE_DETAILS['EMSPULSEOXIMETRY']['default'],
            step=FEATURE_DETAILS['EMSPULSEOXIMETRY']['step']
        )
        input_data['EMSTOTALGCS'] = st.number_input(
            FEATURE_DETAILS['EMSTOTALGCS']['display'],
            min_value=FEATURE_DETAILS['EMSTOTALGCS']['min'],
            max_value=FEATURE_DETAILS['EMSTOTALGCS']['max'],
            value=FEATURE_DETAILS['EMSTOTALGCS']['default'],
            step=FEATURE_DETAILS['EMSTOTALGCS']['step']
        )
    
    return input_data

# ========== 绘制概率条形图 ==========
def plot_probabilities(probs, class_names, colors):
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(class_names, probs, color=colors, edgecolor='black', height=0.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel('概率', fontsize=12)
    ax.set_title('各类别预测概率', fontsize=14, fontweight='bold')
    
    # 在条形上添加数值标签
    for bar, prob in zip(bars, probs):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{prob:.1%}', ha='left', va='center', fontsize=12, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig

# ========== 主函数 ==========
def main():
    # 标题
    st.markdown('<div class="main-header">🏥 创伤伤员伤情分类预测系统</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">基于机器学习FCNN模型的创伤现场检伤分类辅助决策工具</div>', unsafe_allow_html=True)
    
    # 加载模型
    model, scaler = load_models()
    if model is None or scaler is None:
        st.stop()
    
    # 页面布局
    left_col, right_col = st.columns([1, 1.2])
    
    # ========== 左侧：输入区域 ==========
    with left_col:
        st.markdown('<div class="section-title">📋 患者信息录入</div>', unsafe_allow_html=True)
        st.markdown("请填写以下创伤伤员的生命体征和基本信息：")
        
        input_data = create_input_form()
        
        st.markdown("---")
        predict_btn = st.button("🔍 开始预测", use_container_width=True)
    
    # ========== 右侧：结果区域 ==========
    with right_col:
        st.markdown('<div class="section-title">📊 预测结果</div>', unsafe_allow_html=True)
        
        if not predict_btn:
            st.info("👈 请在左侧填写患者信息后点击「开始预测」")
            
            # 显示示例说明
            with st.expander("📖 使用说明"):
                st.markdown("""
                **本系统基于FCNN神经网络模型，输入10项指标预测创伤伤员的ISS伤情分类：**
                
                | 指标 | 正常范围 |
                |------|---------|
                | 收缩压 | 90-120 mmHg |
                | 心率 | 60-100 次/分 |
                | 呼吸频率 | 12-20 次/分 |
                | 血氧饱和度 | 95-100% |
                | 体温 | 36.0-37.2℃ |
                | GCS评分 | 15分（满分） |
                
                **ISS伤情分级标准：**
                - 🟢 轻伤（ISS ≤ 16）
                - 🟡 中等伤（ISS 17-25）
                - 🟠 重伤（ISS 26-50）
                - 🔴 危重伤（ISS > 50）
                """)
        else:
            # 构建输入DataFrame
            patient_df = pd.DataFrame([input_data], columns=FEATURE_ORDER)
            
            # SEX: 1→0(男), 2→1(女) 与训练时一致
            patient_df['SEX'] = patient_df['SEX'].map({1: 0, 2: 1})
            
            # 标准化
            X_scaled = scaler.transform(patient_df.values)
            
            # 预测
            probs = model.predict_proba(X_scaled)[0]
            pred_class = np.argmax(probs)
            
            # ========== 显示预测分类 ==========
            st.markdown("---")
            st.markdown(f'<div class="result-card {CLASS_CSS[pred_class]}">'
                       f'预测结果：{CLASS_NAMES[pred_class]}</div>',
                       unsafe_allow_html=True)
            
            # ========== 各类别概率 ==========
            st.markdown("---")
            st.markdown("**各类别预测概率：**")
            
            cols = st.columns(4)
            for i, (col, name, color, prob) in enumerate(zip(cols, CLASS_NAMES, CLASS_COLORS, probs)):
                with col:
                    st.markdown(f"""
                    <div style="text-align:center; padding:10px; border-radius:8px; 
                                background-color:{color}20; border:2px solid {color};">
                        <div style="font-size:0.9rem; color:#555;">{name}</div>
                        <div style="font-size:1.6rem; font-weight:bold; color:{color};">{prob:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # ========== 输入数据摘要 ==========
            st.markdown("---")
            with st.expander("📋 查看输入数据"):
                display_df = patient_df.copy()
                display_df['SEX'] = display_df['SEX'].map({0: '男', 1: '女'})
                display_df.columns = [FEATURE_DETAILS[col]['display'] for col in display_df.columns]
                st.dataframe(display_df.T, use_container_width=True)

if __name__ == "__main__":
    main()
