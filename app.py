import pickle
import streamlit as st
import pandas as pd
import plotly.graph_objects as go 

st.set_page_config(    
    page_title="乳腺癌内分泌治疗依从性预测模型",
    page_icon="⭕",
    layout="wide"
)

df = pd.read_excel("new_result8.xlsx")
train = df[df["LABEL"]==1][["time", "yaowuzhonglei", "shouru", "ade", "xuanjiao"]]

st.markdown('''
    <h1 style="font-size: 20px; text-align: center; color: black; background: #FCC422; border-radius: .5rem; margin-bottom: 1rem;">
    乳腺癌内分泌治疗依从性预测模型
    </h1>''', unsafe_allow_html=True)

# 导入模型文件
with open("LogisticRegression.pkl", 'rb') as f:
    model = pickle.load(f)

expander = st.expander("**预测输入**", True)
with expander:
    col = st.columns(5)

#d1 = {"依从性差":0, "依从性良好":1}
d2 = {"1年":1, "2年":2, "3年":3, "4年":4, "5年及以上":5}
d3 = {"SERM":1, "AI":2}
d4 = {"低收入":1, "中收入":2, "高收入":3}
d5 = {"有":1, "无":0}
d6 = {"有":1, "无":0}

#MPR	= d1[col[0].selectbox("依从性", ["依从性差", "依从性良好"])]
TIME = d2[col[0].selectbox("用药时间", ["1年", "2年", "3年", "4年", "5年及以上"])]
YAOWUZHONGLEI = d3[col[1].selectbox("药物种类", ["SERM", "AI"])]
SHOURU = d4[col[2].selectbox("经济收入", ["低收入", "中收入", "高收入"])]
ADE	= d5[col[3].selectbox("不良反应", ["有", "无"])]
XUANJIAO = d6[col[4].selectbox("药物宣教", ["有", "无"])]

predata = pd.DataFrame([
    {#"MPR":MPR, 
     "time":TIME, 
     "shouru":SHOURU, 
     "ade":ADE,
     "xuanjiao":XUANJIAO,
     "yaowuzhonglei":YAOWUZHONGLEI}])
predata = predata[list(model.feature_names_in_)]

with expander:
    st.dataframe(predata, hide_index=True, use_container_width=True)
data = predata.copy()

with st.expander("**预测结果**", True):
    #st.write(model.predict(predata))
    d = model.predict_proba(predata).flatten()
    
    # 创建仪表图  
    gauge_fig = go.Figure(go.Indicator(  
        mode='gauge+number',  
        value=d[1],  
        title={'text': f"依从性差概率:{round(d[0], 4)}，依从性良好概率:{round(d[1], 4)}"},  
        gauge={  
            'axis': {'range': [0, 1]},  
            'bar': {'color': '#FCC422'},
            'threshold': {  
                'line': {'color': 'red', 'width': 1},  # 阈值线的样式  
                'value': 0.5                             # 设置阈值  
            }
        }  
    ))  

    # 显示仪表图  
    st.plotly_chart(gauge_fig)

