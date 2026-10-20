"""APEX V2.1 local browser interface. Research only; no brokerage integration."""
import json, re, subprocess, sys, tempfile
from pathlib import Path
import pandas as pd
import streamlit as st

ENGINE = Path(__file__).resolve().parent / 'apex_engine_v21.py'
st.set_page_config(page_title='APEX | Quant Research', page_icon='📈', layout='wide')
st.markdown("""<style>
/* APEX mobile-first light finance dashboard; presentation only. */
.stApp, [data-testid="stAppViewContainer"] {background:#f3f7fc;color:#15253c;}
[data-testid="stHeader"] {background:rgba(243,247,252,.95);}
[data-testid="stSidebar"] {background:#e7effa;}
[data-testid="stSidebar"] * {color:#17304f;}
.block-container {max-width:1180px;padding-top:1.7rem;padding-bottom:3rem;}
h1,h2,h3 {letter-spacing:-.025em;color:#12335b;}
h1 {font-weight:850!important;}
p,li,label {line-height:1.65;}
[data-testid="stMetric"] {
 background:#fff!important;border:1px solid #d9e4f1;border-left:5px solid #367ee9;
 border-radius:16px;padding:16px;box-shadow:0 5px 18px rgba(19,47,85,.07);
}
[data-testid="stMetricLabel"] {color:#4c6382;}
[data-testid="stMetricValue"] {color:#132f53;font-weight:800;}
[data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stMetric"] {border-left-color:#d5a029;}
[data-testid="stHorizontalBlock"] > div:nth-child(3) [data-testid="stMetric"] {border-left-color:#16966c;}
[data-testid="stHorizontalBlock"] > div:nth-child(4) [data-testid="stMetric"] {border-left-color:#d8545c;}
[data-testid="stAlert"] {border-radius:14px;}
[data-testid="stExpander"] {background:#fff;border:1px solid #d9e4f1;border-radius:12px;}
div.stButton > button[kind="primary"] {background:#1763c6;color:white;border:0;border-radius:11px;font-weight:800;min-height:46px;}
div.stButton > button[kind="primary"]:hover {background:#104d9e;color:#fff;}
[data-testid="stDataFrame"],[data-testid="stJson"] {border-radius:12px;}
@media(max-width:700px) {
 .block-container {padding:1.2rem .8rem 2rem;}
 h1 {font-size:2.1rem!important;}
 [data-testid="stMetric"] {padding:12px;}
 [data-testid="stMetricValue"] {font-size:1.45rem;}
}
</style>""", unsafe_allow_html=True)
st.title('APEX · V2.1')
st.caption('Advanced Predictive Equity eXecution | 本機美股量化研究儀表板')
st.warning('研究原型：不連接券商、不會下單。機率是模型估計，不是已驗證的真實勝率。Heston 完整校準及財報事件專屬模型尚未實作。',icon='⚠️')
with st.sidebar:
    st.header('分析設定')
    symbol=st.text_input('美股代號',value='RBLX').strip().upper()
    days=st.select_slider('持有交易日',options=[5,10,15,20,30,40,60],value=20)
    paths=st.select_slider('模擬路徑',options=[1000,3000,6000,12000,25000,50000],value=12000)
    fee=st.number_input('來回交易成本及滑價 (%)',min_value=0.,max_value=10.,value=.30,step=.05)
    uploaded=st.file_uploader('備援歷史行情 CSV（可選）',type=['csv'])
    offline=st.checkbox('只使用上傳 CSV（不連線）',value=False)
    go=st.button('▶ 開始 APEX 分析',type='primary',use_container_width=True)
    st.caption('線上資料：Yahoo Finance via yfinance。不是 TradingView 付費資料直接授權。')

if go:
    if not re.fullmatch(r'[A-Z0-9.^=-]{1,20}',symbol):
        st.error('股票代號格式不正確。')
    elif offline and uploaded is None:
        st.error('勾選離線模式時，請先上傳 CSV。')
    else:
        with st.spinner('下載行情、模擬價格路徑並執行歷史測試中…'):
            with tempfile.TemporaryDirectory(prefix='apexweb_') as tmp:
                out=Path(tmp)/'report.json'
                cmd=[sys.executable,str(ENGINE),symbol,'--days',str(days),'--paths',str(paths),'--fee',str(fee/100),'--out',str(out)]
                if uploaded is not None:
                    f=Path(tmp)/'input.csv';f.write_bytes(uploaded.getvalue());cmd.extend(['--csv',str(f)])
                if offline:cmd.append('--offline')
                try:
                    p=subprocess.run(cmd,cwd=tmp,text=True,capture_output=True,timeout=240)
                    if p.returncode!=0:
                        st.error('分析未完成。請檢查網路或 CSV 格式。')
                        st.code((p.stderr or p.stdout)[-3500:])
                    else:
                        result=json.loads(out.read_text(encoding='utf-8'))
                        st.session_state['result']=result
                        st.session_state['report_json']=out.read_text(encoding='utf-8')
                        st.session_state['report_txt']=out.with_suffix('.txt').read_text(encoding='utf-8')
                except subprocess.TimeoutExpired:
                    st.error('執行超過 240 秒，請先減少模擬路徑數量。')
                except Exception as e:
                    st.error(f'執行失敗：{e}')

if 'result' in st.session_state:
    r=st.session_state['result']; d=r['data']; strat=r['strategy']; sim=r['simulation']; order=sim['order']; test=strat['test_nonoverlap']
    st.success(f"{r['symbol']} 分析完成｜資料截至 {d['last_date']}｜來源：{d['source']}")
    c1,c2,c3,c4=st.columns(4)
    c1.metric('最近收盤',f"${d['last_close']:.2f}")
    c2.metric('研究候選買入',f"${strat['entry']:.2f}")
    c3.metric('目標賣出',f"${strat['take']:.2f}")
    c4.metric('策略停損',f"${strat['stop']:.2f}")
    st.subheader('模擬與回測')
    c1,c2,c3,c4=st.columns(4)
    c1.metric('模擬期末上漲機率',f"{sim['prob_terminal_positive']:.1%}")
    c2.metric('模擬期末中位報酬',f"{sim['median_terminal_return']:.1%}")
    c3.metric('模擬期末5%分位',f"{sim['p05_terminal_return']:.1%}")
    c4.metric('樣本外機會數',str(test['opportunities']))
    st.markdown('**限價單路徑估計**')
    st.json(order,expanded=False)
    st.markdown('**歷史測試（訓練期與樣本外）**')
    st.dataframe(pd.DataFrame([{'區間':'訓練期',**strat['train_nonoverlap']},{'區間':'樣本外',**test}]),use_container_width=True,hide_index=True)
    with st.expander('模型參數與風險診斷'):
        st.json({'parameters':r['parameters'],'evt':r['evt'],'hmm':r['hmm']},expanded=True)
    with st.expander('重要限制',expanded=True):
        for item in r['limitations']:st.write('• '+item)
    a,b=st.columns(2)
    a.download_button('⬇ 下載中文研究報告 TXT',st.session_state['report_txt'],file_name=f"{r['symbol']}_v21_report.txt",mime='text/plain',use_container_width=True)
    b.download_button('⬇ 下載完整 JSON',st.session_state['report_json'],file_name=f"{r['symbol']}_v21_result.json",mime='application/json',use_container_width=True)
else:
    st.info('左側輸入股票代號，按「開始 APEX 分析」。若網路無法取得行情，可上傳本機 CSV。')
    st.markdown('### 運作流程\n歷史行情 → 貝葉斯漂移與經驗跳躍 → 蒙地卡羅 → 候選策略搜尋 → 時序樣本外交易測試 → 風險報告')
