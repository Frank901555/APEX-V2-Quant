#!/usr/bin/env python3
"""APEX V2.1 research prototype. No brokerage orders. Python >=3.9."""
import argparse, json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import genpareto


def load_data(symbol, csv_path=None, offline=False):
    if not offline:
        try:
            import yfinance as yf
            x = yf.download(symbol, period='max', interval='1d', auto_adjust=False, progress=False, threads=False)
            if x.empty: raise ValueError('empty download')
            if isinstance(x.columns, pd.MultiIndex): x.columns = x.columns.get_level_values(0)
            x = x.reset_index().rename(columns={'Date':'date','Close':'close','Open':'open','High':'high','Low':'low','Volume':'volume'})
            x['date'] = pd.to_datetime(x['date']).dt.tz_localize(None)
            x = x[['date','open','high','low','close','volume']]
            x.to_csv(f'{symbol.upper()}.csv', index=False)
            return clean(x), 'Yahoo Finance via yfinance (downloaded, unadjusted)'
        except Exception as e:
            print(f'行情下載失敗，嘗試本機 CSV：{e}')
    p = Path(csv_path or f'{symbol.upper()}.csv')
    if not p.exists(): raise FileNotFoundError(f'無法取得行情，也找不到 {p}；請確認網路或 CSV')
    return clean(pd.read_csv(p)), f'Local CSV: {p}'


def clean(df):
    df.columns = [str(c).strip().lower() for c in df.columns]
    required={'date','close','high','low'}
    if not required.issubset(df.columns): raise ValueError(f'CSV 需包含 {sorted(required)} 欄位；目前為 {list(df.columns)}')
    df['date']=pd.to_datetime(df['date'], errors='coerce')
    for c in ['close','high','low']:
        df[c]=pd.to_numeric(df[c], errors='coerce')
    if 'open' not in df: df['open']=df['close'].shift(1)
    df['open']=pd.to_numeric(df['open'], errors='coerce')
    df=df.dropna(subset=['date','open','high','low','close']).sort_values('date').drop_duplicates('date')
    df=df[(df[['open','high','low','close']]>0).all(axis=1)]
    if len(df)<260: raise ValueError('至少需要 260 根日線；建議 1000 根以上')
    return df.reset_index(drop=True)


def parameters(closes):
    r=np.diff(np.log(np.asarray(closes,dtype=float)))
    med=float(np.median(r)); mad=float(1.4826*np.median(np.abs(r-med)))
    threshold=2.5*max(mad,1e-5)
    jumps=np.abs(r-med)>threshold
    ordinary=r[~jumps]
    if len(ordinary)<50: ordinary=r
    # Explicit decomposition: raw unconditional log drift = diffusion mean + lambda * mean jump.
    lam=float(np.mean(jumps)); jump_samples=r[jumps]
    jump_mean=float(np.mean(jump_samples)) if len(jump_samples) else 0.
    diffusion_mean=float(np.mean(r)-lam*jump_mean)
    diffusion_sd=float(max(np.std(ordinary,ddof=1),1e-5))
    # Simple normal-normal shrinkage on diffusion log drift, no -0.5 variance correction.
    prior_sd=0.10/math.sqrt(252)
    post_var=1/(1/prior_sd**2+len(ordinary)/diffusion_sd**2)
    post_mean=post_var*(len(ordinary)*diffusion_mean/diffusion_sd**2)
    return {'r':r,'jump_rate':lam,'jump_samples':jump_samples,'jump_mean':jump_mean,
            'diffusion_sd':diffusion_sd,'posterior_drift':float(post_mean),
            'posterior_sd':float(math.sqrt(post_var)), 'annual_vol':float(np.std(r,ddof=1)*math.sqrt(252)),
            'threshold':threshold}


def simulate(spot, par, days, paths, seed):
    rng=np.random.default_rng(seed)
    ret=rng.normal(par['posterior_drift'],par['diffusion_sd'],(paths,days))
    jumps=par['jump_samples']
    if len(jumps):
        counts=rng.poisson(par['jump_rate'],size=(paths,days))
        # Poisson compound sum, empirical jump resampling.
        for k in range(int(counts.max())):
            mask=counts>k
            ret[mask]+=rng.choice(jumps,size=int(mask.sum()),replace=True)
    sim=np.column_stack((np.full(paths,spot),spot*np.exp(np.cumsum(ret,axis=1))))
    return sim


def simulated_order(paths, entry, take, stop, days):
    # Daily close observations only: does not model intraday touches or overnight gaps.
    outcomes=[]; filled=0; first_fills=[]
    for p in paths:
        ix=np.flatnonzero(p[1:]<=entry)
        if not len(ix): continue
        t=int(ix[0]+1); filled+=1; first_fills.append(t)
        future=p[t+1:days+1]
        gain=np.flatnonzero(future>=take); loss=np.flatnonzero(future<=stop)
        tg=int(gain[0]) if len(gain) else 10**6
        tl=int(loss[0]) if len(loss) else 10**6
        if tg<tl: exit_px=take
        elif tl<=tg and tl<10**6: exit_px=stop
        else: exit_px=float(p[days])
        outcomes.append(exit_px/entry-1)
    a=np.asarray(outcomes)
    return {'fill_probability':filled/len(paths),'conditional_mean_return':float(a.mean()) if len(a) else None,
            'conditional_win_probability':float(np.mean(a>0)) if len(a) else None,
            'unconditional_expected_return_per_opportunity':float(a.sum()/len(paths)),
            'conditional_p05_return':float(np.quantile(a,.05)) if len(a) else None,
            'note':'Close-only simulation; not an executable intraday fill estimate'}


def historical_trade(df, start, days, entry, take, stop, fee):
    """Trade decision after close at start; enter on later OHLC bars. Conservative same-bar stop first."""
    end=min(start+days,len(df)-1)
    buy_i=None; buy_px=None
    for i in range(start+1,end+1):
        b=df.iloc[i]
        if b.low<=entry:
            buy_i=i; buy_px=min(entry,float(b.open)) if b.open<entry else entry
            break
    if buy_i is None: return {'filled':False,'return':0.,'exit':'no_fill'}
    for i in range(buy_i,end+1):
        b=df.iloc[i]
        # If entry happened today, only apply same-day stop when it could follow entry;
        # stop-first is conservative, but intrabar sequence remains unknown.
        if i==buy_i:
            if b.low<=stop: return {'filled':True,'return':stop/buy_px-1-fee,'exit':'stop_same_bar_ambiguous'}
            continue
        if b.open<=stop: return {'filled':True,'return':float(b.open)/buy_px-1-fee,'exit':'gap_stop'}
        if b.open>=take: return {'filled':True,'return':float(b.open)/buy_px-1-fee,'exit':'gap_take'}
        if b.low<=stop: return {'filled':True,'return':stop/buy_px-1-fee,'exit':'stop'}
        if b.high>=take: return {'filled':True,'return':take/buy_px-1-fee,'exit':'take'}
    return {'filled':True,'return':float(df.iloc[end].close)/buy_px-1-fee,'exit':'time'}


def backtest(df, days, entry_drop, take_gain, stop_loss, fee, indices):
    trades=[]
    for i in indices:
        spot=float(df.iloc[i].close)
        entry=spot*(1-entry_drop)
        trades.append(historical_trade(df,i,days,entry,entry*(1+take_gain),entry*(1-stop_loss),fee))
    a=np.asarray([t['return'] for t in trades if t['filled']],dtype=float)
    n=len(trades); f=len(a)
    if not f: return {'opportunities':n,'filled':0,'fill_rate':0.,'mean_return_filled':None,'win_rate_filled':None,'mean_return_per_opportunity':0.,'max_drawdown_sequential':None}
    equity=np.cumprod(1+a); dd=equity/np.maximum.accumulate(np.r_[1,equity])[1:]-1
    return {'opportunities':n,'filled':f,'fill_rate':f/n,'mean_return_filled':float(a.mean()),
            'win_rate_filled':float(np.mean(a>0)),'mean_return_per_opportunity':float(a.sum()/n),
            'max_drawdown_sequential':float(dd.min()),'trades':trades}


def evt_tail(r):
    losses=-r; u=float(np.quantile(losses,.90)); ex=losses[losses>u]-u
    if len(ex)<25:return {'status':'insufficient','exceedances':int(len(ex))}
    shape,_,scale=genpareto.fit(ex,floc=0)
    return {'status':'fitted_in_sample','exceedances':int(len(ex)), 'threshold':u,'shape':float(shape),'scale':float(scale)}


def hmm_state(r):
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        fit=MarkovRegression(r,k_regimes=2,trend='c',switching_variance=True).fit(disp=False,maxiter=200,em_iter=5,search_reps=3)
        probs=np.asarray(fit.filtered_marginal_probabilities)
        return {'status':'fitted_in_sample','last_filtered_probabilities':probs[-1].tolist(),
                'note':'Regime labels are arbitrary; in-sample parameters, not out-of-sample calibrated'}
    except Exception as e:return {'status':'failed','reason':str(e)[:150]}


def run(args):
    df,source=load_data(args.symbol,args.csv,args.offline)
    p=parameters(df.close.values); spot=float(df.close.iloc[-1]); n=len(df)
    sim=simulate(spot,p,args.days,args.paths,args.seed)
    # Disjoint evaluation windows to avoid overlapping outcomes; train/test time split.
    idx=np.arange(max(260,int(n*.25)),n-args.days-1,args.days,dtype=int)
    split=int(len(idx)*.7); train=idx[:split]; test=idx[split:]
    if len(train)<8 or len(test)<4:raise ValueError('Not enough non-overlapping historical windows for test')
    # Fixed, predeclared grid; no tuning on held-out test.
    grid=[]
    for drop in [.0,.03,.05,.08]:
        for gain in [.08,.12,.18]:
            for loss in [.05,.08]:
                bt=backtest(df,args.days,drop,gain,loss,args.fee,train)
                # Minimum fills; penalize variability and low fill rate.
                rets=np.array([x['return'] for x in bt.get('trades',[]) if x['filled']])
                if len(rets)<6:continue
                score=float(bt['mean_return_per_opportunity']-0.3*np.std(rets,ddof=1)/math.sqrt(len(rets)))
                grid.append((score,drop,gain,loss))
    if not grid:raise ValueError('Insufficient training trades for candidate optimization')
    _,drop,gain,loss=max(grid,key=lambda x:x[0])
    train_bt=backtest(df,args.days,drop,gain,loss,args.fee,train)
    test_bt=backtest(df,args.days,drop,gain,loss,args.fee,test)
    entry=spot*(1-drop); take=entry*(1+gain); stop=entry*(1-loss)
    sim_order=simulated_order(sim,entry,take,stop,args.days)
    returns=sim[:,-1]/spot-1
    result={'engine':'APEX V2.1 research prototype','symbol':args.symbol.upper(),
      'data':{'source':source,'bars':n,'last_date':str(df.date.iloc[-1].date()),'last_close':spot},
      'parameters':{'bayesian_diffusion_drift_daily':p['posterior_drift'],'posterior_sd':p['posterior_sd'],
                    'historical_annualized_vol':p['annual_vol'],'jump_rate_heuristic':p['jump_rate'],'jump_events':len(p['jump_samples'])},
      'simulation':{'paths':args.paths,'days':args.days,'mean_terminal_return':float(returns.mean()),
                    'median_terminal_return':float(np.median(returns)),'p05_terminal_return':float(np.quantile(returns,.05)),
                    'prob_terminal_positive':float(np.mean(returns>0)),'order':sim_order},
      'strategy':{'entry':round(entry,2),'take':round(take,2),'stop':round(stop,2),
                  'holding_days':args.days,'train_nonoverlap':{k:v for k,v in train_bt.items() if k!='trades'},
                  'test_nonoverlap':{k:v for k,v in test_bt.items() if k!='trades'},
                  'test_exits':pd.Series([x['exit'] for x in test_bt['trades']]).value_counts().to_dict(),
                  'roundtrip_fee_slippage_assumption':args.fee},
      'evt':evt_tail(p['r']),'hmm':hmm_state(p['r']),
      'limitations':['Selected grid on chronological train only, evaluated once on later test; NOT a walk-forward refit.',
       'Historical OHLC bar order is unknown; same-day stop is conservative and cannot resolve all intrabar ambiguity.',
       'Monte Carlo uses close-only empirical jump resampling and fixed diffusion volatility, not Heston.',
       'Simulated fill and exit rates are NOT calibrated to actual intraday executions.',
       'Event-specific earnings gaps and option implied volatility surface NOT included.',
       'Train/test strategy statistics are for non-overlapping opportunities, not an investable portfolio equity curve.',
       'Use adjusted prices / corporate actions and timestamped fundamentals for professional deployment.']}
    out=Path(args.out or f'{args.symbol.upper()}_v21_result.json')
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False,default=float),encoding='utf-8')
    lines=[f'APEX V2.1 | {args.symbol.upper()} | {result["data"]["last_date"]}',
           f'來源：{source}；歷史日線：{n}；最新收盤：${spot:.2f}',
           f'研究候選買入 ${entry:.2f} / 停利 ${take:.2f} / 停損 ${stop:.2f} / {args.days}交易日',
           f'歷史訓練樣本：{train_bt["opportunities"]}次機會，成交 {train_bt["filled"]} 次；每次機會平均報酬 {train_bt["mean_return_per_opportunity"]:.2%}',
           f'樣本外測試：{test_bt["opportunities"]}次機會，成交 {test_bt["filled"]} 次；每次機會平均報酬 {test_bt["mean_return_per_opportunity"]:.2%}',
           f'樣本外已成交勝率：{test_bt["win_rate_filled"]:.2%}' if test_bt['win_rate_filled'] is not None else '樣本外無成交',
           f'模擬上漲機率：{np.mean(returns>0):.2%}；模擬限價成交機率：{sim_order["fill_probability"]:.2%}',
           '注意：以上為研究估計，並非可執行報價、保證勝率或實盤績效。',
           'Heston完整校準、財報事件跳躍與嚴格滾動重訓尚未實作。']
    report=out.with_suffix('.txt');report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines));print(f'\n輸出：{out}、{report}')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='APEX V2.1 quant research prototype')
    parser.add_argument('symbol',help='例如 RBLX')
    parser.add_argument('--csv',help='備援本機CSV路徑')
    parser.add_argument('--offline',action='store_true',help='只使用本機CSV')
    parser.add_argument('--days',type=int,default=20)
    parser.add_argument('--paths',type=int,default=12000)
    parser.add_argument('--fee',type=float,default=.003,help='來回交易成本與滑價假設 (0.003=0.3%%)')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--out',help='結果JSON檔名')
    a=parser.parse_args()
    if not 5<=a.days<=60 or not 1000<=a.paths<=100000:parser.error('days must be 5..60, paths 1000..100000')
    run(a)
