import json
import pandas as pd
from stock_digest import add_indicators

def test_bollinger_population_std_and_kd_recursion():
    idx=pd.date_range("2026-01-01",periods=30,freq="B")
    close=pd.Series(range(1,31),index=idx,dtype=float)
    df=pd.DataFrame({"Open":close,"High":close+1,"Low":close-1,"Close":close,"Volume":1000},index=idx)
    cfg={"bollinger_window":20,"bollinger_std":2,"bollinger_ddof":0,"kd_window":9,"kd_alpha":1/3,"kd_initial":50,"overbought":80,"oversold":20}
    out=add_indicators(df,cfg)
    r=out.iloc[-1]
    assert round(r.BB_Mid,8)==20.5
    assert round(r.BB_Upper,8)==round(20.5+2*pd.Series(range(11,31),dtype=float).std(ddof=0),8)
    assert 0 <= r.RSV <= 100
    assert 0 <= r.K <= 100 and 0 <= r.D <= 100
