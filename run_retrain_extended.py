#!/usr/bin/env python3
"""
Retrain on extended dataset (2015-2026), backtest, regenerate 2026 predictions.
Uses historical_rounds_extended.parquet instead of original CSV.
"""
import os, json, warnings, sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import requests
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

PROCESSED = Path("data/processed")
# USE EXTENDED DATASET
HIST_PATH = PROCESSED / "historical_rounds_extended.parquet"
API_KEY = os.getenv("DATAGOLF_API_KEY")
DG_BASE = "https://feeds.datagolf.com"
HARDCODED_CW = {"sg_ott": 0.85, "sg_app": 1.45, "sg_arg": 1.30, "sg_putt": 1.10}

XGB_S1 = {"n_estimators":600,"learning_rate":0.02,"max_depth":4,"subsample":0.8,
           "colsample_bytree":0.8,"min_child_weight":10,"reg_alpha":0.1,"reg_lambda":1.0,
           "objective":"reg:squarederror","random_state":42}
XGB_S2 = {"n_estimators":300,"learning_rate":0.03,"max_depth":3,"subsample":0.8,
           "colsample_bytree":0.7,"min_child_weight":5,"reg_alpha":0.5,"reg_lambda":2.0,
           "objective":"binary:logistic","eval_metric":"auc","random_state":42}
N_SIMS=50000; NOISE=0.10; TPRED=0.14; SEED=42; BLEND=0.90; TEMP=2.5

SG = ["sg_ott","sg_app","sg_arg","sg_putt","sg_t2g","sg_total"]
ROLL = ["sg_ott_3w","sg_app_3w","sg_arg_3w","sg_putt_3w","sg_t2g_3w","sg_total_3w",
        "sg_ott_8w","sg_app_8w","sg_arg_8w","sg_putt_8w","sg_t2g_8w","sg_total_8w",
        "sg_total_std_8","sg_total_momentum","sg_ball_strike_ratio","log_field_size"]
AUG = ["augusta_competitive_rounds","augusta_made_cut_prev_year","augusta_experience_tier",
       "augusta_scoring_trajectory","augusta_rounds_last_2yrs","augusta_best_finish_recent",
       "augusta_starts","augusta_made_cut_rate","augusta_top10_rate","augusta_best_finish",
       "augusta_scoring_avg","augusta_sg_app_career","augusta_sg_total_career",
       "augusta_bogey_avoidance","augusta_round_variance_score","augusta_back9_scoring",
       "tournament_wind_avg","tour_vs_augusta_divergence"]
S2F = ROLL + AUG

def pfn(p):
    if p is None or pd.isna(p): return None
    p=str(p).strip().upper()
    if p in ("CUT","MC","WD","DQ","MDF"): return 999
    p=p.replace("T","").replace("=","")
    try: return int(p)
    except: return None

def norm_name(n):
    n=str(n).strip()
    if "," in n:
        parts=n.split(",",1); return f"{parts[1].strip()} {parts[0].strip()}"
    return n

def build_roll(df):
    df=df.sort_values(["player_name","date"]).copy()
    for c in SG:
        df[f"{c}_3w"]=df.groupby("player_name")[c].transform(lambda x:x.ewm(span=3,min_periods=1).mean())
        df[f"{c}_8w"]=df.groupby("player_name")[c].transform(lambda x:x.ewm(span=8,min_periods=1).mean())
    df["sg_total_std_8"]=df.groupby("player_name")["sg_total"].transform(lambda x:x.rolling(8,min_periods=2).std())
    df["sg_total_momentum"]=df["sg_total_3w"]-df["sg_total_8w"]
    df["sg_ball_strike_ratio"]=(df["sg_ott_8w"]+df["sg_app_8w"])/(df["sg_ott_8w"].abs()+df["sg_app_8w"].abs()+df["sg_arg_8w"].abs()+df["sg_putt_8w"].abs()+1e-6)
    df["log_field_size"]=np.log(df["field_size"].clip(1))
    return df

def apply_cw(df,w):
    df=df.copy()
    for c,wt in w.items():
        if c in df.columns: df[c]=df[c]*wt
    return df

def run_mc(p):
    rng=np.random.RandomState(SEED);n=len(p);mu,s=p.mean(),p.std()
    z=(p-mu)/s*TPRED if s>0 else np.zeros(n)
    w=np.zeros(n);t5=np.zeros(n);t10=np.zeros(n);t20=np.zeros(n)
    for _ in range(N_SIMS):
        sim=z+rng.normal(0,NOISE,n);r=sim.argsort().argsort()+1
        w+=(r==1);t5+=(r<=5);t10+=(r<=10);t20+=(r<=20)
    return {"win":w/N_SIMS,"top5":t5/N_SIMS,"top10":t10/N_SIMS,"top20":t20/N_SIMS}


def main():
    print("="*60)
    print("  RETRAIN ON EXTENDED DATA + BACKTEST + 2026 PREDICTIONS")
    print("="*60)

    # Load data
    tour = pd.read_parquet(HIST_PATH)
    tour["finish_num"]=tour["finish_pos"].apply(pfn)
    unified = pd.read_parquet(PROCESSED/"masters_unified.parquet")
    unified["finish_num"]=unified["finish_pos"].apply(pfn)
    features = pd.read_parquet(PROCESSED/"augusta_player_features.parquet")
    weather = pd.read_parquet(PROCESSED/"masters_weather.parquet") if (PROCESSED/"masters_weather.parquet").exists() else None

    cw_path = PROCESSED/"augusta_sg_weights.json"
    cw = json.load(open(cw_path)) if cw_path.exists() else HARDCODED_CW

    print(f"\n  Tour data: {len(tour):,} rows, seasons {sorted(tour['season'].unique())}")
    print(f"  Unified Masters: {len(unified):,} rows")

    # Precompute rolling features for ALL tour data
    print("  Computing rolling features on full extended dataset...")
    tour_clean = tour.dropna(subset=SG)
    tour_clean = tour_clean[tour_clean["finish_num"]<999]
    tour_w = apply_cw(tour_clean, cw)
    tour_feat = build_roll(tour_w)
    tour_feat["finish_pct"]=((tour_feat["finish_num"]-1)/(tour_feat["field_size"]-1)).clip(0,1)
    print(f"  Rolling features: {len(tour_feat):,} rows")

    # ════════════════════════════════════════
    # BACKTEST (2021-2025)
    # ════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD BACKTEST (2021-2025)")
    print(f"{'='*60}")

    sg_years = sorted(unified[unified["has_sg_data"]==True]["season"].unique())
    bt_years = [y for y in sg_years if y > min(unified["season"].unique())]
    metrics_list = []
    all_bt_results = []

    for year in bt_years:
        # Stage 1
        train = tour_feat[tour_feat["season"]<year].dropna(subset=ROLL+["finish_pct"])
        s1 = xgb.XGBRegressor(**XGB_S1)
        s1.fit(train[ROLL].values, train["finish_pct"].values)

        # Build field
        field = unified[(unified["season"]==year)&(unified["has_sg_data"]==True)].copy()
        if len(field)==0: continue
        ff=[]
        for _,pr in field.iterrows():
            pn=pr["player_name"]
            pt=tour_feat[(tour_feat["player_name"]==pn)&(tour_feat["season"]<year)]
            if len(pt)==0: continue
            lat=pt.iloc[-1]
            d={"player_name":pn}
            for c in ROLL: d[c]=lat.get(c)
            # Augusta features
            fr=features[(features["player_name"]==pn)&(features["season"]==year)]
            if len(fr)>0:
                fr=fr.iloc[0]
                for c in AUG:
                    if c not in ("tournament_wind_avg","tour_vs_augusta_divergence"): d[c]=fr.get(c)
            else:
                for c in AUG:
                    if c not in ("tournament_wind_avg","tour_vs_augusta_divergence"): d[c]=0.0
            d["tournament_wind_avg"]=10.0; d["tour_vs_augusta_divergence"]=0.0
            d["actual_finish_num"]=pr.get("finish_num"); d["actual_finish_pos"]=pr.get("finish_pos")
            ff.append(d)

        fdf=pd.DataFrame(ff).dropna(subset=ROLL)
        if len(fdf)<5: continue

        # Divergence
        sg8=fdf["sg_total_8w"].rank(pct=True)
        hr=fdf["augusta_competitive_rounds"]>=6
        sa=fdf.loc[hr,"augusta_scoring_avg"]
        if len(sa)>2:
            sap=(-sa).rank(pct=True)
            fdf.loc[hr,"tour_vs_augusta_divergence"]=sg8[hr]-sap
        fdf["tour_vs_augusta_divergence"]=fdf["tour_vs_augusta_divergence"].fillna(0)

        # MC
        preds=s1.predict(fdf[ROLL].values)
        mc=run_mc(preds); fdf["mc_top10"]=mc["top10"]

        # Stage 2 (Augusta-only)
        s2t=unified[unified["season"]<year].copy()
        s2t["made_top10"]=(s2t["finish_num"]<=10).astype(int)
        s2t=s2t.merge(features,on=["player_name","season"],how="left")
        s2r=[]
        for _,row in s2t.iterrows():
            pn,ps=row["player_name"],row["season"]
            pt=tour_feat[(tour_feat["player_name"]==pn)&(tour_feat["season"]<=ps)]
            f2={"idx":row.name}
            if len(pt)>0:
                lat=pt.iloc[-1]
                for c in ROLL: f2[c]=lat.get(c)
            else:
                for c in ROLL: f2[c]=np.nan
            f2["tournament_wind_avg"]=10.0; f2["tour_vs_augusta_divergence"]=0.0
            s2r.append(f2)
        s2rf=pd.DataFrame(s2r).set_index("idx")
        for c in S2F:
            if c not in s2t.columns and c in s2rf.columns: s2t[c]=s2rf[c]
            elif c not in s2t.columns: s2t[c]=np.nan
        s2t=s2t[s2t["finish_num"].notna()]
        np_=s2t["made_top10"].sum()
        spw=(len(s2t)-np_)/np_ if np_>0 else 8.0
        s2m=xgb.XGBClassifier(**{**XGB_S2,"scale_pos_weight":spw})
        s2m.fit(s2t[S2F],s2t["made_top10"])

        for c in S2F:
            if c not in fdf.columns: fdf[c]=0.0
        s2p=s2m.predict_proba(fdf[S2F])[:,1]
        fdf["s2_top10"]=s2p
        fdf["blend_top10"]=BLEND*s2p+(1-BLEND)*fdf["mc_top10"]
        fdf["season"]=year

        # Score
        v=fdf[fdf["actual_finish_num"]<999].copy()
        at10=(v["actual_finish_num"]<=10).astype(int)
        m={"year":year}
        if at10.sum()>0 and at10.sum()<len(at10):
            m["s2_auc"]=roc_auc_score(at10,v["s2_top10"])
            m["blend_auc"]=roc_auc_score(at10,v["blend_top10"])
            m["mc_auc"]=roc_auc_score(at10,v["mc_top10"])
        t10p=v.nlargest(10,"blend_top10")
        m["top10_prec"]=(t10p["actual_finish_num"]<=10).sum()/10
        sp,_=stats.spearmanr(v["mc_top10"],-v["actual_finish_num"]) if len(v)>5 else (0,0)
        m["spearman"]=sp
        metrics_list.append(m)
        all_bt_results.append(fdf)
        print(f"  {year}: field={len(fdf)} | S2 AUC={m.get('s2_auc',0):.3f} | T10 Prec={m['top10_prec']:.0%} | Spearman={sp:.3f}")

    # Summary
    print(f"\n  {'Year':<6} {'S2 AUC':>8} {'Blend AUC':>10} {'MC AUC':>8} {'T10 Prec':>9} {'Spearman':>9}")
    print(f"  {'─'*6} {'─'*8} {'─'*10} {'─'*8} {'─'*9} {'─'*9}")
    for m in metrics_list:
        s=lambda k,f=".3f": f"{m[k]:{f}}" if m.get(k) else "N/A"
        print(f"  {m['year']:<6} {s('s2_auc'):>8} {s('blend_auc'):>10} {s('mc_auc'):>8} {s('top10_prec','.0%'):>9} {s('spearman'):>9}")
    def avg(k): vs=[m[k] for m in metrics_list if m.get(k)]; return np.mean(vs) if vs else 0
    print(f"  {'AVG':<6} {avg('s2_auc'):>8.3f} {avg('blend_auc'):>10.3f} {avg('mc_auc'):>8.3f} {avg('top10_prec'):>8.0%} {avg('spearman'):>9.3f}")

    # Save backtest
    if all_bt_results:
        bt_df=pd.concat(all_bt_results,ignore_index=True)
        bt_df.to_parquet(PROCESSED/"backtest_results_v4.parquet",index=False)

    # Compare to V3
    v3_aucs = [0.500, 0.674, 0.669, 0.738, 0.600]  # from CLAUDE.md
    v3_precs = [0.30, 0.50, 0.40, 0.60, 0.30]
    print(f"\n  V3 vs V4 comparison:")
    for i,m in enumerate(metrics_list):
        v3a = v3_aucs[i] if i<len(v3_aucs) else 0
        v4a = m.get("s2_auc",0) or 0
        v3p = v3_precs[i] if i<len(v3_precs) else 0
        v4p = m.get("top10_prec",0)
        print(f"    {m['year']}: S2 AUC {v3a:.3f} -> {v4a:.3f} ({v4a-v3a:+.3f}) | T10 Prec {v3p:.0%} -> {v4p:.0%}")

    # ════════════════════════════════════════
    # 2026 PREDICTIONS
    # ════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  2026 PREDICTIONS (RETRAINED ON EXTENDED DATA)")
    print(f"{'='*60}")

    # Train final S1 on ALL data
    tc=tour_feat.dropna(subset=ROLL+["finish_pct"])
    s1_final=xgb.XGBRegressor(**XGB_S1)
    s1_final.fit(tc[ROLL].values,tc["finish_pct"].values)
    print(f"  S1 trained on {len(tc):,} rows (was ~16K, now {len(tc):,})")

    # Build 2026 field features from extended data
    old_preds=pd.read_parquet(PROCESSED/"predictions_2026.parquet")
    aug_cols=[c for c in old_preds.columns if c.startswith("augusta_") or c in ["tour_vs_augusta_divergence","dg_id","country","dg_rank"]]

    roll_rows=[]
    for _,pr in old_preds.iterrows():
        pn=pr["player_name"]
        pt=tour_feat[tour_feat["player_name"]==pn]
        d={"player_name":pn}
        if len(pt)>0:
            lat=pt.iloc[-1]
            for c in ROLL: d[c]=lat.get(c)
            d["_has_data"]=True
        else:
            d["_has_data"]=False
            for c in ROLL: d[c]=0
        roll_rows.append(d)

    roll_df=pd.DataFrame(roll_rows)
    fdf=roll_df.merge(old_preds[["player_name"]+aug_cols],on="player_name",how="left")
    for c in ROLL: fdf[c]=fdf[c].fillna(0)

    no_data=fdf[fdf["_has_data"]==False]["player_name"].tolist()
    if no_data:
        print(f"  Players without extended tour data: {no_data}")

    # S1 + MC
    preds=s1_final.predict(fdf[ROLL].values)
    mc=run_mc(preds)
    fdf["win_prob"]=mc["win"];fdf["top5_prob"]=mc["top5"]
    fdf["mc_top10"]=mc["top10"];fdf["top20_prob"]=mc["top20"]
    fdf["model_score"]=preds

    # S2 final
    s2t=unified.copy()
    s2t["made_top10"]=(s2t["finish_num"]<=10).astype(int)
    s2t=s2t.merge(features,on=["player_name","season"],how="left")
    s2r=[]
    for _,row in s2t.iterrows():
        pn,ps=row["player_name"],row["season"]
        pt=tour_feat[(tour_feat["player_name"]==pn)&(tour_feat["season"]<=ps)]
        f2={"idx":row.name}
        if len(pt)>0:
            lat=pt.iloc[-1]
            for c in ROLL: f2[c]=lat.get(c)
        else:
            for c in ROLL: f2[c]=np.nan
        f2["tournament_wind_avg"]=10.0;f2["tour_vs_augusta_divergence"]=0.0
        s2r.append(f2)
    s2rf=pd.DataFrame(s2r).set_index("idx")
    for c in S2F:
        if c not in s2t.columns and c in s2rf.columns: s2t[c]=s2rf[c]
        elif c not in s2t.columns: s2t[c]=np.nan
    s2t=s2t[s2t["finish_num"].notna()]
    np_=s2t["made_top10"].sum()
    spw=(len(s2t)-np_)/np_ if np_>0 else 8
    s2_final=xgb.XGBClassifier(**{**XGB_S2,"scale_pos_weight":spw})
    s2_final.fit(s2t[S2F],s2t["made_top10"])

    for c in S2F:
        if c not in fdf.columns: fdf[c]=0.0

    # Divergence
    sg8=fdf["sg_total_8w"].rank(pct=True)
    hr=fdf["augusta_competitive_rounds"]>=6
    sa=fdf.loc[hr,"augusta_scoring_avg"]
    if len(sa)>2:
        sap=(-sa).rank(pct=True)
        fdf.loc[hr,"tour_vs_augusta_divergence"]=sg8[hr]-sap
    fdf["tour_vs_augusta_divergence"]=fdf["tour_vs_augusta_divergence"].fillna(0)

    s2raw=s2_final.predict_proba(fdf[S2F])[:,1]
    fdf["stage2_prob_raw"]=s2raw
    # Temperature scaling
    logits=np.log(np.clip(s2raw,1e-6,1-1e-6)/(1-np.clip(s2raw,1e-6,1-1e-6)))
    s2cal=1/(1+np.exp(-logits/TEMP))
    fdf["stage2_prob_calibrated"]=s2cal
    fdf["top10_prob_calibrated"]=BLEND*s2cal+(1-BLEND)*fdf["mc_top10"]

    # Monotonic
    fdf["make_cut_prob"]=fdf[["top20_prob","mc_top10"]].max(axis=1).clip(upper=0.95)
    fdf["top20_prob"]=fdf[["top20_prob","top10_prob_calibrated"]].max(axis=1)
    fdf["top5_prob"]=np.minimum(fdf["top5_prob"].clip(lower=fdf["win_prob"]).values,fdf["top10_prob_calibrated"].values)

    # Join DK odds
    dk_path=PROCESSED/"dk_odds_2026.csv"
    if dk_path.exists():
        from rapidfuzz import fuzz,process
        dk=pd.read_csv(dk_path);dk_names=dk["player_name"].tolist()
        for _,orow in dk.iterrows():
            best=process.extractOne(orow["player_name"],fdf["player_name"].tolist(),scorer=fuzz.ratio)
            if best and best[1]>=85:
                idx=fdf[fdf["player_name"]==best[0]].index
                if len(idx)>0:
                    fdf.loc[idx[0],"dk_fair_prob_win"]=orow["dk_fair_prob"]
                    fdf.loc[idx[0],"dk_fair_prob_top10"]=orow["dk_fair_top10"]
                    fdf.loc[idx[0],"dk_american_odds"]=orow["dk_american_odds"]
        fdf["kelly_edge_win"]=(fdf["win_prob"]-fdf.get("dk_fair_prob_win",0))/fdf.get("dk_fair_prob_win",1).clip(lower=1e-6)
        fdf["kelly_edge_top10"]=(fdf["top10_prob_calibrated"]-fdf.get("dk_fair_prob_top10",0))/fdf.get("dk_fair_prob_top10",1).clip(lower=1e-6)

    fdf=fdf.sort_values("top10_prob_calibrated",ascending=False).reset_index(drop=True)

    # Print top 20
    print(f"\n  {'Rk':<3} {'Player':<25} {'Win%':>6} {'T10%':>6} {'T20%':>6} {'DK':>8} {'sg3w':>7} {'Tier':>4} {'CutPv':>5} {'Traj':>6}")
    print(f"  {'─'*3} {'─'*25} {'─'*6} {'─'*6} {'─'*6} {'─'*8} {'─'*7} {'─'*4} {'─'*5} {'─'*6}")
    for i,(_,r) in enumerate(fdf.head(20).iterrows()):
        dk=f"+{r['dk_american_odds']:.0f}" if pd.notna(r.get('dk_american_odds')) else "N/A"
        sg3=f"{r.get('sg_total_3w',0):.3f}"
        print(f"  {i+1:<3} {r['player_name']:<25} {r['win_prob']:>5.1%} {r['top10_prob_calibrated']:>5.1%} "
              f"{r['top20_prob']:>5.1%} {dk:>8} {sg3:>7} {r.get('augusta_experience_tier',0):>4.0f} "
              f"{r.get('augusta_made_cut_prev_year',0):>5.0f} {r.get('augusta_scoring_trajectory',0):>+6.2f}")

    # Save
    out_cols=[c for c in [
        "dg_id","player_name","country","dg_rank","win_prob","top5_prob","top10_prob_calibrated",
        "top20_prob","make_cut_prob","model_score","stage2_prob_raw","stage2_prob_calibrated",
        "augusta_competitive_rounds","augusta_experience_tier","augusta_made_cut_prev_year",
        "augusta_scoring_trajectory","tour_vs_augusta_divergence",
        "dk_fair_prob_win","dk_fair_prob_top10","dk_american_odds","kelly_edge_win","kelly_edge_top10",
    ] if c in fdf.columns]
    save=fdf[out_cols]
    save.to_parquet(PROCESSED/"predictions_2026.parquet",index=False)
    save.to_csv(PROCESSED/"predictions_2026.csv",index=False)

    # Spread check
    print(f"\n  Spread: #1 win={fdf['win_prob'].max():.1%} median={fdf['win_prob'].median():.2%} ratio={fdf['win_prob'].max()/fdf['win_prob'].median():.0f}x")
    print(f"  T10 range: {fdf['top10_prob_calibrated'].min():.1%} to {fdf['top10_prob_calibrated'].max():.1%}")

    # Key players comparison
    print(f"\n  KEY PLAYER sg_total_3w (old CSV → extended):")
    old_tour = pd.read_csv("/Users/dylanjaynes/golf_model/golf_model_data/historical_rounds.csv")
    for pn in ["Scottie Scheffler","Jon Rahm","Rory McIlroy","Ludvig Aberg","Davis Riley","Bryson DeChambeau"]:
        old_pt = old_tour[old_tour["player_name"]==pn]
        new_pt = tour_feat[tour_feat["player_name"]==pn]
        old_sg = old_pt.iloc[-1] if len(old_pt)>0 else None
        new_sg = new_pt.iloc[-1] if len(new_pt)>0 else None
        old_val = "N/A" if old_sg is None else f"{old_sg.get('sg_total',0):.3f} ({old_sg.get('season','?')})"
        new_3w = "N/A" if new_sg is None else f"{new_sg.get('sg_total_3w',0):.3f} ({new_sg.get('season','?')})"
        print(f"    {pn:<25} old_last={old_val:<20} new_3w={new_3w}")

    print(f"\n  Saved predictions_2026.parquet/csv ({len(save)} players)")
    print(f"\n{'='*60}")
    print(f"  RETRAIN COMPLETE")
    print(f"{'='*60}")


if __name__=="__main__":
    main()
