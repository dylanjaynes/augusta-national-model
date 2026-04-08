#!/usr/bin/env python3
"""
Final V7 pipeline: scrape 2024 PGA, recompute 2026 Augusta features,
retrain, enforce constraints, full backtest with feature importance.
"""
import os, re, json, time, warnings, sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge
import xgboost as xgb
import requests
from rapidfuzz import fuzz, process

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ.update({k.split("=",1)[0]:k.split("=",1)[1]
                   for k in open(".env").read().strip().split("\n") if "=" in k})

PROCESSED = Path("data/processed"); RAW = Path("data/raw")
API_KEY = os.environ["DATAGOLF_API_KEY"]
DG_BASE = "https://feeds.datagolf.com"
HARDCODED_CW = {"sg_ott":0.85,"sg_app":1.45,"sg_arg":1.30,"sg_putt":1.10}
XGB_S1 = {"n_estimators":600,"learning_rate":0.02,"max_depth":4,"subsample":0.8,
           "colsample_bytree":0.8,"min_child_weight":10,"reg_alpha":0.1,"reg_lambda":1.0,
           "objective":"reg:squarederror","random_state":42}
XGB_S2 = {"n_estimators":300,"learning_rate":0.03,"max_depth":3,"subsample":0.8,
           "colsample_bytree":0.7,"min_child_weight":5,"reg_alpha":0.5,"reg_lambda":2.0,
           "objective":"binary:logistic","eval_metric":"auc","random_state":42}
N_SIMS=50000;NOISE=0.10;TPRED=0.14;SEED=42;BLEND=0.90;TEMP=2.5

SG=["sg_ott","sg_app","sg_arg","sg_putt","sg_t2g","sg_total"]
ROLL=["sg_ott_3w","sg_app_3w","sg_arg_3w","sg_putt_3w","sg_t2g_3w","sg_total_3w",
      "sg_ott_8w","sg_app_8w","sg_arg_8w","sg_putt_8w","sg_t2g_8w","sg_total_8w",
      "sg_total_std_8","sg_total_momentum","sg_ball_strike_ratio","log_field_size"]
AUG=["augusta_competitive_rounds","augusta_made_cut_prev_year","augusta_experience_tier",
     "augusta_scoring_trajectory","augusta_rounds_last_2yrs","augusta_best_finish_recent",
     "augusta_starts","augusta_made_cut_rate","augusta_top10_rate","augusta_best_finish",
     "augusta_scoring_avg","augusta_sg_app_career","augusta_sg_total_career",
     "augusta_bogey_avoidance","augusta_round_variance_score","augusta_back9_scoring",
     "tournament_wind_avg","tour_vs_augusta_divergence"]
S2F=ROLL+AUG

def pfn(p):
    if p is None or pd.isna(p): return None
    p=str(p).strip().upper()
    if p in ("CUT","MC","WD","DQ","MDF"): return 999
    p=p.replace("T","").replace("=","")
    try: return int(p)
    except: return None

def norm_name(n):
    n=str(n).strip()
    if "," in n: parts=n.split(",",1); return f"{parts[1].strip()} {parts[0].strip()}"
    return n

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

# FS lookup
from run_field_strength import FieldStrengthLookup, build_roll_fs


# ═══════════════════════════════════════════════════════════
# TASK 2 — SCRAPE 2024 PGA
# ═══════════════════════════════════════════════════════════

def scrape_2024_pga():
    print("\n" + "="*60)
    print("TASK 2 — SCRAPE 2024 PGA EVENTS")
    print("="*60)

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    session.cookies.set("session", os.environ.get("DG_SESSION_COOKIE",""), domain="datagolf.com")

    ext = pd.read_parquet(PROCESSED/"historical_rounds_extended.parquet")
    existing_2024_pga = set(ext[(ext["season"]==2024) & ~ext["event_name"].str.contains("LIV",na=False)]["event_name"].unique())
    print(f"  Existing 2024 PGA events: {existing_2024_pga}")

    pga_tnums = [2,3,4,5,6,7,9,10,11,12,13,14,16,19,20,21,23,26,27,28,30,32,33,34,41,47,54]
    all_rows = []
    events = {}

    for tid in pga_tnums:
        url = f"https://datagolf.com/past-results/pga-tour/{tid}/2024"
        try:
            r = session.get(url, timeout=15)
        except: continue
        if r.status_code != 200 or len(r.text) < 10000: continue

        for jstr in re.findall(r"JSON\.parse\('(.*?)'\)", r.text, re.DOTALL):
            jstr = jstr.replace("\\'","'")
            if len(jstr) < 2000: continue
            d = json.loads(jstr)
            if not isinstance(d,dict) or "lb" not in d: continue
            info = d.get("info",{})
            ename = info.get("event_name","?")
            yr = info.get("calendar_year",2024)
            if yr != 2024: break
            if ename in existing_2024_pga: break

            lb = d.get("lb",[])
            if not lb: break
            num_rounds = info.get("num_rounds",4)
            course = d.get("course","")
            if isinstance(course,dict): course = course.get("course_name","")

            for p in lb:
                round_sgs = [p.get(f"R{rn}_sg") for rn in range(1,num_rounds+1)]
                round_sgs = [s for s in round_sgs if s is not None]
                all_rows.append({
                    "dg_id":p.get("dg_id"),"player_name":norm_name(p.get("player_name","")),
                    "season":yr,"event_name":ename,"course":str(course),
                    "date":info.get("date",""),"field_size":len(lb),
                    "finish_pos":p.get("fin",""),
                    "sg_ott":None,"sg_app":None,"sg_arg":None,"sg_putt":None,"sg_t2g":None,
                    "sg_total":np.mean(round_sgs) if round_sgs else None,
                })
            events[ename] = len(lb)
            print(f"  {ename}: {len(lb)} players")
            break
        time.sleep(0.3)

    if all_rows:
        df = pd.DataFrame(all_rows)
        combined = pd.concat([ext, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["player_name","season","event_name"], keep="last")
        for c in SG:
            combined.loc[combined[c]==-9999, c] = np.nan
        combined.to_parquet(PROCESSED/"historical_rounds_extended.parquet", index=False)
        print(f"\n  Added {len(events)} new 2024 events, {len(df)} rows")
        print(f"  Total dataset: {len(combined):,} rows")
    else:
        print("  No new 2024 events found")
    return


# ═══════════════════════════════════════════════════════════
# RECOMPUTE 2026 AUGUSTA FEATURES + RETRAIN + PREDICT
# ═══════════════════════════════════════════════════════════

def rebuild_and_predict():
    print("\n" + "="*60)
    print("REBUILDING 2026 PREDICTIONS")
    print("="*60)

    ext = pd.read_parquet(PROCESSED/"historical_rounds_extended.parquet")
    ext["finish_num"] = ext["finish_pos"].apply(pfn)
    unified = pd.read_parquet(PROCESSED/"masters_unified.parquet")
    unified["finish_num"] = unified["finish_pos"].apply(pfn)
    features = pd.read_parquet(PROCESSED/"augusta_player_features.parquet")
    rounds_df = pd.read_parquet(PROCESSED/"masters_sg_rounds.parquet")

    cw = json.load(open(PROCESSED/"augusta_sg_weights.json")) if (PROCESSED/"augusta_sg_weights.json").exists() else HARDCODED_CW
    fs_lookup = FieldStrengthLookup(PROCESSED/"dg_field_strength.csv")

    # Build rolling features
    print("  Computing FS-weighted rolling features...")
    tour_clean = ext.dropna(subset=SG)
    tour_clean = tour_clean[tour_clean["finish_num"]<999]
    tour_cw = apply_cw(tour_clean, cw)
    tour_feat = build_roll_fs(tour_cw, fs_lookup)
    tour_feat["finish_pct"] = ((tour_feat["finish_num"]-1)/(tour_feat["field_size"]-1)).clip(0,1)
    print(f"  Rolling features: {len(tour_feat):,} rows")

    # Train Stage 1
    tc = tour_feat.dropna(subset=ROLL+["finish_pct"])
    s1 = xgb.XGBRegressor(**XGB_S1)
    s1.fit(tc[ROLL].values, tc["finish_pct"].values)
    print(f"  Stage 1 trained on {len(tc):,} rows")

    # Get 2026 field
    field_data = requests.get(f"{DG_BASE}/field-updates",
                              params={"tour":"pga","file_format":"json","key":API_KEY}, timeout=15).json()
    field_players = [{"dg_id":p.get("dg_id"),"player_name":norm_name(p.get("player_name","")),"country":p.get("country",""),"dg_rank":p.get("dg_rank")} for p in field_data.get("field",[])]
    print(f"  2026 field: {len(field_players)} players")

    # Build rolling features for field
    roll_rows = []
    for fp in field_players:
        pn = fp["player_name"]
        pt = tour_feat[tour_feat["player_name"]==pn]
        d = {"player_name":pn, "dg_id":fp["dg_id"], "country":fp["country"], "dg_rank":fp["dg_rank"]}
        if len(pt) > 0:
            lat = pt.iloc[-1]
            for c in ROLL: d[c] = lat.get(c)
        else:
            for c in ROLL: d[c] = 0
        roll_rows.append(d)

    fdf = pd.DataFrame(roll_rows)
    for c in ROLL: fdf[c] = fdf[c].fillna(0)
    fdf = fdf[fdf[ROLL[:6]].abs().sum(axis=1) > 0]  # drop players with all-zero SG
    print(f"  Field with SG data: {len(fdf)}")

    # ── Recompute 2026 Augusta features fresh ─��
    print("  Recomputing 2026 Augusta features...")
    aug_rows = []
    for _, fp in fdf.iterrows():
        pn = fp["player_name"]
        prior = unified[(unified["player_name"]==pn)&(unified["season"]<2026)]
        f = {}
        n = len(prior)
        if n == 0:
            f = {c:0 for c in AUG}
            f["augusta_best_finish"] = 91
            f["augusta_best_finish_recent"] = 91
            f["player_name"] = pn
            aug_rows.append(f)
            continue

        def _cr(row):
            cnt = sum(1 for c in ["r1_score","r2_score","r3_score","r4_score"] if pd.notna(row.get(c)))
            if cnt>0: return cnt
            return 4 if row.get("made_cut")==1 else 2

        comp_rounds = sum(_cr(r) for _,r in prior.iterrows())
        prev_yr = prior[prior["season"]==2025]
        cut_prev = int(len(prev_yr)>0 and prev_yr.iloc[0].get("made_cut")==1)
        tier = 0 if comp_rounds==0 else (1 if comp_rounds<=7 else (2 if comp_rounds<=19 else (3 if comp_rounds<=35 else 4)))
        prior_cut = prior[prior["made_cut"]==1]
        if len(prior_cut)>=2:
            rec2 = prior_cut.nlargest(2,"season")
            rec_avg = rec2["score_vs_field"].dropna().mean()
            all_svf = prior_cut["score_vs_field"].dropna()
            if len(all_svf)>0:
                decay = 0.75**(2026-prior_cut.loc[all_svf.index,"season"])
                car_avg = np.average(all_svf, weights=decay)
                traj = car_avg - rec_avg if pd.notna(rec_avg) else 0
            else: traj = 0
        else: traj = 0
        rec_prior = prior[prior["season"].isin([2025,2024])]
        rds_2yr = sum(_cr(r) for _,r in rec_prior.iterrows())
        rec3 = prior.nlargest(3,"season")
        vf = rec3["finish_num"].dropna()
        vf_good = vf[vf<999]
        best_rec = vf_good.min() if len(vf_good)>0 else 91
        cuts = prior["made_cut"].dropna()
        cut_rate = cuts.mean() if len(cuts)>0 else 0
        fn = prior["finish_num"].dropna()
        t10_rate = (fn<=10).mean() if len(fn)>0 else 0
        best_f = fn[fn<999].min() if (fn<999).any() else 91
        svf = prior["score_vs_field"].dropna()
        d2 = 0.75**(2026-prior.loc[svf.index,"season"]) if len(svf)>0 else pd.Series()
        sc_avg = np.average(svf, weights=d2) if len(svf)>0 else 0
        psg = prior[prior["has_sg_data"]==True]
        if len(psg)>0:
            sd = 0.75**(2026-psg["season"])
            sa = psg["sg_app"].dropna()
            sg_app_c = np.average(sa, weights=sd[sa.index]) if len(sa)>0 else None
            st = psg["sg_total"].dropna()
            sg_tot_c = np.average(st, weights=sd[st.index]) if len(st)>0 else None
        else: sg_app_c=None; sg_tot_c=None

        pr_rds = rounds_df[(rounds_df["player_name"]==pn)&(rounds_df["season"]<2026)]
        if len(pr_rds)>0:
            scores = pr_rds["score"].dropna()
            ba = (scores<=0).sum()/len(scores) if len(scores)>0 else 0
            rv = scores.std() if len(scores)>1 else None
            b9 = pr_rds[pr_rds["round_num"].isin([3,4])]["score"].dropna().mean() if len(pr_rds[pr_rds["round_num"].isin([3,4])])>0 else None
        else: ba=0;rv=None;b9=None

        f = {
            "player_name":pn,
            "augusta_competitive_rounds":comp_rounds,"augusta_made_cut_prev_year":cut_prev,
            "augusta_experience_tier":tier,"augusta_scoring_trajectory":traj,
            "augusta_rounds_last_2yrs":rds_2yr,"augusta_best_finish_recent":best_rec,
            "augusta_starts":n,"augusta_made_cut_rate":cut_rate,"augusta_top10_rate":t10_rate,
            "augusta_best_finish":best_f,"augusta_scoring_avg":sc_avg,
            "augusta_sg_app_career":sg_app_c,"augusta_sg_total_career":sg_tot_c,
            "augusta_bogey_avoidance":ba,"augusta_round_variance_score":rv,
            "augusta_back9_scoring":b9,
            "tournament_wind_avg":10.0,"tour_vs_augusta_divergence":0.0,
        }
        aug_rows.append(f)

    aug_df = pd.DataFrame(aug_rows)
    fdf = fdf.merge(aug_df, on="player_name", how="left")
    for c in AUG:
        if c not in fdf.columns: fdf[c] = 0.0

    # Divergence
    sg8 = fdf["sg_total_8w"].rank(pct=True)
    hr = fdf["augusta_competitive_rounds"]>=6
    sa = fdf.loc[hr,"augusta_scoring_avg"]
    if len(sa)>2:
        sap = (-sa).rank(pct=True)
        fdf.loc[hr,"tour_vs_augusta_divergence"] = sg8[hr]-sap
    fdf["tour_vs_augusta_divergence"] = fdf["tour_vs_augusta_divergence"].fillna(0)

    # Stage 1 predict + MC
    preds = s1.predict(fdf[ROLL].values)
    mc = run_mc(preds)
    fdf["win_prob"]=mc["win"];fdf["top5_prob"]=mc["top5"]
    fdf["mc_top10"]=mc["top10"];fdf["top20_prob"]=mc["top20"]
    fdf["model_score"]=preds

    # Stage 2 (Augusta-only)
    s2t = unified.copy()
    s2t["made_top10"] = (s2t["finish_num"]<=10).astype(int)
    s2t = s2t.merge(features, on=["player_name","season"], how="left")
    s2r = []
    for _,row in s2t.iterrows():
        pn,ps = row["player_name"],row["season"]
        pt = tour_feat[(tour_feat["player_name"]==pn)&(tour_feat["season"]<=ps)]
        f2 = {"idx":row.name}
        if len(pt)>0:
            lat = pt.iloc[-1]
            for c in ROLL: f2[c] = lat.get(c)
        else:
            for c in ROLL: f2[c] = np.nan
        f2["tournament_wind_avg"]=10.0;f2["tour_vs_augusta_divergence"]=0.0
        s2r.append(f2)
    s2rf = pd.DataFrame(s2r).set_index("idx")
    for c in S2F:
        if c not in s2t.columns and c in s2rf.columns: s2t[c]=s2rf[c]
        elif c not in s2t.columns: s2t[c]=np.nan
    s2t = s2t[s2t["finish_num"].notna()]
    np_ = s2t["made_top10"].sum()
    spw = (len(s2t)-np_)/np_ if np_>0 else 8
    s2m = xgb.XGBClassifier(**{**XGB_S2,"scale_pos_weight":spw})
    s2m.fit(s2t[S2F], s2t["made_top10"])

    for c in S2F:
        if c not in fdf.columns: fdf[c]=0.0
    s2raw = s2m.predict_proba(fdf[S2F])[:,1]
    logits = np.log(np.clip(s2raw,1e-6,1-1e-6)/(1-np.clip(s2raw,1e-6,1-1e-6)))
    s2cal = 1/(1+np.exp(-logits/TEMP))
    fdf["stage2_prob_raw"]=s2raw; fdf["stage2_prob_calibrated"]=s2cal
    fdf["top10_prob_calibrated"] = BLEND*s2cal + (1-BLEND)*fdf["mc_top10"]

    # ── TASK 3: Enforce monotonic constraints ──
    print("\n  Enforcing monotonic constraints...")
    violations = 0
    fdf["make_cut_prob"] = fdf[["top20_prob","mc_top10"]].max(axis=1).clip(upper=0.95)
    # top10 >= top20 >= top10 (already handled by taking max)
    fdf["top20_prob"] = fdf[["top20_prob","top10_prob_calibrated"]].max(axis=1)
    # top5 <= top10
    fdf["top5_prob"] = np.minimum(fdf["top5_prob"].clip(lower=fdf["win_prob"]).values, fdf["top10_prob_calibrated"].values)
    # top10 >= 5 * win (minimum consistency)
    min_t10 = fdf["win_prob"] * 5
    mask = fdf["top10_prob_calibrated"] < min_t10
    if mask.any():
        violations += mask.sum()
        fdf.loc[mask, "top10_prob_calibrated"] = min_t10[mask]
        fdf.loc[mask, "top20_prob"] = fdf.loc[mask, ["top20_prob","top10_prob_calibrated"]].max(axis=1)
    print(f"  Monotonic violations fixed: {violations}")

    # Join DK odds
    dk_path = PROCESSED/"dk_odds_2026.csv"
    if dk_path.exists():
        dk = pd.read_csv(dk_path)
        for _,orow in dk.iterrows():
            best = process.extractOne(orow["player_name"], fdf["player_name"].tolist(), scorer=fuzz.ratio)
            if best and best[1]>=85:
                idx = fdf[fdf["player_name"]==best[0]].index
                if len(idx)>0:
                    fdf.loc[idx[0],"dk_fair_prob_win"]=orow["dk_fair_prob"]
                    fdf.loc[idx[0],"dk_fair_prob_top10"]=orow["dk_fair_top10"]
                    fdf.loc[idx[0],"dk_american_odds"]=orow["dk_american_odds"]
        fdf["kelly_edge_win"]=(fdf["win_prob"]-fdf.get("dk_fair_prob_win",0))/fdf.get("dk_fair_prob_win",1).clip(lower=1e-6)
        fdf["kelly_edge_top10"]=(fdf["top10_prob_calibrated"]-fdf.get("dk_fair_prob_top10",0))/fdf.get("dk_fair_prob_top10",1).clip(lower=1e-6)

    fdf = fdf.sort_values("top10_prob_calibrated", ascending=False).reset_index(drop=True)

    # Cameron Young check
    cy = fdf[fdf["player_name"]=="Cameron Young"]
    if len(cy)>0:
        c = cy.iloc[0]
        print(f"\n  Cameron Young fix check:")
        print(f"    made_cut_prev_year: {c['augusta_made_cut_prev_year']}")
        print(f"    win={c['win_prob']:.2%} t10={c['top10_prob_calibrated']:.1%} t20={c['top20_prob']:.1%}")
        print(f"    s2_raw={c['stage2_prob_raw']:.3f} s2_cal={c['stage2_prob_calibrated']:.3f}")

    # ── Backtest with feature importance ──
    print(f"\n{'='*60}")
    print("WALK-FORWARD BACKTEST (V7 FINAL)")
    print(f"{'='*60}")

    sg_years = sorted(unified[unified["has_sg_data"]==True]["season"].unique())
    bt_years = [y for y in sg_years if y > min(unified["season"].unique())]
    metrics = []
    bt_results = []

    for year in bt_years:
        train = tour_feat[tour_feat["season"]<year].dropna(subset=ROLL+["finish_pct"])
        s1_bt = xgb.XGBRegressor(**XGB_S1)
        s1_bt.fit(train[ROLL].values, train["finish_pct"].values)

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
            fr=features[(features["player_name"]==pn)&(features["season"]==year)]
            if len(fr)>0:
                fr=fr.iloc[0]
                for c in AUG:
                    if c not in ("tournament_wind_avg","tour_vs_augusta_divergence"): d[c]=fr.get(c)
            else:
                for c in AUG:
                    if c not in ("tournament_wind_avg","tour_vs_augusta_divergence"): d[c]=0.0
            d["tournament_wind_avg"]=10.0;d["tour_vs_augusta_divergence"]=0.0
            d["actual_finish_num"]=pr.get("finish_num");d["actual_finish_pos"]=pr.get("finish_pos")
            ff.append(d)

        bdf=pd.DataFrame(ff).dropna(subset=ROLL)
        if len(bdf)<5: continue
        sg8b=bdf["sg_total_8w"].rank(pct=True)
        hrb=bdf["augusta_competitive_rounds"]>=6
        sab=bdf.loc[hrb,"augusta_scoring_avg"]
        if len(sab)>2:
            sapb=(-sab).rank(pct=True)
            bdf.loc[hrb,"tour_vs_augusta_divergence"]=sg8b[hrb]-sapb
        bdf["tour_vs_augusta_divergence"]=bdf["tour_vs_augusta_divergence"].fillna(0)

        bp=s1_bt.predict(bdf[ROLL].values)
        bmc=run_mc(bp);bdf["mc_top10"]=bmc["top10"]

        # S2
        s2t2=unified[unified["season"]<year].copy()
        s2t2["made_top10"]=(s2t2["finish_num"]<=10).astype(int)
        s2t2=s2t2.merge(features,on=["player_name","season"],how="left")
        s2r2=[]
        for _,row in s2t2.iterrows():
            pn2,ps2=row["player_name"],row["season"]
            pt2=tour_feat[(tour_feat["player_name"]==pn2)&(tour_feat["season"]<=ps2)]
            f22={"idx":row.name}
            if len(pt2)>0:
                lat2=pt2.iloc[-1]
                for c in ROLL: f22[c]=lat2.get(c)
            else:
                for c in ROLL: f22[c]=np.nan
            f22["tournament_wind_avg"]=10.0;f22["tour_vs_augusta_divergence"]=0.0
            s2r2.append(f22)
        s2rf2=pd.DataFrame(s2r2).set_index("idx")
        for c in S2F:
            if c not in s2t2.columns and c in s2rf2.columns: s2t2[c]=s2rf2[c]
            elif c not in s2t2.columns: s2t2[c]=np.nan
        s2t2=s2t2[s2t2["finish_num"].notna()]
        np2=s2t2["made_top10"].sum()
        spw2=(len(s2t2)-np2)/np2 if np2>0 else 8
        s2bt=xgb.XGBClassifier(**{**XGB_S2,"scale_pos_weight":spw2})
        s2bt.fit(s2t2[S2F],s2t2["made_top10"])
        for c in S2F:
            if c not in bdf.columns: bdf[c]=0.0
        s2p=s2bt.predict_proba(bdf[S2F])[:,1]
        bdf["s2_top10"]=s2p
        bdf["blend_top10"]=BLEND*s2p+(1-BLEND)*bdf["mc_top10"]
        bdf["season"]=year

        v=bdf[bdf["actual_finish_num"]<999].copy()
        at10=(v["actual_finish_num"]<=10).astype(int)
        m={"year":year}
        if at10.sum()>0 and at10.sum()<len(at10):
            m["blend_auc"]=roc_auc_score(at10,v["blend_top10"])
            m["mc_auc"]=roc_auc_score(at10,v["mc_top10"])
            m["s2_auc"]=roc_auc_score(at10,v["s2_top10"])
        t10p=v.nlargest(10,"blend_top10")
        m["top10_prec"]=(t10p["actual_finish_num"]<=10).sum()/10
        sp,_=stats.spearmanr(v["mc_top10"],-v["actual_finish_num"])
        m["spearman"]=sp
        metrics.append(m)
        bt_results.append(bdf)
        print(f"  {year}: field={len(bdf)} | Blend AUC={m.get('blend_auc',0):.3f} | T10 Prec={m['top10_prec']:.0%} | Spearman={sp:.3f}")

    # Feature importance from final S2 model
    print(f"\n  S2 Feature Importance (top 15):")
    imp = s2m.get_booster().get_score(importance_type="gain")
    feat_imp = {}
    for fname, score in imp.items():
        if fname.startswith("f") and fname[1:].isdigit():
            idx = int(fname[1:])
            if idx < len(S2F): feat_imp[S2F[idx]] = score
        else:
            feat_imp[fname] = score
    for i, (f, v) in enumerate(sorted(feat_imp.items(), key=lambda x:-x[1])[:15]):
        tag = " *AUG*" if f in AUG else ""
        print(f"    {i+1:2d}. {f:<35} {v:>10.1f}{tag}")

    # Summary
    print(f"\n  {'Year':<6} {'Blend AUC':>10} {'S2 AUC':>8} {'MC AUC':>8} {'T10 Prec':>9} {'Spearman':>9}")
    for m in metrics:
        print(f"  {m['year']:<6} {m.get('blend_auc',0):>10.3f} {m.get('s2_auc',0):>8.3f} {m.get('mc_auc',0):>8.3f} {m.get('top10_prec',0):>8.0%} {m.get('spearman',0):>9.3f}")
    def avg(k): vs=[m[k] for m in metrics if m.get(k)]; return np.mean(vs) if vs else 0
    print(f"  {'AVG':<6} {avg('blend_auc'):>10.3f} {avg('s2_auc'):>8.3f} {avg('mc_auc'):>8.3f} {avg('top10_prec'):>8.0%} {avg('spearman'):>9.3f}")

    if bt_results:
        pd.concat(bt_results,ignore_index=True).to_parquet(PROCESSED/"backtest_results_v7.parquet",index=False)

    # Save predictions
    out_cols = [c for c in [
        "dg_id","player_name","country","dg_rank","win_prob","top5_prob","top10_prob_calibrated",
        "top20_prob","make_cut_prob","model_score","stage2_prob_raw","stage2_prob_calibrated",
        "augusta_competitive_rounds","augusta_experience_tier","augusta_made_cut_prev_year",
        "augusta_scoring_trajectory","tour_vs_augusta_divergence",
        "dk_fair_prob_win","dk_fair_prob_top10","dk_american_odds","kelly_edge_win","kelly_edge_top10",
    ] if c in fdf.columns]
    save = fdf[out_cols]
    save.to_parquet(PROCESSED/"predictions_2026.parquet",index=False)
    save.to_csv(PROCESSED/"predictions_2026.csv",index=False)

    # Print top 20
    print(f"\n  {'='*95}")
    print(f"  FINAL 2026 TOP 20")
    print(f"  {'='*95}")
    print(f"  {'Rk':<3} {'Player':<25} {'Win%':>6} {'T10%':>6} {'T20%':>6} {'DK':>8} {'Tier':>4} {'CutPv':>5} {'Traj':>6}")
    for i,(_,r) in enumerate(fdf.head(20).iterrows()):
        dk=f"+{r['dk_american_odds']:.0f}" if pd.notna(r.get('dk_american_odds')) else "N/A"
        print(f"  {i+1:<3} {r['player_name']:<25} {r['win_prob']:>5.1%} {r['top10_prob_calibrated']:>5.1%} {r['top20_prob']:>5.1%} {dk:>8} {r.get('augusta_experience_tier',0):>4.0f} {r.get('augusta_made_cut_prev_year',0):>5.0f} {r.get('augusta_scoring_trajectory',0):>+6.2f}")

    # Value plays + fades
    if "dk_fair_prob_top10" in fdf.columns:
        print(f"\n  TOP 5 VALUE (model > market for top-10):")
        val = fdf[fdf["kelly_edge_top10"]>0].nlargest(5,"kelly_edge_top10")
        for _,r in val.iterrows():
            print(f"    {r['player_name']:<25} model={r['top10_prob_calibrated']:.1%} mkt={r['dk_fair_prob_top10']:.1%} edge={r['kelly_edge_top10']:+.0%}")

        print(f"\n  TOP 3 FADES (model < market):")
        fad = fdf[fdf["kelly_edge_top10"]<0].nsmallest(3,"kelly_edge_top10")
        for _,r in fad.iterrows():
            reasons = []
            if r.get("tour_vs_augusta_divergence",0)>0.3: reasons.append("tour→Augusta divergence")
            if r.get("augusta_scoring_trajectory",0)<-1: reasons.append("declining trajectory")
            if r.get("augusta_made_cut_prev_year",0)==0 and r.get("augusta_starts",0)>0: reasons.append("missed cut 2025")
            if not reasons: reasons.append("market overvalues")
            print(f"    {r['player_name']:<25} model={r['top10_prob_calibrated']:.1%} mkt={r['dk_fair_prob_top10']:.1%} edge={r['kelly_edge_top10']:+.0%} — {'; '.join(reasons)}")

    print(f"\n  Spread: #1 win={fdf['win_prob'].max():.1%} median={fdf['win_prob'].median():.2%} ratio={fdf['win_prob'].max()/max(fdf['win_prob'].median(),1e-6):.0f}x")

    return fdf, metrics


def main():
    print("="*60)
    print("  FINAL V7 — SCRAPE 2024, FIX FEATURES, RETRAIN, DEPLOY")
    print("="*60)

    scrape_2024_pga()
    fdf, metrics = rebuild_and_predict()

    print(f"\n{'='*60}")
    print(f"  V7 COMPLETE — READY TO DEPLOY")
    print(f"{'='*60}")
    print(f"\n  Local: python3 -m streamlit run streamlit_app/app.py")
    print(f"  Cloud: share.streamlit.io → dylanjaynes/augusta-national-model → streamlit_app/app.py")


if __name__=="__main__":
    main()
