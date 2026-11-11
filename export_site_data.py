"""Export the METR-LA augmented graph (sensors, road edges, behavioral edges
with their episodes and joint behaviors) to site/data.js for the web viewer.

    python site/export_site_data.py [path/to/layer.pkl]
"""
import json
import pickle
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # the pickle needs the augmentation package

PKL = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "real" / "metr-la_all_dow_laptop.pkl"
P, STEP = 288, timedelta(minutes=5)
NAMES = ["stable", "lin_inc", "lin_dec", "acc_inc", "dec_inc", "acc_dec", "dec_dec"]

d = np.load(ROOT / "data" / "prepared" / "metr-la.npz")
with open(PKL, "rb") as fh:
    obj = pickle.load(fh)
res = obj["res"]
ids = [str(x) for x in d["ids"]]
assert ids == obj["ids"]
days = [datetime.strptime(s.strip(), "%Y-%m-%d") for s in open(ROOT / "data" / "metr-la_days.txt") if s.strip()]


def iso(t):
    return (days[t // P] + (t % P) * STEP).strftime("%Y-%m-%d %H:%M")


# sensors and road edges, as in hygraph_usecase.road_inputs
with open(ROOT / "data" / "adj_METR-LA.pkl", "rb") as fh:
    all_ids, _, W = pickle.load(fh, encoding="latin1")
all_ids = [str(i) for i in all_ids]
keep = [all_ids.index(s) for s in ids]
W = np.asarray(W)[np.ix_(keep, keep)].astype(float)
np.fill_diagonal(W, 0.0)
loc = pd.read_csv(ROOT / "data" / "graph_sensor_locations.csv")
ll = {str(s): (float(a), float(o)) for s, a, o in zip(loc.sensor_id, loc.latitude, loc.longitude)}
dist = pd.read_csv(ROOT / "data" / "distances_la_2012.csv")
dist = {(str(a), str(b)): float(c) for a, b, c in zip(dist["from"], dist["to"], dist["cost"])}

nodes = [{"id": s, "lat": round(ll[s][0], 6), "lon": round(ll[s][1], 6)} for s in ids]
road = [{"s": ids[i], "t": ids[j], "d": dist.get((ids[i], ids[j]))} for i, j in zip(*np.nonzero(W > 0))]

# behavioral edges: one per episode, grouped by pair for drawing
pidx = res.pair_index()
pairs = defaultdict(list)
glob = {"consistent": Counter(), "latent": Counter()}
for e in res.edges:
    a, b = ids[e.src], ids[e.dst]
    ell = int(res.lag[pidx[(min(e.src, e.dst), max(e.src, e.dst))]])
    joint = Counter()
    for (x, y), n in e.joint_labels:
        lead, follow = (x, y) if ell >= 0 else (y, x)
        joint[(lead, follow)] += n
    glob.setdefault(e.kind, Counter()).update(joint)
    tot = sum(joint.values())
    ml, mf = Counter(), Counter()
    for (x, y), n in joint.items():
        ml[x] += n
        mf[y] += n
    # share of the episode, and lift = observed / expected if lead and follow were independent
    top = [[NAMES[x], NAMES[y], round(n / tot, 3), round(n * tot / (ml[x] * mf[y]), 2)]
           for (x, y), n in joint.most_common(6)]
    pairs[(a, b)].append({"lag": int(e.lag), "w": round(float(e.w), 5), "p": float(e.p),
                          "kind": e.kind, "start": iso(e.t_start), "end": iso(e.t_end),
                          "hours": round((e.t_end - e.t_start + 1) * 5 / 60, 1), "joint": top})

beh = []
for (a, b), eps in pairs.items():
    beh.append({"s": a, "t": b, "kind": eps[0]["kind"], "lag": eps[0]["lag"],
                "w": max(x["w"] for x in eps), "episodes": eps})


def summary(c):
    """Joint behaviors of one class, sorted by lift, the ones with at least 0.5% of the steps."""
    tot = sum(c.values())
    ml, mf = Counter(), Counter()
    for (x, y), n in c.items():
        ml[x] += n
        mf[y] += n
    rows = [[NAMES[x], NAMES[y], round(n / tot, 4), round(n * tot / (ml[x] * mf[y]), 2)]
            for (x, y), n in c.items() if n / tot >= 0.005]
    return sorted(rows, key=lambda r: -r[3])


data = {"labels": NAMES, "nodes": nodes, "road": road, "beh": beh,
        "joint_summary": {k: summary(c) for k, c in glob.items() if c},
        "meta": {"first_day": days[0].strftime("%Y-%m-%d"), "last_day": days[-1].strftime("%Y-%m-%d"),
                 "n_days": len(days), "episodes": len(res.edges), "pairs": len(beh)}}
out = Path(__file__).resolve().parent / "data.js"
out.write_text("const DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
print(f"wrote {out}: {len(nodes)} sensors, {len(road)} road edges, {len(beh)} pairs, {len(res.edges)} episodes")
