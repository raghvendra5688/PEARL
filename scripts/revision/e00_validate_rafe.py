import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
from rafe_lib import *
name = sys.argv[1]; cfg = CFG[name]
r = build_rafe(name, "clm"); b = get_base(name)
rd = cfg["rag_dir"]; d = REPO / "data" / ("rag_features_unimol/" + rd.split(":")[1] if rd.startswith("unimol") else "rag_features/" + rd)
st = pd.read_csv(d / f"{cfg['stored']}_test_rag.csv").set_index(SMI)
mine = r["test"].copy(); mine.index = b["test"]["smiles"]
common = mine.index.intersection(st.index); print(name, "overlap", len(common), "of", len(mine), len(st))
rows = []
for c in st.columns:
    if c in mine.columns and c != DS[cfg["ds"]]["label"]:
        a, bb = mine.loc[common, c].values, st.loc[common, c].values
        rows.append((c, np.corrcoef(a, bb)[0, 1] if a.std() > 0 and bb.std() > 0 else np.nan, np.abs(a - bb).max()))
R = pd.DataFrame(rows, columns=["col", "corr", "maxabs"]); print(R.describe().to_string()); print(R[R["corr"] < 0.99].to_string())
print("stored cols not in mine:", [c for c in st.columns if c not in mine.columns][:5])
