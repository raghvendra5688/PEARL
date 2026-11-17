"""Scan the existing artifacts and write grid_catalog.json: every (dataset, embedding model x loss) configuration for which the fine-tuned embeddings,
the ZINC-250k embeddings and the ZINC property table exist. Name = '<ds>@<stem lower-case>', e.g. bace@molformer_finetuned_wl. Used by rafe_lib.CFG and E15."""
import json, pandas as pd
from pathlib import Path
R = Path("/export/qcai-omics/Raghvendra"); E1, E2 = R / "EffiChem_Extras", R / "EffiChem_Extras_v2"
SRC = {  # ds: (smiles-CLM emb csv template, zinc dir, meta, unimol emb template, unimol zinc dir, unimol meta)
 "bace":  (E1/"All_Embeddings/BACE_Embeddings/bace_{sp}_embed.csv", E1/"weighted_loss_flavor/zinc_embeddings/bace", E1/"rag_indices/bace/meta.pkl",
           E1/"weighted_loss_flavor/unimol_embeddings/BACE_Embeddings/bace_{sp}_embed.csv", E1/"weighted_loss_flavor/zinc_embeddings_unimol/bace", E1/"weighted_loss_flavor/rag_indices_unimol/bace/meta.pkl"),
 "bbbp":  (E1/"All_Embeddings/BBBP_Embeddings/bbbp_{sp}_embed.csv", E1/"weighted_loss_flavor/zinc_embeddings/bbbp", E1/"rag_indices/bbbp/meta.pkl",
           E1/"weighted_loss_flavor/unimol_embeddings/BBBP_Embeddings/bbbp_{sp}_embed.csv", E1/"weighted_loss_flavor/zinc_embeddings_unimol/bbbp", E1/"weighted_loss_flavor/rag_indices_unimol/bbbp/meta.pkl"),
 "flavor":(E1/"All_Embeddings/flavor_Embeddings/fart_{sp}_embed.csv", E1/"weighted_loss_flavor/zinc_embeddings/flavor", E1/"rag_indices/flavor/meta.pkl",
           E1/"weighted_loss_flavor/unimol_embeddings/flavor_Embeddings/fart_{sp}_embed.csv", E1/"weighted_loss_flavor/zinc_embeddings_unimol/flavor", E1/"weighted_loss_flavor/rag_indices_unimol/flavor/meta.pkl"),
}
for ds, D in (("dili", "DILI"), ("herg", "HERG"), ("caco2", "CACO2"), ("half_life", "HALF_LIFE")):
    SRC[ds] = (E2/f"finetuned_embeddings/{D}_Embeddings/{ds}_{{sp}}_embed.csv", E2/f"zinc_embeddings/{ds}", E2/f"rag_indices/{ds}/meta.pkl",
               E2/f"unimol_embeddings/{D}_Embeddings/{ds}_{{sp}}_embed.csv", E2/f"zinc_embeddings_unimol/{ds}", E2/f"rag_indices_unimol/{ds}/meta.pkl")
cat, missing = {}, []
for ds, (emb, zdir, meta, uemb, uzdir, umeta) in SRC.items():
    for tpl, zd, mt in ((emb, zdir, meta), (uemb, uzdir, umeta)):
        if not zd.exists(): missing.append((ds, str(zd))); continue
        cols = {c.lower(): c for c in pd.read_csv(str(tpl).format(sp="train"), nrows=1).columns}
        for z in sorted(zd.glob("*.npy")):
            stem = z.stem; col = cols.get(stem.lower() + "_embeddings")
            ok = col and Path(str(tpl).format(sp="train")).exists() and Path(str(tpl).format(sp="test")).exists() and Path(str(tpl).format(sp="eval")).exists() and mt.exists()
            if ok: cat[f"{ds}@{stem.lower()}"] = dict(ds=ds, emb=str(tpl), ecol=col, zinc=str(z), meta=str(mt), stem=stem)
            else: missing.append((ds, stem, "column" if not col else "file"))
json.dump(cat, open(Path(__file__).parent / "grid_catalog.json", "w"), indent=1)
by = {}; [by.setdefault(v["ds"], []).append(k.split("@")[1]) for k, v in cat.items()]
for ds, v in by.items(): print(ds, len(v), v)
print("MISSING:", missing)
