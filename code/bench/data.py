"""Unified loaders for the six-dataset few-shot nanomedicine benchmark."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fp_utils import fingerprints  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

MAIN_METHODS = {
    "luminescence_normalized",
    "luminescence_discretized_normalized",
    "protein_adundance_normalized",
    "luminescence_relative_to_Spikevax",
}
NUM_DRUG = ["MW", "LogP", "TPSA", "HBD", "HBA", "RotB", "HeavyAtomCount"]
NUM_FORM = ["Albumin_Conc_mg_mL", "Drug_to_Albumin_ratio_w_w",
            "Stabilizer_Conc_mg_mL", "Crosslinker_Conc_mM", "pH_Synthesis",
            "Total_Solid_Conc_mg_mL", "AntiSolvent_to_Solvent_Ratio"]
NUM_PHYS = ["Size_nm", "PDI", "Zeta_mV", "EE_percent", "DL_percent"]
CAT = ["Albumin_Type", "Synthesis_Method", "Stabilizer_Name",
       "Crosslinker", "Surface_Modification"]

D6_CAT = ["liposome_type", "carrier_type_group", "targeting_ligand", "phospholipid",
          "helper_lipid", "cholesterol", "other_lipid_components", "peg_lipid_type",
          "cationic_lipid", "drug_name_std", "drug_class", "preparation_method_group",
          "organic_solvent", "hydration_medium", "sonication_type", "extrusion",
          "size_reduction_method", "in_vitro_or_in_vivo", "cell_line_std", "animal_model"]
D6_NUM = ["drug_to_lipid_ratio", "lipid_ratio", "hydration_pH", "lipid_concentration"]


def _num(x):
    if pd.isna(x):
        return np.nan
    m = re.search(r"-?[0-9]+\.?[0-9]*", str(x))
    return float(m.group()) if m else np.nan


def _matrix(df, num_cols, cat_cols, max_cat=10):
    num = df[num_cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
    if cat_cols:
        cats_df = pd.DataFrame(index=df.index)
        for c in cat_cols:
            s = df[c].astype(str)
            top = s.value_counts().head(max_cat).index
            cats_df[c] = s.where(s.isin(top), "OTHER")
        cats = pd.get_dummies(cats_df, dummy_na=True)
        X = np.hstack([num, cats.values.astype(np.float32)])
        return X, list(num_cols) + list(cats.columns)
    return num, list(num_cols)


def _pack(name, task, df, X, y, groups, features, target):
    return {"dataset": name, "task": task, "df": df, "X": X, "y": y,
            "groups": np.asarray(groups), "features": features, "target": target}


# ---------------------------------------------------------------- D1 albumin
def load_d1(group="all", max_cat=10):
    base = RAW / "albumin"
    f = pd.read_excel(base / "Alb_dataset_formulation.xlsx").drop_duplicates("Formulation_ID")
    p = pd.read_excel(base / "Alb_dataset_physicochemical.xlsx").drop_duplicates("Formulation_ID")
    d = pd.read_excel(base / "Alb_dataset_drug.xlsx").drop_duplicates("Drug_ID")
    b = pd.read_excel(base / "Alb_dataset_biological.xlsx")
    b["ic50"] = b["IC50_ug_per_mL"].map(_num)
    b = b.dropna(subset=["ic50"])
    b = b[b["ic50"] > 0]
    b["y"] = np.log10(b["ic50"])
    b = b.groupby(["Formulation_ID", "Cell_Line"], as_index=False)["y"].mean()
    f = f.merge(d[NUM_DRUG + ["Drug_ID"]], on="Drug_ID", how="left")
    f = f.merge(p[NUM_PHYS + ["Formulation_ID"]], on="Formulation_ID", how="left")
    df = b.merge(f, on="Formulation_ID", how="left")
    counts = df.groupby("Cell_Line").size()
    keep = counts[counts >= 20].index
    df = df[df["Cell_Line"].isin(keep)].reset_index(drop=True)
    if group == "drug":
        num, cat = NUM_DRUG, []
    elif group == "form":
        num, cat = NUM_FORM, CAT
    elif group == "phys":
        num, cat = NUM_PHYS, []
    elif group == "drug+form":
        num, cat = NUM_DRUG + NUM_FORM, CAT
    else:
        num, cat = NUM_DRUG + NUM_FORM + NUM_PHYS, CAT
    X, feats = _matrix(df, num, cat, max_cat)
    return _pack("D1", "T1_cellline_ic50", df, X, df["y"].values,
                 df["Cell_Line"].values, feats, "log10(IC50)")


# ---------------------------------------------------------------- D2 LNPDB
def load_d2(n_bits=256):
    d = pd.read_csv(RAW / "LNPDB.csv", encoding="latin-1", low_memory=False)
    d["y"] = pd.to_numeric(d["Experiment_value"], errors="coerce")
    d = d[d["y"].notna() & d["IL_SMILES"].notna()]
    d = d[d["Experiment_method"].isin(MAIN_METHODS)]
    d = (d.groupby(["Experiment_ID", "IL_SMILES", "Model", "Cargo_type"], as_index=False)
           .agg(y=("y", "mean")))
    uniq = d.drop_duplicates("IL_SMILES")
    Xfp, Xd, keep = fingerprints(uniq["IL_SMILES"].tolist(), n_bits=n_bits)
    idx = {s: i for i, s in enumerate(uniq["IL_SMILES"].values[keep])}
    d = d[d["IL_SMILES"].isin(idx)].reset_index(drop=True)
    X = np.hstack([Xfp[[idx[s] for s in d["IL_SMILES"]]],
                   Xd[[idx[s] for s in d["IL_SMILES"]]]])
    y = d["y"].values.astype(float)
    g = d["Experiment_ID"].values
    for u in np.unique(g):
        m = g == u
        sd = y[m].std()
        y[m] = (y[m] - y[m].mean()) / (sd if sd > 1e-6 else 1.0)
    return _pack("D2", "T2_study_delivery", d, X, y, g, [f"fp{i}" for i in range(X.shape[1])],
                 "study z-scored delivery")


# ---------------------------------------------------------------- D3 LNP_ML
def load_d3(n_bits=256):
    d = pd.read_csv(RAW / "LNP_ML_all_data.csv", low_memory=False)
    d = d[["smiles", "unnormalized_delivery", "Experiment_ID",
           "Model_type", "Cargo_type"]].dropna()
    d["y"] = pd.to_numeric(d["unnormalized_delivery"], errors="coerce")
    d = d[d["y"].notna() & d["smiles"].notna()].copy()
    d["unit"] = (d["Experiment_ID"].astype(str) + "|" +
                 d["Model_type"].astype(str) + "|" + d["Cargo_type"].astype(str))
    d = d.groupby(["unit", "smiles"], as_index=False)["y"].mean()
    uniq = d.drop_duplicates("smiles")
    Xfp, Xd, keep = fingerprints(uniq["smiles"].tolist(), n_bits=n_bits)
    idx = {s: i for i, s in enumerate(uniq["smiles"].values[keep])}
    d = d[d["smiles"].isin(idx)].reset_index(drop=True)
    X = np.hstack([Xfp[[idx[s] for s in d["smiles"]]],
                   Xd[[idx[s] for s in d["smiles"]]]])
    y = d["y"].values.astype(float)
    g = d["unit"].values
    for u in np.unique(g):
        m = g == u
        sd = y[m].std()
        y[m] = (y[m] - y[m].mean()) / (sd if sd > 1e-6 else 1.0)
    return _pack("D3", "T3_experiment_delivery", d, X, y,
                 g, [f"fp{i}" for i in range(X.shape[1])], "unit z-scored delivery")


# ---------------------------------------------------------------- D4 Nano-Tumor
def load_d4():
    d = pd.read_excel(RAW / "nano_tumor.xlsx", sheet_name="Tumor & Organ data")
    d = d.dropna(subset=["NM.ID"]).copy()
    d["size"] = d["Size"].map(_num)
    d["pdi"] = d["PDI"].map(_num)
    d["zeta"] = d["Zeta Potential"].map(_num)
    d["dose"] = d["NP Administration (mg/kg)"].map(_num)
    d["tumor_size"] = d["Tumor Size"].map(_num)
    d["tumor_pct"] = pd.to_numeric(d["Biodistribution (%ID)"], errors="coerce")
    # one row per (Reference, NM.ID): max tumor %ID over time
    agg = (d.groupby(["Reference", "NM.ID"], as_index=False)
             .agg(y=("tumor_pct", "max"), size=("size", "first"), pdi=("pdi", "first"),
                  zeta=("zeta", "first"), dose=("dose", "first"),
                  tumor_size=("tumor_size", "first"),
                  particle_type=("Particle Type", "first"),
                  core=("Core material", "first"),
                  targeting=("Targeting Strategy", "first"),
                  shape=("Shape", "first"), tumor_cell=("Tumor Cell", "first"),
                  animal=("Animal", "first")))
    agg = agg[agg["y"] > 0].copy()
    agg["y"] = np.log10(agg["y"])
    X, feats = _matrix(agg, ["size", "pdi", "zeta", "dose", "tumor_size"],
                       ["particle_type", "core", "targeting", "shape",
                        "tumor_cell", "animal"])
    return _pack("D4", "T6_study_tumor_delivery", agg, X, agg["y"].values,
                 agg["Reference"].values, feats, "log10(max tumor %ID)")


# ---------------------------------------------------------------- D5 PLGA
def load_d5(target="particle_size"):
    base = RAW / "plga"
    d = pd.read_csv(base / "NP_dataset.csv")
    ref = pd.read_csv(base / "NP_dataset_formulations.csv")
    if len(ref) == len(d):
        d["reference"] = ref["reference"].values
    else:
        d["reference"] = "global"
    targets = ["particle_size", "EE", "LC"]
    feats = [c for c in d.columns if c not in targets + ["reference"]]
    d = d.dropna(subset=[target]).copy()
    X, names = _matrix(d, feats, [])
    return _pack("D5", f"T4_plga_{target}", d, X, d[target].values.astype(float),
                 d["reference"].values, names, target)


# ---------------------------------------------------------------- D6 LipoDB
def load_d6(target="particle_size_nm", max_cat=10):
    d = pd.read_csv(RAW / "lipodb" / "LipoDB_v2.0_dataset.csv", low_memory=False)
    d[target] = pd.to_numeric(d[target], errors="coerce")
    d = d.dropna(subset=[target]).copy()
    for c in D6_NUM:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    num = [c for c in D6_NUM if c in d.columns]
    cat = [c for c in D6_CAT if c in d.columns]
    X, names = _matrix(d, num, cat, max_cat)
    return _pack("D6", f"T5_lipodb_{target}", d, X, d[target].values.astype(float),
                 d["paper_id"].values, names, target)


# ---------------------------------------------------------------- D7 LNP Atlas
def load_d7(target="particle_size_nm_std", include_ratios=True):
    d = pd.read_csv(RAW / "lnp_atlas.csv", encoding="latin-1", low_memory=False)
    d[target] = pd.to_numeric(d[target], errors="coerce")
    d = d.dropna(subset=[target, "ionizable_lipid_smiles"]).copy()
    # lipid molar ratio "60:20:19.5:0.5"
    ratios = d["lipid_molar_ratio"].astype(str).str.split(":", expand=True)
    for i in range(4):
        d[f"ratio{i}"] = pd.to_numeric(ratios.get(i), errors="coerce")
    uniq = d.drop_duplicates("ionizable_lipid_smiles")
    Xfp, Xd, keep = fingerprints(uniq["ionizable_lipid_smiles"].tolist(), n_bits=256)
    idx = {s: i for i, s in enumerate(uniq["ionizable_lipid_smiles"].values[keep])}
    d = d[d["ionizable_lipid_smiles"].isin(idx)].reset_index(drop=True)
    fp = Xfp[[idx[s] for s in d["ionizable_lipid_smiles"]]]
    desc = Xd[[idx[s] for s in d["ionizable_lipid_smiles"]]]
    X = np.hstack([fp, desc])
    feats = ([f"fp{i}" for i in range(fp.shape[1])] +
             [f"desc{i}" for i in range(desc.shape[1])])
    if include_ratios:
        X = np.hstack([X, d[[f"ratio{i}" for i in range(4)]].values.astype(np.float32)])
        feats = feats + [f"ratio{i}" for i in range(4)]
    return _pack("D7", f"T7_lnpatlas_{target}", d, X, d[target].values.astype(float),
                 d["paper_doi"].values, feats, target)


# ---------------------------------------------------------------- D8 synthetic brain-cancer
D8_NUM = ["mean_particle_size_nm", "size_sd_nm", "polydispersity_index",
          "zeta_potential_mV", "encapsulation_efficiency_percent",
          "drug_loading_percent_w_w", "stability_index_0_100",
          "shelf_condition_proxy", "biodegradation_proxy_0_100",
          "serum_stability_proxy_0_100"]
D8_CAT = ["nanocarrier_class", "formulation_subtype", "payload_type",
          "therapeutic_category", "targeting_strategy", "surface_functionalization",
          "preparation_method", "solvent_system", "surfactant_or_lipid_components",
          "polymer_or_lipid_ratio_descriptor", "morphology_class", "release_profile_class"]


def load_d8(target="BBB_transport_score_0_100"):
    d = pd.read_csv(RAW / "d8_synthetic" / "formulations_master.csv", low_memory=False)
    d[target] = pd.to_numeric(d[target], errors="coerce")
    d = d.dropna(subset=[target]).copy()
    for c in D8_NUM:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    num = [c for c in D8_NUM if c in d.columns]
    cat = [c for c in D8_CAT if c in d.columns]
    X, names = _matrix(d, num, cat)
    return _pack("D8", f"T8_synthetic_{target}", d, X, d[target].values.astype(float),
                 d["formulation_subtype"].values, names, target)


# ---------------------------------------------------------------- D9 MHT SAR
D9_NUM = ["f_avg_core_width_nm", "f_Ms_emu_g_MNP", "f_Hc_Oe", "f_H_Oe", "f_FCY_kHz",
          "f_conc_MNP_mg_mL", "f_avg_core_sd_nm _BOTH", "f_area_core_estimated_nm2",
          "f_volume_core_estimated_nm3", "f_area_volume_ratio", "f_dopant_ratio",
          "Ca_comp", "Co_comp", "Cu_comp", "Fe_comp", "Ga_comp", "Gd_comp", "Mg_comp",
          "Mn_comp", "Nd_comp", "Ni_comp", "Sn_comp", "Tb_comp", "Zn_comp"]
D9_CAT = ["c_core_composition", "is_doped", "c_dopant_element", "f_is_coated",
          "f_is_organic", "f_is_inorganic", "f_is_composite_coating", "c_coating",
          "c_shape_type", "c_magnetic_phase"]


def load_d9():
    d = pd.read_excel(RAW / "mht" / "mht.xlsx")
    d["SAR"] = pd.to_numeric(d["f_SAR_W_g"], errors="coerce")
    d = d[d["SAR"] > 0].copy()
    d["y"] = np.log10(d["SAR"])
    for c in D9_NUM:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    num = [c for c in D9_NUM if c in d.columns]
    cat = [c for c in D9_CAT if c in d.columns]
    X, names = _matrix(d, num, cat)
    return _pack("D9", "T9_mht_sar", d, X, d["y"].values, d["DOI"].values, names,
                 "log10(SAR)")


# ---------------------------------------------------------------- D10 LNPCD cytotoxicity
D10_NUM = ["Ionizable_Lipid_Mol_Ratio", "Helper_Lipid_Mol_Ratio",
           "Cholesterol_Mol_Ratio", "PEG_Lipid_Mol_Ratio",
           "Ionizable_Lipid_to_mRNA_weight_ratio", "Lipid/Cells", "NA/Cells",
           "Lipid_concentration", "NA_concentration", "MolWt", "Num_tails",
           "Num_carbon_in_tail", "num_unsaturated_cc_bonds",
           "num_protonatable_nitrogens", "Topological_Polar_Surface_Area",
           "Hydrogen_Bond_Donors", "LogP", "Fraction_sp3_Carbons",
           "Nitrogen_Count", "Rings", "Aromatic_Rings"]
D10_CAT = ["Helper_lipid_ID", "Cargo_type", "Model_type", "has_ester"]


def load_d10(target="viability"):
    d = pd.read_csv(RAW / "lnpcd" / "lnpcd.csv", low_memory=False)
    d[target] = pd.to_numeric(d[target], errors="coerce")
    d = d.dropna(subset=[target, "smiles"]).copy()
    uniq = d.drop_duplicates("smiles")
    Xfp, Xd, keep = fingerprints(uniq["smiles"].tolist(), n_bits=256)
    idx = {s: i for i, s in enumerate(uniq["smiles"].values[keep])}
    d = d[d["smiles"].isin(idx)].reset_index(drop=True)
    fp = Xfp[[idx[s] for s in d["smiles"]]]
    desc = Xd[[idx[s] for s in d["smiles"]]]
    num = [c for c in D10_NUM if c in d.columns]
    cat = [c for c in D10_CAT if c in d.columns]
    Xnum, names = _matrix(d, num, cat)
    X = np.hstack([fp, desc, Xnum])
    feats = ([f"fp{i}" for i in range(fp.shape[1])] +
             [f"desc{i}" for i in range(desc.shape[1])] + names)
    return _pack("D10", f"T10_lnpcd_{target}", d, X,
                 d[target].values.astype(float), d["Experiment_ID"].values,
                 feats, target)


LOADERS = {"D1": load_d1, "D2": load_d2, "D3": load_d3,
           "D4": load_d4, "D5": load_d5, "D6": load_d6, "D7": load_d7,
           "D8": load_d8, "D9": load_d9, "D10": load_d10}


def load(name, **kwargs):
    return LOADERS[name](**kwargs)