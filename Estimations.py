import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import math
from datetime import datetime
from typing import List, Dict, Tuple

# Optional: file readers & PDF export
try:
    import docx  # python-docx
except Exception:
    docx = None

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    from fpdf import FPDF
except Exception:
    FPDF = None

# Optional: lightweight NLP (auto-fallback if not available)
HF_OK = False
try:
    from transformers import pipeline
    HF_OK = True
except Exception:
    HF_OK = False

# -------------------- STREAMLIT CONFIG & THEME --------------------
st.set_page_config(page_title="Effort Estimation AI Tool", layout="wide")

st.markdown("""
<style>
/* App background */
.stApp {
  background: linear-gradient(135deg, #ffcc33 0%, #ff7a18 100%) !important;
  color: #1d1d1f;
  font-family: "Inter", "Segoe UI", Tahoma, sans-serif;
}
/* Panels */
.block-container { padding-top: 1.2rem; }
/* Cards */
.card {
  background: #fff6d6;
  border: 1px solid #f2c766;
  border-radius: 12px;
  padding: 1rem 1.2rem;
  box-shadow: 0 6px 16px rgba(0,0,0,0.08);
}
/* Headers & badges */
h1, h2, h3 { color: #0b3d91; font-weight: 800; }
.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  background: #003366;
  color: #fff;
  font-size: 0.8rem;
  font-weight: 700;
  margin-right: 6px;
}
/* Metrics */
.metric-box {
  background: #fff3bf;
  border: 1px dashed #e0a800;
  border-radius: 12px;
  padding: 14px;
  text-align: center;
}
/* Inputs & buttons */
.stButton > button {
  background: #0b5ed7;
  color: #fff;
  border: none;
  border-radius: 10px;
  font-weight: 700;
  padding: 0.6rem 1rem;
}
.stDownloadButton > button {
  background: #0b3d91 !important;
  color: #fff !important;
  border-radius: 10px !important;
  font-weight: 700 !important;
}
hr { border: none; height: 1px; background: #f0c36d; margin: 10px 0 6px; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Effort Estimation AI Tool")

# =========================================================
# ===================== HELPERS ===========================
# =========================================================
def read_document(file) -> str:
    """Read text from txt, docx, pdf with graceful degradation."""
    if file is None:
        return ""
    name = file.name.lower()
    try:
        if name.endswith(".txt"):
            return file.read().decode("utf-8", errors="ignore")
        if name.endswith(".docx") and docx:
            d = docx.Document(file)
            return "\n".join([p.text for p in d.paragraphs])
        if name.endswith(".pdf") and PyPDF2:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
            return text
    except Exception:
        return ""
    return ""

def basic_feature_split(text: str) -> List[str]:
    """Fast heuristic to split requirements into features."""
    chunks = re.split(r"(?:\n\s*[-•*]\s+|\n{2,}|;\s*|\.\s+)", text)
    features = [c.strip() for c in chunks if c and len(c.strip()) >= 6]
    # de-duplicate while preserving order
    seen, unique = set(), []
    for f in features:
        if f not in seen:
            unique.append(f)
            seen.add(f)
    return unique[:200]

# ---------- AI-based (optional) complexity per feature ----------
@st.cache_resource(show_spinner=False)
def get_ai_complexity_classifier():
    """
    Try to load a small local transformer pipeline for zero-shot classification.
    If not available or fails, return None (we will fallback to heuristic).
    """
    if not HF_OK:
        return None
    try:
        # small-ish NLI model; if host can't download, we'll fallback gracefully
        clf = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")
        return clf
    except Exception:
        return None

AI_CLF = get_ai_complexity_classifier()

def ai_complexity_score(feature: str) -> Tuple[str, int, str]:
    """
    AI-based complexity classification with fallback heuristic.
    Returns: (class, story_points, rationale)
    """
    labels = ["Low", "Medium", "High"]
    rationale = []
    if AI_CLF:
        try:
            res = AI_CLF(
                feature,
                candidate_labels=labels,
                hypothesis_template="This feature has {label} implementation complexity."
            )
            # Pick best label
            label = res["labels"][0]
            score = float(res["scores"][0])
            rationale.append(f"AI zero-shot classified as {label} (confidence {round(score*100,1)}%).")
            sp = {"Low": 3, "Medium": 5, "High": 8}[label]
            return label, sp, " ".join(rationale)
        except Exception:
            pass

    # Heuristic fallback
    f = feature.lower()
    high_kw = ["integration", "payment", "gateway", "sso", "oauth", "kafka", "stream", "etl",
               "ml", "ai", "analytics", "realtime", "scale", "security", "encryption",
               "compliance", "migration", "microservice", "kubernetes", "distributed",
               "multi-tenant", "observability"]
    low_kw = ["ui", "form", "page", "static", "list", "search", "filter", "report",
              "export", "crud", "login", "signup", "profile", "email", "notification"]
    is_high = any(k in f for k in high_kw)
    is_low  = any(k in f for k in low_kw)
    if is_high and not is_low:
        rationale.append("Keyword heuristic indicates complex integration/AI/security/scale.")
        return "High", 8, " ".join(rationale)
    if is_low and not is_high:
        rationale.append("Keyword heuristic indicates bounded UI/CRUD/reporting.")
        return "Low", 3, " ".join(rationale)
    return "Medium", 5, "Default heuristic (no strong signals)."

def categorize_feature(feature: str) -> str:
    f = feature.lower()
    nf_kw = ["security", "performance", "availability", "latency", "throughput", "compliance",
             "reliability", "observability", "monitor", "backup", "sla", "scalab", "devops"]
    return "Non-Functional" if any(k in f for k in nf_kw) else "Functional"

# ---------- Classical technique implementations ----------
def estimate_cocomo(features: List[str], avg_loc_per_feature: int, mode: str = "organic") -> Dict:
    """
    Basic COCOMO effort (person-months) from KLOC with multipliers by mode.
    Returns dict with effort PM, hours, KLOC and params used.
    """
    loc = max(1, avg_loc_per_feature) * max(0, len(features))
    kloc = loc / 1000.0

    # mode params (Basic COCOMO 81)
    params = {
        "organic": (2.4, 1.05),
        "semi-detached": (3.0, 1.12),
        "embedded": (3.6, 1.20),
    }
    a, b = params.get(mode, params["organic"])

    pm = a * (kloc ** b)  # effort in person-months
    hours = pm * 152  # ~19 wd * 8h
    return {
        "KLOC": round(kloc, 2),
        "PM": round(pm, 2),
        "Hours": round(hours, 1),
        "Mode": mode,
        "a": a, "b": b,
        "avg_loc_per_feature": avg_loc_per_feature,
        "features": len(features)
    }

def estimate_function_points(features: List[str],
                             ei=10, eo=6, eq=4, ilf=5, eif=3,
                             vaf=1.0, hours_per_fp=8.0) -> Dict:
    """
    Simplified FP: approximate counts from features + user overrides.
    UFP = 4*EI + 5*EO + 4*EQ + 10*ILF + 7*EIF (weights simplified here via inputs)
    Effort hours = FP * hours_per_fp * VAF
    """
    n = len(features)
    # naive apportioning from features if the user didn't change defaults
    est_ei  = max(1, int(n * 0.35)) if ei == 10 else ei
    est_eo  = max(1, int(n * 0.25)) if eo == 6  else eo
    est_eq  = max(1, int(n * 0.15)) if eq == 4  else eq
    est_ilf = max(1, int(n * 0.15)) if ilf == 5 else ilf
    est_eif = max(1, int(n * 0.10)) if eif == 3 else eif

    # weighted UFP (using typical mid weights)
    ufp = (est_ei*4) + (est_eo*5) + (est_eq*4) + (est_ilf*10) + (est_eif*7)
    fp = ufp * vaf
    hours = fp * hours_per_fp

    return {
        "EI": est_ei, "EO": est_eo, "EQ": est_eq, "ILF": est_ilf, "EIF": est_eif,
        "UFP": round(ufp, 2),
        "VAF": round(vaf, 2),
        "FP": round(fp, 2),
        "Hours": round(hours, 1),
        "hours_per_fp": hours_per_fp
    }

def estimate_use_case_points(features: List[str],
                             simple_uc=5, avg_uc=8, complex_uc=3,
                             simple_actor=3, avg_actor=3, complex_actor=1,
                             tcf=0.9, ef=1.1, hours_per_ucp=20.0) -> Dict:
    """
    Simplified UCP:
      UUCW = (5 * simple) + (10 * average) + (15 * complex)  [we map to counts directly]
      UAW  = (1 * simple actors) + (2 * avg actors) + (3 * complex actors)
      TCF, EF as inputs; UCP = (UUCW + UAW) * TCF * EF
      Hours = UCP * hours_per_ucp
    We default the counts based on features if the user hasn't changed them.
    """
    n = len(features)
    # if unchanged defaults, derive some counts
    if (simple_uc, avg_uc, complex_uc) == (5, 8, 3):
        simple_uc = max(1, int(n * 0.5))
        avg_uc = max(1, int(n * 0.35))
        complex_uc = max(1, n - simple_uc - avg_uc)

    if (simple_actor, avg_actor, complex_actor) == (3, 3, 1):
        simple_actor = 3
        avg_actor = 2 if n < 20 else 4
        complex_actor = 1 if n < 30 else 2

    uucw = (5*simple_uc) + (10*avg_uc) + (15*complex_uc)
    uaw  = (1*simple_actor) + (2*avg_actor) + (3*complex_actor)
    ucp  = (uucw + uaw) * tcf * ef
    hours = ucp * hours_per_ucp

    return {
        "UUCW": uucw, "UAW": uaw, "TCF": tcf, "EF": ef,
        "UCP": round(ucp, 2),
        "Hours": round(hours, 1),
        "counts": {
            "simple_uc": simple_uc, "avg_uc": avg_uc, "complex_uc": complex_uc,
            "simple_actor": simple_actor, "avg_actor": avg_actor, "complex_actor": complex_actor
        },
        "hours_per_ucp": hours_per_ucp
    }

def estimate_analogy(features: List[str], baseline_hours_per_feature=10.0, complexity_multiplier=1.0) -> Dict:
    """
    Analogy: hours ≈ features * baseline * multiplier
    """
    n = len(features)
    hours = n * baseline_hours_per_feature * complexity_multiplier
    return {
        "Features": n,
        "Baseline h/feature": baseline_hours_per_feature,
        "Multiplier": complexity_multiplier,
        "Hours": round(hours, 1)
    }

def estimate_expert_judgement(expert_hours=160.0, risk_multiplier=1.2) -> Dict:
    """
    Expert provides a base hours estimate, apply risk multiplier.
    """
    hours = expert_hours * risk_multiplier
    return {
        "Expert base hours": expert_hours,
        "Risk multiplier": risk_multiplier,
        "Hours": round(hours, 1)
    }

def ai_per_feature_estimates(features: List[str], hours_per_sp=8.0) -> Tuple[pd.DataFrame, Dict]:
    """
    AI-driven (or heuristic fallback) per-feature complexity -> SP -> hours.
    Returns a dataframe + summary dict.
    """
    rows = []
    for f in features:
        cplx, sp, why = ai_complexity_score(f)
        cat = categorize_feature(f)
        hours = sp * hours_per_sp
        rows.append({
            "Feature": f,
            "Category": cat,
            "AI Complexity": cplx,
            "Story Points": sp,
            "Hours": hours,
            "Rationale": why
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df, {"Hours": 0.0, "SP": 0, "counts": {"High":0,"Medium":0,"Low":0}}

    total_h = float(df["Hours"].sum())
    total_sp = int(df["Story Points"].sum())
    counts = df["AI Complexity"].value_counts().to_dict()
    for k in ["High", "Medium", "Low"]:
        counts.setdefault(k, 0)

    return df, {"Hours": round(total_h,1), "SP": total_sp, "counts": counts}

# ---------- Export helpers ----------
def to_excel_bytes(sheets: Dict[str, pd.DataFrame], summary_text: str = "") -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, index=False, sheet_name=name[:31] or "Sheet1")
        # Add a textual Summary sheet if requested
        if summary_text:
            df_sum = pd.DataFrame({"Summary":[summary_text]})
            df_sum.to_excel(writer, index=False, sheet_name="Summary")
    return output.getvalue()

def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def to_pdf_bytes(summary_lines: List[str], charts: List[bytes], tables: Dict[str, pd.DataFrame]) -> bytes:
    if not FPDF:
        content = "Install 'fpdf' to enable proper PDF export.\n\n" + "\n".join(summary_lines)
        return content.encode("utf-8")

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Effort Estimation Report", ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    pdf.ln(2)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Summary", ln=1)
    pdf.set_font("Arial", "", 11)
    for ln in summary_lines:
        pdf.multi_cell(0, 6, ln)

    # Charts
    for i, chart_bytes in enumerate(charts):
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, f"Chart {i+1}", ln=1)
        img_path = f"chart_{i}.png"
        with open(img_path, "wb") as f:
            f.write(chart_bytes)
        pdf.image(img_path, w=185)

    # Tables (first 25 rows each)
    for name, df in tables.items():
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, name, ln=1)
        pdf.set_font("Arial", "", 10)
        if df.empty:
            pdf.multi_cell(0, 5, "(no rows)")
            continue
        show = df.head(25).copy()
        for _, row in show.iterrows():
            pdf.multi_cell(0, 5, " - " + ", ".join([f"{col}: {row[col]}" for col in show.columns[:6]]))

    out = pdf.output(dest="S").encode("latin-1", errors="ignore")
    return out

# =========================================================
# ===================== SIDEBAR ===========================
# =========================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # Technique selection
    st.markdown("**Estimation Techniques**")
    use_cocomo = st.checkbox("COCOMO (Basic)")
    use_fp = st.checkbox("Function Points")
    use_ucp = st.checkbox("Use Case Points")
    use_analogy = st.checkbox("Analogy-based")
    use_expert = st.checkbox("Expert Judgment")
    use_ai = st.checkbox("AI-based NLP (optional)")

    st.markdown("---")
    st.markdown("**Common Parameters**")
    contingency_pct = st.slider("Contingency (%)", 0, 50, 20)
    want_pdf = st.checkbox("Enable PDF export", value=True)
    want_excel = st.checkbox("Enable Excel export", value=True)

    st.markdown("---")
    st.markdown("**COCOMO Inputs**")
    cocomo_mode = st.selectbox("Mode", ["organic", "semi-detached", "embedded"])
    avg_loc_per_feature = st.number_input("Avg LOC / feature", 20, 2000, 250, step=10)

    st.markdown("---")
    st.markdown("**Function Point Inputs**")
    fp_ei = st.number_input("EI count (or leave default)", 1, 999, 10)
    fp_eo = st.number_input("EO count (or leave default)", 1, 999, 6)
    fp_eq = st.number_input("EQ count (or leave default)", 1, 999, 4)
    fp_ilf = st.number_input("ILF count (or leave default)", 1, 999, 5)
    fp_eif = st.number_input("EIF count (or leave default)", 1, 999, 3)
    fp_vaf = st.slider("Value Adjustment Factor (VAF)", 0.6, 1.4, 1.0, step=0.05)
    fp_hours_per_fp = st.number_input("Hours per FP", 2.0, 20.0, 8.0, step=0.5)

    st.markdown("---")
    st.markdown("**Use Case Points Inputs**")
    ucp_simple = st.number_input("Simple UCs", 0, 999, 5)
    ucp_avg = st.number_input("Average UCs", 0, 999, 8)
    ucp_complex = st.number_input("Complex UCs", 0, 999, 3)
    actor_simple = st.number_input("Simple Actors", 0, 999, 3)
    actor_avg = st.number_input("Average Actors", 0, 999, 3)
    actor_complex = st.number_input("Complex Actors", 0, 999, 1)
    ucp_tcf = st.slider("TCF (0.6–1.4)", 0.6, 1.4, 0.9, step=0.05)
    ucp_ef = st.slider("EF (0.6–1.4)", 0.6, 1.4, 1.1, step=0.05)
    ucp_hours_per = st.number_input("Hours per UCP", 5.0, 40.0, 20.0, step=1.0)

    st.markdown("---")
    st.markdown("**Analogy Inputs**")
    an_hours_per_feature = st.number_input("Baseline hours / feature", 1.0, 80.0, 10.0, step=0.5)
    an_multiplier = st.slider("Complexity multiplier", 0.5, 3.0, 1.0, step=0.1)

    st.markdown("---")
    st.markdown("**Expert Judgment Inputs**")
    expert_hours = st.number_input("Expert base hours", 1.0, 10000.0, 160.0, step=1.0)
    expert_risk = st.slider("Risk multiplier", 0.5, 3.0, 1.2, step=0.1)

    st.markdown("---")
    st.markdown("**AI-based NLP Inputs**")
    ai_hours_per_sp = st.number_input("Hours per Story Point (AI)", 1.0, 16.0, 8.0, step=0.5)
    st.caption("If a Transformer model can't be loaded on your host, the AI method will gracefully fallback to a heuristic.")

# =========================================================
# ===================== INPUT TABS ========================
# =========================================================
tab_paste, tab_upload = st.tabs(["✍️ Paste Requirements", "📄 Upload Document"])

with tab_paste:
    with st.form("paste_form"):
        text_input = st.text_area(
            "Paste project requirements here:",
            height=220,
            placeholder="Paste bullet points, scope, modules, integrations..."
        )
        submitted_paste = st.form_submit_button("🔍 Generate Estimation", use_container_width=True)

with tab_upload:
    with st.form("upload_form"):
        file = st.file_uploader("Upload a document (.txt, .docx, .pdf)", type=["txt", "docx", "pdf"])
        submitted_upload = st.form_submit_button("🔍 Generate Estimation", use_container_width=True)

# Single-click decision
requirements_text = ""
trigger = False
if submitted_paste and text_input.strip():
    requirements_text = text_input.strip()
    trigger = True
elif submitted_upload and file:
    requirements_text = read_document(file).strip()
    trigger = True

# =========================================================
# ===================== RUN ESTIMATION ====================
# =========================================================
if trigger and requirements_text:
    st.markdown("### 🔧 Processing")
    st.markdown('<div class="card">We’re extracting features, applying selected estimation models, and computing capacity & schedule metrics.</div>', unsafe_allow_html=True)

    # Extract candidate features (fast heuristic)
    features = basic_feature_split(requirements_text)
    if not features:
        st.warning("Couldn’t extract features. Please refine the requirements.")
    else:
        # Container to collect results
        rows_summary = []   # for chart + summary table
        sheets = {}         # for Excel export
        charts_bytes = []   # for PDF export

        # ========== AI-based NLP (optional) ==========
        ai_df = pd.DataFrame()
        ai_total_hours = 0.0
        if use_ai:
            ai_df, ai_meta = ai_per_feature_estimates(features, hours_per_sp=ai_hours_per_sp)
            ai_total_hours = ai_meta["Hours"]
            rows_summary.append(["AI-based NLP", ai_total_hours])
            # Save sheet
            sheets["AI-based (per-feature)"] = ai_df

        # ========== COCOMO ==========
        if use_cocomo:
            cocomo = estimate_cocomo(features, avg_loc_per_feature, cocomo_mode)
            rows_summary.append(["COCOMO ("+cocomo_mode+")", cocomo["Hours"]])
            sheets["COCOMO"] = pd.DataFrame([cocomo])

        # ========== Function Points ==========
        if use_fp:
            fp = estimate_function_points(features,
                                          ei=fp_ei, eo=fp_eo, eq=fp_eq, ilf=fp_ilf, eif=fp_eif,
                                          vaf=fp_vaf, hours_per_fp=fp_hours_per_fp)
            rows_summary.append(["Function Points", fp["Hours"]])
            sheets["Function Points"] = pd.DataFrame([fp])

        # ========== Use Case Points ==========
        if use_ucp:
            ucp = estimate_use_case_points(features,
                                           simple_uc=ucp_simple, avg_uc=ucp_avg, complex_uc=ucp_complex,
                                           simple_actor=actor_simple, avg_actor=actor_avg, complex_actor=actor_complex,
                                           tcf=ucp_tcf, ef=ucp_ef, hours_per_ucp=ucp_hours_per)
            rows_summary.append(["Use Case Points", ucp["Hours"]])
            sheets["Use Case Points"] = pd.DataFrame([ucp])

        # ========== Analogy ==========
        if use_analogy:
            an = estimate_analogy(features, baseline_hours_per_feature=an_hours_per_feature, complexity_multiplier=an_multiplier)
            rows_summary.append(["Analogy", an["Hours"]])
            sheets["Analogy"] = pd.DataFrame([an])

        # ========== Expert Judgment ==========
        if use_expert:
            ej = estimate_expert_judgement(expert_hours=expert_hours, risk_multiplier=expert_risk)
            rows_summary.append(["Expert Judgment", ej["Hours"]])
            sheets["Expert Judgment"] = pd.DataFrame([ej])

        # If user forgot to pick any technique
        if not rows_summary:
            st.error("⚠️ Please select at least one estimation technique from the sidebar.")
        else:
            # Combine + add contingency and key metrics
            comp_df = pd.DataFrame(rows_summary, columns=["Technique", "Base Hours"])
            comp_df["With Contingency"] = comp_df["Base Hours"] * (1.0 + contingency_pct/100.0)

            total_base = float(comp_df["Base Hours"].mean())  # average across selected techniques
            total_with_cont = float(comp_df["With Contingency"].mean())

            # ========== METRICS STRIP ==========
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-box"><h3>Avg Base Hours (across chosen)</h3><h2>{}</h2></div>'.format(round(total_base,1)), unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-box"><h3>Avg + Contingency ({}%)</h3><h2>{}</h2></div>'.format(contingency_pct, round(total_with_cont,1)), unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-box"><h3>Features Detected</h3><h2>{}</h2></div>'.format(len(features)), unsafe_allow_html=True)

            # ========== DETAIL TABLES ==========
            st.markdown("### 📋 Technique Comparison")
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # ========== CHARTS ==========
            import matplotlib.pyplot as plt

            # Bar chart of estimates
            fig1, ax1 = plt.subplots()
            ax1.bar(comp_df["Technique"], comp_df["With Contingency"])
            ax1.set_title("Estimated Hours (with Contingency)")
            ax1.set_ylabel("Hours")
            ax1.set_xticklabels(comp_df["Technique"], rotation=20, ha="right")
            st.pyplot(fig1, use_container_width=True)
            charts_bytes.append(fig_to_png_bytes(fig1))

            # If AI per-feature available, show complexity mix pie
            if use_ai and not ai_df.empty:
                st.markdown("### 🧩 AI Complexity Mix")
                mix = ai_df["AI Complexity"].value_counts().to_dict()
                labels = list(mix.keys())
                vals = list(mix.values())
                fig2, ax2 = plt.subplots()
                ax2.pie(vals, labels=labels, autopct="%1.1f%%", startangle=130)
                ax2.set_title("AI-Detected Complexity Distribution")
                st.pyplot(fig2, use_container_width=True)
                charts_bytes.append(fig_to_png_bytes(fig2))

                # Cumulative hours curve
                st.markdown("### ⏱️ AI Cumulative Effort (Top 30 Features)")
                top = ai_df.head(30).copy()
                top["Cum Hours"] = top["Hours"].cumsum()
                fig3, ax3 = plt.subplots()
                ax3.plot(range(1, len(top)+1), top["Cum Hours"], marker="o")
                ax3.set_xlabel("Feature #")
                ax3.set_ylabel("Cumulative Hours")
                ax3.set_title("AI Cumulative Effort Curve")
                st.pyplot(fig3, use_container_width=True)
                charts_bytes.append(fig_to_png_bytes(fig3))

            # ========== DEFENSE NOTES ==========
            st.markdown("### 🛡️ Assumptions & Defense Notes")
            bullets = [
                f"Contingency set to **{contingency_pct}%** to offset unknowns and rework.",
                f"COCOMO uses average LOC/feature = **{avg_loc_per_feature}** and **{cocomo_mode}** mode parameters.",
                f"Function Points weights are simplified mid-weights; **Hours/FP = {fp_hours_per_fp}**; **VAF = {fp_vaf}**.",
                f"Use Case Points assume **Hours/UCP = {ucp_hours_per}**, with TCF={ucp_tcf}, EF={ucp_ef}.",
                f"Analogy baseline **{an_hours_per_feature} h/feature**, multiplier **{an_multiplier}**.",
                f"Expert Judgment base **{expert_hours} h**, risk multiplier **{expert_risk}**.",
                f"AI-based NLP uses a local zero-shot classifier if available; otherwise a keyword heuristic. **Hours/SP = {ai_hours_per_sp}**.",
                "Final recommendation is based on the **average across selected techniques** to reduce single-model bias.",
            ]
            st.markdown('<div class="card">' + "<br/>".join(["• " + b for b in bullets]) + "</div>", unsafe_allow_html=True)

            # ========== EXPORTS ==========
            st.markdown("### 📦 Export")

            # Prepare Excel sheets
            sheets["Technique Comparison"] = comp_df
            if use_ai and not ai_df.empty:
                # also add AI summary sheet
                ai_summary_df = pd.DataFrame([{
                    "AI Hours": total_with_cont if use_ai else 0,
                    "AI Base Hours": ai_total_hours,
                    "Features": len(features),
                    "High": int(ai_df["AI Complexity"].eq("High").sum()),
                    "Medium": int(ai_df["AI Complexity"].eq("Medium").sum()),
                    "Low": int(ai_df["AI Complexity"].eq("Low").sum())
                }])
                sheets["AI Summary"] = ai_summary_df

            # Summary text for file exports
            summary_text = (
                f"Features detected: {len(features)}\n"
                f"Avg base hours (selected techniques): {round(total_base,1)}\n"
                f"Avg hours incl. contingency ({contingency_pct}%): {round(total_with_cont,1)}"
            )

            col_a, col_b = st.columns(2)

            if want_excel:
                with col_a:
                    xls_bytes = to_excel_bytes(sheets, summary_text=summary_text)
                    st.download_button(
                        label="⬇️ Download Estimation (XLSX)",
                        data=xls_bytes,
                        file_name="effort_estimation.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

            if want_pdf:
                with col_b:
                    # Build concise summary lines
                    lines = [
                        f"Features detected: {len(features)}",
                        f"Average Base Hours (across chosen): {round(total_base,1)}",
                        f"Average + Contingency ({contingency_pct}%): {round(total_with_cont,1)}",
                    ]
                    if use_ai and not ai_df.empty:
                        cts = ai_df["AI Complexity"].value_counts().to_dict()
                        lines.append(
                            f"AI complexity counts -> High: {cts.get('High',0)}, "
                            f"Medium: {cts.get('Medium',0)}, Low: {cts.get('Low',0)}"
                        )

                    tables_for_pdf = {"Technique Comparison": comp_df}
                    if use_ai and not ai_df.empty:
                        tables_for_pdf["AI (first 25 rows)"] = ai_df[["Feature","AI Complexity","Story Points","Hours"]]

                    pdf_bytes = to_pdf_bytes(lines, charts_bytes, tables_for_pdf)
                    st.download_button(
                        label="⬇️ Download Report (PDF)",
                        data=pdf_bytes,
                        file_name="effort_estimation.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

else:
    st.info("Paste requirements or upload a document, then click **Generate Estimation**.")
