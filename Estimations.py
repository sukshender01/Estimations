import streamlit as st
import pandas as pd
import io
import re
import math
import random
from datetime import datetime
from typing import List, Dict, Tuple

# Optional file readers & PDF
try:
    import docx  # python-docx
except Exception:
    docx = None

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# ReportLab for robust PDF export
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
.block-container { padding-top: 1.0rem; }
.card {
  background: #fff6d6;
  border: 1px solid #f2c766;
  border-radius: 12px;
  padding: 1rem 1.2rem;
  box-shadow: 0 6px 16px rgba(0,0,0,0.08);
}
h1, h2, h3 { color: #0b3d91; font-weight: 800; }
.badge {
  display: inline-block; padding: 4px 10px; border-radius: 999px;
  background: #003366; color: #fff; font-size: 0.8rem; font-weight: 700; margin-right: 6px;
}
.metric-box {
  background: #fff3bf; border: 1px dashed #e0a800; border-radius: 12px;
  padding: 14px; text-align: center;
}
hr { border: none; height: 1px; background: #f0c36d; margin: 10px 0 6px; }
.help-btn button { width: 100%; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Effort Estimation AI Tool")

# -------------------- HELPERS --------------------
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
    """Heuristic to split requirements into features."""
    chunks = re.split(r"(?:\n\s*[-•*]\s+|\n{2,}|;\s*|\.\s+)", text)
    features = [c.strip() for c in chunks if c and len(c.strip()) >= 6]
    seen, unique = set(), []
    for f in features:
        key = re.sub(r"\s+", " ", f.lower())
        if key not in seen:
            unique.append(f)
            seen.add(key)
    return unique[:200]

def classify_complexity(feature: str) -> Tuple[str, int, str]:
    """
    Classify feature complexity and map to story points with a short rationale.
    Returns: (complexity, story_points, rationale)
    """
    f = feature.lower()
    rationale = []
    high_kw = ["integration", "payment", "gateway", "sso", "oauth", "kafka", "stream", "etl",
               "ml", "ai", "analytics", "realtime", "scalab", "security", "encryption",
               "compliance", "migration", "microservice", "kubernetes", "distributed",
               "multi-tenant", "observability"]
    low_kw = ["ui", "form", "page", "static", "list", "search", "filter", "report", "export",
              "crud", "login", "signup", "profile", "email", "notification"]

    is_high = any(k in f for k in high_kw)
    is_low  = any(k in f for k in low_kw)

    complexity, sp = "Medium", 5
    if is_high and not is_low:
        complexity, sp = "High", 8
        rationale.append("High-risk keywords (integration/AI/security/scale).")
    elif is_low and not is_high:
        complexity, sp = "Low", 3
        rationale.append("UI/CRUD-oriented and bounded scope.")
    elif is_high and is_low:
        complexity, sp = "High", 8
        rationale.append("UI + complex backend integration.")
    else:
        complexity, sp = "Medium", 5
        rationale.append("No strong signals; average scope.")

    vague = ["etc", "optimize", "improve", "efficient", "user-friendly", "robust"]
    if any(v in f for v in vague) and complexity != "High":
        sp += 2
        rationale.append("Ambiguity increases estimate.")

    sp = min(sp, 13)
    return complexity, sp, " ".join(rationale)

def categorize_feature(feature: str) -> str:
    f = feature.lower()
    nf_kw = ["security", "performance", "availability", "latency", "throughput", "compliance",
             "reliability", "observability", "monitor", "backup", "sla", "scalab", "devops"]
    return "Non-Functional" if any(k in f for k in nf_kw) else "Functional"

def estimate_per_feature(features: List[str], hours_per_point: float = 8.0) -> List[Dict]:
    rows = []
    for f in features:
        complexity, sp, rationale = classify_complexity(f)
        hours = sp * hours_per_point
        rows.append({
            "Feature": f,
            "Category": categorize_feature(f),
            "Complexity": complexity,
            "Story Points": sp,
            "Effort (hours)": hours,
            "Rationale": rationale
        })
    return rows

# ---- Five techniques (deterministic formulas; no external APIs) ----
def cocomo_estimation(size_kloc: float):
    # COCOMO Basic (Organic defaults)
    a, b = 2.4, 1.05
    person_months = a * (size_kloc ** b)
    # Convert PM to hours (assume 152 h per PM)
    hours = person_months * 152.0
    # Schedule in months
    time_months = 2.5 * (person_months ** 0.38)
    return max(hours, 0.0), max(time_months, 0.0)

def function_point_estimation(fp_count: int, hours_per_fp: float = 5.0):
    hours = fp_count * hours_per_fp
    time_months = hours / (20.0 * 8.0)  # 20 workdays/month * 8h/day
    return max(hours, 0.0), max(time_months, 0.0)

def use_case_points_estimation(use_cases: int, hours_per_ucp: float = 10.0):
    hours = use_cases * hours_per_ucp
    time_months = hours / (20.0 * 8.0)
    return max(hours, 0.0), max(time_months, 0.0)

def expert_judgment_estimation(requirements_text: str):
    words = len(requirements_text.split())
    # heuristic: 1 hour / 12 words + base
    hours = words / 12.0 + 24.0
    time_months = hours / (20.0 * 8.0)
    return max(hours, 0.0), max(time_months, 0.0)

def ai_based_estimation(requirements_text: str):
    # local NLP-ish heuristic: length, diversity, risk keywords
    text = requirements_text.lower()
    words = re.findall(r"[a-z0-9]+", text)
    vocab = len(set(words))
    risk_terms = sum(1 for w in words if w in {
        "integration","pci","gdpr","hipaa","sso","oauth","kafka","etl","pipeline",
        "realtime","encryption","microservice","kubernetes","ai","ml","compliance",
        "migration","multi-tenant","stream","security"
    })
    base = len(words) / 11.5
    complexity_boost = (vocab / max(len(words),1)) * 180
    risk_boost = risk_terms * 7.5
    hours = base + complexity_boost + risk_boost + 16.0
    time_months = hours / (20.0 * 8.0)
    return max(hours, 0.0), max(time_months, 0.0)

# ---- Metrics & planning ----
def compute_aggregate_metrics(rows: List[Dict],
                              team_size: int,
                              hours_per_day: float,
                              sprint_days: int,
                              contingency_pct: float) -> Dict:
    df = pd.DataFrame(rows)
    total_hours = float(df["Effort (hours)"].sum()) if not df.empty else 0.0
    contingency_hours = total_hours * (contingency_pct / 100.0)
    grand_total = total_hours + contingency_hours

    weekly_capacity = team_size * hours_per_day * 5.0
    weeks = (grand_total / weekly_capacity) if weekly_capacity > 0 else 0
    sprints = math.ceil((weeks * 5.0) / sprint_days) if sprint_days > 0 else 0

    n = len(rows)
    high = int((df["Complexity"] == "High").sum()) if not df.empty else 0
    low  = int((df["Complexity"] == "Low").sum()) if not df.empty else 0
    confidence = min(0.95, 0.6 + (low * 0.01) + (n * 0.002) - (high * 0.005))
    confidence = max(0.4, confidence)

    by_complexity = df.groupby("Complexity")["Effort (hours)"].sum().to_dict() if not df.empty else {}
    by_category   = df.groupby("Category")["Effort (hours)"].sum().to_dict() if not df.empty else {}

    return {
        "total_hours": round(total_hours, 1),
        "contingency_hours": round(contingency_hours, 1),
        "grand_total_hours": round(grand_total, 1),
        "weekly_capacity": round(weekly_capacity, 1),
        "duration_weeks": round(weeks, 2),
        "sprints": int(sprints),
        "by_complexity": by_complexity,
        "by_category": by_category,
        "confidence": round(confidence * 100, 1),
        "counts": {
            "features": n,
            "high": int(high),
            "low": int(low),
            "medium": int((df["Complexity"] == "Medium").sum()) if not df.empty else 0
        }
    }

def suggest_team_and_plan(total_hours: float, hours_per_day: float, sprint_days: int):
    # Assume team members available 5 days/week
    # Start with minimal cross-functional team, scale by workload
    base_team = {"Project Manager": 1, "Business Analyst": 1, "Developers": 2, "QA Engineers": 1, "DevOps": 1}
    # Scale dev & QA with workload
    scale = max(0, total_hours - 400) / 200.0
    add_devs = int(scale * 1.2)
    add_qas  = max(0, int(scale * 0.6))
    team = base_team.copy()
    team["Developers"] += add_devs
    team["QA Engineers"] += add_qas

    weekly_capacity = (sum(team.values()) * hours_per_day * 5.0)
    sprint_capacity = weekly_capacity * (sprint_days / 5.0)
    sprints = max(1, math.ceil(total_hours / max(sprint_capacity, 1.0)))
    return team, sprints

# ---- Export helpers ----
def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def to_excel_bytes(estimation_df: pd.DataFrame,
                   technique_df: pd.DataFrame,
                   summary: Dict) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        estimation_df.to_excel(writer, index=False, sheet_name="Per-Feature Estimation")
        technique_df.to_excel(writer, sheet_name="Techniques", index=True)
        s = pd.DataFrame({
            "Metric": [
                "Total (hours)", "Contingency (hours)", "Grand Total (hours)",
                "Weekly Capacity (hours)", "Duration (weeks)", "Sprints (est.)",
                "Confidence (%)", "Feature Count"
            ],
            "Value": [
                summary["total_hours"], summary["contingency_hours"], summary["grand_total_hours"],
                summary["weekly_capacity"], summary["duration_weeks"], summary["sprints"],
                summary["confidence"], summary["counts"]["features"]
            ]
        })
        s.to_excel(writer, index=False, sheet_name="Summary")
    return output.getvalue()

def to_pdf_bytes(lines: List[str],
                 charts_bytes: List[bytes],
                 tables: Dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Effort Estimation Report</b>", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M"), styles["Normal"]))
    story.append(Spacer(1, 12))

    for ln in lines:
        story.append(Paragraph(ln, styles["Normal"]))
    story.append(Spacer(1, 12))

    # Charts
    for i, ch in enumerate(charts_bytes):
        img = RLImage(io.BytesIO(ch), width=500, height=280)
        story.append(Spacer(1, 8))
        story.append(img)

    # Tables
    for title, df in tables.items():
        story.append(PageBreak())
        story.append(Paragraph(f"<b>{title}</b>", styles["Heading2"]))
        story.append(Spacer(1, 8))
        data = [df.columns.tolist()] + df.values.tolist()
        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2c766")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.black),
            ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("ALIGN", (0,0), (-1,0), "CENTER"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.HexColor("#fff6d6")]),
        ]))
        story.append(table)

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

# ---- Help text for each technique ----
HELP = {
    "COCOMO": """<b>COCOMO</b> estimates based on code size (KLOC).
Example: 12 KLOC (organic) → PM ≈ 2.4 * 12^1.05; Hours ≈ PM * 152; Schedule ≈ 2.5 * PM^0.38.""",
    "Function Points": """<b>Function Points</b> count user-visible functions (inputs/outputs/files).
Example: 120 FP with 5 h/FP → 600 hours.""",
    "Use Case Points": """<b>Use Case Points</b> scale with number/complexity of use cases.
Example: 25 UCP with 10 h/UCP → 250 hours.""",
    "Expert Judgment": """<b>Expert Judgment</b> uses calibrated heuristics from similar projects.
Example: CRM rework of similar size ~ 280 hours.""",
    "AI-based NLP": """<b>AI-based NLP</b> (local heuristic): wordcount, vocabulary diversity & risk keywords
(integrations, compliance, AI/ML) add effort to reflect hidden complexity."""
}

# -------------------- SIDEBAR: PARAMETERS --------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    team_size = st.number_input("Team size (FTE) (for metrics)", 1, 50, 5)
    hours_per_day = st.number_input("Hours per person/day", 1.0, 12.0, 7.0)
    sprint_days = st.number_input("Sprint length (days)", 5, 20, 10)
    hours_per_point = st.number_input("Hours per Story Point (per-feature table)", 4.0, 12.0, 8.0, step=0.5)
    contingency_pct = st.slider("Contingency (%)", 0, 50, 20)

    st.markdown("---")
    st.markdown("**Exports**")
    want_pdf = st.checkbox("Enable PDF export", value=True)
    want_excel = st.checkbox("Enable Excel export", value=True)

# -------------------- INPUT TABS --------------------
tab_paste, tab_upload = st.tabs(["✍️ Paste Requirements", "📄 Upload Document"])
requirements_text = ""
trigger = False

with tab_paste:
    with st.form("paste_form"):
        pasted = st.text_area("Paste project requirements:", height=220, placeholder="Paste bullet points, scope, modules, integrations…")
        # Technique choices & help — in the same form to avoid double click
        st.markdown("#### Select Estimation Techniques")
        cols = st.columns([1,1,1,1,1])
        with cols[0]:
            t_cocomo = st.checkbox("COCOMO", value=True)
            if st.button("❓", key="help_cocomo"):
                st.info(HELP["COCOMO"], icon="ℹ️")
        with cols[1]:
            t_fp = st.checkbox("Function Points", value=True)
            if st.button("❓", key="help_fp"):
                st.info(HELP["Function Points"], icon="ℹ️")
        with cols[2]:
            t_ucp = st.checkbox("Use Case Points", value=True)
            if st.button("❓", key="help_ucp"):
                st.info(HELP["Use Case Points"], icon="ℹ️")
        with cols[3]:
            t_expert = st.checkbox("Expert Judgment", value=True)
            if st.button("❓", key="help_expert"):
                st.info(HELP["Expert Judgment"], icon="ℹ️")
        with cols[4]:
            t_ai = st.checkbox("AI-based NLP", value=True)
            if st.button("❓", key="help_ai"):
                st.info(HELP["AI-based NLP"], icon="ℹ️")

        submitted_paste = st.form_submit_button("🔍 Generate Estimation", use_container_width=True)
        if submitted_paste and pasted.strip():
            requirements_text = pasted.strip()
            trigger = True
            techniques_selected = {
                "COCOMO": t_cocomo, "Function Points": t_fp, "Use Case Points": t_ucp,
                "Expert Judgment": t_expert, "AI-based NLP": t_ai
            }

with tab_upload:
    with st.form("upload_form"):
        file = st.file_uploader("Upload a document (.txt, .docx, .pdf)", type=["txt", "docx", "pdf"])
        # Technique choices & help — mirror the paste form
        st.markdown("#### Select Estimation Techniques")
        cols2 = st.columns([1,1,1,1,1])
        with cols2[0]:
            u_cocomo = st.checkbox("COCOMO ", key="u_cocomo", value=True)
            if st.button("❓ ", key="help_cocomo_u"):
                st.info(HELP["COCOMO"], icon="ℹ️")
        with cols2[1]:
            u_fp = st.checkbox("Function Points ", key="u_fp", value=True)
            if st.button("❓  ", key="help_fp_u"):
                st.info(HELP["Function Points"], icon="ℹ️")
        with cols2[2]:
            u_ucp = st.checkbox("Use Case Points ", key="u_ucp", value=True)
            if st.button("❓   ", key="help_ucp_u"):
                st.info(HELP["Use Case Points"], icon="ℹ️")
        with cols2[3]:
            u_expert = st.checkbox("Expert Judgment ", key="u_expert", value=True)
            if st.button("❓    ", key="help_expert_u"):
                st.info(HELP["Expert Judgment"], icon="ℹ️")
        with cols2[4]:
            u_ai = st.checkbox("AI-based NLP ", key="u_ai", value=True)
            if st.button("❓     ", key="help_ai_u"):
                st.info(HELP["AI-based NLP"], icon="ℹ️")

        submitted_upload = st.form_submit_button("🔍 Generate Estimation", use_container_width=True)
        if submitted_upload and file:
            requirements_text = read_document(file).strip()
            trigger = True
            techniques_selected = {
                "COCOMO": u_cocomo, "Function Points": u_fp, "Use Case Points": u_ucp,
                "Expert Judgment": u_expert, "AI-based NLP": u_ai
            }

# -------------------- RUN ESTIMATION (single click via forms) --------------------
if trigger and requirements_text:
    st.markdown("### 🔧 Processing")
    st.markdown('<div class="card">Extracting features, applying selected techniques, and computing capacity & schedule metrics.</div>', unsafe_allow_html=True)

    # Identify features and per-feature baseline
    features = basic_feature_split(requirements_text)
    if not features:
        st.warning("Couldn’t extract features. Please refine the requirements.")
    else:
        per_feature_rows = estimate_per_feature(features, hours_per_point=hours_per_point)
        per_feature_df = pd.DataFrame(per_feature_rows)

        # Techniques
        results = {}
        # Simple proxies from text to numeric inputs
        approx_kloc = max(1, int(len(re.findall(r"[a-z0-9]+", requirements_text.lower())) / 45))  # rough
        approx_fp   = max(10, int(len(features) * 8))  # 8 FP per feature (heuristic)
        approx_ucp  = len(features)

        if techniques_selected.get("COCOMO"):
            hrs, months = cocomo_estimation(approx_kloc)
            results["COCOMO"] = {"effort": round(hrs, 1), "time_months": round(months, 2)}

        if techniques_selected.get("Function Points"):
            hrs, months = function_point_estimation(approx_fp, hours_per_fp=5.0)
            results["Function Points"] = {"effort": round(hrs, 1), "time_months": round(months, 2)}

        if techniques_selected.get("Use Case Points"):
            hrs, months = use_case_points_estimation(approx_ucp, hours_per_ucp=10.0)
            results["Use Case Points"] = {"effort": round(hrs, 1), "time_months": round(months, 2)}

        if techniques_selected.get("Expert Judgment"):
            hrs, months = expert_judgment_estimation(requirements_text)
            results["Expert Judgment"] = {"effort": round(hrs, 1), "time_months": round(months, 2)}

        if techniques_selected.get("AI-based NLP"):
            hrs, months = ai_based_estimation(requirements_text)
            results["AI-based NLP"] = {"effort": round(hrs, 1), "time_months": round(months, 2)}

        if not results:
            st.error("Please select at least one estimation technique.")
        else:
            # ---------- METRICS STRIP ----------
            # Aggregate using per-feature baseline + contingency etc.
            summary = compute_aggregate_metrics(per_feature_rows, team_size, hours_per_day, sprint_days, contingency_pct)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown('<div class="metric-box"><h3>Total Hours</h3><h2>{}</h2></div>'.format(summary["total_hours"]), unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="metric-box"><h3>Grand Total (+{}%)</h3><h2>{}</h2></div>'.format(contingency_pct, summary["grand_total_hours"]), unsafe_allow_html=True)
            with c3:
                st.markdown('<div class="metric-box"><h3>Duration (weeks)</h3><h2>{}</h2></div>'.format(summary["duration_weeks"]), unsafe_allow_html=True)
            with c4:
                st.markdown('<div class="metric-box"><h3>Confidence</h3><h2>{}%</h2></div>'.format(summary["confidence"]), unsafe_allow_html=True)

            # ---------- IDENTIFIED FEATURES ----------
            st.markdown("### 🧩 Identified Features")
            st.write(", ".join(features[:40]) + (" ..." if len(features) > 40 else ""))

            # ---------- PER-FEATURE TABLE ----------
            st.markdown("### 📋 Per-Feature Estimation (Heuristic Baseline)")
            st.dataframe(per_feature_df, use_container_width=True, hide_index=True)

            # ---------- TECHNIQUE RESULTS ----------
            st.markdown("### 🧪 Technique Results")
            tech_df = pd.DataFrame(results).T.rename(columns={"effort":"Effort (hours)", "time_months":"Time (months)"})
            st.dataframe(tech_df, use_container_width=True)

            # ---------- CHARTS ----------
            charts_bytes = []

            # Bar: Effort by Technique
            fig1, ax1 = plt.subplots()
            tech_df["Effort (hours)"].plot(kind="bar", ax=ax1)
            ax1.set_ylabel("Hours")
            ax1.set_title("Effort by Technique")
            st.pyplot(fig1, use_container_width=True)
            charts_bytes.append(fig_to_png_bytes(fig1))

            # Pie: Per-feature Functional vs Non-Functional (baseline hours)
            st.markdown("### 🧭 Functional vs Non-Functional (Baseline)")
            by_cat = per_feature_df.groupby("Category")["Effort (hours)"].sum()
            if not by_cat.empty:
                fig2, ax2 = plt.subplots()
                ax2.pie(by_cat.values, labels=by_cat.index, autopct="%1.1f%%", startangle=140)
                ax2.set_title("Effort by Category")
                st.pyplot(fig2, use_container_width=True)
                charts_bytes.append(fig_to_png_bytes(fig2))

            # Line: Cumulative Effort over first 30 features
            st.markdown("### ⏱️ Cumulative Effort (Top 30 Features)")
            top = per_feature_df.head(30).copy()
            if not top.empty:
                top["Cum Hours"] = top["Effort (hours)"].cumsum()
                fig3, ax3 = plt.subplots()
                ax3.plot(range(1, len(top)+1), top["Cum Hours"], marker="o")
                ax3.set_xlabel("Feature #")
                ax3.set_ylabel("Cumulative Hours")
                ax3.set_title("Cumulative Effort Curve")
                st.pyplot(fig3, use_container_width=True)
                charts_bytes.append(fig_to_png_bytes(fig3))

            # ---------- TEAM & SPRINT PLAN ----------
            st.markdown("### 👥 Team & Sprint Plan (Suggestion)")
            team_suggestion, sprints_needed = suggest_team_and_plan(summary["grand_total_hours"], hours_per_day, sprint_days)
            colA, colB = st.columns(2)
            with colA:
                st.write("**Suggested Team (FTE)**")
                st.write(pd.DataFrame({"Role": list(team_suggestion.keys()), "Count": list(team_suggestion.values())}))
            with colB:
                st.write(f"**Estimated Sprints Needed:** {sprints_needed} (with sprint length {sprint_days} days)")

            # ---------- DEFENSE NOTES ----------
            st.markdown("### 🛡️ Assumptions & Defense Notes")
            st.markdown(
                f"""
<div class="card">
<span class="badge">Scope Assumptions</span><br/>
• Hours/Story Point (baseline table) = <b>{hours_per_point}</b>.<br/>
• Contingency = <b>{contingency_pct}%</b> to offset unknowns & rework.<br/>
• Team Capacity = <b>{team_size} FTE × {hours_per_day} h/day × 5 days/week</b>.<br/><br/>
<span class="badge">Technique Interpretation</span><br/>
• <b>COCOMO</b> uses proxy KLOC from requirement size to reflect scale.<br/>
• <b>FP</b> approximates 8 FP/feature (tunable).<br/>
• <b>UCP</b> approximates 1 UCP/feature; weight can be adjusted.<br/>
• <b>Expert Judgment</b> calibrated against text size (historical heuristic).<br/>
• <b>AI-based NLP</b> boosts effort for risk keywords & diversity (integration/security/AI).<br/><br/>
<span class="badge">Schedule Rationale</span><br/>
• Estimated <b>{summary["duration_weeks"]} weeks</b> → about <b>{summary["sprints"]} sprints</b> ({sprint_days}d each).<br/><br/>
<span class="badge">Confidence</span><br/>
• Current confidence: <b>{summary["confidence"]}%</b> (improves with clearer requirements and SME validation).<br/>
</div>
                """,
                unsafe_allow_html=True
            )

            # ---------- EXPORTS ----------
            st.markdown("### 📦 Export")
            col_x, col_y = st.columns(2)

            if want_excel:
                with col_x:
                    xls_bytes = to_excel_bytes(per_feature_df, tech_df, summary)
                    st.download_button(
                        label="⬇️ Download Estimation (XLSX)",
                        data=xls_bytes,
                        file_name="effort_estimation.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

            if want_pdf:
                with col_y:
                    # Assemble lines + tables for PDF
                    lines = [
                        f"Total Hours (baseline): {summary['total_hours']}",
                        f"Grand Total (+{contingency_pct}%): {summary['grand_total_hours']}",
                        f"Duration (weeks): {summary['duration_weeks']}",
                        f"Sprints (est.): {summary['sprints']}",
                        f"Confidence: {summary['confidence']}%",
                        f"Features: {summary['counts']['features']} "
                        f"(H:{summary['counts']['high']}, M:{summary['counts']['medium']}, L:{summary['counts']['low']})"
                    ]
                    tables_for_pdf = {
                        "Technique Results": tech_df.reset_index().rename(columns={"index":"Technique"}),
                        "Per-Feature Estimation": per_feature_df
                    }
                    pdf_bytes = to_pdf_bytes(lines, charts_bytes, tables_for_pdf)
                    st.download_button(
                        label="⬇️ Download Report (PDF)",
                        data=pdf_bytes,
                        file_name="effort_estimation.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

else:
    st.info("Paste requirements or upload a document, choose techniques, then click **Generate Estimation**.")
