import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

# ------------------------------
# Utility: Save Excel
# ------------------------------
def to_excel(df_dict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in df_dict.items():
            df.to_excel(writer, index=False, sheet_name=name[:30])
    return output.getvalue()

# ------------------------------
# Utility: Save PDF (ReportLab)
# ------------------------------
def to_pdf(title, df_dict, charts):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 20))

    for section, df in df_dict.items():
        story.append(Paragraph(section, styles["Heading2"]))
        story.append(Spacer(1, 12))

        table_data = [list(df.columns)] + df.values.tolist()
        table = Table(table_data)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 0.5, colors.black)
        ]))
        story.append(table)
        story.append(Spacer(1, 20))

    for chart_title, chart_bytes in charts.items():
        story.append(Paragraph(chart_title, styles["Heading2"]))
        story.append(Spacer(1, 12))
        from reportlab.platypus import Image
        img = Image(io.BytesIO(chart_bytes))
        img._restrictSize(400, 250)
        story.append(img)
        story.append(Spacer(1, 20))

    doc.build(story)
    return buffer.getvalue()

# ------------------------------
# Estimation Techniques
# ------------------------------
def cocomo_estimation(lines):
    effort = 2.94 * (lines ** 1.1)  # simplified
    return round(effort, 2)

def fp_estimation(features):
    fp = len(features) * 5
    effort = fp * 2.5
    return round(effort, 2)

def ucp_estimation(features):
    ucp = len(features) * 1.4
    effort = ucp * 3
    return round(effort, 2)

def ai_estimation(text):
    words = len(text.split())
    effort = words * 0.1
    return round(effort, 2)

def expert_estimation(features):
    return len(features) * 15

# ------------------------------
# Feature Extraction (Dummy NLP)
# ------------------------------
def extract_features(text):
    sentences = text.split(".")
    features = [s.strip() for s in sentences if len(s.strip()) > 5]
    return features

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="AI Effort Estimation Tool", layout="wide")
st.title("🧮 AI-Powered Effort Estimation Tool")

st.sidebar.header("Choose Estimation Techniques")
methods = st.sidebar.multiselect(
    "Select one or more methods:",
    ["COCOMO", "Function Point", "Use Case Point", "AI-based NLP", "Expert-based"],
    default=["COCOMO"]
)

# Help guides
with st.sidebar.expander("📘 Help Guide"):
    st.markdown("""
    **Estimation Methods Explained:**

    - **COCOMO**: Uses code size (KLOC) to estimate effort.  
      Example: 5K lines → Effort ≈ 2.94*(5^1.1).
    - **Function Point (FP)**: Based on user-facing features.  
      Example: 10 features → FP = 50 → Effort = 125.
    - **Use Case Point (UCP)**: Uses number of use cases.  
      Example: 8 cases → Effort = 33.6.
    - **AI-based NLP**: Estimates based on requirement text length.  
      Example: 200 words → Effort = 20.
    - **Expert-based**: Manual assumption per feature.  
      Example: 12 features → Effort = 180.
    """)

st.subheader("📥 Input Project Requirements")
input_option = st.radio("Input type:", ["Paste Text", "Upload Document"])

requirements_text = ""
if input_option == "Paste Text":
    requirements_text = st.text_area("Paste your requirements here:", height=200)
else:
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file:
        requirements_text = uploaded_file.read().decode("utf-8")

if requirements_text:
    st.success("✅ Requirements received")
    features = extract_features(requirements_text)

    st.subheader("🔎 Identified Features")
    st.write(features)

    # Run estimation
    results = []
    for method in methods:
        if method == "COCOMO":
            res = cocomo_estimation(len(requirements_text.split()))
        elif method == "Function Point":
            res = fp_estimation(features)
        elif method == "Use Case Point":
            res = ucp_estimation(features)
        elif method == "AI-based NLP":
            res = ai_estimation(requirements_text)
        elif method == "Expert-based":
            res = expert_estimation(features)
        else:
            res = 0
        results.append((method, res))

    df_results = pd.DataFrame(results, columns=["Method", "Estimated Effort (Person-Hours)"])

    st.subheader("📊 Estimation Results")
    st.dataframe(df_results, use_container_width=True)

    # Charts
    fig, ax = plt.subplots()
    ax.bar(df_results["Method"], df_results["Estimated Effort (Person-Hours)"])
    ax.set_ylabel("Effort (Person-Hours)")
    ax.set_title("Effort Comparison")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    charts = {"Effort Comparison": buf.getvalue()}
    st.pyplot(fig)

    # Team plan
    st.subheader("👥 Suggested Team & Sprint Plan")
    avg_effort = df_results["Estimated Effort (Person-Hours)"].mean()
    team_size = max(2, int(avg_effort // 100))
    sprints = max(1, int(avg_effort // (team_size * 80)))
    team_plan = pd.DataFrame([
        ["Avg Effort", avg_effort],
        ["Recommended Team Size", team_size],
        ["Suggested Sprints", sprints]
    ], columns=["Metric", "Value"])
    st.table(team_plan)

    # Downloads
    df_dict = {"Estimation Results": df_results, "Team Plan": team_plan}
    xls_bytes = to_excel(df_dict)
    pdf_bytes = to_pdf("Effort Estimation Report", df_dict, charts)

    st.download_button("⬇️ Download as Excel", xls_bytes, "estimation.xlsx", "application/vnd.ms-excel")
    st.download_button("⬇️ Download as PDF", pdf_bytes, "estimation.pdf", "application/pdf")
