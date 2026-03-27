"""
╔══════════════════════════════════════════════════════════╗
║   Sales Lead Qualifier System  ·  MD02                  ║
║   Run:  streamlit run app.py                             ║
║   CSV:  leads_main_1000_final.csv  (same folder)         ║
╚══════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, f1_score
)
import io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Sales Lead Qualifier · MD02",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════
#  GLOBAL STYLES  — Poppins (headings) + Inter (body)
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding: 1.4rem 2rem 3rem; }
h1, h2, h3, h4 { font-family: 'Poppins', sans-serif !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #07111c !important;
    border-right: 1px solid #0f2237;
}
section[data-testid="stSidebar"] * { color: #94b8d4 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stRadio label {
    color: #4a7a9b !important; font-size: 0.68rem !important;
    text-transform: uppercase; letter-spacing: 0.09em; font-weight: 700;
}
section[data-testid="stSidebar"] .stButton>button {
    background: linear-gradient(135deg, #0f6fff 0%, #0050cc 100%) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-family: 'Poppins', sans-serif !important;
    padding: 0.7rem 1rem !important; letter-spacing: 0.02em !important;
    font-size: 0.85rem !important;
    box-shadow: 0 4px 16px rgba(15,111,255,0.35) !important;
    transition: all 0.2s ease !important;
}
section[data-testid="stSidebar"] .stButton>button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 22px rgba(15,111,255,0.52) !important;
}

/* ── Hero Header ── */
.main-header {
    background: linear-gradient(135deg, #07111c 0%, #0a1929 50%, #071629 100%);
    border: 1px solid #0f2a3d;
    border-radius: 18px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute; top: 0; right: 0;
    width: 340px; height: 100%;
    background: radial-gradient(ellipse at 80% 50%, rgba(15,111,255,0.09) 0%, transparent 70%);
    pointer-events: none;
}
.main-header .badge {
    display: inline-block;
    background: rgba(15,111,255,0.15);
    border: 1px solid rgba(15,111,255,0.3);
    color: #5aadff;
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; padding: 3px 11px; border-radius: 20px;
    margin-bottom: 8px; font-family: 'JetBrains Mono', monospace;
}
.main-header .acc-pill {
    display: inline-block;
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.25);
    color: #34d399;
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; padding: 3px 11px; border-radius: 20px;
    margin-left: 8px; font-family: 'JetBrains Mono', monospace;
}
.main-header h1 {
    font-family: 'Poppins', sans-serif !important;
    font-size: 1.75rem !important; font-weight: 800 !important;
    color: #e8f3ff !important; margin: 0 0 3px 0 !important;
    letter-spacing: -0.02em; line-height: 1.2;
}
.main-header .subtitle {
    color: #5aadff; font-size: 0.88rem; font-weight: 600;
    font-family: 'Poppins', sans-serif; margin: 0 0 4px 0;
}
.main-header .tagline {
    color: #4a7a9b; font-size: 0.77rem;
    margin: 0 0 12px 0; font-style: italic;
}
.main-header .feature-pills { display: flex; gap: 8px; flex-wrap: wrap; }
.main-header .fp {
    background: rgba(255,255,255,0.04);
    border: 1px solid #0f2a3d; color: #6a9fc0;
    font-size: 0.65rem; padding: 3px 10px;
    border-radius: 20px; font-weight: 500;
}

/* ── KPI Cards ── */
.kpi { background: #fff; border-radius: 14px; padding: 1rem 1.25rem;
       border: 1px solid #e8edf5; box-shadow: 0 2px 10px rgba(0,0,0,0.04); }
.kpi .k-icon { font-size: 1.2rem; margin-bottom: 4px; }
.kpi .k-label { font-size: 0.61rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #9ab0c5; margin-bottom: 5px; }
.kpi .k-val { font-size: 1.9rem; font-weight: 800; color: #07111c;
    font-family: 'Poppins', sans-serif; line-height: 1; }
.kpi .k-sub { font-size: 0.67rem; color: #9ab0c5; margin-top: 4px; }
.kpi.high   { border-top: 3px solid #ef4444; }
.kpi.medium { border-top: 3px solid #f59e0b; }
.kpi.low    { border-top: 3px solid #10b981; }
.kpi.blue   { border-top: 3px solid #3b82f6; }
.kpi.purple { border-top: 3px solid #8b5cf6; }
.kpi.teal   { border-top: 3px solid #06b6d4; }

/* ── Section titles ── */
.sec-title {
    font-family: 'Poppins', sans-serif;
    font-size: 0.64rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.13em; color: #9ab0c5;
    margin: 1.5rem 0 0.8rem; display: flex; align-items: center; gap: 8px;
}
.sec-title::after { content:''; flex:1; height:1px; background:#e8edf5; }

/* ── Info / Alert boxes ── */
.ibox { padding: 0.85rem 1.1rem; border-radius: 10px; font-size: 0.82rem;
    line-height: 1.55; margin-bottom: 0.75rem; }
.ibox.blue   { background:#eff6ff; border-left:4px solid #3b82f6; color:#1e40af; }
.ibox.amber  { background:#fffbeb; border-left:4px solid #f59e0b; color:#92400e; }
.ibox.green  { background:#f0fdf4; border-left:4px solid #10b981; color:#065f46; }
.ibox.red    { background:#fef2f2; border-left:4px solid #ef4444; color:#991b1b; }
.ibox.dark   { background:#07111c; border-left:4px solid #3b82f6; color:#94b8d4; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px; background: #f1f5f9; border-radius: 12px; padding: 4px; border: none;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important; padding: 7px 18px !important;
    font-size: 0.78rem !important; font-weight: 600 !important;
    color: #7a9ab5 !important; background: transparent !important;
    border: none !important; font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.01em;
}
.stTabs [aria-selected="true"] {
    background: #fff !important; color: #0f6fff !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.1) !important;
}

/* ── Prediction result card ── */
.pred-result {
    background: linear-gradient(135deg, #07111c, #0a1929);
    border: 1px solid #0f2a3d; border-radius: 16px; padding: 1.6rem; margin-top: 1rem;
}
.pred-result .tier {
    font-family: 'Poppins', sans-serif; font-size: 2rem; font-weight: 800;
    letter-spacing: -0.02em;
}
.pred-result .score-bar-wrap { background: #0f2237; border-radius: 8px; height: 8px; margin: 8px 0; }
.pred-result .score-bar { height: 8px; border-radius: 8px; }
.pred-result .conf-badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.06em;
    font-family: 'JetBrains Mono', monospace; margin-left: 8px; vertical-align: middle;
}
.conf-high   { background:rgba(16,185,129,0.15); color:#34d399; border:1px solid rgba(16,185,129,0.3); }
.conf-medium { background:rgba(245,158,11,0.15); color:#fbbf24; border:1px solid rgba(245,158,11,0.3); }
.conf-low    { background:rgba(239,68,68,0.15);  color:#f87171; border:1px solid rgba(239,68,68,0.3); }

/* ── Risk warning ── */
.risk-warn {
    background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3);
    color: #d97706; border-radius: 8px; padding: 7px 12px;
    font-size: 0.78rem; font-weight: 600; margin-top: 8px; display: block;
}

/* ── Follow-up action card ── */
.fu-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1rem 1.2rem; margin-top: 0.5rem;
}
.fu-card .fu-title {
    font-family: 'Poppins', sans-serif; font-size: 0.95rem;
    font-weight: 700; color: #07111c; margin-bottom: 8px;
}
.fu-card .fu-row {
    display: flex; gap: 24px; font-size: 0.8rem; color: #64748b; flex-wrap: wrap;
}
.fu-card .fu-row span strong { color: #0f172a; }

/* ── Tooltip hint ── */
.tooltip-info { font-size: 0.68rem; color: #9ab0c5; font-style: italic; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════
PRIORITY_COLORS = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}
SAMPLE_CSV = "leads_main_1000_final.csv"

FOLLOW_UP_CONFIG = {
    "High":   {"icon": "🔴", "timeline": "Within 24 hours",
               "channel": "Phone Call + Email",  "days": 1,
               "msg": "Immediate outreach required. Highest conversion potential — act now before lead goes cold.",
               "ibox": "red"},
    "Medium": {"icon": "🟡", "timeline": "Within 3–5 days",
               "channel": "Email + LinkedIn",    "days": 4,
               "msg": "Schedule follow-up this week. Nurture with relevant content and a soft call-to-action.",
               "ibox": "amber"},
    "Low":    {"icon": "🟢", "timeline": "Within 2 weeks",
               "channel": "Email Newsletter",    "days": 14,
               "msg": "Add to drip campaign. Monitor engagement signals before direct outreach.",
               "ibox": "green"},
}

CAT_COLS = [
    "email_response", "cart_activity", "last_activity", "budget_level",
    "company_size", "previous_interaction", "previous_outcome", "lead_source",
]
NUM_FEATS = [
    "number_of_visits", "time_spent_on_website", "engagement_score",
    "click_rate", "email_open_rate", "inactivity_period",
]
KNOWN_VALUES = {
    "email_response":       ["Clicked", "Replied", "No_Response", "Opened"],
    "cart_activity":        ["Yes", "No"],
    "last_activity":        ["Content_Download", "Webinar", "Demo_Request",
                             "Page_View", "Pricing_Page", "Form_Submit"],
    "budget_level":         ["Low", "Medium", "High"],
    "company_size":         ["Small", "Medium", "Large", "Enterprise"],
    "previous_interaction": ["Yes", "No"],
    "previous_outcome":     ["In_Progress", "No_Contact", "No_Response",
                             "Success", "Failure"],
    "lead_source":          ["Social", "Referral", "Paid_Ads",
                             "Website", "Email_Campaign", "Organic_Search"],
}

# ══════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════
_DEFS = {
    "trained": False, "df": None, "model": None, "scaler": None,
    "encoders": {}, "feature_cols": [], "target_le": None,
    "report": None, "cm": None, "auc": None, "accuracy": None,
    "model_name": None, "logs": [], "trigger_train": False,
}
for k, v in _DEFS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════
#  CORE ML FUNCTIONS
# ══════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    for col in ["converted", "lead_score", "recommended_followup_days",
                "conversion_likelihood"]:
        if col in df.columns:
            df = df.drop(col, axis=1)
    encoders = {}
    for col in CAT_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    feat_cols = (
        [c for c in NUM_FEATS if c in df.columns] +
        [c + "_enc" for c in CAT_COLS if c in df.columns]
    )
    return df, encoders, feat_cols


def train_model(df: pd.DataFrame, feat_cols: list, model_name: str):
    le_target = LabelEncoder()
    y  = le_target.fit_transform(df["lead_category"].astype(str))
    X  = df[feat_cols].fillna(0)
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(
        Xs, y, test_size=0.2, random_state=42, stratify=y
    )
    mdl_map = {
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }
    mdl = mdl_map[model_name]
    mdl.fit(Xtr, ytr)
    yp    = mdl.predict(Xte)
    yprob = mdl.predict_proba(Xte)
    acc   = accuracy_score(yte, yp)
    rep   = classification_report(yte, yp, target_names=le_target.classes_, output_dict=True)
    cm    = confusion_matrix(yte, yp)
    try:
        auc = (roc_auc_score(yte, yprob[:, 1]) if len(le_target.classes_) == 2
               else roc_auc_score(yte, yprob, multi_class="ovr", average="macro"))
    except Exception:
        auc = None
    return mdl, sc, le_target, rep, cm, auc, acc


def predict_all(df: pd.DataFrame, mdl, sc, feat_cols: list, le_target):
    df     = df.copy()
    Xs     = sc.transform(df[feat_cols].fillna(0))
    preds  = mdl.predict(Xs)
    probas = mdl.predict_proba(Xs)
    df["predicted_category"]    = le_target.inverse_transform(preds)
    df["conversion_likelihood"] = probas.max(axis=1).round(4)
    df["lead_score"]            = (df["conversion_likelihood"] * 100).round(0).astype(int)
    return df


def add_followup(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _logic(score):
        if   score >= 70: return ("🔴 Immediate (within 24 hrs)", "Phone Call + Email", 1)
        elif score >= 40: return ("🟡 Within 3–5 days",           "Email + LinkedIn",   4)
        else:             return ("🟢 Within 2 weeks",             "Email Newsletter",   14)

    df["followup_timeline"] = df["lead_score"].apply(lambda s: _logic(s)[0])
    df["followup_channel"]  = df["lead_score"].apply(lambda s: _logic(s)[1])
    _days                   = df["lead_score"].apply(lambda s: _logic(s)[2])
    df["followup_date"]     = _days.apply(
        lambda d: (datetime.today() + timedelta(days=int(d))).strftime("%Y-%m-%d")
    )
    return df


def predict_single(inputs: dict, mdl, sc, encoders, feat_cols, le_target):
    row = {}
    for col, le in encoders.items():
        val = inputs.get(col, le.classes_[0])
        try:    row[col + "_enc"] = int(le.transform([str(val)])[0])
        except: row[col + "_enc"] = 0
    for col in NUM_FEATS:
        row[col] = float(inputs.get(col, 0))
    if "engagement_score" in feat_cols and "engagement_score" not in inputs:
        row["engagement_score"] = (row.get("number_of_visits", 0) *
                                   row.get("time_spent_on_website", 0))
    X  = pd.DataFrame([{k: row.get(k, 0) for k in feat_cols}])
    Xs = sc.transform(X)
    pred      = mdl.predict(Xs)[0]
    probas    = mdl.predict_proba(Xs)[0]
    label     = le_target.inverse_transform([pred])[0]
    prob_dict = {le_target.inverse_transform([i])[0]: round(float(p) * 100, 1)
                 for i, p in enumerate(probas)}
    score = int(max(probas) * 100)
    return label, prob_dict, score


def explain_lead(inputs: dict) -> list:
    reasons = []
    v    = inputs.get("number_of_visits", 0)
    t    = inputs.get("time_spent_on_website", 0)
    inact = inputs.get("inactivity_period", 99)
    if v >= 10: reasons.append(f"Very high visit frequency ({v} visits) — strong interest signal")
    elif v >= 5: reasons.append(f"Good visit frequency ({v} visits)")
    if t >= 20: reasons.append(f"Spends significant time on site ({t:.0f} min)")
    if inputs.get("cart_activity") == "Yes":
        reasons.append("Added items to cart — strong purchase intent 🛒")
    if inputs.get("email_response") == "Replied":
        reasons.append("Actively replied to emails — highly engaged prospect")
    elif inputs.get("email_response") == "Opened":
        reasons.append("Opens emails — shows interest")
    last = inputs.get("last_activity", "")
    if last in ["Demo_Request", "Pricing_Page", "Form_Submit"]:
        reasons.append(f"High-intent last activity: {last.replace('_',' ')}")
    if inputs.get("budget_level") == "High":
        reasons.append("High budget level — strong conversion capability")
    if inputs.get("previous_outcome") == "Success":
        reasons.append("Previous interaction ended successfully — warm prospect")
    if inact <= 7:
        reasons.append(f"Recently active (only {inact} days inactive) — lead is hot")
    if not reasons:
        reasons = ["Low engagement signals across all behavioral dimensions"]
    return reasons


def get_confidence(score: int):
    if   score >= 75: return "High Confidence",     "conf-high"
    elif score >= 45: return "Moderate Confidence", "conf-medium"
    else:             return "Low Confidence",       "conf-low"


def get_risk(inputs: dict):
    inact = inputs.get("inactivity_period", 0)
    if inact >= 45:
        return f"⚠️ Risk: Lead may be cold — {inact} days of inactivity detected"
    if (inputs.get("email_response") == "No_Response" and
            inputs.get("cart_activity") == "No"):
        return "⚠️ Risk: No email engagement and no cart activity — low intent signals"
    return None


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="All Leads")
        for tier in ["High", "Medium", "Low"]:
            if "lead_category" in df.columns:
                sub = df[df["lead_category"] == tier]
                if len(sub):
                    sub.to_excel(w, index=False, sheet_name=f"{tier} Priority")
    return buf.getvalue()

# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem'>
      <div style='font-family:Poppins,sans-serif;font-size:1.1rem;font-weight:700;
                  color:#e8f3ff;'>🎯 Lead Qualifier</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:0.6rem;
                  color:#1a5a8a;margin-top:2px;letter-spacing:0.1em;'>MD02 · SALES AI</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Data Source**")
    src_mode = st.radio("", ["Sample Dataset", "Upload CSV"], label_visibility="collapsed")
    uploaded = None
    if src_mode == "Upload CSV":
        uploaded = st.file_uploader("Drop CSV here", type=["csv"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**ML Algorithm**")
    model_choice = st.selectbox(
        "", ["Random Forest", "Gradient Boosting", "Logistic Regression"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Filters**")
    filter_priority = st.multiselect(
        "Priority Tiers", ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"], label_visibility="collapsed"
    )
    filter_score = st.slider("Min Lead Score", 0, 100, 0, 5, label_visibility="collapsed")

    st.markdown("---")
    run_btn = st.button("🚀  Analyze Leads", use_container_width=True, type="primary")
    if run_btn:
        st.session_state.trigger_train = True

    if st.session_state.trained:
        st.markdown("---")
        acc = st.session_state.accuracy or 0
        auc = st.session_state.auc or 0
        st.markdown(f"""
        <div style='background:#0a1929;border-radius:10px;padding:0.85rem 1rem;
                    border:1px solid #0f2a3d;'>
          <div style='font-size:0.58rem;text-transform:uppercase;letter-spacing:0.1em;
                      color:#1a5a8a;font-weight:700;margin-bottom:8px;'>Model Status · Active</div>
          <div style='display:flex;justify-content:space-between;margin-bottom:5px'>
            <span style='font-size:0.72rem;color:#4a7a9b;'>Accuracy</span>
            <span style='font-family:JetBrains Mono,monospace;font-size:0.75rem;
                         color:#34d399;font-weight:600;'>{acc*100:.1f}%</span>
          </div>
          <div style='display:flex;justify-content:space-between;margin-bottom:5px'>
            <span style='font-size:0.72rem;color:#4a7a9b;'>AUC-ROC</span>
            <span style='font-family:JetBrains Mono,monospace;font-size:0.75rem;
                         color:#5aadff;font-weight:600;'>{auc:.4f}</span>
          </div>
          <div style='display:flex;justify-content:space-between;'>
            <span style='font-size:0.72rem;color:#4a7a9b;'>Algorithm</span>
            <span style='font-size:0.7rem;color:#7aadcc;font-weight:500;'>
              {st.session_state.model_name or "—"}</span>
          </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_sample():
    df = pd.read_csv(SAMPLE_CSV)
    df.columns = df.columns.str.strip().str.lower()
    return df

df_raw = None
if src_mode == "Upload CSV" and uploaded:
    try:
        df_raw = pd.read_csv(uploaded)
        df_raw.columns = df_raw.columns.str.strip().str.lower()
    except Exception as e:
        st.error(f"Could not parse file: {e}")
elif src_mode == "Sample Dataset":
    try:
        df_raw = load_sample()
    except FileNotFoundError:
        st.error("⚠️ Sample file not found. Place `leads_main_1000_final.csv` in the same folder.")

# ══════════════════════════════════════════════════════════
#  TRAIN
# ══════════════════════════════════════════════════════════
if st.session_state.trigger_train and df_raw is not None:
    with st.spinner("Analyzing lead behavior patterns…"):
        df_p, enc, fcols = preprocess(df_raw)
        mdl, sc, le_t, rep, cm, auc, acc = train_model(df_p, fcols, model_choice)
        df_p = predict_all(df_p, mdl, sc, fcols, le_t)
        df_p = add_followup(df_p)
        st.session_state.update({
            "trained": True, "df": df_p, "model": mdl, "scaler": sc,
            "encoders": enc, "feature_cols": fcols, "target_le": le_t,
            "report": rep, "cm": cm, "auc": auc, "accuracy": acc,
            "model_name": model_choice, "trigger_train": False,
        })
    st.success(f"✅ Model trained! Accuracy: {acc*100:.1f}%")

# ══════════════════════════════════════════════════════════
#  HERO HEADER
# ══════════════════════════════════════════════════════════
acc_pill = (f'<span class="acc-pill">✓ {st.session_state.accuracy*100:.1f}% Accuracy</span>'
            if st.session_state.trained else "")
st.markdown(f"""
<div class="main-header">
  <div style="display:flex;align-items:flex-start;justify-content:space-between">
    <div>
      <div class="badge">MD02</div>{acc_pill}
      <h1>Sales Lead Qualifier System</h1>
      <p class="subtitle">AI-Powered Sales Intelligence Platform</p>
      <p class="tagline">Prioritize leads. Predict conversions. Maximize sales efficiency.</p>
      <div class="feature-pills">
        <span class="fp">✔ Real-time Lead Scoring</span>
        <span class="fp">✔ AI-driven Prioritization</span>
        <span class="fp">✔ Smart Follow-up Recommendations</span>
      </div>
    </div>
    <div style="font-size:3.5rem;opacity:0.08;line-height:1;padding-top:4px">🎯</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Pre-train states
if df_raw is None:
    st.markdown('<div class="ibox blue">👈 Select a data source in the sidebar and click <strong>🚀 Analyze Leads</strong> to get started.</div>', unsafe_allow_html=True)
    st.stop()

if not st.session_state.trained:
    st.markdown(
        f'<div class="ibox amber">📊 Data loaded — '
        f'<strong>{len(df_raw):,} rows · {len(df_raw.columns)} features</strong>. '
        f'Click <strong>🚀 Analyze Leads</strong> in the sidebar to begin.</div>',
        unsafe_allow_html=True
    )
    with st.expander("👁  Preview raw data"):
        st.dataframe(df_raw.head(30), use_container_width=True)
    st.stop()

# Filtered view
df: pd.DataFrame = st.session_state.df
df_f = df[
    (df["lead_category"].isin(filter_priority)) &
    (df["lead_score"] >= filter_score)
].copy()

# ══════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════
T = st.tabs([
    "📊 Overview",
    "🏆 Priority Leads",
    "🔮 New Lead",
    "🤖 Model Insights",
    "📅 Action Plan",
    "📝 Activity Log",
    "📤 Export Data",
])

# ══════════════════════════════════════════════════════════
#  TAB 1 · OVERVIEW
# ══════════════════════════════════════════════════════════
with T[0]:
    total = len(df_f)
    if total == 0:
        st.markdown(
            '<div class="ibox amber">⚠️ No leads match the current filters. '
            'Try adjusting the priority tiers or minimum lead score in the sidebar.</div>',
            unsafe_allow_html=True
        )
        st.stop()

    n_high    = int((df_f["lead_category"] == "High").sum())
    n_med     = int((df_f["lead_category"] == "Medium").sum())
    n_low     = int((df_f["lead_category"] == "Low").sum())
    avg_score = df_f["lead_score"].mean()
    avg_eng   = df_f["engagement_score"].mean() if "engagement_score" in df_f.columns else 0
    avg_conv  = df_f["conversion_likelihood"].mean() * 100 if "conversion_likelihood" in df_f.columns else 0
    pct = lambda n: f"{n/total*100:.1f}%"

    st.markdown(
        '<div class="ibox dark" style="margin-bottom:1rem;">'
        'Built to solve real-world sales inefficiency — this system prioritizes high-value leads '
        'using AI-driven behavioral analysis, so sales teams focus effort where it matters most.'
        '</div>', unsafe_allow_html=True
    )

    # KPI Row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for col, css, icon, lbl, val, sub in [
        (c1, "blue",   "📊", "Total Leads",         f"{total:,}",       "current filter"),
        (c2, "high",   "🔴", "High Value Leads",    f"{n_high:,}",      pct(n_high)),
        (c3, "medium", "🟡", "Potential Leads",     f"{n_med:,}",       pct(n_med)),
        (c4, "low",    "🟢", "Nurture Leads",       f"{n_low:,}",       pct(n_low)),
        (c5, "purple", "🎯", "Avg Score",           f"{avg_score:.0f}", "out of 100"),
        (c6, "teal",   "📈", "Expected Conv. Rate", f"{avg_conv:.0f}%", "model estimate"),
    ]:
        col.markdown(
            f'<div class="kpi {css}">'
            f'<div class="k-icon">{icon}</div>'
            f'<div class="k-label">{lbl}</div>'
            f'<div class="k-val">{val}</div>'
            f'<div class="k-sub">{sub}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("")

    # Row 1
    r1a, r1b = st.columns(2)
    with r1a:
        fig = px.pie(df_f, names="lead_category", title="Lead Quality Breakdown",
                     color="lead_category", color_discrete_map=PRIORITY_COLORS, hole=0.58)
        fig.update_traces(textposition="outside", textinfo="percent+label", pull=[0.04]*3)
        fig.update_layout(height=310, margin=dict(t=50,b=10,l=10,r=10), showlegend=False,
                          font=dict(family="Inter"), title_font=dict(family="Poppins", size=14))
        st.plotly_chart(fig, use_container_width=True)
    with r1b:
        fig2 = px.histogram(df_f, x="engagement_score", color="lead_category",
                            nbins=30, title="User Engagement Behavior",
                            color_discrete_map=PRIORITY_COLORS,
                            labels={"engagement_score": "Engagement Score"})
        fig2.update_layout(height=310, margin=dict(t=50,b=10,l=10,r=10), bargap=0.06,
                           font=dict(family="Inter"), title_font=dict(family="Poppins", size=14),
                           legend_title_text="Priority")
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2
    r2a, r2b = st.columns(2)
    with r2a:
        if "lead_source" in df_f.columns:
            src = df_f.groupby(["lead_source","lead_category"]).size().reset_index(name="n")
            fig3 = px.bar(src, x="lead_source", y="n", color="lead_category",
                          title="Leads by Source & Priority", barmode="stack",
                          color_discrete_map=PRIORITY_COLORS,
                          labels={"lead_source":"Source","n":"Leads"})
            fig3.update_layout(height=290, margin=dict(t=50,b=10,l=10,r=10),
                               font=dict(family="Inter"), title_font=dict(family="Poppins",size=14))
            st.plotly_chart(fig3, use_container_width=True)
    with r2b:
        if "company_size" in df_f.columns:
            cs = df_f.groupby(["company_size","lead_category"]).size().reset_index(name="n")
            fig4 = px.bar(cs, x="company_size", y="n", color="lead_category",
                          title="Leads by Company Size", barmode="group",
                          color_discrete_map=PRIORITY_COLORS,
                          labels={"company_size":"Size","n":"Leads"},
                          category_orders={"company_size":["Small","Medium","Large","Enterprise"]})
            fig4.update_layout(height=290, margin=dict(t=50,b=10,l=10,r=10),
                               font=dict(family="Inter"), title_font=dict(family="Poppins",size=14))
            st.plotly_chart(fig4, use_container_width=True)

    # Row 3
    r3a, r3b = st.columns(2)
    with r3a:
        fig5 = px.box(df_f, x="lead_category", y="engagement_score", color="lead_category",
                      title="Engagement Score by Priority", color_discrete_map=PRIORITY_COLORS,
                      points="outliers", category_orders={"lead_category":["High","Medium","Low"]})
        fig5.update_layout(height=290, margin=dict(t=50,b=10,l=10,r=10), showlegend=False,
                           font=dict(family="Inter"), title_font=dict(family="Poppins",size=14))
        st.plotly_chart(fig5, use_container_width=True)
    with r3b:
        fig6 = px.violin(df_f, x="lead_category", y="lead_score", color="lead_category",
                         title="Lead Score Distribution by Priority",
                         color_discrete_map=PRIORITY_COLORS, box=True, points=False,
                         category_orders={"lead_category":["High","Medium","Low"]})
        fig6.update_layout(height=290, margin=dict(t=50,b=10,l=10,r=10), showlegend=False,
                           font=dict(family="Inter"), title_font=dict(family="Poppins",size=14))
        st.plotly_chart(fig6, use_container_width=True)

    # Row 4
    r4a, r4b = st.columns(2)
    with r4a:
        fig7 = px.box(df_f, x="lead_category", y="inactivity_period", color="lead_category",
                      title="Inactivity Period by Priority", color_discrete_map=PRIORITY_COLORS,
                      points="outliers", category_orders={"lead_category":["High","Medium","Low"]},
                      labels={"inactivity_period":"Days Inactive"})
        fig7.update_layout(height=290, margin=dict(t=50,b=10,l=10,r=10), showlegend=False,
                           font=dict(family="Inter"), title_font=dict(family="Poppins",size=14))
        st.plotly_chart(fig7, use_container_width=True)
    with r4b:
        if "budget_level" in df_f.columns:
            bl = df_f.groupby(["budget_level","lead_category"]).size().reset_index(name="n")
            fig8 = px.bar(bl, x="budget_level", y="n", color="lead_category",
                          title="Budget Level vs Priority", barmode="group",
                          color_discrete_map=PRIORITY_COLORS,
                          labels={"budget_level":"Budget","n":"Leads"},
                          category_orders={"budget_level":["Low","Medium","High"]})
            fig8.update_layout(height=290, margin=dict(t=50,b=10,l=10,r=10),
                               font=dict(family="Inter"), title_font=dict(family="Poppins",size=14))
            st.plotly_chart(fig8, use_container_width=True)

    # Scatter
    st.markdown('<div class="sec-title">📈 Engagement × Lead Score Scatter</div>',
                unsafe_allow_html=True)
    sc_size     = "number_of_visits" if "number_of_visits" in df_f.columns else None
    hover_extra = [c for c in ["company_size","lead_source","budget_level"] if c in df_f.columns]
    figsc = px.scatter(
        df_f.sample(min(500, len(df_f)), random_state=42),
        x="engagement_score", y="lead_score", color="lead_category",
        size=sc_size, opacity=0.6, hover_data=hover_extra,
        color_discrete_map=PRIORITY_COLORS,
        title="Engagement Score vs Lead Score  (bubble size = visits)",
        labels={"engagement_score":"Engagement Score","lead_score":"Lead Score (0–100)"}
    )
    figsc.update_layout(height=380, margin=dict(t=50,b=10,l=10,r=10),
                        font=dict(family="Inter"), legend_title_text="Priority",
                        title_font=dict(family="Poppins",size=14))
    st.plotly_chart(figsc, use_container_width=True)

# ══════════════════════════════════════════════════════════
#  TAB 2 · PRIORITY LEADS
# ══════════════════════════════════════════════════════════
with T[1]:
    st.markdown('<div class="sec-title">🏆 Prioritized Lead Rankings</div>',
                unsafe_allow_html=True)

    rc1, rc2, rc3 = st.columns([3, 2, 2])
    search  = rc1.text_input("🔍 Search", placeholder="Filter by any value…",
                             label_visibility="collapsed")
    sort_by = rc2.selectbox(
        "Sort by",
        ["lead_score","conversion_likelihood","engagement_score",
         "number_of_visits","time_spent_on_website"],
        label_visibility="collapsed"
    )
    top_n = rc3.selectbox("Show", [50, 100, 250, 500, "All"],
                          label_visibility="collapsed")

    SHOW_COLS = [
        "lead_category","predicted_category","lead_score","conversion_likelihood",
        "engagement_score","number_of_visits","time_spent_on_website",
        "email_open_rate","click_rate","company_size","budget_level",
        "lead_source","cart_activity","email_response",
        "followup_timeline","followup_channel","followup_date",
    ]
    SHOW_COLS = [c for c in SHOW_COLS if c in df_f.columns]

    df_rank = (df_f[SHOW_COLS].sort_values(sort_by, ascending=False)
               .reset_index(drop=True))
    df_rank.index += 1

    if search:
        mask = df_rank.astype(str).apply(
            lambda r: r.str.contains(search, case=False, na=False)
        ).any(axis=1)
        df_rank = df_rank[mask]

    if top_n != "All":
        df_rank = df_rank.head(int(top_n))

    def _style_priority(val):
        return {
            "High":   "background-color:#fef2f2;color:#dc2626;font-weight:700",
            "Medium": "background-color:#fffbeb;color:#d97706;font-weight:700",
            "Low":    "background-color:#f0fdf4;color:#16a34a;font-weight:700",
        }.get(str(val), "")

    style_cols = [c for c in ["lead_category","predicted_category"] if c in df_rank.columns]
    fmt = {}
    if "conversion_likelihood" in df_rank.columns: fmt["conversion_likelihood"] = "{:.1%}"
    if "lead_score"            in df_rank.columns: fmt["lead_score"]            = "{:.0f}"
    if "engagement_score"      in df_rank.columns: fmt["engagement_score"]      = "{:.1f}"

    styled = df_rank.style.format(fmt)
    for col in style_cols:
        styled = styled.applymap(_style_priority, subset=[col])
    if "lead_score" in df_rank.columns:
        styled = styled.background_gradient(subset=["lead_score"], cmap="RdYlGn", vmin=0, vmax=100)

    st.dataframe(styled, use_container_width=True, height=520)
    st.caption(f"Showing {len(df_rank):,} leads · sorted by {sort_by}")
    st.markdown(
        '<div class="tooltip-info">💡 Lead Score represents the probability of conversion '
        'combined with behavioral engagement signals, scored 0–100.</div>',
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════
#  TAB 3 · NEW LEAD (Single Predict)
# ══════════════════════════════════════════════════════════
with T[2]:
    st.markdown('<div class="sec-title">🔮 Single Lead Prediction</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="ibox blue">Enter lead attributes and click '
        '<strong>🔮 Generate Prediction</strong> for an instant AI classification.</div>',
        unsafe_allow_html=True
    )

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Engagement Signals**")
            visits   = st.slider("Number of Visits",      1,    20,   6)
            time_sp  = st.slider("Time Spent on Site (min)", 1.0, 60.0, 15.0, 0.5)
            inact    = st.slider("Inactivity Period (days)", 1,   75,   20)
            click_r  = st.slider("Click Rate",            0.0, 1.0,  0.4,  0.05)
            email_or = st.slider("Email Open Rate",       0.0, 1.0,  0.35, 0.05)
        with col2:
            st.markdown("**Behavioral Data**")
            email_resp = st.selectbox("Email Response",     KNOWN_VALUES["email_response"])
            cart       = st.selectbox("Cart Activity",      KNOWN_VALUES["cart_activity"])
            last_act   = st.selectbox("Last Activity",      KNOWN_VALUES["last_activity"])
            prev_int   = st.selectbox("Previous Interaction", KNOWN_VALUES["previous_interaction"])
            prev_out   = st.selectbox("Previous Outcome",   KNOWN_VALUES["previous_outcome"])
        with col3:
            st.markdown("**Business Profile**")
            budget   = st.selectbox("Budget Level",  KNOWN_VALUES["budget_level"])
            company  = st.selectbox("Company Size",  KNOWN_VALUES["company_size"])
            lead_src = st.selectbox("Lead Source",   KNOWN_VALUES["lead_source"])

        submitted = st.form_submit_button(
            "🔮  Generate Prediction", type="primary", use_container_width=True
        )

    if submitted:
        inputs = {
            "number_of_visits":      visits,
            "time_spent_on_website": time_sp,
            "engagement_score":      visits * time_sp,
            "click_rate":            click_r,
            "email_open_rate":       email_or,
            "inactivity_period":     inact,
            "email_response":        email_resp,
            "cart_activity":         cart,
            "last_activity":         last_act,
            "budget_level":          budget,
            "company_size":          company,
            "previous_interaction":  prev_int,
            "previous_outcome":      prev_out,
            "lead_source":           lead_src,
        }
        label, prob_dict, score = predict_single(
            inputs, st.session_state.model, st.session_state.scaler,
            st.session_state.encoders, st.session_state.feature_cols,
            st.session_state.target_le
        )
        reasons          = explain_lead(inputs)
        risk_msg         = get_risk(inputs)
        conf_label, conf_cls = get_confidence(score)

        color_map  = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}
        tier_color = color_map.get(label, "#888")
        fu_chan  = {"High": "Phone Call + Email", "Medium": "Email + LinkedIn", "Low": "Email Newsletter"}
        fu_time  = {"High": "Within 24 hours",   "Medium": "Within 3–5 days",  "Low": "Within 2 weeks"}
        fu_days  = {"High": 1,                   "Medium": 4,                  "Low": 14}
        fu_date  = (datetime.today() + timedelta(days=fu_days.get(label, 7))).strftime("%B %d, %Y")
        top_prob = max(prob_dict.values())

        prob_sorted = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
        prob_bars   = ""
        for tier, pv in prob_sorted.items():
            c = color_map.get(tier, "#888")
            prob_bars += f"""
            <div style='margin-bottom:9px'>
              <div style='display:flex;justify-content:space-between;margin-bottom:3px'>
                <span style='font-size:0.75rem;color:#94b8d4;'>{tier}</span>
                <span style='font-family:JetBrains Mono,monospace;font-size:0.75rem;
                             color:{c};font-weight:600;'>{pv:.1f}%</span>
              </div>
              <div style='background:#0f2237;border-radius:4px;height:6px;'>
                <div style='width:{pv}%;height:6px;border-radius:4px;background:{c};'></div>
              </div>
            </div>"""

        res_col, det_col = st.columns([1, 1.2])

        with res_col:
            st.markdown(f"""
            <div class="pred-result">
              <div style='color:#4a7a9b;font-size:0.62rem;text-transform:uppercase;
                           letter-spacing:0.13em;font-weight:700;margin-bottom:4px;'>
                Prediction Result
              </div>
              <div style='display:flex;align-items:center;gap:10px;margin-bottom:4px;'>
                <span class="tier" style='color:{tier_color};'>{label} Priority</span>
                <span class="conf-badge {conf_cls}">{conf_label}</span>
              </div>
              <div class='score-bar-wrap'>
                <div class='score-bar' style='width:{score}%;background:{tier_color};'></div>
              </div>
              <div style='display:flex;justify-content:space-between;margin-bottom:18px;'>
                <span style='font-size:0.7rem;color:#1a5a8a;'>Lead Score</span>
                <span style='font-family:JetBrains Mono,monospace;font-size:0.82rem;
                             color:{tier_color};font-weight:700;'>{score} / 100</span>
              </div>

              <div style='font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;
                           color:#1a5a8a;font-weight:700;margin-bottom:6px;'>
                🔥 Conversion Likelihood
              </div>
              <div style='font-size:1.6rem;font-family:Poppins,sans-serif;font-weight:700;
                           color:{tier_color};margin-bottom:2px;'>{top_prob:.0f}%</div>
              <div style='font-size:0.7rem;color:#2a5a7a;margin-bottom:14px;'>
                probability of conversion
              </div>

              <div style='font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;
                           color:#1a5a8a;font-weight:700;margin-bottom:8px;'>
                Breakdown by Class
              </div>
              {prob_bars}

              <hr style='border:none;border-top:1px solid #0f2237;margin:14px 0;'>
              <div style='font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;
                           color:#1a5a8a;font-weight:700;margin-bottom:8px;'>
                🚀 Recommended Action
              </div>
              <div style='background:#0a1929;border-radius:10px;padding:0.8rem 1rem;
                           border:1px solid #0f2237;'>
                <div style='font-size:0.88rem;color:#e8f3ff;font-weight:600;
                             font-family:Poppins,sans-serif;margin-bottom:8px;'>
                  {fu_chan.get(label)}
                </div>
                <div style='font-size:0.73rem;color:#4a7a9b;margin-bottom:3px;'>
                  ⏰ Timeline: <strong style='color:#94b8d4;'>{fu_time.get(label)}</strong>
                </div>
                <div style='font-size:0.73rem;color:#4a7a9b;'>
                  📅 Follow up by: <strong style='color:#5aadff;'>{fu_date}</strong>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with det_col:
            st.markdown("**🧠 Why this prediction?**")
            for r in reasons:
                icon = "⚠️" if any(w in r.lower() for w in ["risk","cold","warning"]) else "✅"
                st.markdown(f"- {icon} {r}")

            if risk_msg:
                st.markdown(f'<div class="risk-warn">{risk_msg}</div>',
                            unsafe_allow_html=True)

            st.markdown("**📊 Input Summary**")
            summary = {
                "Feature": ["Visits","Time on Site","Inactivity","Cart",
                             "Email Response","Last Activity","Budget","Company"],
                "Value":   [visits, f"{time_sp} min", f"{inact} days", cart,
                             email_resp, last_act.replace("_"," "), budget, company],
            }
            st.dataframe(pd.DataFrame(summary).set_index("Feature"), use_container_width=True)

            st.markdown(
                '<div class="tooltip-info">💡 Lead Score represents the probability of '
                'conversion combined with behavioral engagement signals, scored 0–100.</div>',
                unsafe_allow_html=True
            )

# ══════════════════════════════════════════════════════════
#  TAB 4 · MODEL INSIGHTS
# ══════════════════════════════════════════════════════════
with T[3]:
    rep   = st.session_state.report
    cm    = st.session_state.cm
    auc   = st.session_state.auc
    acc   = st.session_state.accuracy
    mdl   = st.session_state.model
    fcols = st.session_state.feature_cols
    le_t  = st.session_state.target_le

    st.markdown('<div class="sec-title">🤖 Model Performance Metrics</div>',
                unsafe_allow_html=True)

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Algorithm",     st.session_state.model_name or "—")
    mc2.metric("Accuracy",      f"{acc*100:.1f}%" if acc else "—")
    mc3.metric("AUC-ROC",       f"{auc:.4f}"       if auc else "—")
    mc4.metric("Test Split",    "20% holdout")
    mc5.metric("Train Samples", f"{int(len(df)*0.8):,}")

    st.markdown("")
    sub1, sub2, sub3 = st.tabs(["Feature Importance","Confusion Matrix","Classification Report"])

    with sub1:
        fi_vals = (mdl.feature_importances_ if hasattr(mdl, "feature_importances_")
                   else np.abs(mdl.coef_).mean(axis=0) if hasattr(mdl, "coef_") else None)
        if fi_vals is not None:
            fi = pd.DataFrame({"Feature": fcols, "Importance": fi_vals})
            fi = fi.sort_values("Importance", ascending=True).tail(20)
            fi["Feature"] = fi["Feature"].str.replace("_enc", "", regex=False)
            figfi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                           title="Top Feature Importances",
                           color="Importance", color_continuous_scale="Blues")
            figfi.update_layout(height=520, margin=dict(t=50,b=10,l=10,r=10),
                                coloraxis_showscale=False, font=dict(family="Inter"),
                                title_font=dict(family="Poppins",size=14))
            st.plotly_chart(figfi, use_container_width=True)

    with sub2:
        class_names = le_t.classes_.tolist()
        figcm = px.imshow(cm, text_auto=True, aspect="auto", x=class_names, y=class_names,
                          title="Confusion Matrix", color_continuous_scale="Blues",
                          labels=dict(x="Predicted",y="Actual"))
        figcm.update_layout(height=420, font=dict(family="Inter"),
                            title_font=dict(family="Poppins",size=14))
        st.plotly_chart(figcm, use_container_width=True)

    with sub3:
        rows = [{"Class":k,"Precision":round(v["precision"],3),"Recall":round(v["recall"],3),
                 "F1-Score":round(v["f1-score"],3),"Support":int(v.get("support",0))}
                for k, v in rep.items() if isinstance(v, dict)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown('<div class="sec-title">📈 Feature Space: Engagement vs Lead Score</div>',
                unsafe_allow_html=True)
    figsc2 = px.scatter(
        df_f.sample(min(600, len(df_f)), random_state=1),
        x="engagement_score", y="lead_score", color="lead_category", opacity=0.55,
        color_discrete_map=PRIORITY_COLORS, marginal_x="histogram", marginal_y="histogram",
        title="Engagement Score vs Lead Score",
        labels={"engagement_score":"Engagement Score","lead_score":"Lead Score"}
    )
    figsc2.update_layout(height=460, font=dict(family="Inter"),
                         title_font=dict(family="Poppins",size=14))
    st.plotly_chart(figsc2, use_container_width=True)

# ══════════════════════════════════════════════════════════
#  TAB 5 · ACTION PLAN
# ══════════════════════════════════════════════════════════
with T[4]:
    st.markdown('<div class="sec-title">📅 Recommended Action Plan by Priority</div>',
                unsafe_allow_html=True)

    for tier in ["High", "Medium", "Low"]:
        if tier not in filter_priority:
            continue
        cfg = FOLLOW_UP_CONFIG[tier]
        sub = df_f[df_f["lead_category"] == tier]
        cnt = len(sub)
        if cnt == 0:
            continue
        avg_sc = sub["engagement_score"].mean() if "engagement_score" in sub.columns else 0
        avg_ls = sub["lead_score"].mean()

        with st.expander(
            f"{cfg['icon']}  **{tier} Priority** — {cnt:,} leads · "
            f"avg score {avg_ls:.0f} · avg engagement {avg_sc:.0f}",
            expanded=(tier == "High"),
        ):
            st.markdown(f"""
            <div class="fu-card">
              <div class="fu-title">🚀 Recommended Action: {cfg['channel']}</div>
              <div class="fu-row">
                <span>⏰ Timeline: <strong>{cfg['timeline']}</strong></span>
                <span>📡 Channel: <strong>{cfg['channel']}</strong></span>
              </div>
              <p style='margin:8px 0 0;font-size:0.8rem;color:#64748b;'>{cfg['msg']}</p>
            </div>""", unsafe_allow_html=True)

            lc1, lc2 = st.columns(2)
            with lc1:
                top_cols = [c for c in ["lead_score","conversion_likelihood","engagement_score",
                    "company_size","lead_source","budget_level","followup_date"] if c in sub.columns]
                top8 = (sub.sort_values("lead_score", ascending=False)[top_cols]
                        .head(8).reset_index(drop=True))
                top8.index += 1
                st.markdown("**Top leads in this tier**")
                fmt_fu = {}
                if "conversion_likelihood" in top8.columns: fmt_fu["conversion_likelihood"] = "{:.1%}"
                if "lead_score"            in top8.columns: fmt_fu["lead_score"]            = "{:.0f}"
                st.dataframe(top8.style.format(fmt_fu), use_container_width=True)
            with lc2:
                if "lead_source" in sub.columns:
                    src_cnt = sub["lead_source"].value_counts().reset_index()
                    src_cnt.columns = ["Source","Count"]
                    figs = px.bar(src_cnt, x="Source", y="Count", title="Source breakdown",
                                  color_discrete_sequence=[PRIORITY_COLORS[tier]])
                    figs.update_layout(height=270, margin=dict(t=35,b=5,l=5,r=5),
                                       showlegend=False, font=dict(family="Inter"))
                    st.plotly_chart(figs, use_container_width=True)

    st.markdown('<div class="sec-title">📆 Follow-up Calendar</div>', unsafe_allow_html=True)
    if "followup_date" in df_f.columns:
        tl = df_f.groupby(["followup_date","lead_category"]).size().reset_index(name="n")
        figt = px.bar(tl, x="followup_date", y="n", color="lead_category",
                      title="Leads Due per Follow-up Date",
                      color_discrete_map=PRIORITY_COLORS,
                      labels={"followup_date":"Date","n":"Leads"})
        figt.update_layout(height=290, margin=dict(t=50,b=10,l=10,r=10),
                           font=dict(family="Inter"), title_font=dict(family="Poppins",size=14))
        st.plotly_chart(figt, use_container_width=True)

# ══════════════════════════════════════════════════════════
#  TAB 6 · ACTIVITY LOG
# ══════════════════════════════════════════════════════════
with T[5]:
    st.markdown('<div class="sec-title">📝 Track Lead Interactions</div>',
                unsafe_allow_html=True)

    with st.form("log_form", clear_on_submit=True):
        fc1, fc2, fc3, fc4 = st.columns([1.5, 2, 2, 3])
        lead_num = fc1.number_input("Lead Row #", min_value=1, max_value=len(df), step=1, value=1)
        action   = fc2.selectbox("Action", [
            "Email Sent","Phone Call","LinkedIn Message","Meeting Scheduled",
            "Demo Completed","Proposal Sent","Closed Won","Closed Lost",
        ])
        outcome  = fc3.selectbox("Outcome", [
            "Positive","Neutral","No Response","Not Interested","Callback Requested",
        ])
        notes    = fc4.text_input("Notes", placeholder="Any context…")
        submitted_log = st.form_submit_button("➕  Log Interaction", type="primary")

    if submitted_log:
        row = df.iloc[lead_num - 1]
        entry = {
            "Timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Lead #":       int(lead_num),
            "Priority":     row.get("lead_category", "—"),
            "Lead Score":   int(row.get("lead_score", 0)),
            "Conv. Likely": f"{row.get('conversion_likelihood', 0):.1%}",
            "Action":       action, "Outcome": outcome, "Notes": notes,
        }
        st.session_state.logs.append(entry)
        st.success(f"✅ Logged: **{action}** → {outcome} for Lead #{lead_num}")

    if st.session_state.logs:
        log_df = pd.DataFrame(st.session_state.logs)
        st.dataframe(log_df, use_container_width=True)
        lc1, lc2 = st.columns(2)
        with lc1:
            ac = log_df["Action"].value_counts().reset_index()
            ac.columns = ["Action","Count"]
            figa = px.bar(ac, x="Action", y="Count", title="Actions Taken",
                          color_discrete_sequence=["#3b82f6"])
            figa.update_layout(height=270, showlegend=False,
                               margin=dict(t=35,b=5,l=5,r=5), font=dict(family="Inter"))
            st.plotly_chart(figa, use_container_width=True)
        with lc2:
            oc = log_df["Outcome"].value_counts().reset_index()
            oc.columns = ["Outcome","Count"]
            figo = px.pie(oc, names="Outcome", values="Count",
                          title="Outcome Breakdown", hole=0.45)
            figo.update_layout(height=270, font=dict(family="Inter"),
                               margin=dict(t=35,b=5,l=5,r=5))
            st.plotly_chart(figo, use_container_width=True)
        st.download_button("📥 Download Activity Log",
                           data=log_df.to_csv(index=False).encode(),
                           file_name="activity_log.csv", mime="text/csv")
    else:
        st.markdown('<div class="ibox blue">No interactions logged yet. '
                    'Use the form above to start tracking lead activity.</div>',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  TAB 7 · EXPORT DATA
# ══════════════════════════════════════════════════════════
with T[6]:
    st.markdown('<div class="sec-title">📤 Export Qualified Leads</div>',
                unsafe_allow_html=True)

    ec1, ec2 = st.columns(2)
    exp_tiers = ec1.multiselect("Priority tiers", ["High","Medium","Low"],
                                default=["High","Medium"])
    exp_min   = ec2.slider("Min Lead Score", 0, 100, 0, 5)

    safe_cols = [c for c in df_f.columns if not c.endswith("_enc")]
    default_export = [c for c in [
        "lead_category","predicted_category","lead_score","conversion_likelihood",
        "engagement_score","number_of_visits","time_spent_on_website",
        "company_size","budget_level","lead_source","cart_activity","email_response",
        "followup_timeline","followup_channel","followup_date",
    ] if c in safe_cols]
    exp_cols = st.multiselect("Columns to export", safe_cols, default=default_export)

    df_exp = df_f[(df_f["lead_category"].isin(exp_tiers)) & (df_f["lead_score"] >= exp_min)]
    valid  = [c for c in exp_cols if c in df_exp.columns]

    st.markdown(
        f'<div class="ibox green">✅ <strong>{len(df_exp):,} leads</strong> ready for export '
        f'({", ".join(exp_tiers)} priority · min score {exp_min}).</div>',
        unsafe_allow_html=True
    )

    if valid:
        sort_col = "lead_score" if "lead_score" in valid else valid[0]
        preview  = (df_exp[valid].sort_values(sort_col, ascending=False)
                    .reset_index(drop=True))
        preview.index += 1
        fmt2 = {}
        if "conversion_likelihood" in valid: fmt2["conversion_likelihood"] = "{:.1%}"
        if "lead_score"            in valid: fmt2["lead_score"]            = "{:.0f}"
        st.dataframe(preview.style.format(fmt2), use_container_width=True, height=380)

        ex1, ex2 = st.columns(2)
        ex1.download_button(
            "📥  Download CSV",
            data=df_exp[valid].to_csv(index=False).encode(),
            file_name=f"qualified_leads_{datetime.today().strftime('%Y%m%d')}.csv",
            mime="text/csv", use_container_width=True, type="primary",
        )
        try:
            import openpyxl  # noqa
            exp_df = (df_exp[valid] if "lead_category" in valid
                      else df_exp[valid + ["lead_category"]])
            ex2.download_button(
                "📊  Download Excel (tabs per tier)",
                data=to_excel_bytes(exp_df),
                file_name=f"qualified_leads_{datetime.today().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except ImportError:
            ex2.info("Install openpyxl for Excel export: `pip install openpyxl`")
    else:
        st.warning("Select at least one column to preview and export.")