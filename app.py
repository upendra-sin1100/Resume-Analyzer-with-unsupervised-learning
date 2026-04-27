import streamlit as st
import joblib
import re
import pypdf
import numpy as np
from scipy.spatial.distance import cdist

st.set_page_config(page_title="Resume Analyzer AI", layout="wide", page_icon="🧠")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f0f1a; }
[data-testid="stSidebar"] { background: #1a1a2e; }

.hero { text-align: center; padding: 2rem 0 1rem; }
.hero h1 {
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg, #7c9ef8, #c07cf8, #f8c07c);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: #888; font-size: 1rem; }

.stat-card {
    background: #1a1a2e; border-radius: 14px; padding: 18px 12px;
    text-align: center; border: 1px solid #2a2a4e;
    transition: transform 0.2s;
}
.stat-card:hover { transform: translateY(-2px); }
.stat-num { font-size: 2.2rem; font-weight: 800; }
.stat-label { font-size: 0.72rem; color: #888; margin-top: 2px; letter-spacing: 0.05em; }

.resume-line {
    background: #1a1a2e; border-radius: 10px; padding: 12px 18px;
    margin: 6px 0; border-left: 3px solid; color: #ddd;
    font-size: 0.92rem; line-height: 1.5;
}
.source-badge {
    font-size: 0.65rem; padding: 1px 7px; border-radius: 10px;
    float: right; opacity: 0.6; margin-top: 2px;
}
.badge-model { background: #2a2a4e; color: #7c9ef8; }
.badge-rule  { background: #2a1e1e; color: #f8c07c; }

.skill-pill {
    display: inline-block; background: #2a2a1e;
    border: 1px solid #f8c07c44; color: #f8c07c;
    border-radius: 20px; padding: 4px 14px;
    font-size: 0.83rem; margin: 4px 3px;
}
.debug-box {
    background: #1a1a2e; border-radius: 8px; padding: 12px 16px;
    border: 1px solid #333; font-family: monospace;
    font-size: 0.82rem; color: #aaa;
}
</style>
""", unsafe_allow_html=True)


# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        vec   = joblib.load('tfidf_vectorizer.pkl')
        km    = joblib.load('kmeans_model.pkl')
        l_map = joblib.load('label_map.pkl')
        return vec, km, l_map
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.stop()

vectorizer, kmeans, label_map = load_models()


# ── Keyword Rules (fallback when model is uncertain) ──────────────────────────
KEYWORD_RULES = {
    'edu': [
        'b.tech', 'btech', 'm.tech', 'mtech', 'bachelor', 'master', 'phd', 'mba',
        'university', 'college', 'institute', 'school', 'cgpa', 'gpa', 'degree',
        'engineering', 'science', 'graduation', 'diploma', 'higher secondary',
        '10th', '12th', 'hsc', 'ssc', 'board'
    ],
    'skill': [
        'python', 'java', 'c++', 'javascript', 'sql', 'html', 'css', 'react',
        'node', 'django', 'flask', 'tensorflow', 'pytorch', 'pandas', 'numpy',
        'git', 'docker', 'aws', 'linux', 'machine learning', 'deep learning',
        'nlp', 'data analysis', 'mongodb', 'mysql', 'sklearn', 'scikit',
        'languages:', 'tools:', 'frameworks:', 'skills:', 'technologies:',
        'soft skills:', 'web development:', 'data analysis:'
    ],
    'pi': [
        '@gmail', '@yahoo', '@outlook', 'linkedin.com', 'github.com',
        'phone:', 'mobile:', 'email:', 'address:', 'gwalior', 'india',
        'mumbai', 'delhi', 'bangalore', 'hyderabad', '+91'
    ],
    'obj': [
        'objective', 'career objective', 'goal', 'aspiring', 'seeking',
        'motivated', 'enthusiastic', 'passionate about', 'looking for'
    ],
    'sum': [
        'summary', 'professional summary', 'profile', 'about me',
        'overview', 'highlights', 'experienced professional'
    ],
    'exp': [
        'experience', 'worked at', 'internship', 'intern at', 'developer at',
        'engineer at', 'analyst at', 'designed', 'implemented', 'built',
        'developed', 'managed', 'led', 'collaborated', 'deployed'
    ],
}

# Confidence threshold — below this, override model with keyword rules
CONFIDENCE_THRESHOLD = 0.35


def keyword_classify(text: str) -> str | None:
    """Return label if text matches keyword rules, else None."""
    text_lower = text.lower()
    for label, keywords in KEYWORD_RULES.items():
        if any(kw in text_lower for kw in keywords):
            return label
    return None


# ── Text Cleaning — must match advanced_clean_text from training ──────────────
def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r'^(Exp|PI|Sum|Edu|Skill|Obj|QC)\t', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(Exp|PI|Sum|Edu|Skill|Obj|QC)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\+?\d[\d\s\-().]{7,}\d', '', text)   # phone numbers
    text = re.sub(r'\S+@\S+', '', text)                   # emails
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def model_confidence(vector) -> float:
    """
    Measure confidence as 1 - (min_dist / mean_dist).
    Higher = more confident the model placed this in the right cluster.
    """
    dists = cdist(vector.toarray(), kmeans.cluster_centers_, 'euclidean')[0]
    sorted_d = np.sort(dists)
    if sorted_d[0] == 0:
        return 1.0
    return 1.0 - (sorted_d[0] / (np.mean(sorted_d) + 1e-9))


# ── Section Config ────────────────────────────────────────────────────────────
SECTIONS = {
    'exp':   {"label": "Work Experience", "icon": "💼", "color": "#7c9ef8"},
    'edu':   {"label": "Education",       "icon": "🎓", "color": "#a8d8a8"},
    'skill': {"label": "Skills",          "icon": "🛠️", "color": "#f8c07c"},
    'sum':   {"label": "Summary",         "icon": "📝", "color": "#c07cf8"},
    'pi':    {"label": "Personal Info",   "icon": "👤", "color": "#f87c7c"},
    'obj':   {"label": "Objective",       "icon": "🎯", "color": "#7cf8c0"},
    'qc':    {"label": "Other",           "icon": "📄", "color": "#888888"},
}


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🧠 Resume Analyzer AI</h1>
    <p>K-Means clustering + keyword fallback · Handles real-world resumes</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🔍 Model Debug")
    st.markdown("**label_map contents:**")
    st.markdown(f'<div class="debug-box">{label_map}</div>', unsafe_allow_html=True)
    st.markdown("---")
    conf_thresh = st.slider(
        "Confidence threshold", 0.0, 1.0, CONFIDENCE_THRESHOLD, 0.05,
        help="Below this confidence, keyword rules override the model prediction"
    )
    show_source = st.checkbox("Show prediction source (model vs rule)", value=True)


# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📄 Upload your Resume (PDF)", type="pdf")

if uploaded_file:
    with st.spinner("Analyzing resume..."):
        try:
            reader    = pypdf.PdfReader(uploaded_file)
            full_text = "\n".join(
                page.extract_text() for page in reader.pages if page.extract_text()
            )
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            st.stop()

        lines       = full_text.split('\n')
        valid_lines = [l.strip() for l in lines if len(l.strip()) >= 5]

        parsed   = {k: [] for k in SECTIONS}   # (line, source)
        unknown  = []
        stats    = {"model": 0, "rule": 0}

        if valid_lines:
            cleaned  = [clean_text(l) for l in valid_lines]
            vectors  = vectorizer.transform(cleaned)
            cluster_ids = kmeans.predict(vectors)

            for i, (line, cid) in enumerate(zip(valid_lines, cluster_ids)):
                vec_i     = vectors[i]
                model_lbl = label_map.get(int(cid))
                conf      = model_confidence(vec_i)

                # Decide: use model or fall back to keyword rules
                if conf >= conf_thresh and model_lbl in parsed:
                    final_label = model_lbl
                    source      = "model"
                else:
                    kw_lbl = keyword_classify(line)
                    if kw_lbl:
                        final_label = kw_lbl
                        source      = "rule"
                    elif model_lbl in parsed:
                        final_label = model_lbl   # take model even if low-confidence
                        source      = "model"
                    else:
                        unknown.append((line, int(cid), model_lbl))
                        continue

                parsed[final_label].append((line, source))
                stats[source] += 1

    # ── Stats Row ─────────────────────────────────────────────────────────────
    total = len(valid_lines)
    st.success(f"✅ Analyzed **{total}** lines — model: {stats['model']}, keyword fallback: {stats['rule']}")

    cols = st.columns(len(SECTIONS))
    for col, (key, info) in zip(cols, SECTIONS.items()):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-num" style="color:{info['color']}">{len(parsed[key])}</div>
                <div class="stat-label">{info['icon']} {info['label']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    active_keys = [k for k in SECTIONS if parsed[k]]

    if active_keys:
        tabs = st.tabs([
            f"{SECTIONS[k]['icon']} {SECTIONS[k]['label']} ({len(parsed[k])})"
            for k in active_keys
        ])
        for tab, key in zip(tabs, active_keys):
            info = SECTIONS[key]
            with tab:
                if key == 'skill':
                    pills = "".join(
                        f'<span class="skill-pill">{item}</span>'
                        for item, _ in parsed[key]
                    )
                    st.markdown(pills, unsafe_allow_html=True)
                else:
                    for item, source in parsed[key]:
                        badge = ""
                        if show_source:
                            cls  = "badge-model" if source == "model" else "badge-rule"
                            badge = f'<span class="source-badge {cls}">{"🤖 model" if source == "model" else "📋 rule"}</span>'
                        st.markdown(f"""
                        <div class="resume-line" style="border-color:{info['color']}">
                            {badge}{item}
                        </div>""", unsafe_allow_html=True)
    else:
        st.warning("⚠️ No sections populated. Check sidebar debug.")

    if unknown:
        with st.expander(f"⚠️ {len(unknown)} unclassified lines"):
            for line, cid, lbl in unknown:
                st.code(f"cluster {cid} → '{lbl}' | {line[:100]}")