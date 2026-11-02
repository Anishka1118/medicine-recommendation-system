from pathlib import Path
import html
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Healthcare Recommendation System", page_icon="🩺", layout="wide")
MODELS = Path(__file__).resolve().parent.parent / "models"

@st.cache_resource
def load_assets():
    model = joblib.load(MODELS / "disease_model.joblib")
    symptoms = joblib.load(MODELS / "symptom_list.joblib")
    scores = pd.read_csv(MODELS / "medicine_scores.csv")
    scores["disease"] = scores["disease"].astype(str).str.strip()
    return model, symptoms, scores

model, symptom_list, scores = load_assets()
labels = {symptom: symptom.replace("_", " ").strip().title() for symptom in symptom_list}

if "results" not in st.session_state:
    st.session_state.results = None
if "selected_symptoms" not in st.session_state:
    st.session_state.selected_symptoms = []
if "show_dashboard" not in st.session_state:
    st.session_state.show_dashboard = False
if "page" not in st.session_state:
    st.session_state.page = "🩺 Symptom Checker"

st.markdown("""
<style>
.stApp { background: radial-gradient(circle at top right, rgba(139,92,246,.10), transparent 35%), #F7F8FC; color:#1E293B; }
.block-container { padding-top:2rem; padding-bottom:2rem; }
h1,h2,h3 { color:#1E293B !important; } h4,h5,h6 { color:#334155 !important; }
p,label,li { color:#475569; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#F0EDFF,#F8FAFF); border-right:1px solid #E0E7FF; }
section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3 { color:#312E81 !important; }
section[data-testid="stSidebar"] label { color:#475569 !important; font-weight:500; }
div.stButton>button { border-radius:12px; min-height:46px; font-weight:600; border:1px solid #818CF8; background:linear-gradient(100deg,#4F46E5,#7C3AED); color:white; }
div.stButton>button:hover { border:1px solid #A5B4FC; background:linear-gradient(100deg,#4338CA,#6D28D9); color:white; transform:translateY(-2px); }
div[data-baseweb="select"]>div { background:#FFF; border:1px solid #C7D2FE; border-radius:10px; }
div[data-baseweb="select"] input { color:#1E293B !important; }
span[data-baseweb="tag"] { background:#E0E7FF !important; border:1px solid #A5B4FC !important; color:#3730A3 !important; }
span[data-baseweb="tag"] span { color:#3730A3 !important; }
div[data-testid="stAlert"] { border-radius:12px; border:1px solid #D8DEFF; }
hr { border-color:#E2E8F0; }
.welcome-box { padding:45px 30px; border-radius:22px; background:linear-gradient(120deg,#EEF2FF,#EDE9FE,#DDD6FE); text-align:center; border:1px solid #C4B5FD; margin:10px 0 20px; box-shadow:0 12px 40px rgba(99,102,241,.10); }
.welcome-title { color:#312E81; font-size:38px; font-weight:700; line-height:1.35; }
.dashboard-header { padding:22px; border-radius:16px; background:linear-gradient(110deg,#EEF2FF,#EDE9FE,#DDD6FE); border:1px solid #C4B5FD; margin-bottom:18px; }
.dashboard-header h2 { color:#312E81 !important; margin:0; font-size:30px; }
.prediction-card { background:linear-gradient(145deg,#FFF,#F5F3FF); border:1px solid #DDD6FE; border-radius:16px; padding:22px; margin-bottom:12px; box-shadow:0 8px 24px rgba(99,102,241,.08); }
.prediction-label { color:#4C1D95; font-size:15px; font-weight:600; margin-bottom:12px; }
.prediction-score { color:#312E81; font-size:32px; font-weight:700; margin-bottom:14px; }
.prediction-track { background:#E0E7FF; border-radius:20px; height:8px; overflow:hidden; }
.prediction-fill { height:8px; border-radius:20px; background:linear-gradient(90deg,#3B82F6,#8B5CF6); }
.section-card { background:#FFF; border:1px solid #E0E7FF; border-radius:16px; padding:22px; margin-bottom:18px; box-shadow:0 4px 16px rgba(99,102,241,.04); }
.footer-text { text-align:center; color:#64748B; font-size:13px; padding:10px; }
</style>
""", unsafe_allow_html=True)

def show_prediction_card(condition, score):
    safe_condition = html.escape(str(condition))
    score_value = max(0, min(float(score), 100))
    st.markdown(f"""
    <div class="prediction-card">
      <div class="prediction-label">{safe_condition}</div>
      <div class="prediction-score">{score_value:.1f}%</div>
      <div class="prediction-track"><div class="prediction-fill" style="width:{score_value}%;"></div></div>
    </div>""", unsafe_allow_html=True)

if not st.session_state.show_dashboard:
    st.write(""); st.write("")
    st.markdown("""<div class="welcome-box"><div class="welcome-title">🩺 Personalized Healthcare<br>&amp; Medicine Recommendation System</div></div>""", unsafe_allow_html=True)
    st.markdown("""<div style="text-align:center;font-size:17px;color:#475569;line-height:1.8;">Explore an ML-based educational tool that estimates possible conditions from selected symptoms and summarizes medicine reviews from the available dataset.</div>""", unsafe_allow_html=True)
    st.write(""); st.write("")
    st.info("This project is for educational purposes only. It does not diagnose diseases or prescribe medicines.")
    _, center, _ = st.columns([1,1,1])
    with center:
        if st.button("🚀 Get Started", type="primary", use_container_width=True):
            st.session_state.show_dashboard = True
            st.rerun()
    st.markdown('<div class="footer-text">Built with Streamlit and Machine Learning 💙</div>', unsafe_allow_html=True)
else:
    with st.sidebar:
        st.markdown("## 🩺 Healthcare System")
        st.caption("Personalized Healthcare Dashboard")
        st.divider()
        page = st.radio("Navigate", ["🩺 Symptom Checker","📊 Condition Insights","💊 Medicine Reviews","ℹ️ About Project"], key="page")
        st.divider()
        if st.button("🏠 Back to Welcome Page", use_container_width=True):
            st.session_state.show_dashboard = False
            st.rerun()
        st.caption("Educational use only")

    st.markdown('<div class="dashboard-header"><h2>🩺 Personalized Healthcare Dashboard</h2></div>', unsafe_allow_html=True)
    st.caption("Machine Learning • Symptom Analysis • Medicine Reviews")
    st.info("This tool provides educational model outputs only. It does not diagnose diseases or prescribe medicines.")
    st.divider()

    if page == "🩺 Symptom Checker":
        st.subheader("🧬 Select Your Symptoms")
        st.caption("Choose the symptoms you want the model to analyze.")
        chosen = st.multiselect("What symptoms are you experiencing?", options=symptom_list, format_func=lambda symptom: labels[symptom], default=st.session_state.selected_symptoms, placeholder="Search and select symptoms...")
        st.caption(f"Selected symptoms: {len(chosen)}")
        if st.button("✨ Get Recommendations", type="primary", use_container_width=True):
            if not chosen:
                st.warning("Please select at least one symptom.")
                st.session_state.results = None
            else:
                if len(chosen) < 3:
                    st.info("Only a few symptoms are selected. The model output may be less reliable.")
                x = pd.DataFrame([[1 if symptom in chosen else 0 for symptom in symptom_list]], columns=symptom_list)
                probs = model.predict_proba(x)[0]
                top_indices = probs.argsort()[::-1][:3]
                st.session_state.results = [{"Condition":str(model.classes_[i]).strip(),"Model score (%)":round(probs[i]*100,1)} for i in top_indices]
                st.session_state.selected_symptoms = chosen
        if st.session_state.results:
            st.divider()
            st.subheader("🎯 Predicted Conditions")
            st.caption("These scores represent the model's output for the selected symptoms. They are not confirmed diagnoses.")
            cols = st.columns(3)
            for col, result in zip(cols, st.session_state.results):
                with col:
                    show_prediction_card(result["Condition"], result["Model score (%)"])
            st.success("Prediction generated. Use the sidebar to explore condition insights and medicine review summaries.")

    elif page == "📊 Condition Insights":
        st.subheader("📊 Condition Insights")
        if st.session_state.results:
            result_df = pd.DataFrame(st.session_state.results)
            fig = px.bar(result_df, x="Condition", y="Model score (%)", text="Model score (%)", title="Top 3 Predicted Conditions", color="Condition", color_discrete_sequence=["#6366F1","#8B5CF6","#A78BFA"], labels={"Condition":"Predicted Condition","Model score (%)":"Model Score (%)"})
            fig.update_traces(textposition="outside", texttemplate="%{text:.1f}%", textfont=dict(color="#334155",size=14))
            fig.update_layout(template="plotly_white", yaxis_range=[0,100], yaxis_title="Model Score (%)", xaxis_title="Condition", plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", font=dict(color="#334155"), title_font=dict(color="#312E81",size=20), legend_title_text="Condition", margin=dict(l=20,r=20,t=60,b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("📋 Prediction Summary")
            st.dataframe(result_df, hide_index=True, use_container_width=True)
        else:
            st.info("Select symptoms and click Get Recommendations to view condition insights.")

    elif page == "💊 Medicine Reviews":
        st.subheader("💊 Medicine Review Summary")
        st.caption("Medicine rankings are summarized from patient review data. They are not clinical treatment recommendations.")
        if st.session_state.results:
            for result in st.session_state.results:
                name = result["Condition"]
                with st.expander(f'🩺 {name} — {result["Model score (%)"]:.1f}% model score', expanded=True):
                    meds = scores[scores["disease"].str.casefold() == name.casefold()].copy()
                    if meds.empty:
                        st.caption("No matching medicine review records were found in the available dataset.")
                    else:
                        meds = meds.sort_values("score", ascending=False).head(5)
                        display_meds = meds[["drugName","reviews","avg_rating","score"]].rename(columns={"drugName":"Medicine","reviews":"Review Count","avg_rating":"Average Rating","score":"Review Score"})
                        display_meds["Average Rating"] = display_meds["Average Rating"].round(1)
                        display_meds["Review Score"] = display_meds["Review Score"].round(2)
                        for _, medicine in display_meds.iterrows():
                            with st.container(border=True):
                                st.markdown(f"### 💊 {medicine['Medicine']}")
                                rating_col, reviews_col, score_col = st.columns(3)
                                with rating_col:
                                    st.metric("⭐ Average Rating", f"{medicine['Average Rating']:.1f}")
                                with reviews_col:
                                    st.metric("📝 Reviews", f"{int(medicine['Review Count'])}")
                                with score_col:
                                    st.metric("📊 Review Score", f"{medicine['Review Score']:.2f}")
                        st.caption("Higher review scores indicate a higher ranking within the available review dataset.")
        else:
            st.info("Generate condition predictions first to view the related medicine review summaries.")

    elif page == "ℹ️ About Project":
        st.subheader("ℹ️ About This Project")
        st.markdown("This project demonstrates a machine-learning-based healthcare recommendation interface.")
        st.divider()
        st.markdown("### 🔄 How It Works")
        step1, step2 = st.columns(2)
        with step1:
            st.markdown("""<div class="section-card"><h4 style="color:#4F46E5;">01 • Symptom Selection</h4><p>The user selects symptoms from the available symptom list.</p></div>""", unsafe_allow_html=True)
            st.markdown("""<div class="section-card"><h4 style="color:#7C3AED;">03 • Condition Estimation</h4><p>The trained model estimates scores for possible conditions and displays the top three outputs.</p></div>""", unsafe_allow_html=True)
        with step2:
            st.markdown("""<div class="section-card"><h4 style="color:#7C3AED;">02 • Model Processing</h4><p>Selected symptoms are converted into model input features.</p></div>""", unsafe_allow_html=True)
            st.markdown("""<div class="section-card"><h4 style="color:#4F46E5;">04 • Review Summary</h4><p>Medicine review data is summarized for predicted conditions, where available.</p></div>""", unsafe_allow_html=True)
        st.divider()
        st.markdown("### 🧩 Project Components")
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("""<div class="section-card"><h3 style="color:#6366F1;">🤖</h3><h4>Machine Learning</h4><p>Disease prediction model</p></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""<div class="section-card"><h3 style="color:#8B5CF6;">📈</h3><h4>Visualization</h4><p>Plotly condition comparison chart</p></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown("""<div class="section-card"><h3 style="color:#A78BFA;">💊</h3><h4>Medicine Data</h4><p>Patient review summary dataset</p></div>""", unsafe_allow_html=True)
        st.warning("The model's output is not a confirmed diagnosis. Medicine review scores do not establish safety, effectiveness, or suitability for an individual. Consult a qualified healthcare professional for medical concerns.")

    st.divider()
    st.markdown("""<div class="footer-text">Educational project, not medical advice. Model outputs are not confirmed diagnoses. Medicine rankings are based on patient reviews, not clinical evidence. 💙</div>""", unsafe_allow_html=True)
