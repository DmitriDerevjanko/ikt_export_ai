import os
import json
import logging
import re
import time
import traceback
import requests
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
from dotenv import load_dotenv

# ============== CONFIG & INIT ==============
load_dotenv()

# ===================== LOGGING ======================
os.makedirs("logs", exist_ok=True)
log_file_path = os.path.join("logs", "ikt_export.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ikt-export-advisor")

file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(file_handler)

def ui_log(msg: str):
    if "ui_logs" not in st.session_state:
        st.session_state.ui_logs = []
    st.session_state.ui_logs.append(msg)
    logger.info(msg)

# ============== MISTRAL INIT ==============
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-medium")

if not MISTRAL_API_KEY:
    st.error("⚠️ Please set MISTRAL_API_KEY in your .env file")
    st.stop()

def ask_mistral(prompt_text: str) -> str:
    """Send prompt to Mistral API"""
    ui_log(f"➡️ [SEND:MISTRAL] prompt_len={len(prompt_text)} chars")
    try:
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
            json={
                "model": MISTRAL_MODEL,
                "messages": [{"role": "user", "content": prompt_text}],
                "temperature": 0.7,
                "max_tokens": 2000,
            },
            timeout=60,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"].strip()
        ui_log(f"⬅️ [RECV:MISTRAL] len={len(text)}")
        return text
    except Exception as e:
        err = traceback.format_exc()
        ui_log(f"❌ [MISTRAL ERROR] {e}\n{err}")
        return f"Error: {e}"

ui_log(f"✅ Mistral initialized (model={MISTRAL_MODEL})")

# ============== SYSTEM PROMPT ==============
SYSTEM_PROMPT = """
You are "Eeva", an AI Export Advisor from Enterprise Estonia.
Goal: help Estonian ICT, manufacturing, and logistics companies evaluate and improve export readiness.

Rules:
- Ask ONE question at a time.
- Never answer for the user.
- Wait for the user's input before continuing.
- Keep tone professional, warm, and concise.
- After exactly 10 answered questions, generate the final export readiness report automatically.

Ask the following questions in this exact order:
1) What does your company do?
2) Who is your ideal customer (ICP)?
3) Which countries or regions are you targeting for export?
4) What is your previous export experience?
5) What sales channels do you use (direct, partners, distributors, online)?
6) Do you have a website, and is it localized for target markets?
7) Tell me about your team and export capacity.
8) What is your export and marketing budget for the next 12 months?
9) Do you have certifications or compliance standards (ISO, CE, EN)?
10) What are your main export challenges or goals for the next year?

After question 10, automatically generate the final Export Readiness Report including:
- Title and company name (if known)
- Overall readiness score (0–100)
- 🟢 Strengths (5+)
- 🔴 Weaknesses (5+)
- 🚀 Next steps (5–7 actionable)
- Optional: short “Opportunities & Risks” section
Tone: warm, insightful, motivating.
"""


# ============== PAGE STYLE ==============
st.set_page_config(page_title="IKT Export Advisor", page_icon="🌍", layout="centered")
st.markdown("""
<style>
body {
  background: linear-gradient(135deg, #eef2f3 0%, #8ec5fc 100%);
  font-family: "Inter", sans-serif;
}
.big-title {
  text-align:center;
  font-size:2.8rem;
  font-weight:900;
  background: linear-gradient(90deg, #0061ff, #60efff);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
}
.sub-title {
  text-align:center;
  font-size:1.1rem;
  color:#333;
  margin-bottom:2em;
}
.card-container {
  display:flex;
  justify-content:center;
  gap:1.5em;
  flex-wrap:wrap;
  margin:2em auto;
  max-width:900px;
}
.card {
  flex:1 1 280px;
  max-width:310px;
  background:#fff;
  border-radius:18px;
  padding:1.5em;
  box-shadow:0 10px 20px rgba(0,0,0,.08);
  transition:transform .3s, box-shadow .3s;
  text-align:center;
}
.card:hover {
  transform:translateY(-8px);
  box-shadow:0 16px 32px rgba(0,0,0,.15);
}
.start-btn {
  display:block;
  margin:2em auto 0;
  background:linear-gradient(90deg,#0072ff,#00c6ff);
  color:#fff;
  border:none;
  padding:14px 38px;
  border-radius:50px;
  font-size:1.12rem;
  font-weight:700;
  cursor:pointer;
  box-shadow:0 4px 14px rgba(0,0,0,.15);
}
.start-btn:hover { transform:scale(1.05); }
.footer {
  text-align:center;
  margin-top:3em;
  color:#666;
  font-size:.9rem;
}
</style>
""", unsafe_allow_html=True)

# ============== SIDEBAR ==============
with st.sidebar:
    st.markdown("### 🌍 IKT Export Advisor")
    st.caption("Powered by **Mistral AI** and **Streamlit**")

    sector = st.selectbox(
        "Select your sector:",
        ["Manufacturing", "ICT / Software", "Logistics", "Food Industry",
         "Construction", "Energy & Utilities", "Electronics", "Textiles & Fashion",
         "Healthcare / MedTech", "Agriculture & Forestry", "Education & Training",
         "Tourism / Services", "Creative Industries", "Other"],
        index=0
    )

    st.markdown("---")
    if st.button("🔄 Reset conversation"):
        for k in ["messages", "started", "final_text", "final_score",
                  "ui_logs", "ceo_summary"]:
            st.session_state.pop(k, None)
        st.rerun()
        # === Useful Export Links ===
    st.markdown("""
    <div style="
        background: #f8faff;
        border-left: 4px solid #0078ff;
        padding: 10px 15px;
        border-radius: 8px;
        font-size: 0.95rem;
    ">
      <b>🌐 Useful Export Resources</b><br>
      <a href="https://www.eas.ee/en/" target="_blank">🇪🇪 Enterprise Estonia (EAS)</a><br>
      <a href="https://aire-edih.eu/" target="_blank">🤖 AIRE – AI & Robotics Estonia</a><br>
      <a href="https://www.e-resident.gov.ee/" target="_blank">💼 e-Residency for Businesses</a><br>
      <a href="https://www.stat.ee/en" target="_blank">📊 Statistics Estonia – Export Data</a><br>
      <a href="https://madb.europa.eu/" target="_blank">🌍 EU Market Access Database</a><br>
      <a href="https://een.ec.europa.eu/" target="_blank">💡 Enterprise Europe Network (EEN)</a>
    </div>
    """, unsafe_allow_html=True)


# ============== STATE ==============
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    st.session_state.started = False
    st.session_state.final_text = ""
    st.session_state.final_score = None

# ============== LANDING PAGE ==============
if not st.session_state.started:
    st.markdown("<div class='big-title'>IKT Export Readiness AI Agent</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Conversational AI advisor powered by <b>Mistral</b> ✨</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card-container">
      <div class="card">
        <h3>🔍 Analyze</h3>
        <p>Interactive dialogue to assess product-market fit, team capacity, localization, logistics, and compliance.</p>
      </div>
      <div class="card">
        <h3>🤝 Advise</h3>
        <p>Actionable guidance on markets, partners, pricing, certifications, and export processes.</p>
      </div>
      <div class="card">
        <h3>🚀 Plan</h3>
        <p>Step-by-step plan and readiness score with a radar visualization.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Start Conversation", use_container_width=True):
        intro = (
            f"Hello! 👋 I'm **Eeva**, your Export Readiness Advisor for the {sector} sector. "
            "Let's start — what does your company do?"
        )
        st.session_state.messages.append({"role": "assistant", "content": intro})
        st.session_state.started = True
        st.rerun()

    st.markdown("<div class='footer'>© 2025 IKT Export Advisor • Built with ❤️ in Estonia</div>", unsafe_allow_html=True)

# ============== UTILS ==============
def conversation_text(limit=8):
    base = f"[System instructions: {SYSTEM_PROMPT.strip()}]\nSector: {sector}\n\n"
    history = [m for m in st.session_state.messages if m["role"] in ("user", "assistant")]
    for m in history[-limit:]:
        base += f"{m['role'].upper()}: {m['content']}\n"
    return base

# ============== PDF REPORT (UNICODE + ARAIL-STYLE) ==============
def build_pdf_report(text: str, ceo_summary: str = None) -> bytes:
    from fpdf import FPDF
    import re, os, tempfile, requests, zipfile, io

    class PDF(FPDF):
        def header(self):
            self.set_fill_color(0, 97, 255)
            self.rect(0, 0, 220, 25, "F")
            self.set_font("DejaVu", "B", 18)
            self.set_text_color(255, 255, 255)
            self.cell(0, 12, "IKT Export Readiness Report", align="C", new_x="LMARGIN", new_y="NEXT")
            self.ln(8)

        def footer(self):
            self.set_y(-15)
            self.set_font("DejaVu", "I", 9)
            self.set_text_color(120, 120, 120)
            self.cell(0, 10, "Generated by Eeva – AI Export Advisor • Built in Estonia", align="C")

    pdf = PDF("P", "mm", "A4")

    font_dir = os.path.join(tempfile.gettempdir(), "dejavu_fonts")
    font_path = os.path.join(font_dir, "DejaVuSans.ttf")

    if not os.path.exists(font_path):
        os.makedirs(font_dir, exist_ok=True)
        url = "https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.zip"
        r = requests.get(url)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for name in z.namelist():
                if "DejaVuSans.ttf" in name:
                    z.extract(name, font_dir)
                    os.rename(os.path.join(font_dir, name), font_path)
                    break

    pdf.add_font("DejaVu", "", font_path)
    pdf.add_font("DejaVu", "B", font_path)
    pdf.add_font("DejaVu", "I", font_path)

    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_text_color(30, 30, 30)

    def sanitize(s: str) -> str:
        if not s:
            return ""
        s = re.sub(r"(READY_TO_SUMMARIZE|Here's the next question:.*?)", "", s, flags=re.I)
        s = re.sub(r"[*_#`]+", "", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()

    text = sanitize(text or "⚠️ Report content missing.")
    ceo_summary = sanitize(ceo_summary or "Summary not available.")

    pdf.set_font("DejaVu", "", 13)
    pdf.multi_cell(0, 8, text)
    pdf.ln(8)

    pdf.set_draw_color(0, 97, 255)
    pdf.set_line_width(0.8)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(8)

    pdf.set_font("DejaVu", "B", 15)
    pdf.set_text_color(0, 97, 255)
    pdf.cell(0, 10, "CEO Summary", ln=True)
    pdf.set_font("DejaVu", "", 12)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(0, 7, ceo_summary)
    pdf.ln(10)

    pdf.set_font("DejaVu", "I", 11)
    pdf.set_text_color(90, 90, 90)
    pdf.multi_cell(
        0, 6,
        "💡 Note: This report was generated automatically using AI analysis. "
        "It provides strategic insights and recommendations based on your responses. "
        "For tailored consultation, contact Enterprise Estonia or AIRE experts."
    )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)
    with open(tmp.name, "rb") as f:
        data = f.read()
    os.unlink(tmp.name)
    return data




# ============== RADAR CHART ==============
def render_radar(criteria: dict):
    labels, values = list(criteria.keys()), list(criteria.values())
    values += values[:1]; angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2); ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=9)
    st.pyplot(fig)

# ============== FINAL REPORT ==============
def render_final_section():
    st.markdown("## 📄 Export Readiness Report")
    st.markdown(st.session_state.final_text)
    score = st.session_state.final_score or 75
    st.metric("🌟 Readiness Score", f"{score}/100")
    criteria = {
        "ICP clarity": 8, "Market focus": 7, "Website": 6,
        "Budget": 7, "Certifications": 9, "Team": 6,
        "Sales channels": 7, "Partners": 6, "Localization": 5, "Strategy": 8,
    }
    render_radar(criteria)
    if "ceo_summary" not in st.session_state:
        ceo_prompt = conversation_text() + "\nWrite a concise CEO summary (3–4 sentences)."
        st.session_state.ceo_summary = ask_mistral(ceo_prompt)
    st.subheader("🧠 CEO Summary")
    st.markdown(st.session_state.ceo_summary)
    pdf_bytes = build_pdf_report(st.session_state.final_text, st.session_state.ceo_summary)
    st.download_button("📄 Download PDF", data=pdf_bytes, file_name="export_readiness_report.pdf", mime="application/pdf")

# ============== CHAT LOOP (FINAL FIX) ==============
if st.session_state.started:
    st.header("💬 Chat with your AI Export Advisor")

    for m in st.session_state.messages:
        if m["role"] in ("user", "assistant"):
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    if st.session_state.final_text:
        render_final_section()

    user_msg = st.chat_input("Type your reply or ask for advice...")
    if user_msg:
        st.session_state.messages.append({"role": "user", "content": user_msg})
        ui_log(f"👤 USER: {user_msg}")

        with st.chat_message("assistant"):
            with st.spinner("Eeva is thinking…"):
                user_answers = sum(1 for m in st.session_state.messages if m["role"] == "user")
                context = conversation_text()

                if st.session_state.final_text:
                    context += (
                        "\n\n[POST-REPORT MODE] The export readiness report has been generated. "
                        "Now continue as an experienced export consultant. "
                        "Answer the user's new questions or comments based on the report, "
                        "and provide helpful, practical guidance."
                    )
                    reply = ask_mistral(context)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.markdown(reply)

                elif user_answers < 10:
                    reply = ask_mistral(context)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.markdown(reply)
                    st.rerun()

                else:
                    ui_log("🧾 Generating final report via Mistral…")
                    summary_prompt = context + """
Please now produce the final export readiness report.
Include: title, company name, readiness score (0–100),
at least 5 strengths, 5 weaknesses, 7–10 next steps,
and a short Opportunities & Risks section in Markdown.
Use professional, structured Markdown style.
"""
                    final_text = ask_mistral(summary_prompt)
                    st.session_state.final_text = final_text
                    st.session_state.messages.append({"role": "assistant", "content": "✅ Report generated."})

                    render_final_section()
