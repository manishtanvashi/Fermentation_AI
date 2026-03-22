import streamlit as st
import requests
import base64

st.set_page_config(page_title="Fermentation AI", layout="centered")

# ---------- LOAD IMAGE ----------
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_base64 = get_base64_image("kevin-kandlbinder-WrjxzLskZK0-unsplash.jpg")

# ---------- STYLE ----------
st.markdown(f"""
<style>

/* BACKGROUND */
.stApp {{
    background-image: url("data:image/jpg;base64,{img_base64}");
    background-size: cover;
    background-position: center;
}}

/* OVERLAY */
.stApp::before {{
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    backdrop-filter: blur(6px);
    background: rgba(0,0,0,0.6);
    z-index: -1;
}}

/* REMOVE EXTRA SPACE */
.block-container {{
    padding-top: 3rem !important;
}}

/* TEXT */
h1, h2, h3 {{
    color: white !important;
    text-align: center;
}}

label {{
    color: white !important;
}}

/* INPUT */
input {{
    background-color: rgba(255,255,255,0.9) !important;
    border-radius: 8px !important;
    color: black !important;
}}

/* HOME BUTTON CLEAN */
button[kind="secondary"] {{
    background: transparent !important;
    color: white !important;
    border: none !important;
    font-size: 14px !important;
    padding: 4px 8px !important;
}}

button[kind="secondary"]:hover {{
    color: #00aced !important;
}}

/* FOOTER FIX (CENTER PROPERLY) */
.footer {{
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    text-align: center;
    background: rgba(0,0,0,0.9);
    color: white;
    padding: 10px 0;
    font-size: 14px;
}}

.footer-inner {{
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 8px;
}}

.footer a {{
    color: #00aced;
    text-decoration: none;
}}

</style>

<div class="footer">
  <div class="footer-inner">
    <span>Email: manishtanvashi150@gmail.com</span> |
    <a href="https://linkedin.com/in/manish-tanvashi" target="_blank">LinkedIn</a>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- SESSION ----------
if "page" not in st.session_state:
    st.session_state.page = "welcome"

if "answer" not in st.session_state:
    st.session_state.answer = None

if "sources" not in st.session_state:
    st.session_state.sources = []

# ---------- WELCOME ----------
if st.session_state.page == "welcome":

    st.markdown("<h1>Fermentation Research Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<h2>Explore fermentation knowledge using AI</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("Start", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()

# ---------- CHAT ----------
elif st.session_state.page == "chat":

    # HOME (TOP RIGHT CLEAN)
    col1, col2 = st.columns([9,1])
    with col2:
        if st.button("Home"):
            st.session_state.page = "welcome"
            st.session_state.answer = None
            st.session_state.sources = []
            st.rerun()

    st.markdown("<h1>Ask Your Question</h1>", unsafe_allow_html=True)

    # INPUT
    question = st.text_input(
        "Enter your question:",
        placeholder="Enter your text here..."
    )

    # CENTER BUTTON
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        submit = st.button("Get Answer", use_container_width=True)

    # API CALL
    if submit:
        if question:
            with st.spinner("Processing..."):
                response = requests.post(
                    "http://127.0.0.1:8000/ask",
                    json={"question": question}
                )
                result = response.json()

            st.session_state.answer = result["answer"]
            st.session_state.sources = result["sources"]
        else:
            st.warning("Enter a question")

    # ---------- DISPLAY ----------
    if st.session_state.answer:

        st.markdown("## Answer")

        st.markdown(
            f"""
            <div style="
                background: rgba(255,255,255,0.15);
                padding:20px;
                border-radius:10px;
                color:white;
                font-size:18px;
                backdrop-filter: blur(4px);
            ">
            {st.session_state.answer}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("## Sources")
        for s in st.session_state.sources:
            st.write(f"- {s}")

        # ASK ANOTHER CENTER
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("Ask Another", use_container_width=True):
                st.session_state.answer = None
                st.session_state.sources = []
                st.rerun()
