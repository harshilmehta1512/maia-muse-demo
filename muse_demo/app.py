"""
MAIA MUSE — AI Music Detection Demo
=====================================
Drag-and-drop browser demo for the Thursday client presentation.
"""

import time
import tempfile
import os
import textwrap
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MAIA MUSE · AI Music Detection",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Inject CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Reset & Base ── */
* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #F4F6FB !important;
    color: #0F172A !important;
}
[data-testid="stAppViewContainer"] > .main { background: #F4F6FB !important; }
[data-testid="stHeader"] { display: none !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }
[data-testid="stSidebarNav"] { display: none; }
[data-testid="collapsedControl"] { display: none; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #F4F6FB; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 4px; }

/* ── Main padding ── */
.block-container, .stMainBlockContainer {
    padding: 1.5rem 2.5rem 2rem 2.5rem !important;
    max-width: 100% !important;
}

/* ── Top nav bar ── */
.topnav {
    background: #FFFFFF;
    border-bottom: 1px solid #E2E8F0;
    padding: 0 3rem;
    height: 72px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 1px 8px rgba(0,0,0,0.06);
}
.topnav-brand {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #7C3AED;
}
.topnav-badge {
    background: linear-gradient(135deg,#7C3AED,#06B6D4);
    color: white;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 100px;
}

/* ── Page body padding ── */
.page-body { padding: 2.5rem 3rem; max-width: 1400px; margin: 0 auto; }

/* ── White card ── */
.card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.04);
    position: relative;
}
.card-accent-top {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #7C3AED, #06B6D4);
    border-radius: 16px 16px 0 0;
}

/* ── Section label ── */
.section-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 1rem;
}

/* ── Upload zone ── */
[data-testid="stFileUploadDropzone"] {
    background: #FAFAFF !important;
    border: 2px dashed #C4B5FD !important;
    border-radius: 14px !important;
    transition: border-color 0.25s ease !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    background: #F5F0FF !important;
    border-color: #7C3AED !important;
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #7C3AED, #06B6D4) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 12px rgba(124,58,237,0.25) !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(124,58,237,0.35) !important;
}

/* ── Verdict badge ── */
.verdict-ai {
    display: inline-flex; align-items: center; gap: 8px;
    background: #FEF2F2; border: 1px solid #FECACA;
    border-radius: 100px; padding: 8px 20px;
    color: #DC2626; font-size: 0.8rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
}
.verdict-real {
    display: inline-flex; align-items: center; gap: 8px;
    background: #F0FDF4; border: 1px solid #BBF7D0;
    border-radius: 100px; padding: 8px 20px;
    color: #16A34A; font-size: 0.8rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
}
.verdict-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; animation: pulse-dot 2s infinite; }
.dot-ai { background: #EF4444; }
.dot-real { background: #22C55E; }
@keyframes pulse-dot {
    0%,100% { opacity:1; transform:scale(1); }
    50% { opacity:0.5; transform:scale(1.4); }
}

/* ── Big verdict text ── */
.verdict-title-ai {
    font-size: 3rem; font-weight: 900; line-height: 1;
    color: #DC2626; letter-spacing: -0.02em; margin: 14px 0 4px;
}
.verdict-title-real {
    font-size: 3rem; font-weight: 900; line-height: 1;
    color: #16A34A; letter-spacing: -0.02em; margin: 14px 0 4px;
}

/* ── Stat chip ── */
.stat-chip {
    background: #F8FAFC; border: 1px solid #E2E8F0;
    border-radius: 12px; padding: 0.9rem 1rem; text-align: center;
}
.stat-chip .label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #94A3B8; margin-bottom: 4px;
}
.stat-chip .value { font-size: 1.4rem; font-weight: 800; color: #0F172A; }

/* ── Progress bar ── */
.conf-bar-wrap {
    background: #F1F5F9; border-radius: 100px; height: 8px; overflow: hidden; margin-top: 8px;
}
.conf-bar-fill-ai {
    height: 100%; border-radius: 100px;
    background: linear-gradient(90deg, #F87171, #DC2626); transition: width 1s ease;
}
.conf-bar-fill-real {
    height: 100%; border-radius: 100px;
    background: linear-gradient(90deg, #4ADE80, #16A34A); transition: width 1s ease;
}

/* ── Divider ── */
.divider { height: 1px; background: #E2E8F0; margin: 1.5rem 0; }

/* ── Waveform bars (decorative) ── */
.wave-bars { display:flex; align-items:center; gap:3px; height:40px; margin:0 auto; width:fit-content; }
.wave-bar {
    width: 3px; border-radius: 2px;
    background: linear-gradient(to top, #7C3AED, #06B6D4);
    animation: wave-anim var(--dur) ease-in-out infinite alternate;
    animation-delay: var(--delay);
}
@keyframes wave-anim {
    from { transform: scaleY(0.3); opacity: 0.4; }
    to   { transform: scaleY(1.0); opacity: 1.0; }
}

/* ── Spinner text ── */
.analyzing-text {
    color: #7C3AED; font-size: 0.85rem; font-weight: 700;
    letter-spacing: 0.18em; text-transform: uppercase; text-align: center;
}

/* ── Feature pill ── */
.feature-pill {
    display: inline-flex; align-items: center; gap: 5px;
    background: #F1F5F9; border: 1px solid #E2E8F0;
    border-radius: 100px; padding: 5px 13px;
    font-size: 0.75rem; color: #64748B; font-weight: 500; margin: 3px;
}

/* ── How-it-works icon box ── */
.how-icon {
    width:36px; height:36px; border-radius:10px;
    display:flex; align-items:center; justify-content:center;
    font-size:1.1rem; flex-shrink:0;
}

/* ── Plotly override ── */
.js-plotly-plot .plotly .modebar { display: none !important; }
.js-plotly-plot { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load detector (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_detector():
    from detector import MUSEDetector
    return MUSEDetector()


# ── Plotly helpers ────────────────────────────────────────────────────────────
def make_gauge(probability: float, is_ai: bool) -> go.Figure:
    color = "#DC2626" if is_ai else "#16A34A"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability * 100, 1),
        number={
            "suffix": "%",
            "font": {"size": 42, "color": color, "family": "Inter"},
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 0,
                "tickcolor": "rgba(0,0,0,0)",
                "visible": False,
            },
            "bar": {"color": color, "thickness": 0.6},
            "bgcolor": "#F1F5F9",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  50], "color": "rgba(22,163,74,0.08)"},
                {"range": [50, 100], "color": "rgba(220,38,38,0.08)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.85,
                "value": probability * 100,
            },
        },
    ))

    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Inter"},
    )
    return fig


def make_waveform(waveform: np.ndarray, is_ai: bool, sr: int = 16000) -> go.Figure:
    color_line = "#DC2626" if is_ai else "#16A34A"
    color_fill = "rgba(220,38,38,0.10)" if is_ai else "rgba(22,163,74,0.10)"

    # Downsample for display
    target = 2000
    if len(waveform) > target:
        step = len(waveform) // target
        waveform = waveform[::step][:target]

    t = np.linspace(0, len(waveform) / sr, len(waveform))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=waveform,
        mode="lines",
        line=dict(color=color_line, width=1.2),
        fill="tozeroy",
        fillcolor=color_fill,
        hoverinfo="skip",
    ))

    fig.update_layout(
        height=160,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


def make_fakeprint_chart(fakeprint: np.ndarray, freq_range: np.ndarray, is_ai: bool) -> go.Figure:
    color = "#DC2626" if is_ai else "#16A34A"
    fill  = "rgba(220,38,38,0.08)" if is_ai else "rgba(22,163,74,0.08)"

    freqs = freq_range if len(freq_range) == len(fakeprint) else np.linspace(1000, 8000, len(fakeprint))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs / 1000, y=fakeprint,
        mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=fill,
        hovertemplate="%{x:.1f} kHz: %{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        height=180,
        margin=dict(l=0, r=0, t=10, b=30),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(
            title=dict(text="Frequency (kHz)", font=dict(size=11, color="#94A3B8")),
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=False,
            tickfont=dict(color="#94A3B8", size=10),
            tickcolor="rgba(0,0,0,0)",
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
    )
    return fig


# ── Animated wave bars HTML ───────────────────────────────────────────────────
def wave_bars_html(n: int = 28, color_a: str = "#7C3AED", color_b: str = "#06B6D4") -> str:
    bars = ""
    for i in range(n):
        h = 10 + (abs(np.sin(i * 0.6)) * 30)
        dur = f"{0.6 + (i % 5) * 0.15:.2f}s"
        delay = f"{(i * 0.07) % 0.8:.2f}s"
        bars += (
            f'<div class="wave-bar" style="height:{h:.0f}px;'
            f'--dur:{dur};--delay:{delay};'
            f'background:linear-gradient(to top,{color_a},{color_b})"></div>'
        )
    return f'<div class="wave-bars">{bars}</div>'


# ── Top nav bar ───────────────────────────────────────────────────────────────
logo_path   = Path(__file__).parent / "assets" / "logo.png"
client_logo = Path(__file__).parent / "assets" / "client_logo.png"

nav_left, nav_center, nav_right = st.columns([2, 3, 2])
with nav_left:
    if logo_path.exists():
        st.image(str(logo_path), width=300)
    else:
        st.markdown(
            '<div style="padding:1.2rem 0;">'
            '<span style="font-size:1.3rem;font-weight:800;color:#7C3AED;letter-spacing:-0.02em;">'
            'soundsafe<span style="color:#06B6D4;">.ai</span></span></div>',
            unsafe_allow_html=True,
        )

with nav_center:
    st.markdown(
        '<div style="text-align:center;padding:0.8rem 0;">'
        '<div style="font-size:0.65rem;font-weight:700;letter-spacing:0.28em;'
        'text-transform:uppercase;color:#94A3B8;margin-bottom:6px;">Powered by MAIA</div>'
        '<div style="font-size:2rem;font-weight:900;letter-spacing:-0.03em;'
        'background:linear-gradient(135deg,#7C3AED 0%,#06B6D4 100%);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
        'background-clip:text;line-height:1.1;">MAIA MUSE™</div>'
        '<div style="font-size:0.8rem;color:#94A3B8;margin-top:5px;font-weight:500;">'
        'AI-Generated Music Detection'
        '</div></div>',
        unsafe_allow_html=True,
    )

with nav_right:
    if client_logo.exists():
        st.image(str(client_logo), width=240)
    else:
        st.markdown(
            '<div style="height:60px;"></div>',
            unsafe_allow_html=True,
        )

# Divider
st.markdown('<div style="height:1px;background:#E2E8F0;margin:0.5rem 0 1.5rem;"></div>', unsafe_allow_html=True)

# ── Upload section ────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">🎵 &nbsp; Analyze Track</p>', unsafe_allow_html=True)

upload_col, info_col = st.columns([3, 2], gap="large")

with upload_col:
    st.markdown(
        '<div class="card">'
        '<div class="card-accent-top"></div>'
        '<div style="margin-bottom:1.25rem;">'
        '<div style="font-size:1.15rem;font-weight:800;color:#0F172A;margin-bottom:4px;">Analyze a Track</div>'
        '<div style="font-size:0.82rem;color:#64748B;">Upload any audio file and get an instant AI-generation verdict.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(wave_bars_html(36), unsafe_allow_html=True)
    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop audio file here, or click to browse",
        type=["mp3", "wav", "flac", "ogg", "m4a"],
        label_visibility="visible",
    )

    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:1rem;">'
        + "".join(f'<span class="feature-pill">{fmt}</span>' for fmt in ["MP3", "WAV", "FLAC", "OGG", "M4A"])
        + '<span class="feature-pill" style="margin-left:6px;">Max 100 MB</span>'
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with info_col:
    st.markdown("""
<div class="card" style="height:100%;">
<div class="card-accent-top"></div>
<div style="font-size:1.15rem;font-weight:800;color:#0F172A;margin-bottom:1.4rem;">How It Works</div>
<div style="display:flex;flex-direction:column;gap:0;">

  <div style="display:flex;gap:0;align-items:stretch;">
    <div style="display:flex;flex-direction:column;align-items:center;margin-right:16px;">
      <div style="width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,#7C3AED,#06B6D4);
                  display:flex;align-items:center;justify-content:center;flex-shrink:0;
                  font-size:0.75rem;font-weight:800;color:white;">01</div>
      <div style="width:2px;background:#E2E8F0;flex:1;margin:6px 0;"></div>
    </div>
    <div style="padding-bottom:1.4rem;">
      <div style="font-size:0.87rem;font-weight:700;color:#0F172A;margin-bottom:3px;">Spectral Fakeprint Extraction</div>
      <div style="font-size:0.77rem;color:#64748B;line-height:1.6;">STFT converts audio to frequency domain. A convex-hull baseline is subtracted to isolate periodic artifacts.</div>
    </div>
  </div>

  <div style="display:flex;gap:0;align-items:stretch;">
    <div style="display:flex;flex-direction:column;align-items:center;margin-right:16px;">
      <div style="width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,#7C3AED,#06B6D4);
                  display:flex;align-items:center;justify-content:center;flex-shrink:0;
                  font-size:0.75rem;font-weight:800;color:white;">02</div>
      <div style="width:2px;background:#E2E8F0;flex:1;margin:6px 0;"></div>
    </div>
    <div style="padding-bottom:1.4rem;">
      <div style="font-size:0.87rem;font-weight:700;color:#0F172A;margin-bottom:3px;">ONNX Classifier Inference</div>
      <div style="font-size:0.77rem;color:#64748B;line-height:1.6;">A 14 KB ONNX model — trained on 17,866 tracks — scores the residue pattern in under 2 seconds on CPU.</div>
    </div>
  </div>

  <div style="display:flex;gap:0;align-items:flex-start;">
    <div style="display:flex;flex-direction:column;align-items:center;margin-right:16px;">
      <div style="width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,#7C3AED,#06B6D4);
                  display:flex;align-items:center;justify-content:center;flex-shrink:0;
                  font-size:0.75rem;font-weight:800;color:white;">03</div>
    </div>
    <div>
      <div style="font-size:0.87rem;font-weight:700;color:#0F172A;margin-bottom:3px;">Verdict &amp; Evidence</div>
      <div style="font-size:0.77rem;color:#64748B;line-height:1.6;">Returns AI probability, confidence score, waveform, and the fakeprint chart as forensic evidence.</div>
    </div>
  </div>

</div>
</div>
""", unsafe_allow_html=True)


# ── Inference ─────────────────────────────────────────────────────────────────
if uploaded is not None:
    st.markdown('<div style="height:1px;background:#E2E8F0;margin:2rem 0;"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">📊 &nbsp; Analysis Results</p>', unsafe_allow_html=True)

    # Load model
    with st.spinner(""):
        loading_placeholder = st.empty()
        loading_placeholder.markdown(
            '<div style="text-align:center;padding:2rem;">'
            + wave_bars_html(40, "#7C3AED", "#06B6D4")
            + '<div class="analyzing-text" style="margin-top:16px;">Analyzing audio signature…</div>'
            + "</div>",
            unsafe_allow_html=True,
        )
        try:
            detector = load_detector()
        except Exception as e:
            loading_placeholder.empty()
            st.error(f"Failed to load model: {e}")
            st.stop()

    # Run inference
    try:
        suffix = "." + uploaded.name.split(".")[-1].lower()
        audio  = detector.load_audio_bytes(uploaded.getvalue(), suffix=suffix)
        wav_np = detector.get_waveform(audio)
        result = detector.predict(audio)
        time.sleep(0.3)  # brief pause for UX
        loading_placeholder.empty()
    except Exception as e:
        loading_placeholder.empty()
        st.error(f"Analysis failed: {e}")
        st.stop()

    prob   = result["probability"]
    is_ai  = result["is_ai"]
    conf   = result["confidence"]
    label  = result["label"]
    fp     = result["fakeprint"]

    # ── Result layout ──
    res_left, res_right = st.columns([1, 1], gap="large")

    with res_left:
        # Verdict card
        badge_html = (
            f'<div class="verdict-ai"><span class="verdict-dot dot-ai"></span>AI GENERATED</div>'
            if is_ai else
            f'<div class="verdict-real"><span class="verdict-dot dot-real"></span>AUTHENTIC MUSIC</div>'
        )
        title_class = "verdict-title-ai" if is_ai else "verdict-title-real"
        title_text  = "AI Generated" if is_ai else "Authentic"

        prob_color = '#DC2626' if is_ai else '#16A34A'
        bar_cls    = 'ai' if is_ai else 'real'
        verdict_lbl = 'AI' if is_ai else 'Real'
        card_html = textwrap.dedent(f"""
<div class="card">
{badge_html}
<div class="{title_class}" style="margin:16px 0 4px;">{title_text}</div>
<div style="color:#475569;font-size:0.85rem;font-weight:500;margin-bottom:24px;">{uploaded.name}</div>
<div class="divider"></div>
<div style="margin-bottom:20px;">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
<span style="font-size:0.78rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:#64748B;">AI Probability</span>
<span style="font-size:1rem;font-weight:700;color:{prob_color};">{prob:.1%}</span>
</div>
<div class="conf-bar-wrap"><div class="conf-bar-fill-{bar_cls}" style="width:{prob*100:.1f}%"></div></div>
</div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;">
<div class="stat-chip"><div class="label">Confidence</div><div class="value" style="color:{prob_color};">{conf:.0%}</div></div>
<div class="stat-chip"><div class="label">AI Score</div><div class="value">{prob:.3f}</div></div>
<div class="stat-chip"><div class="label">Verdict</div><div class="value" style="font-size:1.1rem;color:{prob_color};">{verdict_lbl}</div></div>
</div>
</div>
        """).strip()
        st.markdown(card_html, unsafe_allow_html=True)

        # Gauge
        st.plotly_chart(
            make_gauge(prob, is_ai),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with res_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        # Waveform
        st.markdown(
            '<div class="section-label">Waveform</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            make_waveform(wav_np, is_ai),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Spectral fakeprint
        st.markdown(
            '<div class="section-label">Spectral Fakeprint (1–8 kHz)</div>',
            unsafe_allow_html=True,
        )
        freq_range = np.linspace(1000, 8000, len(fp))
        st.plotly_chart(
            make_fakeprint_chart(fp, freq_range, is_ai),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # Explanation
        if is_ai:
            explanation = (
                "Elevated periodic peaks detected in the high-frequency band — "
                "a known artifact of AI music generators (Suno, Udio) caused by "
                "deconvolution layer aliasing."
            )
        else:
            explanation = (
                "No significant periodic artifacts detected. The spectral residue "
                "is consistent with naturally recorded or produced music."
            )

        st.markdown(
            f'<div style="background:#F8FAFC;border:1px solid #E2E8F0;'
            f'border-radius:12px;padding:14px 16px;font-size:0.8rem;color:#64748B;'
            f'line-height:1.6;margin-top:4px;">'
            f'<span style="color:#7C3AED;font-weight:700;">ℹ &nbsp;</span>{explanation}'
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Reset button ──
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    col_btn = st.columns([1, 1, 1])
    with col_btn[1]:
        if st.button("🔄 &nbsp; Analyze Another Track", use_container_width=True):
            st.rerun()

else:
    pass


# ── Product Overview ──────────────────────────────────────────────────────────
slide_path = Path(__file__).parent / "assets" / "slide.png"

st.markdown('<div style="height:2.5rem;"></div>', unsafe_allow_html=True)

st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:1.5rem;">
  <div style="height:1px;background:#E2E8F0;flex:1;"></div>
  <span style="background:#F3EEFF;border:1px solid #DDD6FE;color:#7C3AED;
               font-size:0.68rem;font-weight:700;letter-spacing:0.18em;
               text-transform:uppercase;padding:5px 14px;border-radius:100px;
               white-space:nowrap;">Product Overview</span>
  <div style="height:1px;background:#E2E8F0;flex:1;"></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;margin-bottom:2.5rem;">
  <h2 style="font-size:2rem;font-weight:900;color:#0F172A;letter-spacing:-0.03em;margin:0 0 10px;">
    The Industry&apos;s Most Precise<br>
    <span style="background:linear-gradient(135deg,#7C3AED,#06B6D4);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                 background-clip:text;">AI Music Detector</span>
  </h2>
  <p style="font-size:0.92rem;color:#64748B;max-width:560px;margin:0 auto;line-height:1.7;">
    MAIA MUSE™ finds the machine in the music — providing the certainty the human ear can no longer deliver.
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:2.5rem;">
  <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:14px;padding:1.5rem;
              box-shadow:0 1px 3px rgba(0,0,0,0.04),0 4px 12px rgba(0,0,0,0.03);">
    <div style="width:40px;height:40px;border-radius:10px;background:#F3EEFF;
                display:flex;align-items:center;justify-content:center;font-size:1.25rem;margin-bottom:12px;">⚠️</div>
    <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;
                color:#7C3AED;margin-bottom:6px;">The Problem</div>
    <div style="font-size:0.95rem;font-weight:700;color:#0F172A;margin-bottom:8px;line-height:1.3;">AI that sounds too human to catch</div>
    <div style="font-size:0.78rem;color:#64748B;line-height:1.65;">Human intuition can no longer detect the machine. Without a filter, the music economy is handed to the very models cannibalizing it.</div>
  </div>
  <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:14px;padding:1.5rem;
              box-shadow:0 1px 3px rgba(0,0,0,0.04),0 4px 12px rgba(0,0,0,0.03);">
    <div style="width:40px;height:40px;border-radius:10px;background:#ECFEFF;
                display:flex;align-items:center;justify-content:center;font-size:1.25rem;margin-bottom:12px;">🔬</div>
    <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;
                color:#0891B2;margin-bottom:6px;">The Power-Up</div>
    <div style="font-size:0.95rem;font-weight:700;color:#0F172A;margin-bottom:8px;line-height:1.3;">Full-spectrum forensic detection</div>
    <div style="font-size:0.78rem;color:#64748B;line-height:1.65;">Scans every second of a track, detecting frequency phase anomalies and pinpointing exactly when and where generative artifacts appear.</div>
  </div>
  <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:14px;padding:1.5rem;
              box-shadow:0 1px 3px rgba(0,0,0,0.04),0 4px 12px rgba(0,0,0,0.03);">
    <div style="width:40px;height:40px;border-radius:10px;background:#F0FDF4;
                display:flex;align-items:center;justify-content:center;font-size:1.25rem;margin-bottom:12px;">🛡️</div>
    <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.2em;text-transform:uppercase;
                color:#16A34A;margin-bottom:6px;">The Transformation</div>
    <div style="font-size:0.95rem;font-weight:700;color:#0F172A;margin-bottom:8px;line-height:1.3;">From doubt to catalog integrity</div>
    <div style="font-size:0.78rem;color:#64748B;line-height:1.65;">Users move from questioning if a track is real or fake — to enforcing absolute catalog integrity with forensic-grade evidence.</div>
  </div>
</div>
""", unsafe_allow_html=True)

if slide_path.exists():
    st.markdown("""
<div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:20px;
            padding:16px;box-shadow:0 4px 24px rgba(0,0,0,0.07),0 1px 4px rgba(0,0,0,0.04);">
  <div style="border-radius:12px;overflow:hidden;">
""", unsafe_allow_html=True)
    st.image(str(slide_path), use_container_width=True)
    st.markdown("""
  </div>
  <div style="display:flex;justify-content:center;margin-top:12px;">
    <span style="font-size:0.72rem;color:#94A3B8;font-weight:500;letter-spacing:0.05em;">
      MAIA MUSE™ &nbsp;·&nbsp; Serial A00002b &nbsp;·&nbsp; Confidential
    </span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="margin-top:4rem;padding:2rem 0 1.5rem;
                border-top:1px solid #E2E8F0;
                display:flex;justify-content:space-between;align-items:center;
                flex-wrap:wrap;gap:12px;">
        <div>
            <span style="font-size:0.85rem;font-weight:800;color:#7C3AED;">MAIA MUSE™</span>
            <span style="font-size:0.78rem;color:#94A3B8;margin-left:8px;">
                by SoundSafe &nbsp;·&nbsp; AI Music Detection Platform
            </span>
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
            <span class="feature-pill">Model: Fakeprint + ONNX</span>
            <span class="feature-pill">Accuracy: 99.88%</span>
            <span class="feature-pill">Suno · Udio Detection</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
