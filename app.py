"""
LLM-Powered Equity Research Assistant
Neela Patil

Enter a stock ticker to get key fundamentals, technical indicators, recent news
and an analyst-style research report written by an LLM on Groq (default: openai/gpt-oss-120b).

The Groq API key is read from Streamlit secrets (GROQ_API_KEY), never from code.
"""

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="Equity Research Assistant", page_icon="📈", layout="wide")

MAX_REPORTS_PER_SESSION = 5
DEFAULT_MODEL = "openai/gpt-oss-120b"  # Groq retired llama-3.3-70b-versatile on 16 Aug 2026

UP, DOWN, LINE, ACCENT, MUTED = "#2E7D5B", "#B23A48", "#1E4D8C", "#C08A2B", "#8A94A6"


# ─── DATA ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock(ticker: str):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    try:
        info = stock.info or {}
    except Exception:
        info = {}
    try:
        news = (stock.news or [])[:8]
    except Exception:
        news = []
    return info, hist, news


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_peers(tickers: tuple):
    out = {}
    for t in tickers:
        try:
            h = yf.Ticker(t).history(period="1y")["Close"]
            if len(h):
                out[t] = h / h.iloc[0] * 100
        except Exception:
            pass
    return pd.DataFrame(out)


def fmt(val, kind=None, currency=""):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    try:
        if kind == "pct":
            return f"{val * 100:.2f}%"
        if kind == "B":
            return f"{currency}{val / 1e9:,.2f}B"
        if kind == "x":
            return f"{val:.2f}x"
        if kind == "price":
            return f"{currency}{val:,.2f}"
        return f"{round(val, 2)}"
    except Exception:
        return str(val)


def dividend_yield(info):
    # yfinance changed 'dividendYield' units across versions, so compute it directly.
    rate, price = info.get("dividendRate"), info.get("currentPrice")
    if rate and price:
        return rate / price
    return info.get("trailingAnnualDividendYield")


def fundamentals_table(info):
    cur = info.get("currency", "")
    cur = "$" if cur == "USD" else (f"{cur} " if cur else "")
    de = info.get("debtToEquity")  # reported by Yahoo as a percentage
    return {
        "Current price": fmt(info.get("currentPrice"), "price", cur),
        "52-week high": fmt(info.get("fiftyTwoWeekHigh"), "price", cur),
        "52-week low": fmt(info.get("fiftyTwoWeekLow"), "price", cur),
        "Market cap": fmt(info.get("marketCap"), "B", cur),
        "P/E (TTM)": fmt(info.get("trailingPE"), "x"),
        "Forward P/E": fmt(info.get("forwardPE"), "x"),
        "P/B": fmt(info.get("priceToBook"), "x"),
        "EV/EBITDA": fmt(info.get("enterpriseToEbitda"), "x"),
        "Revenue (TTM)": fmt(info.get("totalRevenue"), "B", cur),
        "Gross margin": fmt(info.get("grossMargins"), "pct"),
        "Operating margin": fmt(info.get("operatingMargins"), "pct"),
        "Net margin": fmt(info.get("profitMargins"), "pct"),
        "ROE": fmt(info.get("returnOnEquity"), "pct"),
        "ROA": fmt(info.get("returnOnAssets"), "pct"),
        "Debt/Equity": fmt(de / 100 if de is not None else None, "x"),
        "Current ratio": fmt(info.get("currentRatio")),
        "Dividend yield": fmt(dividend_yield(info), "pct"),
        "Beta": fmt(info.get("beta")),
        "Analyst target price": fmt(info.get("targetMeanPrice"), "price", cur),
        "Analyst consensus": (info.get("recommendationKey") or "N/A").replace("_", " ").title(),
    }


def add_indicators(hist):
    df = hist.copy()
    c = df["Close"]
    df["MA_20"], df["MA_50"], df["MA_200"] = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss)
    std20 = c.rolling(20).std()
    df["BB_upper"], df["BB_lower"] = df["MA_20"] + 2 * std20, df["MA_20"] - 2 * std20
    df["MACD"] = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["Daily_Return"] = c.pct_change()
    return df


def technical_summary(df):
    last = df.iloc[-1]
    vol = df["Daily_Return"].std() * np.sqrt(252)
    ret_1y = df["Close"].iloc[-1] / df["Close"].iloc[0] - 1
    rsi = last["RSI"]
    rsi_state = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"

    def vs(ma):
        return "N/A" if pd.isna(last[ma]) else ("Above" if last["Close"] > last[ma] else "Below")

    return {
        "rsi": rsi, "rsi_state": rsi_state,
        "macd": "Bullish" if last["MACD"] > last["Signal"] else "Bearish",
        "vs50": vs("MA_50"), "vs200": vs("MA_200"),
        "vol": vol, "ret_1y": ret_1y,
    }


def headline(article):
    content = article.get("content") or {}
    title = content.get("title") or article.get("title") or "Untitled"
    link = ((content.get("canonicalUrl") or {}).get("url")
            or (content.get("clickThroughUrl") or {}).get("url")
            or article.get("link"))
    source = (content.get("provider") or {}).get("displayName") or article.get("publisher") or ""
    return title, link, source


# ─── LLM ─────────────────────────────────────────────────────────────────────
def build_prompt(company, ticker, fundamentals, tech, headlines):
    fund = "\n".join(f"{k}: {v}" for k, v in fundamentals.items())
    tech_txt = (
        f"RSI (14d): {tech['rsi']:.1f} ({tech['rsi_state']})\n"
        f"MACD: {tech['macd']}\nPrice vs MA50: {tech['vs50']}\nPrice vs MA200: {tech['vs200']}\n"
        f"Annualised volatility: {tech['vol'] * 100:.1f}%\n1Y return: {tech['ret_1y'] * 100:.1f}%"
    )
    news = "\n".join(f"- {h}" for h in headlines) or "- No recent headlines available"
    return f"""You are a senior equity research analyst at a top investment bank.
Write a structured, professional research report for {company} ({ticker}).

FUNDAMENTALS:
{fund}

TECHNICAL INDICATORS:
{tech_txt}

RECENT NEWS HEADLINES:
{news}

Structure the report with these exact sections:
1. **Executive Summary** (3-4 sentences, include a clear BUY/HOLD/SELL recommendation)
2. **Business Overview** (sector, business model, competitive position)
3. **Fundamental Analysis** (valuation multiples, margins, balance sheet health)
4. **Technical Analysis** (price trend, momentum, support/resistance)
5. **Key Risks** (3-4 specific risks)
6. **Investment Thesis** (bull case, bear case, and 12-month price target rationale)

Be specific and data-driven, and use the actual numbers provided. If a figure is N/A, do not invent it."""


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def generate_report(prompt: str, model: str, day: str):
    # 'day' keeps the cache to one report per ticker per day, which protects the API quota.
    from groq import Groq

    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a senior equity research analyst. Write detailed, data-driven reports."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_completion_tokens=4000,
        **extra,
    )
    return resp.choices[0].message.content


def has_key():
    try:
        return bool(st.secrets.get("GROQ_API_KEY"))
    except Exception:
        return False


# ─── CHARTS ──────────────────────────────────────────────────────────────────
def price_chart(df, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(30,77,140,0.10)", name="Bollinger bands", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color=LINE, width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_50"], name="50-day MA", line=dict(color=ACCENT, width=1.2, dash="dash")))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_200"], name="200-day MA", line=dict(color=DOWN, width=1.2, dash="dash")))
    fig.update_layout(title=title, height=420, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"), hovermode="x unified")
    return fig


def indicator_charts(df):
    fig = make_subplots(rows=2, cols=2, subplot_titles=("RSI (14-day)", "MACD", "Volume (millions)",
                                                        "Daily return distribution (%)"),
                        vertical_spacing=0.18, horizontal_spacing=0.08)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color=LINE, width=1.4), name="RSI"), 1, 1)
    for lvl, col in ((70, DOWN), (30, UP)):
        fig.add_hline(y=lvl, line=dict(color=col, dash="dash", width=1), row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=1, col=1)
    hist_vals = df["MACD"] - df["Signal"]
    fig.add_trace(go.Bar(x=df.index, y=hist_vals, marker_color=np.where(hist_vals >= 0, UP, DOWN),
                         opacity=0.5, name="Histogram"), 1, 2)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], line=dict(color=LINE, width=1.4), name="MACD"), 1, 2)
    fig.add_trace(go.Scatter(x=df.index, y=df["Signal"], line=dict(color=ACCENT, width=1.4), name="Signal"), 1, 2)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"] / 1e6,
                         marker_color=np.where(df["Daily_Return"].fillna(0) >= 0, UP, DOWN),
                         opacity=0.7, name="Volume"), 2, 1)
    fig.add_trace(go.Histogram(x=df["Daily_Return"].dropna() * 100, nbinsx=50, marker_color=LINE,
                               opacity=0.8, name="Returns"), 2, 2)
    fig.add_vline(x=0, line=dict(color=DOWN, dash="dash", width=1), row=2, col=2)
    fig.update_layout(height=620, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ─── PAGE ────────────────────────────────────────────────────────────────────
st.title("Equity Research Assistant")
st.caption("Fundamentals, technicals and news for any listed stock, summarised into an analyst-style "
           "report by an LLM. Built by Neela Patil. For learning purposes only, not investment advice.")

with st.sidebar:
    st.header("Pick a stock")
    with st.form("ticker_form"):
        ticker = st.text_input("Ticker", value="AAPL",
                               help="Yahoo Finance symbol, e.g. NVDA, TSLA, RELIANCE.NS, TCS.NS").strip().upper()
        peers_raw = st.text_input("Peers to compare", value="MSFT, GOOGL, META")
        want_report = st.checkbox("Write the AI research report", value=True)
        submitted = st.form_submit_button("Analyse", type="primary", width="stretch")
    st.caption("Market data from Yahoo Finance via yfinance. Reports by an open-weight LLM on Groq.")

if "ticker" not in st.session_state or submitted:
    st.session_state.ticker = ticker
    st.session_state.peers = tuple(p.strip().upper() for p in peers_raw.split(",") if p.strip())
    st.session_state.want_report = want_report

T = st.session_state.ticker
if not T:
    st.info("Enter a ticker in the sidebar to start.")
    st.stop()

with st.spinner(f"Fetching market data for {T}…"):
    try:
        info, hist, news = fetch_stock(T)
    except Exception as e:
        st.error(f"Couldn't reach Yahoo Finance for {T} ({e.__class__.__name__}). Wait a minute and try again.")
        st.stop()

if hist is None or hist.empty or len(hist) < 30:
    st.error(f"No price history found for “{T}”. Check the symbol; Indian stocks need a suffix like .NS or .BO.")
    st.stop()

company = info.get("longName") or info.get("shortName") or T
df = add_indicators(hist)
tech = technical_summary(df)
fundamentals = fundamentals_table(info)

st.subheader(f"{company} ({T})")
meta = [x for x in (info.get("sector"), info.get("industry"), info.get("country")) if x]
if meta:
    st.write(", ".join(meta))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Price", fundamentals["Current price"] if fundamentals["Current price"] != "N/A" else f"{df['Close'].iloc[-1]:,.2f}")
c2.metric("1-year return", f"{tech['ret_1y'] * 100:.1f}%")
c3.metric("RSI (14d)", f"{tech['rsi']:.1f}", tech["rsi_state"], delta_color="off")
c4.metric("MACD", tech["macd"])
c5.metric("Annualised volatility", f"{tech['vol'] * 100:.1f}%")

tab_report, tab_charts, tab_fund, tab_news, tab_peers = st.tabs(
    ["Research report", "Charts", "Fundamentals", "News", "Peer comparison"])

headlines_full = [headline(a) for a in news]

with tab_report:
    if not st.session_state.want_report:
        st.info("Tick “Write the AI research report” in the sidebar and press Analyse to generate one.")
    elif not has_key():
        st.warning("The report generator isn't configured: add GROQ_API_KEY in the app's secrets.")
    else:
        st.session_state.setdefault("reports_made", set())
        key = f"{T}-{date.today()}"
        if key not in st.session_state.reports_made and len(st.session_state.reports_made) >= MAX_REPORTS_PER_SESSION:
            st.warning(f"This demo writes up to {MAX_REPORTS_PER_SESSION} reports per visit. Refresh the page to start a new session.")
        else:
            prompt = build_prompt(company, T, fundamentals, tech, [h[0] for h in headlines_full])
            model = st.secrets.get("GROQ_MODEL", DEFAULT_MODEL)
            with st.spinner("Writing the research report…"):
                try:
                    report = generate_report(prompt, model, str(date.today()))
                    st.session_state.reports_made.add(key)
                except Exception as e:
                    report = None
                    st.error(f"The report couldn't be generated ({e.__class__.__name__}). "
                             "The LLM service may be busy; try again in a minute.")
            if report:
                st.markdown(report)
                st.download_button("Download report (.md)",
                                   f"# Equity Research Report: {company} ({T})\n\n{report}",
                                   file_name=f"{T}_research_report.md", mime="text/markdown")
                st.caption("Generated by an LLM from the data on this page. It can be wrong; check the figures.")

with tab_charts:
    st.plotly_chart(price_chart(df, f"{company}: price, moving averages and Bollinger bands"), width="stretch")
    st.plotly_chart(indicator_charts(df), width="stretch")

with tab_fund:
    st.dataframe(pd.DataFrame(fundamentals.items(), columns=["Metric", "Value"]),
                 hide_index=True, width="stretch", height=740)

with tab_news:
    if not headlines_full:
        st.info("No recent headlines from Yahoo Finance for this ticker.")
    for title, link, source in headlines_full:
        line = f"[{title}]({link})" if link else title
        st.markdown(f"- {line}" + (f"  \n  <span style='color:{MUTED}'>{source}</span>" if source else ""),
                    unsafe_allow_html=True)

with tab_peers:
    tickers = tuple(dict.fromkeys((T,) + st.session_state.peers))
    with st.spinner("Loading peers…"):
        peers = fetch_peers(tickers)
    if peers.shape[1] < 2:
        st.info("Add at least one valid peer ticker in the sidebar to compare.")
    else:
        fig = go.Figure()
        for col in peers.columns:
            fig.add_trace(go.Scatter(x=peers.index, y=peers[col], name=col,
                                     line=dict(width=3 if col == T else 1.4)))
        fig.add_hline(y=100, line=dict(color=MUTED, dash="dash", width=1))
        fig.update_layout(title="1-year performance, indexed to 100", height=460,
                          margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified")
        st.plotly_chart(fig, width="stretch")
        last = peers.ffill().iloc[-1].sub(100).sort_values(ascending=False)
        st.dataframe(last.map(lambda v: f"{v:+.1f}%").rename("1-year return").to_frame(),
                     width="content")
