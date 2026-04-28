# Market Intel Agent

Market Intel Agent is a Streamlit application that turns a single free-text market brief into an executive-ready competitive intelligence memo with mandatory citations, an explicit research plan, and an automated red-team critique; it supports instant demo mode, live BYOK mode (OpenAI + Tavily), local Chroma caching for speed/cost control, and an optional memo revision loop based on critique findings.

## Architecture

```mermaid
flowchart TD
    A[User Brief] --> B[1) Scope Planner]
    B --> C[2) Parallel Intel Gather]
    C --> D[3) Memo Synthesizer]
    D --> E[4) Critic Red Team]
    E --> F[Render Tabs + Export]
    E -->|revise requested| G[Optional Reviser]
    G --> F
```

## Local Setup

```bash
git clone https://github.com/<your-user>/market-intel-agent.git
cd market-intel-agent
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

## API Keys

- OpenAI API key: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- Tavily API key (free tier available): [https://app.tavily.com/home](https://app.tavily.com/home)

In the app, leave Demo Mode ON for no-key walkthroughs. For live mode, enter keys in the sidebar; they are session-only and never stored.

## Streamlit Cloud Deployment

1. Push this folder to GitHub.
2. In Streamlit Community Cloud, create app from the repo/branch/path.
3. Add secrets:
   - `OPENAI_API_KEY`
   - `TAVILY_API_KEY`
4. Deploy and run.

## Cost Notes

- Typical 3-company run target: **~$0.27-$0.43** depending on model selection and cache hit rate.
- Recommended monthly budget for light team use: **$40-$120**.
- Hard cap recommendation: set **$20 hard cap** on the OpenAI key in billing controls.

## Sample Memo (from `samples/sample_run.json`)

> **Bottom Line**: Independent inference vendors are gaining technical mindshare, but hyperscalers retain enterprise buying leverage [1][4][5].
>
> **Recommended Actions**:
> 1) Build co-sell options with top specialist platforms while preserving cloud-neutral API strategy [1][2][3].  
> 2) Track procurement friction indicators as a leading share signal [4][5].  
> 3) Re-run assessment monthly with fresh investor signals [2][6].

## Known Limitations

- Streamlit Community Cloud instances can sleep after inactivity windows (often around 12 hours).
- Tavily free tier limits can constrain monthly query volume (for example 1,000 searches/month).
- News publication dates are best-effort parsed from search metadata.
- `robots.txt` is not respected by the lightweight webpage fetch helper.
