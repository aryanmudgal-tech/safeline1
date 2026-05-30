# Safeline — Police Voice Documentation Agent

> Built for the YC Voice Agents Hackathon on top of the Pipecat starter kit.

Safeline is a voice agent that police officers call after an incident. The officer narrates what happened in natural speech; the agent listens, asks targeted follow-up questions to fill gaps, then generates the required police documentation (incident / arrest / use-of-force / accident reports) and texts the officer a link to review, edit, and approve the drafts. Every correction the officer makes is fed back to improve future reports.

**How it beats Axon Draft One**

| Draft One | Safeline |
| --- | --- |
| Deletes the AI draft after export (no audit trail) | Every incident is persisted with both the **immutable AI draft** and the **officer-edited version** in `server/data/incidents/` |
| Hallucinates from ambient audio ("officer shapeshifted into a frog") | `extraction_accuracy` evaluator flags any field not grounded in the transcript; the model is instructed to use `null` over invention |
| Passive (only summarizes body-cam footage) | Active two-phase voice interview: open narrative → one-question-at-a-time gap filling |
| Never learns from corrections | Three-layer auto-improvement loop; Layer 1 injects similar past officer corrections as few-shot examples at generation time |

## Architecture

The Pipecat pipeline, Twilio transport, and Pipecat Cloud deploy are **unchanged** from the starter kit. Only the business logic was swapped in.

| File | Role |
| --- | --- |
| `server/bot-gpt.py` | Voice bot (Gradium STT → GPT → Gradium TTS). **Run this one first.** |
| `server/bot-nemotron.py` | Same bot on NVIDIA Nemotron STT+LLM (use once endpoints are provided). |
| `server/police_agent.py` | Shared police logic: two-phase system prompt, roster lookup, LLM tools, transcript capture, report generation + review-link SMS. |
| `server/report_generator.py` | LLM field extraction + the permanent, file-backed report store (AI draft + officer version + diff). |
| `server/templates/*.json` | Required-field definitions for each report type. |
| `server/mock_data/officer_roster.json` | Badge → officer lookup. |
| `server/mock_data/demo_scenarios.json` | Three pre-written 60s narratives for testing without calls. |
| `server/improvement/correction_store.py` | Auto-improvement Layer 1: semantic store of officer corrections (faiss + MiniLM if installed, token-overlap fallback otherwise). |
| `server/review_ui/` | Standalone FastAPI app + SPA where the officer reviews/edits/approves reports. |
| `server/cekura_eval.py` | Four evaluators: completeness, extraction accuracy, escalation correctness, question coverage. |

## Run the demo (under 3 minutes)

```bash
cd server
uv sync                 # add `--extra improve` for faiss/MiniLM dense retrieval (optional)

# Terminal 1 — voice bot (WebRTC at http://localhost:7860)
uv run bot-gpt.py

# Terminal 2 — review UI (http://localhost:8080)
uv run python review_ui/api.py
```

Open http://localhost:7860, click **Connect**, and run a scenario from `mock_data/demo_scenarios.json` (e.g. give badge **3892**, then read the DUI narrative). When the agent says it's generating reports, it stores them and (if Twilio creds + a caller number are set) texts the review link. The link opens the review UI; the link is also logged to the bot's console for local WebRTC testing where there's no phone number.

## Six-digit review portal

The review UI is now code-first. Open the review server URL, enter the six-digit code spoken by the voice agent, and the portal will poll until reports are ready.

API contract:

```http
GET /api/reports/{6-digit-code}
POST /api/reports/{6-digit-code}/approve
```

`GET` returns `status: "generating"` while report generation is still running, then `status: "pending_review"` with editable report payloads. `POST` accepts:

```json
{
  "edited_reports": {
    "incident_report": {
      "incident_date": "2026-05-30"
    }
  }
}
```

Approvals preserve the immutable AI draft, store the officer-edited version, compute diffs, and feed those corrections into the learning loop.

**Test report generation without a call:**

```bash
uv run python -c "import asyncio,json; from dotenv import load_dotenv; load_dotenv(); \
from report_generator import generate_reports; \
s=json.load(open('mock_data/demo_scenarios.json'))[1]; \
print(asyncio.run(generate_reports('OFFICER: '+s['narrative'], s['expected_reports'], s['badge_number'])))"
```

**Evaluate a stored incident (Cekura-style):**

```bash
uv run python cekura_eval.py --all          # or pass an INC-... id
```

## Cekura integration

The four evaluators in `cekura_eval.py` mirror the custom evaluators to register in the Cekura dashboard:

1. `report_completeness` — are all required fields populated (non-null)?
2. `extraction_accuracy` — was the right info extracted from the transcript (no hallucinations)?
3. `escalation_correctness` — did emergencies get flagged for human review instead of documented?
4. `question_coverage` — did the agent ask about every required field it didn't hear?

Drive Cekura end-to-end from Claude Code:

```
/plugin marketplace add cekura-ai/cekura-skills
/plugin install cekura@cekura-skills
/cekura-report
```

`GET /api/reports/{incident_id}/diff` exposes the officer-vs-AI corrections for Cekura to consume.

## Auto-improvement loop

When an officer approves a report with edits, `approve_report()` computes the field-level diff and calls `correction_store.add_correction()`. Those corrections are embedded and, on the next generation, the most similar ones are injected into the extraction prompt as few-shot examples — so the agent stops repeating mistakes. Seed examples live in `mock_data/synthetic_corrections.json`.

> Safety: if the officer reports a serious injury, fatality, or ongoing emergency, the agent escalates to a supervisor and **never** generates a report.

---

# YC Voice Agents Hackathon

Welcome to the YC Voice Agents Hackathon, hosted by [Cekura](https://cekura.com) and [Daily](https://daily.co), in partnership with [NVIDIA](https://nvidia.com), [AWS](https://aws.amazon.com), and [Twilio](https://twilio.com).

The goal of this event is to learn about building, scaling, evaluating, and continuously improving voice agents.

## Schedule, rules, and prizes

This is a one-day event. Please arrive by 8:30. We'll kick things off at 9:00.

### Schedule

  - 8:00 AM – Doors open & registration
  - 8:30 AM – Breakfast
  - 9:00 AM – Welcome / Hackathon begins
  - 12:00 PM – Lunch
  - 6:00 PM – Submissions due
  - 6:00 - 8:00 PM – Dinner, demos, and conversation
  - 8:00 PM – Judges' presentations
  - 9:00 PM – We all go home

### General guidance

First of all, please respect the YC space. We very much appreciate YC hosting these events. Stay in the designated areas, clean up after meals, and in general be a good guest.

Build something new for this hackathon. Use the tools from Cekura to evaluate and improve the performance of what you build. Use Pipecat as the orchestration framework for your voice agent. We also encourage you to use the open source models from NVIDIA, but it's okay to use any models that work well for your project.

There will be engineers from Cekura, Daily, NVIDIA, AWS, and Twilio available to help you with your project. Don't hesitate to find us.

Judging will start at 6:00. In general, the judges want to showcase interesting projects rather than just pick winners. So don't worry too much about what the judges are looking for in a project. Build something that demonstrates creativity, is interesting on a technical level, or solves a real problem! But do keep in mind that the judges want to see great examples of using Cekura to improve voice agent performance, and using open source models from NVIDIA.


# Tech stack and starting points.

This repo contains two versions of a voice agent built with [Pipecat](https://pipecat.ai).

The demo bot **Field & Flower** is a neighborhood flower shop: callers order a bouquet for delivery while the bot looks up the catalog, captures delivery details, and places the order. All backend calls are mocked, so the starter runs with nothing but AI service keys.

## Version 1 — GPT-4.1

You can start with this before the hackathon, if you want to. Or test GPT-4.1 and Nemotron side-by-side during the hackathon, using Cekura.

This bot only requires a Gradium API key and an OpenAI API key. Sign up for free at [Gradium](https://gradium.ai). We'll provide a code for Gradium credits, during the event.

- **STT:** [Gradium](https://gradium.ai)
- **LLM:** [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) (GPT-4.1)
- **TTS:** [Gradium](https://gradium.ai)
- **Transports:** SmallWebRTC (local dev) and [Twilio](https://www.twilio.com/en-us) (production telephony)
- **Deploy target:** [Pipecat Cloud](https://pipecat.daily.co)

## Version 2

NVIDIA models hosted on AWS, available during the hackathon. We'll share endpoints for the NVIDIA ASR (STT) and LLM models at the beginning of the day.

- **STT:** [Nemotron Speech Streaming](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
- **LLM:** [Nemotron 3 Super 120B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16)
- **TTS:** [Gradium](https://gradium.ai)
- **Transports:** SmallWebRTC (local dev) and Twilio (production telephony)
- **Deploy target:** [Pipecat Cloud](https://pipecat.daily.co)

## Develop locally

Get the bot running over WebRTC in your browser before you push to the cloud or wire up the phone, for a faster iteration loop.

### Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) package manager
- API keys for [OpenAI](https://platform.openai.com) and [Gradium](https://gradium.ai)

### Setup

1. **Clone and enter the server directory:**

   ```bash
   git clone https://github.com/pipecat-ai/yc-voice-agents-hackathon.git
   cd yc-voice-agents-hackathon/server
   ```

2. **Configure API keys:**

   ```bash
   cp .env.example .env
   # Edit .env and fill in OPENAI_API_KEY, GRADIUM_API_KEY.
   # TWILIO_* keys are only needed when you wire up the phone (next section).
   ```

3. **Install dependencies:**

   ```bash
   uv sync
   ```

4. **Run the bot:**

   ```bash
   # run one or the other of these
   uv run bot-gpt.py
   uv run bot-nemotron.py
   ```

   Open [http://localhost:7860](http://localhost:7860) and click **Connect** to start talking. First launch takes ~20s while Pipecat downloads VAD and turn-detection models.

## Deploy to Pipecat Cloud

Once the bot works locally, deploy to Pipecat Cloud and connect it to a Twilio phone number so anyone can call in.

### Prerequisites

1. [Sign up for Pipecat Cloud](https://pipecat.daily.co/sign-up)
2. Install the [Pipecat CLI](https://github.com/pipecat-ai/pipecat-cli) and log in:

   ```bash
   uv tool install pipecat-ai-cli
   pc cloud auth login
   ```

### Configure Twilio

1. [Add credits / upgrade your Twilio account](https://twil.io/yc-hack)

2. [Buy a phone number](https://help.twilio.com/articles/223135247) with voice capability.

3. Get your Pipecat Cloud organization name:

   ```bash
   pc cloud organizations list
   ```

4. [Create a TwiML Bin](https://www.twilio.com/docs/serverless/twiml-bins/getting-started#create-a-new-twiml-bin) with this configuration:

   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <Response>
     <Connect>
       <Stream url="wss://api.pipecat.daily.co/ws/twilio">
         <Parameter name="_pipecatCloudServiceHost"
           value="flower-bot.YOUR_ORG_NAME"/>
       </Stream>
     </Connect>
   </Response>
   ```

   Replace `YOUR_ORG_NAME` with the org name from step 2.

5. [Attach the TwiML Bin](https://www.twilio.com/docs/serverless/twiml-bins/getting-started#wire-your-twiml-bin-up-to-an-incoming-phone-call) to your Twilio number: Go to [your phone numbers](https://console.twilio.com/go?to=/account/__account__/us1/senders-hub/list/phone-numbers/inventory) → select your
number → under **Voice Configuration**, set method to the **TwiML Bin** you created → Save.

6. [Optional] Use [Twilio Dev phone](https://www.twilio.com/docs/labs/dev-phone) for testing.

### Review the deployment configuration

Your deployment details are specified in the `pcc-deploy.toml` file. You can learn more about options in the [docs](https://docs.pipecat.ai/api-reference/cli/cloud/deploy#configuration-file-pcc-deploy-toml).

### Upload secrets

```bash
pc cloud secrets set flower-bot-secrets --file .env
```

This uploads everything from `.env` to Pipecat Cloud's secure storage. The bot reads from there at runtime, so you don't bake keys into the image.

### Deploy

Build and run your bot on Pipecat Cloud:

```bash
pc cloud deploy
```

Learn more about [cloud builds](https://docs.pipecat.ai/pipecat-cloud/guides/cloud-builds).

### Call your bot

Dial the Twilio number you set up. 🌷

## Test your agent with Cekura

[Cekura](https://cekura.com) tests and observes voice agents. For this hackathon, use it to **test the Pipecat bot you build in this repo** — run real conversations against it, score the transcripts, and fix what's failing before you demo.

### Sign up

Create your account at **[dashboard.cekura.ai](https://dashboard.cekura.ai)**. If you're approved for this hackathon, just sign up and your credits will show up automatically. If you don't see them, find someone from the Cekura team, they're on-site.

### Onboarding (or skip it)

On first login you'll land on a short setup flow that helps you create your first agent and test. Feel free to click through it — **or hit _Skip_** and jump straight to the dashboard if you'd rather set things up yourself. Either way takes a minute.

### Recommended: start by testing your agent (via Claude Code)

The fastest path — and what we recommend for the hackathon — is to drive Cekura from **Claude Code** using our MCP server + skills. You stay in your terminal, and Cekura handles agent creation, scenario generation, and running the test.

**1. Install the Cekura skills + MCP** (Claude Code marketplace plugin — bundles the skills, slash commands, and auto-configured MCP server):

```bash
/plugin marketplace add cekura-ai/cekura-skills
/plugin install cekura@cekura-skills
```

Repo: [github.com/cekura-ai/cekura-skills](https://github.com/cekura-ai/cekura-skills) · Full setup + other agents (Cursor, Codex, etc.): **[docs.cekura.ai → Claude Code guide](https://docs.cekura.ai/mcp/claude-code-guide)** and **[Skills](https://docs.cekura.ai/mcp/skills)**.

**2. Run an end-to-end test** of your agent with a single command:

```
/cekura-report
```

This spins up anything from 10–20 evaluators (what Cekura calls test cases), runs scenarios against your Pipecat agent, and gives you back a full report — transcripts, scores, and what failed — so you can iterate fast.

> When connecting your agent, **select `Pipecat` as the provider.** Details: [docs.cekura.ai → Pipecat](https://docs.cekura.ai/documentation/integrations/pipecat/automated).

## Learn more

### Pipecat

- [Pipecat Documentation](https://docs.pipecat.ai/)
- [Pipecat Cloud Deployment](https://docs.pipecat.ai/pipecat-cloud/introduction)
- [Pipecat Examples](https://github.com/pipecat-ai/pipecat-examples)
- [Pipecat Discord](https://discord.gg/pipecat)

### Twilio

- [Twilio Developer Hub](https://www.twilio.com/en-us/developers)
- [Twilio Documentation](https://www.twilio.com/docs)
- [Twilio Dev phone](https://www.twilio.com/docs/labs/dev-phone)

### Cekura

- [Claude Code guide](https://docs.cekura.ai/mcp/claude-code-guide) — MCP + skills setup
- [Cekura skills](https://docs.cekura.ai/mcp/skills) — all slash commands
- [Pipecat integration](https://docs.cekura.ai/documentation/integrations/pipecat/automated)
- [Cekura docs](https://docs.cekura.ai) · [dashboard](https://dashboard.cekura.ai)
