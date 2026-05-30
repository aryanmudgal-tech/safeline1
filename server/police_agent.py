"""Safeline police-documentation agent: shared logic for both bot variants.

This module holds everything that makes the bot a *police documentation
assistant* rather than the flower-shop starter, so ``bot-gpt.py`` and
``bot-nemotron.py`` stay nearly identical to the original and only swap in
this business logic:

  * the two-phase system prompt (open narrative -> gap filling),
  * officer roster lookup by badge number,
  * the LLM tools (lookup_officer, escalate_emergency, generate_reports,
    end_session),
  * transcript capture, report generation, persistence, and the review-link SMS.

The Pipecat pipeline, Twilio transport, and Pipecat Cloud deploy are untouched.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

import aiohttp
from loguru import logger
from pipecat.frames.frames import EndTaskFrame, FunctionCallResultProperties
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallParams

from report_generator import generate_reports as generate_reports_impl
from report_generator import store_reports

_SERVER_DIR = Path(__file__).resolve().parent
_ROSTER_PATH = _SERVER_DIR / "mock_data" / "officer_roster.json"

VALID_REPORT_TYPES = {
    "incident_report",
    "arrest_report",
    "use_of_force",
    "accident_report",
}

# The synthetic kickoff message we inject to start the call. Filtered out of the
# transcript so it never ends up in a generated report.
GREETING_KICKOFF = (
    "An officer just called the Safeline documentation line. Greet them briefly "
    "and ask for their badge number so you can look up their name. Do not ask "
    "anything else yet."
)


# --------------------------------------------------------------------------- #
# System prompt
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are Safeline, a police documentation assistant. Police officers call you \
after handling an incident to document what happened.

Your job has two phases:

PHASE 1 - OPEN NARRATIVE:
When the officer calls, greet them and ask for their badge number. Call the \
lookup_officer tool with the badge number to look up their name from the roster. \
Then say: "Go ahead, Officer [name]. Tell me what happened. I'll listen and ask \
follow-up questions when you're done."
Listen to the officer's full narrative without interrupting. While listening, \
silently extract:
- Incident type (domestic disturbance, traffic stop, theft, use of force, etc.)
- Location and time
- Names of suspect, victim, witnesses
- Whether an arrest was made
- Whether force was used
- Injuries
- Evidence collected
- Charges

PHASE 2 - GAP FILLING:
After the officer finishes their narrative, suggest which reports to generate:
- Always: Incident Report
- If arrest mentioned: Arrest Report
- If force used: Use of Force Report
- If vehicle accident: Accident Report

Ask ONE targeted follow-up question at a time to fill missing required fields. \
Prioritize in this order:
1. Legally critical: Was Miranda given? What force was used and why? What charges \
are being filed?
2. Identity: Suspect full name, DOB, victim name, witness contacts
3. Timeline: Exact time of arrest, sequence of events
4. Evidence: Items collected, photos taken
5. Administrative: Case number, assisting officers

When all critical fields are filled, say: "I have everything I need. I'm \
generating your reports now and will send you a link to review them." In that \
SAME turn, call the generate_reports tool with the list of report types you \
decided on. After it returns, briefly confirm the review link was sent, then \
call end_session.

RULES:
- Ask only ONE question at a time.
- Never ask about something already mentioned in the narrative.
- Never make up information - only use what the officer told you.
- If the officer mentions a serious injury, a fatality, or an ongoing emergency, \
immediately say "This sounds like it needs immediate attention - please contact \
your supervisor directly" and call the escalate_emergency tool. Do NOT generate \
any report in that case.
- Keep responses brief and professional.
- You are talking to a law enforcement officer, use appropriate terminology.
"""


# --------------------------------------------------------------------------- #
# Roster
# --------------------------------------------------------------------------- #
def load_roster() -> dict:
    try:
        with open(_ROSTER_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Officer roster not found at {_ROSTER_PATH}")
        return {}


def _normalize_badge(badge: str) -> str:
    return "".join(ch for ch in str(badge) if ch.isdigit())


# --------------------------------------------------------------------------- #
# Transcript
# --------------------------------------------------------------------------- #
def transcript_from_context(context) -> str:
    """Build a readable transcript from the LLM context, dropping the system
    prompt and the synthetic kickoff message."""
    lines = []
    for msg in context.get_messages():
        role = msg.get("role")
        if role == "system":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if not content:
            continue
        if content.strip() == GREETING_KICKOFF.strip():
            continue
        speaker = {"user": "OFFICER", "assistant": "SAFELINE"}.get(role, role.upper())
        lines.append(f"{speaker}: {content.strip()}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# SMS (Twilio REST API over aiohttp — no extra SDK dependency)
# --------------------------------------------------------------------------- #
async def send_review_sms(to_number: str, review_url: str) -> bool:
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_PHONE_NUMBER")

    if not (account_sid and auth_token and from_number and to_number):
        logger.warning(
            "Skipping SMS (missing Twilio creds, from-number, or caller number). "
            f"Review link: {review_url}"
        )
        return False

    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    data = {
        "To": to_number,
        "From": from_number,
        "Body": f"Your Safeline reports are ready for review: {review_url}",
    }
    try:
        auth = aiohttp.BasicAuth(account_sid, auth_token)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data, auth=auth) as resp:
                if resp.status >= 300:
                    logger.error(f"Twilio SMS error ({resp.status}): {await resp.text()}")
                    return False
        logger.info(f"Sent review SMS to {to_number}: {review_url}")
        return True
    except Exception as e:
        logger.error(f"Failed to send review SMS: {e}")
        return False


def review_url_for(incident_id: str) -> str:
    base = os.getenv("APP_BASE_URL", "http://localhost:8080").rstrip("/")
    return f"{base}/review?incident_id={incident_id}"


# --------------------------------------------------------------------------- #
# Tool factory
# --------------------------------------------------------------------------- #
def create_tools(session: dict) -> list:
    """Build the per-call tool functions, closed over a mutable ``session`` dict.

    ``session`` must contain ``context`` (the LLMContext, set by the caller after
    it's created) and may contain ``from_number`` (caller phone for SMS).
    """
    roster = load_roster()

    async def lookup_officer(params: FunctionCallParams, badge_number: str) -> None:
        """Look up the calling officer in the department roster by badge number.

        Call this once, right after the officer gives their badge number, before
        inviting them to narrate the incident.

        Args:
            badge_number: The officer's badge number as digits (e.g. "4521").
        """
        badge = _normalize_badge(badge_number)
        officer = roster.get(badge)
        session["badge"] = badge
        if officer:
            session["officer"] = {**officer, "badge": badge}
            logger.info(f"Officer identified: badge={badge} name={officer['name']}")
            await params.result_callback(
                {
                    "found": True,
                    "name": officer["name"],
                    "unit": officer["unit"],
                    "supervisor": officer["supervisor"],
                }
            )
        else:
            session["officer"] = {"name": None, "badge": badge}
            logger.info(f"Badge {badge} not in roster")
            await params.result_callback(
                {
                    "found": False,
                    "note": (
                        "Badge not found in roster. Politely confirm the badge number "
                        "once; if they confirm, proceed and record it as-is."
                    ),
                }
            )

    async def escalate_emergency(params: FunctionCallParams, reason: str) -> None:
        """Escalate to a human supervisor and END the session WITHOUT generating
        any report. Call this if the officer reports a serious injury, a
        fatality, or an ongoing emergency.

        You must already have spoken the line telling them to contact their
        supervisor directly in this same turn before calling this.

        Args:
            reason: Short description of why this is being escalated.
        """
        session["escalated"] = True
        session["escalation_reason"] = reason
        logger.warning(f"EMERGENCY ESCALATION (no report generated): {reason}")
        await params.llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
        await params.result_callback(
            {"escalated": True}, properties=FunctionCallResultProperties(run_llm=False)
        )

    async def generate_reports(params: FunctionCallParams, report_types: list[str]) -> None:
        """Generate the police reports from the conversation and text the officer
        a review link. Call this only after all legally-critical fields are filled
        and you've told the officer you're generating their reports.

        Args:
            report_types: Which reports to generate. Valid values:
                "incident_report", "arrest_report", "use_of_force",
                "accident_report". Always include "incident_report".
        """
        if session.get("escalated"):
            await params.result_callback(
                {"ok": False, "reason": "Session escalated to supervisor; no report generated."}
            )
            return

        requested = [rt for rt in (report_types or []) if rt in VALID_REPORT_TYPES]
        if "incident_report" not in requested:
            requested.insert(0, "incident_report")

        context = session.get("context")
        transcript = transcript_from_context(context) if context else ""
        badge = session.get("badge", "")
        officer = session.get("officer") or {"badge": badge}

        incident_id = f"INC-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        session["incident_id"] = incident_id
        logger.info(f"Generating reports {requested} for incident {incident_id}")

        try:
            reports = await generate_reports_impl(transcript, requested, badge)
            store_reports(
                incident_id,
                reports,
                transcript=transcript,
                officer=officer,
                officer_badge=badge,
            )
        except Exception as e:
            logger.error(f"Report generation/store failed: {e}")
            await params.result_callback(
                {"ok": False, "reason": "Report generation hit an error; please try again."}
            )
            return

        review_url = review_url_for(incident_id)
        sms_sent = await send_review_sms(session.get("from_number"), review_url)

        await params.result_callback(
            {
                "ok": True,
                "incident_id": incident_id,
                "reports_generated": list(reports.keys()),
                "review_url": review_url,
                "sms_sent": sms_sent,
            }
        )

    async def end_session(params: FunctionCallParams) -> None:
        """End the call. Only call this AFTER you've said a brief closing line in
        the same turn (e.g. "Stay safe out there.")."""
        logger.info("end_session invoked — pushing EndTaskFrame upstream")
        await params.llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
        await params.result_callback(
            {"ok": True}, properties=FunctionCallResultProperties(run_llm=False)
        )

    return [lookup_officer, escalate_emergency, generate_reports, end_session]
