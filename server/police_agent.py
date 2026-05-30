"""Safeline police-documentation agent: shared logic for both bot variants.

This module holds everything that makes the bot a *police documentation
assistant* rather than the flower-shop starter, so ``bot-gpt.py`` and
``bot-nemotron.py`` stay nearly identical to the original and only swap in
this business logic:

  * the two-phase system prompt (open narrative -> gap filling),
  * officer roster lookup by badge number,
  * the LLM tools (lookup_officer, escalate_emergency, generate_reports,
    end_session),
  * transcript capture, report generation, persistence, and handing the officer
    a 6-digit access code (the bot POSTs reports to the review portal over HTTP
    and speaks the code aloud — no SMS).

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

from report_generator import create_pending_incident
from report_generator import generate_reports as generate_reports_impl
from report_generator import mark_generation_failed
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
generating your reports now." In that SAME turn, call the generate_reports \
tool with the list of report types you decided on. The tool returns the \
officer's access code in the "spoken_code" field. The code is ALWAYS EXACTLY \
SIX DIGITS.

ACCESS CODE HANDOFF (follow exactly):
- Read the code using the EXACT words given in the tool's "spoken_code" field, \
ONE digit at a time, slowly (for example: "four... eight... one... nine... \
two... zero").
- The code is EXACTLY SIX digits. Never add, drop, change, or invent digits. \
Never read it as one big number (do NOT say "four hundred eighty-one \
thousand..."). Say six separate digits only.
- Then tell the officer to enter it at the Safeline portal and ask: "Did you \
get the code?"
- WAIT for the officer's reply. Do NOT end the call yet.
- ONLY when the officer explicitly confirms they have the code (for example \
"I got the code", "got it", "yes I have it"), say a brief closing line \
(e.g. "Stay safe out there.") and call the confirm_code_received tool.
- If the officer has not confirmed, did not hear it, or asks you to repeat, \
read the SAME six digits again (digit by digit) and ask "Did you get the \
code?" again. Keep repeating until they confirm. NEVER hang up or call \
confirm_code_received before the officer confirms they received the code.

RULES:
- Ask only ONE question at a time.
- Never ask about something already mentioned in the narrative.
- Never make up information - only use what the officer told you.
- The access code is ALWAYS exactly six digits. Read only the digits provided \
by the generate_reports tool, exactly six of them, one at a time.
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
# Review portal hand-off (6-digit access code)
# --------------------------------------------------------------------------- #
_DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


def _digits_to_words(code: str) -> str:
    """Spell a code as separate digit words so the LLM reads it verbatim and
    never collapses it into one large number (e.g. '481920' -> 'four, eight,
    one, nine, two, zero')."""
    return ", ".join(_DIGIT_WORDS.get(ch, ch) for ch in code)


def _strip_report_metadata(reports: dict) -> dict:
    """Drop the internal ``_``-prefixed metadata so the portal only shows real
    report fields."""
    return {
        report_type: {k: v for k, v in fields.items() if not k.startswith("_")}
        for report_type, fields in reports.items()
    }


async def submit_reports_to_review_ui(
    code: str,
    incident_id: str,
    officer_badge: str,
    officer_name: str,
    reports: dict,
    transcript: str,
) -> bool:
    """POST generated reports to the review portal, keyed by ``code``.

    The bot runs remotely on Pipecat Cloud and the portal runs elsewhere
    (exposed via ngrok), so they hand off over HTTP rather than shared storage.
    """
    review_ui_url = os.getenv("REVIEW_UI_URL", "http://localhost:8080").rstrip("/")
    payload = {
        "code": code,
        "incident_id": incident_id,
        "officer_badge": officer_badge,
        "officer_name": officer_name,
        "reports": reports,
        "transcript": transcript,
        "generated_at": datetime.datetime.now().isoformat(),
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{review_ui_url}/api/reports/save", json=payload
            ) as resp:
                if resp.status != 200:
                    logger.error(
                        f"Review UI save failed ({resp.status}): {await resp.text()}"
                    )
                    return False
        logger.info(f"Posted reports to review portal {review_ui_url} under code {code}")
        return True
    except Exception as e:
        logger.error(f"Failed to post reports to review portal: {e}")
        return False


def code_repeat_instruction(code: str) -> str:
    """System-message text that tells the agent to re-read the SAME six digits
    and re-ask for confirmation. Used by the bot's idle watchdog so the agent
    repeats the code every ~15s of officer silence until they confirm."""
    spaced = _digits_to_words(code)
    return (
        "The officer has NOT yet confirmed they received the access code. "
        "Say: 'Did you get the code? Let me repeat it.' Then read these six "
        f"digits one at a time, slowly: {spaced}. Then ask: 'Did you get the "
        "code?' Read ONLY these six digits, exactly six, and do not add any "
        "digits. Do not call any tool until the officer confirms they received "
        "the code."
    )


# --------------------------------------------------------------------------- #
# Tool factory
# --------------------------------------------------------------------------- #
def create_tools(session: dict) -> list:
    """Build the per-call tool functions, closed over a mutable ``session`` dict.

    ``session`` must contain ``context`` (the LLMContext, set by the caller
    after it's created). The generated 6-digit access code is stored back on
    ``session["access_code"]``.
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
        """Generate the police reports from the conversation, hand them to the
        review portal, and return a six-digit access code the officer will use
        to review them. Call this only after all legally-critical fields are
        filled and you've told the officer you're generating their reports.

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
        officer_name = officer.get("name") or f"Officer (badge {badge or 'unknown'})"

        incident_id = f"INC-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        session["incident_id"] = incident_id
        pending = create_pending_incident(
            incident_id,
            transcript=transcript,
            officer=officer,
            officer_badge=badge,
            report_types=requested,
        )
        access_code = pending["access_code"]
        logger.info(
            f"Generating reports {requested} for incident {incident_id} "
            f"access_code={access_code}"
        )

        try:
            reports = await generate_reports_impl(transcript, requested, badge)
            # Keep the permanent on-disk audit trail (immutable AI draft) as well.
            store_reports(
                incident_id,
                reports,
                transcript=transcript,
                officer=officer,
                officer_badge=badge,
                access_code=access_code,
            )
        except Exception as e:
            logger.error(f"Report generation/store failed: {e}")
            mark_generation_failed(incident_id, str(e))
            await params.result_callback(
                {"ok": False, "reason": "Report generation hit an error; please try again."}
            )
            return

        clean_reports = _strip_report_metadata(reports)
        code = access_code
        session["access_code"] = code
        # Until the officer confirms receipt, the idle watchdog will re-read the
        # code every ~15s of silence (see the bot pipeline's IdleFrameProcessor).
        session["awaiting_code_confirmation"] = True
        posted = await submit_reports_to_review_ui(
            code=code,
            incident_id=incident_id,
            officer_badge=badge,
            officer_name=officer_name,
            reports=clean_reports,
            transcript=transcript,
        )

        await params.result_callback(
            {
                "ok": True,
                "incident_id": incident_id,
                "reports_generated": list(reports.keys()),
                "access_code": code,
                "spoken_code": _digits_to_words(code),
                "digit_count": 6,
                "posted_to_portal": posted,
                "instructions": (
                    "Read EXACTLY these six digits to the officer, one at a time, "
                    "using the words in spoken_code. Do not add or change any "
                    "digits. Then ask 'Did you get the code?' and do NOT end the "
                    "call until the officer confirms they received it; only then "
                    "call confirm_code_received."
                ),
            }
        )

    async def confirm_code_received(params: FunctionCallParams) -> None:
        """End the call AFTER the officer has explicitly confirmed they received
        the six-digit access code (e.g. they said "I got the code", "got it", or
        "yes I have it").

        Call this ONLY once the officer confirms. You must say a brief closing
        line (e.g. "Stay safe out there.") in the SAME turn before calling it.
        Do NOT call this if the officer has not yet confirmed they have the code.
        """
        session["awaiting_code_confirmation"] = False
        logger.info("Officer confirmed code receipt — ending session")
        await params.llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
        await params.result_callback(
            {"ok": True}, properties=FunctionCallResultProperties(run_llm=False)
        )

    async def end_session(params: FunctionCallParams) -> None:
        """End the call. Do NOT use this to end after generating reports — use
        confirm_code_received once the officer confirms they got the code. Only
        use this for unrelated endings, and only AFTER saying a brief closing
        line in the same turn."""
        session["awaiting_code_confirmation"] = False
        logger.info("end_session invoked — pushing EndTaskFrame upstream")
        await params.llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
        await params.result_callback(
            {"ok": True}, properties=FunctionCallResultProperties(run_llm=False)
        )

    return [
        lookup_officer,
        escalate_emergency,
        generate_reports,
        confirm_code_received,
        end_session,
    ]
