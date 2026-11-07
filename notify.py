"""
OrganRoute SD - team alerts via Twilio (SMS or WhatsApp). No UI here.

Set in .env:
  TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN   from console.twilio.com
  TWILIO_FROM   your Twilio number (+1555...) for SMS,
                or whatsapp:+14155238886 for the WhatsApp Sandbox
  ALERT_TO      comma-separated phone numbers in +1... format

Trial accounts only reach verified numbers. US SMS also needs carrier (A2P 10DLC) registration;
unregistered texts are accepted by Twilio but dropped by the carrier (error 30034 in the Twilio logs).
The WhatsApp Sandbox avoids that: each recipient texts "join <code>" to the sandbox number once.
Never include patient-identifying details in alerts.
"""
from __future__ import annotations

import os

REQUIRED = ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM", "ALERT_TO")
ICONS = {"red": "🚨", "yellow": "⚠️", "green": "✅"}
SMS_MAX_CHARS = 320   # ~2-3 SMS segments; each segment is billed
HINTS = {
    21608: "on a trial account, verify this number in the Twilio console first",
    21211: "number format should be +1XXXXXXXXXX",
    63015: "this WhatsApp number hasn't joined the sandbox yet (text 'join <code>' to it)",
    20003: "check TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN",
}


def enabled() -> bool:
    return all(os.environ.get(k) for k in REQUIRED)


def channel() -> str:
    return "WhatsApp" if os.environ.get("TWILIO_FROM", "").startswith("whatsapp:") else "SMS"


def recipients() -> list[str]:
    return [n.strip() for n in os.environ.get("ALERT_TO", "").split(",") if n.strip()]


def send_alert(title: str, message: str, status: str = "yellow") -> tuple[bool, str]:
    """Send to everyone in ALERT_TO. Returns (any_sent, note for the UI). Never raises."""
    missing = [k for k in REQUIRED if not os.environ.get(k)]
    if missing:
        return False, f"Alerts off: set {', '.join(missing)} in .env."
    try:
        from twilio.rest import Client
        from twilio.base.exceptions import TwilioRestException
    except ImportError:
        return False, "Alerts off: run pip install -r requirements.txt (twilio missing)."

    sender = os.environ["TWILIO_FROM"]
    whatsapp = sender.startswith("whatsapp:")
    body = f"{ICONS.get(status, '')} {title}\n{message}".strip()
    if not whatsapp and len(body) > SMS_MAX_CHARS:
        body = body[:SMS_MAX_CHARS - 1] + "…"

    client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
    sent, errors = 0, []
    for to in recipients():
        addr = f"whatsapp:{to}" if whatsapp and not to.startswith("whatsapp:") else to
        try:
            client.messages.create(to=addr, from_=sender, body=body)
            sent += 1
        except TwilioRestException as e:
            hint = HINTS.get(e.code, "")
            errors.append(f"{to[-4:].rjust(len(to), '•')}: {e.msg}" + (f" ({hint})" if hint else ""))
        except Exception as e:  # network etc.
            errors.append(f"{type(e).__name__}: {str(e)[:80]}")
    note = f"{channel()}: queued for {sent}/{len(recipients())} recipient(s)."
    if sent and not whatsapp:
        note += " If no text arrives, check Twilio Console → Monitor → Messaging logs (30034 = carrier registration needed)."
    if errors:
        note += " Errors: " + "; ".join(errors)
    return sent > 0, note
