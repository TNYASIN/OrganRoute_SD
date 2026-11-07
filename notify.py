"""
OrganRoute SD - team alerts. No UI here. Sends through every channel configured in .env:

ntfy (free phone push; recommended for the demo)
  NTFY_TOPIC    a hard-to-guess topic name, e.g. organroute-x7k2q9
  NTFY_SERVER   optional, default https://ntfy.sh (use ntfy.sh for reliable iPhone push)
  Install the ntfy app (iOS/Android) and subscribe to the same topic.
  Topics are public to anyone who knows the name: never include patient-identifying details.

Twilio (SMS or WhatsApp; needs an upgraded account for custom messages)
  TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, ALERT_TO (comma-separated +1 numbers)
  TWILIO_FROM   your Twilio number, or whatsapp:+14155238886 for the WhatsApp Sandbox
  Free trials only send preset templates; US SMS also needs carrier (A2P 10DLC) registration.
"""
from __future__ import annotations

import json
import os
import ssl
import urllib.request

TWILIO_REQUIRED = ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM", "ALERT_TO")
ICONS = {"red": "🚨", "yellow": "⚠️", "green": "✅"}
SMS_MAX_CHARS = 320   # ~2-3 SMS segments; each segment is billed
HINTS = {
    21608: "on a trial account, verify this number in the Twilio console first",
    21211: "number format should be +1XXXXXXXXXX",
    63015: "this WhatsApp number hasn't joined the sandbox yet (text 'join <code>' to it)",
    20003: "check TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN",
    0: "free trials only send Twilio's preset templates; upgrade for custom messages",
}
NTFY_PRIORITY = {"red": 5, "yellow": 4, "green": 3}   # 5 = urgent (long vibration, bypasses quiet)
NTFY_TAGS = {"red": "rotating_light", "yellow": "warning", "green": "white_check_mark"}

try:  # python.org macOS builds often lack system CA certs
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL = ssl.create_default_context()


def ntfy_enabled() -> bool:
    return bool(os.environ.get("NTFY_TOPIC", "").strip())


def twilio_enabled() -> bool:
    return all(os.environ.get(k) for k in TWILIO_REQUIRED)


def enabled() -> bool:
    return ntfy_enabled() or twilio_enabled()


def recipients() -> list[str]:
    return [n.strip() for n in os.environ.get("ALERT_TO", "").split(",") if n.strip()]


def twilio_channel() -> str:
    return "WhatsApp" if os.environ.get("TWILIO_FROM", "").startswith("whatsapp:") else "SMS"


def describe() -> str:
    """Short label for the UI, e.g. 'ntfy + SMS to 2'."""
    parts = []
    if ntfy_enabled():
        parts.append("ntfy")
    if twilio_enabled():
        n = len(recipients())
        parts.append(f"{twilio_channel()} to {n}")
    return " + ".join(parts) or "off"


def setup_hint() -> str:
    return "set NTFY_TOPIC in .env (free; install the ntfy app and subscribe to the same topic)"


def _send_ntfy(title: str, message: str, status: str) -> tuple[bool, str]:
    topic = os.environ["NTFY_TOPIC"].strip()
    server = os.environ.get("NTFY_SERVER", "https://ntfy.sh").strip().rstrip("/")
    # JSON publish (to the server root) keeps emoji/accents safe; HTTP headers would need ASCII
    payload = {"topic": topic, "title": title, "message": message or title,
               "priority": NTFY_PRIORITY.get(status, 3), "tags": [NTFY_TAGS.get(status, "bell")]}
    req = urllib.request.Request(server, data=json.dumps(payload).encode("utf-8"),
                                 headers={"User-Agent": "OrganRoute-SD", "Content-Type": "application/json"})
    try:  # never echo the topic: it's effectively the password to read alerts
        with urllib.request.urlopen(req, timeout=8, context=_SSL) as r:
            ok = 200 <= r.status < 300
        note = "Pushed to team phones (ntfy)." if ok else "ntfy push failed."
        if example_topic():
            note += " ⚠️ You're using the example topic from .env.example, which anyone can subscribe to; set your own NTFY_TOPIC."
        return ok, note
    except Exception as e:
        return False, f"ntfy failed: {type(e).__name__} {str(e)[:80]}"


def example_topic() -> bool:
    return "change-me" in os.environ.get("NTFY_TOPIC", "")


def _send_twilio(title: str, message: str, status: str) -> tuple[bool, str]:
    try:
        from twilio.rest import Client
        from twilio.base.exceptions import TwilioRestException
    except ImportError:
        return False, "Twilio: run pip install -r requirements.txt (twilio missing)."
    sender = os.environ["TWILIO_FROM"].strip()
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
            errors.append(f"…{to[-4:]}: {e.msg[:90]}" + (f" ({hint})" if hint else ""))
        except Exception as e:  # network etc.
            errors.append(f"{type(e).__name__}: {str(e)[:80]}")
    note = f"{twilio_channel()}: queued for {sent}/{len(recipients())}."
    if sent and not whatsapp:
        note += " If no text arrives, check Twilio's Messaging logs (30034 = carrier registration needed)."
    if errors:
        note += " Errors: " + "; ".join(errors)
    return sent > 0, note


def send_alert(title: str, message: str, status: str = "yellow") -> tuple[bool, str]:
    """Send through every configured channel. Returns (any_sent, note for the UI). Never raises."""
    if not enabled():
        return False, f"Alerts off: {setup_hint()}."
    results = []
    if ntfy_enabled():
        results.append(_send_ntfy(title, message, status))
    if twilio_enabled():
        results.append(_send_twilio(title, message, status))
    return any(ok for ok, _ in results), " ".join(note for _, note in results)
