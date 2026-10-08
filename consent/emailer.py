"""Email participants their own copy of the study documents.

Sends through SMTP with the account in these env vars (Heroku config vars),
e.g. a study Gmail account with an app password (Google account -> Security ->
2-Step Verification -> App passwords):

    SMTP_USER       sender address, e.g. ucl.mp.study@gmail.com
    SMTP_PASSWORD   the app password (not the account password)
    SMTP_HOST       default smtp.gmail.com
    SMTP_PORT       default 587 (STARTTLS)
    SMTP_REPLY_TO   optional, where participants' replies go (e.g. a UCL address)

Without SMTP_USER and SMTP_PASSWORD, the email option is hidden.

The recipient address is never stored in the oTree database or logged. Note
that Gmail keeps a copy of every message sent via SMTP in the account's Sent
folder, so clear that folder regularly.

The address a participant gives on the information sheet page is also held in
this process's memory (remember_address) for a few hours, so the consent page
can offer to email the signed form to it; it never reaches the database.

Sending runs in a background thread: a live method blocks the whole server
while it runs, including the other laptop's game.

Kept free of oTree imports so it can be tested on its own.
"""
import logging
import os
import re
import smtplib
import ssl
import threading
import time
from email.message import EmailMessage
from pathlib import Path

logger = logging.getLogger(__name__)

_EMAIL = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')

SENDING, SENT, FAILED = 'sending', 'sent', 'failed'

# key -> SENDING / SENT / FAILED; in-process, so it only needs to outlive one
# page view (Heroku runs one web process)
_status = {}
# key -> (address, time given); in-process only, see remember_address
_addresses = {}
ADDRESS_TTL = 3 * 3600
_lock = threading.Lock()


def smtp_settings(env=os.environ):
    """SMTP settings from the environment, or None if email isn't set up."""
    user = env.get('SMTP_USER', '').strip()
    password = env.get('SMTP_PASSWORD', '').strip()
    if not (user and password):
        return None
    return dict(
        host=env.get('SMTP_HOST', '').strip() or 'smtp.gmail.com',
        port=int(env.get('SMTP_PORT', '').strip() or 587),
        user=user,
        password=password,
        reply_to=env.get('SMTP_REPLY_TO', '').strip(),
    )


def clean_address(raw):
    """(address, error): the trimmed address, or an error message for the participant."""
    address = str(raw or '').strip()
    if not _EMAIL.match(address) or len(address) > 200:
        return address, 'Please enter a valid email address.'
    return address, None


def build_message(to, settings, subject, body, attachments):
    """attachments: PDFs, each a path (attached under its file name) or (file name, bytes)."""
    msg = EmailMessage()
    msg['From'] = settings['user']
    msg['To'] = to
    if settings['reply_to']:
        msg['Reply-To'] = settings['reply_to']
    msg['Subject'] = subject
    msg.set_content(body)
    for item in attachments:
        name, data = item if isinstance(item, tuple) else (Path(item).name, Path(item).read_bytes())
        msg.add_attachment(data, maintype='application', subtype='pdf', filename=name)
    return msg


def _send(key, msg, settings):
    try:
        with smtplib.SMTP(settings['host'], settings['port'], timeout=30) as smtp:
            smtp.starttls(context=ssl.create_default_context())
            smtp.login(settings['user'], settings['password'])
            smtp.send_message(msg)
        result = SENT
    except Exception as e:  # report only the type: the message would contain the address
        logger.warning('consent: sending the documents by email failed (%s)', type(e).__name__)
        result = FAILED
    with _lock:
        _status[key] = result


def send_in_background(key, msg, settings, send=_send):
    """Start sending msg; poll status(key) for the result."""
    with _lock:
        _status[key] = SENDING
    threading.Thread(target=send, args=(key, msg, settings), daemon=True).start()


def status(key):
    with _lock:
        return _status.get(key)


def remember_address(key, address, now=None):
    now = time.time() if now is None else now
    with _lock:
        for k in [k for k, (_, t) in _addresses.items() if now - t > ADDRESS_TTL]:
            del _addresses[k]
        _addresses[key] = (address, now)


def recall_address(key, now=None):
    """The address remembered for key, or '' if none (or it has expired)."""
    now = time.time() if now is None else now
    with _lock:
        address, t = _addresses.get(key, ('', 0))
    return address if now - t <= ADDRESS_TTL else ''


def forget_address(key):
    with _lock:
        _addresses.pop(key, None)
