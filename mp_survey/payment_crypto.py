"""Payment details: validation, public-key encryption, and the payer's CLI.

The oTree server only ever holds the PUBLIC key (env var PAYMENT_PUBLIC_KEY), so
it can encrypt but never decrypt: the database, its backups and every export
contain only ciphertext. The PRIVATE key stays with whoever makes the payments,
never on Heroku and never in git. Destroying it at the end of the study makes
every stored row unreadable (no need to hunt down old DB copies).

Kept free of oTree imports so the CLI runs anywhere:

    # once: make a keypair; put the printed public key in Heroku's config vars
    uv run python mp_survey/payment_crypto.py keygen ~/.mp_payments/private.key

    # to pay: download mp_survey's custom export (admin -> Data), then
    uv run python mp_survey/payment_crypto.py decrypt export.csv \\
        --key ~/.mp_payments/private.key --out payments.csv

payments.csv holds bank details in plain text: keep it out of the repo and
delete it once payments are done.
"""
import argparse
import base64
import csv
import json
import os
import re
import sys

from nacl.public import PrivateKey, PublicKey, SealedBox

FIELDS = ['full_name', 'email', 'sort_code', 'account_number']

_EMAIL = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')


def clean_details(raw):
    """Normalise and validate the submitted form.

    Returns (details, errors): details has every field in FIELDS (sort code and
    account number reduced to digits); errors maps field -> message, empty if valid.
    """
    raw = raw if isinstance(raw, dict) else {}
    details = {f: str(raw.get(f) or '').strip() for f in FIELDS}
    details['sort_code'] = re.sub(r'[\s-]', '', details['sort_code'])
    details['account_number'] = re.sub(r'\s', '', details['account_number'])

    errors = {}
    if not details['full_name']:
        errors['full_name'] = 'Please enter your name.'
    elif len(details['full_name']) > 100:
        errors['full_name'] = 'Name is too long.'
    if not _EMAIL.match(details['email']) or len(details['email']) > 200:
        errors['email'] = 'Please enter a valid email address.'
    if not re.fullmatch(r'\d{6}', details['sort_code']):
        errors['sort_code'] = 'Sort code should be 6 digits, e.g. 12-34-56.'
    if not re.fullmatch(r'\d{8}', details['account_number']):
        errors['account_number'] = 'Account number should be 8 digits.'
    return details, errors


def encrypt_details(details, public_key_b64):
    """Encrypt a details dict to base64 ciphertext readable only with the private key."""
    box = SealedBox(PublicKey(base64.b64decode(public_key_b64)))
    return base64.b64encode(box.encrypt(json.dumps(details).encode())).decode()


def decrypt_details(ciphertext_b64, private_key_b64):
    box = SealedBox(PrivateKey(base64.b64decode(private_key_b64)))
    return json.loads(box.decrypt(base64.b64decode(ciphertext_b64)))


def generate_keypair():
    """(private_b64, public_b64)"""
    sk = PrivateKey.generate()
    return base64.b64encode(bytes(sk)).decode(), base64.b64encode(bytes(sk.public_key)).decode()


def latest_per_participant(rows):
    """Keep each participant's most recent submission (they may resubmit to correct a typo)."""
    latest = {}
    for row in rows:
        code = row['participant_code']
        if code not in latest or float(row['submitted_at']) > float(latest[code]['submitted_at']):
            latest[code] = row
    return list(latest.values())


def _keygen(args):
    if os.path.exists(args.private_key_path):
        sys.exit(f'{args.private_key_path} already exists; refusing to overwrite a key that may '
                 'be needed to read existing submissions.')
    private_b64, public_b64 = generate_keypair()
    os.makedirs(os.path.dirname(os.path.abspath(args.private_key_path)), exist_ok=True)
    fd = os.open(args.private_key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write(private_b64 + '\n')
    print(f'Private key written to {args.private_key_path} (keep it safe; never commit or upload it).')
    print('Public key, for Heroku: heroku config:set PAYMENT_PUBLIC_KEY=' + public_b64)


def _decrypt(args):
    with open(args.key) as f:
        private_b64 = f.read().strip()
    with open(args.export_csv, newline='') as f:
        rows = latest_per_participant(list(csv.DictReader(f)))
    # test_run = 1: a lab-notes test session (questionnaire optional), not a participant to pay
    meta = ['session_code', 'participant_code', 'participant_label', 'test_run', 'submitted_at']
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=meta + FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: float(r['submitted_at'])):
            writer.writerow({**{k: row.get(k, '') for k in meta}, **decrypt_details(row['ciphertext'], private_b64)})
    print(f'{len(rows)} participants written to {args.out}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser('keygen', help='make a keypair; writes the private key, prints the public one')
    p.add_argument('private_key_path')
    p.set_defaults(func=_keygen)
    p = sub.add_parser('decrypt', help="decrypt mp_survey's custom export into a payments CSV")
    p.add_argument('export_csv')
    p.add_argument('--key', required=True, help='private key file from keygen')
    p.add_argument('--out', required=True)
    p.set_defaults(func=_decrypt)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
