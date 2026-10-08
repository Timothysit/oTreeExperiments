"""The signed consent form as a PDF: emailed to the participant when they sign,
and written to the lab's records from the decrypted consent export:

    # admin -> Data -> consent (custom export), then
    uv run python -m consent.signed_form archive consent_export.csv \\
        --key ~/.mp_payments/private.key \\
        --out-dir /mnt/ogma/delab/lab-members/Tim/human_mp/consent_forms

One PDF per participant (their latest signature), named
<date>_<participant code>_consent.pdf; existing files are left alone, so
re-running after each testing day only adds the new ones. Test runs are skipped.

The PDFs hold names: keep them out of git and out of anything shared beyond the
research team.
"""
import argparse
import csv
import logging
import sys
import time
from pathlib import Path

from fpdf import FPDF

from mp_survey.payment_crypto import decrypt_details, latest_per_participant

from . import form_text

FONTS = Path(__file__).resolve().parent / 'fonts'  # DejaVu: covers accented and non-Latin European names
logging.getLogger('fontTools').setLevel(logging.WARNING)  # font subsetting logs every table it prunes

HEADER = [
    ('Title of Study', form_text.STUDY_TITLE),
    ('Department', 'Psychology and Language Sciences (PaLS)'),
    ('Researcher(s)', 'Timothy Sit, tim.sit.18@ucl.ac.uk; Julia Nicklaus, julia.nicklaus.25@ucl.ac.uk; '
                      'Ann Duan, c.duan@ucl.ac.uk'),
    ('Principal Researcher', 'Essi Viding, e.viding@ucl.ac.uk'),
    ('UCL Data Protection Officer', 'Alexandra Potts, data-protection@ucl.ac.uk'),
    ('UCL Research Ethics Committee Project ID', form_text.ETHICS_ID),
]
PREAMBLE = (
    'I confirm that I understand that by ticking each box below I am consenting to this element of the study. '
    'I understand that it will be assumed that unticked boxes means that I DO NOT consent to that part of the '
    'study. I understand that by not giving consent for any one element that I may be deemed ineligible for '
    'the study.'
)


def signed_date(record):
    """'8 October 2026' from the record's signed_at ('2026-10-08 16:38:00 UTC')."""
    t = time.strptime(record['signed_at'][:10], '%Y-%m-%d')
    return f"{t.tm_mday} {time.strftime('%B %Y', t)}"


def text_block(pdf, w, h, text, align='L'):
    """Wrapped text from the current x; leaves the cursor at the left margin below it."""
    pdf.multi_cell(w, h, text, align=align, new_x='LMARGIN', new_y='NEXT')


def render_signed_form(record, participant_code=''):
    """PDF bytes of a completed consent form. record: the dict consent.save_consent encrypts."""
    pdf = FPDF(format='A4')
    pdf.set_margins(18, 16, 18)
    pdf.set_auto_page_break(True, margin=16)
    pdf.add_font('DejaVu', '', FONTS / 'DejaVuSans.ttf')
    pdf.add_font('DejaVu', 'B', FONTS / 'DejaVuSans-Bold.ttf')
    pdf.add_page()
    width = pdf.epw

    pdf.set_font('DejaVu', 'B', 12)
    text_block(pdf, width, 7, 'CONSENT FORM FOR ADULT PARTICIPANTS IN RESEARCH STUDIES', align='C')
    pdf.ln(2)
    for label, value in HEADER:
        pdf.set_font('DejaVu', 'B', 8.5)
        text_block(pdf, width, 4.5, label + ':')
        pdf.set_font('DejaVu', '', 8.5)
        text_block(pdf, width, 4.5, value)
    pdf.ln(3)
    pdf.set_font('DejaVu', 'B', 9)
    text_block(pdf, width, 4.8, PREAMBLE)
    pdf.ln(2)

    box = 5
    text_w = width - box - 12
    for i, (field, paragraphs) in enumerate(zip(form_text.STATEMENT_FIELDS, form_text.STATEMENTS), start=1):
        # keep a statement and its box on one page
        if pdf.get_y() > pdf.h - 45:
            pdf.add_page()
        top = pdf.get_y()
        pdf.set_font('DejaVu', '', 8.5)
        for j, para in enumerate(paragraphs):
            prefix = f'{i}.  ' if j == 0 else ''
            if para.startswith('- '):
                pdf.set_x(pdf.l_margin + 10)
                text_block(pdf, text_w - 10, 4.5, '•  ' + para[2:])
            else:
                pdf.set_x(pdf.l_margin + (0 if j == 0 else 6))
                text_block(pdf, text_w - (0 if j == 0 else 6), 4.5, prefix + para)
        if i in form_text.OPTIONAL_STATEMENTS:
            pdf.set_x(pdf.l_margin + 6)
            pdf.set_font('DejaVu', '', 7.5)
            text_block(pdf, text_w - 6, 4.2, '(optional)')
        bottom = pdf.get_y()
        x = pdf.l_margin + width - box
        pdf.rect(x, top + 0.5, box, box)
        if record.get(field):
            pdf.set_xy(x, top + 0.3)
            pdf.set_font('DejaVu', 'B', 10)
            pdf.cell(box, box, '✓', align='C')
        pdf.set_y(max(bottom, top + box + 1) + 2)

    if pdf.get_y() > pdf.h - 70:
        pdf.add_page()
    pdf.ln(2)
    pdf.set_font('DejaVu', '', 8.5)
    text_block(pdf, width, 4.5, (
        'If you would like your contact details to be retained so that you can be contacted in the future by '
        'UCL researchers who would like to invite you to participate in follow up studies to this project, or '
        'in future studies of a similar nature, please tick the appropriate box below.'
    ))
    pdf.ln(1)
    for choice in form_text.FUTURE_CONTACT:
        top = pdf.get_y()
        pdf.rect(pdf.l_margin + 4, top + 0.5, 4, 4)
        if record.get('future_contact') == choice:
            pdf.set_xy(pdf.l_margin + 4, top + 0.2)
            pdf.set_font('DejaVu', 'B', 9)
            pdf.cell(4, 4, '✓', align='C')
            pdf.set_font('DejaVu', '', 8.5)
        pdf.set_xy(pdf.l_margin + 11, top)
        pdf.cell(width - 11, 5, choice, new_x='LMARGIN', new_y='NEXT')

    date = signed_date(record)
    pdf.ln(6)
    for role, name, note in [
        ('Name of participant', record.get('full_name', ''), 'Signed by typing their name on screen'),
        ('Researcher', record.get('researcher', ''), ''),
    ]:
        pdf.set_font('DejaVu', 'B', 11)
        pdf.cell(width * 0.6, 7, name)
        pdf.cell(width * 0.4, 7, date, new_x='LMARGIN', new_y='NEXT')
        pdf.set_font('DejaVu', '', 7.5)
        pdf.cell(width * 0.6, 4, role + (f' ({note})' if note else ''))
        pdf.cell(width * 0.4, 4, 'Date', new_x='LMARGIN', new_y='NEXT')
        pdf.ln(4)

    pdf.set_font('DejaVu', '', 7)
    pdf.set_text_color(110)
    footer = f"Signed {record.get('signed_at', '')}"
    if participant_code:
        footer += f' · participant {participant_code}'
    text_block(pdf, width, 4, footer)
    return bytes(pdf.output())


def archive(export_csv, private_key_b64, out_dir, include_test_runs=False):
    """Write one signed-form PDF per participant; returns (written, skipped_existing)."""
    with open(export_csv, newline='') as f:
        rows = latest_per_participant(list(csv.DictReader(f)))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written, existing = [], []
    for row in sorted(rows, key=lambda r: float(r['submitted_at'])):
        if row.get('test_run') == '1' and not include_test_runs:
            continue
        record = decrypt_details(row['ciphertext'], private_key_b64)
        path = out_dir / f"{record['signed_at'][:10]}_{row['participant_code']}_consent.pdf"
        if path.exists():
            existing.append(path)
            continue
        path.write_bytes(render_signed_form(record, row['participant_code']))
        written.append(path)
    return written, existing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser('archive', help='decrypt the consent export into one signed-form PDF per participant')
    p.add_argument('export_csv')
    p.add_argument('--key', required=True, help='private key file (mp_survey/payment_crypto.py keygen)')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--include-test-runs', action='store_true')
    args = parser.parse_args(argv)
    with open(args.key) as f:
        private_b64 = f.read().strip()
    written, existing = archive(args.export_csv, private_b64, args.out_dir, args.include_test_runs)
    print(f'{len(written)} written, {len(existing)} already there, in {args.out_dir}')


if __name__ == '__main__':
    main(sys.argv[1:])
