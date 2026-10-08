"""Wording of the consent form (consent_form_Tim.pdf), shared by the oTree page
and signed_form.py's PDF. Kept free of oTree imports so the PDF can also be
made on a laptop from the decrypted export.
"""

STUDY_TITLE = 'Decision-Making in a Dynamic Social Context'
ETHICS_ID = 'CEHP/2024/596'

# Consent form statements, worded as in consent_form_Tim.pdf. Each is a list
# of paragraphs; the lines starting with '- ' render as a list.
STATEMENTS = [
    ['I confirm that I have read and understood the Information Sheet for the above study. I have had '
     'an opportunity to consider the information and what will be expected of me. I have also had the '
     'opportunity to ask questions which have been answered to my satisfaction.'],
    ['I understand that my participation is voluntary and that I am free to withdraw at any time without '
     'giving a reason, without the care I receive or my legal rights being affected.'],
    ['I understand that I will be able to withdraw my anonymous data at any point in time up until the '
     'publication of this data, and withdraw my personal information at any point without requiring a '
     'reason.',
     'I understand that if I decide to withdraw:',
     '- Any personal data I have provided up to that point will be deleted unless I agree otherwise.',
     '- Any published or pre-print (anonymous) data will remain available and open-access.'],
    ['I consent to participate in the study. I understand that my personal information (name, email '
     'address, gender, and date of birth) will be used for the purposes explained to me. I understand '
     'that according to data protection legislation, public task will be the lawful basis for processing.'],
    ['Use of the information',
     'I understand that all personal information will remain confidential and that all efforts will be '
     'made to ensure I cannot be identified (personal information that can be used to identify my data '
     'will be protected and only accessible to researchers on the study, and stored for up to 10 years '
     'after the completion of the project).'],
    ['I understand that the data gathered in this study will be stored pseudonymously and securely. It '
     'will be assigned a coded designation that will deprive the collected data of any connection to my '
     'identity. It will not be possible to identify me in any publications or scientific communication, '
     'where this data will be presented anonymously.'],
    ['I understand the potential risks of participating and the support that will be available to me '
     'should I become distressed during the course of the research.'],
    ['I understand that the data will not be made available to any commercial organisations but is '
     'solely the responsibility of the researcher(s) undertaking this study.'],
    ['I understand that I will not benefit financially from this study or from any possible outcome it '
     'may result in in the future.'],
    ['I understand that I will be compensated for the portion of time spent in the study, with additional '
     'compensation based on my performance, and will still be fully compensated if I later choose to '
     'withdraw.'],
    ['I agree that my pseudonymised research data may be used by others for future research. Only '
     'researchers undertaking the current study will be able to identify me from this data, and no one '
     'will be able to identify me from any published data. *Note, not agreeing to this will not preclude '
     'you from taking part in this study'],
    ['I hereby confirm that I understand the inclusion criteria as detailed in the Information Sheet and '
     'explained to me by the researcher.'],
    ['I hereby confirm that:',
     '- (a) I understand the exclusion criteria as detailed in the Information Sheet and explained to me '
     'by the researcher; and',
     '- (b) I do not fall under the exclusion criteria.'],
    ['I am aware of who I should contact if I wish to lodge a complaint.'],
    ['I voluntarily agree to take part in this study.'],
    ['Use of information for this project and beyond:',
     'I agree that my personal information (name, date of birth, gender, and email) will be stored '
     'securely in on UCL’s Data Safe Haven, for up to 10 years after the completion of this study, and '
     'that only study researchers will be able to associate my person information with my data. I am '
     'aware that no personal information will be included in any publication resulting from this study '
     'or otherwise.',
     'I would be happy for the data I provide to be securely archived at UCL until project completion.',
     'I understand that other authenticated researchers working on this study at UCL will have access '
     'to my pseudonymised data.'],
]
OPTIONAL_STATEMENTS = [11]  # data sharing: "not agreeing to this will not preclude you"
FUTURE_CONTACT = [
    'Yes, I would be happy to be contacted in this way',
    'No, I would not like to be contacted',
]

STATEMENT_FIELDS = [f'consent_{i:02d}' for i in range(1, len(STATEMENTS) + 1)]
