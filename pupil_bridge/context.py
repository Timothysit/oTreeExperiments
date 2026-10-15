"""What an oTree page passes to _static/pupil_bridge.js, via js_vars."""


def pupil_bridge_enabled(player):
    """Whether this participant's laptop uses the pupil bridge.

    Only laptops listed in the session config's `pupil_bridge_labels`
    (comma-separated room labels) do, so a laptop without a headset never tries
    to reach a local address. Participants without a label (e.g. demo links)
    always do.
    """
    participant = player.participant
    labels = player.session.config.get("pupil_bridge_labels", "")
    allowed = {s.strip() for s in labels.split(",") if s.strip()}
    return not participant.label or participant.label in allowed


def pupil_js_vars(player, **extra):
    """Pass as js_vars `pupil` and call PupilBridge.init(js_vars.pupil) in the page.

    The returned fields are added to every annotation the page sends, so each
    Pupil recording can be matched to the oTree participant and session.
    """
    participant = player.participant
    return dict(
        enabled=pupil_bridge_enabled(player),
        participant_code=participant.code,
        participant_label=participant.label or "",
        session_code=player.session.code,
        player_id=player.id_in_group,
        **extra,
    )
