"""Parse user-study conversation transcripts into replayable turn sequences.

A transcript line has the form ``HH:MM - <actor>: <message>``. The three
inputs the graph asks for at each cycle (datetime, patient request,
situation perception) are exactly the fields recorded by
``utils.logs.conversation_message``, so a transcript can be replayed
verbatim through the full LangGraph pipeline.

Stdlib only: importable without the project's runtime dependencies.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

# Transcripts are written by open() with the default Windows codepage.
TRANSCRIPT_ENCODING = "cp1252"

_LINE = re.compile(r"^(?P<time>\d{2}:\d{2}) - (?P<actor>[^:]+): (?P<msg>.*)$")
_HEADER = re.compile(r"^(?P<time>\d{2}:\d{2}) - Patient ID: (?P<pid>\S+)\s*$")
_DOCTOR = re.compile(r"^Doctor specifications:\s*(?P<specs>.*)$")

# The user-study sessions label perception lines "Perception"; the
# development sessions in llm4hrc/conversation label them
# "Robot (perception)". Both mark the same field.
_PERCEPTION_ACTORS = frozenset({"Perception", "Robot (perception)"})

# Participant folders are named "<n> - <real name>". Only the number is
# carried into parsed output, so no personal name reaches the results.
_PARTICIPANT_DIR = re.compile(r"^(?P<num>\d+)\s*-")


@dataclass
class Turn:
    """One perception cycle: what the replay feeds to robot_perception_node."""

    time: str
    patient_request: str | None
    situation_perception: str


@dataclass
class Transcript:
    path: Path
    patient_id: str
    doctor_specs: str
    start_time: str
    turns: list[Turn] = field(default_factory=list)
    participant: str = ""
    content_hash: str = ""

    @property
    def name(self) -> str:
        """Identifier safe to publish: participant number, never a name."""
        session = self.path.name.replace("_conversation.txt", "")
        return f"{self.participant}/{session}" if self.participant else session


def parse_transcript(path: str | Path) -> Transcript:
    """Parse one transcript file into a Transcript."""
    path = Path(path)
    raw = path.read_text(encoding=TRANSCRIPT_ENCODING, errors="replace")

    patient_id = ""
    doctor_specs = ""
    start_time = ""
    turns: list[Turn] = []
    pending_request: str | None = None
    pending_time: str | None = None

    for line in raw.splitlines():
        line = line.rstrip()
        if not line:
            continue

        header = _HEADER.match(line)
        if header:
            patient_id = header.group("pid")
            start_time = header.group("time")
            continue

        doctor = _DOCTOR.match(line)
        if doctor:
            doctor_specs = doctor.group("specs").strip()
            continue

        entry = _LINE.match(line)
        if not entry:
            continue

        actor = entry.group("actor").strip()
        msg = entry.group("msg").strip()

        if actor == "Patient":
            # "None" is how an empty request was serialised.
            pending_request = None if msg == "None" else msg
            pending_time = entry.group("time")
        elif actor in _PERCEPTION_ACTORS:
            # A perception always closes the pair opened by a Patient line.
            turns.append(
                Turn(
                    time=pending_time or entry.group("time"),
                    patient_request=pending_request,
                    situation_perception=msg or "Patient is speaking",
                )
            )
            pending_request = None
            pending_time = None

    participant = ""
    folder = _PARTICIPANT_DIR.match(path.parent.name)
    if folder:
        participant = f"P{int(folder.group('num')):02d}"

    return Transcript(
        path=path,
        patient_id=patient_id,
        doctor_specs=doctor_specs,
        start_time=start_time or "08:00",
        turns=turns,
        participant=participant,
        content_hash=hashlib.md5(raw.encode("utf-8", "replace")).hexdigest(),
    )


def load_corpus(
    directory: str | Path = "conversation",
    min_turns: int = 1,
    recursive: bool = False,
    deduplicate: bool = True,
) -> list[Transcript]:
    """Parse the transcripts under *directory*, keeping those with enough turns.

    The user-study export stores one folder per participant, and a few
    sessions were copied into two folders; *deduplicate* keeps only the
    first occurrence of each byte-identical transcript so a session is not
    replayed (and counted) twice. Results are sorted by participant number,
    then by turn count.
    """
    directory = Path(directory)
    pattern = "**/*_conversation.txt" if recursive else "*_conversation.txt"
    transcripts = [parse_transcript(p) for p in sorted(directory.glob(pattern))]

    kept: list[Transcript] = []
    seen: dict[str, Transcript] = {}
    for transcript in transcripts:
        if len(transcript.turns) < min_turns:
            continue
        if deduplicate and transcript.content_hash in seen:
            continue
        seen[transcript.content_hash] = transcript
        kept.append(transcript)

    return sorted(kept, key=lambda t: (t.participant, -len(t.turns)))


def find_duplicates(transcripts: list[Transcript]) -> dict[str, list[str]]:
    """Group transcript names by content hash, returning only the collisions."""
    groups: dict[str, list[str]] = {}
    for transcript in transcripts:
        groups.setdefault(transcript.content_hash, []).append(transcript.name)
    return {h: names for h, names in groups.items() if len(names) > 1}
