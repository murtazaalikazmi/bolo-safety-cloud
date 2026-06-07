"""
Bolo Safety Cloud — Processing Pipeline
Reads audio from Dropbox → Whisper → GPT → Supabase DB
Called from app.py when an admin presses the Process button.
"""

import os
import re
import json
import tempfile
import traceback
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# LIMITS
# ─────────────────────────────────────────────────────────────
MAX_DURATION_SECONDS = 300    # 5 minutes
MAX_FILE_SIZE_MB     = 20
AUDIO_EXTENSIONS     = {".wav", ".m4a", ".mp3", ".ogg", ".flac", ".aac", ".wma", ".mp4"}

DOMAIN_TERMS = {
    "palazan": "paraxylene", "para xylene": "paraxylene",
    "edc": "EDC", "vcm": "VCM", "lpg": "LPG", "lng": "LNG",
    "jetty": "Jetty", "mcc": "MCC", "loading bay": "Loading Bay",
    "loading area": "Loading Area", "tank farm": "Tank Farm",
    "tan farm": "Tank Farm", "ppe": "PPE", "loto": "LOTO",
    "ptw": "PTW", "hot work": "Hot Work", "confined space": "Confined Space",
    "dcs": "DCS", "scada": "SCADA",
}

CATEGORY_PROMPT = """You are an expert HSE officer at a chemical/petrochemical plant.
Classify the safety observation into EXACTLY ONE category:
  • Unsafe Act       — a person's action that increases accident risk (not wearing PPE, bypassing safety interlocks)
  • Unsafe Condition — a physical/environmental hazard (oil spill, damaged equipment, blocked exit)
  • Near Miss        — event that had potential to cause harm but did NOT (tool dropped near worker)
  • Incident         — event that DID result in injury, damage or environmental release

Observation: {text}

Respond with valid JSON only:
{{"category": "...", "severity": "High/Medium/Low", "location": "...", "reasoning": "one sentence"}}"""


# ─────────────────────────────────────────────────────────────
# DROPBOX HELPERS
# ─────────────────────────────────────────────────────────────
def get_dropbox_client(app_key, app_secret, refresh_token):
    import dropbox
    # Uses refresh token — auto-renews access, never expires
    return dropbox.Dropbox(
        app_key=app_key,
        app_secret=app_secret,
        oauth2_refresh_token=refresh_token
    )


def list_dropbox_audio(dbx, folder_path):
    """Return list of audio file metadata dicts from the Dropbox folder."""
    import dropbox
    try:
        result = dbx.files_list_folder(folder_path)
        files  = []
        while True:
            for entry in result.entries:
                if isinstance(entry, dropbox.files.FileMetadata):
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in AUDIO_EXTENSIONS:
                        files.append({
                            "name":          entry.name,
                            "path":          entry.path_lower,
                            "size_bytes":    entry.size,
                            "server_modified": str(entry.server_modified),
                        })
            if not result.has_more:
                break
            result = dbx.files_list_folder_continue(result.cursor)
        return files
    except Exception as e:
        raise RuntimeError(f"Could not list Dropbox folder '{folder_path}': {e}")


def download_from_dropbox(dbx, dropbox_path):
    """Download a file from Dropbox and return its bytes."""
    _, response = dbx.files_download(dropbox_path)
    return response.content


# ─────────────────────────────────────────────────────────────
# AUDIO HELPERS
# ─────────────────────────────────────────────────────────────
def check_limits(file_bytes, filename):
    """Returns (ok: bool, reason: str). Checks size then duration."""
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large: {size_mb:.1f} MB (max {MAX_FILE_SIZE_MB} MB)"

    ext = os.path.splitext(filename)[1].lower() or ".wav"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(file_bytes)
        tmp = f.name
    try:
        from pydub import AudioSegment
        duration = len(AudioSegment.from_file(tmp)) / 1000
        if duration > MAX_DURATION_SECONDS:
            return False, (
                f"Recording too long: {duration/60:.1f} min "
                f"(max {MAX_DURATION_SECONDS//60} min). "
                "Please keep observations under 5 minutes."
            )
    finally:
        os.remove(tmp)
    return True, ""


def preprocess_audio(file_bytes, filename):
    """Normalise volume, return path to a temp WAV file."""
    from pydub import AudioSegment
    from pydub.effects import normalize as pydub_normalize

    ext = os.path.splitext(filename)[1].lower() or ".wav"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(file_bytes)
        src = f.name
    try:
        audio = pydub_normalize(AudioSegment.from_file(src))
    finally:
        os.remove(src)

    fd, wav_path = tempfile.mkstemp(suffix="_processed.wav")
    os.close(fd)
    audio.export(wav_path, format="wav")
    return wav_path


# ─────────────────────────────────────────────────────────────
# WHISPER
# ─────────────────────────────────────────────────────────────
def normalize_domain_terms(text):
    for wrong, correct in DOMAIN_TERMS.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", correct, text, flags=re.I)
    return text


def transcribe_and_translate(wav_path, whisper_model):
    args         = dict(language="ur", temperature=0.0, fp16=False)
    urdu_text    = whisper_model.transcribe(wav_path, task="transcribe", **args).get("text", "").strip()
    english_text = normalize_domain_terms(
        whisper_model.transcribe(wav_path, task="translate", **args).get("text", "").strip()
    )
    return urdu_text, english_text


# ─────────────────────────────────────────────────────────────
# GPT HELPERS
# ─────────────────────────────────────────────────────────────
def extract_reporter_name(urdu_text, openai_client):
    if not urdu_text:
        return "Not available"
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": (
                "Extract the reporter's name from this Urdu safety observation. "
                "Return ONLY the name. If none found, return: Not available\n\n"
                f"Urdu text:\n{urdu_text}"
            )}],
            temperature=0, max_tokens=30
        )
        return response.choices[0].message.content.strip() or "Not available"
    except Exception as e:
        print(f"  Name extraction error: {e}")
        return "Not available"


def categorize_hse_observation(english_text, openai_client):
    valid    = {"Unsafe Act", "Unsafe Condition", "Near Miss", "Incident"}
    fallback = {"category": "Unclassified", "severity": "Not specified",
                "location": "Not specified", "reasoning": "Categorisation failed"}
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": CATEGORY_PROMPT.format(text=english_text)}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        if result.get("category") not in valid:
            result["category"] = "Unclassified"
        return result
    except Exception as e:
        print(f"  Categorisation error: {e}")
        return fallback


def extract_time_from_filename(filename):
    stem = os.path.splitext(filename)[0]
    for fmt in ("%Y_%m_%d_%H_%M_%S", "%Y%m%d_%H%M%S", "%Y-%m-%d_%H-%M-%S"):
        try:
            return datetime.strptime(stem[:len(fmt)], fmt).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
def run_pipeline(ui, supabase_client, whisper_model, openai_client,
                 dropbox_app_key, dropbox_app_secret, dropbox_refresh_token,
                 dropbox_folder):
    """
    ui              — Streamlit container for live log output
    supabase_client — initialised supabase-py Client (for DB only)
    whisper_model   — loaded Whisper model
    openai_client   — OpenAI client
    dropbox_token   — Dropbox access token
    dropbox_folder  — Dropbox folder path e.g. '/ASR Recordings'
    """

    def log(msg, kind="write"):
        print(msg)
        if kind == "error":  ui.error(msg)
        elif kind == "warn": ui.warning(msg)
        elif kind == "ok":   ui.success(msg)
        elif kind == "info": ui.info(msg)
        else:                ui.write(msg)

    print("\n" + "="*60)
    print("PIPELINE STARTED")

    # ── 1. Connect to Dropbox & list files ───────────────────
    try:
        dbx         = get_dropbox_client(dropbox_app_key, dropbox_app_secret, dropbox_refresh_token)
        audio_files = list_dropbox_audio(dbx, dropbox_folder)
    except Exception as e:
        log(f"❌ Dropbox error: {e}", "error")
        return 0

    log(f"📂 Dropbox folder: **{dropbox_folder}**")
    log(f"   Audio files found: **{len(audio_files)}**")

    # ── 2. Find already-processed filenames from Supabase DB ─
    try:
        processed_resp = supabase_client.table("processed_files").select("filename").execute()
        processed      = {row["filename"] for row in processed_resp.data}
    except Exception as e:
        log(f"❌ Could not query processed_files table: {e}", "error")
        return 0

    pending = [f for f in audio_files if f["name"] not in processed]
    log(f"   Already processed: **{len(processed)}** | **New: {len(pending)}**")

    if not pending:
        log("✅ No new recordings to process.", "info")
        return 0

    rows_saved = 0
    progress   = ui.progress(0, text="Starting…")

    for i, file_meta in enumerate(pending):
        filename = file_meta["name"]
        pct      = int((i / len(pending)) * 100)
        progress.progress(pct, text=f"Processing {i+1}/{len(pending)}: {filename}")
        log(f"---\n🎙️ **`{filename}`**")

        wav_path = None
        try:
            # ── Download from Dropbox ─────────────────────────
            log("  ⏳ Downloading from Dropbox…")
            file_bytes = download_from_dropbox(dbx, file_meta["path"])

            # ── Enforce limits ────────────────────────────────
            ok, reason = check_limits(file_bytes, filename)
            if not ok:
                log(f"  ⛔ Skipped — {reason}", "warn")
                supabase_client.table("processed_files").insert(
                    {"filename": filename, "skipped": True, "reason": reason}
                ).execute()
                continue

            # ── Pre-process ───────────────────────────────────
            log("  ⏳ Converting audio…")
            wav_path = preprocess_audio(file_bytes, filename)

            # ── Transcribe ────────────────────────────────────
            log("  ⏳ Transcribing Urdu speech…")
            urdu_text, english_text = transcribe_and_translate(wav_path, whisper_model)
            log(f"  🗣 Urdu: `{urdu_text[:100]}`")
            log(f"  🌐 English: `{english_text[:100]}`")

            # ── Name + Category ───────────────────────────────
            log("  ⏳ Extracting name & categorising…")
            name        = extract_reporter_name(urdu_text, openai_client)
            cat         = categorize_hse_observation(english_text, openai_client)
            report_time = extract_time_from_filename(filename)

            log(f"  ✅ **{cat['category']}** | Severity: {cat['severity']} "
                f"| Location: {cat['location']} | Reporter: {name}")

            # ── Save to Supabase DB ───────────────────────────
            supabase_client.table("observations").insert({
                "time_of_reporting":   report_time,
                "name":                name,
                "urdu_observation":    urdu_text,
                "english_translation": english_text,
                "hse_categorization":  cat["category"],
                "severity":            cat["severity"],
                "location":            cat["location"],
                "ai_reasoning":        cat["reasoning"],
                "audio_file":          filename,
            }).execute()

            supabase_client.table("processed_files").insert(
                {"filename": filename, "skipped": False, "reason": ""}
            ).execute()

            rows_saved += 1

        except Exception as e:
            tb = traceback.format_exc()
            print(f"  ❌ ERROR on {filename}: {e}\n{tb}")
            ui.error(f"❌ **Error on `{filename}`:** {e}")
            ui.code(tb)
        finally:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)

    progress.progress(100, text="Done.")
    progress.empty()

    if rows_saved:
        log(f"✅ Pipeline complete — {rows_saved} new observation(s) saved.", "ok")
    else:
        log("⚠️ Pipeline ran but nothing was saved. Check errors above.", "warn")

    print("="*60 + "\n")
    return rows_saved
