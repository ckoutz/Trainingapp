
import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta
import calendar
from typing import Dict, Any, List, Optional
import ast
import xml.etree.ElementTree as ET

# ------------------------
# OPENAI CLIENT (AI COACH)
# ------------------------
try:
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    client = None  # handled gracefully later


# ------------------------
# CONFIG / FILE PATHS
# ------------------------
st.set_page_config(page_title="2025–26 Training Planner & Log", layout="wide")

LOG_FILE = "training_log.csv"
WORK_FILE = "work_schedule.csv"

CARDIO_MODES = ["Run", "Bike", "Incline Walk"]

# Strength variants
BENCH_ALTS = ["Primary (Barbell Bench)", "Dumbbell Bench", "Floor Press", "Push-Ups", "Machine/Smith Bench"]
PULLUP_ALTS = ["Primary (Pull-Ups)", "Lat Pulldown", "Assisted Pull-Up", "Inverted Row"]
INCLINE_PRESS_ALTS = ["Primary (Incline Press)", "DB Incline Press", "Machine Incline", "Pike Push-Ups"]
ROW_ALTS = ["Primary (Row)", "DB Row", "Cable Row", "Machine Row"]
LATERAL_ALTS = ["Primary (DB Lateral Raise)", "Cable Lateral Raise", "Plate Raise"]
TRICEPS_ALTS = ["Primary (Triceps)", "Rope Pushdown", "DB OH Extension", "Close-Grip Pushup"]

DEADLIFT_ALTS = ["Primary (Deadlift/RDL)", "KB Deadlift", "DB RDL", "Back Extensions"]
SPLIT_SQUAT_ALTS = ["Primary (Split Squat/Lunge)", "DB Lunge", "Bulgarian Split Squat", "Step-Ups"]
HIPTHRUST_ALTS = ["Primary (Hip Thrust)", "DB Hip Thrust", "Glute Bridge", "Single-Leg Hip Bridge"]
BICEPS_ALTS = ["Primary (Biceps Curl)", "DB Curl", "Hammer Curl", "Band Curl"]
CORE_ALTS = ["Primary (Core)", "Plank", "Side Plank", "Dead Bug"]

ME_ALTS = ["Primary (Step-Ups)", "StairStepper", "Incline Grind (sustained)", "Ruck Walk", "Box Step Ladder"]


# ------------------------
# WORK SCHEDULE HELPERS
# ------------------------
def load_work_schedule() -> pd.DataFrame:
    try:
        df = pd.read_csv(WORK_FILE)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["date", "is_work"])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["date"] = df["date"].dt.date
    return df


def save_work_schedule(df: pd.DataFrame) -> None:
    df_out = df.copy()
    if not df_out.empty:
        df_out["date"] = pd.to_datetime(df_out["date"])
    df_out.to_csv(WORK_FILE, index=False)


def is_workday(d: date) -> bool:
    df = load_work_schedule()
    if df.empty:
        return False
    return bool(df[df["date"] == d]["is_work"].any())


def update_work_schedule_for_month(year: int, month: int, selections: Dict[int, bool]) -> None:
    df = load_work_schedule()
    if not df.empty:
        mask_same_month = df["date"].apply(lambda x: x.year == year and x.month == month)
        df = df[~mask_same_month]

    new_rows = [{"date": date(year, month, day), "is_work": flag} for day, flag in selections.items()]
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    save_work_schedule(df)


# ------------------------
# TCX PARSER
# ------------------------
def parse_tcx(file_obj):
    """Parse TCX and return dict of metrics, or None if parse fails."""
    try:
        tree = ET.parse(file_obj)
        root = tree.getroot()
    except Exception:
        return None

    ns = {"tcx": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}
    tps = root.findall(".//tcx:Trackpoint", ns)

    times, hrs, cadence, altitude, distance = [], [], [], [], []

    for tp in tps:
        t = tp.find("tcx:Time", ns)
        hr = tp.find(".//tcx:HeartRateBpm/tcx:Value", ns)
        cad = tp.find(".//tcx:Cadence", ns)
        alt = tp.find(".//tcx:AltitudeMeters", ns)
        dist = tp.find(".//tcx:DistanceMeters", ns)

        if t is not None and t.text:
            try:
                times.append(datetime.fromisoformat(t.text.replace("Z", "+00:00")))
            except Exception:
                pass
        if hr is not None and hr.text:
            try:
                hrs.append(float(hr.text))
            except Exception:
                pass
        if cad is not None and cad.text:
            try:
                cadence.append(float(cad.text))
            except Exception:
                pass
        if alt is not None and alt.text:
            try:
                altitude.append(float(alt.text))
            except Exception:
                pass
        if dist is not None and dist.text:
            try:
                distance.append(float(dist.text))
            except Exception:
                pass

    if not times:
        return None

    duration_sec = (times[-1] - times[0]).total_seconds()
    distance_m = distance[-1] if distance else 0.0
    avg_hr = sum(hrs) / len(hrs) if hrs else 0.0
    max_hr = max(hrs) if hrs else 0.0
    avg_cad = sum(cadence) / len(cadence) if cadence else 0.0

    elevation_gain = 0.0
    for i in range(1, len(altitude)):
        diff = altitude[i] - altitude[i - 1]
        if diff > 0:
            elevation_gain += diff

    if distance_m > 0:
        pace_min_per_km = (duration_sec / 60.0) / (distance_m / 1000.0)
    else:
        pace_min_per_km = 0.0

    half = len(hrs) // 2
    if half > 0:
        first = sum(hrs[:half]) / half
        second = sum(hrs[half:]) / (len(hrs) - half)
        hr_drift = (second - first) / first if first > 0 else 0.0
    else:
        hr_drift = 0.0

    return {
        "duration_sec": duration_sec,
        "distance_m": distance_m,
        "avg_hr": avg_hr,
        "max_hr": max_hr,
        "avg_cadence": avg_cad,
        "elevation_gain_m": elevation_gain,
        "pace_min_per_km": pace_min_per_km,
        "hr_drift": hr_drift,
    }


# ------------------------
# TRAINING PLAN LOGIC
# ------------------------
def get_phase_and_day_plan(d: date):
    """
    Base deterministic plan.
    Returns (phase, day_type, planned_description, kind).
    """
    dt = d

    # Macro dates
    start_2a = date(2025, 12, 8)
    end_2a   = date(2026, 1, 19)

    start_2b = date(2026, 1, 20)
    end_2b   = date(2026, 2, 23)

    start_2c = date(2026, 2, 24)
    end_2c   = date(2026, 3, 30)

    start_3a = date(2026, 3, 31)
    end_3a   = date(2026, 5, 4)

    start_3b = date(2026, 5, 5)
    end_3b   = date(2026, 6, 22)

    start_4  = date(2026, 6, 23)
    end_4    = date(2026, 8, 10)

    start_taper = date(2026, 8, 11)
    race_day    = date(2026, 8, 29)

    if dt < start_2a or dt > race_day:
        return (
            "Unprogrammed",
            "Manual / Free Day",
            "No structured plan programmed for this date yet. Use as free day or log a manual workout.",
            "Manual",
        )

    dow = dt.weekday()  # Mon=0

    def tempo_desc():
        return (
            "Tempo Run — 30–35 min:\n"
            "• 5–10 min warm-up (easy).\n"
            "• 2–3 × 6–8 min at tempo (RPE 6–7) with 2–3 min easy between.\n"
            "• 5–10 min cool-down.\n"
            "Goal: aerobic power and running economy."
        )

    def long_z2_desc(mode="Bike/Run/Hike"):
        return (
            f"Long Z2 {mode} — 75–90 min.\n"
            "Stay in comfortable Z2 (RPE 4–6). You should be able to talk."
        )

    def flex_z2_desc():
        return (
            "Flexible Z2 Cardio — 30–45 min.\n"
            "Choose Run, Bike, or Incline Walk. Stay easy-moderate (Z2)."
        )

    def incline_desc(long=False):
        if long:
            return "Long Incline/Hike — 60–90 min continuous uphill Z2."
        return "Incline Z2 — 40–60 min uphill treadmill or outdoor hiking in Z2."

    def hill_desc():
        return "Hill Hike — 45–75 min up/down hills in Z2. Grind, don't sprint."

    def me_desc():
        return (
            "Muscular Endurance Step-Ups + Easy Cardio:\n"
            "• 3 rounds: 6 min continuous step-ups + 2 min easy.\n"
            "• Then 20–25 min easy Z1–Z2 cardio."
        )

    def brick_desc(long=False):
        if long:
            return (
                "Long Brick (Bike → Run):\n"
                "• Bike 60–90 min Z2 with some race-like sections.\n"
                "• Quick change.\n"
                "• Run 15–25 min Z2–low Z3."
            )
        return (
            "Brick (Bike → Run):\n"
            "• Bike 30–45 min Z2–Z3-.\n"
            "• Quick change.\n"
            "• Run 10–15 min easy-steady."
        )

    def race_like_desc(modality):
        return (
            f"Race-Pace {modality} session.\n"
            "Short intervals at target race effort with easy recovery.\n"
            "Goal: lock in feel of race pace without big fatigue."
        )

    # 2A
    if start_2a <= dt <= end_2a:
        phase = "Phase 2A – Strength + Aerobic Base"
        if dow == 0:
            return phase, "Strength A – Upper Focus", "Bench, Pull-Ups, Incline Press, Row, Laterals, Triceps.", "StrengthA"
        if dow == 1:
            return phase, "Tempo Run", tempo_desc(), "Tempo"
        if dow == 2:
            return phase, "Z2 Hill Hike", hill_desc(), "Incline"
        if dow == 3:
            return phase, "Strength B – Lower/Hypertrophy", "Deadlift/RDL, Split Squat/Lunge, Hip Thrust, Row Var, Biceps, Core.", "StrengthB"
        if dow == 4:
            return phase, "Aerobic Flex", flex_z2_desc(), "FlexCardio"
        if dow == 5:
            return phase, "Long Z2", long_z2_desc(), "LongZ2"
        return phase, "Off / Recovery", "Off day. Absorb the week.", "Off"

    # 2B
    if start_2b <= dt <= end_2b:
        phase = "Phase 2B – Muscular Endurance"
        if dow == 0:
            return phase, "Z2 Incline", incline_desc(False), "Incline"
        if dow == 1:
            return phase, "Strength A – Upper (De-emphasized)", "Keep loads submax; upper maintenance.", "StrengthA"
        if dow == 2:
            return phase, "ME Step-Ups + Easy Cardio", me_desc(), "ME"
        if dow == 3:
            return phase, "Z2 Incline", incline_desc(False), "Incline"
        if dow == 4:
            return phase, "Strength B – Lower (Light)", "Movement quality > heavy loading.", "StrengthB"
        if dow == 5:
            return phase, "Long Incline / Hike", incline_desc(True), "LongZ2"
        return phase, "Off / Recovery", "Off day.", "Off"

    # 2C
    if start_2c <= dt <= end_2c:
        phase = "Phase 2C – Aerobic Power / Transition"
        if dow == 0:
            return phase, "Strength A – Upper", "Upper strength; no grinders.", "StrengthA"
        if dow == 1:
            return phase, "Tempo / Threshold Run", tempo_desc(), "Tempo"
        if dow == 2:
            return phase, "Z2 Bike / Incline", flex_z2_desc(), "FlexCardio"
        if dow == 3:
            return phase, "Strength B – Lower", "Maintain leg strength as intensity rises.", "StrengthB"
        if dow == 4:
            return phase, "Easy Z2 Cardio", flex_z2_desc(), "FlexCardio"
        if dow == 5:
            return phase, "Long Z2", long_z2_desc("Bike/Run combo"), "LongZ2"
        return phase, "Off / Recovery", "Off day.", "Off"

    # 3A
    if start_3a <= dt <= end_3a:
        phase = "Phase 3A – Triathlon Base"
        if dow == 0:
            return phase, "Swim + Short Strength A", "Technique swim + short upper strength.", "StrengthA"
        if dow == 1:
            return phase, "Z2 Bike", flex_z2_desc(), "FlexCardio"
        if dow == 2:
            return phase, "Z2 Run", flex_z2_desc(), "FlexCardio"
        if dow == 3:
            return phase, "Swim + Strength B (Light)", "Swim + lighter lower strength.", "StrengthB"
        if dow == 4:
            return phase, "Tempo Bike / Run", tempo_desc(), "Tempo"
        if dow == 5:
            return phase, "Long Z2 Bike / Run", long_z2_desc("Bike/Run"), "LongZ2"
        return phase, "Off / Recovery", "Off day.", "Off"

    # 3B
    if start_3b <= dt <= end_3b:
        phase = "Phase 3B – Triathlon Build"
        if dow == 0:
            return phase, "Swim Intervals + Optional Strength A", "Harder swim + optional upper.", "StrengthA"
        if dow == 1:
            return phase, "Bike Tempo", "Bike tempo intervals 45–75 min.", "Tempo"
        if dow == 2:
            return phase, "Run Intervals", "Run intervals 30–45 min.", "Tempo"
        if dow == 3:
            return phase, "Swim + Easy Strength B", "Maintain lower strength lightly.", "StrengthB"
        if dow == 4:
            return phase, "Brick (Bike → Run)", brick_desc(False), "TriBrick"
        if dow == 5:
            return phase, "Long Brick / Long Bike", brick_desc(True), "TriBrick"
        return phase, "Off / Recovery", "Off day.", "Off"

    # 4
    if start_4 <= dt <= end_4:
        phase = "Phase 4 – Peak / Specific"
        if dow == 0:
            return phase, "Race-Pace Swim", race_like_desc("Swim"), "TriRaceLike"
        if dow == 1:
            return phase, "Race-Pace Bike", race_like_desc("Bike"), "TriRaceLike"
        if dow == 2:
            return phase, "Race-Pace Run", race_like_desc("Run"), "TriRaceLike"
        if dow == 3:
            return phase, "Easy Swim + Light Strength", "Movement quality only.", "FlexCardio"
        if dow == 4:
            return phase, "Race Simulation Brick", brick_desc(False), "TriBrick"
        if dow == 5:
            return phase, "Long Race-Specific", brick_desc(True), "TriBrick"
        return phase, "Off / Recovery", "Off day.", "Off"

    # Taper
    if start_taper <= dt <= race_day:
        phase = "Taper"
        if dt == race_day:
            return phase, "RACE DAY – Tahoe Triathlon", "Execute, stay calm, and have fun.", "TriRaceLike"

        dow = dt.weekday()
        if dow in [0, 2]:
            return phase, "Short Race-Pace Sharpening", race_like_desc("Swim/Bike/Run"), "TriRaceLike"
        if dow in [1, 3]:
            return phase, "Easy Z2 Cardio", flex_z2_desc(), "FlexCardio"
        if dow == 4:
            return phase, "Very Short Brick", brick_desc(False), "TriBrick"
        if dow == 5:
            return phase, "Optional Easy Session", "Skip if tired.", "FlexCardio"
        return phase, "Off / Pre-Race Rest", "Off and prep.", "Off"

    return "Unprogrammed", "Manual / Free Day", "No structured plan.", "Manual"


def adjust_for_workday(phase, day_type, planned, kind, work: bool):
    """Light adjustment if it's a work day."""
    if not work:
        return phase, day_type, planned, kind

    if kind in ["Off", "Manual"]:
        return phase, day_type, planned, kind

    if kind in ["LongZ2", "ME", "TriBrick", "TriRaceLike"]:
        new_day = day_type + " (Workday adjusted)"
        new_plan = (
            "Travel/work-day version:\n"
            "25–40 min easy Z1–Z2 cardio of choice.\n"
            "Goal: circulation, not heroics."
        )
        return phase, new_day, new_plan, "FlexCardio"

    if kind in ["StrengthA", "StrengthB"]:
        new_day = day_type + " (Workday lighter)"
        new_plan = planned + "\n\nWorkday tweak: drop one set and keep 2–4 reps in reserve."
        return phase, new_day, new_plan, kind

    if kind == "Tempo":
        new_day = day_type + " (Workday version)"
        new_plan = (
            "Either 1–2 short tempo reps, or convert to 30–40 min pure Z2 if beat up.\n"
            "Work/travel makes tempo feel harder; back it off if needed."
        )
        return phase, new_day, new_plan, kind

    return phase, day_type, planned, kind


# ------------------------
# LOG LOAD / SAVE (SANITIZED)
# ------------------------
def load_log() -> pd.DataFrame:
    try:
        df = pd.read_csv(LOG_FILE)
    except FileNotFoundError:
        return pd.DataFrame()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["date"] = df["date"].dt.date
    return df


def save_log_for_date(d: date, row_data: Dict[str, Any]) -> None:
    df = load_log()
    if "date" in df.columns and not df.empty:
        df = df[df["date"] != d]

    row_out = row_data.copy()
    row_out["date"] = d
    df = pd.concat([df, pd.DataFrame([row_out])], ignore_index=True)

    df_out = df.copy()
    if not df_out.empty:
        df_out["date"] = pd.to_datetime(df_out["date"])
    df_out.to_csv(LOG_FILE, index=False)


def get_log_for_date(d: date) -> Dict[str, Any]:
    df = load_log()
    if df.empty or "date" not in df.columns:
        return {}
    rows = df[df["date"] == d]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def delete_log_for_date(d: date) -> None:
    df = load_log()
    if df.empty:
        return
    df = df[df["date"] != d]
    df_out = df.copy()
    if not df_out.empty:
        df_out["date"] = pd.to_datetime(df_out["date"])
    df_out.to_csv(LOG_FILE, index=False)


# ------------------------
# EFFECTIVE PLAN (with overrides)
# ------------------------
def get_effective_plan(d: date):
    base_phase, base_day_type, base_planned, base_kind = get_phase_and_day_plan(d)
    entry = get_log_for_date(d)
    if entry.get("use_plan_override"):
        return (
            entry.get("override_phase", base_phase),
            entry.get("override_day_type", base_day_type),
            entry.get("override_planned", base_planned),
            entry.get("override_kind", base_kind),
        )
    return base_phase, base_day_type, base_planned, base_kind


# ------------------------
# STRENGTH HELPERS
# ------------------------
def get_strength_exercises(kind: str) -> List[Dict[str, Any]]:
    if kind == "StrengthA":
        return [
            {"name": "Bench Press", "alts": BENCH_ALTS},
            {"name": "Pull-Ups", "alts": PULLUP_ALTS},
            {"name": "Incline Press", "alts": INCLINE_PRESS_ALTS},
            {"name": "Row", "alts": ROW_ALTS},
            {"name": "Lateral Raises", "alts": LATERAL_ALTS},
            {"name": "Triceps", "alts": TRICEPS_ALTS},
        ]
    if kind == "StrengthB":
        return [
            {"name": "Deadlift / RDL", "alts": DEADLIFT_ALTS},
            {"name": "Split Squat / Lunge", "alts": SPLIT_SQUAT_ALTS},
            {"name": "Hip Thrust", "alts": HIPTHRUST_ALTS},
            {"name": "Row Variation", "alts": ROW_ALTS},
            {"name": "Biceps", "alts": BICEPS_ALTS},
            {"name": "Core", "alts": CORE_ALTS},
        ]
    if kind == "ME":
        return [{"name": "ME Step-Ups", "alts": ME_ALTS}]
    return []


def get_last_strength_entry(df: pd.DataFrame, current_date: date, exercise_name: str, variant_name: str):
    if df.empty or "strength_main" not in df.columns:
        return None

    past = df[df["date"] < current_date]
    if past.empty:
        return None

    past = past.dropna(subset=["strength_main", "strength_extra"], how="all")
    if past.empty:
        return None

    past = past.sort_values("date", ascending=False)

    for _, row in past.iterrows():
        for col in ["strength_main", "strength_extra"]:
            block_str = row.get(col, "")
            if not isinstance(block_str, str) or not block_str.strip():
                continue
            try:
                entries = ast.literal_eval(block_str)
            except Exception:
                entries = []
            if not isinstance(entries, list):
                continue
            for e in entries:
                if not isinstance(e, dict):
                    continue
                ex_name = e.get("exercise")
                var_name = e.get("variant") or e.get("alt")
                if ex_name == exercise_name and var_name == variant_name:
                    result = e.copy()
                    result["date"] = row["date"]
                    return result
    return None


# ------------------------
# AI COACH
# ------------------------
def init_ai_state():
    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hey, I'm your AI coach. I can see your plan, recent training, "
                    "and upcoming week. Ask me about adjustments, fatigue, or how to "
                    "shape today around your goals."
                ),
            }
        ]


def build_context_summary(selected_date: date) -> str:
    parts = []

    # Today's plan
    phase, day_type, planned, kind = get_effective_plan(selected_date)
    work_flag = is_workday(selected_date)
    parts.append(
        f"Today ({selected_date}): phase={phase}, day_type={day_type}, kind={kind}, workday={work_flag}. "
        f"Planned: {planned}"
    )

    # Upcoming 7 days
    upcoming_lines = []
    for i in range(1, 8):
        d = selected_date + timedelta(days=i)
        ph, dtp, _, knd = get_effective_plan(d)
        upcoming_lines.append(f"{d}: {dtp} ({ph}, kind={knd})")
    parts.append("Next 7 days programmed: " + " | ".join(upcoming_lines))

    # Recent log (last 7 days)
    df = load_log()
    if not df.empty:
        recent = df[(df["date"] <= selected_date)].sort_values("date", ascending=False).head(7)
        daily_lines = []
        for _, row in recent.iterrows():
            d = row.get("date")
            dt_adj = row.get("day_type_adjusted", "")
            mode = row.get("mode", "")
            work = row.get("workday", False)
            cmode_main = row.get("cardio_main_mode", "")
            dur_main = row.get("cardio_main_duration_min", 0)
            daily_lines.append(
                f"{d}: {dt_adj}, mode={mode}, workday={work}, main_cardio={cmode_main} {dur_main}min"
            )
        if daily_lines:
            parts.append("Last 7 logged days: " + " || ".join(daily_lines))

    return "\n".join(parts)


def ai_add_message(role: str, content: str):
    st.session_state.ai_messages.append({"role": role, "content": content})


def ai_call_coach(user_message: str) -> str:
    if client is None:
        return "AI is not configured yet. Check that OPENAI_API_KEY is set in Streamlit secrets."

    try:
        selected_date = st.session_state.get("selected_date", date.today())
        context_text = build_context_summary(selected_date)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise, practical training coach. "
                    "You help the user interpret workouts, adjust plan for soreness/work travel, "
                    "and prepare for the Tahoe Triathlon (Aug 29, 2026) plus GBRS-style strength. "
                    "Use the provided context about their schedule and recent logs. "
                    "Give short, concrete suggestions, not essays."
                ),
            },
            {
                "role": "system",
                "content": "Training context:\n" + context_text,
            },
        ]

        for m in st.session_state.ai_messages:
            messages.append({"role": m["role"], "content": m["content"]})

        messages.append({"role": "user", "content": user_message})

        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            max_tokens=400,
            temperature=0.5,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"OpenAI error: {e}"


def render_ai_coach_panel():
    st.markdown("---")
    with st.expander("🧠 AI Coach (context-aware)", expanded=False):
        for m in st.session_state.ai_messages:
            if m["role"] == "user":
                st.markdown(f"**You:** {m['content']}")
            else:
                st.markdown(f"**Coach:** {m['content']}")

        st.markdown("---")
        key = f"ai_input_{len(st.session_state.ai_messages)}"
        user_q = st.text_input("Ask your coach something:", key=key)
        send = st.button("Send to Coach")

        if send and user_q.strip():
            ai_add_message("user", user_q.strip())
            reply = ai_call_coach(user_q.strip())
            ai_add_message("assistant", reply)


# ------------------------
# COACHING TIPS
# ------------------------
def coaching_tips_for_kind(kind: str) -> str:
    if kind == "Tempo":
        return (
            "Tempo tips:\n"
            "• Start conservative; first rep should feel almost too easy.\n"
            "• You should never sprint; pace should feel 'comfortably hard'.\n"
            "• If HR spikes or breathing is wild, shorten the reps and extend recovery.\n"
            "• If you’re exhausted, convert the whole thing to Z2 — still a win."
        )
    if kind in ["LongZ2", "FlexCardio", "Incline"]:
        return (
            "Z2 / Incline tips:\n"
            "• You should be able to speak in full sentences.\n"
            "• Keep ego out of it: slower is fine if HR stays smoother.\n"
            "• The goal here is volume and capillary work, not speed.\n"
            "• The last 20 minutes of a long Z2 day do most of the adaptation."
        )
    if kind in ["StrengthA", "StrengthB", "ME"]:
        return (
            "Strength tips:\n"
            "• Leave 1–3 reps in reserve on most sets; no grinders.\n"
            "• Prioritize controlled eccentrics and consistent rest times.\n"
            "• If legs are smoked from hills/ME, reduce lower-body volume.\n"
            "• Track weight and RPE so progression is deliberate, not random."
        )
    if kind in ["TriBrick", "TriRaceLike"]:
        return (
            "Tri-specific tips:\n"
            "• For bricks, accept that the first 3–5 minutes of the run feel awkward.\n"
            "• Focus on cadence and posture instead of pace chasing.\n"
            "• Race-pace work should feel repeatable, not maximal.\n"
            "• Better to be 5% undercooked than 1% overcooked on race day."
        )
    return (
        "General tips:\n"
        "• Match today's effort to your sleep, stress, and soreness.\n"
        "• One workout never makes you; one missed workout never ruins you.\n"
        "• Keep notes on how sessions feel so future you can adjust smarter."
    )


# ------------------------
# UI HELPERS – STRENGTH & CARDIO BLOCKS
# ------------------------
def render_strength_block(label: str, kind: str, d: date, existing_block: Optional[str]) -> str:
    df_log = load_log()
    exercises = get_strength_exercises(kind)
    st.markdown(f"#### {label} Strength Block")

    block_entries = []
    existing_entries = []
    if isinstance(existing_block, str) and existing_block.strip():
        try:
            parsed = ast.literal_eval(existing_block)
            if isinstance(parsed, list):
                existing_entries = parsed
        except Exception:
            existing_entries = []

    for i, ex in enumerate(exercises):
        st.markdown(f"**{ex['name']}**")
        default_variant = ex["alts"][0]
        existing_variant = default_variant
        existing_sets = ""
        existing_reps = ""
        existing_weight = ""
        existing_rpe = ""

        if existing_entries:
            for entry in existing_entries:
                if entry.get("exercise") == ex["name"]:
                    existing_variant = entry.get("variant", default_variant)
                    existing_sets = entry.get("sets", "")
                    existing_reps = entry.get("reps", "")
                    existing_weight = entry.get("weight", "")
                    existing_rpe = entry.get("rpe", "")
                    break

        variant = st.selectbox(
            "Variant",
            ex["alts"],
            index=ex["alts"].index(existing_variant) if existing_variant in ex["alts"] else 0,
            key=f"{d}_{label}_variant_{i}",
        )

        last = get_last_strength_entry(df_log, d, ex["name"], variant)
        if last:
            st.caption(
                f"Last ({variant}) on {last.get('date')}: "
                f"{last.get('sets', '')}×{last.get('reps', '')} @ {last.get('weight', '')} "
                f"(RPE {last.get('rpe', '')})"
            )
        else:
            st.caption(f"Last ({variant}): no previous log yet.")

        c1, c2, c3, c4 = st.columns(4)
        sets = c1.text_input("Sets", value=str(existing_sets), key=f"{d}_{label}_sets_{i}")
        reps = c2.text_input("Reps", value=str(existing_reps), key=f"{d}_{label}_reps_{i}")
        weight = c3.text_input("Weight", value=str(existing_weight), key=f"{d}_{label}_wt_{i}")
        rpe = c4.text_input("RPE", value=str(existing_rpe), key=f"{d}_{label}_rpe_{i}")

        block_entries.append(
            {
                "exercise": ex["name"],
                "variant": variant,
                "sets": sets,
                "reps": reps,
                "weight": weight,
                "rpe": rpe,
            }
        )

    return str(block_entries)


def render_cardio_block(label: str, d: date, existing: Dict[str, Any], allow_tcx: bool):
    st.markdown(f"#### {label} Cardio Block")

    cardio_mode = st.selectbox(
        "Cardio mode",
        CARDIO_MODES,
        index=CARDIO_MODES.index(existing.get("mode", "Run")) if existing.get("mode", "Run") in CARDIO_MODES else 0,
        key=f"{d}_{label}_mode",
    )
    duration = st.number_input(
        "Duration (min)",
        min_value=0,
        max_value=600,
        value=int(existing.get("duration", 0)) if isinstance(existing.get("duration", 0), (int, float)) else 0,
        key=f"{d}_{label}_dur",
    )
    distance = st.text_input(
        "Distance (mi/km)",
        value=str(existing.get("distance", "")),
        key=f"{d}_{label}_dist",
    )
    avg_hr = st.text_input(
        "Avg HR (bpm)",
        value=str(existing.get("avg_hr", "")),
        key=f"{d}_{label}_avg_hr",
    )
    rpe = st.slider(
        "Cardio RPE (1–10)",
        1,
        10,
        int(existing.get("rpe", 6)) if isinstance(existing.get("rpe", 6), (int, float)) else 6,
        key=f"{d}_{label}_rpe",
    )

    tcx_metrics = None
    if allow_tcx:
        st.markdown("Optional: Upload TCX to auto-fill")
        tcx_file = st.file_uploader(f"Upload TCX file (.tcx) for {label}", type=None, key=f"{d}_{label}_tcx")
        if tcx_file is not None:
            if not tcx_file.name.lower().endswith(".tcx"):
                st.error("Please upload a file with the .tcx extension.")
            else:
                parsed = parse_tcx(tcx_file)
                if parsed:
                    tcx_metrics = parsed
                    duration_min = int(parsed["duration_sec"] / 60)
                    distance_mi = round(parsed["distance_m"] / 1609.34, 2) if parsed["distance_m"] > 0 else 0.0
                    st.success("TCX parsed.")
                    st.info(
                        f"Duration: {duration_min} min | Distance: {distance_mi} mi | "
                        f"Avg HR: {int(parsed['avg_hr'])} | Max HR: {int(parsed['max_hr'])} | "
                        f"Elev: {int(parsed['elevation_gain_m'])} m | Pace: {parsed['pace_min_per_km']:.2f} min/km"
                    )
                else:
                    st.error("Could not parse TCX file; logging manual values.")

    card = {
        "mode": cardio_mode,
        "duration": duration,
        "distance": distance,
        "avg_hr": avg_hr,
        "rpe": rpe,
    }
    return card, tcx_metrics


# ------------------------
# TODAY PAGE (DAILY EDITOR)
# ------------------------
def render_today_page():
    st.title("Today")

    # Selected date in session
    today = date.today()
    if "selected_date" not in st.session_state:
        st.session_state["selected_date"] = today

    selected = st.date_input("Choose date", value=st.session_state["selected_date"])
    st.session_state["selected_date"] = selected

    existing = get_log_for_date(selected)
    base_phase, base_day_type, base_planned, base_kind = get_effective_plan(selected)
    work_flag = is_workday(selected)

    existing_mode = existing.get("mode", "Auto (Programmed/Adjusted)")
    mode = st.radio(
        "Workout Mode",
        ["Auto (Programmed/Adjusted)", "Manual / Custom", "Rest / Skip"],
        index=["Auto (Programmed/Adjusted)", "Manual / Custom", "Rest / Skip"].index(existing_mode)
        if existing_mode in ["Auto (Programmed/Adjusted)", "Manual / Custom", "Rest / Skip"]
        else 0,
        horizontal=True,
    )

    if mode == "Auto (Programmed/Adjusted)":
        override = st.checkbox(
            "Use original programmed workout even if this is a workday",
            value=bool(existing.get("override_original", False)),
        )
        if override:
            phase, day_type, planned, kind = base_phase, base_day_type, base_planned, base_kind
        else:
            phase, day_type, planned, kind = adjust_for_workday(base_phase, base_day_type, base_planned, base_kind, work_flag)
    elif mode == "Manual / Custom":
        phase = existing.get("phase", "Manual")
        day_type = existing.get("day_type_adjusted", "Manual / Custom")
        planned = existing.get("planned_manual", "Manual custom session. Use blocks below to record.")
        kind = "Manual"
    else:
        phase = base_phase if base_phase != "Unprogrammed" else "Rest"
        day_type = "Rest / Skip"
        planned = "Rest or intentional skip. One missed day in a good block is noise, not disaster."
        kind = "Off"

    # Header card
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader(phase)
        st.caption(f"Day Type: {day_type}")
    with c2:
        if work_flag:
            st.markdown("🧳 **Workday**")
        else:
            st.markdown("🏠 **Home day**")

    st.markdown("**Planned Session Details:**")
    st.write(planned)

    with st.expander("🧠 Coaching Tips", expanded=False):
        st.write(coaching_tips_for_kind(kind))

    # Cardio
    st.markdown("---")
    st.markdown("### Cardio")

    cardio_kinds = ["Tempo", "FlexCardio", "LongZ2", "Incline", "ME", "TriBrick", "TriRaceLike"]
    is_plan_cardio_day = kind in cardio_kinds

    cardio_main = {}
    cardio_extra = {}
    tcx_main = None

    if is_plan_cardio_day:
        existing_main = {
            "mode": existing.get("cardio_main_mode", ""),
            "duration": existing.get("cardio_main_duration_min", 0),
            "distance": existing.get("cardio_main_distance", ""),
            "avg_hr": existing.get("cardio_main_avg_hr", ""),
            "rpe": existing.get("cardio_main_rpe_1_10", 6),
        }
        st.markdown("#### Main Cardio (from plan)")
        cardio_main, tcx_main = render_cardio_block("Main", selected, existing_main, allow_tcx=True)
    else:
        st.caption("No specific cardio prescribed today, but you can still log optional work.")
        cardio_main = {}
        tcx_main = None

    add_extra_cardio = st.checkbox(
        "Add an extra cardio session today?",
        value=bool(existing.get("cardio_extra_mode", "")),
    )

    if add_extra_cardio:
        existing_extra = {
            "mode": existing.get("cardio_extra_mode", ""),
            "duration": existing.get("cardio_extra_duration_min", 0),
            "distance": existing.get("cardio_extra_distance", ""),
            "avg_hr": existing.get("cardio_extra_avg_hr", ""),
            "rpe": existing.get("cardio_extra_rpe_1_10", 6),
        }
        cardio_extra, _ = render_cardio_block("Extra", selected, existing_extra, allow_tcx=False)
    else:
        cardio_extra = {}

    # Strength
    st.markdown("---")
    st.markdown("### Strength / ME")

    strength_main_block = ""
    strength_extra_block = ""

    strength_kinds = ["StrengthA", "StrengthB", "ME"]
    if kind in strength_kinds:
        strength_main_block = render_strength_block(
            "Main",
            kind,
            selected,
            existing.get("strength_main", ""),
        )

    add_extra_strength = st.checkbox(
        "Add an extra strength session today?",
        value=bool(existing.get("strength_extra", "")),
    )
    if add_extra_strength:
        extra_kind = st.selectbox(
            "Extra strength type",
            ["StrengthA", "StrengthB", "ME"],
            index=0,
        )
        strength_extra_block = render_strength_block(
            "Extra",
            extra_kind,
            selected,
            existing.get("strength_extra", ""),
        )

    # Trackers
    st.markdown("---")
    st.markdown("### Daily Trackers")
    df_existing = existing

    cA, cB, cC = st.columns(3)
    hrv = cA.text_input("HRV", value=str(df_existing.get("hrv", "")))
    sleep_hrs = cB.text_input("Sleep hours", value=str(df_existing.get("sleep_hours", "")))
    mood = cC.slider(
        "Mood (1–5)",
        1,
        5,
        int(df_existing.get("mood_1_5", 3)) if isinstance(df_existing.get("mood_1_5", 3), (int, float)) else 3,
    )

    cD, cE, _ = st.columns(3)
    soreness = cD.slider(
        "Soreness (1–5)",
        1,
        5,
        int(df_existing.get("soreness_1_5", 3)) if isinstance(df_existing.get("soreness_1_5", 3), (int, float)) else 3,
    )
    energy = cE.slider(
        "Energy (1–5)",
        1,
        5,
        int(df_existing.get("energy_1_5", 3)) if isinstance(df_existing.get("energy_1_5", 3), (int, float)) else 3,
    )

    notes = st.text_area(
        "Notes",
        value=str(df_existing.get("notes", "")),
        height=120,
    )

    # Assemble row + autosave
    row: Dict[str, Any] = {
        "phase": phase,
        "day_type": base_day_type,
        "day_type_adjusted": day_type,
        "kind": base_kind,
        "kind_adjusted": kind,
        "workday": work_flag,
        "mode": mode,
    }

    # Strength
    row["strength_main"] = strength_main_block
    row["strength_extra"] = strength_extra_block

    # Cardio main
    if cardio_main:
        if tcx_main:
            duration_min = int(tcx_main["duration_sec"] / 60)
            distance_mi = round(tcx_main["distance_m"] / 1609.34, 2) if tcx_main["distance_m"] > 0 else 0.0
            row.update(
                {
                    "cardio_main_mode": cardio_main["mode"],
                    "cardio_main_duration_min": duration_min,
                    "cardio_main_distance": distance_mi,
                    "cardio_main_avg_hr": int(tcx_main["avg_hr"]),
                    "cardio_main_rpe_1_10": cardio_main["rpe"],
                    "cardio_main_max_hr": int(tcx_main["max_hr"]),
                    "cardio_main_elev_gain_m": int(tcx_main["elevation_gain_m"]),
                    "cardio_main_avg_cadence": int(tcx_main["avg_cadence"]),
                    "cardio_main_pace_min_per_km": round(tcx_main["pace_min_per_km"], 2),
                    "cardio_main_hr_drift": round(tcx_main["hr_drift"], 4),
                }
            )
        else:
            row.update(
                {
                    "cardio_main_mode": cardio_main["mode"],
                    "cardio_main_duration_min": cardio_main["duration"],
                    "cardio_main_distance": cardio_main["distance"],
                    "cardio_main_avg_hr": cardio_main["avg_hr"],
                    "cardio_main_rpe_1_10": cardio_main["rpe"],
                }
            )
    else:
        row.update(
            {
                "cardio_main_mode": "",
                "cardio_main_duration_min": 0,
                "cardio_main_distance": "",
                "cardio_main_avg_hr": "",
                "cardio_main_rpe_1_10": 0,
            }
        )

    # Cardio extra
    if cardio_extra:
        row.update(
            {
                "cardio_extra_mode": cardio_extra["mode"],
                "cardio_extra_duration_min": cardio_extra["duration"],
                "cardio_extra_distance": cardio_extra["distance"],
                "cardio_extra_avg_hr": cardio_extra["avg_hr"],
                "cardio_extra_rpe_1_10": cardio_extra["rpe"],
            }
        )
    else:
        row.update(
            {
                "cardio_extra_mode": "",
                "cardio_extra_duration_min": 0,
                "cardio_extra_distance": "",
                "cardio_extra_avg_hr": "",
                "cardio_extra_rpe_1_10": 0,
            }
        )

    # Trackers
    row.update(
        {
            "hrv": hrv,
            "sleep_hours": sleep_hrs,
            "mood_1_5": mood,
            "soreness_1_5": soreness,
            "energy_1_5": energy,
            "notes": notes,
        }
    )

    # Save if anything non-empty
    has_content = any(
        (isinstance(v, (int, float)) and v not in [0, 0.0])
        or (isinstance(v, str) and v.strip())
        for v in row.values()
    )

    if has_content:
        save_log_for_date(selected, row)
        st.success("Auto-saved ✅")
    else:
        st.info("No data entered yet for this day. Nothing saved.")

    if st.button(f"Delete entire log for {selected}"):
        delete_log_for_date(selected)
        st.warning(f"Deleted log for {selected}.")

    render_ai_coach_panel()


# ------------------------
# WEEK VIEW
# ------------------------
def get_week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())  # Monday


def render_week_view():
    st.title("Week View")

    today = date.today()
    if "week_start" not in st.session_state:
        st.session_state["week_start"] = get_week_start(today)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if st.button("« Previous week"):
            st.session_state["week_start"] = st.session_state["week_start"] - timedelta(days=7)
    with c3:
        if st.button("Next week »"):
            st.session_state["week_start"] = st.session_state["week_start"] + timedelta(days=7)

    week_start = st.session_state["week_start"]
    st.caption(f"Week of {week_start}")

    df_log = load_log()
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    cols = st.columns(7)

    for i in range(7):
        d = week_start + timedelta(days=i)
        phase, day_type, _, kind = get_effective_plan(d)
        work_flag = is_workday(d)

        day_logs = df_log[df_log["date"] == d] if not df_log.empty else pd.DataFrame()
        has_log = not day_logs.empty

        with cols[i]:
            st.markdown(f"**{day_labels[i]}**")
            st.caption(f"{d.month}/{d.day}")
            st.write(day_type)
            if work_flag:
                st.markdown("🧳 Work")
            if has_log:
                st.markdown("✅ Logged")
            else:
                st.markdown("⭕ Not logged")
            if st.button("Open", key=f"open_{d}"):
                st.session_state["selected_date"] = d
                # Switch to Today page by updating radio via session_state trick
                st.session_state["page"] = "Today"

    render_ai_coach_panel()


# ------------------------
# PHASE OVERVIEW
# ------------------------
def render_phase_overview():
    st.title("Phase Overview")

    # Define phase ranges for display
    phases = [
        {
            "name": "Phase 2A – Strength + Aerobic Base",
            "start": date(2025, 12, 8),
            "end": date(2026, 1, 19),
            "goals": [
                "Build aerobic base with Z2 and tempo.",
                "Increase upper/lower strength without excessive fatigue.",
                "Prepare legs for upcoming ME block.",
            ],
            "weekly_pattern": [
                "Mon – Strength A (Upper)",
                "Tue – Tempo Run",
                "Wed – Hill Z2 Hike",
                "Thu – Strength B (Lower)",
                "Fri – Flex Z2 Cardio",
                "Sat – Long Z2",
                "Sun – Off / Recovery",
            ],
        },
        {
            "name": "Phase 2B – Muscular Endurance",
            "start": date(2026, 1, 20),
            "end": date(2026, 2, 23),
            "goals": [
                "Develop leg durability for hills and ME work.",
                "Maintain upper strength with reduced volume.",
                "Handle work/travel while stacking step-up volume.",
            ],
            "weekly_pattern": [
                "Mon – Z2 Incline",
                "Tue – Strength A (Upper light)",
                "Wed – ME Step-Ups + Easy Cardio",
                "Thu – Z2 Incline",
                "Fri – Strength B (Lower light)",
                "Sat – Long Incline / Hike",
                "Sun – Off / Recovery",
            ],
        },
        {
            "name": "Phase 2C – Aerobic Power / Transition",
            "start": date(2026, 2, 24),
            "end": date(2026, 3, 30),
            "goals": [
                "Bridge into more intense work without losing base.",
                "Keep legs strong but not destroyed for running.",
                "Dial in sustainable tempo pacing.",
            ],
            "weekly_pattern": [
                "Mon – Strength A (Upper)",
                "Tue – Tempo / Threshold Run",
                "Wed – Z2 Bike / Incline",
                "Thu – Strength B (Lower)",
                "Fri – Easy Z2 Cardio",
                "Sat – Long Z2 (Bike/Run combo)",
                "Sun – Off / Recovery",
            ],
        },
        {
            "name": "Phase 3A – Triathlon Base",
            "start": date(2026, 3, 31),
            "end": date(2026, 5, 4),
            "goals": [
                "Introduce swim–bike–run structure.",
                "Keep strength while volume increases.",
                "Build durable, repeatable Z2 tri training.",
            ],
            "weekly_pattern": [
                "Mon – Swim + Short Strength A",
                "Tue – Z2 Bike",
                "Wed – Z2 Run",
                "Thu – Swim + Strength B (Light)",
                "Fri – Tempo Bike / Run",
                "Sat – Long Z2 Bike / Run",
                "Sun – Off / Recovery",
            ],
        },
        {
            "name": "Phase 3B – Triathlon Build",
            "start": date(2026, 5, 5),
            "end": date(2026, 6, 22),
            "goals": [
                "Sharpen triathlon-specific fitness.",
                "Introduce more intervals and brick work.",
                "Maintain strength without compromising key sessions.",
            ],
            "weekly_pattern": [
                "Mon – Swim Intervals + Optional Strength A",
                "Tue – Bike Tempo",
                "Wed – Run Intervals",
                "Thu – Swim + Easy Strength B",
                "Fri – Brick (Bike → Run)",
                "Sat – Long Brick / Long Bike",
                "Sun – Off / Recovery",
            ],
        },
        {
            "name": "Phase 4 – Peak / Specific",
            "start": date(2026, 6, 23),
            "end": date(2026, 8, 10),
            "goals": [
                "Dial in race pace for all three legs.",
                "Practice race-like sessions and fueling.",
                "Keep fatigue under control while sharpening.",
            ],
            "weekly_pattern": [
                "Mon – Race-Pace Swim",
                "Tue – Race-Pace Bike",
                "Wed – Race-Pace Run",
                "Thu – Easy Swim + Light Strength",
                "Fri – Race Simulation Brick",
                "Sat – Long Race-Specific Session",
                "Sun – Off / Recovery",
            ],
        },
        {
            "name": "Taper & Race",
            "start": date(2026, 8, 11),
            "end": date(2026, 8, 29),
            "goals": [
                "Reduce fatigue while keeping sharpness.",
                "Practice race-day routines and pacing.",
                "Arrive fresh and mentally ready for Tahoe Triathlon.",
            ],
            "weekly_pattern": [
                "Short race-pace touches",
                "Plenty of Z2 / easy movement",
                "No new fitness, just freshness",
            ],
        },
    ]

    today = date.today()

    for ph in phases:
        is_current = ph["start"] <= today <= ph["end"]
        with st.expander(
            f"{'⭐ ' if is_current else ''}{ph['name']}  ({ph['start']} → {ph['end']})",
            expanded=is_current,
        ):
            st.markdown("**Goals for this phase:**")
            for g in ph["goals"]:
                st.markdown(f"- {g}")
            st.markdown("**Typical week structure:**")
            for w in ph["weekly_pattern"]:
                st.markdown(f"- {w}")

    render_ai_coach_panel()


# ------------------------
# HISTORY PAGE
# ------------------------
def render_history_page():
    st.title("History")

    df = load_log()
    if df.empty:
        st.info("No logs saved yet.")
        render_ai_coach_panel()
        return

    df = df.sort_values("date", ascending=False)
    st.dataframe(
        df[[
            "date",
            "phase",
            "day_type_adjusted",
            "mode",
            "workday",
            "cardio_main_mode",
            "cardio_main_duration_min",
            "cardio_extra_mode",
            "cardio_extra_duration_min",
            "hrv",
            "sleep_hours",
            "mood_1_5",
            "soreness_1_5",
            "energy_1_5",
        ]],
        height=400,
    )

    st.markdown("### Delete a specific day")
    unique_dates = sorted(df["date"].unique(), reverse=True)
    del_date = st.selectbox("Choose a date to delete", unique_dates)
    if st.button("Delete selected date"):
        delete_log_for_date(del_date)
        st.warning(f"Deleted log for {del_date}. Refresh the page to see changes.")

    render_ai_coach_panel()


# ------------------------
# WORK SCHEDULE PAGE
# ------------------------
def render_work_schedule_page():
    st.title("Work Schedule (Tours / Duty Days)")

    today = date.today()
    year = int(st.number_input("Year", min_value=2025, max_value=2030, value=today.year))
    month = int(st.number_input("Month", min_value=1, max_value=12, value=today.month))

    df_work = load_work_schedule()
    month_flags: Dict[int, bool] = {}
    num_days = calendar.monthrange(year, month)[1]

    for day in range(1, num_days + 1):
        d = date(year, month, day)
        if not df_work.empty:
            is_w = bool(df_work[df["date"] == d]["is_work"].any()) if "date" in df_work.columns else False
        else:
            is_w = False
        month_flags[day] = is_w

    st.subheader(f"{calendar.month_name[month]} {year}")
    first_weekday, _ = calendar.monthrange(year, month)
    day_counter = 1

    for week in range(6):
        cols = st.columns(7)
        for wd in range(7):
            if week == 0 and wd < first_weekday:
                cols[wd].write(" ")
            elif day_counter > num_days:
                cols[wd].write(" ")
            else:
                dnum = day_counter
                default = month_flags[dnum]
                month_flags[dnum] = cols[wd].checkbox(
                    f"{dnum}", value=default, key=f"work_{year}_{month}_{dnum}"
                )
                day_counter += 1
        if day_counter > num_days:
            break

    if st.button("Save this month’s work schedule"):
        update_work_schedule_for_month(year, month, month_flags)
        st.success("Work schedule saved ✅")

    render_ai_coach_panel()


# ------------------------
# MAIN APP
# ------------------------
def main():
    init_ai_state()

    # Sidebar navigation (hybrid style)
    page = st.sidebar.radio(
        "Navigate",
        ["Phase Overview", "Week View", "Today", "History", "Work Schedule"],
        key="page",
    )

    if page == "Phase Overview":
        render_phase_overview()
    elif page == "Week View":
        render_week_view()
    elif page == "Today":
        render_today_page()
    elif page == "History":
        render_history_page()
    elif page == "Work Schedule":
        render_work_schedule_page()


if __name__ == "__main__":
    main()
