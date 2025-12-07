
import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta
import calendar
from typing import Dict, Any, List
import ast
import xml.etree.ElementTree as ET

# Attempt to create OpenAI client using secret key
try:
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    client = None  # We'll handle gracefully in debug / coach


# ------------------------
# CONFIG / FILE PATHS
# ------------------------

LOG_FILE = "training_log.csv"
WORK_FILE = "work_schedule.csv"

CARDIO_MODES = ["Run", "Bike", "Incline Walk"]

# Strength variant lists
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
# WORK SCHEDULE
# ------------------------

def load_work_schedule() -> pd.DataFrame:
    try:
        df = pd.read_csv(WORK_FILE, parse_dates=["date"])
        df["date"] = df["date"].dt.date
    except FileNotFoundError:
        df = pd.DataFrame(columns=["date", "is_work"])
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
    """
    Parse TCX and return dict of metrics, or None if parse fails.
    """
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
    Return (phase, day_type, planned_description, kind)
    kind drives widgets: StrengthA, StrengthB, Tempo, FlexCardio, LongZ2, Incline, ME, TriBrick, TriRaceLike, Off, Manual
    """
    dt = d

    # Macro dates 2025–26
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
            return (
                phase,
                "Strength A – Upper Focus",
                "Bench, Pull-Ups, Incline Press, Row, Laterals, Triceps.\nLeave 1–3 reps in reserve.",
                "StrengthA",
            )
        if dow == 1:
            return phase, "Tempo Run", tempo_desc(), "Tempo"
        if dow == 2:
            return phase, "Z2 Hill Hike", hill_desc(), "Incline"
        if dow == 3:
            return (
                phase,
                "Strength B – Lower/Hypertrophy",
                "Deadlift/RDL, Split Squat/Lunge, Hip Thrust, Row Var, Biceps, Core.",
                "StrengthB",
            )
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
            return (
                phase,
                "Strength A – Upper (De-emphasized)",
                "Keep loads submax; upper maintenance.",
                "StrengthA",
            )
        if dow == 2:
            return phase, "ME Step-Ups + Easy Cardio", me_desc(), "ME"
        if dow == 3:
            return phase, "Z2 Incline", incline_desc(False), "Incline"
        if dow == 4:
            return (
                phase,
                "Strength B – Lower (Light)",
                "Movement quality > heavy loading.",
                "StrengthB",
            )
        if dow == 5:
            return phase, "Long Incline / Hike", incline_desc(True), "LongZ2"
        return phase, "Off / Recovery", "Off day. Lean into rest if ME buried you.", "Off"

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
            return (
                phase,
                "Swim + Short Strength A",
                "Technique swim + short upper strength.",
                "StrengthA",
            )
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
            return (
                phase,
                "RACE DAY – Tahoe Triathlon",
                "Execute, stay calm, and have fun.",
                "TriRaceLike",
            )

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

    # Don't adjust off/manual days
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


# ------------------------
# LOG LOAD/SAVE
# ------------------------

def load_log() -> pd.DataFrame:
    try:
        df = pd.read_csv(LOG_FILE, parse_dates=["date"])
        df["date"] = df["date"].dt.date
    except FileNotFoundError:
        df = pd.DataFrame()
    return df


def save_log_row(row: dict) -> None:
    df = load_log()
    new_row = pd.DataFrame([row])
    df = pd.concat([df, new_row], ignore_index=True)
    df_out = df.copy()
    if not df_out.empty:
        df_out["date"] = pd.to_datetime(df_out["date"])
    df_out.to_csv(LOG_FILE, index=False)


def get_last_strength_entry(df: pd.DataFrame, current_date: date, exercise_name: str, variant_name: str):
    if df.empty or "strength_block" not in df.columns:
        return None

    past = df[df["date"] < current_date]
    if past.empty:
        return None
    past = past.dropna(subset=["strength_block"])
    if past.empty:
        return None

    past = past.sort_values("date", ascending=False)

    for _, row in past.iterrows():
        block_str = row.get("strength_block", "")
        if not isinstance(block_str, str) or not block_str.strip():
            continue
        try:
            entries = ast.literal_eval(block_str)
        except Exception:
            continue
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
# AI COACH (MODE C – CONTEXTUAL)
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
    """Build a compact text summary of plan + recent logs + upcoming week for the AI."""
    parts = []

    # Today's plan
    phase, day_type, planned, kind = get_phase_and_day_plan(selected_date)
    work_flag = is_workday(selected_date)
    parts.append(f"Today ({selected_date}): phase={phase}, day_type={day_type}, kind={kind}, workday={work_flag}.")

    # Upcoming 7 days
    upcoming_lines = []
    for i in range(1, 8):
        d = selected_date + timedelta(days=i)
        ph, dtp, _, knd = get_phase_and_day_plan(d)
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
            cmode = row.get("cardio_mode", "")
            dur = row.get("cardio_duration_min", 0)
            hr = row.get("cardio_avg_hr", "")
            daily_lines.append(
                f"{d}: {dt_adj}, mode={mode}, workday={work}, cardio={cmode} {dur}min HR={hr}"
            )
        if daily_lines:
            parts.append("Last 7 logged days: " + " || ".join(daily_lines))

        # Recent strength summary (last ~10 entries flattened)
        strength_rows = []
        for _, row in recent.iterrows():
            sb = row.get("strength_block", "")
            if isinstance(sb, str) and sb.strip():
                try:
                    entries = ast.literal_eval(sb)
                except Exception:
                    entries = []
                if isinstance(entries, list):
                    for e in entries:
                        if not isinstance(e, dict):
                            continue
                        strength_rows.append(
                            f"{row.get('date')}: {e.get('exercise')} [{e.get('variant')}] "
                            f"{e.get('sets')}x{e.get('reps')} @ {e.get('weight')} RPE {e.get('rpe')}"
                        )
        if strength_rows:
            parts.append("Recent strength sets: " + " || ".join(strength_rows[:10]))

    return "\n".join(parts)


def ai_add_message(role: str, content: str):
    st.session_state.ai_messages.append({"role": role, "content": content})


def ai_call_coach(user_message: str) -> str:
    """Call OpenAI with full training context and return reply text (or error)."""
    if client is None:
        return "AI is not configured yet. Check that OPENAI_API_KEY is set in Streamlit secrets."

    try:
        selected_date = st.session_state.get("last_selected_date", date.today())
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

        # Chat history
        for m in st.session_state.ai_messages:
            messages.append({"role": m["role"], "content": m["content"]})

        # New user message
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
        # Show history
        for m in st.session_state.ai_messages:
            if m["role"] == "user":
                st.markdown(f"**You:** {m['content']}")
            else:
                st.markdown(f"**Coach:** {m['content']}")

        st.markdown("---")
        # Create a fresh key each render so the box clears automatically
        msg_key = f"ai_input_{len(st.session_state.ai_messages)}"

        user_q = st.text_input("Ask your coach something:", key=msg_key)
        send = st.button("Send to Coach")

        if send and user_q.strip():
            ai_add_message("user", user_q.strip())
            reply = ai_call_coach(user_q.strip())
            ai_add_message("assistant", reply)
            # No need to modify session_state — input box clears because the key changed
            

# ------------------------
# STREAMLIT APP
# ------------------------

st.set_page_config(page_title="2025–26 Training Log", layout="centered")
init_ai_state()

page = st.sidebar.radio("Page", ["Today", "Work Schedule", "History", "Debug"])


# ===== TODAY PAGE =====
if page == "Today":
    st.title("2025–26 Training Log")

    today = date.today()
    selected_date = st.date_input("Date", value=today)
    st.session_state["last_selected_date"] = selected_date

    base_phase, base_day_type, base_planned, base_kind = get_phase_and_day_plan(selected_date)
    work_flag = is_workday(selected_date)

    st.markdown("### Workout Mode")
    mode = st.selectbox(
        "Choose how to set today’s workout:",
        ["Auto (Programmed/Adjusted)", "Manual / Custom", "Rest / Skip"],
    )

    phase = base_phase
    day_type = base_day_type
    planned = base_planned
    kind = base_kind

    if mode == "Auto (Programmed/Adjusted)":
        override = st.checkbox(
            "Use original programmed workout even if this is a workday",
            value=False,
        )
        if not override:
            phase, day_type, planned, kind = adjust_for_workday(
                base_phase, base_day_type, base_planned, base_kind, work_flag
            )
    elif mode == "Manual / Custom":
        phase = "Manual"
        day_type = "Manual / Custom"
        planned = "Manual custom session. Use Strength/Cardio/Notes below to record."
        kind = "Manual"
    else:  # Rest / Skip
        phase = base_phase if base_phase != "Unprogrammed" else "Rest"
        day_type = "Rest / Skip"
        planned = "Rest or intentional skip. One missed day in a good block is noise, not disaster."
        kind = "Off"

    st.subheader(phase)
    st.caption(f"Day Type: {day_type}")
    st.write("**Planned Session Details:**")
    st.write(planned)

    if work_flag:
        st.info("This date is marked as a WORKDAY in your schedule.")

    st.markdown("---")

    # Base log dict
    log_data: Dict[str, Any] = {
        "date": selected_date,
        "phase": phase,
        "day_type": base_day_type,
        "day_type_adjusted": day_type,
        "kind": base_kind,
        "kind_adjusted": kind,
        "workday": work_flag,
        "mode": mode,
    }

    # Cardio
    tcx_data = None
    cardio_mode = ""
    cardio_duration = 0
    cardio_distance = ""
    cardio_avg_hr = ""
    cardio_rpe = 0

    if kind in ["Tempo", "FlexCardio", "LongZ2", "Incline", "ME", "TriBrick", "TriRaceLike"]:
        st.markdown("### Cardio Log")
        cardio_mode = st.selectbox("Cardio mode", CARDIO_MODES, index=0)
        cardio_duration = st.number_input("Cardio duration (min)", min_value=0, max_value=600, value=0)
        cardio_distance = st.text_input("Cardio distance (mi/km)", value="")
        cardio_avg_hr = st.text_input("Cardio avg HR (bpm)", value="")
        cardio_rpe = st.slider("Cardio RPE (1–10)", 1, 10, 6)

        st.markdown("#### Optional: Upload TCX to auto-fill")
        tcx_file = st.file_uploader("Upload TCX file", type=["tcx"])
        if tcx_file is not None:
            parsed = parse_tcx(tcx_file)
            if parsed:
                tcx_data = parsed
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

    st.markdown("---")

    # Strength
    df_log = load_log()
    exercises = get_strength_exercises(kind)
    strength_entries = []

    if exercises:
        st.markdown("### Strength / ME Session")
        for i, ex in enumerate(exercises):
            st.markdown(f"**{ex['name']}**")
            variant = st.selectbox("Variant", ex["alts"], key=f"{selected_date}_variant_{i}")

            last = get_last_strength_entry(df_log, selected_date, ex["name"], variant)
            if last:
                st.caption(
                    f"Last ({variant}) on {last.get('date')}: "
                    f"{last.get('sets', '')}×{last.get('reps', '')} @ {last.get('weight', '')} "
                    f"(RPE {last.get('rpe', '')})"
                )
            else:
                st.caption(f"Last ({variant}): no previous log yet.")

            c1, c2, c3, c4 = st.columns(4)
            sets = c1.text_input("Sets", key=f"{selected_date}_sets_{i}")
            reps = c2.text_input("Reps", key=f"{selected_date}_reps_{i}")
            weight = c3.text_input("Weight", key=f"{selected_date}_wt_{i}")
            rpe = c4.text_input("RPE", key=f"{selected_date}_rpe_{i}")

            strength_entries.append(
                {
                    "exercise": ex["name"],
                    "variant": variant,
                    "sets": sets,
                    "reps": reps,
                    "weight": weight,
                    "rpe": rpe,
                }
            )
        log_data["strength_block"] = str(strength_entries)
    else:
        log_data["strength_block"] = ""

    # Finalize cardio fields
    if kind in ["Tempo", "FlexCardio", "LongZ2", "Incline", "ME", "TriBrick", "TriRaceLike"]:
        if tcx_data:
            duration_min = int(tcx_data["duration_sec"] / 60)
            distance_mi = round(tcx_data["distance_m"] / 1609.34, 2) if tcx_data["distance_m"] > 0 else 0.0
            log_data.update(
                {
                    "cardio_mode": cardio_mode,
                    "cardio_duration_min": duration_min,
                    "cardio_distance": distance_mi,
                    "cardio_avg_hr": int(tcx_data["avg_hr"]),
                    "cardio_rpe_1_10": cardio_rpe,
                    "cardio_max_hr": int(tcx_data["max_hr"]),
                    "cardio_elev_gain_m": int(tcx_data["elevation_gain_m"]),
                    "cardio_avg_cadence": int(tcx_data["avg_cadence"]),
                    "cardio_pace_min_per_km": round(tcx_data["pace_min_per_km"], 2),
                    "cardio_hr_drift": round(tcx_data["hr_drift"], 4),
                }
            )
        else:
            log_data.update(
                {
                    "cardio_mode": cardio_mode,
                    "cardio_duration_min": cardio_duration,
                    "cardio_distance": cardio_distance,
                    "cardio_avg_hr": cardio_avg_hr,
                    "cardio_rpe_1_10": cardio_rpe,
                    "cardio_max_hr": "",
                    "cardio_elev_gain_m": "",
                    "cardio_avg_cadence": "",
                    "cardio_pace_min_per_km": "",
                    "cardio_hr_drift": "",
                }
            )
    else:
        log_data.update(
            {
                "cardio_mode": "",
                "cardio_duration_min": 0,
                "cardio_distance": "",
                "cardio_avg_hr": "",
                "cardio_rpe_1_10": 0,
                "cardio_max_hr": "",
                "cardio_elev_gain_m": "",
                "cardio_avg_cadence": "",
                "cardio_pace_min_per_km": "",
                "cardio_hr_drift": "",
            }
        )

    st.markdown("---")

    st.markdown("### Daily Trackers")
    cA, cB, cC = st.columns(3)
    hrv = cA.text_input("HRV", value="")
    sleep_hrs = cB.text_input("Sleep hours", value="")
    mood = cC.slider("Mood (1–5)", 1, 5, 3)

    cD, cE, _ = st.columns(3)
    soreness = cD.slider("Soreness (1–5)", 1, 5, 3)
    energy = cE.slider("Energy (1–5)", 1, 5, 3)

    notes = st.text_area("Notes", height=120)

    log_data.update(
        {
            "hrv": hrv,
            "sleep_hours": sleep_hrs,
            "mood_1_5": mood,
            "soreness_1_5": soreness,
            "energy_1_5": energy,
            "notes": notes,
        }
    )

    if st.button("Save today’s log"):
        save_log_row(log_data)
        st.success("Logged ✅")

    # AI Coach panel
    render_ai_coach_panel()


# ===== WORK SCHEDULE PAGE =====
elif page == "Work Schedule":
    st.title("Work Schedule (Tours/Duty Days)")

    today = date.today()
    year = int(st.number_input("Year", min_value=2025, max_value=2030, value=today.year))
    month = int(st.number_input("Month", min_value=1, max_value=12, value=today.month))

    df_work = load_work_schedule()
    month_flags: Dict[int, bool] = {}
    monthrange = calendar.monthrange(year, month)[1]

    for day in range(1, monthrange + 1):
        d = date(year, month, day)
        if not df_work.empty:
            is_w = bool(df_work[df_work["date"] == d]["is_work"].any())
        else:
            is_w = False
        month_flags[day] = is_w

    st.subheader(f"{calendar.month_name[month]} {year}")
    first_weekday, num_days = calendar.monthrange(year, month)
    days_iter = 1

    for _week in range(6):
        cols = st.columns(7)
        for wd in range(7):
            if _week == 0 and wd < first_weekday:
                cols[wd].write(" ")
            elif days_iter > num_days:
                cols[wd].write(" ")
            else:
                dnum = days_iter
                default = month_flags[dnum]
                month_flags[dnum] = cols[wd].checkbox(
                    f"{dnum}",
                    value=default,
                    key=f"work_{year}_{month}_{dnum}",
                )
                days_iter += 1
        if days_iter > num_days:
            break

    if st.button("Save this month’s work schedule"):
        update_work_schedule_for_month(year, month, month_flags)
        st.success("Work schedule saved ✅")

    render_ai_coach_panel()


# ===== HISTORY PAGE =====
elif page == "History":
    st.title("Training History")

    df = load_log()
    if df.empty:
        st.info("No logs saved yet.")
    else:
        df = df.sort_values("date", ascending=False)

        st.dataframe(
            df[
                [
                    "date",
                    "phase",
                    "day_type_adjusted",
                    "mode",
                    "workday",
                    "cardio_mode",
                    "cardio_duration_min",
                    "hrv",
                    "sleep_hours",
                    "mood_1_5",
                    "soreness_1_5",
                    "energy_1_5",
                ]
            ],
            height=400,
        )

        st.markdown("### Export for analysis")
        st.caption("Exports a tall CSV with daily, cardio, and strength rows.")

        records: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            rdate = row.get("date")
            records.append(
                {
                    "record_type": "daily",
                    "date": rdate,
                    "phase": row.get("phase"),
                    "day_type": row.get("day_type"),
                    "day_type_adjusted": row.get("day_type_adjusted"),
                    "mode": row.get("mode"),
                    "workday": row.get("workday"),
                    "hrv": row.get("hrv"),
                    "sleep_hours": row.get("sleep_hours"),
                    "mood_1_5": row.get("mood_1_5"),
                    "soreness_1_5": row.get("soreness_1_5"),
                    "energy_1_5": row.get("energy_1_5"),
                    "notes": row.get("notes"),
                }
            )

            cardio_mode = row.get("cardio_mode", "")
            cardio_dur = row.get("cardio_duration_min", 0)
            cardio_dist = row.get("cardio_distance", "")
            cardio_hr = row.get("cardio_avg_hr", "")
            cardio_rpe = row.get("cardio_rpe_1_10", 0)
            cardio_max_hr = row.get("cardio_max_hr", "")
            cardio_elev = row.get("cardio_elev_gain_m", "")
            cardio_cad = row.get("cardio_avg_cadence", "")
            cardio_pace = row.get("cardio_pace_min_per_km", "")
            cardio_drift = row.get("cardio_hr_drift", "")

            if (
                (isinstance(cardio_mode, str) and cardio_mode.strip())
                or (isinstance(cardio_dur, (int, float)) and cardio_dur > 0)
                or (isinstance(cardio_dist, str) and cardio_dist.strip())
            ):
                records.append(
                    {
                        "record_type": "cardio",
                        "date": rdate,
                        "phase": row.get("phase"),
                        "day_type": row.get("day_type"),
                        "day_type_adjusted": row.get("day_type_adjusted"),
                        "mode": row.get("mode"),
                        "workday": row.get("workday"),
                        "cardio_mode": cardio_mode,
                        "duration_min": cardio_dur,
                        "distance": cardio_dist,
                        "avg_hr": cardio_hr,
                        "cardio_rpe": cardio_rpe,
                        "max_hr": cardio_max_hr,
                        "elev_gain_m": cardio_elev,
                        "avg_cadence": cardio_cad,
                        "pace_min_per_km": cardio_pace,
                        "hr_drift": cardio_drift,
                        "notes": row.get("notes"),
                    }
                )

            strength_block = row.get("strength_block", "")
            if isinstance(strength_block, str) and strength_block.strip():
                try:
                    entries = ast.literal_eval(strength_block)
                except Exception:
                    entries = []
                if isinstance(entries, list):
                    for e in entries:
                        if not isinstance(e, dict):
                            continue
                        records.append(
                            {
                                "record_type": "strength",
                                "date": rdate,
                                "phase": row.get("phase"),
                                "day_type": row.get("day_type"),
                                "day_type_adjusted": row.get("day_type_adjusted"),
                                "mode": row.get("mode"),
                                "workday": row.get("workday"),
                                "exercise": e.get("exercise"),
                                "variant": e.get("variant") or e.get("alt"),
                                "sets": e.get("sets"),
                                "reps": e.get("reps"),
                                "weight": e.get("weight"),
                                "rpe": e.get("rpe"),
                                "notes": row.get("notes"),
                            }
                        )

        if records:
            df_export = pd.DataFrame(records)
            csv_bytes = df_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download analytics_export.csv",
                data=csv_bytes,
                file_name="analytics_export.csv",
                mime="text/csv",
            )
        else:
            st.info("No data available to export yet.")

    render_ai_coach_panel()


# ===== DEBUG PAGE =====
else:  # page == "Debug"
    st.title("Debug / Diagnostics")

    st.subheader("Session State")
    st.json(st.session_state)

    st.subheader("Test OpenAI API")
    if client is None:
        st.error("OpenAI client is not initialised. Check OPENAI_API_KEY in secrets.")
    else:
        if st.button("Run quick API test"):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": "Say hi in one short sentence."}],
                    max_tokens=20,
                )
                st.success("API call succeeded:")
                st.write(resp.choices[0].message.content)
            except Exception as e:
                st.error("OpenAI API error:")
                st.error(str(e))

    render_ai_coach_panel()
