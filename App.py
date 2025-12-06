import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from datetime import date, datetime, timedelta
import calendar
from typing import Tuple, List, Dict, Any
import ast
import xml.etree.ElementTree as ET
from openai import OpenAI

# ------------------------
# CONFIG & FILE PATHS
# ------------------------

LOG_FILE = "training_log.csv"
WORK_FILE = "work_schedule.csv"

CARDIO_MODES = ["Run", "Bike", "Incline Walk"]

# Alt lists (with "Primary" as first option)
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

# OpenAI client (expects OPENAI_API_KEY in Streamlit Secrets)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# ------------------------
# AI COACH CHAT LOGIC
# ------------------------

def init_chat_state():
    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = [
            {
                "role": "assistant",
                "content": "Hey, I'm your AI coach. Ask me about today's workout, adjustments for soreness, or how to interpret your cardio data."
            }
        ]


def add_ai_message(role: str, content: str):
    st.session_state.ai_messages.append({"role": role, "content": content})


def call_ai_coach(user_message: str) -> str:
    """
    Sends chat history + the new user_message to OpenAI and returns the assistant reply.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise, practical training coach. "
                "You help the user interpret their workouts, adjust training for soreness, "
                "and make decisions about runs, lifting, and recovery. "
                "The user is training for the Tahoe Triathlon on August 29, 2026, cares about GBRS standards, "
                "triathlon, hypertrophy, and uphill hiking. "
                "Answer in short, clear paragraphs with specific recommendations."
            ),
        }
    ]

    for m in st.session_state.ai_messages:
        messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": user_message})

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            max_tokens=400,
            temperature=0.5,
        )
        reply = resp.choices[0].message.content
        return reply
    except Exception as e:
        return f"Error talking to AI coach: {e}"


# ------------------------
# WORK SCHEDULE DATA
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
    df_out["date"] = pd.to_datetime(df_out["date"])
    df_out.to_csv(WORK_FILE, index=False)


def is_workday(d: date) -> bool:
    df = load_work_schedule()
    if df.empty:
        return False
    return bool(df[df["date"] == d]["is_work"].any())


def update_work_schedule_for_month(year: int, month: int, selections: Dict[int, bool]) -> None:
    df = load_work_schedule()
    mask_same_month = df["date"].apply(lambda x: x.year == year and x.month == month)
    df = df[~mask_same_month]

    new_rows = [{"date": date(year, month, day), "is_work": flag} for day, flag in selections.items()]
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    save_work_schedule(df)


# ------------------------
# TCX PARSING
# ------------------------

def parse_tcx(file_obj):
    """
    Parses a TCX file and returns a dict with:
    duration_sec, distance_m, avg_hr, max_hr, avg_cadence, elevation_gain_m,
    pace_min_per_km, hr_drift
    """
    tree = ET.parse(file_obj)
    root = tree.getroot()

    ns = {"tcx": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}
    tps = root.findall(".//tcx:Trackpoint", ns)

    times, hrs, cadence, altitude, distance = [], [], [], [], []

    for tp in tps:
        t = tp.find("tcx:Time", ns)
        hr = tp.find(".//tcx:HeartRateBpm/tcx:Value", ns)
        cad = tp.find(".//tcx:Cadence", ns)
        alt = tp.find(".//tcx:AltitudeMeters", ns)
        dist = tp.find(".//tcx:DistanceMeters", ns)

        if t is not None:
            try:
                times.append(datetime.fromisoformat(t.text.replace("Z", "+00:00")))
            except Exception:
                pass
        if hr is not None:
            try:
                hrs.append(float(hr.text))
            except Exception:
                pass
        if cad is not None:
            try:
                cadence.append(float(cad.text))
            except Exception:
                pass
        if alt is not None:
            try:
                altitude.append(float(alt.text))
            except Exception:
                pass
        if dist is not None:
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

def get_phase_and_day_plan(d: date) -> Tuple[str, str, str, str]:
    """
    Returns (phase, day_type, planned, kind)
    kind drives UI: StrengthA, StrengthB, Tempo, Hill, FlexCardio,
    LongZ2, Incline, ME, TriBrick, TriRaceLike, Off, Manual
    """
    dt = d

    # Macro dates for 2025–26
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
            "No structured plan for this date yet. Use this as a free day or enter a manual workout.",
            "Manual",
        )

    dow = dt.weekday()  # Mon=0

    # Helper descriptions
    def tempo_run_desc():
        return (
            "Tempo Run — 30–35 minutes\n\n"
            "• Warm-up: 5–10 min easy Z1–low Z2 jog or brisk walk.\n"
            "• Main Set: 2–3 × 6–8 min at controlled tempo (RPE 6–7), "
            "with 2–3 min easy Z1–Z2 between reps.\n"
            "• Cool-down: 5–10 min easy.\n\n"
            "Goal: Build aerobic power and running economy.\n"
            "Coaching Tip: Relax shoulders/jaw and keep cadence smooth."
        )

    def long_z2_desc(mode: str = "Bike/Run"):
        return (
            f"Long Z2 {mode} — 75–90 minutes\n\n"
            "• Warm-up: 10 min easy Z1.\n"
            "• Main: 60–70 min steady Z2 (RPE 4–6).\n"
            "• Cool-down: 5–10 min easy.\n\n"
            "Goal: Build base engine and fat-burning capacity.\n"
            "Coaching Tip: If it feels exciting, you're probably going too hard."
        )

    def flex_z2_desc():
        return (
            "Flexible Z2 Cardio — 30–45 minutes\n\n"
            "• Choose: Run, Bike, or Incline Walk.\n"
            "• Stay in easy Z2 (RPE 3–5).\n\n"
            "Goal: Maintain aerobic volume without digging a hole.\n"
            "Coaching Tip: Let fatigue and joints decide the modality."
        )

    def incline_desc(long: bool = False):
        if long:
            return (
                "Long Incline / Hike — 60–90 minutes\n\n"
                "• Continuous uphill walking/hiking in Z2.\n"
                "Goal: Build climbing-specific strength.\n"
                "Coaching Tip: Shorten stride and lean slightly from ankles."
            )
        return (
            "Incline Z2 — 40–60 minutes\n\n"
            "• Steady incline treadmill or outdoor uphill walking in Z2.\n"
            "Goal: Low-impact aerobic work for uphill strength.\n"
            "Coaching Tip: Use incline, not speed, to set difficulty."
        )

    def hill_hike_desc():
        return (
            "Hill Hike — 45–75 minutes\n\n"
            "• Continuous or looped hiking on hills/stairs in Z2.\n"
            "Goal: Leg strength/durability for climbs.\n"
            "Coaching Tip: Grind, don't sprint; keep breathing steady."
        )

    def me_desc():
        return (
            "Muscular Endurance Step-Ups + Easy Cardio\n\n"
            "• 3 rounds:\n"
            "  – 6 min continuous step-ups with pack.\n"
            "  – 2 min easy.\n"
            "• Then: 20–25 min easy Z1–Z2 cardio.\n\n"
            "Goal: Local muscular endurance for climbs.\n"
            "Coaching Tip: Smooth, controlled steps; legs burn, breathing steady."
        )

    def brick_desc(long: bool = False):
        if long:
            return (
                "Long Brick — Bike → Run\n\n"
                "• Bike: 60–90 min Z2 with race-like segments.\n"
                "• Transition: 5–10 min.\n"
                "• Run: 15–25 min Z2–low Z3.\n\n"
                "Goal: Run well off the bike and practice fueling.\n"
                "Coaching Tip: First 5 min of run should feel too easy."
            )
        return (
            "Brick — Bike → Run\n\n"
            "• Bike: 30–45 min Z2–Z3-.\n"
            "• Transition.\n"
            "• Run: 10–15 min easy-steady.\n\n"
            "Goal: Groove the feel of running off the bike.\n"
            "Coaching Tip: Focus on cadence and relaxed upper body."
        )

    def race_like_desc(modality: str):
        return (
            f"Race-Pace {modality}\n\n"
            "• Short intervals at target race effort with plenty of easy recovery.\n\n"
            "Goal: Lock in race-pace feel and movement economy.\n"
            "Coaching Tip: If you couldn't hold it on race day, back it off."
        )

    # Phase 2A – Strength + Aerobic Base
    if start_2a <= dt <= end_2a:
        phase = "Phase 2A – Strength + Aerobic Base"
        if dow == 0:
            planned = (
                "Strength A – Upper Focus\n\n"
                "• Bench Press, Pull-Ups, Incline Press, Row, Lateral Raises, Triceps.\n"
                "Goal: Build upper-body strength and muscle.\n"
                "Coaching Tip: Leave 1–3 reps in the tank on main lifts."
            )
            return phase, "Strength A – Upper Focus", planned, "StrengthA"
        if dow == 1:
            return phase, "Aerobic Tempo Run", tempo_run_desc(), "Tempo"
        if dow == 2:
            return phase, "Z2 Hill Hike", hill_hike_desc(), "Hill"
        if dow == 3:
            planned = (
                "Strength B – Lower / Hypertrophy\n\n"
                "• Deadlift/RDL, Split Squat/Lunge, Hip Thrust, Row Variation, Biceps, Core.\n"
                "Goal: Lower-body strength and muscle.\n"
                "Coaching Tip: Control the eccentric; no ugly grinders."
            )
            return phase, "Strength B – Lower/Hypertrophy", planned, "StrengthB"
        if dow == 4:
            return phase, "Aerobic Flex Day", flex_z2_desc(), "FlexCardio"
        if dow == 5:
            return phase, "Long Z2 Endurance", long_z2_desc("Bike/Run/Hike"), "LongZ2"
        planned = (
            "Off / Recovery\n\n"
            "Goal: Absorb the week's work.\n"
            "Coaching Tip: If unsure whether to train, rest."
        )
        return phase, "Off / Recovery", planned, "Off"

    # Phase 2B – ME
    if start_2b <= dt <= end_2b:
        phase = "Phase 2B – Muscular Endurance"
        if dow == 0:
            return phase, "Z2 Incline", incline_desc(False), "Incline"
        if dow == 1:
            planned = (
                "Strength A – Upper (De-emphasized)\n\n"
                "Goal: Maintain upper strength while ME is priority.\n"
                "Coaching Tip: 2–3 reps in reserve; don't crush yourself."
            )
            return phase, "Strength A – Upper Focus", planned, "StrengthA"
        if dow == 2:
            return phase, "ME Step-Ups + Easy Cardio", me_desc(), "ME"
        if dow == 3:
            return phase, "Z2 Incline", incline_desc(False), "Incline"
        if dow == 4:
            planned = (
                "Strength B – Lower (Light)\n\n"
                "Goal: Support ME without heavy fatigue.\n"
                "Coaching Tip: Treat as movement practice."
            )
            return phase, "Strength B – Lower (Light)", planned, "StrengthB"
        if dow == 5:
            return phase, "Long Incline / Hike", incline_desc(True), "LongZ2"
        planned = "Off / Recovery\n\nCoaching Tip: Lean into rest if ME days bury you."
        return phase, "Off / Recovery", planned, "Off"

    # Phase 2C – Transition
    if start_2c <= dt <= end_2c:
        phase = "Phase 2C – Aerobic Power / Transition"
        if dow == 0:
            planned = (
                "Strength A – Upper\n\n"
                "Goal: Keep building upper strength as aerobic work sharpens.\n"
                "Coaching Tip: No grinders; smooth bar speed."
            )
            return phase, "Strength A – Upper", planned, "StrengthA"
        if dow == 1:
            return phase, "Tempo / Threshold Run", tempo_run_desc(), "Tempo"
        if dow == 2:
            return phase, "Z2 Bike / Incline", flex_z2_desc(), "FlexCardio"
        if dow == 3:
            planned = (
                "Strength B – Lower\n\n"
                "Goal: Maintain leg strength as intensity rises.\n"
                "Coaching Tip: Trim sets if legs are cooked."
            )
            return phase, "Strength B – Lower", planned, "StrengthB"
        if dow == 4:
            return phase, "Easy Z2 Cardio", flex_z2_desc(), "FlexCardio"
        if dow == 5:
            return phase, "Long Z2 Endurance", long_z2_desc("Bike/Run combo"), "LongZ2"
        planned = "Off / Recovery\n\nCoaching Tip: De-stress life today too."
        return phase, "Off / Recovery", planned, "Off"

    # Phase 3A – Triathlon Base
    if start_3a <= dt <= end_3a:
        phase = "Phase 3A – Triathlon Base"
        if dow == 0:
            planned = (
                "Swim (Technique) + Short Strength A\n\n"
                "Goal: Build swim comfort while keeping upper strength.\n"
                "Coaching Tip: Smooth water feel > effort."
            )
            return phase, "Swim + Strength A", planned, "StrengthA"
        if dow == 1:
            return phase, "Z2 Bike", flex_z2_desc(), "FlexCardio"
        if dow == 2:
            return phase, "Z2 Run", flex_z2_desc(), "FlexCardio"
        if dow == 3:
            planned = (
                "Swim + Strength B (Light)\n\n"
                "Goal: Maintain strength without blocking tri volume."
            )
            return phase, "Swim + Strength B", planned, "StrengthB"
        if dow == 4:
            return phase, "Tempo Bike / Run", tempo_run_desc(), "Tempo"
        if dow == 5:
            return phase, "Long Z2 Bike / Run", long_z2_desc("Bike/Run"), "LongZ2"
        planned = "Off / Recovery\n\nCoaching Tip: Sleep and fueling matter most now."
        return phase, "Off / Recovery", planned, "Off"

    # Phase 3B – Tri Build
    if start_3b <= dt <= end_3b:
        phase = "Phase 3B – Triathlon Build"
        if dow == 0:
            planned = (
                "Swim Intervals + Optional Short Strength A\n\n"
                "Coaching Tip: Drop strength first on heavy weeks."
            )
            return phase, "Swim Intervals", planned, "StrengthA"
        if dow == 1:
            planned = (
                "Bike Tempo — 45–75 min\n\n"
                "Goal: Raise sustained power.\n"
                "Coaching Tip: 85–95 rpm, stable torso."
            )
            return phase, "Bike Tempo", planned, "Tempo"
        if dow == 2:
            planned = (
                "Run Intervals — 30–45 min\n\n"
                "Goal: Raise running speed and resilience.\n"
                "Coaching Tip: First few reps should feel too easy."
            )
            return phase, "Run Intervals", planned, "Tempo"
        if dow == 3:
            planned = (
                "Swim + Easy Strength B\n\n"
                "Goal: Maintain strength with minimal leg fatigue."
            )
            return phase, "Swim + Strength B", planned, "StrengthB"
        if dow == 4:
            return phase, "Brick (Bike → Run)", brick_desc(False), "TriBrick"
        if dow == 5:
            return phase, "Long Bike / Long Brick", brick_desc(True), "TriBrick"
        planned = "Off / Recovery\n\nCoaching Tip: Rest is where adaptation happens."
        return phase, "Off / Recovery", planned, "Off"

    # Phase 4 – Peak
    if start_4 <= dt <= end_4:
        phase = "Phase 4 – Peak / Specific Prep"
        if dow == 0:
            return phase, "Race-Pace Swim", race_like_desc("Swim"), "TriRaceLike"
        if dow == 1:
            return phase, "Race-Pace Bike", race_like_desc("Bike"), "TriRaceLike"
        if dow == 2:
            return phase, "Race-Pace Run", race_like_desc("Run"), "TriRaceLike"
        if dow == 3:
            planned = (
                "Easy Swim + Very Light Strength\n\n"
                "Goal: Movement quality, not fatigue."
            )
            return phase, "Easy Swim + Light Strength", planned, "FlexCardio"
        if dow == 4:
            return phase, "Race Simulation Brick", brick_desc(False), "TriBrick"
        if dow == 5:
            return phase, "Long Race-Specific Session", brick_desc(True), "TriBrick"
        planned = "Off / Recovery\n\nCoaching Tip: Protect sleep like it's training."
        return phase, "Off / Recovery", planned, "Off"

    # Taper
    if start_taper <= dt <= race_day:
        phase = "Taper"
        if dt == race_day:
            planned = (
                "RACE DAY – Tahoe Triathlon\n\n"
                "Goal: Execute, stay calm, and have fun.\n"
                "Coaching Tip: Trust the work; no last-minute heroics."
            )
            return phase, "RACE DAY – Tahoe Triathlon", planned, "TriRaceLike"

        if dow in [0, 2]:
            return phase, "Short Race-Pace Sharpening", race_like_desc("Swim/Bike/Run"), "TriRaceLike"
        if dow in [1, 3]:
            planned = (
                "Easy Z2 Cardio — 30–45 min\n\n"
                "Goal: Keep the engine idling.\n"
                "Coaching Tip: You should finish feeling better than you started."
            )
            return phase, "Easy Z2 Cardio", planned, "FlexCardio"
        if dow == 4:
            return phase, "Very Short Brick", brick_desc(False), "TriBrick"
        if dow == 5:
            planned = (
                "Optional Easy Session\n\n"
                "Coaching Tip: If slightly run down, skip and rest."
            )
            return phase, "Optional Easy Session", planned, "FlexCardio"
        planned = (
            "Off / Pre-Race Rest\n\n"
            "Coaching Tip: Hydrate, eat normally, visualize smooth execution."
        )
        return phase, "Off / Pre-Race Rest", planned, "Off"

    return "Unprogrammed", "Manual / Free Day", "No structured plan.", "Manual"


def adjust_for_workday(phase: str, day_type: str, planned: str, kind: str, work: bool) -> Tuple[str, str, str, str, bool]:
    if not work:
        return phase, day_type, planned, kind, False

    if kind in ["Off", "Manual"]:
        return phase, day_type, planned, kind, False

    if kind in ["LongZ2", "ME", "TriBrick", "TriRaceLike"]:
        new_day_type = f"{day_type} (Adjusted for Workday)"
        new_planned = (
            "Travel/work day adjustment:\n\n"
            "• Swap to 25–40 min easy Z1–Z2 cardio (Run/Bike/Incline).\n"
            "• Goal is circulation and headspace, not fitness PRs.\n\n"
            "Coaching Tip: On long duty days, 'did something easy' is a win."
        )
        return phase, new_day_type, new_planned, "FlexCardio", True

    if kind in ["StrengthA", "StrengthB"]:
        new_day_type = f"{day_type} (Workday – lighter loads)"
        new_planned = planned + "\n\nWorkday Adjustment: Keep 2–4 reps in reserve and cut one set if drained."
        return phase, new_day_type, new_planned, kind, True

    if kind == "Tempo":
        new_day_type = f"{day_type} (Workday – controlled)"
        new_planned = (
            "Tempo (Workday version)\n\n"
            "• Option 1: 1–2 short tempo reps instead of full workout.\n"
            "• Option 2: Convert to 30–40 min pure Z2 if very fatigued.\n\n"
            "Coaching Tip: Travel and sleep debt make tempo feel harder."
        )
        return phase, new_day_type, new_planned, kind, True

    return phase, day_type, planned, kind, False


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
# LOG DATA
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
        except (ValueError, SyntaxError):
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
# STREAMLIT APP
# ------------------------

st.set_page_config(page_title="2025–26 Training Log", layout="centered")
init_chat_state()

page = st.sidebar.radio("Page", ["Today", "Work Schedule", "History"])

# ------------- TODAY PAGE -------------

if page == "Today":
    st.title("2025–26 Training Log")

    today = date.today()
    selected_date = st.date_input("Date", value=today)

    work_flag = is_workday(selected_date)
    base_phase, base_day_type, base_planned, base_kind = get_phase_and_day_plan(selected_date)

    st.markdown("### Workout Mode")
    workout_mode = st.selectbox(
        "Use which workout today?",
        ["Auto (Programmed / Adjusted)", "Choose from This Week", "Manual / Custom", "Rest / Skip"],
    )

    effective_phase = base_phase
    effective_day_type = base_day_type
    effective_planned = base_planned
    effective_kind = base_kind
    adjusted = False
    source_date = selected_date

    week_monday = selected_date - timedelta(days=selected_date.weekday())
    week_days = [week_monday + timedelta(days=i) for i in range(7)]

    if workout_mode == "Auto (Programmed / Adjusted)":
        override = st.checkbox(
            "Override workday adjustment (use original programmed workout)",
            value=False,
        )
        if override:
            adjusted = False
        else:
            effective_phase, effective_day_type, effective_planned, effective_kind, adjusted = adjust_for_workday(
                base_phase, base_day_type, base_planned, base_kind, work_flag
            )

    elif workout_mode == "Choose from This Week":
        options = []
        for d in week_days:
            ph, dtp, pln, knd = get_phase_and_day_plan(d)
            label = f"{d.strftime('%a %m-%d')} – {dtp} ({ph})"
            options.append({"date": d, "label": label, "phase": ph, "day_type": dtp, "planned": pln, "kind": knd})
        labels = [o["label"] for o in options]
        choice = st.selectbox("Pick a workout from this week:", labels)
        chosen = next(o for o in options if o["label"] == choice)

        effective_phase = chosen["phase"]
        effective_day_type = chosen["day_type"]
        effective_planned = chosen["planned"]
        effective_kind = chosen["kind"]
        source_date = chosen["date"]
        adjusted = False

    elif workout_mode == "Manual / Custom":
        effective_phase = "Manual"
        effective_day_type = "Manual / Custom"
        effective_planned = "Manual custom session. Use Strength / Cardio sections and Notes to describe what you did."
        effective_kind = "Manual"
        adjusted = False

    elif workout_mode == "Rest / Skip":
        effective_phase = base_phase if base_phase != "Unprogrammed" else "Rest"
        effective_day_type = "Rest / Skip"
        effective_planned = (
            "Rest / Skip Day\n\nUse when you intentionally skip training (travel, fatigue, life). "
            "One skipped day in a well-run block is noise, not disaster."
        )
        effective_kind = "Off"
        adjusted = False

    st.subheader(effective_phase)
    if workout_mode == "Auto (Programmed / Adjusted)":
        if adjusted:
            st.caption(f"Original: {base_day_type}")
            st.markdown(f"**Adjusted for Workday:** {effective_day_type}")
        else:
            st.caption(f"Day Type: {effective_day_type}")
    elif workout_mode == "Choose from This Week":
        st.caption(f"Using: {effective_day_type} (from {source_date.strftime('%a %m-%d')})")
    else:
        st.caption(f"Day Type: {effective_day_type}")

    st.write(f"**Planned Session Details:**\n\n{effective_planned}")

    if work_flag and workout_mode == "Auto (Programmed / Adjusted)":
        st.info("This date is marked as a WORKDAY in your schedule.")
    elif work_flag:
        st.info("This is a WORKDAY, but you've manually chosen today's workout.")

    st.markdown("---")

    log_data: Dict[str, Any] = {
        "date": selected_date,
        "phase": effective_phase,
        "day_type": base_day_type,
        "day_type_adjusted": effective_day_type,
        "kind": base_kind,
        "kind_adjusted": effective_kind,
        "workday": work_flag,
        "mode": workout_mode,
        "template_source_date": source_date,
    }

    # Cardio log
    tcx_data = None
    cardio_mode = ""
    cardio_duration = 0
    cardio_distance = ""
    cardio_avg_hr = ""
    cardio_rpe = 0

    if effective_kind in ["Tempo", "Hill", "FlexCardio", "LongZ2", "Incline", "ME", "TriBrick", "TriRaceLike"]:
        st.markdown("### Cardio Log")
        cardio_mode = st.selectbox("Cardio mode", CARDIO_MODES, index=0)
        cardio_duration = st.number_input("Cardio duration (min)", min_value=0, max_value=600, value=0)
        cardio_distance = st.text_input("Cardio distance (mi/km)", value="")
        cardio_avg_hr = st.text_input("Cardio avg HR (bpm)", value="")
        cardio_rpe = st.slider("Cardio RPE (1–10)", 1, 10, 6)

        st.markdown("#### Upload TCX to auto-fill (optional)")
        st.caption("If you upload a TCX file, its data will override manual cardio fields when you save.")
        tcx_file = st.file_uploader("Upload TCX file", type=["tcx"], key=f"tcx_{selected_date}")
        if tcx_file is not None:
            parsed = parse_tcx(tcx_file)
            if parsed:
                tcx_data = parsed
                duration_min = int(parsed["duration_sec"] / 60)
                distance_mi = round(parsed["distance_m"] / 1609.34, 2) if parsed["distance_m"] > 0 else 0.0
                st.success("TCX file parsed successfully!")
                st.info(
                    f"Auto metrics from TCX:\n"
                    f"- Duration: {duration_min} min\n"
                    f"- Distance: {distance_mi} mi\n"
                    f"- Avg HR: {round(parsed['avg_hr'])}\n"
                    f"- Max HR: {round(parsed['max_hr'])}\n"
                    f"- Elevation Gain: {round(parsed['elevation_gain_m'])} m\n"
                    f"- Avg Cadence: {round(parsed['avg_cadence'])}\n"
                    f"- Pace: {round(parsed['pace_min_per_km'], 2)} min/km\n"
                    f"- HR Drift: {round(parsed['hr_drift'] * 100, 1)} %"
                )
            else:
                st.error("Could not parse TCX file. Check the file and try again.")

    st.markdown("---")

    # Strength / ME block
    df_log = load_log()
    exercises = get_strength_exercises(effective_kind)
    strength_entries = []

    if exercises:
        st.markdown("### Strength / ME Session")
        for i, ex in enumerate(exercises):
            st.markdown(f"**{ex['name']}**")
            variant = st.selectbox("Variant", ex["alts"], key=f"{selected_date}_variant_{i}")

            last = get_last_strength_entry(df_log, selected_date, ex["name"], variant)
            if last:
                last_date = last.get("date")
                st.caption(
                    f"Last ({variant}) — {last_date}: {last.get('sets','')}×"
                    f"{last.get('reps','')} @ {last.get('weight','')} (RPE {last.get('rpe','')})"
                )
            else:
                st.caption(f"Last ({variant}) — no previous log for this variant yet.")

            col1, col2, col3, col4 = st.columns(4)
            sets = col1.text_input("Sets", key=f"{selected_date}_sets_{i}")
            reps = col2.text_input("Reps", key=f"{selected_date}_reps_{i}")
            weight = col3.text_input("Weight", key=f"{selected_date}_wt_{i}")
            rpe = col4.text_input("RPE", key=f"{selected_date}_rpe_{i}")

            strength_entries.append(
                {"exercise": ex["name"], "variant": variant, "sets": sets, "reps": reps, "weight": weight, "rpe": rpe}
            )
        log_data["strength_block"] = str(strength_entries)
    else:
        log_data["strength_block"] = ""

    # Finalize cardio fields
    if effective_kind in ["Tempo", "Hill", "FlexCardio", "LongZ2", "Incline", "ME", "TriBrick", "TriRaceLike"]:
        if tcx_data:
            duration_min = int(tcx_data["duration_sec"] / 60)
            distance_mi = round(tcx_data["distance_m"] / 1609.34, 2) if tcx_data["distance_m"] > 0 else 0.0
            log_data.update(
                {
                    "cardio_mode": cardio_mode,
                    "cardio_duration_min": duration_min,
                    "cardio_distance": distance_mi,
                    "cardio_avg_hr": round(tcx_data["avg_hr"]),
                    "cardio_rpe_1_10": cardio_rpe,
                    "cardio_max_hr": round(tcx_data["max_hr"]),
                    "cardio_elev_gain_m": round(tcx_data["elevation_gain_m"]),
                    "cardio_avg_cadence": round(tcx_data["avg_cadence"]),
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
    colA, colB, colC = st.columns(3)
    hrv = colA.text_input("HRV", value="")
    sleep_hrs = colB.text_input("Sleep hours", value="")
    mood = colC.slider("Mood (1–5)", 1, 5, 3)

    colD, colE, _ = st.columns(3)
    soreness = colD.slider("Soreness (1–5)", 1, 5, 3)
    energy = colE.slider("Energy (1–5)", 1, 5, 3)

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

    st.markdown("---")

    if st.button("Save today’s log"):
        save_log_row(log_data)
        st.success("Logged ✅")


# ------------- WORK SCHEDULE PAGE -------------

elif page == "Work Schedule":
    st.title("Work Schedule")
    st.write("Tap the days you are **working (on tour/duty)**. Auto mode will adjust those days.")

    today = date.today()
    year = int(st.number_input("Year", min_value=2025, max_value=2030, value=today.year))
    month = int(st.number_input("Month", min_value=1, max_value=12, value=today.month))

    df_work = load_work_schedule()
    month_flags: Dict[int, bool] = {}
    monthrange = calendar.monthrange(year, month)[1]

    for day in range(1, monthrange + 1):
        d = date(year, month, day)
        if not df_work.empty and "date" in df_work.columns:
            is_w = bool(df_work[df_work["date"] == d]["is_work"].any())
        else:
            is_w = False
        month_flags[day] = is_w

    st.subheader(f"{calendar.month_name[month]} {year}")

    first_weekday, num_days = calendar.monthrange(year, month)
    days_iter = 1
    for _ in range(6):
        cols = st.columns(7)
        for wd in range(7):
            if _ == 0 and wd < first_weekday:
                cols[wd].write(" ")
            elif days_iter > num_days:
                cols[wd].write(" ")
            else:
                dnum = days_iter
                default_val = month_flags[dnum]
                month_flags[dnum] = cols[wd].checkbox(
                    f"{dnum}", value=default_val, key=f"work_{year}_{month}_{dnum}"
                )
                days_iter += 1
        if days_iter > num_days:
            break

    if st.button("Save this month’s work schedule"):
        update_work_schedule_for_month(year, month, month_flags)
        st.success("Work schedule saved ✅")


# ------------- HISTORY PAGE -------------

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
            use_container_width=True,
        )

        st.markdown("### Inspect a single day")
        dates_available = df["date"].unique()
        selected_hist_date = st.selectbox("Select date", dates_available)
        day_df = df[df["date"] == selected_hist_date]
        st.write(day_df.T)

        st.markdown("### Export Analytics File (CSV)")
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
                except (ValueError, SyntaxError):
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


# ------------- FLOATING AI COACH BUBBLE -------------

# Chat input form (functional part)
with st.form(key="ai_coach_form", clear_on_submit=True):
    user_prompt = st.text_input(
        "Ask your AI coach something:",
        value="",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Send to AI Coach")
    if submitted and user_prompt.strip():
        add_ai_message("user", user_prompt.strip())
        reply = call_ai_coach(user_prompt.strip())
        add_ai_message("assistant", reply)
        st.rerun()

# Build HTML for floating bubble & sliding drawer
chat_html_messages = ""
for m in st.session_state.ai_messages:
    role_class = "ai-msg-user" if m["role"] == "user" else "ai-msg-assistant"
    chat_html_messages += f"""
    <div class="ai-msg {role_class}">
        <div class="ai-msg-role">{m['role'].capitalize()}</div>
        <div class="ai-msg-content">{m['content']}</div>
    </div>
    """

bubble_html = f"""
<style>
#ai-coach-bubble {{
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 56px;
    height: 56px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4f46e5, #22c55e);
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    z-index: 9999;
    color: white;
    font-weight: 700;
    font-size: 26px;
}}

#ai-coach-bubble:hover {{
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.4);
}}

#ai-coach-drawer {{
    position: fixed;
    bottom: 90px;
    right: 20px;
    width: 320px;
    max-height: 70vh;
    background: #0b1120;
    color: #e5e7eb;
    border-radius: 16px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.6);
    display: none;
    flex-direction: column;
    overflow: hidden;
    z-index: 9998;
    border: 1px solid #1f2937;
}}

#ai-coach-header {{
    padding: 10px 14px;
    background: #111827;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 14px;
    border-bottom: 1px solid #1f2937;
}}
#ai-coach-header span {{
    font-weight: 600;
}}

#ai-coach-close {{
    cursor: pointer;
    font-size: 18px;
    line-height: 1;
}}

#ai-coach-body {{
    padding: 10px;
    overflow-y: auto;
    max-height: 55vh;
    font-size: 13px;
}}

.ai-msg {{
    margin-bottom: 8px;
    padding: 8px;
    border-radius: 8px;
}}
.ai-msg-user {{
    background: #1e293b;
}}
.ai-msg-assistant {{
    background: #020617;
    border: 1px solid #1e293b;
}}
.ai-msg-role {{
    font-size: 10px;
    text-transform: uppercase;
    opacity: 0.6;
    margin-bottom: 2px;
}}
.ai-msg-content {{
    white-space: pre-wrap;
}}

#ai-coach-footer {{
    padding: 8px 10px;
    background: #020617;
    font-size: 11px;
    color: #9ca3af;
    border-top: 1px solid #1f2937;
}}
</style>

<div id="ai-coach-bubble" onclick="toggleAICoachDrawer()">
    💬
</div>

<div id="ai-coach-drawer">
    <div id="ai-coach-header">
        <span>AI Coach</span>
        <div id="ai-coach-close" onclick="toggleAICoachDrawer()">×</div>
    </div>
    <div id="ai-coach-body">
        {chat_html_messages}
    </div>
    <div id="ai-coach-footer">
        Type your question in the input at the very bottom of the page and hit Send.
    </div>
</div>

<script>
function toggleAICoachDrawer() {{
    var drawer = window.parent.document.getElementById('ai-coach-drawer');
    if (!drawer) {{
        drawer = document.getElementById('ai-coach-drawer');
    }}
    if (drawer.style.display === 'flex') {{
        drawer.style.display = 'none';
    }} else {{
        drawer.style.display = 'flex';
    }}
}}
</script>
"""

components.html(bubble_html, height=0, width=0)
