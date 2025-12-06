import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta
import calendar
from typing import Tuple, List, Dict, Any
import ast  # for parsing stored strength history

# ------------------------
# CONFIG
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

    # Remove existing entries for this month
    mask_same_month = df["date"].apply(lambda x: x.year == year and x.month == month)
    df = df[~mask_same_month]

    # Add new entries for month
    new_rows = []
    for day, work_flag in selections.items():
        new_rows.append({"date": date(year, month, day), "is_work": work_flag})
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    save_work_schedule(df)


# ------------------------
# TRAINING PLAN LOGIC
# ------------------------

def get_phase_and_day_plan(d: date) -> Tuple[str, str, str, str]:
    """
    Returns (phase, day_type, planned, kind)
    kind drives UI: StrengthA, StrengthB, Tempo, Hill, FlexCardio,
    LongZ2, Incline, ME, TriBrick, TriRaceLike, Off, Manual
    """
    dt = d  # pure date

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

    # --- Helper functions for rich cardio descriptions ---
    def tempo_run_desc():
        return (
            "Tempo Run — 30–35 minutes\n\n"
            "• Warm-up: 5–10 min easy Z1–low Z2 jog or brisk walk.\n"
            "• Main Set: 2–3 × 6–8 min at controlled tempo (RPE 6–7, comfortably hard), "
            "with 2–3 min easy Z1–Z2 between reps.\n"
            "• Cool-down: 5–10 min easy.\n\n"
            "Goal: Build aerobic power and running economy without going into a race effort.\n"
            "Coaching Tip: Stay relaxed in shoulders and jaw, and aim for a smooth, repeatable cadence."
        )

    def long_z2_desc(mode: str = "Bike/Run"):
        return (
            f"Long Z2 {mode} — 75–90 minutes\n\n"
            "• Warm-up: 10 min easy Z1.\n"
            "• Main Set: 60–70 min steady Z2 (RPE 4–6). You should be able to talk in full sentences.\n"
            "• Cool-down: 5–10 min very easy.\n\n"
            "Goal: Build your base engine and fat-burning capacity.\n"
            "Coaching Tip: Think 'smooth and boring.' If it feels exciting, you're probably going too hard."
        )

    def flex_z2_desc():
        return (
            "Flexible Z2 Cardio — 30–45 minutes\n\n"
            "• Choose: Run, Bike, or Incline Walk depending on fatigue and joints.\n"
            "• Stay in easy Z2 (RPE 3–5) where breathing is steady and controlled.\n\n"
            "Goal: Maintain aerobic volume without digging a recovery hole.\n"
            "Coaching Tip: This is the day to respect soreness and travel fatigue. Err on the easier side."
        )

    def incline_desc(long: bool = False):
        if long:
            return (
                "Long Incline / Hike — 60–90 minutes\n\n"
                "• Continuous uphill walking or hiking at Z2 effort (RPE 4–6).\n"
                "• Use treadmill incline or outdoor hills.\n\n"
                "Goal: Build climbing-specific strength for hills and bike climbing.\n"
                "Coaching Tip: Shorten your stride, lean slightly from the ankles, and keep breathing steady."
            )
        else:
            return (
                "Incline Z2 — 40–60 minutes\n\n"
                "• Steady incline treadmill or outdoor uphill walking in Z2 (RPE 4–5).\n\n"
                "Goal: Low-impact aerobic work that directly supports uphill running and bike climbing.\n"
                "Coaching Tip: Let the incline, not speed, drive the difficulty. If HR creeps too high, slow down."
            )

    def hill_hike_desc():
        return (
            "Hill Hike — 45–75 minutes\n\n"
            "• Continuous or looped hiking on hills or stair machines, staying mostly in Z2.\n\n"
            "Goal: Build leg strength and durability for climbs without pounding from running.\n"
            "Coaching Tip: Keep steps deliberate and posture tall; don't sprint the hills, just grind steadily."
        )

    def me_desc():
        return (
            "Muscular Endurance Step-Ups + Easy Cardio\n\n"
            "• 3 rounds:\n"
            "  – 6 minutes continuous step-ups with pack (moderate weight).\n"
            "  – 2 minutes easy standing or slow walk recovery.\n"
            "• Then: 20–25 minutes easy Z1–Z2 Run/Bike/Incline.\n\n"
            "Goal: Build local muscular endurance in quads and glutes for climbing and bike power.\n"
            "Coaching Tip: Smooth, controlled steps. No bouncing. Legs should burn; breathing stays controlled."
        )

    def brick_desc(long: bool = False):
        if long:
            return (
                "Long Brick — Bike → Run\n\n"
                "• Bike: 60–90 min Z2 with short segments at race-like effort.\n"
                "• Transition: 5–10 min quick change.\n"
                "• Run: 15–25 min off the bike at easy–moderate pace (Z2–Z3-).\n\n"
                "Goal: Teach your legs to run well off the bike and practice fueling.\n"
                "Coaching Tip: First 5 minutes of the run should feel laughably easy—let your legs come to you."
            )
        else:
            return (
                "Brick — Bike → Run\n\n"
                "• Bike: 30–45 min Z2–Z3-.\n"
                "• Transition: quick change.\n"
                "• Run: 10–15 min off the bike at easy–steady pace.\n\n"
                "Goal: Groove the feel of running off the bike without huge fatigue.\n"
                "Coaching Tip: Focus on cadence and relaxed upper body on the run; don't chase speed."
            )

    def race_like_desc(modality: str):
        return (
            f"Race-Pace {modality}\n\n"
            "• Short intervals at your target race effort with plenty of easy recovery.\n"
            "• Total hard time is small; quality > quantity.\n\n"
            "Goal: Lock in the feel of race pace and movement economy.\n"
            "Coaching Tip: Ask yourself: 'Could I hold this on race day if conditions were good?' "
            "If the answer is no, back it off slightly."
        )

    # Phase 2A – Strength + Aerobic Base
    if start_2a <= dt <= end_2a:
        phase = "Phase 2A – Strength + Aerobic Base"
        if dow == 0:
            planned = (
                "Strength A – Upper Focus\n\n"
                "• Bench Press, Pull-Ups, Incline Press, Row, Lateral Raises, Triceps.\n"
                "• Optional: 5–10 min very easy cardio at the end.\n\n"
                "Goal: Build upper-body strength and muscle to support overall performance and GBRS work.\n"
                "Coaching Tip: Leave 1–3 reps in the tank on main lifts; this is not a max-out phase."
            )
            return phase, "Strength A – Upper Focus", planned, "StrengthA"
        elif dow == 1:
            return phase, "Aerobic Tempo Run", tempo_run_desc(), "Tempo"
        elif dow == 2:
            return phase, "Z2 Hill Hike", hill_hike_desc(), "Hill"
        elif dow == 3:
            planned = (
                "Strength B – Lower / Hypertrophy\n\n"
                "• Deadlift/RDL, Split Squat or Lunge, Hip Thrust, Row Variation, Biceps, Core.\n\n"
                "Goal: Build lower-body strength and muscle to support hills, biking and overall power.\n"
                "Coaching Tip: Control the eccentric (lowering), and don't chase fatigue at the expense of form."
            )
            return phase, "Strength B – Lower/Hypertrophy", planned, "StrengthB"
        elif dow == 4:
            return phase, "Aerobic Flex Day", flex_z2_desc(), "FlexCardio"
        elif dow == 5:
            return phase, "Long Z2 Endurance", long_z2_desc("Bike/Run/Hike"), "LongZ2"
        else:
            planned = (
                "Off / Recovery\n\n"
                "• Rest, light mobility, optional very easy walk.\n\n"
                "Goal: Absorb the week's work and show up fresh for key days.\n"
                "Coaching Tip: If you're unsure whether to train, default to recovery here."
            )
            return phase, "Off / Recovery", planned, "Off"

    # Phase 2B – Muscular Endurance (ME)
    if start_2b <= dt <= end_2b:
        phase = "Phase 2B – Muscular Endurance"
        if dow == 0:
            return phase, "Z2 Incline", incline_desc(long=False), "Incline"
        elif dow == 1:
            planned = (
                "Strength A – Upper Focus (Slightly De-Emphasized)\n\n"
                "• Bench, Pull-Ups, Incline, Row, Delts, Triceps.\n\n"
                "Goal: Maintain and slightly build strength while ME takes priority.\n"
                "Coaching Tip: Keep 2–3 reps in reserve and don't let this session crush your legs/central fatigue."
            )
            return phase, "Strength A – Upper Focus", planned, "StrengthA"
        elif dow == 2:
            return phase, "ME Step-Ups + Easy Cardio", me_desc(), "ME"
        elif dow == 3:
            return phase, "Z2 Incline", incline_desc(long=False), "Incline"
        elif dow == 4:
            planned = (
                "Strength B – Lower (Light)\n\n"
                "• Light Deadlift/RDL, Carries, Rear Delts, Biceps, Core.\n\n"
                "Goal: Support ME work without overloading legs.\n"
                "Coaching Tip: Treat this as 'movement practice' more than a 'crush yourself' day."
            )
            return phase, "Strength B – Lower (Light)", planned, "StrengthB"
        elif dow == 5:
            return phase, "Long Incline / Hike", incline_desc(long=True), "LongZ2"
        else:
            planned = (
                "Off / Recovery\n\n"
                "• Rest, light mobility, optional walk.\n"
                "Coaching Tip: If ME days are burying you, lean into this rest day."
            )
            return phase, "Off / Recovery", planned, "Off"

    # Phase 2C – Aerobic Power Transition
    if start_2c <= dt <= end_2c:
        phase = "Phase 2C – Aerobic Power / Transition"
        if dow == 0:
            planned = (
                "Strength A – Upper\n\n"
                "• Bench, Pull-Ups, Incline, Row, Delts, Triceps.\n\n"
                "Goal: Keep building upper strength while aerobic work turns slightly sharper.\n"
                "Coaching Tip: Maintain controlled form and consistent progress; no grinders."
            )
            return phase, "Strength A – Upper", planned, "StrengthA"
        elif dow == 1:
            return phase, "Tempo / Threshold Run", tempo_run_desc(), "Tempo"
        elif dow == 2:
            return phase, "Z2 Bike / Incline", flex_z2_desc(), "FlexCardio"
        elif dow == 3:
            planned = (
                "Strength B – Lower\n\n"
                "• Deadlift/RDL, Split Squat/Lunge, Hip Thrust, Row Variation, Biceps, Core.\n\n"
                "Goal: Maintain leg strength while run/bike intensity starts to rise.\n"
                "Coaching Tip: If your legs feel overly cooked, trim sets slightly and keep bar speed snappy."
            )
            return phase, "Strength B – Lower", planned, "StrengthB"
        elif dow == 4:
            return phase, "Easy Z2 Cardio", flex_z2_desc(), "FlexCardio"
        elif dow == 5:
            return phase, "Long Z2 Endurance", long_z2_desc("Bike/Run combo"), "LongZ2"
        else:
            planned = (
                "Off / Recovery\n\n"
                "• Rest, mobility, easy walking.\n"
                "Coaching Tip: Use today to de-stress life outside training too."
            )
            return phase, "Off / Recovery", planned, "Off"

    # Phase 3A – Triathlon Base
    if start_3a <= dt <= end_3a:
        phase = "Phase 3A – Triathlon Base"
        if dow == 0:
            planned = (
                "Swim (Technique) + Strength A (Short)\n\n"
                "• Swim: 20–30 min of easy technique drills and short repeats.\n"
                "• Strength A: Fewer sets than full day; focus on quality.\n\n"
                "Goal: Build swim comfort while keeping upper-body strength.\n"
                "Coaching Tip: Smooth water feel > effort. On strength, leave 3 reps in the tank."
            )
            return phase, "Swim + Strength A", planned, "StrengthA"
        elif dow == 1:
            return phase, "Z2 Bike", flex_z2_desc(), "FlexCardio"
        elif dow == 2:
            return phase, "Z2 Run", flex_z2_desc(), "FlexCardio"
        elif dow == 3:
            planned = (
                "Swim + Strength B (Light)\n\n"
                "• Swim: Technique / easy aerobic.\n"
                "• Strength B: Light sets for lower body.\n\n"
                "Goal: Maintain strength without interfering with growing tri volume.\n"
                "Coaching Tip: If in doubt, cut sets before you cut swim technique work."
            )
            return phase, "Swim + Strength B", planned, "StrengthB"
        elif dow == 4:
            return phase, "Tempo Bike / Run", tempo_run_desc(), "Tempo"
        elif dow == 5:
            return phase, "Long Z2 Bike / Run", long_z2_desc("Bike/Run"), "LongZ2"
        else:
            planned = (
                "Off / Recovery\n\n"
                "Coaching Tip: Sleep and fueling matter more now than 'one extra easy session.'"
            )
            return phase, "Off / Recovery", planned, "Off"

    # Phase 3B – Triathlon Build
    if start_3b <= dt <= end_3b:
        phase = "Phase 3B – Triathlon Build"
        if dow == 0:
            planned = (
                "Swim Intervals + Optional Short Strength A\n\n"
                "• Swim: Intervals at moderate-hard effort with easy rest.\n"
                "• Optional Strength A: cut volume if you feel cooked.\n\n"
                "Goal: Raise swim fitness while preserving strength.\n"
                "Coaching Tip: On a long travel week, skip the strength before skipping the swim."
            )
            return phase, "Swim Intervals", planned, "StrengthA"
        elif dow == 1:
            planned = (
                "Bike Tempo — 45–75 minutes\n\n"
                "• Warm-up: 10 min easy.\n"
                "• Main: 2–4 × 8–12 min at strong tempo (RPE 7) with 4–5 min easy between.\n"
                "• Cool-down: 5–10 min easy.\n\n"
                "Goal: Boost sustained power on the bike.\n"
                "Coaching Tip: Focus on smooth cadence (85–95 rpm) and stable torso."
            )
            return phase, "Bike Tempo", planned, "Tempo"
        elif dow == 2:
            planned = (
                "Run Intervals — 30–45 minutes\n\n"
                "• Warm-up: 10 min easy.\n"
                "• Main: 4–6 × 3–4 min near 10K/threshold pace (RPE 7–8) "
                "with equal or slightly shorter easy jog recovery.\n"
                "• Cool-down: 5–10 min easy.\n\n"
                "Goal: Raise running speed and resilience at faster paces.\n"
                "Coaching Tip: First couple intervals should feel too easy—save the grind for the last one or two."
            )
            return phase, "Run Intervals", planned, "Tempo"
        elif dow == 3:
            planned = (
                "Swim + Easy Strength B\n\n"
                "• Swim: Aerobic or light interval work.\n"
                "• Strength B: Very submaximal, focus on joint health and posture.\n\n"
                "Goal: Maintain strength without beating up tired legs.\n"
                "Coaching Tip: If brick/long sessions feel heavy, drop this strength session entirely."
            )
            return phase, "Swim + Strength B", planned, "StrengthB"
        elif dow == 4:
            return phase, "Brick (Bike → Run)", brick_desc(long=False), "TriBrick"
        elif dow == 5:
            return phase, "Long Bike / Long Brick", brick_desc(long=True), "TriBrick"
        else:
            planned = "Off / Recovery\n\nCoaching Tip: Full rest days are where the adaptation happens."
            return phase, "Off / Recovery", planned, "Off"

    # Phase 4 – Peak Prep
    if start_4 <= dt <= end_4:
        phase = "Phase 4 – Peak / Specific Prep"
        if dow == 0:
            return phase, "Race-Pace Swim", race_like_desc("Swim"), "TriRaceLike"
        elif dow == 1:
            return phase, "Race-Pace Bike", race_like_desc("Bike"), "TriRaceLike"
        elif dow == 2:
            return phase, "Race-Pace Run", race_like_desc("Run"), "TriRaceLike"
        elif dow == 3:
            planned = (
                "Easy Swim + Very Light Strength\n\n"
                "Goal: Keep movement quality high without adding fatigue.\n"
                "Coaching Tip: Treat strength as mobility with resistance here."
            )
            return phase, "Easy Swim + Light Strength", planned, "FlexCardio"
        elif dow == 4:
            return phase, "Race Simulation Brick", brick_desc(long=False), "TriBrick"
        elif dow == 5:
            return phase, "Long Race-Specific Session", brick_desc(long=True), "TriBrick"
        else:
            planned = (
                "Off / Recovery\n\n"
                "Coaching Tip: Protect sleep like it's part of the training plan (it is)."
            )
            return phase, "Off / Recovery", planned, "Off"

    # Taper
    if start_taper <= dt <= race_day:
        phase = "Taper"
        if dt == race_day:
            planned = (
                "RACE DAY – Tahoe Triathlon\n\n"
                "Goal: Execute, stay calm, and have fun. Trust the work you've done.\n"
                "Coaching Tip: Control the controllables. Nutrition, pacing, and mindset beat last-minute heroics."
            )
            return phase, "RACE DAY – Tahoe Triathlon", planned, "TriRaceLike"

        if dow in [0, 2]:
            planned = race_like_desc("Swim/Bike/Run (short)")
            return phase, "Short Race-Pace Sharpening", planned, "TriRaceLike"
        elif dow in [1, 3]:
            planned = (
                "Easy Z2 Cardio — 30–45 minutes\n\n"
                "Goal: Keep the engine idling without adding fatigue.\n"
                "Coaching Tip: You should finish these feeling better than you started."
            )
            return phase, "Easy Z2 Cardio", planned, "FlexCardio"
        elif dow == 4:
            return phase, "Very Short Brick", brick_desc(long=False), "TriBrick"
        elif dow == 5:
            planned = (
                "Optional Easy Session\n\n"
                "Coaching Tip: If you feel even slightly run down, skip this and rest."
            )
            return phase, "Optional Easy Session", planned, "FlexCardio"
        else:
            planned = (
                "Off / Pre-Race Rest\n\n"
                "Coaching Tip: Hydrate, eat normally, and visualize smooth race execution."
            )
            return phase, "Off / Pre-Race Rest", planned, "Off"

    # Fallback
    return "Unprogrammed", "Manual / Free Day", "No structured plan.", "Manual"


def adjust_for_workday(phase: str, day_type: str, planned: str, kind: str, work: bool) -> Tuple[str, str, str, str, bool]:
    """
    Moderate adjustment: reshape workouts on work days to be less demanding if they are long/ME/brick.
    Returns (phase, day_type_adj, planned_adj, kind_adj, adjusted_flag)
    """
    if not work:
        return phase, day_type, planned, kind, False

    # Don't adjust off/manual
    if kind in ["Off", "Manual"]:
        return phase, day_type, planned, kind, False

    if kind in ["LongZ2", "ME", "TriBrick", "TriRaceLike"]:
        new_day_type = f"{day_type} (Adjusted for Workday)"
        new_planned = (
            "Travel/work day adjustment:\n\n"
            "• Swap to 25–40 min easy Z1–Z2 cardio (Run/Bike/Incline).\n"
            "• Keep effort low; goal is circulation and headspace, not fitness PRs.\n\n"
            "Coaching Tip: On long duty days, the win is just 'did something easy and didn't dig a hole.'"
        )
        new_kind = "FlexCardio"
        return phase, new_day_type, new_planned, new_kind, True

    if kind in ["StrengthA", "StrengthB"]:
        new_day_type = f"{day_type} (Workday – lighter loads)"
        new_planned = planned + "\n\nWorkday Adjustment: Keep 2–4 reps in reserve and cut one set if you feel drained."
        return phase, new_day_type, new_planned, kind, True

    if kind == "Tempo":
        new_day_type = f"{day_type} (Workday – controlled)"
        new_planned = (
            "Tempo (Workday version)\n\n"
            "• Option 1: 1–2 short tempo reps (6–8 min) instead of full workout.\n"
            "• Option 2: Convert to 30–40 min Z2 only if very fatigued.\n\n"
            "Coaching Tip: Travel and sleep debt make tempo feel harder; it's okay to back off."
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
    elif kind == "StrengthB":
        return [
            {"name": "Deadlift / RDL", "alts": DEADLIFT_ALTS},
            {"name": "Split Squat / Lunge", "alts": SPLIT_SQUAT_ALTS},
            {"name": "Hip Thrust", "alts": HIPTHRUST_ALTS},
            {"name": "Row Variation", "alts": ROW_ALTS},
            {"name": "Biceps", "alts": BICEPS_ALTS},
            {"name": "Core", "alts": CORE_ALTS},
        ]
    elif kind == "ME":
        return [
            {"name": "ME Step-Ups", "alts": ME_ALTS},
        ]
    else:
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


def get_last_strength_entry(
    df: pd.DataFrame,
    current_date: date,
    exercise_name: str,
    variant_name: str,
):
    """
    Look back in the log for the most recent entry of this exercise+variant
    before current_date.

    Returns dict with {date, exercise, variant, sets, reps, weight, rpe} or None.
    """
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
# STREAMLIT UI
# ------------------------

st.set_page_config(page_title="2025–26 Training Log", layout="centered")

page = st.sidebar.radio("Page", ["Today", "Work Schedule", "History"])

# ------------- TODAY PAGE -------------

if page == "Today":
    st.title("2025–26 Training Log")

    today = date.today()
    selected_date = st.date_input("Date", value=today)

    # Base programmed workout for this calendar date
    work_flag = is_workday(selected_date)
    base_phase, base_day_type, base_planned, base_kind = get_phase_and_day_plan(selected_date)

    # Workout mode selector
    st.markdown("### Workout Mode")
    workout_mode = st.selectbox(
        "Use which workout today?",
        [
            "Auto (Programmed / Adjusted)",
            "Choose from This Week",
            "Manual / Custom",
            "Rest / Skip",
        ],
    )

    # Prepare final values
    effective_phase = base_phase
    effective_day_type = base_day_type
    effective_planned = base_planned
    effective_kind = base_kind
    adjusted = False
    source_date = selected_date  # date the template came from
    mode_label = workout_mode

    # Helper: compute this week’s programmed days (Mon–Sun)
    week_monday = selected_date - timedelta(days=selected_date.weekday())
    week_days = [week_monday + timedelta(days=i) for i in range(7)]

    if workout_mode == "Auto (Programmed / Adjusted)":
        # Workday override for auto-adjust
        override = st.checkbox(
            "Override workday adjustment (use original programmed workout)",
            value=False,
        )

        if override:
            effective_phase = base_phase
            effective_day_type = base_day_type
            effective_planned = base_planned
            effective_kind = base_kind
            adjusted = False
        else:
            effective_phase, effective_day_type, effective_planned, effective_kind, adjusted = adjust_for_workday(
                base_phase, base_day_type, base_planned, base_kind, work_flag
            )

    elif workout_mode == "Choose from This Week":
        # Build list of options for this week
        options = []
        for d in week_days:
            ph, dtp, pln, knd = get_phase_and_day_plan(d)
            label = f"{d.strftime('%a %m-%d')} – {dtp} ({ph})"
            options.append(
                {
                    "date": d,
                    "label": label,
                    "phase": ph,
                    "day_type": dtp,
                    "planned": pln,
                    "kind": knd,
                }
            )

        labels = [o["label"] for o in options]
        choice = st.selectbox("Pick a workout from this week:", labels)
        chosen = next(o for o in options if o["label"] == choice)

        effective_phase = chosen["phase"]
        effective_day_type = chosen["day_type"]
        effective_planned = chosen["planned"]
        effective_kind = chosen["kind"]
        source_date = chosen["date"]
        adjusted = False  # this is deliberate choice, not auto-adjust

    elif workout_mode == "Manual / Custom":
        effective_phase = "Manual"
        effective_day_type = "Manual / Custom"
        effective_planned = (
            "Manual custom session.\n\n"
            "Use the Strength / Cardio sections and Notes to describe what you actually did."
        )
        effective_kind = "Manual"
        adjusted = False

    elif workout_mode == "Rest / Skip":
        effective_phase = base_phase if base_phase != "Unprogrammed" else "Rest"
        effective_day_type = "Rest / Skip"
        effective_planned = (
            "Rest / Skip Day\n\n"
            "Use this when you intentionally skip training (travel, fatigue, life stuff).\n"
            "Coaching Tip: One skipped day in a well-run block is noise, not a disaster."
        )
        effective_kind = "Off"
        adjusted = False

    # Display phase + planned
    st.subheader(effective_phase)
    if workout_mode == "Auto (Programmed / Adjusted)":
        if adjusted:
            st.caption(f"Original: {base_day_type}")
            st.markdown(f"**Adjusted for Workday:** {effective_day_type}")
        else:
            st.caption(f"Day Type: {effective_day_type}")
    else:
        # Non-auto mode: show what you chose and where it came from
        if workout_mode == "Choose from This Week":
            st.caption(f"Using: {effective_day_type} (from {source_date.strftime('%a %m-%d')})")
        else:
            st.caption(f"Day Type: {effective_day_type}")

    st.write(f"**Planned Session Details:**\n\n{effective_planned}")

    if work_flag and workout_mode == "Auto (Programmed / Adjusted)":
        st.info("This date is marked as a WORKDAY in your schedule.")
    elif work_flag and workout_mode != "Auto (Programmed / Adjusted)":
        st.info("This is a WORKDAY, but you have manually chosen today's workout.")

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

    # Cardio log (only if this workout involves cardio)
    if effective_kind in ["Tempo", "Hill", "FlexCardio", "LongZ2", "Incline", "ME", "TriBrick", "TriRaceLike"]:
        st.markdown("### Cardio Log")
        cardio_mode = st.selectbox("Cardio mode", CARDIO_MODES, index=0)
        cardio_duration = st.number_input("Cardio duration (min)", min_value=0, max_value=600, value=0)
        cardio_distance = st.text_input("Cardio distance (mi/km)", value="")
        cardio_avg_hr = st.text_input("Cardio avg HR (bpm)", value="")
        cardio_rpe = st.slider("Cardio RPE (1–10)", 1, 10, 6)

        log_data.update(
            {
                "cardio_mode": cardio_mode,
                "cardio_duration_min": cardio_duration,
                "cardio_distance": cardio_distance,
                "cardio_avg_hr": cardio_avg_hr,
                "cardio_rpe_1_10": cardio_rpe,
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
            }
        )

    st.markdown("---")

    # Strength / ME block (only if this workout is a strength/ME type)
    df_log = load_log()
    exercises = get_strength_exercises(effective_kind)
    strength_entries = []

    if exercises:
        st.markdown("### Strength / ME Session")

        for i, ex in enumerate(exercises):
            st.markdown(f"**{ex['name']}**")

            # 1) Variant selector first
            variant = st.selectbox(
                "Variant",
                ex["alts"],
                key=f"{selected_date}_variant_{i}",
            )

            # 2) Variant-specific last attempt
            last = get_last_strength_entry(df_log, selected_date, ex["name"], variant)

            if last:
                last_sets = last.get("sets", "")
                last_reps = last.get("reps", "")
                last_wt = last.get("weight", "")
                last_rpe = last.get("rpe", "")
                last_date = last.get("date")
                st.caption(
                    f"Last ({variant}) — {last_date}: "
                    f"{last_sets}×{last_reps} @ {last_wt} (RPE {last_rpe})"
                )
            else:
                st.caption(f"Last ({variant}) — no previous log for this variant yet.")

            # 3) Today's log for this variant
            col1, col2, col3, col4 = st.columns(4)
            sets = col1.text_input("Sets", key=f"{selected_date}_sets_{i}")
            reps = col2.text_input("Reps", key=f"{selected_date}_reps_{i}")
            weight = col3.text_input("Weight", key=f"{selected_date}_wt_{i}")
            rpe = col4.text_input("RPE", key=f"{selected_date}_rpe_{i}")

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

    st.markdown("---")

    st.markdown("### Daily Trackers")
    colA, colB, colC = st.columns(3)
    hrv = colA.text_input("HRV", value="")
    sleep_hrs = colB.text_input("Sleep hours", value="")
    mood = colC.slider("Mood (1–5)", 1, 5, 3)

    colD, colE, colF = st.columns(3)
    soreness = colD.slider("Soreness (1–5)", 1, 5, 3)
    energy = colE.slider("Energy (1–5)", 1, 5, 3)
    # colF reserved

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

    st.write("Tap the days you are **working (on tour/duty)**. The plan will auto-adjust those days in Auto mode.")

    today = date.today()
    year = st.number_input("Year", min_value=2025, max_value=2030, value=today.year)
    month = st.number_input("Month", min_value=1, max_value=12, value=today.month)

    year = int(year)
    month = int(month)

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
    for week in range(6):
        cols = st.columns(7)
        for wd in range(7):
            if week == 0 and wd < first_weekday:
                cols[wd].write(" ")
            elif days_iter > num_days:
                cols[wd].write(" ")
            else:
                dnum = days_iter
                label = f"{dnum}"
                default_val = month_flags[dnum]
                month_flags[dnum] = cols[wd].checkbox(
                    label,
                    value=default_val,
                    key=f"work_{year}_{month}_{dnum}"
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
