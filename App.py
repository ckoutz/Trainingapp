import streamlit as st
import pandas as pd
from datetime import date, datetime
import calendar
from typing import Tuple, List, Dict, Any

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
    """
    selections: { day -> is_work }
    """
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
    LongZ2, Incline, ME, TriEasy, TriBrick, TriRaceLike, Off, Manual
    """
    dt = d  # pure date

    # Macro dates for 2025–26
    start_2a = date(2025, 12, 8)
    end_2a = date(2026, 1, 19)

    start_2b = date(2026, 1, 20)
    end_2b = date(2026, 2, 23)

    start_2c = date(2026, 2, 24)
    end_2c = date(2026, 3, 30)

    start_3a = date(2026, 3, 31)
    end_3a = date(2026, 5, 4)

    start_3b = date(2026, 5, 5)
    end_3b = date(2026, 6, 22)

    start_4 = date(2026, 6, 23)
    end_4 = date(2026, 8, 10)

    start_taper = date(2026, 8, 11)
    race_day = date(2026, 8, 29)

    if dt < start_2a or dt > race_day:
        return (
            "Unprogrammed",
            "Manual / Free Day",
            "No structured plan for this date yet. Use this as a free day or enter a manual workout.",
            "Manual",
        )

    dow = dt.weekday()  # Mon=0

    # Phase 2A – Strength + Aerobic Base
    if start_2a <= dt <= end_2a:
        phase = "Phase 2A – Strength + Aerobic Base"
        if dow == 0:
            return phase, "Strength A – Upper Focus", "Bench, Pull-Ups, Incline Press, Row, Delts, Triceps. Optional 10-min very easy jog.", "StrengthA"
        elif dow == 1:
            return phase, "Aerobic Tempo Run", "20–30 min steady aerobic run (high Z2 / low Z3). Cardio: Run/Bike/Incline.", "Tempo"
        elif dow == 2:
            return phase, "Z2 Hill Hike", "45–75 min Z2 hill hiking. Cardio: Hike/Incline/Run/Bike.", "Hill"
        elif dow == 3:
            return phase, "Strength B – Lower/Hypertrophy", "Deadlift/RDL, Split Squat/Lunge, Hip Thrust, Row, Biceps, Core.", "StrengthB"
        elif dow == 4:
            return phase, "Aerobic Flex Day", "Easy 25–35 min Z1–Z2. Choose Run/Bike/Incline based on fatigue.", "FlexCardio"
        elif dow == 5:
            return phase, "Long Z2 Endurance", "60–90 min continuous Z2. Prefer Bike or Long Hike; Run only if calves feel great.", "LongZ2"
        else:
            return phase, "Off / Recovery", "Rest, light mobility, optional easy walk.", "Off"

    # Phase 2B – Muscular Endurance (ME)
    if start_2b <= dt <= end_2b:
        phase = "Phase 2B – Muscular Endurance"
        if dow == 0:
            return phase, "Z2 Incline", "40–60 min Z2 incline walk. Cardio: Run/Bike/Incline.", "Incline"
        elif dow == 1:
            return phase, "Strength A – Upper Focus", "Bench, Pull-Ups, Incline Press, Row, Delts, Triceps.", "StrengthA"
        elif dow == 2:
            return phase, "ME Step-Ups + Easy Cardio", "3×6 min step-ups with pack (2-min rest) + 20–25 min very easy Run/Bike/Incline.", "ME"
        elif dow == 3:
            return phase, "Z2 Incline", "40–60 min Z2 incline walk. Cardio: Run/Bike/Incline.", "Incline"
        elif dow == 4:
            return phase, "Strength B – Lower (Light)", "Light Deadlift/RDL, Carries, Rear Delts, Biceps, Core.", "StrengthB"
        elif dow == 5:
            return phase, "Long Incline / Hike", "60–90 min Z2 long incline walk or hike. Cardio: Hike/Incline/Bike.", "LongZ2"
        else:
            return phase, "Off / Recovery", "Rest, light mobility, optional easy walk.", "Off"

    # Phase 2C – Aerobic Power Transition
    if start_2c <= dt <= end_2c:
        phase = "Phase 2C – Aerobic Power / Transition"
        if dow == 0:
            return phase, "Strength A – Upper", "Bench, Pull-Ups, Incline, Row, Delts, Triceps.", "StrengthA"
        elif dow == 1:
            return phase, "Tempo / Threshold Run", "25–35 min tempo/threshold-style run or split intervals. Cardio: Run/Bike/Incline.", "Tempo"
        elif dow == 2:
            return phase, "Z2 Bike or Incline", "45–75 min Z2 Bike or Incline walk.", "FlexCardio"
        elif dow == 3:
            return phase, "Strength B – Lower", "Deadlift/RDL, Split Squat/Lunge, Hip Thrust, Row, Biceps, Core.", "StrengthB"
        elif dow == 4:
            return phase, "Easy Z2 Cardio", "30–45 min easy Z2. Choose Run/Bike/Incline.", "FlexCardio"
        elif dow == 5:
            return phase, "Long Z2 Endurance", "75–90 min Z2 (Bike/Run combo).", "LongZ2"
        else:
            return phase, "Off / Recovery", "Rest, light mobility, optional easy walk.", "Off"

    # Phase 3A – Triathlon Base
    if start_3a <= dt <= end_3a:
        phase = "Phase 3A – Triathlon Base"
        if dow == 0:
            return phase, "Swim + Strength A", "Technique-focused swim + short Strength A (upper).", "StrengthA"
        elif dow == 1:
            return phase, "Z2 Bike", "45–75 min Z2 bike.", "FlexCardio"
        elif dow == 2:
            return phase, "Z2 Run", "30–45 min Z2 run.", "FlexCardio"
        elif dow == 3:
            return phase, "Swim + Strength B", "Swim + short Strength B (lower-light).", "StrengthB"
        elif dow == 4:
            return phase, "Tempo Bike / Run", "Short tempo intervals on bike or run.", "Tempo"
        elif dow == 5:
            return phase, "Long Z2 Bike or Run", "90+ min Z2 bike or 60+ min Z2 run.", "LongZ2"
        else:
            return phase, "Off / Recovery", "Rest, light mobility.", "Off"

    # Phase 3B – Triathlon Build
    if start_3b <= dt <= end_3b:
        phase = "Phase 3B – Triathlon Build"
        if dow == 0:
            return phase, "Swim Intervals", "Swim intervals + optional short Strength A.", "StrengthA"
        elif dow == 1:
            return phase, "Bike Tempo", "45–75 min bike with tempo segments.", "Tempo"
        elif dow == 2:
            return phase, "Run Intervals", "Short intervals around 10K/threshold pace.", "Tempo"
        elif dow == 3:
            return phase, "Swim + Easy Strength B", "Swim + strength maintenance.", "StrengthB"
        elif dow == 4:
            return phase, "Brick (Bike → Run)", "Moderate bike followed immediately by short run off-bike.", "TriBrick"
        elif dow == 5:
            return phase, "Long Bike / Long Brick", "Long bike or bike+run brick in Z2–Z3.", "TriBrick"
        else:
            return phase, "Off / Recovery", "Rest, light mobility.", "Off"

    # Phase 4 – Peak Prep
    if start_4 <= dt <= end_4:
        phase = "Phase 4 – Peak / Specific Prep"
        if dow == 0:
            return phase, "Race-Pace Swim", "Race-pace swim intervals.", "TriRaceLike"
        elif dow == 1:
            return phase, "Race-Pace Bike", "Bike intervals at race power/HR.", "TriRaceLike"
        elif dow == 2:
            return phase, "Race-Pace Run", "Run intervals at race effort.", "TriRaceLike"
        elif dow == 3:
            return phase, "Easy Swim + Strength (very light)", "Technique swim + minimal strength.", "FlexCardio"
        elif dow == 4:
            return phase, "Brick (Race Simulation)", "Short race-pace brick (bike→run).", "TriBrick"
        elif dow == 5:
            return phase, "Long Race-Specific Session", "Long bike or brick near race intensity.", "TriBrick"
        else:
            return phase, "Off / Recovery", "Rest, light mobility.", "Off"

    # Taper
    if start_taper <= dt <= race_day:
        phase = "Taper"
        if dow in [0, 2]:
            return phase, "Short Race-Pace Sharpening", "Short race-pace efforts, low volume.", "TriRaceLike"
        elif dow in [1, 3]:
            return phase, "Easy Z2 Cardio", "Easy Z2 swim/bike/run.", "FlexCardio"
        elif dow == 4:
            return phase, "Very Short Brick", "Very small brick, mostly just to stay sharp.", "TriBrick"
        elif dow == 5:
            return phase, "Optional Easy Session", "Optional easy Z1–Z2; otherwise rest.", "FlexCardio"
        else:
            if dt == race_day:
                return phase, "RACE DAY – Tahoe Triathlon", "Race! Execute the plan and have fun.", "TriRaceLike"
            return phase, "Off / Pre-Race Rest", "Rest, mental review, short shakeouts only.", "Off"

    # Fallback
    return "Unprogrammed", "Manual / Free Day", "No structured plan.", "Manual"


def adjust_for_workday(phase: str, day_type: str, planned: str, kind: str, work: bool) -> Tuple[str, str, str, str, bool]:
    """
    Moderate adjustment (B): reshapes workouts on work days to be less demanding if they are long/ME/brick.
    Returns (phase, day_type_adj, planned_adj, kind_adj, adjusted_flag)
    """
    if not work:
        return phase, day_type, planned, kind, False

    # Don't adjust off/manual
    if kind in ["Off", "Manual"]:
        return phase, day_type, planned, kind, False

    # For moderate adjustment:
    # - LongZ2, ME, TriBrick, TriRaceLike -> simplified FlexCardio on work day
    if kind in ["LongZ2", "ME", "TriBrick", "TriRaceLike"]:
        new_day_type = f"{day_type} (Adjusted for Workday)"
        new_planned = "Travel/work day: swap to 25–40 min easy Z1–Z2 cardio. Choose Run/Bike/Incline based on fatigue."
        new_kind = "FlexCardio"
        return phase, new_day_type, new_planned, new_kind, True

    # Keep Strength but mentally flagged as adjusted
    if kind in ["StrengthA", "StrengthB"]:
        new_day_type = f"{day_type} (Workday – consider lighter loads)"
        new_planned = planned + " Adjust loads down if fatigued; prioritize technique."
        return phase, new_day_type, new_planned, kind, True

    # Tempo on workday -> slightly tamed
    if kind == "Tempo":
        new_day_type = f"{day_type} (Workday – controlled)"
        new_planned = "Keep tempo section short and controlled (e.g., 2×6–8 min) or do 30–40 min Z2 only."
        return phase, new_day_type, new_planned, kind, True

    # FlexCardio / Incline / TriEasy remain basically unchanged
    return phase, day_type, planned, kind, False


def get_strength_exercises(kind: str) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts:
    { name, alts }
    """
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

    work_flag = is_workday(selected_date)

    phase, day_type, planned, kind = get_phase_and_day_plan(selected_date)
    phase, day_type_adj, planned_adj, kind_adj, adjusted = adjust_for_workday(
        phase, day_type, planned, kind, work_flag
    )

    st.subheader(phase)
    if adjusted:
        st.caption(f"Original: {day_type}")
        st.markdown(f"**Adjusted for Workday:** {day_type_adj}")
    else:
        st.caption(f"Day Type: {day_type_adj}")
    st.write(f"**Planned Workout:** {planned_adj}")

    if work_flag:
        st.info("This date is marked as a WORKDAY in your schedule.")

    st.markdown("---")

    log_data: Dict[str, Any] = {
        "date": selected_date,
        "phase": phase,
        "day_type": day_type,
        "day_type_adjusted": day_type_adj,
        "kind": kind,
        "kind_adjusted": kind_adj,
        "workday": work_flag,
    }

    # Cardio block for days that include cardio
    if kind_adj in ["Tempo", "Hill", "FlexCardio", "LongZ2", "Incline", "ME", "TriBrick", "TriRaceLike"]:
        st.markdown("### Cardio")
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

    # Strength / ME block
    exercises = get_strength_exercises(kind_adj)
    strength_entries = []

    if exercises:
        st.markdown("### Strength / ME Session")
        for i, ex in enumerate(exercises):
            st.markdown(f"**{ex['name']}**")
            alt_choice = st.selectbox(
                "Select variant", ex["alts"], key=f"{selected_date}_alt_{i}"
            )

            col1, col2, col3, col4 = st.columns(4)
            sets = col1.text_input("Sets", key=f"{selected_date}_sets_{i}")
            reps = col2.text_input("Reps", key=f"{selected_date}_reps_{i}")
            weight = col3.text_input("Weight", key=f"{selected_date}_wt_{i}")
            rpe = col4.text_input("RPE", key=f"{selected_date}_rpe_{i}")

            strength_entries.append(
                {
                    "exercise": ex["name"],
                    "alt": alt_choice,
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
    # colF reserved for future

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

    st.write("Mark which days are **work days** (tours / duty). The training plan will auto-adjust on those days.")

    today = date.today()
    year = st.number_input("Year", min_value=2025, max_value=2030, value=today.year)
    month = st.number_input("Month", min_value=1, max_value=12, value=today.month)

    year = int(year)
    month = int(month)

    df_work = load_work_schedule()
    # Current month existing flags
    month_flags = {}
    monthrange = calendar.monthrange(year, month)[1]

    for day in range(1, monthrange + 1):
        d = date(year, month, day)
        if not df_work.empty:
            is_w = bool(df_work[df_work["date"] == d]["is_work"].any())
        else:
            is_w = False
        month_flags[day] = is_w

    st.subheader(f"{calendar.month_name[month]} {year}")

    # Render as calendar grid
    first_weekday, num_days = calendar.monthrange(year, month)
    # first_weekday: Monday=0
    # Build rows of checkboxes
    days_iter = 1
    for week in range(6):  # max 6 weeks in a month
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
