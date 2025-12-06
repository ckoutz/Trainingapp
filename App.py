import streamlit as st
import pandas as pd
from datetime import date, datetime

# ------------------------
# CONFIG
# ------------------------

LOG_FILE = "training_log.csv"

CARDIO_MODES = ["Run", "Bike", "Incline Walk"]

BENCH_ALTS = ["Dumbbell Bench", "Floor Press", "Push-Ups", "Machine/Smith Bench"]
PULLUP_ALTS = ["Lat Pulldown", "Assisted Pull-Up", "Inverted Row"]
INCLINE_PRESS_ALTS = ["DB Incline Press", "Machine Incline", "Pike Push-Ups"]
ROW_ALTS = ["DB Row", "Cable Row", "Machine Row"]
LATERAL_ALTS = ["DB Lateral Raise", "Cable Lateral Raise", "Plate Raise"]
TRICEPS_ALTS = ["Rope Pushdown", "DB OH Extension", "Close-Grip Pushup"]

DEADLIFT_ALTS = ["KB Deadlift", "DB RDL", "Back Extensions"]
SPLIT_SQUAT_ALTS = ["DB Lunge", "Bulgarian Split Squat", "Step-Ups"]
HIPTHRUST_ALTS = ["DB Hip Thrust", "Glute Bridge", "Single-Leg Hip Bridge"]
BICEPS_ALTS = ["DB Curl", "Hammer Curl", "Band Curl"]
CORE_ALTS = ["Plank", "Side Plank", "Dead Bug"]

ME_ALTS = ["Step-Ups", "StairStepper", "Incline Grind (sustained)", "Ruck Walk", "Box Step Ladder"]


# ------------------------
# TRAINING LOGIC
# ------------------------

def get_phase_and_day_plan(d: date):
    """
    Returns (phase, day_type, planned, kind)
    kind is used to drive UI logic: StrengthA, StrengthB, Tempo, Hill, FlexCardio, LongZ2, Incline, ME, Off
    """

    dt = datetime(d.year, d.month, d.day)

    # Our programmed window: Dec 8, 2024 – Jan 31, 2025
    start_2a = datetime(2024, 12, 8)
    end_2a = datetime(2025, 1, 12)
    start_2b = datetime(2025, 1, 13)
    end_2b = datetime(2025, 1, 31)

    if dt < start_2a or dt > end_2b:
        return (
            "Unprogrammed",
            "Manual / Free Day",
            "No structured plan for this date in the Dec–Jan block. Use this as a free day or enter a manual workout.",
            "Manual",
        )

    # Phase 2A – Strength + Aerobic Base
    if start_2a <= dt <= end_2a:
        phase = "Phase 2A – Strength + Aerobic Base"
        dow = dt.weekday()  # Mon=0
        if dow == 0:
            day_type = "Strength A – Upper Focus"
            planned = "Bench, Pull-Ups, Incline Press, Row, Delts, Triceps. Optional 10-min very easy jog."
            kind = "StrengthA"
        elif dow == 1:
            day_type = "Aerobic Tempo Run"
            planned = "20–30 min steady aerobic run (high Z2 / low Z3). Cardio mode: Run/Bike/Incline."
            kind = "Tempo"
        elif dow == 2:
            day_type = "Z2 Hill Hike"
            planned = "45–75 min Z2 hill hiking. Cardio mode: Hike/Incline/Run/Bike."
            kind = "Hill"
        elif dow == 3:
            day_type = "Strength B – Lower/Hypertrophy"
            planned = "Deadlift/RDL, Split Squat or Lunge, Hip Thrust, Row, Biceps, Core."
            kind = "StrengthB"
        elif dow == 4:
            day_type = "Aerobic Flex Day"
            planned = "Easy 25–35 min Z1–Z2. Choose Run/Bike/Incline based on fatigue."
            kind = "FlexCardio"
        elif dow == 5:
            day_type = "Long Z2 Endurance"
            planned = "60–90 min continuous Z2. Prefer Bike or Long Hike; Run only if calves feel great."
            kind = "LongZ2"
        else:
            day_type = "Off / Recovery"
            planned = "Rest, light mobility, optional easy walk."
            kind = "Off"

    # Phase 2B – Muscular Endurance
    elif start_2b <= dt <= end_2b:
        phase = "Phase 2B – Muscular Endurance"
        dow = dt.weekday()
        if dow == 0:
            day_type = "Z2 Incline"
            planned = "40–60 min Z2 incline walk. Cardio mode: Run/Bike/Incline."
            kind = "Incline"
        elif dow == 1:
            day_type = "Strength A – Upper Focus"
            planned = "Bench, Pull-Ups, Incline Press, Row, Delts, Triceps."
            kind = "StrengthA"
        elif dow == 2:
            day_type = "ME Step-Ups + Easy Run"
            planned = "3×6 min step-ups with pack (2 min rest) + 20–25 min very easy Run/Bike/Incline."
            kind = "ME"
        elif dow == 3:
            day_type = "Z2 Incline"
            planned = "40–60 min Z2 incline walk. Cardio mode: Run/Bike/Incline."
            kind = "Incline"
        elif dow == 4:
            day_type = "Strength B – Lower (Light)"
            planned = "Light Deadlift/RDL, Carries, Rear Delts, Biceps, Core."
            kind = "StrengthB"
        elif dow == 5:
            day_type = "Long Incline / Hike"
            planned = "60–90 min Z2 long incline or hike. Cardio mode: Hike/Incline/Bike."
            kind = "LongZ2"
        else:
            day_type = "Off / Recovery"
            planned = "Rest, light mobility, optional easy walk."
            kind = "Off"
    else:
        # Fallback – should not hit
        phase = "Unprogrammed"
        day_type = "Manual / Free Day"
        planned = "No structured plan."
        kind = "Manual"

    return phase, day_type, planned, kind


def get_strength_exercises(kind: str):
    """
    Returns a list of dicts for strength exercises:
    { name, alt_list }
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
# DATA HANDLING
# ------------------------

def load_log():
    try:
        df = pd.read_csv(LOG_FILE, parse_dates=["date"])
    except FileNotFoundError:
        df = pd.DataFrame()
    return df


def save_log_row(row: dict):
    df = load_log()
    new_row = pd.DataFrame([row])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)


# ------------------------
# UI
# ------------------------

st.set_page_config(page_title="Dec–Jan Training Log", layout="centered")

page = st.sidebar.radio("Page", ["Today", "History"])

if page == "Today":
    st.title("Dec–Jan Training Log")

    # Date selector (defaults to today)
    today = date.today()
    selected_date = st.date_input("Date", value=today)

    phase, day_type, planned, kind = get_phase_and_day_plan(selected_date)

    st.subheader(f"{phase}")
    st.caption(f"Day Type: {day_type}")
    st.write(f"**Planned Workout:** {planned}")

    st.markdown("---")

    log_data = {
        "date": selected_date,
        "phase": phase,
        "day_type": day_type,
        "kind": kind,
    }

    # Cardio block (where relevant)
    if kind in ["Tempo", "Hill", "FlexCardio", "LongZ2", "Incline", "ME"]:
        st.markdown("### Cardio")
        cardio_mode = st.selectbox("Cardio mode", CARDIO_MODES, index=0)
        cardio_duration = st.number_input("Cardio duration (min)", min_value=0, max_value=300, value=0)
        cardio_distance = st.text_input("Cardio distance (mi/km)", value="")
        cardio_avg_hr = st.text_input("Cardio avg HR (bpm)", value="")
        log_data.update(
            {
                "cardio_mode": cardio_mode,
                "cardio_duration_min": cardio_duration,
                "cardio_distance": cardio_distance,
                "cardio_avg_hr": cardio_avg_hr,
            }
        )
    else:
        log_data.update(
            {
                "cardio_mode": "",
                "cardio_duration_min": 0,
                "cardio_distance": "",
                "cardio_avg_hr": "",
            }
        )

    st.markdown("---")

    # Strength / ME block
    exercises = get_strength_exercises(kind)
    strength_entries = []

    if exercises:
        st.markdown("### Strength / ME Session")
        for i, ex in enumerate(exercises):
            st.markdown(f"**{ex['name']}**")
            st.caption("Alts: " + ", ".join(ex["alts"]))

            col1, col2, col3, col4 = st.columns(4)
            sets = col1.text_input(f"Sets ({ex['name']})", key=f"{selected_date}_sets_{i}")
            reps = col2.text_input(f"Reps ({ex['name']})", key=f"{selected_date}_reps_{i}")
            weight = col3.text_input(f"Weight ({ex['name']})", key=f"{selected_date}_wt_{i}")
            rpe = col4.text_input(f"RPE ({ex['name']})", key=f"{selected_date}_rpe_{i}")

            strength_entries.append(
                {
                    "exercise": ex["name"],
                    "sets": sets,
                    "reps": reps,
                    "weight": weight,
                    "rpe": rpe,
                    "alts": ", ".join(ex["alts"]),
                }
            )

        # Flatten strength data a bit for storage (we'll store as JSON-ish string)
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
    # Leave colF blank or use for something else later

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

elif page == "History":
    st.title("Training History")

    df = load_log()
    if df.empty:
        st.info("No logs saved yet.")
    else:
        # Sort by date descending
        df = df.sort_values("date", ascending=False)

        st.dataframe(
            df[["date", "phase", "day_type", "cardio_mode", "cardio_duration_min", "hrv", "sleep_hours", "mood_1_5", "soreness_1_5", "energy_1_5"]],
            use_container_width=True,
        )

        # Optionally select a single day to inspect
        st.markdown("### Inspect single day")
        dates_available = df["date"].dt.date.unique()
        selected_hist_date = st.selectbox("Select date", dates_available)
        day_df = df[df["date"].dt.date == selected_hist_date]

        st.write(day_df.T)
