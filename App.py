import os
import json
from datetime import date, datetime, timedelta
from typing import Optional, Dict, Any, List

import pandas as pd
import streamlit as st

# Optional OpenAI import (app still runs if library is missing)
try:  # pragma: no cover - optional dependency
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore


# -----------------------------
# Paths & constants
# -----------------------------

LOG_FILE = "training_log.csv"
WORK_FILE = "work_schedule.csv"
WEEK_OVERRIDE_FILE = "week_overrides.csv"

PLAN_START = date(2025, 12, 8)  # Phase 2A start (Monday)


# -----------------------------
# Plan & workouts
# -----------------------------

WORKOUTS: Dict[str, Dict[str, Any]] = {
    "Strength A – Upper Focus": {
        "category": "strength",
        "description": "Upper body hypertrophy. Main press (BB/DB), row or pull-up, then accessories for arms and shoulders.",
        "duration_min": 60,
    },
    "Strength B – Lower / ME": {
        "category": "strength",
        "description": "Lower body and muscular endurance. Squat/leg press, hinge (RDL), lunges, and core work.",
        "duration_min": 60,
    },
    "Tempo Run": {
        "category": "cardio",
        "description": "Continuous run at comfortably hard pace (~10K effort). 10 min easy warm-up, 20–25 min tempo, 5–10 min cool-down.",
        "duration_min": 45,
    },
    "Z2 Hill Hike": {
        "category": "cardio",
        "description": "Steady incline hike in mid‑Zone 2. Keep breathing easy and focus on smooth, efficient steps.",
        "duration_min": 60,
    },
    "Z2 Run": {
        "category": "cardio",
        "description": "Easy conversational Zone 2 run. You should be able to talk in full sentences the whole time.",
        "duration_min": 40,
    },
    "Long Z2": {
        "category": "cardio",
        "description": "Longer easy endurance session (run/hike/bike). Stay relaxed and nose‑breathing as much as possible.",
        "duration_min": 75,
    },
    "Rest / Mobility": {
        "category": "rest",
        "description": "No structured training. Optional light walking and 10–20 minutes of easy mobility or foam rolling.",
        "duration_min": 0,
    },
}

PHASES: List[Dict[str, Any]] = []

# Build simple sequential phases for the year
phase_names = [
    "Phase 2A – Strength + Aerobic Base",
    "Phase 2B – Aerobic Emphasis",
    "Phase 2C – Threshold & Power",
    "Phase 3A – Race Specific Build",
    "Phase 3B – Peak & Taper",
]
phase_goals = [
    "Build strength and aerobic base while keeping fatigue manageable. Main focus is consistency.",
    "Add more aerobic volume and quality while maintaining strength from 2A.",
    "Introduce more sustained tempo / threshold work on top of the aerobic base.",
    "Shift toward race‑specific pacing and terrain. Sharpen but don’t bury yourself.",
    "Freshen up, keep the engine primed, and arrive sharp for key events.",
]

base_pattern = {
    0: "Strength A – Upper Focus",
    1: "Tempo Run",
    2: "Z2 Hill Hike",
    3: "Strength B – Lower / ME",
    4: "Z2 Run",
    5: "Long Z2",
    6: "Rest / Mobility",
}

start = PLAN_START
for i, (name, goal) in enumerate(zip(phase_names, phase_goals)):
    # each phase ~6 weeks, except last which we just extend
    if i < len(phase_names) - 1:
        end = start + timedelta(weeks=6) - timedelta(days=1)
    else:
        end = start + timedelta(weeks=12)  # tail of the year, approximate
    PHASES.append(
        {
            "name": name,
            "start": start,
            "end": end,
            "weekly_pattern": base_pattern.copy(),
            "goal": goal,
        }
    )
    start = end + timedelta(days=1)


# -----------------------------
# Data helpers
# -----------------------------

def _ensure_date_col(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def load_logs() -> pd.DataFrame:
    if not os.path.exists(LOG_FILE):
        cols = [
            "date",
            "mode",
            "planned_name",
            "ai_name",
            "cardio_name",
            "cardio_distance_km",
            "cardio_time_min",
            "cardio_tcx_file",
            "strength_name",
            "strength_notes",
            "hrv",
            "sleep_hrs",
            "soreness",
            "energy",
            "notes",
        ]
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(LOG_FILE)
    return _ensure_date_col(df)


def save_log_row(row: Dict[str, Any]) -> None:
    df = load_logs()
    df = df[df["date"] != row["date"]]  # replace if exists
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)


def delete_log_for_date(d: date) -> None:
    if not os.path.exists(LOG_FILE):
        return
    df = load_logs()
    df = df[df["date"] != d]
    df.to_csv(LOG_FILE, index=False)


def load_work_schedule() -> pd.DataFrame:
    if not os.path.exists(WORK_FILE):
        df = pd.DataFrame(columns=["date", "is_work"])
    else:
        df = pd.read_csv(WORK_FILE)
    return _ensure_date_col(df)


def save_work_schedule(df: pd.DataFrame) -> None:
    df.to_csv(WORK_FILE, index=False)


def load_week_overrides() -> pd.DataFrame:
    if not os.path.exists(WEEK_OVERRIDE_FILE):
        df = pd.DataFrame(columns=["date", "override_name"])
    else:
        df = pd.read_csv(WEEK_OVERRIDE_FILE)
    return _ensure_date_col(df)


def save_week_overrides(df: pd.DataFrame) -> None:
    df.to_csv(WEEK_OVERRIDE_FILE, index=False)


# -----------------------------
# Plan helpers
# -----------------------------

def get_phase_for_date(d: date) -> Optional[Dict[str, Any]]:
    for phase in PHASES:
        if phase["start"] <= d <= phase["end"]:
            return phase
    return None


def get_base_planned_workout(d: date) -> Optional[str]:
    phase = get_phase_for_date(d)
    if phase is None:
        return None
    weekday = d.weekday()  # 0=Mon
    return phase["weekly_pattern"].get(weekday)


def get_planned_workout(d: date) -> Optional[str]:
    overrides = load_week_overrides()
    row = overrides[overrides["date"] == d]
    if not row.empty:
        return str(row.iloc[0]["override_name"])
    return get_base_planned_workout(d)


def shuffle_week(start_of_week: date) -> None:
    """
    Randomly shuffle this week's planned workouts and store overrides.
    """
    dates = [start_of_week + timedelta(days=i) for i in range(7)]
    planned = [get_planned_workout(d) or get_base_planned_workout(d) for d in dates]
    # filter out None
    planned_non_none = [p for p in planned if p is not None]
    if not planned_non_none:
        return
    import random

    random.shuffle(planned_non_none)
    # reassign shuffled only to days that had workouts
    overrides = load_week_overrides()
    # drop any overrides for this week
    overrides = overrides[(overrides["date"] < dates[0]) | (overrides["date"] > dates[-1])]

    idx = 0
    new_rows = []
    for d, p in zip(dates, planned):
        if p is None:
            continue
        new_rows.append({"date": d, "override_name": planned_non_none[idx]})
        idx += 1

    if new_rows:
        overrides = pd.concat([overrides, pd.DataFrame(new_rows)], ignore_index=True)
    save_week_overrides(overrides)


# -----------------------------
# OpenAI helpers
# -----------------------------

@st.cache_resource(show_spinner=False)
def get_openai_client() -> Optional[Any]:
    api_key = None
    try:
        if "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


def build_history_summary(logs: pd.DataFrame, days: int = 7) -> str:
    if logs.empty:
        return "No logged training yet."
    recent = logs.sort_values("date").tail(days)
    lines = []
    for _, r in recent.iterrows():
        parts = [str(r["date"])]
        if isinstance(r.get("mode"), str):
            parts.append(f"mode={r['mode']}")
        if isinstance(r.get("cardio_name"), str) and r["cardio_name"]:
            dist = r.get("cardio_distance_km")
            time_min = r.get("cardio_time_min")
            extra = []
            if pd.notna(dist):
                extra.append(f"{dist:.1f} km")
            if pd.notna(time_min):
                extra.append(f"{time_min:.0f} min")
            parts.append("cardio=" + r["cardio_name"] + (" (" + ", ".join(extra) + ")" if extra else ""))
        if isinstance(r.get("strength_name"), str) and r["strength_name"]:
            parts.append("strength=" + r["strength_name"])
        if pd.notna(r.get("hrv")):
            parts.append(f"HRV={r['hrv']}")
        if pd.notna(r.get("sleep_hrs")):
            parts.append(f"sleep={r['sleep_hrs']}h")
        if pd.notna(r.get("soreness")):
            parts.append(f"soreness={r['soreness']}")
        if pd.notna(r.get("energy")):
            parts.append(f"energy={r['energy']}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def ai_suggest_workout_for_day(d: date, logs: pd.DataFrame) -> str:
    client = get_openai_client()
    if client is None:
        return "AI suggestion unavailable: no OpenAI API key or library configured."

    phase = get_phase_for_date(d)
    today_plan = get_planned_workout(d)
    upcoming = []
    for offset in range(1, 4):
        nd = d + timedelta(days=offset)
        p = get_planned_workout(nd)
        if p:
            upcoming.append(f"{nd.isoformat()}: {p}")

    history_summary = build_history_summary(logs, days=10)

    system_msg = (
        "You are a friendly endurance + strength coach. "
        "You write concise, practical suggestions (max ~200 words). "
        "Output only plain text, no markdown lists."
    )
    user_msg = f"""
Today is {d.isoformat()}.
Current phase: {phase['name'] if phase else 'Pre-plan / manual period'}.
Planned workout (if any): {today_plan or 'None / manual'}.

Recent training (most recent last):
{history_summary}

Upcoming planned sessions:
{chr(10).join(upcoming) if upcoming else 'None scheduled.'}

Please propose ONE concrete workout for today, including:
- type (cardio/strength/mixed)
- main focus
- approximate duration
- simple structure (warm-up / main / cool-down)
- how hard it should feel.

Keep it aligned with the phase and don’t bury the athlete if fatigue or soreness look high.
"""
    try:
        resp = client.chat.completions.create(  # type: ignore[attr-defined]
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
            max_tokens=400,
        )
        return resp.choices[0].message.content.strip()  # type: ignore[index]
    except Exception as e:  # pragma: no cover - network
        return f"OpenAI error: {e}"


def ai_coach_answer(prompt: str, context: str) -> str:
    client = get_openai_client()
    if client is None:
        return "AI coach unavailable: no OpenAI API key or library configured."

    system_msg = (
        "You are a pragmatic, encouraging training coach. "
        "Use the provided training context to answer questions. "
        "Be concise, specific, and avoid fluff."
    )
    user_msg = f"Context:\n{context}\n\nQuestion from athlete:\n{prompt}"
    try:
        resp = client.chat.completions.create(  # type: ignore[attr-defined]
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
            max_tokens=400,
        )
        return resp.choices[0].message.content.strip()  # type: ignore[index]
    except Exception as e:  # pragma: no cover - network
        return f"OpenAI error: {e}"


# -----------------------------
# TCX parsing (very light)
# -----------------------------

def parse_tcx_basic(file_bytes: bytes) -> Dict[str, Optional[float]]:
    """
    Very simple TCX parser to extract total distance (km) and time (min).
    Safe to fail – returns None on error.
    """
    try:
        text = file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return {"distance_km": None, "time_min": None}

    import re

    dist_match = re.search(r"<DistanceMeters>([\d\.]+)</DistanceMeters>", text)
    time_match = re.search(r"<TotalTimeSeconds>([\d\.]+)</TotalTimeSeconds>", text)
    dist_km = float(dist_match.group(1)) / 1000.0 if dist_match else None
    time_min = float(time_match.group(1)) / 60.0 if time_match else None
    return {"distance_km": dist_km, "time_min": time_min}


# -----------------------------
# UI helpers
# -----------------------------

def init_session_state() -> None:
    if "selected_date" not in st.session_state:
        st.session_state["selected_date"] = date.today()
    if "page" not in st.session_state:
        st.session_state["page"] = "Today"
    if "ai_today_text" not in st.session_state:
        st.session_state["ai_today_text"] = ""
    if "ai_history_answer" not in st.session_state:
        st.session_state["ai_history_answer"] = ""


def week_monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


# -----------------------------
# Pages
# -----------------------------

def page_today(logs: pd.DataFrame, work_df: pd.DataFrame) -> None:
    d = st.session_state["selected_date"]
    today_real = date.today()

    # Header with date picker
    st.markdown("### Today")
    cols = st.columns([2, 1])
    with cols[0]:
        st.write(d.strftime("%A, %B %d, %Y"))
    with cols[1]:
        new_d = st.date_input(" ", d, key="today_date_picker")
        if new_d != d:
            st.session_state["selected_date"] = new_d
            d = new_d

    # Day type: check work schedule
    work_row = work_df[work_df["date"] == d]
    is_work_day = False
    if not work_row.empty and bool(work_row.iloc[0]["is_work"]):
        is_work_day = True
    st.caption("On‑tour / work day" if is_work_day else "Home / off‑tour day")

    planned_name = get_planned_workout(d)
    phase = get_phase_for_date(d)

    st.markdown("#### Day Mode")
    mode = st.radio(
        "How do you want to handle today?",
        options=["Auto (plan)", "Manual", "Rest", "AI suggestion"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key="today_mode_radio",
    )

    st.markdown("#### Planned Workout")
    if mode == "Rest":
        st.info("Today is set as a **Rest** day. No structured training planned.")
        planned_for_log = "Rest / Mobility"
    elif mode == "Manual":
        if planned_name:
            st.caption(f"Phase plan for today would be: **{planned_name}**")
        st.info("Manual day – choose what you actually did below.")
        planned_for_log = None
    elif mode == "AI suggestion":
        if planned_name:
            st.caption(f"Phase plan for today would be: **{planned_name}**")
        st.info("You can request an AI‑generated session below, and optionally mark it as today's workout.")
        planned_for_log = planned_name
    else:  # Auto
        if planned_name is None:
            st.warning("No structured workout in the plan for this date. Treat as manual or rest.")
        else:
            w = WORKOUTS.get(planned_name, {})
            st.subheader(planned_name)
            st.write(w.get("description", ""))
            dur = w.get("duration_min")
            if dur:
                st.caption(f"Estimated duration: ~{int(dur)} minutes")
        planned_for_log = planned_name

    # --- Manual / actual training blocks ---
    st.markdown("---")
    st.markdown("#### Cardio Session (optional)")

    cardio_name = ""
    cardio_distance_km: Optional[float] = None
    cardio_time_min: Optional[float] = None
    cardio_tcx_file = ""

    if mode in ["Auto (plan)", "AI suggestion", "Manual"]:
        cardio_name = st.text_input("Cardio session name (e.g. Z2 Run, Hill Hike)", key="cardio_name_input")
        tcx_file = st.file_uploader(
            "Upload TCX file (optional)",
            type=None,  # accept any, we will just parse if looks like TCX
            key="cardio_tcx_uploader",
        )
        if tcx_file is not None:
            parsed = parse_tcx_basic(tcx_file.read())
            cardio_distance_km = parsed["distance_km"]
            cardio_time_min = parsed["time_min"]
            cardio_tcx_file = tcx_file.name
            if cardio_distance_km or cardio_time_min:
                txt_bits = []
                if cardio_distance_km:
                    txt_bits.append(f"{cardio_distance_km:.1f} km")
                if cardio_time_min:
                    txt_bits.append(f"{cardio_time_min:.0f} min")
                st.caption("Parsed from TCX: " + ", ".join(txt_bits))
            else:
                st.caption("Could not parse distance/time, but file name will be stored.")

        c_cols = st.columns(2)
        with c_cols[0]:
            manual_dist = st.number_input(
                "Distance (km – override or fill if no TCX)",
                min_value=0.0,
                value=float(cardio_distance_km or 0.0),
                step=0.1,
            )
        with c_cols[1]:
            manual_time = st.number_input(
                "Time (minutes – override or fill if no TCX)",
                min_value=0.0,
                value=float(cardio_time_min or 0.0),
                step=1.0,
            )
        if manual_dist:
            cardio_distance_km = manual_dist
        if manual_time:
            cardio_time_min = manual_time

    st.markdown("#### Strength Session (optional)")
    strength_name = ""
    strength_notes = ""
    if mode in ["Auto (plan)", "AI suggestion", "Manual"]:
        strength_name = st.text_input("Strength session name (e.g. Strength A, upper pump)", key="strength_name_input")
        strength_notes = st.text_area("Key lifts / notes", key="strength_notes_input")

    # --- AI Suggested workout for today ---
    ai_name = ""
    if mode == "AI suggestion":
        st.markdown("---")
        st.markdown("#### AI Suggested Workout")
        if st.button("Generate AI suggestion for today"):
            st.session_state["ai_today_text"] = ai_suggest_workout_for_day(d, logs)
        if st.session_state.get("ai_today_text"):
            st.write(st.session_state["ai_today_text"])
            use_ai = st.checkbox("Use this AI suggestion as today's primary workout name", value=False)
            if use_ai:
                ai_name = "AI‑Suggested Session"
        else:
            st.caption("Press the button above to get an AI‑generated session based on your phase and history.")

    # --- Daily trackers ---
    st.markdown("---")
    st.markdown("#### Daily Trackers")

    tracker_cols = st.columns(2)
    with tracker_cols[0]:
        hrv = st.number_input("HRV (ms)", min_value=0.0, value=0.0, step=1.0, key="hrv_input")
    with tracker_cols[1]:
        sleep_hrs = st.number_input("Sleep (hours)", min_value=0.0, value=0.0, step=0.25, key="sleep_input")

    s_cols = st.columns(2)
    with s_cols[0]:
        soreness = st.slider("Soreness (1–10)", min_value=1, max_value=10, value=3, key="soreness_slider")
    with s_cols[1]:
        energy = st.slider("Energy (1–10)", min_value=1, max_value=10, value=7, key="energy_slider")

    notes = st.text_area("Notes for today", key="notes_input")

    # --- Save / delete ---
    st.markdown("---")
    save_col, delete_col = st.columns(2)
    with save_col:
        if st.button("Save today's log"):
            row = {
                "date": d,
                "mode": mode,
                "planned_name": planned_for_log or "",
                "ai_name": ai_name,
                "cardio_name": cardio_name,
                "cardio_distance_km": cardio_distance_km,
                "cardio_time_min": cardio_time_min,
                "cardio_tcx_file": cardio_tcx_file,
                "strength_name": strength_name,
                "strength_notes": strength_notes,
                "hrv": hrv,
                "sleep_hrs": sleep_hrs,
                "soreness": soreness,
                "energy": energy,
                "notes": notes,
            }
            save_log_row(row)
            st.success("Saved today's log.")

    with delete_col:
        if st.button("Delete this day's log"):
            delete_log_for_date(d)
            st.warning("Deleted any existing log for this date.")

    # --- AI coach at bottom (context-aware about today + recent history) ---
    st.markdown("---")
    st.markdown("#### 🧠 AI Coach (for today)")
    coach_prompt = st.text_input("Ask your coach something about today or this week:", key="coach_today_prompt")
    if st.button("Ask AI Coach (today)") and coach_prompt.strip():
        hist_summary = build_history_summary(logs, days=10)
        context_parts = [
            f"Date: {d.isoformat()}",
            f"Current phase: {phase['name'] if phase else 'None'}",
            f"Planned workout: {planned_for_log or planned_name or 'None'}",
            f"Today work day: {is_work_day}",
            "Recent logs:",
            hist_summary,
        ]
        answer = ai_coach_answer(coach_prompt.strip(), "\n".join(context_parts))
        st.session_state["ai_today_answer"] = answer

    if st.session_state.get("ai_today_answer"):
        st.write(st.session_state["ai_today_answer"])


def page_this_week(logs: pd.DataFrame, work_df: pd.DataFrame) -> None:
    d = st.session_state["selected_date"]
    monday = week_monday(d)
    st.markdown("### This Week")
    nav_cols = st.columns(3)
    with nav_cols[0]:
        if st.button("« Previous week"):
            monday_prev = monday - timedelta(weeks=1)
            st.session_state["selected_date"] = monday_prev
            d = monday_prev
            monday = week_monday(d)
    with nav_cols[1]:
        if st.button("Go to Today"):
            st.session_state["selected_date"] = date.today()
            d = date.today()
            monday = week_monday(d)
    with nav_cols[2]:
        if st.button("Next week »"):
            monday_next = monday + timedelta(weeks=1)
            st.session_state["selected_date"] = monday_next
            d = monday_next
            monday = week_monday(d)

    st.write(f"Week of {monday.isoformat()}")

    if st.button("Shuffle this week's workouts"):
        shuffle_week(monday)
        st.info("Shuffled this week's planned workouts.")

    for offset in range(7):
        day = monday + timedelta(days=offset)
        phase = get_phase_for_date(day)
        planned = get_planned_workout(day)
        st.markdown("---")
        st.markdown(f"**{day.strftime('%a %m/%d')}**")
        if phase:
            st.caption(phase["name"])
        if planned:
            w = WORKOUTS.get(planned, {})
            st.write(planned)
            st.write(w.get("description", ""))
            dur = w.get("duration_min")
            if dur:
                st.caption(f"Estimated: ~{int(dur)} minutes")
        else:
            st.write("No structured workout planned for this day.")


def page_calendar(logs: pd.DataFrame, work_df: pd.DataFrame) -> None:
    st.markdown("### Work Schedule Calendar")

    year = st.number_input("Year", min_value=2024, max_value=2030, value=st.session_state["selected_date"].year)
    month = st.number_input("Month", min_value=1, max_value=12, value=st.session_state["selected_date"].month)

    # build calendar
    first = date(int(year), int(month), 1)
    start_day = first.weekday()  # 0=Mon
    # we will show weeks starting Sunday for iOS vibe: S M T W T F S
    # convert to Sunday index
    sunday_index = (first.weekday() + 1) % 7
    st.markdown("S  M  T  W  T  F  S")

    # Build list of all days in month
    if month == 12:
        next_month_first = date(int(year) + 1, 1, 1)
    else:
        next_month_first = date(int(year), int(month) + 1, 1)
    num_days = (next_month_first - first).days
    days = [first + timedelta(days=i) for i in range(num_days)]

    # We'll render rows of 7 using st.columns
    # Determine offset from Sunday
    offset = (first.weekday() + 1) % 7
    idx = 0
    # ensure work_df indexed by date for quick lookup
    work_df = work_df.copy()
    work_df = _ensure_date_col(work_df)

    # first row
    cols = st.columns(7)
    for i in range(offset):
        cols[i].write(" ")
    col_idx = offset
    while idx < len(days):
        d = days[idx]
        if col_idx == 7:
            cols = st.columns(7)
            col_idx = 0
        is_work = False
        row = work_df[work_df["date"] == d]
        if not row.empty and bool(row.iloc[0]["is_work"]):
            is_work = True
        # style for work day: red fill, else outline
        label = str(d.day)
        button_kw = {"key": f"ws_{d.isoformat()}"}
        if is_work:
            # use label emoji dot to simulate red fill
            label = f"🔴 {d.day}"
        else:
            label = f"{d.day}"
        if cols[col_idx].button(label, **button_kw):
            # toggle work status
            if is_work:
                work_df.loc[work_df["date"] == d, "is_work"] = False
            else:
                if row.empty:
                    work_df = pd.concat(
                        [work_df, pd.DataFrame([{"date": d, "is_work": True}])],
                        ignore_index=True,
                    )
                else:
                    work_df.loc[work_df["date"] == d, "is_work"] = True
            save_work_schedule(work_df)
            st.experimental_rerun()
        col_idx += 1
        idx += 1


def page_phase() -> None:
    st.markdown("### Phase Overview")

    # Determine current phase index based on selected_date
    d = st.session_state["selected_date"]
    current_idx = 0
    for i, ph in enumerate(PHASES):
        if ph["start"] <= d <= ph["end"]:
            current_idx = i
            break

    # Allow manual navigation
    idx = st.number_input("Phase index", min_value=0, max_value=len(PHASES) - 1, value=current_idx, step=1)
    phase = PHASES[int(idx)]
    st.subheader(phase["name"])
    st.caption(f"{phase['start'].isoformat()} → {phase['end'].isoformat()}")
    st.write(phase["goal"])

    st.markdown("**Typical week structure**")
    for wd in range(7):
        day_name = (phase["start"] + timedelta(days=wd)).strftime("%A")
        w = phase["weekly_pattern"].get(wd)
        if w:
            st.write(f"{day_name}: {w}")
        else:
            st.write(f"{day_name}: (no structured session)")


def page_history(logs: pd.DataFrame, work_df: pd.DataFrame) -> None:
    st.markdown("### History & Export")

    if logs.empty:
        st.info("No training logs yet.")
    else:
        logs_sorted = logs.sort_values("date", ascending=False)
        st.dataframe(logs_sorted)

        # delete a day
        dates_str = [d.strftime("%Y-%m-%d") for d in logs_sorted["date"]]
        del_date_str = st.selectbox("Delete log for date:", options=["--"] + dates_str)
        if del_date_str != "--":
            if st.button("Delete selected date"):
                delete_log_for_date(datetime.strptime(del_date_str, "%Y-%m-%d").date())
                st.warning(f"Deleted log for {del_date_str}.")
                st.experimental_rerun()

        # export
        if st.button("Download CSV history"):
            csv = logs_sorted.to_csv(index=False)
            st.download_button(
                "Click to download",
                csv,
                file_name="training_history.csv",
                mime="text/csv",
            )

    # AI coach on history
    st.markdown("---")
    st.markdown("#### 🧠 AI Coach on History")
    hist_prompt = st.text_input("Ask about your overall training history, trends, or adjustments:", key="hist_prompt")
    if st.button("Ask AI Coach (history)") and hist_prompt.strip():
        context = build_history_summary(logs, days=30)
        answer = ai_coach_answer(hist_prompt.strip(), context)
        st.session_state["ai_history_answer"] = answer

    if st.session_state.get("ai_history_answer"):
        st.write(st.session_state["ai_history_answer"])


# -----------------------------
# Main app
# -----------------------------

def main() -> None:
    st.set_page_config(page_title="2025–26 Training Log", layout="centered", initial_sidebar_state="collapsed")
    init_session_state()

    logs = load_logs()
    work_df = load_work_schedule()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    if st.sidebar.button("Today"):
        st.session_state["selected_date"] = date.today()
        st.session_state["page"] = "Today"

    page = st.sidebar.radio(
        "Go to",
        ["Today", "This Week", "Calendar", "Phase", "History"],
        index=["Today", "This Week", "Calendar", "Phase", "History"].index(st.session_state.get("page", "Today")),
        key="nav_radio",
    )
    st.session_state["page"] = page

    if page == "Today":
        page_today(logs, work_df)
    elif page == "This Week":
        page_this_week(logs, work_df)
    elif page == "Calendar":
        page_calendar(logs, work_df)
    elif page == "Phase":
        page_phase()
    elif page == "History":
        page_history(logs, work_df)


if __name__ == "__main__":
    main()
