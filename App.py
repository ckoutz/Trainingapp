import os
import json
from datetime import date, datetime, timedelta
from typing import Optional, Dict, Any, List

import pandas as pd
import streamlit as st

# Optional OpenAI import
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore


LOG_FILE = "training_log.csv"
WORK_FILE = "work_schedule.csv"
WEEK_OVERRIDE_FILE = "week_overrides.csv"

PLAN_START = date(2025, 12, 8)


# -----------------------------
# Plan & workout library
# -----------------------------

WORKOUTS: Dict[str, Dict[str, Any]] = {
    # Strength
    "Strength A – Upper Focus": {
        "category": "strength",
        "description": "Upper body hypertrophy: bench/press, rows/pull-ups, shoulders, and arms.",
        "duration_min": 60,
    },
    "Strength B – Lower / ME": {
        "category": "strength",
        "description": "Lower-body strength + muscular endurance: squat/leg press, hinge, lunges, glutes, hamstrings, and ME finisher.",
        "duration_min": 60,
    },
    "Full Body Strength": {
        "category": "strength",
        "description": "Full-body strength: hinge, push, row, squat or lunge, shoulders, and core.",
        "duration_min": 60,
    },

    # Cardio – Z2 / endurance
    "Z2 – Outdoor Run": {
        "category": "cardio",
        "description": "Easy outdoor run in mid-Zone 2. Conversational pace, relaxed breathing.",
        "duration_min": 40,
    },
    "Z2 – Treadmill": {
        "category": "cardio",
        "description": "Zone 2 treadmill run. Adjust incline slightly to mimic outdoor loading.",
        "duration_min": 40,
    },
    "Z2 – Trail Run": {
        "category": "cardio",
        "description": "Easy trail run. Focus on footing and staying relaxed on climbs.",
        "duration_min": 45,
    },
    "Z2 – Incline Treadmill Walk": {
        "category": "cardio",
        "description": "Incline treadmill walk in Zone 2. Great for uphill strength without pounding.",
        "duration_min": 45,
    },
    "Z2 – Hike": {
        "category": "cardio",
        "description": "Outdoor Z2 hike. Use poles if needed, stay in control on the climbs.",
        "duration_min": 60,
    },
    "Z2 – Ruck": {
        "category": "cardio",
        "description": "Weighted pack ruck in Zone 2. Think durable, steady effort.",
        "duration_min": 60,
    },
    "Z2 – Bike Outdoor": {
        "category": "cardio",
        "description": "Outdoor Z2 ride. Smooth cadence, low RPE, keep HR stable.",
        "duration_min": 75,
    },
    "Z2 – Bike Indoor": {
        "category": "cardio",
        "description": "Indoor bike Z2 spin. Perfect for easy days with low impact.",
        "duration_min": 60,
    },

    # Tempo / long
    "Tempo Run": {
        "category": "cardio",
        "description": "Continuous tempo: 10 min easy, 20–25 min comfortably hard, 5–10 min cool-down.",
        "duration_min": 45,
    },
    "Long Z2 – Run/Trail": {
        "category": "cardio",
        "description": "Long Z2 run or trail run. Fuel/hydrate like a long event.",
        "duration_min": 75,
    },
    "Long Z2 – Bike": {
        "category": "cardio",
        "description": "Long easy Z2 ride. Stay relaxed and smooth, nothing heroic.",
        "duration_min": 90,
    },

    # Rest
    "Rest / Mobility": {
        "category": "rest",
        "description": "Full rest from structured training. Optional light walking and mobility only.",
        "duration_min": 0,
    },
}

PHASES: List[Dict[str, Any]] = []

phase_names = [
    "Phase 2A – Strength + Aerobic Base",
    "Phase 2B – Aerobic Emphasis",
]

phase_goals = [
    "Build strength and aerobic base while keeping fatigue manageable. Focus on consistent lifting + Z2.",
    "Add a bit more aerobic quality and long Z2 while maintaining strength from 2A.",
]

weekly_pattern_2A = {
    0: "Strength A – Upper Focus",
    1: "Tempo Run",
    2: "Z2 – Hike",
    3: "Strength B – Lower / ME",
    4: "Z2 – Outdoor Run",
    5: "Long Z2 – Run/Trail",
    6: "Rest / Mobility",
}

weekly_pattern_2B = {
    0: "Strength A – Upper Focus",
    1: "Tempo Run",
    2: "Z2 – Incline Treadmill Walk",
    3: "Strength B – Lower / ME",
    4: "Z2 – Bike Indoor",
    5: "Long Z2 – Bike",
    6: "Rest / Mobility",
}

start = PLAN_START
for i, (name, goal) in enumerate(zip(phase_names, phase_goals)):
    weekly_pattern = weekly_pattern_2A if i == 0 else weekly_pattern_2B
    end = start + timedelta(weeks=6) - timedelta(days=1)
    PHASES.append(
        {
            "name": name,
            "start": start,
            "end": end,
            "weekly_pattern": weekly_pattern,
            "goal": goal,
        }
    )
    start = end + timedelta(days=1)


# -----------------------------
# Data helpers
# -----------------------------

def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def load_logs() -> pd.DataFrame:
    if not os.path.exists(LOG_FILE):
        cols = [
            "date",
            "mode",
            "planned_name",
            "ai_title",
            "strength_name",
            "strength_log",
            "cardio_name",
            "cardio_variant",
            "cardio_log",
            "hrv",
            "sleep_hrs",
            "soreness",
            "energy",
            "notes",
        ]
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(LOG_FILE)
    return _ensure_date(df)


def save_log_row(row: Dict[str, Any]) -> None:
    df = load_logs()
    df = df[df["date"] != row["date"]]
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
    return _ensure_date(df)


def save_work_schedule(df: pd.DataFrame) -> None:
    df.to_csv(WORK_FILE, index=False)


def load_week_overrides() -> pd.DataFrame:
    if not os.path.exists(WEEK_OVERRIDE_FILE):
        df = pd.DataFrame(columns=["date", "override_name"])
    else:
        df = pd.read_csv(WEEK_OVERRIDE_FILE)
    return _ensure_date(df)


def save_week_overrides(df: pd.DataFrame) -> None:
    df.to_csv(WEEK_OVERRIDE_FILE, index=False)


# -----------------------------
# Plan helpers
# -----------------------------

def get_phase_for_date(d: date) -> Optional[Dict[str, Any]]:
    for ph in PHASES:
        if ph["start"] <= d <= ph["end"]:
            return ph
    return None


def base_planned_name_for_date(d: date) -> Optional[str]:
    phase = get_phase_for_date(d)
    if not phase:
        return None
    return phase["weekly_pattern"].get(d.weekday())


def get_planned_name_for_date(d: date) -> Optional[str]:
    overrides = load_week_overrides()
    row = overrides[overrides["date"] == d]
    if not row.empty:
        return str(row.iloc[0]["override_name"])
    return base_planned_name_for_date(d)


def swap_two_days_in_week(monday: date, day_a: int, day_b: int) -> None:
    """Swap the planned workouts for two days in this week (0–6)."""
    overrides = load_week_overrides()
    dates = [monday + timedelta(days=i) for i in range(7)]

    # current planned list
    planned: List[Optional[str]] = []
    for d in dates:
        planned.append(get_planned_name_for_date(d))

    if planned[day_a] is None and planned[day_b] is None:
        return

    planned[day_a], planned[day_b] = planned[day_b], planned[day_a]

    # wipe overrides for this week
    overrides = overrides[(overrides["date"] < dates[0]) | (overrides["date"] > dates[-1])]
    new_rows = []
    for d, name in zip(dates, planned):
        if name is not None:
            new_rows.append({"date": d, "override_name": name})
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
        api_key = st.secrets.get("OPENAI_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        pass
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


def build_history_summary(logs: pd.DataFrame, days: int = 10) -> str:
    if logs.empty:
        return "No logged training yet."
    recent = logs.sort_values("date").tail(days)
    lines = []
    for _, r in recent.iterrows():
        parts = [str(r["date"])]
        if isinstance(r.get("mode"), str):
            parts.append(f"mode={r['mode']}")
        if isinstance(r.get("cardio_name"), str) and r["cardio_name"]:
            parts.append(f"cardio={r['cardio_name']}")
        if isinstance(r.get("strength_name"), str) and r["strength_name"]:
            parts.append(f"strength={r['strength_name']}")
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
    today_plan = get_planned_name_for_date(d)
    upcoming = []
    for offset in range(1, 4):
        nd = d + timedelta(days=offset)
        p = get_planned_name_for_date(nd)
        if p:
            upcoming.append(f"{nd.isoformat()}: {p}")

    history_summary = build_history_summary(logs, days=10)

    system_msg = (
        "You are a hybrid endurance + strength coach. "
        "You write concise, practical workouts as plain text (no bullet lists)."
    )
    user_msg = f"""
Today is {d.isoformat()}.
Current phase: {phase['name'] if phase else 'Pre-plan/manual'}.
Planned workout (if any): {today_plan or 'None / manual'}.

Recent training (most recent last):
{history_summary}

Upcoming planned sessions:
{chr(10).join(upcoming) if upcoming else 'None scheduled.'}

Propose ONE concrete workout for today, aligned with the phase and planned load:
- what type (strength/cardio/mixed)
- main focus
- rough duration
- simple structure (warm-up / main / cool-down)
- how hard it should feel (RPE).

Keep it under about 200 words.
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
    except Exception as e:
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
    except Exception as e:
        return f"OpenAI error: {e}"


# -----------------------------
# TCX parsing
# -----------------------------

def parse_tcx(file_bytes: bytes) -> Dict[str, Optional[float]]:
    """
    Light TCX parser: distance, time, HR, elevation (if present).
    """
    try:
        text = file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return {
            "distance_km": None,
            "time_min": None,
            "hr_avg": None,
            "hr_max": None,
            "elev_gain": None,
        }

    import re

    def first_float(pattern: str) -> Optional[float]:
        m = re.search(pattern, text)
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None

    dist_m = first_float(r"<DistanceMeters>([\d\.]+)</DistanceMeters>")
    time_s = first_float(r"<TotalTimeSeconds>([\d\.]+)</TotalTimeSeconds>")
    hr_avg = first_float(r"<AverageHeartRateBpm>.*?<Value>(\d+)</Value>")
    hr_max = first_float(r"<MaximumHeartRateBpm>.*?<Value>(\d+)</Value>")
    elev_gain = first_float(r"<TotalAscent>([\d\.]+)</TotalAscent>")

    distance_km = dist_m / 1000.0 if dist_m is not None else None
    time_min = time_s / 60.0 if time_s is not None else None

    return {
        "distance_km": distance_km,
        "time_min": time_min,
        "hr_avg": hr_avg,
        "hr_max": hr_max,
        "elev_gain": elev_gain,
    }


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
    if "ai_today_answer" not in st.session_state:
        st.session_state["ai_today_answer"] = ""
    if "ai_history_answer" not in st.session_state:
        st.session_state["ai_history_answer"] = ""


def week_monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def cardio_options() -> List[str]:
    return [k for k, v in WORKOUTS.items() if v["category"] == "cardio"]


def strength_options() -> List[str]:
    return [k for k, v in WORKOUTS.items() if v["category"] == "strength"]


# -----------------------------
# Pages
# -----------------------------

def page_today(logs: pd.DataFrame, work_df: pd.DataFrame) -> None:
    d: date = st.session_state["selected_date"]

    st.markdown("### Today")
    header_cols = st.columns([2, 1])
    with header_cols[0]:
        st.write(d.strftime("%A, %B %d, %Y"))
    with header_cols[1]:
        new_d = st.date_input("Go to date", value=d, key="today_date_picker")
        if new_d != d:
            d = new_d
            st.session_state["selected_date"] = d

    work_row = work_df[work_df["date"] == d]
    is_work_day = (not work_row.empty) and bool(work_row.iloc[0]["is_work"])
    st.caption("On-tour / work day" if is_work_day else "Home / off-tour day")

    planned_name = get_planned_name_for_date(d)
    phase = get_phase_for_date(d)

    st.markdown("#### Mode")
    mode = st.radio(
        "Mode",
        options=["Auto (plan)", "Manual", "Rest", "AI suggestion"],
        horizontal=True,
        label_visibility="collapsed",
        key="today_mode",
    )

    st.markdown("#### Planned Workout")

    planned_for_log: Optional[str] = None
    if mode == "Rest":
        st.info("Rest day selected. No structured training planned.")
        planned_for_log = "Rest / Mobility"
    elif mode == "Manual":
        if planned_name:
            st.caption(f"Phase plan for today would be: **{planned_name}**")
        st.info("Manual day – use the strength/cardio add buttons below.")
    elif mode == "AI suggestion":
        if planned_name:
            st.caption(f"Phase plan for today would be: **{planned_name}**")
        st.info("Use AI to propose a session, then log what you actually did.")
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
                st.caption(f"Estimated duration: ~{int(dur)} min")
        planned_for_log = planned_name

    # ----- Strength block -----
    st.markdown("---")
    st.markdown("#### Strength Session")

    strength_name = ""
    strength_log: Dict[str, Any] = {}

    show_strength = mode in ["Auto (plan)", "AI suggestion", "Manual"]
    if show_strength:
        if mode == "Manual":
            add_strength = st.checkbox("Add strength session today?", value=False, key="add_strength_cb")
            if add_strength:
                strength_name = st.selectbox("Strength workout", [""] + strength_options(), key="strength_sel")
        else:
            if planned_name in strength_options():
                strength_name = planned_name
            else:
                add_strength2 = st.checkbox("Add strength session today?", value=False, key="add_strength2_cb")
                if add_strength2:
                    strength_name = st.selectbox("Strength workout", [""] + strength_options(), key="strength_sel_auto")

        if strength_name:
            w = WORKOUTS.get(strength_name, {})
            st.write(f"**{strength_name}**")
            st.write(w.get("description", ""))
            st.caption("Log what you actually did:")
            s_cols = st.columns(3)
            with s_cols[0]:
                sets = st.number_input("Total working sets", min_value=0, step=1, key="str_sets")
            with s_cols[1]:
                top_weight = st.number_input("Top set weight (lbs/kg)", min_value=0.0, step=1.0, key="str_weight")
            with s_cols[2]:
                rpe = st.slider("Overall RPE", min_value=1, max_value=10, value=7, key="str_rpe")
            details = st.text_area("Key lifts / rep schemes (e.g., 4x6 bench @ 155, 3x10 row)", key="str_details")
            strength_log = {
                "sets": sets,
                "top_weight": top_weight,
                "rpe": rpe,
                "details": details,
            }
        else:
            st.caption("No strength session logged for today.")

    # ----- Cardio block -----
    st.markdown("---")
    st.markdown("#### Cardio Session")

    cardio_name = ""
    cardio_variant = ""
    cardio_log: Dict[str, Any] = {}

    show_cardio = mode in ["Auto (plan)", "AI suggestion", "Manual"]
    if show_cardio:
        if mode == "Manual":
            add_cardio = st.checkbox("Add cardio session today?", value=False, key="add_cardio_cb")
            if add_cardio:
                cardio_name = st.selectbox("Cardio workout", [""] + cardio_options(), key="cardio_sel")
        else:
            if planned_name in cardio_options():
                cardio_name = planned_name
            else:
                add_cardio2 = st.checkbox("Add cardio session today?", value=False, key="add_cardio2_cb")
                if add_cardio2:
                    cardio_name = st.selectbox("Cardio workout", [""] + cardio_options(), key="cardio_sel_auto")

        if cardio_name:
            w = WORKOUTS.get(cardio_name, {})
            st.write(f"**{cardio_name}**")
            st.write(w.get("description", ""))
            dur = w.get("duration_min")
            if dur:
                st.caption(f"Planned duration: ~{int(dur)} min")

            cardio_variant = st.text_input(
                "Cardio variant label (e.g. 'Trail', 'Treadmill 2%', 'Ruck 30lb')",
                key="cardio_variant",
            )

            st.caption("Log what you actually did:")
            c_cols = st.columns(2)
            with c_cols[0]:
                dist_km = st.number_input("Distance (km)", min_value=0.0, step=0.1, key="cardio_dist")
            with c_cols[1]:
                time_min = st.number_input("Time (min)", min_value=0.0, step=1.0, key="cardio_time")

            tcx_file = st.file_uploader("Upload TCX (optional)", type=None, key="tcx_uploader")
            tcx_summary: Dict[str, Any] = {}
            if tcx_file is not None:
                parsed = parse_tcx(tcx_file.read())
                tcx_summary = parsed
                bits = []
                if parsed["distance_km"]:
                    bits.append(f"{parsed['distance_km']:.1f} km")
                if parsed["time_min"]:
                    bits.append(f"{parsed['time_min']:.0f} min")
                if parsed["hr_avg"]:
                    bits.append(f"HR avg {parsed['hr_avg']:.0f}")
                if parsed["elev_gain"]:
                    bits.append(f"+{parsed['elev_gain']:.0f} m")
                if bits:
                    st.caption("From TCX: " + ", ".join(bits))
                if dist_km == 0.0 and parsed["distance_km"] is not None:
                    dist_km = parsed["distance_km"]
                if time_min == 0.0 and parsed["time_min"] is not None:
                    time_min = parsed["time_min"]

            cardio_log = {
                "distance_km": dist_km,
                "time_min": time_min,
                "variant": cardio_variant,
                "tcx_summary": tcx_summary,
            }
        else:
            st.caption("No cardio session logged for today.")

    # ----- AI Suggested Workout (text) -----
    ai_title = ""
    if mode == "AI suggestion":
        st.markdown("---")
        st.markdown("#### AI Suggested Workout")
        if st.button("Generate AI suggestion for today"):
            st.session_state["ai_today_text"] = ai_suggest_workout_for_day(d, logs)
        if st.session_state.get("ai_today_text"):
            st.write(st.session_state["ai_today_text"])
            use_ai = st.checkbox("Mark this as today's primary AI session", value=False)
            if use_ai:
                ai_title = "AI-Suggested Session"
        else:
            st.caption("Press the button above to get an AI-generated idea.")

    # ----- Daily trackers -----
    st.markdown("---")
    st.markdown("#### Daily Trackers")

    t_cols = st.columns(2)
    with t_cols[0]:
        hrv = st.number_input("HRV (ms)", min_value=0.0, step=1.0, key="hrv_input")
    with t_cols[1]:
        sleep_hrs = st.number_input("Sleep (hours)", min_value=0.0, step=0.25, key="sleep_input")

    s_cols = st.columns(2)
    with s_cols[0]:
        soreness = st.slider("Soreness (1–10)", min_value=1, max_value=10, value=3, key="soreness_slider")
    with s_cols[1]:
        energy = st.slider("Energy (1–10)", min_value=1, max_value=10, value=7, key="energy_slider")

    notes = st.text_area("Notes for today", key="notes_input")

    # ----- Save / delete -----
    st.markdown("---")
    save_col, delete_col = st.columns(2)
    with save_col:
        if st.button("Save today's log"):
            row = {
                "date": d,
                "mode": mode,
                "planned_name": planned_for_log or "",
                "ai_title": ai_title,
                "strength_name": strength_name,
                "strength_log": json.dumps(strength_log) if strength_log else "",
                "cardio_name": cardio_name,
                "cardio_variant": cardio_variant,
                "cardio_log": json.dumps(cardio_log) if cardio_log else "",
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

    # ----- AI Coach (today) -----
    st.markdown("---")
    st.markdown("#### 🧠 AI Coach – Today")
    coach_prompt = st.text_input("Ask your coach about today or this week:", key="coach_today_prompt")
    if st.button("Ask AI Coach (today)") and coach_prompt.strip():
        ctx = [
            f"Date: {d.isoformat()}",
            f"Phase: {phase['name'] if phase else 'None'}",
            f"Planned: {planned_name}",
            f"Mode: {mode}",
            f"Work day: {is_work_day}",
            "",
            "Recent training:",
            build_history_summary(logs, days=10),
        ]
        ans = ai_coach_answer(coach_prompt.strip(), "\n".join(ctx))
        st.session_state["ai_today_answer"] = ans

    if st.session_state.get("ai_today_answer"):
        st.write(st.session_state["ai_today_answer"])


def page_this_week(logs: pd.DataFrame, work_df: pd.DataFrame) -> None:
    st.markdown("### This Week")

    d = st.session_state["selected_date"]
    monday = week_monday(d)

    nav_cols = st.columns(3)
    with nav_cols[0]:
        if st.button("« Previous week"):
            monday = monday - timedelta(weeks=1)
            st.session_state["selected_date"] = monday
    with nav_cols[1]:
        if st.button("Go to current week"):
            today = date.today()
            monday = week_monday(today)
            st.session_state["selected_date"] = today
    with nav_cols[2]:
        if st.button("Next week »"):
            monday = monday + timedelta(weeks=1)
            st.session_state["selected_date"] = monday

    st.caption(f"Week of {monday.isoformat()}")

    # Swap UI instead of random shuffle
    st.markdown("#### Swap two days")
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    swap_cols = st.columns(3)
    with swap_cols[0]:
        day_a_label = st.selectbox("Day A", day_labels, index=0, key="swap_day_a")
    with swap_cols[1]:
        day_b_label = st.selectbox("Day B", day_labels, index=1, key="swap_day_b")
    with swap_cols[2]:
        if st.button("Swap workouts"):
            idx_a = day_labels.index(day_a_label)
            idx_b = day_labels.index(day_b_label)
            swap_two_days_in_week(monday, idx_a, idx_b)
            st.success(f"Swapped workouts for {day_a_label} and {day_b_label}.")

    st.markdown("---")
    for i in range(7):
        day = monday + timedelta(days=i)
        ph = get_phase_for_date(day)
        planned = get_planned_name_for_date(day)
        st.markdown(f"**{day.strftime('%a %m/%d')}**")
        if ph:
            st.caption(ph["name"])
        if planned:
            w = WORKOUTS.get(planned, {})
            st.write(planned)
            st.write(w.get("description", ""))
            dur = w.get("duration_min")
            if dur:
                st.caption(f"Estimated: ~{int(dur)} min")
        else:
            st.write("No structured workout planned.")
        st.markdown("---")


def page_calendar(logs: pd.DataFrame, work_df: pd.DataFrame) -> None:
    st.markdown("### Work Schedule Calendar")

    cur = st.session_state["selected_date"]
    year = st.number_input("Year", min_value=2024, max_value=2030, value=cur.year, step=1)
    month = st.number_input("Month", min_value=1, max_value=12, value=cur.month, step=1)

    first = date(int(year), int(month), 1)
    if month == 12:
        next_first = date(int(year) + 1, 1, 1)
    else:
        next_first = date(int(year), int(month) + 1, 1)
    num_days = (next_first - first).days
    days = [first + timedelta(days=i) for i in range(num_days)]

    st.markdown("**S   M   T   W   T   F   S**")

    # offset: Sunday=0
    offset = (first.weekday() + 1) % 7

    work_df = _ensure_date(work_df.copy())

    idx = 0
    cols = st.columns(7)
    for _ in range(offset):
        cols[_].write(" ")
    col_idx = offset

    for d in days:
        if col_idx == 7:
            cols = st.columns(7)
            col_idx = 0

        row = work_df[work_df["date"] == d]
        is_work = (not row.empty) and bool(row.iloc[0]["is_work"])

        # calendar-style box
        label_html = f"""
        <div style="border-radius:6px; padding:4px 0; text-align:center;
                    background-color:{'#ff4b4b' if is_work else 'transparent'};
                    color:{'white' if is_work else 'inherit'};
                    border:1px solid #555; min-width:28px;">
            {d.day}
        </div>
        """

        if cols[col_idx].button(str(d), key=f"cal_btn_{d.isoformat()}"):
            # toggle work flag
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
            st.session_state["selected_date"] = d

        cols[col_idx].markdown(label_html, unsafe_allow_html=True)
        col_idx += 1
        idx += 1


def page_phase() -> None:
    st.markdown("### Phase Overview")

    d = st.session_state["selected_date"]
    current_idx = 0
    for i, ph in enumerate(PHASES):
        if ph["start"] <= d <= ph["end"]:
            current_idx = i
            break

    if "phase_idx" not in st.session_state:
        st.session_state["phase_idx"] = current_idx

    header_cols = st.columns([1, 3, 1])
    with header_cols[0]:
        if st.button("◀", key="phase_prev"):
            st.session_state["phase_idx"] = max(0, st.session_state["phase_idx"] - 1)
    with header_cols[2]:
        if st.button("▶", key="phase_next"):
            st.session_state["phase_idx"] = min(len(PHASES) - 1, st.session_state["phase_idx"] + 1)

    phase = PHASES[st.session_state["phase_idx"]]
    st.subheader(phase["name"])
    st.caption(f"{phase['start'].isoformat()} → {phase['end'].isoformat()}")
    st.write(phase["goal"])

    st.markdown("**Typical week structure**")
    for wd in range(7):
        day_name = (phase["start"] + timedelta(days=wd)).strftime("%A")
        w = phase["weekly_pattern"].get(wd)
        st.write(f"{day_name}: {w or '(no structured session)'}")


def page_history(logs: pd.DataFrame, work_df: pd.DataFrame) -> None:
    st.markdown("### History & Export")

    logs = load_logs()  # refresh from disk
    if logs.empty:
        st.info("No training logs yet.")
    else:
        logs_sorted = logs.sort_values("date", ascending=False)
        st.dataframe(logs_sorted)

        unique_dates = [d.strftime("%Y-%m-%d") for d in logs_sorted["date"].unique()]
        del_date_str = st.selectbox("Delete log for date:", options=["--"] + unique_dates)
        if del_date_str != "--":
            if st.button("Delete selected date"):
                delete_log_for_date(datetime.strptime(del_date_str, "%Y-%m-%d").date())
                st.warning(f"Deleted log for {del_date_str}.")
                logs = load_logs()
                logs_sorted = logs.sort_values("date", ascending=False)
                st.dataframe(logs_sorted)

        if st.button("Download CSV history"):
            csv = logs_sorted.to_csv(index=False)
            st.download_button(
                "Click to download",
                csv,
                file_name="training_history.csv",
                mime="text/csv",
            )

    st.markdown("---")
    st.markdown("#### 🧠 AI Coach – History")
    hist_prompt = st.text_input(
        "Ask about trends, load management, or how to adjust the next block:",
        key="hist_prompt",
    )
    if st.button("Ask AI Coach (history)") and hist_prompt.strip():
        ctx = build_history_summary(load_logs(), days=30)
        ans = ai_coach_answer(hist_prompt.strip(), ctx)
        st.session_state["ai_history_answer"] = ans

    if st.session_state.get("ai_history_answer"):
        st.write(st.session_state["ai_history_answer"])


# -----------------------------
# Main app
# -----------------------------

def main() -> None:
    st.set_page_config(page_title="2025–26 Training Planner", layout="centered")
    init_session_state()

    logs = load_logs()
    work_df = load_work_schedule()

    st.sidebar.title("Navigation")
    prev_page = st.session_state.get("page", "Today")
    page = st.sidebar.radio(
        "Go to",
        ["Today", "This Week", "Calendar", "Phase", "History"],
        index=["Today", "This Week", "Calendar", "Phase", "History"].index(prev_page),
        key="nav_radio",
    )

    # If user just switched to Today from another page, reset date to real today
    if page == "Today" and prev_page != "Today":
        st.session_state["selected_date"] = date.today()

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
