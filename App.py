
import json
from datetime import date, datetime, timedelta
import pathlib
from typing import Dict, Any, List, Optional

import streamlit as st
import pandas as pd
import openai

# =====================
# CONFIG
# =====================

st.set_page_config(page_title="Training Planner", layout="wide")

DATA_DIR = pathlib.Path(".")
PLAN_START_DATE = date(2025, 12, 8)  # Phase 2A start

LOG_CSV = DATA_DIR / "training_log.csv"
WORK_CSV = DATA_DIR / "work_schedule.csv"

DAYS_SHORT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAYS_LONG = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# OpenAI key (optional – app still works without it)
openai.api_key = st.secrets.get("OPENAI_API_KEY", None)


# =====================
# UTILS
# =====================


def load_csv(path: pathlib.Path, columns: List[str]) -> pd.DataFrame:
    if path.exists():
        try:
            df = pd.read_csv(path)
            # Ensure expected columns exist
            for c in columns:
                if c not in df.columns:
                    df[c] = None
            return df
        except Exception:
            # Corrupt file -> rename and start fresh
            backup = path.with_suffix(".bak")
            path.rename(backup)
    return pd.DataFrame(columns=columns)


def save_csv(df: pd.DataFrame, path: pathlib.Path) -> None:
    df.to_csv(path, index=False)


def get_week_index_for_date(d: date) -> int:
    """0-based week index from PLAN_START_DATE."""
    delta = (d - PLAN_START_DATE).days
    if delta < 0:
        return 0
    return delta // 7


def get_day_index_for_date(d: date) -> int:
    delta = (d - PLAN_START_DATE).days
    if delta < 0:
        return 0
    return delta % 7


def nice_date_label(d: date) -> str:
    return d.strftime("%A, %B %d, %Y")


def today_date() -> date:
    return date.today()


# =====================
# PLAN LOGIC
# =====================


def base_week_template(phase: str) -> List[Dict[str, Any]]:
    """
    Return a 7-day template for a given phase.
    This is deliberately simple but structured, and can be evolved later.
    """
    if phase == "2A":
        return [
            dict(
                name="Strength A – Upper Focus",
                mode="auto",
                kind="strength",
                has_cardio=True,
                has_strength=True,
                est_minutes=60,
                description="Upper push + pull, bench emphasis, followed by 15–20min Z2 run or bike.",
            ),
            dict(
                name="Tempo Run",
                mode="auto",
                kind="cardio",
                has_cardio=True,
                has_strength=False,
                est_minutes=35,
                description="10 min easy, 15 min comfortably hard tempo, 10 min easy.",
            ),
            dict(
                name="Z2 Hill Hike",
                mode="auto",
                kind="cardio",
                has_cardio=True,
                has_strength=False,
                est_minutes=45,
                description="Moderate uphill hike in Z2, continuous if possible.",
            ),
            dict(
                name="Strength B – Lower/Posterior",
                mode="auto",
                kind="strength",
                has_cardio=False,
                has_strength=True,
                est_minutes=60,
                description="Deadlift / hinge focus + glutes, accessories for hamstrings and low back.",
            ),
            dict(
                name="Easy Cardio + Pull-ups",
                mode="auto",
                kind="mixed",
                has_cardio=True,
                has_strength=True,
                est_minutes=40,
                description="Easy Z2 run or bike plus focused pull-up practice (several submax sets).",
            ),
            dict(
                name="Long Z2 Cardio",
                mode="auto",
                kind="cardio",
                has_cardio=True,
                has_strength=False,
                est_minutes=75,
                description="Long steady Z2 run, ride, or hike, relaxed conversational pace.",
            ),
            dict(
                name="Rest / Mobility",
                mode="auto",
                kind="rest",
                has_cardio=False,
                has_strength=False,
                est_minutes=20,
                description="Off day. Optional light mobility, walking, or easy spin only.",
            ),
        ]
    else:  # phase 2B – slightly sharper
        return [
            dict(
                name="Strength A – Upper (Heavier)",
                mode="auto",
                kind="strength",
                has_cardio=True,
                has_strength=True,
                est_minutes=65,
                description="Heavier bench and row variations, slightly lower reps. 10–15min Z2 finisher.",
            ),
            dict(
                name="Tempo / Cruise Intervals",
                mode="auto",
                kind="cardio",
                has_cardio=True,
                has_strength=False,
                est_minutes=40,
                description="Warmup then 3–4 × 5min at tempo with 2min easy between. Cooldown.",
            ),
            dict(
                name="Z2 Hill Hike or Run",
                mode="auto",
                kind="cardio",
                has_cardio=True,
                has_strength=False,
                est_minutes=45,
                description="Sustained uphill Z2. Short run segments allowed if legs feel good.",
            ),
            dict(
                name="Strength B – Lower (Heavier)",
                mode="auto",
                kind="strength",
                has_cardio=False,
                has_strength=True,
                est_minutes=65,
                description="Heavier deadlift or squat focus with accessories. Emphasis on quality.",
            ),
            dict(
                name="GBRS Pull / Push Support",
                mode="auto",
                kind="mixed",
                has_cardio=True,
                has_strength=True,
                est_minutes=45,
                description="Support work for pull-ups, push-ups, and core; 15–20min easy cardio.",
            ),
            dict(
                name="Long Z2 w/ strides",
                mode="auto",
                kind="cardio",
                has_cardio=True,
                has_strength=False,
                est_minutes=80,
                description="Long Z2 run/ride with 4–6 short relaxed strides or spin-ups at the end.",
            ),
            dict(
                name="Rest / Mobility",
                mode="auto",
                kind="rest",
                has_cardio=False,
                has_strength=False,
                est_minutes=20,
                description="Off day with optional mobility and walking only.",
            ),
        ]


def phase_for_week(week_index: int) -> str:
    # Weeks 0–7: Phase 2A, Weeks 8–15: Phase 2B, later weeks = maintenance 2B
    if week_index < 8:
        return "2A"
    else:
        return "2B"


def get_plan_for_date(d: date) -> Dict[str, Any]:
    """
    Return a dict describing the plan for a given date.
    """
    if d < PLAN_START_DATE:
        return dict(
            phase="Pre-Plan",
            name="No structured plan yet",
            mode="auto",
            kind="none",
            has_cardio=False,
            has_strength=False,
            est_minutes=0,
            description="Your 2A plan begins on Dec 8, 2025. Before that, treat days as manual.",
        )

    week_index = get_week_index_for_date(d)
    phase = phase_for_week(week_index)
    template = base_week_template(phase)
    day_index = (d - PLAN_START_DATE).days % 7
    day = template[day_index].copy()
    day["phase"] = f"Phase {phase}"
    return day


# =====================
# WORK SCHEDULE
# =====================


def load_work_schedule() -> pd.DataFrame:
    cols = ["date", "is_work"]
    df = load_csv(WORK_CSV, cols)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["is_work"] = df["is_work"].astype(bool)
    return df


def save_work_schedule(df: pd.DataFrame) -> None:
    df_out = df.copy()
    df_out["date"] = df_out["date"].astype(str)
    save_csv(df_out, WORK_CSV)


def is_workday(d: date, work_df: pd.DataFrame) -> bool:
    if work_df.empty:
        return False
    row = work_df.loc[work_df["date"] == d]
    if row.empty:
        return False
    return bool(row.iloc[0]["is_work"])


def toggle_workday(d: date, work_df: pd.DataFrame) -> pd.DataFrame:
    if work_df.empty:
        work_df = pd.DataFrame(columns=["date", "is_work"])
    mask = work_df["date"] == d
    if mask.any():
        work_df.loc[mask, "is_work"] = ~work_df.loc[mask, "is_work"]
    else:
        work_df = pd.concat(
            [work_df, pd.DataFrame([{"date": d, "is_work": True}])],
            ignore_index=True,
        )
    return work_df


# =====================
# LOGGING
# =====================


def load_logs() -> pd.DataFrame:
    cols = [
        "date",
        "mode",
        "used_ai",
        "cardio_json",
        "strength_json",
        "hrv",
        "sleep_hours",
        "soreness",
        "energy",
        "notes",
    ]
    df = load_csv(LOG_CSV, cols)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def save_logs(df: pd.DataFrame) -> None:
    df_out = df.copy()
    df_out["date"] = df_out["date"].astype(str)
    save_csv(df_out, LOG_CSV)


def get_log_for_date(d: date, logs: pd.DataFrame) -> Optional[pd.Series]:
    if logs.empty:
        return None
    row = logs.loc[logs["date"] == d]
    if row.empty:
        return None
    return row.iloc[0]


def upsert_log_for_date(
    d: date,
    logs: pd.DataFrame,
    payload: Dict[str, Any],
) -> pd.DataFrame:
    payload = payload.copy()
    payload["date"] = d
    if logs.empty:
        logs = pd.DataFrame([payload])
    else:
        mask = logs["date"] == d
        if mask.any():
            logs.loc[mask, :] = payload
        else:
            logs = pd.concat([logs, pd.DataFrame([payload])], ignore_index=True)
    return logs


# =====================
# AI HELPERS
# =====================


def call_openai_chat(system_prompt: str, user_prompt: str) -> str:
    if not openai.api_key:
        return "AI is not configured (no API key found)."

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message["content"]
    except Exception as e:
        return f"OpenAI error: {e}"


def generate_ai_workout_suggestion(
    d: date,
    plan: Dict[str, Any],
    logs: pd.DataFrame,
    work_df: pd.DataFrame,
) -> str:
    # Gather last 7 and next 7 days context
    context_lines = []
    for offset in range(-7, 8):
        dd = d + timedelta(days=offset)
        plan_dd = get_plan_for_date(dd)
        log_dd = get_log_for_date(dd, logs)
        line = {
            "date": str(dd),
            "phase": plan_dd.get("phase"),
            "name": plan_dd.get("name"),
            "est_minutes": plan_dd.get("est_minutes"),
            "is_workday": is_workday(dd, work_df),
            "logged": bool(log_dd is not None),
        }
        context_lines.append(line)

    system = (
        "You are a training planner AI for a hybrid triathlon / strength athlete. "
        "Use the provided context—phase, planned sessions, workdays, and recent logs—to suggest a sensible workout "
        "for TODAY that respects fatigue, preserves key sessions, and fits in approximately the planned duration. "
        "Output in a compact, structured human-readable description, with sections: "
        "Title, Estimated Time, Structure, and Reasoning."
    )

    user = json.dumps(
        {
            "today": str(d),
            "today_plan": plan,
            "context": context_lines,
        },
        indent=2,
    )

    return call_openai_chat(system, user)


# =====================
# UI HELPERS
# =====================


def render_cardio_editor(key_prefix: str, existing: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    col1, col2 = st.columns(2)
    with col1:
        ctype = st.selectbox(
            "Cardio type",
            ["Run", "Bike", "Row", "Swim", "Hill Hike", "Other"],
            index=0 if not existing else 0,
            key=f"{key_prefix}_ctype",
        )
    with col2:
        duration = st.number_input(
            "Duration (min)",
            min_value=0,
            max_value=300,
            value=int(existing.get("duration_min", 0)) if existing else 0,
            step=5,
            key=f"{key_prefix}_dur",
        )

    dist = st.number_input(
        "Distance (km)",
        min_value=0.0,
        max_value=200.0,
        value=float(existing.get("distance_km", 0.0)) if existing else 0.0,
        step=0.1,
        key=f"{key_prefix}_dist",
    )

    rpe = st.slider(
        "Effort (RPE)",
        min_value=1,
        max_value=10,
        value=int(existing.get("rpe", 6)) if existing and existing.get("rpe") else 6,
        key=f"{key_prefix}_rpe",
    )

    return {
        "type": ctype,
        "duration_min": duration,
        "distance_km": dist,
        "rpe": rpe,
    }


def render_strength_editor(key_prefix: str, existing: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    st.markdown("**Main Lift / Block**")
    main_name = st.text_input(
        "Main lift or pattern",
        value=existing.get("main_name", "") if existing else "",
        key=f"{key_prefix}_main",
    )
    sets = st.number_input(
        "Sets",
        min_value=0,
        max_value=10,
        value=int(existing.get("sets", 0)) if existing else 0,
        key=f"{key_prefix}_sets",
    )
    reps = st.text_input(
        "Reps (e.g. 5, 5, 5 or 3x8)",
        value=existing.get("reps", "") if existing else "",
        key=f"{key_prefix}_reps",
    )
    rpe = st.slider(
        "Effort (RPE)",
        min_value=1,
        max_value=10,
        value=int(existing.get("rpe", 7)) if existing and existing.get("rpe") else 7,
        key=f"{key_prefix}_rpe",
    )

    st.markdown("**Accessories / Notes**")
    notes = st.text_area(
        "Accessory work",
        value=existing.get("accessories", "") if existing else "",
        key=f"{key_prefix}_acc",
    )

    return {
        "main_name": main_name,
        "sets": sets,
        "reps": reps,
        "rpe": rpe,
        "accessories": notes,
    }


def parse_tcx(file) -> Dict[str, Any]:
    import xml.etree.ElementTree as ET

    try:
        tree = ET.parse(file)
        root = tree.getroot()
        ns = {"tcx": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}

        total_time = 0.0
        total_distance = 0.0
        hr_values = []

        for lap in root.findall(".//tcx:Lap", ns):
            time_el = lap.find("tcx:TotalTimeSeconds", ns)
            dist_el = lap.find("tcx:DistanceMeters", ns)
            if time_el is not None and time_el.text:
                total_time += float(time_el.text)
            if dist_el is not None and dist_el.text:
                total_distance += float(dist_el.text)

        for tp in root.findall(".//tcx:Trackpoint", ns):
            hr_el = tp.find(".//tcx:HeartRateBpm/tcx:Value", ns)
            if hr_el is not None and hr_el.text:
                try:
                    hr_values.append(int(hr_el.text))
                except ValueError:
                    pass

        distance_km = total_distance / 1000.0 if total_distance else 0.0
        duration_min = total_time / 60.0 if total_time else 0.0
        avg_hr = sum(hr_values) / len(hr_values) if hr_values else None

        return {
            "distance_km": distance_km,
            "duration_min": duration_min,
            "avg_hr": avg_hr,
        }
    except Exception as e:
        return {"error": str(e)}


# =====================
# PAGES
# =====================


def page_today(selected_date: date, logs: pd.DataFrame, work_df: pd.DataFrame):
    st.title("Today")

    st.subheader(nice_date_label(selected_date))

    # Plan + log
    plan = get_plan_for_date(selected_date)
    log_row = get_log_for_date(selected_date, logs)

    day_mode = log_row["mode"] if log_row is not None else "auto"
    mode = st.radio(
        "Day mode",
        options=["auto", "manual", "rest", "ai"],
        format_func=lambda x: {
            "auto": "Auto (use planned workout)",
            "manual": "Manual builder",
            "rest": "Rest day",
            "ai": "AI suggestion",
        }[x],
        index=["auto", "manual", "rest", "ai"].index(day_mode) if day_mode in ["auto", "manual", "rest", "ai"] else 0,
    )

    # Workday badge
    work_badge = "✈️ Workday" if is_workday(selected_date, work_df) else "🏠 Home day"
    st.markdown(f"**{work_badge}**")

    st.markdown(f"**Phase:** {plan.get('phase', '')}")
    st.markdown(f"**Planned workout:** {plan['name']}")
    if plan.get("est_minutes"):
        st.markdown(f"**Estimated time:** {plan['est_minutes']} minutes")

    st.markdown("**Planned Session Details:**")
    st.write(plan.get("description", ""))

    # Coaching tips placeholder
    with st.expander("🧠 Coaching Tips"):
        st.write(
            "Treat this phase as building strength and aerobic base for Tahoe & GBRS. "
            "On workdays, it's OK to trim warmups or accessories to fit duty schedule."
        )

    cardio_data = {}
    strength_data = {}
    used_ai = False

    if mode == "rest":
        st.info("Rest day selected. No workout blocks for today.")
    elif mode == "auto":
        # Show planned blocks if they exist, and allow logging
        if plan.get("has_cardio"):
            st.markdown("### Main Cardio")
            existing_cardio = None
            if log_row is not None and isinstance(log_row.get("cardio_json"), str):
                try:
                    existing_cardio = json.loads(log_row["cardio_json"])
                except Exception:
                    existing_cardio = None
            cardio_data = render_cardio_editor("auto_cardio", existing_cardio)
            tcx_file = st.file_uploader("Upload TCX file (optional)", type=["tcx"], key="tcx_auto")
            if tcx_file is not None:
                parsed = parse_tcx(tcx_file)
                if "error" in parsed:
                    st.warning(f"TCX parse error: {parsed['error']}")
                else:
                    st.markdown(
                        f"**From TCX:** {parsed['distance_km']:.2f} km, {parsed['duration_min']:.1f} min, "
                        f"Avg HR: {parsed['avg_hr']:.0f}" if parsed["avg_hr"] else ""
                    )

        if plan.get("has_strength"):
            st.markdown("### Main Strength / ME")
            existing_str = None
            if log_row is not None and isinstance(log_row.get("strength_json"), str):
                try:
                    existing_str = json.loads(log_row["strength_json"])
                except Exception:
                    existing_str = None
            strength_data = render_strength_editor("auto_strength", existing_str)

    elif mode == "manual":
        st.markdown("### Manual Builder")
        st.markdown("Add only what you actually did today.")

        add_cardio = st.checkbox("Add a cardio session today?")
        if add_cardio:
            st.markdown("#### Cardio Session")
            cardio_data = render_cardio_editor("manual_cardio", None)
            tcx_file = st.file_uploader("Upload TCX file (optional)", type=["tcx"], key="tcx_manual")
            if tcx_file is not None:
                parsed = parse_tcx(tcx_file)
                if "error" in parsed:
                    st.warning(f"TCX parse error: {parsed['error']}")
                else:
                    st.markdown(
                        f"**From TCX:** {parsed['distance_km']:.2f} km, {parsed['duration_min']:.1f} min, "
                        f"Avg HR: {parsed['avg_hr']:.0f}" if parsed["avg_hr"] else ""
                    )

        add_strength = st.checkbox("Add a strength session today?")
        if add_strength:
            st.markdown("#### Strength Session")
            strength_data = render_strength_editor("manual_strength", None)

    elif mode == "ai":
        st.markdown("### AI-Suggested Session (stacked with planned)")

        if "ai_suggestion_cache" not in st.session_state or st.session_state.ai_suggestion_cache.get(
            "date"
        ) != selected_date:
            with st.spinner("Asking AI for a suggestion..."):
                suggestion = generate_ai_workout_suggestion(selected_date, plan, logs, work_df)
            st.session_state.ai_suggestion_cache = {"date": selected_date, "text": suggestion}
        else:
            suggestion = st.session_state.ai_suggestion_cache["text"]

        st.markdown("#### Planned Workout")
        st.write(plan.get("description", ""))

        st.markdown("#### AI Suggested Workout")
        st.write(suggestion)

        use_ai = st.checkbox("Use AI suggestion as today's workout?", value=False)
        if use_ai:
            used_ai = True
            # For now we don't parse the AI text into structured fields; we just store it as notes.

    st.markdown("---")
    st.markdown("### Daily Trackers")

    col1, col2 = st.columns(2)
    with col1:
        hrv = st.number_input(
            "HRV (ms)", min_value=0.0, max_value=300.0, value=float(log_row["hrv"]) if log_row is not None and pd.notna(log_row["hrv"]) else 0.0
        )
        sleep = st.number_input(
            "Sleep (hours)", min_value=0.0, max_value=14.0, value=float(log_row["sleep_hours"]) if log_row is not None and pd.notna(log_row["sleep_hours"]) else 0.0
        )
    with col2:
        soreness = st.slider(
            "Soreness (1–10)",
            min_value=1,
            max_value=10,
            value=int(log_row["soreness"]) if log_row is not None and pd.notna(log_row["soreness"]) else 3,
        )
        energy = st.slider(
            "Energy (1–10)",
            min_value=1,
            max_value=10,
            value=int(log_row["energy"]) if log_row is not None and pd.notna(log_row["energy"]) else 7,
        )

    notes = st.text_area(
        "Notes",
        value=str(log_row["notes"]) if log_row is not None and pd.notna(log_row["notes"]) else "",
    )

    if st.button("Save today's log"):
        payload = {
            "mode": mode,
            "used_ai": used_ai,
            "cardio_json": json.dumps(cardio_data) if cardio_data else "",
            "strength_json": json.dumps(strength_data) if strength_data else "",
            "hrv": hrv,
            "sleep_hours": sleep,
            "soreness": soreness,
            "energy": energy,
            "notes": notes,
        }
        new_logs = upsert_log_for_date(selected_date, logs, payload)
        save_logs(new_logs)
        st.success("Saved.")
        st.experimental_rerun()



def page_this_week(selected_date: date, logs: pd.DataFrame, work_df: pd.DataFrame):
    st.title("This Week")

    # Find Monday of current week (relative to selected_date)
    monday = selected_date - timedelta(days=selected_date.weekday())
    st.caption(f"Week of {monday.strftime('%Y-%m-%d')}")

    # Navigation
    col_prev, col_today, col_next = st.columns(3)
    with col_prev:
        if st.button("« Previous week"):
            st.session_state.selected_date = monday - timedelta(days=7)
            st.experimental_rerun()
    with col_today:
        if st.button("Go to Today"):
            st.session_state.selected_date = today_date()
            st.experimental_rerun()
    with col_next:
        if st.button("Next week »"):
            st.session_state.selected_date = monday + timedelta(days=7)
            st.experimental_rerun()

    st.markdown("---")

    # For each day of the week, show ONLY planned workout info
    for i in range(7):
        d = monday + timedelta(days=i)
        plan = get_plan_for_date(d)

        st.markdown(f"#### {DAYS_LONG[i]} {d.strftime('%m/%d')}")
        st.markdown(f"*{plan.get('phase', '')}*")
        st.markdown(f"**{plan['name']}**")
        if plan.get("est_minutes"):
            st.markdown(f"_Est. time: {plan['est_minutes']} min_")
        if plan.get("description"):
            st.write(plan["description"])

        if st.button("Open", key=f"open_{d}"):
            st.session_state.selected_date = d
            st.session_state.sidebar_page = "Today"
            st.experimental_rerun()

        st.markdown("---")
def render_work_calendar(selected_year: int, selected_month: int, work_df: pd.DataFrame):
    st.subheader(f"{datetime(selected_year, selected_month, 1).strftime('%B %Y')}")

    # Navigation
    col_prev, col_today, col_next = st.columns(3)
    with col_prev:
        if st.button("« Prev month"):
            prev = datetime(selected_year, selected_month, 1) - timedelta(days=1)
            st.session_state.cal_year = prev.year
            st.session_state.cal_month = prev.month
            st.experimental_rerun()
    with col_today:
        if st.button("This month"):
            today = today_date()
            st.session_state.cal_year = today.year
            st.session_state.cal_month = today.month
            st.experimental_rerun()
    with col_next:
        if st.button("Next month »"):
            nxt = datetime(selected_year, selected_month, 28) + timedelta(days=4)
            nxt = datetime(nxt.year, nxt.month, 1)
            st.session_state.cal_year = nxt.year
            st.session_state.cal_month = nxt.month
            st.experimental_rerun()

    st.markdown("---")

    # Calendar grid (Sunday start like iOS)
    st.markdown("**S   M   T   W   T   F   S**")

    first = date(selected_year, selected_month, 1)
    start_weekday = (first.weekday() + 1) % 7  # convert Monday=0 to Sunday=0
    if selected_month < 12:
        days_in_month = (date(selected_year, selected_month + 1, 1) - timedelta(days=1)).day
    else:
        days_in_month = (date(selected_year + 1, 1, 1) - timedelta(days=1)).day

    cells = []
    for _ in range(start_weekday):
        cells.append(None)
    for day_num in range(1, days_in_month + 1):
        cells.append(date(selected_year, selected_month, day_num))

    # Pad to multiple of 7
    while len(cells) % 7 != 0:
        cells.append(None)

    for row_start in range(0, len(cells), 7):
        cols = st.columns(7)
        for i, d in enumerate(cells[row_start : row_start + 7]):
            with cols[i]:
                if d is None:
                    st.write(" ")
                else:
                    is_w = is_workday(d, work_df)
                    label = str(d.day)
                    today = today_date()
                    # blue dot for today
                    if d == today:
                        label = f"🔵 {label}"
                    # red circle for workday
                    if is_w:
                        label = f"🔴 {label}"
                    if st.button(label, key=f"cal_{d}"):
                        new_df = toggle_workday(d, work_df)
                        save_work_schedule(new_df)
                        st.experimental_rerun()


def page_work_schedule(work_df: pd.DataFrame):
    st.title("Work Schedule")

    if "cal_year" not in st.session_state or "cal_month" not in st.session_state:
        t = today_date()
        st.session_state.cal_year = t.year
        st.session_state.cal_month = t.month

    render_work_calendar(st.session_state.cal_year, st.session_state.cal_month, work_df)


def page_phase():
    st.title("Phase Overview")
    today = today_date()
    week_index = get_week_index_for_date(today)
    phase = phase_for_week(week_index)

    st.markdown(f"### Current phase: Phase {phase}")
    st.write(
        "Phase 2A = Strength + Aerobic Base, building your foundation and GBRS lifts.\n\n"
        "Phase 2B = Sharpening slightly with more quality tempo work and heavier strength, "
        "while still supporting triathlon volume."
    )

    st.markdown("### Outlook")
    for offset in range(0, 4):
        w = week_index + offset
        ph = phase_for_week(w)
        monday = PLAN_START_DATE + timedelta(weeks=w)
        st.markdown(f"**Week {w + 1} ({monday.strftime('%b %d')}) — Phase {ph}**")
        template = base_week_template(ph)
        names = ", ".join(d["name"] for d in template)
        st.caption(names)


def page_history(logs: pd.DataFrame):
    st.title("History")

    if logs.empty:
        st.info("No logs yet.")
        return

    logs_sorted = logs.sort_values("date", ascending=False)
    for _, row in logs_sorted.iterrows():
        d = row["date"]
        st.markdown(f"#### {nice_date_label(d)}")
        st.caption(f"Mode: {row['mode']}, AI used: {'yes' if row.get('used_ai') else 'no'}")
        if row.get("cardio_json"):
            try:
                c = json.loads(row["cardio_json"])
                st.write(f"Cardio: {c.get('type')} — {c.get('duration_min')} min, {c.get('distance_km')} km, RPE {c.get('rpe')}")
            except Exception:
                pass
        if row.get("strength_json"):
            try:
                s = json.loads(row["strength_json"])
                st.write(
                    f"Strength: {s.get('main_name')} — {s.get('sets')} sets, reps {s.get('reps')}, RPE {s.get('rpe')}"
                )
            except Exception:
                pass
        st.caption(
            f"HRV: {row.get('hrv')} | Sleep: {row.get('sleep_hours')}h | "
            f"Soreness: {row.get('soreness')} | Energy: {row.get('energy')}"
        )
        if row.get("notes"):
            st.write(row["notes"])
        st.markdown("---")


def page_ai_coach(selected_date: date, logs: pd.DataFrame, work_df: pd.DataFrame):
    st.title("AI Coach (context-aware)")

    if "coach_messages" not in st.session_state:
        st.session_state.coach_messages = [
            {
                "role": "assistant",
                "content": "Hey, I'm your AI coach. I know your phase, plan, and recent logs. Ask me about adjustments, "
                "fatigue, or how to shape today around your goals.",
            }
        ]

    # Build context summary
    today = selected_date
    lines = []
    for offset in range(-7, 8):
        d = today + timedelta(days=offset)
        plan = get_plan_for_date(d)
        log = get_log_for_date(d, logs)
        lines.append(
            f"{d}: {plan.get('phase')} — {plan.get('name')} | "
            f"workday={is_workday(d, work_df)} | logged={log is not None}"
        )

    context_text = "\n".join(lines)

    for msg in st.session_state.coach_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask your coach something...")
    if prompt:
        st.session_state.coach_messages.append({"role": "user", "content": prompt})
        system = (
            "You are a hybrid endurance + strength training coach. Use the provided summary of the athlete's "
            "plan, work schedule, and logs to give concise, practical advice."
        )
        user_prompt = f"Context:\n{context_text}\n\nUser question:\n{prompt}"
        reply = call_openai_chat(system, user_prompt)
        st.session_state.coach_messages.append({"role": "assistant", "content": reply})
        st.experimental_rerun()


# =====================
# MAIN
# =====================


def main():
    # Global state
    if "selected_date" not in st.session_state:
        st.session_state.selected_date = today_date()
    if "sidebar_page" not in st.session_state:
        st.session_state.sidebar_page = "Today"

    logs = load_logs()
    work_df = load_work_schedule()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    if st.sidebar.button("Today"):
        st.session_state.selected_date = today_date()
        st.session_state.sidebar_page = "Today"
    if st.sidebar.button("This Week"):
        st.session_state.sidebar_page = "This Week"
    if st.sidebar.button("Phase"):
        st.session_state.sidebar_page = "Phase"
    if st.sidebar.button("Work Schedule"):
        st.session_state.sidebar_page = "Work Schedule"
    if st.sidebar.button("History"):
        st.session_state.sidebar_page = "History"
    if st.sidebar.button("AI Coach"):
        st.session_state.sidebar_page = "AI Coach"

    page = st.session_state.sidebar_page
    selected_date = st.session_state.selected_date

    if page == "Today":
        page_today(selected_date, logs, work_df)
    elif page == "This Week":
        page_this_week(selected_date, logs, work_df)
    elif page == "Phase":
        page_phase()
    elif page == "Work Schedule":
        page_work_schedule(work_df)
    elif page == "History":
        page_history(logs)
    elif page == "AI Coach":
        page_ai_coach(selected_date, logs, work_df)


if __name__ == "__main__":
    main()
