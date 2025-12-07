
import json
from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Optional

import streamlit as st
import pandas as pd

try:
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
except Exception:
    client = None

st.set_page_config(page_title="Training Planner", layout="wide")

PLAN_START_DATE = date(2025, 12, 8)
LOG_CSV = "training_log.csv"
WORK_CSV = "work_schedule.csv"

DAYS_LONG = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CARDIO_TYPES = ["Run", "Bike", "Row", "Swim", "Hill Hike", "Other"]


def today_date() -> date:
    return date.today()


def nice_date_label(d: date) -> str:
    return d.strftime("%A, %B %d, %Y")


def load_csv(path: str, columns: List[str]) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=columns)
    for c in columns:
        if c not in df.columns:
            df[c] = None
    return df


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def phase_for_week(week_index: int) -> str:
    if week_index < 8:
        return "2A"
    else:
        return "2B"


def base_week_template(phase: str) -> List[Dict[str, Any]]:
    if phase == "2A":
        return [
            dict(
                name="Strength A – Upper Focus",
                kind="mixed",
                has_cardio=True,
                has_strength=True,
                est_minutes=60,
                description=(
                    "Bench + row emphasis with accessories, then 10–20 min easy Z2 cardio. "
                    "Goal: build pressing strength and upper back without frying recovery."
                ),
            ),
            dict(
                name="Tempo Run",
                kind="cardio",
                has_cardio=True,
                has_strength=False,
                est_minutes=35,
                description=(
                    "10 min easy warm-up, 15 min comfortably-hard tempo (RPE 6–7), "
                    "10 min easy cool-down. Smooth, controlled, non-sprint."
                ),
            ),
            dict(
                name="Z2 Hill Hike",
                kind="cardio",
                has_cardio=True,
                has_strength=False,
                est_minutes=45,
                description=(
                    "Continuous uphill Z2 hike. Breathe steady, no grinding. "
                    "Perfect for leg durability without pounding."
                ),
            ),
            dict(
                name="Strength B – Lower/Posterior",
                kind="strength",
                has_cardio=False,
                has_strength=True,
                est_minutes=60,
                description=(
                    "Deadlift / RDL + glutes and hamstrings. Technique > load. "
                    "Leave 1–3 reps in reserve on all heavy sets."
                ),
            ),
            dict(
                name="Easy Cardio + GBR Support",
                kind="mixed",
                has_cardio=True,
                has_strength=True,
                est_minutes=45,
                description=(
                    "20–25 min very easy Z2 cardio plus submaximal pull-up / push-up work. "
                    "Think practice, not testing."
                ),
            ),
            dict(
                name="Long Z2 Cardio",
                kind="cardio",
                has_cardio=True,
                has_strength=False,
                est_minutes=75,
                description=(
                    "Long steady Z2 run, ride, or hike at conversational pace. "
                    "Fuel and hydrate as you would on long days."
                ),
            ),
            dict(
                name="Rest / Mobility",
                kind="rest",
                has_cardio=False,
                has_strength=False,
                est_minutes=20,
                description=(
                    "Off day. Optional light mobility, walking, or spin only. "
                    "Let the week sink in."
                ),
            ),
        ]
    else:
        return [
            dict(
                name="Strength A – Upper (Heavier)",
                kind="mixed",
                has_cardio=True,
                has_strength=True,
                est_minutes=65,
                description=(
                    "Heavier bench and row variations in lower rep ranges. "
                    "Finish with a short 10–15 min Z2 cooldown."
                ),
            ),
            dict(
                name="Tempo / Cruise Intervals",
                kind="cardio",
                has_cardio=True,
                has_strength=False,
                est_minutes=40,
                description=(
                    "Warm-up then 3–4 × 5 min at tempo with 2 min easy jog between. "
                    "Cool-down easy. Sharp but repeatable effort."
                ),
            ),
            dict(
                name="Z2 Hill Hike or Run",
                kind="cardio",
                has_cardio=True,
                has_strength=False,
                est_minutes=45,
                description=(
                    "Sustained uphill Z2. Mix hiking and light running as legs allow. "
                    "Keep HR smooth and under control."
                ),
            ),
            dict(
                name="Strength B – Lower (Heavier)",
                kind="strength",
                has_cardio=False,
                has_strength=True,
                est_minutes=65,
                description=(
                    "Heavier hinge/squat work. Limit grinding. "
                    "This supports hills, ME, and GBRS strength without wrecking you."
                ),
            ),
            dict(
                name="GBRS Support + Easy Cardio",
                kind="mixed",
                has_cardio=True,
                has_strength=True,
                est_minutes=45,
                description=(
                    "Pull-up, push-up, and core support plus 15–20 min easy Z2. "
                    "Think skill practice and muscular endurance."
                ),
            ),
            dict(
                name="Long Z2 w/ Strides",
                kind="cardio",
                has_cardio=True,
                has_strength=False,
                est_minutes=80,
                description=(
                    "Long Z2 run or ride with 4–6 relaxed 10–20s strides/spin-ups "
                    "near the end. Smooth, not sprinting."
                ),
            ),
            dict(
                name="Rest / Mobility",
                kind="rest",
                has_cardio=False,
                has_strength=False,
                est_minutes=20,
                description=(
                    "Off day again. Focus on moving just enough to feel better tomorrow."
                ),
            ),
        ]


def get_plan_for_date(d: date) -> Dict[str, Any]:
    if d < PLAN_START_DATE:
        return dict(
            phase="Pre-Plan",
            name="No structured plan",
            kind="none",
            has_cardio=False,
            has_strength=False,
            est_minutes=0,
            description="Your structured 2A/2B plan begins on Dec 8, 2025. Before that, treat days as manual.",
        )

    days_from_start = (d - PLAN_START_DATE).days
    week_idx = days_from_start // 7
    day_idx = days_from_start % 7

    phase = phase_for_week(week_idx)
    template = base_week_template(phase)
    day_plan = template[day_idx].copy()
    day_plan["phase"] = f"Phase {phase}"
    day_plan["week_index"] = week_idx + 1
    return day_plan


def load_work_schedule() -> pd.DataFrame:
    df = load_csv(WORK_CSV, ["date", "is_work"])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df["is_work"] = df["is_work"].fillna(False).astype(bool)
    return df


def save_work_schedule(df: pd.DataFrame) -> None:
    out = df.copy()
    out["date"] = out["date"].astype(str)
    save_csv(out, WORK_CSV)


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


def load_logs() -> pd.DataFrame:
    df = load_csv(
        LOG_CSV,
        [
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
        ],
    )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    return df


def save_logs(df: pd.DataFrame) -> None:
    out = df.copy()
    out["date"] = out["date"].astype(str)
    save_csv(out, LOG_CSV)


def get_log_for_date(d: date, logs: pd.DataFrame) -> Optional[pd.Series]:
    if logs.empty:
        return None
    row = logs.loc[logs["date"] == d]
    if row.empty:
        return None
    return row.iloc[0]


def upsert_log(d: date, logs: pd.DataFrame, payload: Dict[str, Any]) -> pd.DataFrame:
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


def ai_chat(system_prompt: str, user_prompt: str) -> str:
    if client is None:
        return "AI is not configured (no API key found in Streamlit secrets)."
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"OpenAI error: {e}"


def ai_suggest_workout(d: date, plan: Dict[str, Any], logs: pd.DataFrame, work_df: pd.DataFrame) -> str:
    lines = []
    for offset in range(-7, 8):
        dd = d + timedelta(days=offset)
        p = get_plan_for_date(dd)
        l = get_log_for_date(dd, logs)
        lines.append(
            {
                "date": str(dd),
                "phase": p.get("phase"),
                "name": p.get("name"),
                "est_minutes": p.get("est_minutes"),
                "is_workday": is_workday(dd, work_df),
                "logged": bool(l is not None),
            }
        )
    system = (
        "You are a hybrid triathlon + strength coach. Given the athlete's planned session, "
        "work schedule, and what they've recently logged, suggest a smart workout for TODAY "
        "that fits in roughly the planned duration. Respond with sections: Title, Estimated Time, "
        "Structure, and Reasoning. Keep it tight and practical."
    )
    user = json.dumps(
        {
            "today": str(d),
            "today_plan": plan,
            "context": lines,
        },
        indent=2,
    )
    return ai_chat(system, user)


def parse_tcx_file(uploaded_file) -> Dict[str, Any]:
    import xml.etree.ElementTree as ET
    try:
        uploaded_file.seek(0)
        tree = ET.parse(uploaded_file)
        root = tree.getroot()
        ns = {"tcx": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}

        total_time = 0.0
        total_dist = 0.0
        hrs = []

        for lap in root.findall(".//tcx:Lap", ns):
            t_el = lap.find("tcx:TotalTimeSeconds", ns)
            d_el = lap.find("tcx:DistanceMeters", ns)
            if t_el is not None and t_el.text:
                total_time += float(t_el.text)
            if d_el is not None and d_el.text:
                total_dist += float(d_el.text)

        for tp in root.findall(".//tcx:Trackpoint", ns):
            h_el = tp.find(".//tcx:HeartRateBpm/tcx:Value", ns)
            if h_el is not None and h_el.text:
                try:
                    hrs.append(int(h_el.text))
                except ValueError:
                    pass

        duration_min = total_time / 60.0 if total_time else 0.0
        distance_km = total_dist / 1000.0 if total_dist else 0.0
        avg_hr = sum(hrs) / len(hrs) if hrs else None

        return {
            "duration_min": duration_min,
            "distance_km": distance_km,
            "avg_hr": avg_hr,
        }
    except Exception as e:
        return {"error": str(e)}


def render_cardio_editor(key_prefix: str, existing: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    st.markdown("**Cardio Session**")
    c1, c2 = st.columns(2)
    with c1:
        ctype = st.selectbox(
            "Type",
            CARDIO_TYPES,
            index=CARDIO_TYPES.index(existing.get("type", "Run")) if existing and existing.get("type") in CARDIO_TYPES else 0,
            key=f"{key_prefix}_ctype",
        )
    with c2:
        duration = st.number_input(
            "Duration (min)",
            min_value=0,
            max_value=300,
            step=5,
            value=int(existing.get("duration_min", 0)) if existing else 0,
            key=f"{key_prefix}_dur",
        )

    dist = st.number_input(
        "Distance (km)",
        min_value=0.0,
        max_value=200.0,
        step=0.1,
        value=float(existing.get("distance_km", 0.0)) if existing else 0.0,
        key=f"{key_prefix}_dist",
    )

    rpe = st.slider(
        "Effort (RPE)",
        min_value=1,
        max_value=10,
        value=int(existing.get("rpe", 6)) if existing and existing.get("rpe") else 6,
        key=f"{key_prefix}_rpe",
    )

    st.caption("📱 iPhone tip: If your file has a cloud icon in Files, tap it first so it fully downloads, then upload.")
    tcx_file = st.file_uploader(
        "Upload TCX / FIT / GPX (optional)",
        type=None,
        accept_multiple_files=False,
        key=f"{key_prefix}_tcx",
    )

    if tcx_file is not None:
        name = tcx_file.name.lower()
        if name.endswith((".tcx", ".xml")):
            parsed = parse_tcx_file(tcx_file)
            if "error" in parsed:
                st.warning(f"TCX parse error: {parsed['error']}")
            else:
                st.success(
                    f"From TCX: {parsed['distance_km']:.2f} km, {parsed['duration_min']:.1f} min"
                    + (f", Avg HR {parsed['avg_hr']:.0f}" if parsed.get("avg_hr") else "")
                )
                if duration == 0:
                    duration = int(parsed["duration_min"])
                if dist == 0:
                    dist = float(parsed["distance_km"])
        else:
            st.warning("File doesn't look like a TCX/XML. We'll still log manually-entered data.")

    return {
        "type": ctype,
        "duration_min": duration,
        "distance_km": dist,
        "rpe": rpe,
    }


def render_strength_editor(key_prefix: str, existing: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    st.markdown("**Strength Session**")
    main = st.text_input(
        "Main lift / pattern",
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
        "Reps (e.g. 5x5, 3x8)",
        value=existing.get("reps", "") if existing else "",
        key=f"{key_prefix}_reps",
    )
    rpe = st.slider(
        "Top set RPE",
        min_value=1,
        max_value=10,
        value=int(existing.get("rpe", 7)) if existing and existing.get("rpe") else 7,
        key=f"{key_prefix}_rpe",
    )
    acc = st.text_area(
        "Accessories / notes",
        value=existing.get("accessories", "") if existing else "",
        key=f"{key_prefix}_acc",
    )
    return {
        "main_name": main,
        "sets": sets,
        "reps": reps,
        "rpe": rpe,
        "accessories": acc,
    }


def render_ai_coach(selected_date: date, logs: pd.DataFrame, work_df: pd.DataFrame):
    st.markdown("---")
    st.subheader("🧠 AI Coach")

    if "coach_messages" not in st.session_state:
        st.session_state.coach_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hey, I'm your AI coach. I know your plan, workdays, and logs. "
                    "Ask me about how to adjust today, manage fatigue, or stack sessions."
                ),
            }
        ]

    ctx_lines = []
    for offset in range(-7, 8):
        d = selected_date + timedelta(days=offset)
        p = get_plan_for_date(d)
        l = get_log_for_date(d, logs)
        ctx_lines.append(
            f"{d}: {p.get('phase')} — {p.get('name')} | workday={is_workday(d, work_df)} | logged={l is not None}"
        )
    ctx = "\n".join(ctx_lines)

    for msg in st.session_state.coach_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask your coach something...")
    if prompt:
        st.session_state.coach_messages.append({"role": "user", "content": prompt})
        system = (
            "You are a concise, practical hybrid triathlon + strength coach. Use the provided context about "
            "the plan, work schedule, and logs to give short, concrete advice."
        )
        user = f"Context:\n{ctx}\n\nUser:\n{prompt}"
        reply = ai_chat(system, user)
        st.session_state.coach_messages.append({"role": "assistant", "content": reply})


def page_today(logs: pd.DataFrame, work_df: pd.DataFrame):
    selected_date: date = st.session_state.get("selected_date", today_date())

    st.title("Today")
    st.subheader(nice_date_label(selected_date))

    week_monday = selected_date - timedelta(days=selected_date.weekday())
    week_days = [week_monday + timedelta(days=i) for i in range(7)]
    labels = [f"{DAYS_LONG[i]} {d.strftime('%m/%d')}" for i, d in enumerate(week_days)]
    try:
        idx_cur = week_days.index(selected_date)
    except ValueError:
        idx_cur = 0

    sel_idx = st.selectbox("View a day this week:", range(7), format_func=lambda i: labels[i], index=idx_cur)
    chosen_date = week_days[sel_idx]
    if chosen_date != selected_date:
        st.session_state["selected_date"] = chosen_date
        selected_date = chosen_date

    plan = get_plan_for_date(selected_date)
    log_row = get_log_for_date(selected_date, logs)
    work_flag = is_workday(selected_date, work_df)

    default_mode = (
        log_row["mode"]
        if log_row is not None and log_row.get("mode") in ["planned", "manual", "rest", "ai"]
        else "planned"
    )
    mode = st.radio(
        "Mode for this day",
        ["planned", "manual", "rest", "ai"],
        format_func=lambda x: {
            "planned": "Planned workout",
            "manual": "Manual builder",
            "rest": "Rest day",
            "ai": "AI suggested workout",
        }[x],
        index=["planned", "manual", "rest", "ai"].index(default_mode),
        horizontal=True,
    )

    st.markdown(f"**Phase:** {plan.get('phase')} · Week {plan.get('week_index', 1)}")
    st.markdown(f"**Day type:** {plan['name']}")
    if plan.get("est_minutes"):
        st.markdown(f"**Estimated time:** {plan['est_minutes']} min")

    badge = "✈️ Workday" if work_flag else "🏠 Home day"
    st.markdown(f"**{badge}**")

    used_ai = False
    cardio_data: Dict[str, Any] = {}
    strength_data: Dict[str, Any] = {}

    if mode == "rest":
        st.info("Rest day selected. No workout blocks to log today.")
    elif mode == "planned":
        st.markdown("### Planned Workout")
        st.write(plan.get("description", ""))

        if plan.get("has_cardio"):
            st.markdown("#### Cardio (log what you actually did)")
            existing_cardio = None
            if log_row is not None and isinstance(log_row.get("cardio_json"), str) and log_row["cardio_json"]:
                try:
                    existing_cardio = json.loads(log_row["cardio_json"])
                except Exception:
                    existing_cardio = None
            cardio_data = render_cardio_editor("planned_cardio", existing_cardio)

        if plan.get("has_strength"):
            st.markdown("#### Strength (log what you actually did)")
            existing_str = None
            if log_row is not None and isinstance(log_row.get("strength_json"), str) and log_row["strength_json"]:
                try:
                    existing_str = json.loads(log_row["strength_json"])
                except Exception:
                    existing_str = None
            strength_data = render_strength_editor("planned_strength", existing_str)

    elif mode == "manual":
        st.markdown("### Manual Builder")
        st.write("Build only what you actually did today.")

        if st.checkbox("Add cardio session today?", value=False):
            cardio_data = render_cardio_editor("manual_cardio", None)

        if st.checkbox("Add strength session today?", value=False):
            strength_data = render_strength_editor("manual_strength", None)

    elif mode == "ai":
        st.markdown("### AI Suggested Workout")
        st.caption("This uses your plan, work schedule, and recent logs to generate a smart session for today.")

        if st.button("Generate / Refresh AI suggestion"):
            suggestion = ai_suggest_workout(selected_date, plan, logs, work_df)
            st.session_state["ai_suggestion_cache"] = {"date": selected_date, "text": suggestion}

        suggestion_text = ""
        cache = st.session_state.get("ai_suggestion_cache")
        if cache and cache.get("date") == selected_date:
            suggestion_text = cache.get("text", "")
        if suggestion_text:
            st.markdown("#### AI Suggested Session")
            st.write(suggestion_text)
            used_ai = True
        else:
            st.info("Click the button above to get an AI-suggested session for today.")

        st.markdown("You can log details for this AI session below if you want structured history.")
        if st.checkbox("Add cardio for AI session?", value=False):
            cardio_data = render_cardio_editor("ai_cardio", None)
        if st.checkbox("Add strength for AI session?", value=False):
            strength_data = render_strength_editor("ai_strength", None)

    st.markdown("---")
    st.markdown("### Daily Trackers")

    c1, c2 = st.columns(2)
    with c1:
        hrv_val = float(log_row["hrv"]) if log_row is not None and pd.notna(log_row["hrv"]) else 0.0
        sleep_val = float(log_row["sleep_hours"]) if log_row is not None and pd.notna(log_row["sleep_hours"]) else 0.0
        hrv = st.number_input("HRV (ms)", min_value=0.0, max_value=300.0, value=hrv_val)
        sleep_hours = st.number_input("Sleep (hours)", min_value=0.0, max_value=14.0, value=sleep_val)
    with c2:
        soreness_val = int(log_row["soreness"]) if log_row is not None and pd.notna(log_row["soreness"]) else 3
        energy_val = int(log_row["energy"]) if log_row is not None and pd.notna(log_row["energy"]) else 7
        soreness = st.slider("Soreness (1–10)", 1, 10, soreness_val)
        energy = st.slider("Energy (1–10)", 1, 10, energy_val)

    notes = st.text_area(
        "Notes",
        value=str(log_row["notes"]) if log_row is not None and pd.notna(log_row["notes"]) else "",
    )

    if st.button("💾 Save today's log"):
        payload = {
            "mode": mode,
            "used_ai": used_ai,
            "cardio_json": json.dumps(cardio_data) if cardio_data else "",
            "strength_json": json.dumps(strength_data) if strength_data else "",
            "hrv": hrv,
            "sleep_hours": sleep_hours,
            "soreness": soreness,
            "energy": energy,
            "notes": notes,
        }
        new_logs = upsert_log(selected_date, logs, payload)
        save_logs(new_logs)
        st.success("Saved today.")

    render_ai_coach(selected_date, logs, work_df)


def page_this_week(logs: pd.DataFrame, work_df: pd.DataFrame):
    st.title("This Week")

    current_date: date = st.session_state.get("selected_date", today_date())

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("« Previous week"):
            current_date = current_date - timedelta(days=7)
            st.session_state["selected_date"] = current_date
    with col2:
        if st.button("Go to Today"):
            current_date = today_date()
            st.session_state["selected_date"] = current_date
    with col3:
        if st.button("Next week »"):
            current_date = current_date + timedelta(days=7)
            st.session_state["selected_date"] = current_date

    monday = current_date - timedelta(days=current_date.weekday())
    st.caption(f"Week of {monday.strftime('%Y-%m-%d')}")
    st.markdown("---")

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
            st.session_state["selected_date"] = d
        st.markdown("---")


def render_calendar(year: int, month: int, logs: pd.DataFrame, work_df: pd.DataFrame):
    st.subheader(f"{datetime(year, month, 1).strftime('%B %Y')}")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("« Prev"):
            prev = datetime(year, month, 1) - timedelta(days=1)
            st.session_state["cal_year"] = prev.year
            st.session_state["cal_month"] = prev.month
    with col2:
        if st.button("This month"):
            t = today_date()
            st.session_state["cal_year"] = t.year
            st.session_state["cal_month"] = t.month
    with col3:
        if st.button("Next »"):
            nxt = datetime(year, month, 28) + timedelta(days=4)
            nxt = datetime(nxt.year, nxt.month, 1)
            st.session_state["cal_year"] = nxt.year
            st.session_state["cal_month"] = nxt.month

    st.markdown("**S   M   T   W   T   F   S**")

    first = date(year, month, 1)
    first_weekday = (first.weekday() + 1) % 7
    if month == 12:
        days_in_month = (date(year + 1, 1, 1) - timedelta(days=1)).day
    else:
        days_in_month = (date(year, month + 1, 1) - timedelta(days=1)).day

    cells: List[Optional[date]] = []
    for _ in range(first_weekday):
        cells.append(None)
    for day_num in range(1, days_in_month + 1):
        cells.append(date(year, month, day_num))
    while len(cells) % 7 != 0:
        cells.append(None)

    today = today_date()
    for i in range(0, len(cells), 7):
        cols = st.columns(7)
        for j, d in enumerate(cells[i:i+7]):
            with cols[j]:
                if d is None:
                    st.write(" ")
                else:
                    label = str(d.day)
                    if d == today:
                        label = f"🔵 {label}"
                    if is_workday(d, work_df):
                        label = f"🔴 {label}"
                    has_log = get_log_for_date(d, logs) is not None
                    if has_log and "🔵" not in label and "🔴" not in label:
                        label = f"• {label}"
                    if st.button(label, key=f"cal_{d}"):
                        new_df = toggle_workday(d, work_df)
                        save_work_schedule(new_df)
                        st.session_state["work_df"] = new_df
                        st.session_state["selected_date"] = d


def page_calendar(logs: pd.DataFrame, work_df: pd.DataFrame):
    st.title("Calendar (Work + Training)")

    if "cal_year" not in st.session_state or "cal_month" not in st.session_state:
        t = today_date()
        st.session_state["cal_year"] = t.year
        st.session_state["cal_month"] = t.month

    year = st.session_state["cal_year"]
    month = st.session_state["cal_month"]

    render_calendar(year, month, logs, work_df)


def page_phase():
    st.title("Phase Overview")

    today = today_date()
    days_from_start = (today - PLAN_START_DATE).days
    week_idx = max(0, days_from_start // 7)
    phase = phase_for_week(week_idx)

    st.markdown(f"### Current phase: Phase {phase}")
    if phase == "2A":
        st.write(
            "Phase 2A = Strength + Aerobic Base. Build upper/lower strength, GBRS support, and aerobic volume "
            "without going too hard yet."
        )
    else:
        st.write(
            "Phase 2B = Slightly sharper work with heavier strength and more structured tempo while still supporting "
            "your hybrid goals."
        )

    st.markdown("### Upcoming weeks (high-level)")
    for offset in range(0, 4):
        w = week_idx + offset
        ph = phase_for_week(w)
        monday = PLAN_START_DATE + timedelta(weeks=w)
        st.markdown(f"**Week {w + 1} ({monday.strftime('%b %d')}) – Phase {ph}**")
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
        st.caption(f"Mode: {row.get('mode', '')} | AI used: {'yes' if row.get('used_ai') else 'no'}")
        if row.get("cardio_json"):
            try:
                c = json.loads(row["cardio_json"])
                st.write(
                    f"Cardio: {c.get('type')} — {c.get('duration_min')} min, "
                    f"{c.get('distance_km')} km, RPE {c.get('rpe')}"
                )
            except Exception:
                pass
        if row.get("strength_json"):
            try:
                s = json.loads(row["strength_json"])
                st.write(
                    f"Strength: {s.get('main_name')} — {s.get('sets')} sets, reps {s.get('reps')}, "
                    f"RPE {s.get('rpe')}"
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


def main():
    if "selected_date" not in st.session_state:
        st.session_state["selected_date"] = today_date()

    logs = load_logs()
    work_df = load_work_schedule()
    st.session_state["work_df"] = work_df

    st.sidebar.title("Navigation")

    if st.sidebar.button("Today"):
        st.session_state["selected_date"] = today_date()
        st.session_state["page"] = "Today"

    page = st.sidebar.radio(
        "Go to",
        ["Today", "This Week", "Calendar", "Phase", "History"],
        index=["Today", "This Week", "Calendar", "Phase", "History"].index(
            st.session_state.get("page", "Today")
        ),
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
        page_history(logs)


if __name__ == "__main__":
    main()
