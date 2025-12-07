
import json
from datetime import datetime, timedelta, date
import streamlit as st

# =====================
# CONFIG
# =====================

st.set_page_config(page_title="Training Planner", layout="wide")

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

STRENGTH_CATEGORIES = [
    "None",
    "Full Body",
    "Legs",
    "Glutes",
    "Back",
    "Chest",
    "Shoulders",
    "Arms",
    "Core",
]

STRENGTH_EXERCISES = {
    "None": ["None"],
    "Full Body": ["Deadlift", "Clean", "Snatch", "Kettlebell Swing"],
    "Legs": ["Back Squat", "Front Squat", "Leg Press", "Lunge", "Leg Curl", "Leg Extension"],
    "Glutes": ["Hip Thrust", "Glute Bridge", "Step-up", "Bulgarian Split Squat"],
    "Back": ["Pull-up", "Lat Pulldown", "Barbell Row", "Dumbbell Row", "Seated Row"],
    "Chest": ["Bench Press", "Incline Bench", "Dumbbell Press", "Push-up"],
    "Shoulders": ["Overhead Press", "Lateral Raise", "Front Raise", "Rear Delt Fly"],
    "Arms": ["Biceps Curl", "Triceps Extension", "Hammer Curl", "Dips"],
    "Core": ["Plank", "Hanging Leg Raise", "Cable Crunch", "Russian Twist"],
}

CARDIO_TYPES = ["Run", "Bike", "Walk", "Row", "Swim", "Other"]

N_WEEKS_DEFAULT = 24  # how many weeks to keep in memory


# =====================
# DATA MODEL
# =====================

def default_day(week_index: int, day_index: int) -> dict:
    return {
        "week_index": week_index,
        "day_index": day_index,
        "id": f"w{week_index}-d{day_index}",
        "primary_type": "rest",  # "cardio" | "strength" | "rest"
        "primary_cardio": None,  # {type, duration_min, distance_km}
        "primary_strength": [],  # list of {category, exercise, sets, reps}
        "addons": {
            "cardio": [],   # list of cardio add-ons
            "strength": [], # list of strength add-ons
        },
        "notes": "",
        "is_work_day": False,
    }


def init_session_state():
    if "plan_weeks" not in st.session_state:
        st.session_state.plan_weeks = [
            [default_day(w, d) for d in range(7)] for w in range(N_WEEKS_DEFAULT)
        ]

    if "plan_start_date" not in st.session_state:
        # Week 1 starts on the Monday of current week
        today = date.today()
        monday = today - timedelta(days=today.weekday())
        st.session_state.plan_start_date = monday

    if "current_week_index" not in st.session_state:
        st.session_state.current_week_index = 0

    if "selected_day_index" not in st.session_state:
        st.session_state.selected_day_index = datetime.today().weekday()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "settings" not in st.session_state:
        st.session_state.settings = {
            "units": "km",  # or "mi"
            "default_cardio_duration": 45,
        }


def get_week_label(week_index: int) -> str:
    start_date: date = st.session_state.plan_start_date + timedelta(weeks=week_index)
    end_date: date = start_date + timedelta(days=6)
    # Example: "Week 3 — Feb 03–Feb 09"
    start_str = start_date.strftime("%b %d")
    end_str = end_date.strftime("%b %d")
    return f"Week {week_index + 1} — {start_str}–{end_str}"


def summarize_workout(day: dict) -> str:
    primary_type = day.get("primary_type", "rest")
    parts = []

    if primary_type == "cardio" and day.get("primary_cardio"):
        c = day["primary_cardio"]
        s = f"{c['type']} {c['duration_min']}min"
        if c.get("distance_km"):
            s += f", {c['distance_km']}km"
        parts.append(s)
    elif primary_type == "strength" and day.get("primary_strength"):
        cats = {blk["category"] for blk in day["primary_strength"] if blk.get("category")}
        if cats:
            parts.append("Strength: " + ", ".join(sorted(cats)))
        else:
            parts.append("Strength")
    else:
        parts.append("Rest")

    addons = day.get("addons", {})
    addon_bits = []
    if addons.get("cardio"):
        addon_bits.append(f"+{len(addons['cardio'])} cardio add-on(s)")
    if addons.get("strength"):
        addon_bits.append(f"+{len(addons['strength'])} strength add-on(s)")
    if addon_bits:
        parts.append(" / " + " & ".join(addon_bits))

    if day.get("is_work_day"):
        parts.append(" / Work")

    return "".join(parts)


# =====================
# UI HELPERS
# =====================

def strength_block_ui(prefix: str, idx: int, default_block: dict | None):
    key_base = f"{prefix}_strength_{idx}"
    default_category = (default_block or {}).get("category", "None")
    default_exercise = (default_block or {}).get("exercise", "None")
    default_sets = (default_block or {}).get("sets", 3)
    default_reps = (default_block or {}).get("reps", 10)

    col1, col2, col3, col4 = st.columns([1.4, 1.4, 0.8, 0.8])
    with col1:
        category = st.selectbox(
            "Category",
            STRENGTH_CATEGORIES,
            index=STRENGTH_CATEGORIES.index(default_category) if default_category in STRENGTH_CATEGORIES else 0,
            key=f"{key_base}_category",
        )
    with col2:
        exercises = STRENGTH_EXERCISES.get(category, ["None"])
        exercise = st.selectbox(
            "Exercise",
            exercises,
            index=exercises.index(default_exercise) if default_exercise in exercises else 0,
            key=f"{key_base}_exercise",
        )
    with col3:
        sets = st.number_input(
            "Sets",
            min_value=1,
            max_value=10,
            value=int(default_sets) if isinstance(default_sets, int) else 3,
            step=1,
            key=f"{key_base}_sets",
        )
    with col4:
        reps = st.number_input(
            "Reps",
            min_value=1,
            max_value=30,
            value=int(default_reps) if isinstance(default_reps, int) else 10,
            step=1,
            key=f"{key_base}_reps",
        )

    if category == "None":
        return None

    return {
        "category": category,
        "exercise": exercise,
        "sets": int(sets),
        "reps": int(reps),
    }


def cardio_block_ui(prefix: str, default_block: dict | None):
    key_base = f"{prefix}_cardio"
    default_type = (default_block or {}).get("type", "Run")
    default_duration = (default_block or {}).get("duration_min", st.session_state.settings["default_cardio_duration"])
    default_distance = (default_block or {}).get("distance_km", 0.0)

    col1, col2, col3 = st.columns([1.4, 1.0, 1.0])
    with col1:
        ctype = st.selectbox(
            "Cardio Type",
            CARDIO_TYPES,
            index=CARDIO_TYPES.index(default_type) if default_type in CARDIO_TYPES else 0,
            key=f"{key_base}_type",
        )
    with col2:
        duration_min = st.number_input(
            "Duration (min)",
            min_value=0,
            max_value=300,
            value=int(default_duration) if isinstance(default_duration, (int, float)) else 45,
            step=5,
            key=f"{key_base}_duration",
        )
    with col3:
        distance_km = st.number_input(
            "Distance (km)",
            min_value=0.0,
            max_value=100.0,
            value=float(default_distance) if isinstance(default_distance, (int, float)) else 0.0,
            step=0.1,
            key=f"{key_base}_distance",
        )

    if duration_min == 0 and distance_km == 0:
        return None

    return {
        "type": ctype,
        "duration_min": int(duration_min),
        "distance_km": float(distance_km),
    }


def day_editor(week_index: int, day_index: int, compact: bool = False):
    day = st.session_state.plan_weeks[week_index][day_index]
    prefix = f"w{week_index}_d{day_index}"

    if not compact:
        st.markdown(f"### {DAYS_OF_WEEK[day_index]} — {get_week_label(week_index)}")
    else:
        st.markdown(f"**Edit {DAYS_OF_WEEK[day_index]} ({get_week_label(week_index)})**")

    primary_type = st.radio(
        "Primary workout type",
        options=["cardio", "strength", "rest"],
        index=["cardio", "strength", "rest"].index(day.get("primary_type", "rest")),
        horizontal=True,
        key=f"{prefix}_primary_type",
    )

    primary_cardio = None
    primary_strength_blocks = []

    if primary_type == "cardio":
        st.markdown("**Primary Cardio**")
        primary_cardio = cardio_block_ui(prefix=f"{prefix}_primary", default_block=day.get("primary_cardio"))
    elif primary_type == "strength":
        st.markdown("**Primary Strength Blocks**")
        existing_blocks = day.get("primary_strength") or []
        max_blocks = 4
        for i in range(max_blocks):
            st.markdown(f"Block {i + 1}")
            default_block = existing_blocks[i] if i < len(existing_blocks) else None
            blk = strength_block_ui(prefix=f"{prefix}_primary", idx=i, default_block=default_block)
            if blk:
                primary_strength_blocks.append(blk)
            st.markdown("---")

    addons_cardio = []
    addons_strength = []

    with st.expander("➕ Cardio Add-On (optional)"):
        include_cardio_addon = st.checkbox(
            "Include a cardio add-on",
            value=bool(day.get("addons", {}).get("cardio")),
            key=f"{prefix}_addon_cardio_toggle",
        )
        if include_cardio_addon:
            existing_cardio_addons = day.get("addons", {}).get("cardio") or []
            primary_addon = existing_cardio_addons[0] if existing_cardio_addons else None
            addon = cardio_block_ui(prefix=f"{prefix}_addon", default_block=primary_addon)
            if addon:
                addons_cardio.append(addon)

    with st.expander("➕ Strength Add-On (optional)"):
        include_strength_addon = st.checkbox(
            "Include a strength add-on",
            value=bool(day.get("addons", {}).get("strength")),
            key=f"{prefix}_addon_strength_toggle",
        )
        if include_strength_addon:
            st.markdown("You can define up to 3 additional strength blocks.")
            existing_strength_addons = day.get("addons", {}).get("strength") or []
            max_addon_blocks = 3
            for i in range(max_addon_blocks):
                st.markdown(f"Add-on Block {i + 1}")
                default_block = existing_strength_addons[i] if i < len(existing_strength_addons) else None
                blk = strength_block_ui(prefix=f"{prefix}_addon", idx=i, default_block=default_block)
                if blk:
                    addons_strength.append(blk)
                st.markdown("---")

    notes = st.text_area("Notes (optional)", value=day.get("notes", ""), key=f"{prefix}_notes")

    colw1, colw2 = st.columns(2)
    with colw1:
        is_work_day = st.checkbox(
            "Mark as work day",
            value=day.get("is_work_day", False),
            key=f"{prefix}_is_work_day",
        )
    with colw2:
        save_clicked = st.button("Save Day", key=f"{prefix}_save")

    if save_clicked:
        updated = default_day(week_index, day_index)
        updated["primary_type"] = primary_type
        updated["notes"] = notes
        updated["is_work_day"] = is_work_day

        if primary_type == "cardio" and primary_cardio:
            updated["primary_cardio"] = primary_cardio
        elif primary_type == "strength" and primary_strength_blocks:
            updated["primary_strength"] = primary_strength_blocks

        updated["addons"] = {
            "cardio": addons_cardio,
            "strength": addons_strength,
        }

        st.session_state.plan_weeks[week_index][day_index] = updated
        st.success(f"Saved workout for {DAYS_OF_WEEK[day_index]} ({get_week_label(week_index)}).")


# =====================
# PAGES
# =====================

def page_today():
    st.title("Today")

    today = date.today()
    start_date: date = st.session_state.plan_start_date
    delta_days = (today - start_date).days

    if delta_days < 0:
        week_index = 0
        day_index = 0
    else:
        week_index = min(delta_days // 7, N_WEEKS_DEFAULT - 1)
        day_index = min(delta_days % 7, 6)

    day = st.session_state.plan_weeks[week_index][day_index]

    st.caption(get_week_label(week_index))
    st.subheader(f"{DAYS_OF_WEEK[day_index]}")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("#### Summary")
        st.write(summarize_workout(day))

        with st.expander("Edit today's workout"):
            day_editor(week_index, day_index, compact=True)

    with col2:
        st.markdown("#### TCX Upload (today)")
        uploaded = st.file_uploader("Upload TCX for today", type=["tcx"], key="tcx_today")
        if uploaded is not None:
            parse_and_show_tcx(uploaded)

    st.markdown("---")
    st.markdown("#### Coach (stub)")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask your coach...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        reply = handle_coach_reply(prompt, day, week_index)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.experimental_rerun()


def page_this_week():
    st.title("This Week")

    week_index = st.selectbox(
        "Select week",
        options=list(range(N_WEEKS_DEFAULT)),
        index=st.session_state.current_week_index,
        format_func=get_week_label,
        key="week_selector",
    )
    st.session_state.current_week_index = week_index

    week_days = st.session_state.plan_weeks[week_index]

    st.markdown("### Week Overview")
    cols = st.columns(7)
    for i, col in enumerate(cols):
        with col:
            day = week_days[i]
            st.markdown(f"**{DAYS_OF_WEEK[i][:3]}**")
            st.caption(summarize_workout(day))
            if st.button("Edit", key=f"edit_w{week_index}_d{i}"):
                st.session_state.selected_day_index = i

    st.markdown("---")
    st.markdown("### Edit Selected Day")

    day_index = st.session_state.selected_day_index
    day_editor(week_index, day_index)

    st.markdown("---")
    st.markdown("### Shuffle / Reorder This Week")
    with st.expander("Shuffle workouts within this week"):
        options = list(range(7))
        labels = [f"{DAYS_OF_WEEK[i]}: {summarize_workout(week_days[i])}" for i in options]
        mapping = {}
        for i in range(7):
            mapping[i] = st.selectbox(
                f"{DAYS_OF_WEEK[i]} gets workout from:",
                options=options,
                format_func=lambda x, labels=labels: labels[x],
                index=i,
                key=f"reorder_w{week_index}_d{i}",
            )

        if st.button("Apply Shuffle", key=f"shuffle_apply_w{week_index}"):
            new_week = []
            for dest_day in range(7):
                src_idx = mapping[dest_day]
                src_day = week_days[src_idx]
                copied = json.loads(json.dumps(src_day))
                copied["day_index"] = dest_day
                copied["id"] = f"w{week_index}-d{dest_day}"
                new_week.append(copied)
            st.session_state.plan_weeks[week_index] = new_week
            st.success("Week reordered.")


def page_work_schedule():
    st.title("Work Schedule")

    st.markdown(
        "Mark work days or apply a simple 7-on / 7-off pattern across all weeks. "
        "Work days will be visible in your weekly and today views."
    )

    pattern = st.radio(
        "Pattern helper (optional)",
        options=["None", "7-on / 7-off (starting Week 1, Monday)"],
        index=0,
        horizontal=True,
    )

    if pattern != "None":
        if st.button("Apply pattern to all weeks"):
            for w in range(N_WEEKS_DEFAULT):
                # 7 on (week 1), 7 off (week 2), repeat
                on_block = (w % 2 == 0)
                for d in range(7):
                    st.session_state.plan_weeks[w][d]["is_work_day"] = on_block
            st.success("Applied 7-on / 7-off pattern to all weeks.")

    st.markdown("---")
    st.markdown("### Edit work days for a specific week")

    week_index = st.selectbox(
        "Select week to edit",
        options=list(range(N_WEEKS_DEFAULT)),
        index=st.session_state.current_week_index,
        format_func=get_week_label,
        key="work_week_selector",
    )
    st.session_state.current_week_index = week_index

    cols = st.columns(7)
    week_days = st.session_state.plan_weeks[week_index]
    for i, col in enumerate(cols):
        with col:
            day = week_days[i]
            st.markdown(f"**{DAYS_OF_WEEK[i][:3]}**")
            is_work = st.checkbox(
                "Work",
                value=day.get("is_work_day", False),
                key=f"work_w{week_index}_d{i}",
            )
            day["is_work_day"] = is_work

    st.info(
        "Work flags are saved immediately. You can fine-tune training intensity on work days from the "
        "'This Week' page by editing each day."
    )


def page_phase():
    st.title("Phase Overview")

    current_week = st.session_state.current_week_index
    total_weeks = N_WEEKS_DEFAULT

    # Simple example phase logic: 1–8 = Phase 1, 9–16 = Phase 2, etc.
    if current_week < 8:
        phase = "Phase 1 (Base)"
    elif current_week < 16:
        phase = "Phase 2 (Build)"
    else:
        phase = "Phase 3 (Peak)"
    st.subheader(f"Current Phase: {phase}")

    st.markdown(
        "- **Phase 1 (Base)**: Emphasis on easy Zone 2, basic strength technique, and building consistency.\n"
        "- **Phase 2 (Build)**: More tempo/threshold work, heavier strength, and specific hill sessions.\n"
        "- **Phase 3 (Peak)**: Sharper intensity, race-specific prep, and strategic deloads."
    )

    st.markdown("---")
    st.markdown("### Weekly Snapshot")

    cols = st.columns(3)
    with cols[0]:
        st.metric("Current week", current_week + 1)
    with cols[1]:
        st.metric("Total weeks in plan (in memory)", total_weeks)
    with cols[2]:
        st.metric("Weeks remaining", max(total_weeks - (current_week + 1), 0))

    st.markdown("---")
    st.markdown("### Debug / Raw Plan Data (optional)")
    with st.expander("Show raw JSON for all weeks"):
        st.code(json.dumps(st.session_state.plan_weeks, indent=2), language="json")


def page_settings():
    st.title("Settings")

    settings = st.session_state.settings

    units = st.radio(
        "Distance units",
        options=["km", "mi"],
        index=["km", "mi"].index(settings.get("units", "km")),
        horizontal=True,
    )
    settings["units"] = units

    default_duration = st.number_input(
        "Default cardio duration (min)",
        min_value=10,
        max_value=180,
        value=int(settings.get("default_cardio_duration", 45)),
        step=5,
    )
    settings["default_cardio_duration"] = int(default_duration)

    st.session_state.settings = settings

    st.markdown("---")
    st.markdown("### Export / Import Plan (manual)")

    st.markdown("**Export current plan JSON**")
    st.code(json.dumps(st.session_state.plan_weeks, indent=2), language="json")

    st.markdown("**Import plan JSON**")
    uploaded = st.file_uploader("Upload JSON exported from this app", type=["json"], key="import_json")
    if uploaded is not None:
        try:
            data = json.loads(uploaded.read().decode("utf-8"))
            if isinstance(data, list):
                st.session_state.plan_weeks = data
                st.success("Imported plan JSON.")
            else:
                st.error("JSON format not recognized. Expected a list.")
        except Exception as e:
            st.error(f"Failed to import JSON: {e}")


# =====================
# TCX PARSER
# =====================

def parse_and_show_tcx(uploaded_file):
    try:
        content = uploaded_file.read().decode("utf-8")
    except Exception:
        st.error("Could not read TCX file. Make sure it's a valid text-based .tcx file.")
        return

    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(content)
        ns = {"tcx": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}

        total_time = 0.0
        total_distance = 0.0
        hr_values = []

        for lap in root.findall(".//tcx:Lap", ns):
            time_el = lap.find("tcx:TotalTimeSeconds", ns)
            dist_el = lap.find("tcx:DistanceMeters", ns)
            if time_el is not None:
                total_time += float(time_el.text or 0)
            if dist_el is not None:
                total_distance += float(dist_el.text or 0)

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

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Distance (km)", f"{distance_km:.2f}")
        with col2:
            st.metric("Duration (min)", f"{duration_min:.1f}")
        with col3:
            if avg_hr:
                st.metric("Avg HR (bpm)", f"{avg_hr:.0f}")
            else:
                st.metric("Avg HR (bpm)", "N/A")

    except Exception as e:
        st.error(f"Failed to parse TCX: {e}")


# =====================
# COACH STUB
# =====================

def handle_coach_reply(prompt: str, today_day: dict, week_index: int) -> str:
    """Very simple stub that at least is aware of today's workout and week index."""
    summary = summarize_workout(today_day)
    label = get_week_label(week_index)
    return (
        f"You're in {label}. Today's planned work is: {summary}.\n\n"
        "I'm a placeholder coach for now, but I can already see your structure. "
        "Use the 'This Week' page to make sure you have a sensible mix of easy and hard days, "
        "and try to keep hard days away from the heaviest work days.")


# =====================
# MAIN
# =====================

def main():
    init_session_state()

    st.sidebar.title("Training Planner")
    page = st.sidebar.radio(
        "Navigate",
        options=["Today", "This Week", "Work Schedule", "Phase", "Settings"],
    )

    if page == "Today":
        page_today()
    elif page == "This Week":
        page_this_week()
    elif page == "Work Schedule":
        page_work_schedule()
    elif page == "Phase":
        page_phase()
    elif page == "Settings":
        page_settings()


if __name__ == "__main__":
    main()
