
import json
from datetime import datetime
import streamlit as st

st.set_page_config(page_title="Training Planner + Coach", layout="wide")

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


def default_workout(day_index: int) -> dict:
    """Return a default rest day workout object."""
    return {
        "id": f"day-{day_index}",
        "day_index": day_index,
        "primary_type": "rest",  # "cardio" | "strength" | "rest"
        "primary_cardio": None,  # {type, duration_min, distance_km}
        "primary_strength": [],  # list of strength blocks
        "addons": {
            "cardio": [],  # list of cardio add-ons
            "strength": [],  # list of strength add-ons
        },
        "notes": "",
    }


def init_session_state():
    if "plan" not in st.session_state:
        st.session_state.plan = [default_workout(i) for i in range(7)]
    if "selected_day" not in st.session_state:
        st.session_state.selected_day = 0
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def strength_block_ui(prefix: str, idx: int, default_block: dict | None):
    """Render inputs for a single strength block and return the resulting dict or None.
    prefix: unique prefix per context (e.g. 'primary', 'addon')
    idx: index inside that context
    default_block: existing data or None
    """
    key_base = f"{prefix}_strength_{idx}"
    default_category = (default_block or {}).get("category", "None")
    default_exercise = (default_block or {}).get("exercise", "None")
    default_sets = (default_block or {}).get("sets", 3)
    default_reps = (default_block or {}).get("reps", 10)

    col1, col2, col3, col4 = st.columns([1.2, 1.2, 0.7, 0.7])
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
    """Render inputs for a cardio block and return resulting dict or None."""
    key_base = f"{prefix}_cardio"
    default_type = (default_block or {}).get("type", "Run")
    default_duration = (default_block or {}).get("duration_min", 30)
    default_distance = (default_block or {}).get("distance_km", 0.0)

    col1, col2, col3 = st.columns([1.2, 0.8, 0.8])
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
            value=int(default_duration) if isinstance(default_duration, (int, float)) else 30,
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


def summarize_workout(day: dict) -> str:
    primary_type = day.get("primary_type", "rest")
    summary_parts = []

    if primary_type == "cardio" and day.get("primary_cardio"):
        c = day["primary_cardio"]
        part = f"{c['type']} {c['duration_min']}min"
        if c.get("distance_km"):
            part += f", {c['distance_km']}km"
        summary_parts.append(part)
    elif primary_type == "strength" and day.get("primary_strength"):
        cats = {blk["category"] for blk in day["primary_strength"] if blk.get("category")}
        if cats:
            summary_parts.append("Strength: " + ", ".join(sorted(cats)))
        else:
            summary_parts.append("Strength")
    else:
        summary_parts.append("Rest")

    addons = day.get("addons", {})
    addon_strs = []
    if addons.get("cardio"):
        addon_strs.append(f"+{len(addons['cardio'])} cardio add-on(s)")
    if addons.get("strength"):
        addon_strs.append(f"+{len(addons['strength'])} strength add-on(s)")

    if addon_strs:
        summary_parts.append(" / " + " & ".join(addon_strs))

    return "".join(summary_parts) if summary_parts else "Rest"


def weekly_overview():
    st.subheader("📆 Weekly Schedule")
    plan = st.session_state.plan

    cols = st.columns(7)
    for i, col in enumerate(cols):
        with col:
            day = plan[i]
            st.markdown(f"**{DAYS_OF_WEEK[i]}**")
            st.caption(summarize_workout(day))


def reorder_section():
    st.markdown("---")
    with st.expander("🔀 Shuffle / Reorder workouts this week"):
        plan = st.session_state.plan
        st.write(
            "Assign which existing workout should go to each day. "
            "If you pick the same source day more than once, that workout will be duplicated."
        )

        options = list(range(7))
        labels = [f"{DAYS_OF_WEEK[i]}: {summarize_workout(plan[i])}" for i in options]

        mapping = {}
        for i, day_name in enumerate(DAYS_OF_WEEK):
            mapping[i] = st.selectbox(
                f"{day_name} gets workout from:",
                options=options,
                format_func=lambda x, labels=labels: labels[x],
                index=i,
                key=f"reorder_{i}",
            )

        if st.button("Apply Reorder"):
            new_plan = []
            for dest_day in range(7):
                src_idx = mapping[dest_day]
                src_day = plan[src_idx]
                # Create a shallow copy but update day_index
                copied = json.loads(json.dumps(src_day))
                copied["day_index"] = dest_day
                copied["id"] = f"day-{dest_day}"
                new_plan.append(copied)
            st.session_state.plan = new_plan
            st.success("Reordered workouts for this week.")


def edit_manual_section():
    st.markdown("---")
    st.subheader("✏️ Edit / Manual Workout Entry")

    day_idx = st.selectbox(
        "Select day to edit",
        options=list(range(7)),
        format_func=lambda i: DAYS_OF_WEEK[i],
        index=st.session_state.selected_day,
    )
    st.session_state.selected_day = day_idx
    day = st.session_state.plan[day_idx]

    with st.form("edit_workout_form"):
        st.markdown(f"### {DAYS_OF_WEEK[day_idx]}")

        primary_type = st.radio(
            "Primary workout type",
            options=["cardio", "strength", "rest"],
            index=["cardio", "strength", "rest"].index(day.get("primary_type", "rest")),
            horizontal=True,
        )

        primary_cardio = None
        primary_strength_blocks = []

        if primary_type == "cardio":
            st.markdown("**Primary Cardio**")
            primary_cardio = cardio_block_ui(
                prefix=f"primary_{day_idx}",
                default_block=day.get("primary_cardio"),
            )
        elif primary_type == "strength":
            st.markdown("**Primary Strength Blocks**")
            existing_blocks = day.get("primary_strength") or []
            max_blocks = 4
            for i in range(max_blocks):
                st.markdown(f"Block {i + 1}")
                default_block = existing_blocks[i] if i < len(existing_blocks) else None
                blk = strength_block_ui(prefix=f"primary_{day_idx}", idx=i, default_block=default_block)
                if blk:
                    primary_strength_blocks.append(blk)
                st.markdown("---")

        addons_cardio = []
        addons_strength = []

        with st.expander("➕ Cardio Add-On (optional)"):
            include_cardio_addon = st.checkbox(
                "Include a cardio add-on",
                value=bool(day.get("addons", {}).get("cardio")),
                key=f"addon_cardio_toggle_{day_idx}",
            )
            if include_cardio_addon:
                existing_cardio_addons = day.get("addons", {}).get("cardio") or []
                primary_addon = existing_cardio_addons[0] if existing_cardio_addons else None
                addon = cardio_block_ui(prefix=f"addon_{day_idx}", default_block=primary_addon)
                if addon:
                    addons_cardio.append(addon)

        with st.expander("➕ Strength Add-On (optional)"):
            include_strength_addon = st.checkbox(
                "Include a strength add-on",
                value=bool(day.get("addons", {}).get("strength")),
                key=f"addon_strength_toggle_{day_idx}",
            )
            if include_strength_addon:
                st.markdown("You can define up to 3 additional strength blocks.")
                existing_strength_addons = day.get("addons", {}).get("strength") or []
                max_addon_blocks = 3
                for i in range(max_addon_blocks):
                    st.markdown(f"Add-on Block {i + 1}")
                    default_block = existing_strength_addons[i] if i < len(existing_strength_addons) else None
                    blk = strength_block_ui(prefix=f"addon_{day_idx}", idx=i, default_block=default_block)
                    if blk:
                        addons_strength.append(blk)
                    st.markdown("---")

        notes = st.text_area("Notes (optional)", value=day.get("notes", ""))

        submitted = st.form_submit_button("Save Day")
        if submitted:
            updated = default_workout(day_idx)
            updated["primary_type"] = primary_type
            updated["notes"] = notes

            if primary_type == "cardio" and primary_cardio:
                updated["primary_cardio"] = primary_cardio
            elif primary_type == "strength" and primary_strength_blocks:
                updated["primary_strength"] = primary_strength_blocks

            updated["addons"] = {
                "cardio": addons_cardio,
                "strength": addons_strength,
            }

            st.session_state.plan[day_idx] = updated
            st.success(f"Saved workout for {DAYS_OF_WEEK[day_idx]}.")


def coach_chat_section():
    st.markdown("---")
    st.subheader("🤖 AI Coach (simple stub)")
    st.caption(
        "This is a basic chat UI. Wire it up to your preferred model/API inside the handle_coach_reply() function."
    )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask your coach something about your training...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        # Placeholder coach reply:
        reply = handle_coach_reply(prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.experimental_rerun()


def handle_coach_reply(prompt: str) -> str:
    """Stub for AI coach logic. Replace with a real model call if you want.

    You can import openai / other SDK here and use your API key from environment.
    This stub keeps the file self-contained and runnable without extra setup.
    """
    return (
        "I'm a placeholder coach for now. Once you connect me to a real model, "
        "I can analyze your plan, TCX data, and history in detail. "
        "For now, tell me what type of feedback you want most (pacing, volume, recovery, etc.)."
    )


def tcx_upload_section():
    st.markdown("---")
    st.subheader("📁 Upload a TCX file (optional)")
    st.caption("Quick-and-dirty summary so the coach can eventually learn from your real workouts.")

    uploaded = st.file_uploader("Upload TCX", type=["tcx"])
    if not uploaded:
        return

    try:
        content = uploaded.read().decode("utf-8")
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

        distance_km = total_distance / 1000.0 if total_distance else 0
        duration_min = total_time / 60.0 if total_time else 0
        avg_hr = sum(hr_values) / len(hr_values) if hr_values else None

        st.write("**Parsed Summary**")
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


def main():
    init_session_state()

    st.title("Training Planner")
    st.caption(
        "Plan your week, shuffle workouts freely, enter manual days, and attach optional strength/cardio "
        "to any day — plus a stub AI coach and TCX upload."
    )

    weekly_overview()
    reorder_section()
    edit_manual_section()
    tcx_upload_section()
    coach_chat_section()

    with st.expander("📦 Export / Debug data"):
        st.code(json.dumps(st.session_state.plan, indent=2), language="json")


if __name__ == "__main__":
    main()
