from datetime import datetime, timedelta
import math

from utils.interfaces import Activity, Task
from utils.logs import log_message

TOLERANCE_START_TIME_ACTIVITY = timedelta(minutes=30)

START_DAY_TIME_LIMIT = datetime.combine(
    datetime.today(), datetime.strptime("07:00", "%H:%M").time()
)
END_DAY_TIME_LIMIT = datetime.combine(
    datetime.today(), datetime.strptime("22:00", "%H:%M").time()
)


def set_skipped_activities(task: Task, current_time: datetime) -> None:
    for activity in task["activities"]:
        if activity["done"] is True:
            activity["skipped"] = False
            continue

        if (
            activity["skipped"] is True
            or activity["without_time"] is True
            or activity["activity_time"] is None
        ):
            continue

        administration_time = datetime.strptime(
            activity["activity_time"], "%H:%M"
        ).replace(
            year=current_time.year, month=current_time.month, day=current_time.day
        )

        if current_time - administration_time >= TOLERANCE_START_TIME_ACTIVITY:
            activity["skipped"] = True
            activity["done"] = False


def parse_time(time_str: str | None):
    """Convert time string to datetime object. Returns None if time_str is None."""
    return (
        datetime.combine(datetime.today(), datetime.strptime(time_str, "%H:%M").time())
        if time_str
        else None
    )


def format_time(time_obj: datetime | None):
    """Convert datetime object to string format HH:MM. Returns 'No fixed time' if None."""
    return time_obj.strftime("%H:%M") if time_obj else "No fixed time"


def conflicts(activity1: Activity, activity2: Activity):
    """Check if two activities have overlapping time slots."""
    start1 = parse_time(activity1.get("activity_time"))
    start2 = parse_time(activity2.get("activity_time"))
    if not start1 or not start2:
        return False

    duration1 = timedelta(minutes=activity1["activity_duration"])
    duration2 = timedelta(minutes=activity2["activity_duration"])
    tolerance1 = timedelta(
        minutes=math.ceil(duration1.total_seconds() / 60 * 0.1)
    )  # 10% tolerance
    tolerance2 = timedelta(minutes=math.ceil(duration2.total_seconds() / 60 * 0.1))

    # Pessemistic approach: consider the worst case scenario for the end time adding the tolerance
    end1 = start1 + duration1 + tolerance1
    end2 = start2 + duration2 + tolerance2

    return start1 < end2 and start2 < end1


def resolve_conflicts(new_activity: Activity, activities: list[Activity]):
    """Resolve conflicts by prioritizing critical activities first, then priority (1 is highest)."""
    conflicting_activities: list[Activity] = []

    for act in activities:
        if conflicts(new_activity, act):
            conflicting_activities.append(act)

    if not conflicting_activities:
        return activities + [new_activity]

    for act in conflicting_activities:
        if new_activity["critical"] and not act["critical"]:
            log_message(
                f"Replacing {act['activity_name']} with {new_activity['activity_name']} due to critical status"
            )
            activities.remove(act)
        elif act["critical"] and not new_activity["critical"]:
            log_message(
                f"Keeping {act['activity_name']} over {new_activity['activity_name']} due to critical status"
            )
            return activities  # Keep existing activity
        elif new_activity["priority"] < act["priority"]:
            log_message(
                f"Replacing {act['activity_name']} with {new_activity['activity_name']} due to higher priority"
            )
            activities.remove(act)
        elif new_activity["priority"] > act["priority"]:
            log_message(
                f"Keeping {act['activity_name']} over {new_activity['activity_name']} due to higher priority"
            )
            return activities  # Keep existing activity

    log_message(
        f"Conflict between {new_activity['activity_name']} and {[a['activity_name'] for a in conflicting_activities]}. Keeping both for user decision."
    )
    return activities + [new_activity]


def schedule_activities(
    activity_list: list[Activity], current_time: datetime | None
) -> tuple[list[Activity], list[Activity]]:
    """Schedule activities and return the next activities to do based on current time."""
    activities: list[Activity] = []
    for act in activity_list:
        activity_time = parse_time(act.get("activity_time"))
        activities = (
            resolve_conflicts(act, activities)
            if activity_time
            else auto_schedule(act, activities)
        )
    activities.sort(key=lambda a: parse_time(a["activity_time"]) or datetime.max)
    scheduled_activities: list[Activity] = [
        {**a, "activity_time": a["activity_time"]} for a in activities
    ]
    return scheduled_activities, get_next_activity(activities, current_time)


def auto_schedule(activity: Activity, activities: list[Activity]):
    """Assign a self-schedulable activity to the earliest available time slot."""
    last_end_time = START_DAY_TIME_LIMIT
    for act in sorted(activities, key=lambda x: x["activity_time"] or datetime.max):
        activity_time = parse_time(act["activity_time"])
        if activity_time and (activity_time - last_end_time) >= timedelta(
            minutes=activity["activity_duration"]
        ):
            activity["activity_time"] = format_time(last_end_time)
            return activities + [activity]
        if activity_time:
            last_end_time = max(
                last_end_time,
                activity_time + timedelta(minutes=act["activity_duration"]),
            )
    if (END_DAY_TIME_LIMIT - last_end_time) >= timedelta(
        minutes=activity["activity_duration"]
    ):
        activity["activity_time"] = format_time(last_end_time)
        return activities + [activity]
    log_message(f"No slot for {activity['activity_name']}")
    return activities


def get_next_activity(
    activities: list[Activity], current_time: datetime | None
) -> list[Activity]:
    """Return the next activity (or multiple activities if they have the same critical status and priority)."""
    # If current time is not set or outside the day time limit, return empty list
    if (
        not current_time
        or current_time < START_DAY_TIME_LIMIT
        or current_time > END_DAY_TIME_LIMIT
    ):
        return []

    upcoming_activities: list[Activity] = []
    for a in activities:
        activity_time = parse_time(a["activity_time"])
        if (
            activity_time
            and current_time - TOLERANCE_START_TIME_ACTIVITY
            <= activity_time
            <= current_time + TOLERANCE_START_TIME_ACTIVITY
            and not a["done"]
            and not a["skipped"]
        ):
            upcoming_activities.append(a)
    if not upcoming_activities:
        return []

    upcoming_activities.sort(
        key=lambda a: parse_time(a["activity_time"]) or datetime.max
    )
    top_activity = upcoming_activities[0]
    result: list[Activity] = [
        a
        for a in upcoming_activities
        if a["critical"] == top_activity["critical"]
        and a["priority"] == top_activity["priority"]
    ]
    return result
