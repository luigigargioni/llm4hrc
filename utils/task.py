from datetime import datetime, timedelta

from utils.interfaces import Task

TOLERANCE_START_TIME_ACTIVITY = timedelta(minutes=30)


def set_skipped_activities(task: Task, current_time: datetime) -> None:
    for activity in task["activities"]:
        if (
            activity["done"] is True
            or activity["skipped"] is True
            or activity["without_time"] is True
        ):
            continue

        if activity["activity_time"] is None:
            continue

        administration_time = datetime.strptime(
            activity["activity_time"], "%H:%M"
        ).replace(
            year=current_time.year, month=current_time.month, day=current_time.day
        )

        if current_time - administration_time >= TOLERANCE_START_TIME_ACTIVITY:
            activity["skipped"] = True
            activity["done"] = False
