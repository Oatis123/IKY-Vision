import time


class PersonTracker:
    def __init__(self, time_limit: float = 20.0):
        self.tracked_persons: dict[str, float] = {}
        self.time_limit = time_limit

    def track(self, persons: list[str]) -> list[str]:
        """Возвращает имена людей, которые пропали из кадра дольше чем на time_limit секунд."""
        now = time.perf_counter()

        visible_persons = {p for p in persons if p != "Unknown"}

        for person in visible_persons:
            self.tracked_persons[person] = now

        missing_persons = []

        absent_persons = set(self.tracked_persons.keys()) - visible_persons

        for name in absent_persons:
            if now - self.tracked_persons[name] >= self.time_limit:
                missing_persons.append(name)
                
                del self.tracked_persons[name]

        return missing_persons