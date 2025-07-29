import time


class PerformanceTracker:
    """Utility class to track performance metrics"""

    def __init__(self):
        self.stages = {}
        self.current_stage = None
        self.current_stage_start = None

    def start_stage(self, stage_name):
        """Start timing a new stage"""
        if self.current_stage:
            self.end_stage()
        self.current_stage = stage_name
        self.current_stage_start = time.time()
        print(f"⏱️ Starting stage: {stage_name}")
        return self.current_stage_start

    def end_stage(self):
        """End timing the current stage"""
        if self.current_stage and self.current_stage_start:
            duration = time.time() - self.current_stage_start
            self.stages[self.current_stage] = duration
            print(f"✅ Completed stage: {self.current_stage} in {duration:.3f}s")
            self.current_stage = None
            self.current_stage_start = None
            return duration
        return 0

    def get_summary(self):
        """Get a summary of all timed stages"""
        return {
            "stages": self.stages,
            "total_time": sum(self.stages.values()),
            "longest_stage": (
                max(self.stages.items(), key=lambda x: x[1]) if self.stages else None
            ),
        }

    def print_summary(self):
        """Print a summary of performance metrics"""
        summary = self.get_summary()

        print("\n📊 Performance Summary:")

        if summary["longest_stage"]:
            stage, duration = summary["longest_stage"]
            print(f"- Longest stage: {stage} ({duration:.2f}s)")

        print(f"- Total timed operations: {summary['total_time']:.2f}s")

        print("\nDetailed stage timings:")
        for stage, duration in sorted(
            summary["stages"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"- {stage}: {duration:.2f}s")
