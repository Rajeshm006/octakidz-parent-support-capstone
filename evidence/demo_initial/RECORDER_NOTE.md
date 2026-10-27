# Preserved first final-demo attempt

All ten replies and 85 inspected boundary checks completed, but the recorder then called a nonexistent snapshot helper. It stopped safely rather than inventing a final success/reset result. The recorder now calls memory_snapshot(); the agent/runtime is unchanged. This is a recorder failure, not a model or provider outage. The fresh corrected run is kept separately. Its initial cold-index request is included in whole-step accounting.
