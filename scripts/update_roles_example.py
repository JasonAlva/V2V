#!/usr/bin/env python
"""Example script demonstrating dynamic role updates during pipeline execution."""

from run_live_pipeline import update_roles

# Example 1: Switch ego to a different vehicle
update_roles(ego_id="vehicle_5", coop_ids=["vehicle_3", "vehicle_7"])

# Example 2: Change the coop radius for neighborhood detection
update_roles(ego_id="ego_vehicle", coop_ids=["coop_vehicle"], coop_radius=100.0)

# Example 3: Add multiple cooperative vehicles
update_roles(ego_id="ego_vehicle", coop_ids=["coop_1", "coop_2", "coop_3"], coop_radius=80.0)

print("✓ Roles updated successfully. Changes take effect next frame in the pipeline!")
