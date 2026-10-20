import json
import shutil
from pathlib import Path

project = Path(r"C:\Users\AAA\Desktop\camera_tamper_detection")
notebook = project / "camera_tamper_attack.ipynb"
backup = Path(r"C:\Users\AAA\AppData\Local\Temp\camera_tamper_attack_before_false_alarm_fix.ipynb")
shutil.copy2(notebook, backup)
data = json.loads(notebook.read_text(encoding="utf-8"))
lines = "".join(data["cells"][7]["source"]).splitlines(True)

start = next(i for i, line in enumerate(lines) if line.startswith("            low_variance ="))
end = next(i for i, line in enumerate(lines[start:], start) if line.startswith("            person_moving ="))
new_lines = [
    "            # A plain wall can already have low variance/entropy.\n",
    "            # Only treat these as abnormal when they drop from a textured baseline.\n",
    "            low_variance = (\n",
    "                baseline_metrics[\"variance\"] > 20.0\n",
    "                and quality[\"variance\"] < baseline_metrics[\"variance\"] * 0.35\n",
    "            )\n",
    "            low_entropy = (\n",
    "                baseline_metrics[\"entropy\"] > 2.0\n",
    "                and quality[\"entropy\"] < baseline_metrics[\"entropy\"] * 0.65\n",
    "            )\n",
    "            edge_drop = (\n",
    "                baseline_metrics[\"edge_ratio\"] > 0.01\n",
    "                and quality[\"edge_ratio\"] < baseline_metrics[\"edge_ratio\"] * 0.55\n",
    "            )\n",
    "            hard_black_or_white = (\n",
    "                quality[\"brightness\"] < 15 or quality[\"brightness\"] > 245\n",
    "            )\n",
    "\n",
    "            # Require a larger persistent change before calling it occlusion.\n",
    "            global_scene_change = (\n",
    "                (anchor_diff > 0.12 and adaptive_diff > 0.08)\n",
    "                or (\n",
    "                    changed_ratio > 0.25\n",
    "                    and (\n",
    "                        brightness_change > 0.12\n",
    "                        or low_variance\n",
    "                        or low_entropy\n",
    "                        or edge_drop\n",
    "                        or hard_black_or_white\n",
    "                    )\n",
    "                )\n",
    "            )\n",
    "            slow_occlusion = (\n",
    "                changed_ratio > 0.25\n",
    "                and adaptive_diff > 0.08\n",
    "                and motion_ratio < 0.20\n",
    "                and (\n",
    "                    brightness_change > 0.12\n",
    "                    or low_variance\n",
    "                    or low_entropy\n",
    "                    or edge_drop\n",
    "                    or hard_black_or_white\n",
    "                )\n",
    "            )\n",
    "            quality_candidate = global_scene_change or slow_occlusion\n",
    "\n",
    "            person_moving = motion_ratio > 0.20 and not (\n",
]
lines[start:end] = new_lines
lines = [line.replace("quality_abnormal_count >= 5", "quality_abnormal_count >= 15") for line in lines]
lines = [line.replace("quality_recovery_count >= 10", "quality_recovery_count >= 20") for line in lines]
data["cells"][7]["source"] = lines
notebook.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
print("updated", notebook)
print("backup", backup)
