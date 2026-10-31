"""
GlacierAI historical-data audit
--------------------------------
Run from the GlacierAI project root:
    python backend/scripts/audit_historical_results.py

This script DOES NOT download imagery or rerun the U-Net. It audits the files
already produced by the historical-analysis pipeline and writes a JSON report.
It checks:
- historical_results.json consistency and year-to-year changes
- satellite GeoTIFF dimensions, band count, all-zero RGB pixels, and band-4
  validity coverage when present
- whether missing-data coverage makes an estimate unsafe to compare

It deliberately does not label model estimates as ground truth.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import rasterio
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'rasterio'. Activate the GlacierAI environment "
        "and install project requirements before running this audit."
    ) from exc


ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_ROOT = ROOT / "datasets" / "historical"
REPORT_PATH = ROOT / "datasets" / "historical" / "historical_quality_audit.json"
ZERO_RGB_WARNING_PERCENT = 5.0
MISSING_DATA_WARNING_PERCENT = 5.0


def find_result_files():
    return sorted(HISTORICAL_ROOT.glob("**/historical_results.json"))


def load_results(path: Path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return [], f"Could not read JSON: {exc}"

    # Handle common shapes used by historical pipeline versions.
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)], None

    if isinstance(payload, dict):
        for key in ("observations", "results", "historical_results", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)], None
        if "year" in payload:
            return [payload], None

    return [], "No observations/results list found in JSON structure."


def inspect_tiff(path: Path):
    info = {
        "path": str(path.relative_to(ROOT)),
        "width": None,
        "height": None,
        "bands": None,
        "crs": None,
        "nodata": None,
        "rgb_all_zero_percent": None,
        "band4_valid_percent": None,
        "warnings": [],
        "error": None,
    }
    try:
        with rasterio.open(path) as src:
            info.update(
                width=src.width,
                height=src.height,
                bands=src.count,
                crs=str(src.crs) if src.crs else None,
                nodata=src.nodata,
            )
            if src.count < 3:
                info["warnings"].append("Fewer than 3 bands; not an RGB satellite image.")
                return info

            # Windowed read avoids loading huge rasters all at once.
            rgb_zero = 0
            rgb_total = 0
            band4_valid = 0
            band4_total = 0
            for _, window in src.block_windows(1):
                rgb = src.read([1, 2, 3], window=window)
                all_zero = np.all(rgb == 0, axis=0)
                rgb_zero += int(np.count_nonzero(all_zero))
                rgb_total += int(all_zero.size)

                if src.count >= 4:
                    validity = src.read(4, window=window)
                    band4_valid += int(np.count_nonzero(validity > 0))
                    band4_total += int(validity.size)

            if rgb_total:
                info["rgb_all_zero_percent"] = round(100.0 * rgb_zero / rgb_total, 3)
                if info["rgb_all_zero_percent"] > ZERO_RGB_WARNING_PERCENT:
                    info["warnings"].append(
                        f"{info['rgb_all_zero_percent']}% of pixels are zero in all RGB bands."
                    )
            if band4_total:
                info["band4_valid_percent"] = round(100.0 * band4_valid / band4_total, 3)
                missing = 100.0 - info["band4_valid_percent"]
                if missing > MISSING_DATA_WARNING_PERCENT:
                    info["warnings"].append(
                        f"Explicit band-4 validity mask marks {missing:.3f}% as invalid."
                    )
                # A likely inconsistency: all RGB-zero pixels but band 4 claims valid.
                if (
                    info["rgb_all_zero_percent"] is not None
                    and info["rgb_all_zero_percent"] > ZERO_RGB_WARNING_PERCENT
                    and info["band4_valid_percent"] > 99.0
                ):
                    info["warnings"].append(
                        "RGB contains many all-zero pixels but band 4 marks almost all pixels valid; "
                        "verify the downloader's dataMask output and band order."
                    )
    except Exception as exc:
        info["error"] = str(exc)
    return info


def locate_satellite_tiffs(glacier_dir: Path):
    candidates = []
    satellite_dir = glacier_dir / "satellite"
    if satellite_dir.exists():
        candidates.extend(satellite_dir.glob("*.tif"))
        candidates.extend(satellite_dir.glob("*.tiff"))
    # Only use the glacier-specific folder to avoid auditing unrelated rasters.
    return sorted(set(p for p in candidates if p.is_file()))


def result_year(item):
    try:
        if item.get("year") is not None:
            return int(item["year"])
        date_value = item.get("date") or item.get("datetime")
        if date_value:
            return datetime.fromisoformat(str(date_value).replace("Z", "+00:00")).year
    except Exception:
        return None
    return None


def result_area(item):
    for key in ("area_km2", "estimated_area_km2", "glacier_area_km2"):
        value = item.get(key)
        try:
            if value is not None and math.isfinite(float(value)):
                return float(value)
        except (TypeError, ValueError):
            pass
    return None


def main():
    if not HISTORICAL_ROOT.exists():
        raise SystemExit(f"Historical data folder not found: {HISTORICAL_ROOT}")

    result_files = find_result_files()
    if not result_files:
        raise SystemExit(
            f"No historical_results.json found under {HISTORICAL_ROOT}. "
            "Run this from the GlacierAI project root."
        )

    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_root": str(ROOT),
        "summary": [],
        "glaciers": [],
        "overall_warnings": [],
    }

    for result_file in result_files:
        glacier_dir = result_file.parent
        items, error = load_results(result_file)
        glacier = {
            "folder": str(glacier_dir.relative_to(ROOT)),
            "results_file": str(result_file.relative_to(ROOT)),
            "json_error": error,
            "observations": [],
            "satellite_files": [],
            "warnings": [],
        }

        if error:
            glacier["warnings"].append(error)

        years_seen = set()
        for item in items:
            year = result_year(item)
            area = result_area(item)
            observation = {
                "year": year,
                "date": item.get("date") or item.get("datetime"),
                "area_km2": area,
                "scene_id": item.get("scene_id") or item.get("scene"),
                "cloud_percent": item.get("catalog_cloud_percent", item.get("cloud_cover")),
                "source_status": item.get("status"),
            }
            glacier["observations"].append(observation)
            if year is not None:
                years_seen.add(year)

        for tif in locate_satellite_tiffs(glacier_dir):
            tif_info = inspect_tiff(tif)
            glacier["satellite_files"].append(tif_info)
            if tif_info["warnings"]:
                glacier["warnings"].extend(
                    [f"{Path(tif_info['path']).name}: {w}" for w in tif_info["warnings"]]
                )

        measured = [
            (o["year"], o["area_km2"], o["date"])
            for o in glacier["observations"]
            if o["year"] is not None and o["area_km2"] is not None
        ]
        measured.sort(key=lambda row: row[0])
        for previous, current in zip(measured, measured[1:]):
            y0, a0, _ = previous
            y1, a1, _ = current
            if a0 > 0:
                pct = (a1 - a0) / a0 * 100.0
                if abs(pct) >= 20:
                    glacier["warnings"].append(
                        f"Estimated area changes {pct:+.1f}% from {y0} to {y1}; "
                        "inspect scene dates, valid coverage, snow, and segmentation before interpreting."
                    )

        if not glacier["observations"]:
            glacier["warnings"].append("No observations could be parsed from historical_results.json.")
        if not glacier["satellite_files"]:
            glacier["warnings"].append("No satellite GeoTIFFs found in this glacier's satellite folder.")

        glacier["warnings"] = list(dict.fromkeys(glacier["warnings"]))
        report["glaciers"].append(glacier)

    for glacier in report["glaciers"]:
        for warning in glacier["warnings"]:
            report["overall_warnings"].append(f"{glacier['folder']}: {warning}")
        report["summary"].append({
            "folder": glacier["folder"],
            "observation_count": len(glacier["observations"]),
            "satellite_tiff_count": len(glacier["satellite_files"]),
            "warning_count": len(glacier["warnings"]),
        })

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n========== GLACIERAI HISTORICAL QUALITY AUDIT ==========")
    print(f"Project: {ROOT}")
    print(f"JSON report: {REPORT_PATH}")
    print()
    for glacier in report["glaciers"]:
        print(f"Glacier folder: {glacier['folder']}")
        for observation in sorted(
            glacier["observations"],
            key=lambda x: (x["year"] is None, x["year"] or 0),
        ):
            if observation["area_km2"] is not None:
                print(
                    f"  {observation['year']}: {observation['area_km2']:.6f} km²"
                    f" | date={observation['date']}"
                )
        for tif in glacier["satellite_files"]:
            if tif["error"]:
                print(f"  TIFF ERROR: {tif['path']}: {tif['error']}")
                continue
            print(
                f"  TIFF: {Path(tif['path']).name} | "
                f"{tif['width']}x{tif['height']} | bands={tif['bands']} | "
                f"RGB-zero={tif['rgb_all_zero_percent']}% | "
                f"band4-valid={tif['band4_valid_percent']}%"
            )
        for warning in glacier["warnings"]:
            print(f"  WARNING: {warning}")
        print()

    print("Audit complete. This is a diagnostic report, not a ground-truth validation.")
    print("========================================================\n")


if __name__ == "__main__":
    main()
