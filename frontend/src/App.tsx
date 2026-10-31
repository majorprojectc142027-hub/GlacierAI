import { useEffect, useRef, useState } from "react";
import {
  Mountain,
  LayoutDashboard,
  Globe2,
  ScanSearch,
  History,
  Layers3,
  FileText,
  Sun,
  Moon,
  Monitor,
  //ChevronDown,
  Menu,
  ChevronRight,
  Snowflake,
  MapPin,
  Download,
  ArrowUpRight,
  CircleCheck,
  X,
  Search,
  LoaderCircle,
} from "lucide-react";
import "./App.css";
const API_BASE_URL = "http://127.0.0.1:8000";
type Page =
  | "Overview"
  | "Global Map"
  | "AI Analysis"
  | "History"
  | "Debris"
  | "Reports";
type Glacier = {
  rgi_id: string;
  name: string;
  latitude?: number;
  longitude?: number;
  area_km2?: number;
  region?: string;
  country?: string;
  geometry?: Record<string, unknown> | null;
  raw: Record<string, unknown>;
};
const navigation = [
  { label: "Overview", icon: LayoutDashboard },
  { label: "Global Map", icon: Globe2 },
  { label: "AI Analysis", icon: ScanSearch },
  { label: "History", icon: History },
  { label: "Debris", icon: Layers3 },
  { label: "Reports", icon: FileText },
] as const;
function asText(...values: unknown[]): string {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
    if (typeof value === "number" && Number.isFinite(value)) {
      return String(value);
    }
  }
  return "";
}
function asNumber(...values: unknown[]): number | undefined {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
    if (
      typeof value === "string" &&
      value.trim() !== "" &&
      Number.isFinite(Number(value))
    ) {
      return Number(value);
    }
  }
  return undefined;
}
function normalizeGlacier(item: unknown, index: number): Glacier {
  const value =
    item && typeof item === "object"
      ? (item as Record<string, unknown>)
      : {};
  const properties =
    value.properties && typeof value.properties === "object"
      ? (value.properties as Record<string, unknown>)
      : {};
  const merged = { ...properties, ...value };
  return {
    rgi_id: asText(
      merged.rgi_id,
      merged.rgiId,
      merged.RGIId,
      merged.id,
      merged.glacier_id,
      merged.glacierId
    ) || `result-${index}`,
    name: asText(
      merged.name,
      merged.glacier_name,
      merged.glacierName,
      merged.NAME,
      merged.glac_name,
      merged.glacier
    ) || "Unnamed glacier",
    latitude: asNumber(
      merged.latitude,
      merged.lat,
      merged.center_lat,
      merged.cenlat
    ),
    longitude: asNumber(
      merged.longitude,
      merged.lon,
      merged.lng,
      merged.center_lon,
      merged.cenlon
    ),
    area_km2: asNumber(
      merged.area_km2,
      merged.areaKm2,
      merged.area,
      merged.Area
    ),
    region: asText(merged.region, merged.region_name, merged.regionName),
    country: asText(merged.country, merged.country_name, merged.countryName),
    geometry: (merged.geometry ?? merged.geojson ?? merged.polygon ?? merged.boundary ?? null) as Record<string, unknown> | null,
    raw: merged,
  };
}
function extractGlaciers(data: unknown): Glacier[] {
  let items: unknown[] = [];
  if (Array.isArray(data)) {
    items = data;
  } else if (data && typeof data === "object") {
    const response = data as Record<string, unknown>;
    for (const key of ["glaciers", "results", "items", "data"]) {
      if (Array.isArray(response[key])) {
        items = response[key] as unknown[];
        break;
      }
    }
  }
  return items
    .map(normalizeGlacier)
    .filter((glacier) => glacier.name !== "Unnamed glacier");
}
type JsonRecord = Record<string, unknown>;
function recordOf(value: unknown): JsonRecord {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as JsonRecord
    : {};
}
function displayNumber(value: unknown, digits = 2): string {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString(undefined, { maximumFractionDigits: digits })
    : "—";
}
function outputUrl(value: unknown): string {
  if (typeof value !== "string" || !value.trim()) return "";
  if (/^https?:\/\//i.test(value)) return value;
  return `${API_BASE_URL}${value.startsWith("/") ? "" : "/"}${value}`;
}
function buildGlacierMapDocument(glacier: Glacier | null): string {
  const isOverview = glacier === null;
  const lat = glacier?.latitude ?? 20;
  const lon = glacier?.longitude ?? 0;
  const zoom = isOverview ? 2 : 9;
  const raw = glacier?.raw ?? {};
  const candidate =
    (raw.geometry as Record<string, unknown> | undefined) ??
    (raw.geojson as Record<string, unknown> | undefined) ??
    (raw.polygon as Record<string, unknown> | undefined) ??
    (raw.boundary as Record<string, unknown> | undefined) ??
    ((raw.feature as Record<string, unknown> | undefined)?.geometry as Record<string, unknown> | undefined) ??
    ((raw.rgi_geometry as Record<string, unknown> | undefined) ?? undefined);
  const geoJson = candidate
    ? (candidate.type === "Feature" || candidate.type === "FeatureCollection"
        ? candidate
        : { type: "Feature", properties: { name: glacier?.name, rgi_id: glacier?.rgi_id }, geometry: candidate })
    : null;
  const safeJson = JSON.stringify(geoJson).replace(/</g, "\\u003c");
  const title = isOverview ? "Global glacier overview" : `${glacier.name} — glacier boundary`;
  return `<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
html,body,#map{height:100%;width:100%;margin:0;font-family:Arial,sans-serif}
.leaflet-control-layers{border-radius:10px!important}
.map-note{background:#fff;padding:9px 12px;border-radius:8px;box-shadow:0 1px 8px #0002;font-size:12px;max-width:250px;line-height:1.4}
.legend{display:flex;align-items:center;gap:7px}.swatch{width:20px;height:4px;background:#ffd54a;border:1px solid #183d52}
</style></head><body><div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const map=L.map('map',{zoomControl:true}).setView([${lat},${lon}],${zoom});
const street=L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap'});
const imagery=L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{maxZoom:19,attribution:'Tiles © Esri'});
const terrain=L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',{maxZoom:17,attribution:'© OpenTopoMap'});
street.addTo(map);
L.control.layers({'Street map':street,'Satellite imagery':imagery,'Terrain':terrain},null,{collapsed:false}).addTo(map);
if(!${isOverview}){
  const point=L.circleMarker([${lat},${lon}],{radius:6,color:'#ffffff',weight:2,fillColor:'#00b8a9',fillOpacity:1}).addTo(map);
  point.bindPopup(${JSON.stringify(title)}).openPopup();
}
const boundaryData=${safeJson};
if(boundaryData){
  const boundary=L.geoJSON(boundaryData,{style:{color:'#ffd54a',weight:3,opacity:1,fillColor:'#19b9b0',fillOpacity:0.18}});
  boundary.addTo(map);
  const bounds=boundary.getBounds();
  if(bounds.isValid()) map.fitBounds(bounds.pad(0.12),{maxZoom:13});
  L.control.layers(null,{'Glacier boundary':boundary},{collapsed:true}).addTo(map);
  const legend=L.control({position:'bottomleft'});
  legend.onAdd=function(){const d=L.DomUtil.create('div','map-note');d.innerHTML='<div class="legend"><span class="swatch"></span><strong>RGI glacier boundary</strong></div><div style="margin-top:5px">Yellow outline · translucent glacier area</div>';return d;};
  legend.addTo(map);
}else{
  const note=L.control({position:'bottomleft'});
  note.onAdd=function(){const d=L.DomUtil.create('div','map-note');d.innerHTML=${JSON.stringify(isOverview ? '<strong>Global map</strong><br>Search the inventory above to locate a glacier. This view does not display an inventory-wide glacier layer yet.' : '<strong>Glacier location shown</strong><br>Boundary outline is unavailable because the glacier search response did not include polygon geometry. No boundary has been invented.')};return d;};
  note.addTo(map);
}
</script></body></html>`;
}
function App() {
const [historicalResult, setHistoricalResult] =
  useState<Record<string, unknown> | null>(null);
const [historicalLoading, setHistoricalLoading] = useState(false);
const [historicalError, setHistoricalError] = useState("");
const [debrisResult, setDebrisResult] =
  useState<Record<string, unknown> | null>(null);
const [debrisLoading, setDebrisLoading] = useState(false);
const [debrisError, setDebrisError] = useState("");
const [startYear, setStartYear] = useState(2023);
const [endYear, setEndYear] = useState(2026);
  const [page, setPage] = useState<Page>("Overview");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [theme, setTheme] = useState<"light" | "dark" | "system">("system");
  const [showThemeMenu, setShowThemeMenu] = useState(false);
  // Search starts empty. No glacier is selected by default.
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<Glacier[]>([]);
  const [selectedGlacier, setSelectedGlacier] = useState<Glacier | null>(
    null
  );
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState("");
  const [showResults, setShowResults] = useState(false);
  const [activeSearchIndex, setActiveSearchIndex] = useState(0);
  const [notice, setNotice] = useState("");
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysisError, setAnalysisError] = useState("");
  const [analysisResult, setAnalysisResult] = useState<Record<string, unknown> | null>(null);
  // Incremented whenever the user changes study glacier; prevents an older request
  // from restoring results for the previous glacier.
  const analysisGeneration = useRef(0);
  function clearGlacierResults() {
    analysisGeneration.current += 1;
    setAnalysisResult(null);
    setAnalysisError("");
    setAnalysisLoading(false);
    setHistoricalResult(null);
    setHistoricalError("");
    setHistoricalLoading(false);
    setDebrisResult(null);
    setDebrisError("");
    setDebrisLoading(false);
    setNotice("");
  }
async function runDebrisAnalysis() {
  if (debrisLoading) return;
  const requestGeneration = analysisGeneration.current;
  if (!selectedGlacier) {
    setDebrisError("Search for and select a glacier first.");
    return;
  }
  setDebrisLoading(true);
  setDebrisError("");
  try {
    const params = new URLSearchParams({
      glacier: selectedGlacier.name,
      rgi_id: selectedGlacier.rgi_id,
    });
    if (selectedGlacier.latitude != null) {
      params.set("latitude", String(selectedGlacier.latitude));
    }
    if (selectedGlacier.longitude != null) {
      params.set("longitude", String(selectedGlacier.longitude));
    }
    const response = await fetch(
      `${API_BASE_URL}/api/debris/analyze?${params.toString()}`
    );
    const body: unknown = await response.json();
    if (!response.ok) {
      const detail =
        body && typeof body === "object" && "detail" in body
          ? String((body as { detail: unknown }).detail)
          : `Debris analysis failed (${response.status})`;
      throw new Error(detail);
    }
    if (requestGeneration === analysisGeneration.current) {
      setDebrisResult(body as Record<string, unknown>);
    }
  } catch (error) {
    if (requestGeneration === analysisGeneration.current) {
      setDebrisError(
        error instanceof Error
          ? error.message
          : "Debris analysis failed."
      );
    }
  } finally {
    if (requestGeneration === analysisGeneration.current) setDebrisLoading(false);
  }
}
async function runHistoricalAnalysis() {
  if (historicalLoading) return;
  const requestGeneration = analysisGeneration.current;
  if (!selectedGlacier) {
    setHistoricalError("Search for and select a glacier first.");
    return;
  }
  if (startYear > endYear) {
    setHistoricalError("Start year must not exceed end year.");
    return;
  }
  setHistoricalLoading(true);
  setHistoricalError("");
  try {
    const response = await fetch(
      `${API_BASE_URL}/api/historical/analyze`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: selectedGlacier.name,
          rgi_id: selectedGlacier.rgi_id,
          start_year: startYear,
          end_year: endYear,
        }),
      }
    );
    const body: unknown = await response.json();
    if (!response.ok) {
      const detail =
        body && typeof body === "object" && "detail" in body
          ? String((body as { detail: unknown }).detail)
          : `Historical analysis failed (${response.status})`;
      throw new Error(detail);
    }
    if (requestGeneration === analysisGeneration.current) {
      setHistoricalResult(body as Record<string, unknown>);
    }
  } catch (error) {
    if (requestGeneration === analysisGeneration.current) {
      setHistoricalError(
        error instanceof Error
          ? error.message
          : "Historical analysis failed."
      );
    }
  } finally {
    if (requestGeneration === analysisGeneration.current) setHistoricalLoading(false);
  }
}
 async function runGlacierAnalysis() {
  if (analysisLoading) return;
  const requestGeneration = analysisGeneration.current;
  if (!selectedGlacier) {
    setAnalysisError("Search for and select a glacier before starting analysis.");
    setPage("AI Analysis");
    return;
  }
  // Reuse the completed result for this glacier.
  const currentResult = analysisResult;
  const currentRgiId = currentResult
    ? String(
        (currentResult.glacier as Record<string, unknown> | undefined)?.rgi_id ?? ""
      )
    : "";
  if (currentResult && currentRgiId === selectedGlacier.rgi_id) {
    setPage("AI Analysis");
    setSidebarOpen(false);
    return;
  }
  setPage("AI Analysis");
  setSidebarOpen(false);
  setAnalysisLoading(true);
  setAnalysisError("");
  try {
    const params = new URLSearchParams({
      name: selectedGlacier.name,
      rgi_id: selectedGlacier.rgi_id,
    });
    if (selectedGlacier.latitude != null) {
      params.set("latitude", String(selectedGlacier.latitude));
    }
    if (selectedGlacier.longitude != null) {
      params.set("longitude", String(selectedGlacier.longitude));
    }
    const response = await fetch(
      `${API_BASE_URL}/api/live/analyze-live?${params.toString()}`
    );
    const body: unknown = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail =
        body && typeof body === "object" && "detail" in body
          ? String((body as { detail: unknown }).detail)
          : `Analysis failed with HTTP ${response.status}`;
      throw new Error(detail);
    }
    if (requestGeneration === analysisGeneration.current) {
      setAnalysisResult(
        body && typeof body === "object"
          ? (body as Record<string, unknown>)
          : { result: body }
      );
    }
  } catch (error) {
    if (requestGeneration === analysisGeneration.current) {
      setAnalysisError(
        error instanceof Error
          ? error.message
          : "Unable to complete glacier analysis."
      );
    }
  } finally {
    if (requestGeneration === analysisGeneration.current) setAnalysisLoading(false);
  }
}
  useEffect(() => {
    const root = document.documentElement;
    const systemDark = window.matchMedia("(prefers-color-scheme: dark)");
    const applyTheme = () => {
      const resolved = theme === "system" ? (systemDark.matches ? "dark" : "light") : theme;
      root.dataset.theme = resolved;
      root.style.colorScheme = resolved;
    };
    applyTheme();
    if (theme === "system") {
      systemDark.addEventListener("change", applyTheme);
      return () => systemDark.removeEventListener("change", applyTheme);
    }
  }, [theme]);
  useEffect(() => {
    const query = searchQuery.trim();
    const controller = new AbortController();
    // A selected glacier's name stays in the input for context. Do not treat
    // that value as a fresh search, or the dropdown reopens after selection.
    if (
      selectedGlacier &&
      query.toLocaleLowerCase() === selectedGlacier.name.trim().toLocaleLowerCase()
    ) {
      setSearchResults([]);
      setShowResults(false);
      setSearchLoading(false);
      setSearchError("");
      return () => controller.abort();
    }
    if (query.length < 2) {
      setSearchResults([]);
      setSearchLoading(false);
      setSearchError("");
      return () => controller.abort();
    }
    const timer = window.setTimeout(async () => {
      setSearchLoading(true);
      setSearchError("");
      try {
const params = new URLSearchParams({
  query,
  limit: "50",
});
        const response = await fetch(
          `${API_BASE_URL}/api/glaciers/search?${params.toString()}`,
          { signal: controller.signal }
        );
        if (!response.ok) {
          const message = await response.text();
          throw new Error(
            message || `Glacier search failed (${response.status})`
          );
        }
        const data: unknown = await response.json();
        setSearchResults(extractGlaciers(data));
        setShowResults(true);
      } catch (error) {
        if (error instanceof Error && error.name !== "AbortError") {
          setSearchError(
            error.message.includes("Failed to fetch")
              ? "Cannot reach the backend. Make sure FastAPI is running."
              : error.message
          );
          setSearchResults([]);
        }
      } finally {
        if (!controller.signal.aborted) {
          setSearchLoading(false);
        }
      }
    }, 300);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [searchQuery, selectedGlacier]);
  async function selectGlacier(glacier: Glacier) {
    if (selectedGlacier?.rgi_id !== glacier.rgi_id) clearGlacierResults();
    // Select immediately so the UI responds to the first click. Then fetch the
    // full inventory record, which may include the real GeoJSON boundary.
    setSelectedGlacier(glacier);
    setSearchQuery(glacier.name);
    setSearchResults([]);
    setShowResults(false);
    setSearchError("");
    setNotice("");
    if (!glacier.rgi_id || glacier.rgi_id.startsWith("result-")) return;
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/glaciers/${encodeURIComponent(glacier.rgi_id)}`
      );
      if (!response.ok) {
        // Keep the glacier selected even if detail/geometry is unavailable.
        console.warn(`Glacier details unavailable (${response.status}).`);
        return;
      }
      const payload: unknown = await response.json();
      const root = recordOf(payload);
      const details = recordOf(root.glacier ?? root.data ?? root);
      const detailProperties = recordOf(details.properties);
      const merged = { ...glacier.raw, ...detailProperties, ...details };
      const geometryCandidate =
        details.geometry ??
        detailProperties.geometry ??
        details.geojson ??
        details.polygon ??
        details.boundary ??
        details.rgi_geometry ??
        glacier.geometry ??
        glacier.raw.geometry ??
        glacier.raw.geojson ??
        glacier.raw.polygon ??
        glacier.raw.boundary;
      setSelectedGlacier((current) => {
        if (!current || current.rgi_id !== glacier.rgi_id) return current;
        return {
          ...current,
          latitude: asNumber(details.latitude, details.lat, details.cenlat, current.latitude),
          longitude: asNumber(details.longitude, details.lon, details.lng, details.cenlon, current.longitude),
          area_km2: asNumber(details.area_km2, details.area, details.Area, current.area_km2),
          geometry: geometryCandidate && typeof geometryCandidate === "object"
            ? geometryCandidate as Record<string, unknown>
            : null,
          raw: { ...merged, geometry: geometryCandidate },
        };
      });
    } catch (error) {
      console.warn("Could not load glacier geometry; keeping the selected marker.", error);
    }
  }
  function clearGlacierSearch() {
    clearGlacierResults();
    setSearchQuery("");
    setSelectedGlacier(null);
    setSearchResults([]);
    setSearchError("");
    setShowResults(false);
  }
  function selectPage(nextPage: Page) {
    setPage(nextPage);
    setSidebarOpen(false);
    setNotice("");
  }
  function buildReportText() {
    const ai = recordOf(analysisResult);
    const aiGlacier = recordOf(ai.glacier);
    const aiAnalysis = recordOf(ai.analysis);
    const aiSatellite = recordOf(ai.satellite);
    const debris = recordOf(debrisResult);
    const sentinel = recordOf(debris.sentinel);
    const historical = recordOf(historicalResult);
    const change = recordOf(historical.change_analysis);
    const trend = recordOf(change.overall_trend);
    const prediction = recordOf(historical.prediction);
    const predictionRows = Array.isArray(prediction.predictions) ? prediction.predictions.map(recordOf) : [];
    const observationsData = recordOf(historical.data);
    const observations = Array.isArray(observationsData.observations) ? observationsData.observations.map(recordOf) : [];
    return [
      "GLACIERAI — GLACIER MONITORING REPORT",
      `Generated: ${new Date().toLocaleString()}`,
      `Glacier: ${selectedGlacier?.name || asText(aiGlacier.requested_name, debris.glacier, "Not selected")}`,
      `RGI ID: ${selectedGlacier?.rgi_id || asText(aiGlacier.rgi_id, debris.rgi_id, "Unavailable")}`,
      "",
      "1. AI GLACIER SEGMENTATION",
      `Status: ${analysisResult ? asText(ai.status, "Completed") : "No AI analysis in this session"}`,
      `Estimated glacier area: ${displayNumber(asNumber(aiAnalysis.area_km2), 3)} km²`,
      `RGI inventory area: ${displayNumber(asNumber(aiGlacier.rgi_area_km2, selectedGlacier?.area_km2), 3)} km²`,
      `AI − RGI area difference: ${displayNumber((() => { const aiArea = asNumber(aiAnalysis.area_km2); const rgiArea = asNumber(aiGlacier.rgi_area_km2, selectedGlacier?.area_km2); return aiArea != null && rgiArea != null ? aiArea - rgiArea : undefined; })(), 3)} km²`,
      `Coverage: ${displayNumber(asNumber(aiAnalysis.coverage_percent), 2)}%`,
      `Glacier type: ${asText(recordOf(ai.classification).display_name, recordOf(ai.classification).class, "Unavailable")}`,
      `Observation date: ${asText(aiSatellite.observation_date, "Unavailable")}`,
      `Scene ID: ${asText(aiSatellite.scene_id, "Unavailable")}`,
      "",
      "2. DEBRIS ESTIMATION",
      `Status: ${debrisResult ? asText(debris.status, "Completed") : "No debris analysis in this session"}`,
      `Glacier area: ${displayNumber(asNumber(debris.glacier_area_km2), 3)} km²`,
      `Estimated debris area: ${displayNumber(asNumber(debris.debris_area_km2), 5)} km²`,
      `Debris percentage: ${displayNumber(asNumber(debris.debris_percentage), 3)}%`,
      `Debris pixels: ${displayNumber(asNumber(debris.debris_pixels), 0)}`,
      `RGB scene ID: ${asText(sentinel.rgb_scene_id, recordOf(debris.rgb_download).selected_scene_id, "Unavailable")}`,
      `Exact scene match verified: ${sentinel.exact_scene_match_verified === true ? "Yes" : "No"}`,
      asText(debris.scientific_note, "Debris values are spectral estimates and require independent validation."),
      "",
      "3. HISTORICAL AREA OBSERVATIONS",
      ...observations.map((item) => `${asText(item.year)}: ${displayNumber(asNumber(item.area_km2), 3)} km²; date ${asText(item.date, "unavailable")}; scene ${asText(item.scene_id, "unavailable")}`),
      `Retreat trend status: ${asText(trend.status, "Not calculated")}`,
      asText(trend.note, "Historical area estimates require quality validation before retreat conclusions."),
      "",
      "4. FUTURE AREA ESTIMATES",
      ...predictionRows.map((item) => `${asText(item.year)}: ${displayNumber(asNumber(item.predicted_area_km2), 3)} km²`),
      asText(prediction.note, "No prediction available. Predictions are not physical climate forecasts."),
      "",
      "SCIENTIFIC LIMITATION",
      "Model-estimated area and debris are not automatically validated measurements. Do not interpret unvalidated differences as physical glacier retreat.",
    ].join("\n");
  }
  function buildReportHtml() {
    const report = buildReportText();
    const ai = recordOf(analysisResult);
    const outputs = recordOf(ai.outputs);
    const imageItems = [
      ["AI segmentation overlay", outputs.overlay],
      ["Glacier mask", outputs.mask_png],
      ["Probability heatmap", outputs.probability_heatmap],
      ["Probability overlay", outputs.probability_overlay],
      ["RGI reference mask", outputs.rgi_mask],
    ].filter((item) => typeof item[1] === "string" && item[1]);
    const photos = imageItems.length
      ? `<div class="photos">${imageItems.map(([label, path]) => `<figure><img src="${outputUrl(path)}" alt="${String(label)}" onerror="this.style.display='none';this.nextElementSibling.style.display='block'"><div class="missing" style="display:none">Image unavailable</div><figcaption>${String(label)}</figcaption></figure>`).join("")}</div>`
      : `<p class="note">No AI output photos are available in the current analysis result.</p>`;
    const selected = selectedGlacier;
    const profile = selected
      ? `<section><h2>Glacier profile and location</h2><div class="grid">
          <div><span>Glacier</span><strong>${selected.name}</strong></div>
          <div><span>RGI ID</span><strong>${selected.rgi_id}</strong></div>
          <div><span>Region</span><strong>${selected.region || selected.country || "Unavailable"}</strong></div>
          <div><span>Coordinates</span><strong>${selected.latitude != null && selected.longitude != null ? `${selected.latitude.toFixed(5)}, ${selected.longitude.toFixed(5)}` : "Unavailable"}</strong></div>
          <div><span>Inventory area</span><strong>${displayNumber(selected.area_km2, 3)} km²</strong></div>
        </div></section>`
      : "";
    const escapeHtml = (value: string) => value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    const lines = report.split("\n");
    const reportSections: { title: string; lines: string[] }[] = [];
    let current: { title: string; lines: string[] } | null = null;
    for (const line of lines) {
      if (/^\d+\. /.test(line) || line === "SCIENTIFIC LIMITATION") {
        current = { title: line.replace(/^\d+\. /, ""), lines: [] };
        reportSections.push(current);
      } else if (current && line.trim()) current.lines.push(line);
    }
    const sectionsHtml = reportSections.map((section) => {
      const isLimit = section.title === "SCIENTIFIC LIMITATION";
      const rows = section.lines.map((line) => {
        const split = line.indexOf(":");
        if (split > 0 && split < 70) return `<div class="metric"><span>${escapeHtml(line.slice(0, split))}</span><strong>${escapeHtml(line.slice(split + 1).trim())}</strong></div>`;
        return `<p class="detail">${escapeHtml(line)}</p>`;
      }).join("");
      return `<section class="${isLimit ? "limitation" : ""}"><h2>${escapeHtml(section.title)}</h2><div class="metrics">${rows}</div></section>`;
    }).join("");
    return `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
      <title>GlacierAI Monitoring Report</title><style>
      *{box-sizing:border-box}body{margin:0;background:#edf3f6;color:#19364a;font:15px/1.6 Arial,sans-serif}
      main{max-width:1000px;margin:28px auto;padding:0 20px 36px}.hero{padding:28px;border-radius:16px;background:linear-gradient(135deg,#12344a,#087e8b);color:white}
      .eyebrow{font-size:12px;letter-spacing:2px;text-transform:uppercase;opacity:.85}.hero h1{margin:8px 0;font-size:30px}.hero p{margin:5px 0}
      .actions{display:flex;gap:10px;margin:16px 0}.actions button{border:0;border-radius:8px;padding:10px 16px;background:#087e8b;color:white;cursor:pointer}
      section{margin:16px 0;padding:22px;border-radius:14px;background:white;box-shadow:0 2px 12px #12344a0b}h2{margin:0 0 14px;font-size:20px}
      .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}.grid>div{padding:12px;border:1px solid #e0e9ee;border-radius:9px}
      .grid span{display:block;color:#637a88;font-size:12px}.grid strong{display:block;overflow-wrap:anywhere}
      .metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px}.metric{padding:12px;border:1px solid #e0e9ee;border-radius:9px;min-width:0}.metric span{display:block;color:#637a88;font-size:12px;margin-bottom:4px}.metric strong{display:block;overflow-wrap:anywhere;font-weight:600}.detail{grid-column:1/-1;margin:0;padding:12px;background:#f4f8fa;border-radius:8px;overflow-wrap:anywhere}
      .limitation{border:1px solid #e6c36a;background:#fffaf0}.photos{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}
      figure{margin:0;border:1px solid #dce7ec;border-radius:10px;overflow:hidden;background:#f7fafb}figure img{display:block;width:100%;height:auto;min-height:120px;object-fit:contain}figcaption{padding:10px;font-weight:600}
      .note{color:#637a88;font-size:13px}.footer{padding:18px;text-align:center;color:#637a88;font-size:12px}
      @media(max-width:600px){main{padding:0 10px 24px;margin:10px auto}.hero{padding:20px}.hero h1{font-size:24px}}
      @media print{body{background:white}main{max-width:none;margin:0;padding:0}.actions{display:none}section,.hero,figure{box-shadow:none;break-inside:avoid}.hero{border-radius:0}.photos figure{break-inside:avoid}}
      </style></head><body><main><header class="hero"><div class="eyebrow">GlacierAI · Earth observation</div><h1>Glacier Monitoring Report</h1><p>Generated ${new Date().toLocaleString()}</p><p>Model estimates are shown separately from validated measurements.</p></header>
      <div class="actions"><button onclick="window.print()">Print / Save as PDF</button></div>
      ${profile}
      ${sectionsHtml}
      <section><h2>AI output images</h2>${photos}</section>
      <div class="footer">Model-estimated results require validation before being interpreted as measured glacier retreat.</div>
      </main></body></html>`;
  }
  function downloadReport() {
    const html = buildReportHtml();
    const url = URL.createObjectURL(new Blob([html], { type: "text/html;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "GlacierAI-monitoring-report.html";
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function viewReport() {
    const reportWindow = window.open("", "_blank");
    if (!reportWindow) {
      setNotice("Allow pop-ups to view the report, or use Export instead.");
      return;
    }
    reportWindow.document.open();
    reportWindow.document.write(buildReportHtml());
    reportWindow.document.close();
  }
  const descriptions: Record<Page, string> = {
    Overview: "A clear view of glacier monitoring activity.",
    "Global Map": "Search and select glaciers from the global inventory.",
    "AI Analysis": "Prepare a selected glacier for image analysis.",
    History: "Compare historical observations with future glacier-area projections. Forecasts are estimates, not validated retreat measurements.",
    Debris: "Explore debris-covered glacier monitoring.",
    Reports: "Export a workspace report.",
  };
  function GlacierSearch() {
    return (
      <div className="search-wrapper">
        <label className="field-label" htmlFor="glacier-search">
          SEARCH GLOBAL GLACIER INVENTORY
        </label>
        <div className="search-control">
        <div className="search-input-wrap">
          <Search size={19} className="search-input-icon" />
          <input
            id="glacier-search"
            className="form-control glacier-search-input"
            type="text"
            value={searchQuery}
            autoComplete="off"
            spellCheck={false}
            placeholder="Start typing a glacier name, region or RGI ID..."
            role="combobox"
            aria-label="Search the global glacier inventory by glacier name, region or RGI ID"
            aria-autocomplete="list"
            aria-expanded={showResults && (searchResults.length > 0 || searchLoading)}
            aria-controls="glacier-search-results"
            aria-activedescendant={showResults && searchResults[activeSearchIndex] ? `glacier-option-${activeSearchIndex}` : undefined}
            aria-busy={searchLoading}
            onFocus={() => {
              if (searchQuery.trim().length >= 2 && !selectedGlacier) setShowResults(true);
            }}
            onKeyDown={(event) => {
              if (event.key === "ArrowDown" && searchResults.length > 0) {
                event.preventDefault();
                setShowResults(true);
                setActiveSearchIndex((index) => (index + 1) % searchResults.length);
              } else if (event.key === "ArrowUp" && searchResults.length > 0) {
                event.preventDefault();
                setActiveSearchIndex((index) => (index - 1 + searchResults.length) % searchResults.length);
              } else if (event.key === "Enter" && showResults && searchResults[activeSearchIndex]) {
                event.preventDefault();
                void selectGlacier(searchResults[activeSearchIndex]);
              } else if (event.key === "Escape") {
                setShowResults(false);
              }
            }}
            onChange={(event) => {
              const nextQuery = event.target.value;
              if (nextQuery !== searchQuery) clearGlacierResults();
              setSearchQuery(nextQuery);
              setSelectedGlacier(null);
              setActiveSearchIndex(0);
              setShowResults(true);
            }}
          />
          {searchLoading && (
            <LoaderCircle className="search-spinner" size={19} />
          )}
          {searchQuery && (
            <button
              type="button"
              className="icon-button search-clear"
              aria-label="Clear search"
              onClick={clearGlacierSearch}
            >
              <X size={17} />
            </button>
          )}
        </div>
        {showResults && (searchLoading || searchResults.length > 0) && (
          <div className="glacier-search-results" id="glacier-search-results" role="listbox" aria-label="Matching glaciers">
            {searchLoading && searchResults.length === 0 && (
              <div className="search-dropdown-status"><LoaderCircle size={16} className="search-spinner-inline" /> Searching the global glacier inventory...</div>
            )}
            {searchResults.map((glacier, index) => (
              <button
                type="button"
                key={glacier.rgi_id}
                id={`glacier-option-${index}`}
                className={`glacier-result ${activeSearchIndex === index ? "active-result" : ""}`}
                role="option"
                aria-selected={activeSearchIndex === index}
                onMouseEnter={() => setActiveSearchIndex(index)}
                onClick={() => void selectGlacier(glacier)}
              >
                <span className="result-icon">
                  <Mountain size={19} />
                </span>
                <span className="result-info">
                  <strong>{glacier.name}</strong>
                  <small>
                    {glacier.region ||
                      glacier.country ||
                      glacier.rgi_id}
                  </small>
                  <small className="result-rgi">
                    RGI ID: {glacier.rgi_id}
                  </small>
                </span>
                <ChevronRight size={17} />
              </button>
            ))}
          </div>
        )}
        </div>
        {!selectedGlacier && searchQuery.trim().length === 0 && (
          <p className="search-help">
            Enter at least two characters to search the glacier inventory.
          </p>
        )}
        {searchError && (
          <div className="search-message search-error" role="alert">
            {searchError}
          </div>
        )}
        {showResults &&
  searchQuery.trim().length >= 2 &&
  !searchLoading &&
  !searchError &&
  searchResults.length === 0 &&
  !(
    selectedGlacier &&
    selectedGlacier.name.toLowerCase() === searchQuery.trim().toLowerCase()
  ) && (
    <div className="search-message">
      No matching glaciers found. Try another name.
    </div>
  )}

        {selectedGlacier && (
          <div className="selected-glacier">
            <div className="selected-glacier-icon">
              <CircleCheck size={20} />
            </div>
            <div className="selected-glacier-info">
              <strong>{selectedGlacier.name}</strong>
              <span>RGI ID: {selectedGlacier.rgi_id}</span>
              {selectedGlacier.latitude != null &&
                selectedGlacier.longitude != null && (
                  <span>
                    {selectedGlacier.latitude.toFixed(5)},{" "}
                    {selectedGlacier.longitude.toFixed(5)}
                  </span>
                )}
              {selectedGlacier.area_km2 != null && (
                <span>
                  Inventory area: {selectedGlacier.area_km2.toFixed(3)} km²
                </span>
              )}
            </div>
            <button
              type="button"
              className="secondary-button"
              onClick={runGlacierAnalysis}
            >
              Analyze <ArrowUpRight size={15} />
            </button>
          </div>
        )}
      </div>
    );
  }
  return (
    <div className="app-shell">
      {sidebarOpen && (
        <button
          className="mobile-shade"
          aria-label="Close navigation"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        <div className="brand">
          <button type="button" className="brand-home" aria-label="Go to GlacierAI home" title="Go to home" onClick={() => selectPage("Overview")}>
          <div className="brand-icon">
            <Snowflake size={25} />
          </div>
          <div>
            <strong>
              Glacier<span>AI</span>
            </strong>
            <small>EARTH OBSERVATION</small>
          </div>
          </button>
          <button
            className="icon-button sidebar-close"
            aria-label="Close navigation"
            onClick={() => setSidebarOpen(false)}
          >
            <X size={19} />
          </button>
        </div>
        <div className="workspace-label">WORKSPACE</div>
        <nav className="nav-list">
          {navigation.map(({ label, icon: Icon }) => (
            <button
              key={label}
              className={`nav-link ${page === label ? "active" : ""}`}
              aria-current={page === label ? "page" : undefined}
              onClick={() => selectPage(label)}
            >
              <Icon size={19} />
              <span>{label}</span>
              {page === label && (
                <ChevronRight className="nav-chevron" size={16} />
              )}
            </button>
          ))}
        </nav>
      </aside>
      <a className="skip-link" href="#main-content">Skip to content</a>
      <main className="main-area" id="main-content" tabIndex={-1}>
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="icon-button menu-button"
              aria-label="Open navigation"
              onClick={() => setSidebarOpen(true)}
            >
              <Menu size={22} />
            </button>
            <div className="breadcrumb">
              Workspace <span>/</span> <strong>{page}</strong>
            </div>
          </div>
          <div className="topbar-actions">
            <div className="theme-wrapper" style={{ position: "relative" }}>
              <button className="icon-button" type="button" aria-label="Change theme" title="Change theme" onClick={() => setShowThemeMenu((v) => !v)}>
                {theme === "dark" ? <Moon size={19} /> : theme === "light" ? <Sun size={19} /> : <Monitor size={19} />}
              </button>
              {showThemeMenu && <div className="theme-menu" style={{ position: "absolute", right: 0, top: "calc(100% + 8px)", zIndex: 1000, background: "var(--panel, #fff)", border: "1px solid var(--border, #dce7ec)", borderRadius: 12, padding: 8, minWidth: 160, boxShadow: "0 12px 32px #102a4325" }}>
                {([{value:"light",label:"Light",icon:Sun},{value:"dark",label:"Dark",icon:Moon},{value:"system",label:"System",icon:Monitor}] as const).map((item) => <button key={item.value} type="button" className="theme-option" onClick={() => { setTheme(item.value); setShowThemeMenu(false); }} style={{ display:"flex",alignItems:"center",gap:9,width:"100%",padding:"9px 10px",border:0,borderRadius:8,background:theme===item.value?"var(--soft, #edf5f8)":"transparent",color:"var(--text, #19364a)",cursor:"pointer",textAlign:"left" }}><item.icon size={16}/>{item.label}{theme===item.value ? " ✓" : ""}</button>)}
              </div>}
            </div>
          </div>
        </header>
        <div className="page-content">
          <section className="welcome-row">
            <div>
              <div className="eyebrow">
                <span className="eyebrow-line" />
                GLACIER MONITORING PLATFORM
              </div>
              <h1>
                {page === "Overview" ? (
                  <>
                    Monitor the <span>frozen Earth.</span>
                  </>
                ) : (
                  page
                )}
              </h1>
              <p className="page-subtitle">{descriptions[page]}</p>
            </div>
            {page !== "Overview" && page !== "AI Analysis" && (
              <button
                className="primary-button"
                onClick={() => selectPage("AI Analysis")}
              >
                <ScanSearch size={17} /> New analysis
              </button>
            )}
          </section>
          {notice && (
            <div className="notice">
              <span>{notice}</span>
              <button
                onClick={() => setNotice("")}
                aria-label="Dismiss notification"
              >
                <X size={16} />
              </button>
            </div>
          )}
          {page === "Overview" && (
            <>
              <section className="hero-panel">
                <div className="hero-copy">
                  <span className="hero-tag">
                    <Snowflake size={14} /> CRYOSPHERE INTELLIGENCE
                  </span>
                  <h2>
                    Understand change.
                    <br />
                    <span>Protect what remains.</span>
                  </h2>
                  <p>
                    Explore glacier inventory records, satellite imagery
                    and AI-assisted glacier monitoring in one workspace.
                  </p>
                  <div className="hero-actions">
                    <button
                      className="hero-button"
                      onClick={() => selectPage("AI Analysis")}
                    >
                      Start AI analysis <ArrowUpRight size={16} />
                    </button>
                  </div>
                </div>
                <div className="hero-art" aria-hidden="true">
                  <div className="orbit orbit-one" />
                  <div className="orbit orbit-two" />
                  <div className="mountain mountain-back" />
                  <div className="mountain mountain-front" />
                  <div className="mountain-snow snow-back" />
                  <div className="mountain-snow snow-front" />
                  <div className="art-sun" />
                  <div className="art-star star-one">✳</div>
                  <div className="art-star star-two">✳</div>
                  <div className="art-caption">
                    <MapPin size={13} /> GLOBAL OBSERVATION
                  </div>
                </div>
              </section>
              <div className="section-heading lower-heading">
                <div>
                  <h2>Find a glacier</h2>
                  <p>Search the global RGI inventory.</p>
                </div>
              </div>
              <section className="content-card">
                {GlacierSearch()}
              </section>
            </>
          )}
          {page === "Global Map" && (
            <section className="content-card">
              <div className="card-heading">
                <div>
                  <h2>Global glacier map</h2>
                  <p>Explore the world map, then search the global glacier inventory to zoom to a selected glacier.</p>
                </div>
                <Globe2 size={22} />
              </div>
              {GlacierSearch()}
              <div className="real-map-shell" style={{ marginTop: "20px" }}>
                <iframe
                  title={selectedGlacier ? `Interactive map for ${selectedGlacier.name}` : "Interactive global glacier map"}
                  className="real-glacier-map"
                  width="100%"
                  height="520"
                  loading="lazy"
                  style={{ border: 0, display: "block" }}
                  srcDoc={buildGlacierMapDocument(selectedGlacier)}
                />
              </div>
              <div className="map-info-row" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "10px", padding: "4px 2px 0" }}>
                <div>
                  <strong>{selectedGlacier ? selectedGlacier.name : "Global overview"}</strong>
                  <p style={{ margin: "5px 0 0", fontSize: "12px" }}>
                    {selectedGlacier && selectedGlacier.latitude != null && selectedGlacier.longitude != null
                      ? `Coordinates: ${selectedGlacier.latitude.toFixed(4)}, ${selectedGlacier.longitude.toFixed(4)}`
                      : selectedGlacier
                        ? "This inventory record has no coordinates; the map remains at global view."
                        : "Pan and zoom the map. Search above to locate a glacier with recorded coordinates."}
                  </p>
                </div>
                {selectedGlacier && selectedGlacier.latitude != null && selectedGlacier.longitude != null && (
                  <a
                    href={`https://www.openstreetmap.org/?mlat=${selectedGlacier.latitude}&mlon=${selectedGlacier.longitude}#map=9/${selectedGlacier.latitude}/${selectedGlacier.longitude}`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Open larger map ↗
                  </a>
                )}
              </div>
              <p className="scientific-note" style={{ marginTop: "14px", borderRadius: "8px" }}>
                The base map is for geographic context. Glacier markers and boundaries are shown only when supported by inventory data returned by the backend.
              </p>
            </section>
          )}
          {page === "AI Analysis" && (
            <section className="content-card">
              <div className="card-heading">
                <div>
                  <h2>Glacier image analysis</h2>
                  <p>
                    Search and select a glacier before proceeding.
                  </p>
                </div>
                <ScanSearch size={22} />
              </div>
              {GlacierSearch()}
              {selectedGlacier && (
                <div className="data-note">
                </div>
              )}
              {analysisLoading && (
                <div className="analysis-progress" role="status" aria-live="polite">
  <LoaderCircle className="analysis-spinner" size={22} />
  <div className="analysis-progress-content">
    <p className="analysis-progress-title">
      Analysis in progress
    </p>
    <p className="analysis-progress-description">
      Downloading satellite imagery and running glacier segmentation.
      This can take several minutes.
    </p>
  </div>
</div>
              )}
              {analysisError && (
                <div className="analysis-error" role="alert">
                  <strong>Analysis failed</strong>
                  <p>{analysisError}</p>
                  <button className="secondary-button" type="button" onClick={runGlacierAnalysis} disabled={analysisLoading || !selectedGlacier}>Try again</button>
                </div>
              )}
              {analysisResult && (() => {
                const classification = recordOf(analysisResult.classification);
                const glacier = recordOf(analysisResult.glacier);
                const satellite = recordOf(analysisResult.satellite);
                const model = recordOf(analysisResult.model);
                const analysis = recordOf(analysisResult.analysis);
                const outputs = recordOf(analysisResult.outputs);
                const aiAreaKm2 = asNumber(analysis.area_km2);
                const rgiAreaKm2 = asNumber(
                  glacier.rgi_area_km2,
                  selectedGlacier?.area_km2
                );
                const areaDifferenceKm2 =
                  aiAreaKm2 != null && rgiAreaKm2 != null
                    ? aiAreaKm2 - rgiAreaKm2
                    : undefined;
                const areaDifferencePercent =
                  areaDifferenceKm2 != null && rgiAreaKm2 != null && rgiAreaKm2 > 0
                    ? (areaDifferenceKm2 / rgiAreaKm2) * 100
                    : undefined;
                const images = [
                  { key: "mask_png", title: "Glacier mask", description: "Binary segmentation result" },
                  { key: "overlay", title: "Final mask preview", description: "Binary segmentation output; not a satellite composite" },
                  { key: "probability_heatmap", title: "Probability heatmap", description: "Pixel-level glacier probability" },
                  { key: "probability_overlay", title: "Probability overlay", description: "Confidence mapped over imagery" },
                  { key: "rgi_mask", title: "RGI reference mask", description: "Inventory geometry used for analysis" },
                ].map((item) => ({ ...item, src: outputUrl(outputs[item.key]) })).filter((item) => item.src);
                return (
                  <div className="analysis-results">
                    <div className="results-title">
                      <div>
                        <h3>{asText(glacier.requested_name, selectedGlacier?.name, "Glacier analysis")}</h3>
                        <p>{asText(analysisResult.status, "Analysis response received")} · {asText(satellite.source, "Satellite source unavailable")} · {asText(model.name, "Model unavailable")}</p>
                      </div>
                      <span
                        className="success-pill"
                        role="status"
                        style={{
                          background: analysis.complete === true ? undefined : "#fff3cd",
                          color: analysis.complete === true ? undefined : "#805b00",
                        }}
                      >
                        {analysis.complete === true
                          ? "All tiles processed — validation needed"
                          : "Partial result — validation needed"}
                      </span>
                    </div>
                    <div className="analysis-metrics-grid">
                      <article className="analysis-metric">
                        <span>AI-estimated area</span>
                        <strong>{displayNumber(analysis.area_km2)} km²</strong>
                        <small>From current segmentation</small>
                      </article>
                      <article className="analysis-metric">
                        <span>RGI inventory area</span>
                        <strong>
                          {displayNumber(
                            asNumber(glacier.rgi_area_km2, selectedGlacier?.area_km2)
                          )} km²
                        </strong>
                        <small>Reference area from glacier inventory</small>
                      </article>
                      <article className="analysis-metric">
                        <span>Area difference</span>
                        <strong>
                          {areaDifferenceKm2 == null
                            ? "—"
                            : `${areaDifferenceKm2 > 0 ? "+" : ""}${displayNumber(areaDifferenceKm2, 3)} km²`}
                        </strong>
                        <small>AI estimate minus RGI reference</small>
                      </article>
                      <article className="analysis-metric">
                        <span>Percentage difference</span>
                        <strong>
                          {areaDifferencePercent == null
                            ? "—"
                            : `${areaDifferencePercent > 0 ? "+" : ""}${displayNumber(areaDifferencePercent, 2)}%`}
                        </strong>
                        <small>Relative to RGI inventory area</small>
                      </article>
                      <article className="analysis-metric">
                        <span>Glacier coverage</span>
                        <strong>{displayNumber(analysis.coverage_percent, 1)}%</strong>
                        <small>Of analyzed pixels</small>
                      </article>
                      <article className="analysis-metric">
                        <span>Glacier type</span>
                        <strong>{asText(classification.display_name, classification.class, "—")}</strong>
                        <small>{asText(classification.confidence_level, "Confidence")} confidence</small>
                      </article>
                      <article className="analysis-metric">
                        <span>Type confidence</span>
                        <strong>{displayNumber(classification.confidence_percent)}%</strong>
                        <small>{asText(classification.source, "Classification model")}</small>
                      </article>
                    </div>
                    <div className="analysis-inline-note" role="note" style={{ marginTop: "12px" }}>
                      {analysis.complete === true
                        ? "The difference compares the AI-estimated area with the RGI inventory reference. It is not proof of glacier growth or retreat because the inventory and satellite observation may represent different dates, and the model estimate needs independent validation."
                        : "This analysis is incomplete, so the AI-estimated area and its difference from the RGI reference may be underestimated. Wait for all tiles to finish and validate the boundary before drawing conclusions."}
                    </div>
                    <div className="analysis-section">
                      <div className="analysis-section-heading">
                        <div><h4>Segmentation outputs</h4><p>Generated files returned by your backend.</p></div>
                        <span>{images.length} images</span>
                      </div>
                      {images.length > 0 ? (
                        <div className="ai-image-grid">
                          {images.map((item) => (
                            <figure className="ai-image-card" key={item.key}>
                              <a href={item.src} target="_blank" rel="noreferrer" aria-label={`Open ${item.title} image`}>
                                <img src={item.src} alt={item.title} loading="lazy" />
                              </a>
                              <figcaption><strong>{item.title}</strong><span>{item.description}</span></figcaption>
                            </figure>
                          ))}
                        </div>
                      ) : (
                        <div className="analysis-inline-note">The backend returned no displayable image paths for this analysis.</div>
                      )}
                    </div>
                  </div>
                );
              })()}{!analysisLoading && !analysisError && !analysisResult && (
                <div className="empty-state">
                  <div className="empty-icon">
                    <ScanSearch size={25} />
                  </div>
                  <h3>Ready for image analysis</h3>
                <p>
                  Select a glacier above and run analysis. Results from the connected backend will appear here, including segmentation outputs and estimated glacier type.
                </p>
              </div>
              )}
            </section>
          )}
          {page === "History" && (
            <section className="content-card">
              <div className="card-heading">
                <div>
                  <h2>Historical observations</h2>
                  <p>
                    Verified observation dates and measured glacier
                    changes.
                  </p>
                </div>
                <History size={22} />
              </div>
              {GlacierSearch()}
              <div className="history-controls" style={{ display: "flex", flexWrap: "wrap", alignItems: "end", gap: "12px", marginTop: "20px" }}>
                <label className="field-label">
                  START YEAR
                  <select className="form-control" value={startYear} onChange={(event) => setStartYear(Number(event.target.value))} style={{ display: "block", minWidth: "130px", marginTop: "6px" }}>
                    {Array.from({ length: 27 }, (_, i) => 2000 + i).map((year) => <option key={year} value={year}>{year}</option>)}
                  </select>
                </label>
                <label className="field-label">
                  END YEAR
                  <select className="form-control" value={endYear} onChange={(event) => setEndYear(Number(event.target.value))} style={{ display: "block", minWidth: "130px", marginTop: "6px" }}>
                    {Array.from({ length: 27 }, (_, i) => 2000 + i).map((year) => <option key={year} value={year}>{year}</option>)}
                  </select>
                </label>
                <button className="primary-button" type="button" onClick={runHistoricalAnalysis} disabled={historicalLoading || !selectedGlacier}>
                  {historicalLoading ? <><LoaderCircle className="analysis-spinner" size={17} /> Running analysis…</> : <>Run historical analysis <ArrowUpRight size={16} /></>}
                </button>
              </div>
              {!selectedGlacier && <p className="search-help">Search above and select a glacier before running historical analysis.</p>}
              {historicalError && <div className="analysis-error" role="alert"><strong>Historical analysis failed</strong><p>{historicalError}</p></div>}
              {historicalLoading && <div className="analysis-progress" role="status"><LoaderCircle className="analysis-spinner" size={22} /><div><p className="analysis-progress-title">Historical analysis in progress</p><p className="analysis-progress-description">Loading historical observations, change analysis and future prediction. This may take several minutes.</p></div></div>}
              {historicalResult && (
                <div className="analysis-results" style={{ marginTop: "22px" }}>
                  <div className="results-title"><div><h3>{asText(recordOf(historicalResult.glacier).name, selectedGlacier?.name, "Historical analysis")}</h3><p>{startYear}–{endYear} · {asText(historicalResult.status, "Response received")}</p></div><span className="success-pill"><CircleCheck size={14} /> Results received</span></div>
                  {(["data", "change_analysis", "prediction", "glims_reference"] as const).map((key) => {
                    const value = historicalResult[key];
                    if (value == null) return null;
                    return <section className="analysis-detail-card" key={key} style={{ marginTop: "14px" }}>
                      <h4>{({ data: "Historical observations", change_analysis: "Change analysis", prediction: "Future prediction", glims_reference: "GLIMS reference" } as Record<string, string>)[key]}</h4>
                      {key === "data" ? (() => {
                        const dataRecord = recordOf(value);
                        const observations = Array.isArray(dataRecord.observations) ? dataRecord.observations.map(recordOf) : [];
                        const usable = observations.map((item) => ({
                          year: asNumber(item.year),
                          area: asNumber(item.area_km2),
                          date: asText(item.date),
                          scene: asText(item.scene_id),
                        })).filter((item) => item.year != null && item.area != null);
                        const historicalPoints = usable.map((item) => ({ year: item.year as number, area: item.area as number }));
                        const predictionRoot = recordOf(historicalResult?.prediction);
                        const predictionData = recordOf(predictionRoot.data);
                        const predictionStatus = asText(predictionData.status, predictionRoot.status, "unknown").toUpperCase();
                        const forecastSource = Array.isArray(predictionData.predictions)
                          ? predictionData.predictions
                          : Array.isArray(predictionRoot.predictions)
                            ? predictionRoot.predictions
                            : [];
                        const forecastRows = predictionStatus === "SUCCESS"
                          ? forecastSource.map(recordOf).map((item) => ({
                              year: asNumber(item.year),
                              area: asNumber(item.predicted_area_km2 ?? item.area_km2 ?? item.predicted),
                            })).filter((item): item is { year: number; area: number } => item.year != null && item.area != null)
                          : [];
                        const combined = [...historicalPoints, ...forecastRows].sort((a,b) => a.year-b.year);
                        const maxArea = Math.max(1, ...combined.map((item) => item.area));
                        const minYear = Math.min(...combined.map((item) => item.year));
                        const maxYear = Math.max(...combined.map((item) => item.year));
                        const xAt = (year: number) => 52 + (maxYear === minYear ? 0 : ((year - minYear) / (maxYear - minYear)) * 496);
                        const yAt = (area: number) => 174 - (area / maxArea) * 132;
                        const historicalPolyline = historicalPoints.map((item) => `${xAt(item.year)},${yAt(item.area)}`).join(" ");
                        const forecastPolyline = [...(historicalPoints.length ? [historicalPoints[historicalPoints.length-1]] : []), ...forecastRows].map((item) => `${xAt(item.year)},${yAt(item.area)}`).join(" ");
                        const forecastDirection = forecastRows.length > 1 ? forecastRows[forecastRows.length-1].area - forecastRows[0].area : 0;
                        return <>
                          <div className="analysis-metrics-grid">
                            {usable.map((item) => <article className="analysis-metric" key={item.year}><span>{item.year} observation</span><strong>{displayNumber(item.area, 3)} km²</strong><small>{item.date ? new Date(item.date).toLocaleDateString() : "Date unavailable"}</small></article>)}
                          </div>
                          <h4 style={{ marginTop: "18px" }}>Historical area and future retreat projection</h4>
                          {(historicalPoints.length + forecastRows.length) > 1 ? <svg viewBox="0 0 600 235" role="img" aria-label="Historical glacier area and future area projection graph" style={{ width: "100%", maxHeight: "320px", background: "var(--soft, #f7fbfd)", borderRadius: "12px" }}>
                            {[0,1,2,3].map((n) => <g key={n}><line x1="42" x2="570" y1={40+n*42} y2={40+n*42} stroke="var(--border, #dce8ef)" /><text x="35" y={44+n*42} textAnchor="end" fontSize="10" fill="var(--muted, #637a88)">{displayNumber(maxArea*(1-n/3),1)}</text></g>)}
                            {historicalPolyline && <polyline points={historicalPolyline} fill="none" stroke="#1498a8" strokeWidth="4" strokeLinejoin="round" strokeLinecap="round" />}
                            {forecastPolyline && forecastRows.length > 0 && <polyline points={forecastPolyline} fill="none" stroke="#e08a37" strokeWidth="3" strokeDasharray="7 6" strokeLinejoin="round" strokeLinecap="round" />}
                            {historicalPoints.map((item) => <g key={`hist-${item.year}`}><circle cx={xAt(item.year)} cy={yAt(item.area)} r="5" fill="#087e8b" /><text x={xAt(item.year)} y="205" textAnchor="middle" fontSize="11" fill="var(--text, #36556a)">{item.year}</text></g>)}
                            {forecastRows.map((item) => <g key={`future-${item.year}`}><circle cx={xAt(item.year)} cy={yAt(item.area)} r="5" fill="#e08a37" /><text x={xAt(item.year)} y="205" textAnchor="middle" fontSize="11" fill="var(--text, #36556a)">{item.year}</text></g>)}
                            <g><line x1="64" y1="225" x2="86" y2="225" stroke="#1498a8" strokeWidth="4"/><text x="92" y="229" fontSize="11" fill="var(--text, #36556a)">Historical estimates</text><line x1="260" y1="225" x2="282" y2="225" stroke="#e08a37" strokeWidth="3" strokeDasharray="5 4"/><text x="288" y="229" fontSize="11" fill="var(--text, #36556a)">Future projection</text></g>
                          </svg> : <div className="analysis-inline-note">At least two historical or future points are needed to draw the trend chart.</div>}
                          {predictionStatus === "INSUFFICIENT_VALIDATION" && <div className="analysis-inline-note" role="status" style={{ marginTop: 10 }}>Future projection unavailable: fewer than two historical observations passed validation. Historical estimates remain visible for reference, but their changes should not be interpreted as confirmed glacier retreat.</div>}
                          {forecastRows.length > 1 && <div className="analysis-inline-note" style={{ marginTop: 10 }}>{forecastDirection < 0 ? "The future area estimates trend downward, consistent with projected area loss. This is a model projection, not a validated physical retreat measurement." : forecastDirection > 0 ? "Warning: the future area estimates trend upward, so these values do not indicate glacier retreat. Review the prediction method and input observations before interpreting the result." : "The future area estimates are flat; they do not show projected area retreat."}</div>}
                        </>;
                      })() : key === "change_analysis" ? (() => {
                        const change = recordOf(value);
                        const trend = recordOf(change.overall_trend);
                        const rows = Array.isArray(change.year_to_year_area_change) ? change.year_to_year_area_change.map(recordOf) : [];
                        return <>
                          <div className="data-note"><p><strong>Overall trend:</strong> {asText(trend.status, "Not available")}. {asText(trend.note, "Use caution when interpreting historical estimates.")}</p></div>
                          {rows.map((row, index) => <div className="source-row" key={index}><div><strong>{asText(row.from_year)} → {asText(row.to_year)}</strong><p>Area change: {displayNumber(asNumber(row.area_change_km2), 3)} km² · {row.suitable_for_retreat === true ? "Passed retreat quality checks" : "Not validated for physical retreat claims"}</p></div></div>)}
                        </>;
                      })() : key === "prediction" ? (() => {
                        const prediction = recordOf(value);
                        const predictions = Array.isArray(prediction.predictions) ? prediction.predictions.map(recordOf) : [];
                        return <>
                          <p>{asText(prediction.prediction_method, "Trend-based estimate")} · Confidence: {asText(prediction.prediction_confidence, "unknown")}</p>
                          <div className="analysis-metrics-grid">{predictions.map((item) => <article className="analysis-metric" key={String(item.year)}><span>Predicted {asText(item.year)}</span><strong>{displayNumber(asNumber(item.predicted_area_km2), 2)} km²</strong><small>Lower-confidence trend estimate, not a climate forecast</small></article>)}</div>
                          <p className="scientific-note">{asText(prediction.note, "Predictions require validated observations.")}</p>
                        </>;
                      })() : <p>GLIMS reference: {asText(recordOf(value).status, recordOf(value).success === true ? "Available" : "No reference records returned")} · Records: {displayNumber(asNumber(recordOf(value).count) ?? (Array.isArray(recordOf(value).records) ? (recordOf(value).records as unknown[]).length : 0), 0)}</p>}
                    </section>;
                  })}
                </div>
              )}
            </section>
          )}
          {page === "Debris" && (
            <section className="content-card">
              <div className="card-heading">
                <div>
                  <h2>Debris monitoring</h2>
                  <p>
                    Inspect debris analysis for a selected glacier.
                  </p>
                </div>
                <Layers3 size={22} />
              </div>
              {GlacierSearch()}
              <div style={{ marginTop: "20px" }}>
                <button className="primary-button" type="button" onClick={() => { void runDebrisAnalysis(); }} disabled={debrisLoading || !selectedGlacier}>
                  {debrisLoading ? <><LoaderCircle className="analysis-spinner" size={17} /> Analyzing debris…</> : <>Run debris detection <ArrowUpRight size={16} /></>}
                </button>
                {!selectedGlacier && <p className="search-help">Search above and select a glacier before running debris detection.</p>}
              </div>
              {debrisError && <div className="analysis-error" role="alert"><strong>Debris analysis failed</strong><p>{debrisError}</p><button className="secondary-button" type="button" onClick={() => { void runDebrisAnalysis(); }} disabled={debrisLoading || !selectedGlacier}>Try again</button></div>}
              {debrisLoading && <div className="analysis-progress" role="status"><LoaderCircle className="analysis-spinner" size={22} /><div><p className="analysis-progress-title">Debris detection in progress</p><p className="analysis-progress-description">Downloading satellite inputs and generating the debris-analysis output. This may take several minutes.</p></div></div>}
              {debrisResult && (() => {
                const glacierArea = asNumber(debrisResult.glacier_area_km2);
                const debrisArea = asNumber(debrisResult.debris_area_km2);
                const debrisPercent = asNumber(debrisResult.debris_percentage);
                const glacierPixels = asNumber(debrisResult.glacier_pixels);
                const debrisPixels = asNumber(debrisResult.debris_pixels);
                const spectral = recordOf(debrisResult.spectral_statistics);
                const sentinel = recordOf(debrisResult.sentinel);
                const rgbDownload = recordOf(debrisResult.rgb_download);
                const imageCandidates = [
                  { key: "debris_probability_heatmap_url", title: "Debris probability heatmap", description: "Debris probability visualization" },
                  { key: "debris_probability_overlay_url", title: "Debris probability overlay", description: "Probability overlay over imagery" },
                  { key: "debris_full_overlay_url", title: "Debris mask overlay", description: "Detected debris regions" },
                  { key: "rgb_image_url", title: "Satellite image", description: "RGB Sentinel-2 input" },
                ].map((item) => {
                  const slug = (selectedGlacier?.rgi_id || "").toLowerCase().replace(/[.-]/g, "_");
                  const fallbackNames: Record<string, string> = {
                    debris_probability_heatmap_url: `${slug}_debris_probability_heatmap.png`,
                    debris_probability_overlay_url: `${slug}_debris_probability_overlay.png`,
                    debris_full_overlay_url: `${slug}_debris_full_overlay.png`,
                    rgb_image_url: `${slug}_debris_full_rgb.png`,
                  };
                  const returned = outputUrl(debrisResult[item.key]);
                  return { ...item, src: returned || (slug ? `${API_BASE_URL}/datasets/live/predictions/${fallbackNames[item.key]}` : "") };
                });
                return (
                  <div className="analysis-results" style={{ marginTop: "22px" }}>
                    <div className="results-title">
                      <div>
                        <h3>{asText(debrisResult.glacier, selectedGlacier?.name, "Debris detection")}</h3>
                        <p>RGI ID: {asText(debrisResult.rgi_id, selectedGlacier?.rgi_id, "—")}</p>
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
                        <span className="success-pill"><CircleCheck size={14} /> Analysis complete</span>
                        <button
                          className="secondary-button"
                          type="button"
                          onClick={() => { void runDebrisAnalysis(); }}
                          disabled={debrisLoading || !selectedGlacier}
                        >
                          {debrisLoading ? <><LoaderCircle className="analysis-spinner" size={15} /> Re-running…</> : <>Re-run analysis <ArrowUpRight size={15} /></>}
                        </button>
                      </div>
                    </div>
                    <div className="analysis-metrics-grid">
                      <article className="analysis-metric">
                        <span>Estimated glacier area</span>
                        <strong>{displayNumber(glacierArea, 3)} km²</strong>
                        <small>{displayNumber(glacierPixels, 0)} glacier pixels</small>
                      </article>
                      <article className="analysis-metric">
                        <span>Estimated debris area</span>
                        <strong>{displayNumber(debrisArea, 3)} km²</strong>
                        <small>{displayNumber(debrisPixels, 0)} debris pixels</small>
                      </article>
                      <article className="analysis-metric">
                        <span>Debris percentage</span>
                        <strong>{displayNumber(debrisPercent, 2)}%</strong>
                        <small>Of the detected glacier mask</small>
                      </article>
                      <article className="analysis-metric">
                        <span>Satellite resolution</span>
                        <strong>{displayNumber(asNumber(debrisResult.resolution_m), 2)} m</strong>
                        <small>{asText(debrisResult.crs, "CRS unavailable")}</small>
                      </article>
                    </div>
                    <section className="analysis-section">
                      <div className="analysis-section-heading">
                        <div><h4>Debris visualizations</h4><p>Generated debris maps from the latest analysis.</p></div>
                        <span>{imageCandidates.length} images</span>
                      </div>
                      {imageCandidates.length > 0 ? (
                        <div className="ai-image-grid">
                          {imageCandidates.map((item) => (
                            <figure className="ai-image-card" key={item.key}>
                              <a href={item.src} target="_blank" rel="noreferrer" aria-label={`Open ${item.title} image`}>
                                <img src={item.src} alt={item.title} loading="lazy" />
                              </a>
                              <figcaption><strong>{item.title}</strong><span>{item.description}</span></figcaption>
                            </figure>
                          ))}
                        </div>
                      ) : (
                        <div className="analysis-inline-note">
                          If an image does not load, its file may not exist at the expected output path for this glacier.
                        </div>
                      )}
                    </section>
                    <div className="analysis-detail-grid">
                      <section className="analysis-detail-card">
                        <h4>Observation details</h4>
                        <dl>
                          <div><dt>RGB scene ID</dt><dd>{asText(sentinel.rgb_scene_id, rgbDownload.selected_scene_id, "—")}</dd></div>
                          <div><dt>Acquisition date</dt><dd>{asText(sentinel.rgb_acquisition_datetime, rgbDownload.selected_datetime, "—")}</dd></div>
                          <div><dt>Cloud cover</dt><dd>{asNumber(rgbDownload.selected_cloud_cover) != null ? `${displayNumber(asNumber(rgbDownload.selected_cloud_cover))}%` : "—"}</dd></div>
                          <div><dt>Coordinate reference</dt><dd>{asText(debrisResult.crs, "—")}</dd></div>
                          <div><dt>Exact scene match verified</dt><dd>{sentinel.exact_scene_match_verified === true ? "Yes" : "No"}</dd></div>
                        </dl>
                      </section>
                      <section className="analysis-detail-card">
                        <h4>Spectral statistics</h4>
                        <dl>
                          <div><dt>Mean B08 (NIR)</dt><dd>{displayNumber(asNumber(spectral.mean_b08), 4)}</dd></div>
                          <div><dt>Mean B11 (SWIR1)</dt><dd>{displayNumber(asNumber(spectral.mean_b11), 4)}</dd></div>
                          <div><dt>Mean B12 (SWIR2)</dt><dd>{displayNumber(asNumber(spectral.mean_b12), 4)}</dd></div>
                          <div><dt>Mean SWIR1/NIR</dt><dd>{displayNumber(asNumber(spectral.mean_swir1_nir), 4)}</dd></div>
                          <div><dt>Mean SWIR2/NIR</dt><dd>{displayNumber(asNumber(spectral.mean_swir2_nir), 4)}</dd></div>
                        </dl>
                      </section>
                    </div>
                    <p className="scientific-note">
                      {asText(debrisResult.scientific_note, "Debris values are spectral estimates and require independent validation before scientific interpretation.")}
                      {sentinel.exact_scene_match_verified !== true ? " The spectral downloader has not independently verified an exact match to the RGB scene." : ""}
                    </p>
                  </div>
                );
              })()}
            </section>
          )}
          {page === "Reports" && (
            <section className="content-card">
              <div className="card-heading">
                <div>
                  <h2>Monitoring reports</h2>
                  <p>
                    View and export the latest available glacier analysis results.
                  </p>
                </div>
                <FileText size={22} />
              </div>
              <div className="report-row">
                <div className="report-icon">
                  <FileText size={22} />
                </div>
                <div>
                  <strong>Glacier monitoring report</strong>
                  <p>Includes available AI segmentation, debris estimates, historical observations, retreat-quality status, predictions and observation metadata.</p>
                </div>
                <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
                  <button className="secondary-button" onClick={viewReport}><FileText size={16} /> View report</button>
                  <button className="primary-button" onClick={downloadReport}><Download size={16} /> Export report</button>
                </div>
              </div>
            </section>
          )}
          <footer className="footer">
            <div className="footer-brand">
              <Snowflake size={17} /> GlacierAI
            </div>
            <span>AI-based glacier monitoring and retreat analysis</span>
            <span>Research workspace</span>
          </footer>
        </div>
      </main>
    </div>
  );
}
export default App;
