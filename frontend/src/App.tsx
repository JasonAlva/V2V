import {
  CarFront,
  Gauge,
  List,
  LocateFixed,
  Map as MapIcon,
  Pause,
  Play,
  RefreshCcw,
  Rows3,
  SkipBack,
  SkipForward,
  Wifi,
  WifiOff,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Badge } from "./components/ui/badge";
import { Button } from "./components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./components/ui/card";

type Vehicle = {
  id: string;
  x: number;
  y: number;
  heading_rad?: number;
  width?: number;
  speed_ms?: number;
};

type FrameData = {
  step: number;
  sim_time: number;
  ego?: Vehicle;
  coop?: Vehicle;
  all_vehicles: Vehicle[];
};

type LiveManifest = {
  frame_index: number;
  sim_time: number;
  vehicles: string[];
  ego_id?: string;
  coop_id?: string;
  frame_json: string;
  bev_image: string;
  fused_json: string;
  dashboards: Record<string, string>;
  updated_at: number;
  bev_by_observer?: Record<string, string>;
};

type VehicleStatus = "ego" | "visible" | "blind" | "v2v";

type SumoMapLane = {
  id: string;
  points: Array<[number, number]>;
};

type SumoMapJunction = {
  id: string;
  x: number;
  y: number;
};

type SumoMap = {
  bounds: {
    minX: number;
    minY: number;
    maxX: number;
    maxY: number;
  };
  lanes: SumoMapLane[];
  junctions: SumoMapJunction[];
};

const MAX_FRAME = 1000;
const FOV_HALF_RAD = Math.PI / 3;
const SIGHT_RANGE_M = 120;
const MAP_FOCUS_RANGE_M = 200;
const DEFAULT_MAP_BOUNDS = { minX: 0, minY: 0, maxX: 400, maxY: 400 };

function parseSumoNet(xmlText: string): SumoMap | null {
  const parser = new DOMParser();
  const doc = parser.parseFromString(xmlText, "application/xml");
  const parseError = doc.querySelector("parsererror");
  if (parseError) return null;

  const location = doc.querySelector("location");
  const convBoundary = location?.getAttribute("convBoundary") ?? "";
  const boundsParts = convBoundary.split(",").map((value) => Number(value));
  const bounds =
    boundsParts.length === 4 &&
    boundsParts.every((value) => Number.isFinite(value))
      ? {
          minX: boundsParts[0],
          minY: boundsParts[1],
          maxX: boundsParts[2],
          maxY: boundsParts[3],
        }
      : DEFAULT_MAP_BOUNDS;

  const lanes: SumoMapLane[] = [];
  const edges = Array.from(doc.querySelectorAll("edge"));
  for (const edge of edges) {
    if (edge.getAttribute("function") === "internal") continue;
    const laneNodes = Array.from(edge.querySelectorAll("lane"));
    for (const lane of laneNodes) {
      const shape = lane.getAttribute("shape") ?? "";
      if (!shape) continue;
      const points = shape
        .trim()
        .split(" ")
        .map(
          (pair) =>
            pair.split(",").map((value) => Number(value)) as [number, number],
        )
        .filter((pair) => Number.isFinite(pair[0]) && Number.isFinite(pair[1]));
      if (points.length < 2) continue;
      lanes.push({ id: lane.getAttribute("id") ?? "lane", points });
    }
  }

  const junctions: SumoMapJunction[] = Array.from(
    doc.querySelectorAll("junction"),
  )
    .filter((junction) => !junction.getAttribute("id")?.startsWith(":"))
    .map((junction) => ({
      id: junction.getAttribute("id") ?? "junction",
      x: Number(junction.getAttribute("x")),
      y: Number(junction.getAttribute("y")),
    }))
    .filter(
      (junction) => Number.isFinite(junction.x) && Number.isFinite(junction.y),
    );

  return { bounds, lanes, junctions };
}

function toLocal(
  observer: Vehicle,
  target: Vehicle,
): { forward: number; lateral: number; dist: number } {
  const dx = target.x - observer.x;
  const dy = target.y - observer.y;
  const h = observer.heading_rad ?? 0;
  const forward = dx * Math.cos(h) + dy * Math.sin(h);
  const lateral = -dx * Math.sin(h) + dy * Math.cos(h);
  return { forward, lateral, dist: Math.hypot(dx, dy) };
}

function hasLineOfSight(
  observer: Vehicle,
  target: Vehicle,
  vehicles: Vehicle[],
): boolean {
  const rel = toLocal(observer, target);
  if (rel.forward <= 0 || rel.dist > SIGHT_RANGE_M) {
    return false;
  }

  const angle = Math.atan2(rel.lateral, rel.forward);
  if (Math.abs(angle) > FOV_HALF_RAD) {
    return false;
  }

  const tx = target.x - observer.x;
  const ty = target.y - observer.y;
  const targetDistSq = tx * tx + ty * ty;
  if (targetDistSq < 1e-6) {
    return true;
  }

  for (const blocker of vehicles) {
    if (blocker.id === observer.id || blocker.id === target.id) {
      continue;
    }

    const bx = blocker.x - observer.x;
    const by = blocker.y - observer.y;
    const projection = (bx * tx + by * ty) / targetDistSq;

    if (projection <= 0 || projection >= 1) {
      continue;
    }

    const closestX = projection * tx;
    const closestY = projection * ty;
    const perpDist = Math.hypot(bx - closestX, by - closestY);
    const blockerWidth = blocker.width ?? 2;
    const targetWidth = target.width ?? 2;
    const blockRadius = (blockerWidth + targetWidth) * 0.45;

    if (perpDist < blockRadius) {
      return false;
    }
  }

  return true;
}

function statusColor(status: VehicleStatus): string {
  if (status === "ego") return "bg-cyan-400 border-cyan-100";
  if (status === "visible") return "bg-emerald-500 border-emerald-100";
  if (status === "v2v") return "bg-amber-300 border-yellow-50";
  return "bg-rose-500 border-rose-100";
}

function statusText(status: VehicleStatus): string {
  if (status === "ego") return "EGO";
  if (status === "visible") return "VISIBLE";
  if (status === "v2v") return "V2V";
  return "BLIND";
}

type WsStatus = "connected" | "reconnecting" | "off";

// Resolve the WebSocket URL:
// • In dev/prod we go through the Vite proxy (/ws → port 8765) so we stay
//   on the same origin and avoid CORS / port-blocking issues entirely.
// • Override by setting VITE_WS_URL (e.g. ws://myserver:8765/ws/live).
const DEFAULT_WS_URL = (() => {
  const override = import.meta.env.VITE_WS_URL as string | undefined;
  if (override) return override;
  // Build from window.location so it works on any host/port Vite listens on.
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/live`;
})();

function App() {
  const [frameIndex, setFrameIndex] = useState(0);
  const [frameData, setFrameData] = useState<FrameData | null>(null);
  const [live, setLive] = useState(true);
  const [liveManifest, setLiveManifest] = useState<LiveManifest | null>(null);
  const [observerId, setObserverId] = useState("");
  const [assistId, setAssistId] = useState("");
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState("");
  const [wsStatus, setWsStatus] = useState<WsStatus>("reconnecting");
  const [sumoMap, setSumoMap] = useState<SumoMap | null>(null);
  const [mapError, setMapError] = useState("");

  // Monotonically-incrementing counter used as the image cache-buster.
  // Only bumps when a payload arrives that contains a *real* bev_image URL
  // that differs from the previous one, ensuring the browser re-fetches the
  // finished render rather than the stale placeholder.
  const imgRevisionRef = useRef(0);
  const [imgRevision, setImgRevision] = useState(0);
  const lastBevImageRef = useRef("");

  // Track the highest known frame index for replay bounds
  const maxFrameRef = useRef(0);

  // WebSocket ref so we can close it on cleanup without stale closure issues
  const wsRef = useRef<WebSocket | null>(null);
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const retryDelayRef = useRef(1000); // starts at 1 s, doubles up to 16 s

  const connectWs = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.onclose = null; // prevent retry loop firing on deliberate close
      wsRef.current.close();
      wsRef.current = null;
    }

    const ws = new WebSocket(DEFAULT_WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setWsStatus("connected");
      retryDelayRef.current = 1000; // reset back-off on successful connect
    };

    ws.onmessage = (event: MessageEvent<string>) => {
      try {
        const payload = JSON.parse(event.data) as LiveManifest;
        if (payload.frame_index > maxFrameRef.current) {
          maxFrameRef.current = payload.frame_index;
        }
        console.log(
          `[WS] frame=${payload.frame_index} t=${payload.sim_time?.toFixed(2)}s` +
          ` vehicles=[${(payload.vehicles ?? []).join(", ")}]` +
          ` bev=${payload.bev_image ?? "(none)"}` +
          ` dashboards=${Object.keys(payload.dashboards ?? {}).join(", ") || "(none)"}` +
          ` bev_by_observer=${Object.keys(payload.bev_by_observer ?? {}).join(", ") || "(none)"}`
        );
        setLiveManifest(payload);
        setFrameIndex(payload.frame_index);
        // Only bump the image revision when a *real* BEV render has landed.
        // The quick pre-render payload has an empty bev_by_observer dict and
        // a placeholder bev_image path that may not exist yet.  The finished
        // payload (pushed by _heavy_work) carries real dashboards/bev_by_observer
        // entries AND a distinct bev_image.  Using the presence of dashboards or
        // bev_by_observer content as the signal prevents the browser from
        // hammering 404s on in-flight paths.
        const hasRealRender =
          (payload.bev_image &&
            payload.bev_image !== lastBevImageRef.current) ||
          Object.keys(payload.dashboards ?? {}).length > 0 ||
          Object.keys(payload.bev_by_observer ?? {}).length > 0;
        if (hasRealRender) {
          lastBevImageRef.current = payload.bev_image ?? "";
          imgRevisionRef.current += 1;
          setImgRevision(imgRevisionRef.current);
          console.log(
            `[WS] 🖼️  Real render detected — bumping image revision to ${imgRevisionRef.current}` +
            `  bev=${payload.bev_image ?? "(none)"}` +
            `  dashKeys=[${Object.keys(payload.dashboards ?? {}).join(", ")}]`
          );
        }
      } catch {
        // malformed frame — ignore
      }
    };

    ws.onclose = () => {
      setWsStatus("reconnecting");
      wsRef.current = null;
      // Exponential back-off: 1 s → 2 s → 4 s → … → 16 s max
      const delay = retryDelayRef.current;
      retryDelayRef.current = Math.min(delay * 2, 16_000);
      retryTimerRef.current = setTimeout(connectWs, delay);
    };

    ws.onerror = () => {
      // onclose fires right after onerror — let it handle reconnection
      ws.close();
    };
  }, []);

  // Start / stop WebSocket based on `live` toggle
  useEffect(() => {
    if (!live) {
      // Tear down WS gracefully when user turns off live mode
      if (retryTimerRef.current) {
        clearTimeout(retryTimerRef.current);
        retryTimerRef.current = null;
      }
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
        wsRef.current = null;
      }
      setWsStatus("off");
      return;
    }

    connectWs();

    return () => {
      if (retryTimerRef.current) clearTimeout(retryTimerRef.current);
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [live, connectWs]);

  // Fallback HTTP polling — always runs on mount to jump to the latest saved frame,
  // and keeps polling while live mode is on (as a safety net beside WS).
  // When WS is connected we still poll but at a slower rate (1 s) solely to
  // detect finished renders that the WS quick-payload missed.  We do NOT
  // override frameIndex while WS is connected to avoid fighting the WS feed.
  const wsStatusRef = useRef<WsStatus>("reconnecting");
  wsStatusRef.current = wsStatus; // keep ref in sync without adding to deps

  useEffect(() => {
    let cancelled = false;

    async function pollLatestManifest() {
      try {
        const response = await fetch(`/live/latest.json?ts=${Date.now()}`);
        if (!response.ok) return;
        const payload = (await response.json()) as LiveManifest;
        if (cancelled) return;
        if (payload.frame_index > maxFrameRef.current) {
          maxFrameRef.current = payload.frame_index;
        }
        // When WS is connected, only update the manifest for image refresh;
        // do NOT reset frameIndex so we don't fight the WS stream.
        if (wsStatusRef.current !== "connected") {
          setFrameIndex(payload.frame_index);
        }
        setLiveManifest(payload);
        // Bump image revision if a real render has landed.
        const hasRealRender =
          (payload.bev_image &&
            payload.bev_image !== lastBevImageRef.current) ||
          Object.keys(payload.dashboards ?? {}).length > 0 ||
          Object.keys(payload.bev_by_observer ?? {}).length > 0;
        if (hasRealRender) {
          lastBevImageRef.current = payload.bev_image ?? "";
          imgRevisionRef.current += 1;
          setImgRevision(imgRevisionRef.current);
          console.log(
            `[POLL] 🖼️  Real render detected — revision=${imgRevisionRef.current}` +
            `  frame=${payload.frame_index}` +
            `  bev=${payload.bev_image ?? "(none)"}` +
            `  dashKeys=[${Object.keys(payload.dashboards ?? {}).join(", ")}]`
          );
        } else {
          console.log(
            `[POLL] frame=${payload.frame_index} t=${payload.sim_time?.toFixed(2)}s — no new render` +
            ` (wsStatus=${wsStatusRef.current})`
          );
        }
      } catch {
        // Keep the last visible frame if the live manifest is not ready yet.
      }
    }

    // Always poll once on mount to initialise to the latest frame
    pollLatestManifest();

    if (!live) {
      return () => {
        cancelled = true;
      };
    }

    // In live mode: poll every 500 ms as safety net (slower when WS is active)
    const timer = window.setInterval(pollLatestManifest, 500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [live]);

  useEffect(() => {
    let cancelled = false;

    async function loadMap() {
      try {
        setMapError("");
        const response = await fetch(`/sumo_map.net.xml?ts=${Date.now()}`);
        if (!response.ok) {
          throw new Error("SUMO map not found");
        }
        const xmlText = await response.text();
        if (cancelled) return;
        const parsed = parseSumoNet(xmlText);
        if (!parsed) {
          throw new Error("Failed to parse SUMO map");
        }
        setSumoMap(parsed);
      } catch (err) {
        if (!cancelled) {
          setMapError((err as Error).message);
        }
      }
    }

    loadMap();
    return () => {
      cancelled = true;
    };
  }, []);

  // Replay animation timer — only active in non-live mode when playing
  useEffect(() => {
    if (live || !playing) return;
    const timer = window.setInterval(() => {
      setFrameIndex((value) => {
        const max = Math.max(maxFrameRef.current, 1);
        return value >= max ? 0 : value + 1;
      });
    }, 220);
    return () => window.clearInterval(timer);
  }, [playing, live]);

  useEffect(() => {
    let cancelled = false;
    const fileName = `/frames/frame_${frameIndex.toString().padStart(6, "0")}.json`;

    async function loadFrame() {
      try {
        setError("");
        console.log(`[MAP] Loading frame JSON: ${fileName}`);
        const response = await fetch(`${fileName}?ts=${Date.now()}`);
        if (!response.ok) {
          throw new Error(`Frame ${frameIndex} not found (${response.status})`);
        }

        const json = (await response.json()) as FrameData;

        if (cancelled) return;

        console.log(
          `[MAP] ✅ Frame ${json.step} loaded` +
          `  t=${json.sim_time.toFixed(2)}s` +
          `  vehicles=[${json.all_vehicles.map((v) => v.id).join(", ")}]` +
          `  ego=${json.ego?.id ?? "(none)"}` +
          `  coop=${json.coop?.id ?? "(none)"}`
        );

        setFrameData(json);
        const ids = new Set(json.all_vehicles.map((vehicle) => vehicle.id));

        setObserverId((current) => {
          if (current && ids.has(current)) return current;
          if (json.ego && ids.has(json.ego.id)) return json.ego.id;
          return json.all_vehicles[0]?.id ?? "";
        });

        setAssistId((current) => {
          if (current && ids.has(current)) return current;
          if (json.coop && ids.has(json.coop.id)) return json.coop.id;
          return "";
        });
      } catch (err) {
        if (!cancelled) {
          console.warn(`[MAP] ❌ Failed to load frame ${frameIndex}:`, (err as Error).message);
          setError((err as Error).message);
          setFrameData(null);
        }
      }
    }

    loadFrame();
    return () => {
      cancelled = true;
    };
  }, [frameIndex]);

  const vehicles = frameData?.all_vehicles ?? [];
  const observer =
    vehicles.find((vehicle) => vehicle.id === observerId) ?? null;
  const assistVehicle =
    assistId && assistId !== observerId
      ? (vehicles.find((vehicle) => vehicle.id === assistId) ?? null)
      : null;

  const statusByVehicleId = useMemo(() => {
    const result = new Map<string, VehicleStatus>();
    if (!observer) return result;

    for (const target of vehicles) {
      if (target.id === observer.id) {
        result.set(target.id, "ego");
        continue;
      }

      const visibleFromObserver = hasLineOfSight(observer, target, vehicles);
      if (visibleFromObserver) {
        result.set(target.id, "visible");
        continue;
      }

      const visibleFromAssist = assistVehicle
        ? hasLineOfSight(assistVehicle, target, vehicles)
        : false;
      result.set(target.id, visibleFromAssist ? "v2v" : "blind");
    }

    return result;
  }, [assistVehicle, observer, vehicles]);

  const observerDashboard =
    liveManifest?.dashboards?.[observerId] ??
    (frameData?.ego?.id === observerId
      ? `/images/frame_${frameIndex.toString().padStart(6, "0")}_ego.png`
      : undefined) ??
    (frameData?.coop?.id === observerId
      ? `/images/frame_${frameIndex.toString().padStart(6, "0")}_coop.png`
      : undefined);

  const bevImage =
    (observerId ? liveManifest?.bev_by_observer?.[observerId] : undefined) ??
    liveManifest?.bev_image ??
    `/fused/bev/frame_${frameIndex.toString().padStart(6, "0")}_bev.png`;

  // imgRevision is our cache-buster: it only increments when a finished
  // render payload (with real BEV/dashboard paths) has been received.
  const liveRefreshKey = imgRevision;

  const selectedStatus = observer
    ? (statusByVehicleId.get(observer.id) ?? "blind")
    : null;

  const mapBounds = sumoMap?.bounds ?? DEFAULT_MAP_BOUNDS;
  const focusCenterX = observer?.x ?? (mapBounds.minX + mapBounds.maxX) / 2;
  const focusCenterY = observer?.y ?? (mapBounds.minY + mapBounds.maxY) / 2;
  const focusHalfSpan = observer
    ? MAP_FOCUS_RANGE_M / 2
    : Math.max(
        mapBounds.maxX - mapBounds.minX,
        mapBounds.maxY - mapBounds.minY,
      ) / 2;

  // Always show the full SUMO map bounds (zoomed-out view)
  const mapViewBounds = mapBounds;

  const mapWidth = Math.max(1, mapViewBounds.maxX - mapViewBounds.minX);
  const mapHeight = Math.max(1, mapViewBounds.maxY - mapViewBounds.minY);

  const mapX = (x: number) => ((x - mapViewBounds.minX) / mapWidth) * 100;
  const mapY = (y: number) => ((mapViewBounds.maxY - y) / mapHeight) * 100;

  const toSvgPoint = (x: number, y: number) => ({
    x: x - mapViewBounds.minX,
    y: mapViewBounds.maxY - y,
  });

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-[1500px] flex-col gap-4 px-3 py-4 md:px-6 md:py-6">
      <Card className="border-slate-700/70 bg-slate-900/80 backdrop-blur">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-xl text-slate-100 md:text-2xl">
            <CarFront className="h-6 w-6 text-cyan-300" />
            Realtime SUMO V2V Dashboard
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-wrap items-center gap-2 text-slate-200">
          <Button
            variant={live ? "default" : "outline"}
            onClick={() => setLive((value) => !value)}
          >
            <RefreshCcw className="mr-1 h-4 w-4" />
            {live ? "Live ON" : "Live OFF"}
          </Button>

          {/* Fit Map control removed — map now always shows full SUMO bounds */}

          {/* WebSocket connection status indicator */}
          {live && (
            <span
              className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${
                wsStatus === "connected"
                  ? "bg-emerald-900/50 text-emerald-300 ring-1 ring-emerald-500/40"
                  : "bg-amber-900/50 text-amber-300 ring-1 ring-amber-500/40"
              }`}
              aria-label={`WebSocket status: ${wsStatus}`}
              title={
                wsStatus === "connected"
                  ? `Connected to ${DEFAULT_WS_URL}`
                  : "Reconnecting…"
              }
            >
              {wsStatus === "connected" ? (
                <Wifi className="h-3 w-3" />
              ) : (
                <WifiOff className="h-3 w-3 animate-pulse" />
              )}
              {wsStatus === "connected" ? "WS Connected" : "Reconnecting…"}
            </span>
          )}

          {/* Replay controls — show when: not in live mode, OR live mode but WS disconnected (sim ended/stopped) */}
          {(!live || wsStatus === "reconnecting") && (
            <>
              <Button
                variant="outline"
                onClick={() => setFrameIndex((value) => Math.max(0, value - 1))}
              >
                <SkipBack className="mr-1 h-4 w-4" />
                Prev
              </Button>
              <Button onClick={() => setPlaying((value) => !value)}>
                {playing ? (
                  <Pause className="mr-1 h-4 w-4" />
                ) : (
                  <Play className="mr-1 h-4 w-4" />
                )}
                {playing ? "Pause" : "Replay"}
              </Button>
              <Button
                variant="outline"
                onClick={() =>
                  setFrameIndex((value) =>
                    Math.min(maxFrameRef.current, value + 1),
                  )
                }
              >
                <SkipForward className="mr-1 h-4 w-4" />
                Next
              </Button>
              <label
                className="ml-1 text-sm text-slate-300"
                htmlFor="frame-input"
              >
                Frame
              </label>
              <input
                id="frame-input"
                className="w-24 rounded-md border border-slate-600 bg-slate-950 px-2 py-1 text-sm text-slate-100"
                type="number"
                min={0}
                max={maxFrameRef.current || MAX_FRAME}
                value={frameIndex}
                onChange={(event) =>
                  setFrameIndex(
                    Math.max(
                      0,
                      Math.min(
                        maxFrameRef.current || MAX_FRAME,
                        Number(event.target.value) || 0,
                      ),
                    ),
                  )
                }
              />
              <span className="text-xs text-slate-400">
                / {maxFrameRef.current}
              </span>
            </>
          )}

          <Badge className="ml-auto" aria-label="Map interaction hint">
            Click any vehicle marker or use vehicle list to select
          </Badge>
        </CardContent>
      </Card>

      <div className="grid gap-4 lg:grid-cols-[2.1fr,1fr]">
        <Card className="border-slate-700/70 bg-slate-900/80 backdrop-blur lg:row-span-2">
          <CardContent className="p-3 md:p-4">
            <div className="grid gap-2 pb-3 text-sm text-slate-200 md:grid-cols-6">
              <Badge
                variant="outline"
                className="justify-center border-cyan-300 text-cyan-200"
              >
                Observer: {observer?.id ?? "none"}
              </Badge>
              <Badge
                variant="outline"
                className="justify-center border-emerald-300 text-emerald-200"
              >
                Visible
              </Badge>
              <Badge
                variant="outline"
                className="justify-center border-rose-300 text-rose-200"
              >
                Blind
              </Badge>
              <Badge
                variant="outline"
                className="justify-center border-amber-200 text-amber-100"
              >
                V2V detected
              </Badge>
              <label htmlFor="assist-select" className="sr-only">
                Choose assist vehicle
              </label>
              <select
                id="assist-select"
                className="rounded-md border border-slate-600 bg-slate-950 px-2 py-1 text-sm text-slate-100"
                value={assistId}
                onChange={(event) => setAssistId(event.target.value)}
                aria-label="Assist vehicle selector"
              >
                <option value="">No assist car</option>
                {vehicles
                  .filter((vehicle) => vehicle.id !== observerId)
                  .map((vehicle) => (
                    <option key={vehicle.id} value={vehicle.id}>
                      Assist: {vehicle.id}
                    </option>
                  ))}
              </select>
            </div>

            <div className="relative h-[40vh] min-h-[260px] w-full overflow-hidden rounded-xl border border-slate-700/70 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800">
              <svg
                className="absolute inset-0 h-full w-full"
                viewBox={`0 0 ${mapWidth} ${mapHeight}`}
                preserveAspectRatio="xMidYMid slice"
                role="img"
                aria-label="SUMO top-down map focused on the selected vehicle"
              >
                <rect
                  x={0}
                  y={0}
                  width={mapWidth}
                  height={mapHeight}
                  fill="#0f0f1e"
                  opacity="0.9"
                />

                <g
                  opacity="0.08"
                  stroke="rgba(255,255,255,0.25)"
                  strokeWidth="1"
                >
                  {Array.from({ length: 9 }).map((_, i) => {
                    const x = (mapWidth / 8) * i;
                    return (
                      <line
                        key={`vgrid-${i}`}
                        x1={x}
                        y1={0}
                        x2={x}
                        y2={mapHeight}
                      />
                    );
                  })}
                  {Array.from({ length: 9 }).map((_, i) => {
                    const y = (mapHeight / 8) * i;
                    return (
                      <line
                        key={`hgrid-${i}`}
                        x1={0}
                        y1={y}
                        x2={mapWidth}
                        y2={y}
                      />
                    );
                  })}
                </g>

                <g
                  id="sumo-lanes"
                  stroke="rgba(140,170,200,0.6)"
                  strokeWidth="2"
                  fill="none"
                >
                  {sumoMap?.lanes.map((lane) => (
                    <polyline
                      key={lane.id}
                      points={lane.points
                        .map((point) => {
                          const mapped = toSvgPoint(point[0], point[1]);
                          return `${mapped.x},${mapped.y}`;
                        })
                        .join(" ")}
                    />
                  ))}
                </g>

                <g id="sumo-junctions">
                  {sumoMap?.junctions.map((junction) => {
                    const mapped = toSvgPoint(junction.x, junction.y);
                    return (
                      <g key={junction.id}>
                        <circle
                          cx={mapped.x}
                          cy={mapped.y}
                          r={Math.max(3, mapWidth * 0.01)}
                          fill="rgba(100,150,255,0.65)"
                          stroke="rgba(150,200,255,0.95)"
                          strokeWidth="1.5"
                        />
                        <text
                          x={mapped.x}
                          y={mapped.y - 6}
                          textAnchor="middle"
                          fontSize={Math.max(8, mapWidth * 0.02)}
                          fill="rgba(255,255,255,0.9)"
                          fontWeight="bold"
                        >
                          {junction.id}
                        </text>
                      </g>
                    );
                  })}
                </g>

                {vehicles.map((vehicle) => {
                  const status = statusByVehicleId.get(vehicle.id) ?? "blind";
                  const isSelected = vehicle.id === observerId;

                  let fillColor = "rgba(255,100,100,0.8)";
                  if (status === "ego") fillColor = "rgba(100,200,255,1)";
                  else if (status === "visible")
                    fillColor = "rgba(100,255,100,1)";
                  else if (status === "v2v") fillColor = "rgba(255,200,100,1)";

                  const mapped = toSvgPoint(vehicle.x, vehicle.y);
                  const x = mapped.x;
                  const y = mapped.y;
                  const heading = vehicle.heading_rad ?? 0;

                  const vr = Math.max(2, mapWidth * 0.025); // vehicle radius scales with view
                  const selRing = vr * 1.6;
                  const headLen = vr * 1.4;
                  return (
                    <g key={vehicle.id} opacity={isSelected ? 1 : 0.75}>
                      {isSelected && (
                        <circle
                          cx={x}
                          cy={y}
                          r={selRing}
                          fill="none"
                          stroke="rgba(255,255,100,0.6)"
                          strokeWidth={Math.max(1, mapWidth * 0.006)}
                        />
                      )}
                      <circle
                        cx={x}
                        cy={y}
                        r={vr}
                        fill={fillColor}
                        stroke="rgba(255,255,255,0.8)"
                        strokeWidth={Math.max(0.5, mapWidth * 0.003)}
                      />
                      <line
                        x1={x}
                        y1={y}
                        x2={x + headLen * Math.cos(heading)}
                        y2={y + headLen * Math.sin(heading)}
                        stroke="rgba(255,255,255,0.9)"
                        strokeWidth={Math.max(0.5, mapWidth * 0.003)}
                      />
                    </g>
                  );
                })}
              </svg>

              {vehicles.map((vehicle) => {
                const xPct = mapX(vehicle.x);
                const yPct = mapY(vehicle.y);
                const status = statusByVehicleId.get(vehicle.id) ?? "blind";
                const isSelected = vehicle.id === observerId;

                return (
                  <button
                    key={vehicle.id}
                    type="button"
                    aria-label={`Inspect ${vehicle.id}`}
                    aria-pressed={isSelected}
                    onClick={() => setObserverId(vehicle.id)}
                    className={`absolute -translate-x-1/2 -translate-y-1/2 rounded-full shadow-lg transition hover:scale-125 ${
                      isSelected
                        ? "ring-2 ring-yellow-300 ring-offset-2 ring-offset-slate-900"
                        : ""
                    }`}
                    style={{
                      left: `${xPct}%`,
                      top: `${yPct}%`,
                      width: "32px",
                      height: "32px",
                      zIndex: isSelected ? 40 : 10,
                    }}
                    title={`${vehicle.id} - ${status}`}
                  >
                    <span
                      className={`absolute inset-0 rounded-full border-2 ${statusColor(status)}`}
                    />
                  </button>
                );
              })}

              {observer && (
                <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between rounded-md border border-slate-700/80 bg-slate-950/85 px-3 py-2 text-xs text-slate-200">
                  <span className="inline-flex items-center gap-2">
                    <LocateFixed className="h-3.5 w-3.5 text-cyan-300" />
                    Selected: <strong>{observer.id}</strong>
                  </span>
                  <span className="inline-flex items-center gap-2">
                    <Badge
                      variant="outline"
                      className="border-slate-500 text-slate-200"
                    >
                      {selectedStatus ? statusText(selectedStatus) : "N/A"}
                    </Badge>
                    <span>
                      {observer.speed_ms !== undefined
                        ? `${(observer.speed_ms * 3.6).toFixed(1)} km/h`
                        : "Speed N/A"}
                    </span>
                  </span>
                </div>
              )}
            </div>

            {frameData && (
              <p className="pt-3 text-sm text-slate-300">
                Step {frameData.step} | t={frameData.sim_time.toFixed(2)}s |
                Vehicles: {vehicles.length} | Map focus: full
              </p>
            )}
            {mapError && (
              <p className="pt-2 text-xs text-amber-200">
                Map load: {mapError}
              </p>
            )}
            {error && <p className="pt-3 text-sm text-rose-300">{error}</p>}
          </CardContent>
        </Card>

        <Card className="border-slate-700/70 bg-slate-900/80 backdrop-blur">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm text-slate-100">
              <List className="h-4 w-4 text-slate-300" />
              Vehicles
            </CardTitle>
          </CardHeader>
          <CardContent className="max-h-[50vh] overflow-y-auto p-2">
            <div
              className="space-y-1"
              role="listbox"
              aria-label="Vehicle selector"
            >
              {vehicles.length === 0 ? (
                <p className="text-xs text-slate-400">No vehicles</p>
              ) : (
                vehicles.map((vehicle) => {
                  const status = statusByVehicleId.get(vehicle.id) ?? "blind";
                  const isSelected = vehicle.id === observerId;
                  let statusBg = "bg-rose-900/40";
                  if (status === "ego") statusBg = "bg-cyan-900/40";
                  else if (status === "visible") statusBg = "bg-emerald-900/40";
                  else if (status === "v2v") statusBg = "bg-amber-900/40";

                  return (
                    <button
                      key={vehicle.id}
                      onClick={() => setObserverId(vehicle.id)}
                      aria-selected={isSelected}
                      className={`w-full rounded px-2 py-2 text-left text-sm transition ${
                        isSelected
                          ? "border border-cyan-400 bg-cyan-600/60"
                          : `${statusBg} border border-slate-700 hover:border-slate-500`
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-slate-100">
                          {vehicle.id}
                        </span>
                        <span className="rounded bg-slate-800/80 px-1.5 py-0.5 text-[10px] text-slate-200">
                          {statusText(status)}
                        </span>
                      </div>
                      {vehicle.speed_ms !== undefined && (
                        <div className="mt-1 text-xs text-slate-400">
                          {(vehicle.speed_ms * 3.6).toFixed(1)} km/h
                        </div>
                      )}
                    </button>
                  );
                })
              )}
            </div>
          </CardContent>
        </Card>

        <Card className="border-slate-700/70 bg-slate-900/80 backdrop-blur">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm text-slate-100">
              <MapIcon className="h-4 w-4 text-slate-300" />
              Visibility Legend
            </CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-2 gap-2 text-xs text-slate-200">
            <div className="inline-flex items-center gap-2 rounded border border-slate-700 px-2 py-1">
              <span className="h-3 w-3 rounded-full border border-cyan-100 bg-cyan-400" />{" "}
              EGO
            </div>
            <div className="inline-flex items-center gap-2 rounded border border-slate-700 px-2 py-1">
              <span className="h-3 w-3 rounded-full border border-emerald-100 bg-emerald-500" />{" "}
              VISIBLE
            </div>
            <div className="inline-flex items-center gap-2 rounded border border-slate-700 px-2 py-1">
              <span className="h-3 w-3 rounded-full border border-rose-100 bg-rose-500" />{" "}
              BLIND
            </div>
            <div className="inline-flex items-center gap-2 rounded border border-slate-700 px-2 py-1">
              <span className="h-3 w-3 rounded-full border border-yellow-50 bg-amber-300" />{" "}
              V2V
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="border-slate-700/70 bg-slate-900/80 backdrop-blur">
          <CardHeader className="pb-1 pt-3">
            <CardTitle className="flex items-center gap-2 text-sm text-slate-100">
              <Gauge className="h-4 w-4 text-slate-300" />
              Selected Car Dashboard
            </CardTitle>
          </CardHeader>
          <CardContent className="p-2">
            {observerDashboard ? (
              <img
                src={`${observerDashboard}?ts=${liveRefreshKey}`}
                alt="Selected vehicle synthetic dashboard"
                className="max-h-64 w-full rounded-md border border-slate-700 object-contain"
                onLoad={() =>
                  console.log(
                    `[DASH] 🟢 Image loaded  frame=${frameIndex}  rev=${liveRefreshKey}  src=${observerDashboard}`
                  )
                }
                onError={() =>
                  console.warn(
                    `[DASH] 🔴 Image error   frame=${frameIndex}  rev=${liveRefreshKey}  src=${observerDashboard}`
                  )
                }
              />
            ) : (
              <div className="flex h-40 items-center justify-center rounded-md border border-slate-700/50 bg-slate-950/40">
                <p className="text-xs text-slate-400">No dashboard available</p>
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="border-slate-700/70 bg-slate-900/80 backdrop-blur">
          <CardHeader className="pb-1 pt-3">
            <CardTitle className="flex items-center gap-2 text-sm text-slate-100">
              <Rows3 className="h-4 w-4 text-slate-300" />
              Realtime BEV
            </CardTitle>
          </CardHeader>
          <CardContent className="p-2">
            <img
              src={`${bevImage}?ts=${liveRefreshKey}`}
              alt="Realtime bird eye view"
              className="max-h-64 w-full rounded-md border border-slate-700 object-contain"
              onLoad={() =>
                console.log(
                  `[BEV] 🟢 Image loaded  frame=${frameIndex}  rev=${liveRefreshKey}  src=${bevImage}`
                )
              }
              onError={() =>
                console.warn(
                  `[BEV] 🔴 Image error   frame=${frameIndex}  rev=${liveRefreshKey}  src=${bevImage}`
                )
              }
            />
          </CardContent>
        </Card>
      </div>
    </main>
  );
}

export default App;
