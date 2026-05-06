import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Simulation data is now served exclusively through the ws_server (port 8765)
  // so Vite never races with atomic writes to latest.json / BEV images.
  // publicDir left as default ("public") — only static frontend assets live there.
  appType: "mpa",
  server: {
    fs: {
      allow: [".."],
    },
    proxy: {
      // WebSocket upgrade for live stream
      "/ws": {
        target: "ws://localhost:8765",
        ws: true,
        rewriteWsOrigin: true,
      },
      // All simulation data — served by FastAPI/StaticFiles (no Vite race)
      "/live": { target: "http://localhost:8765", changeOrigin: true },
      "/frames": { target: "http://localhost:8765", changeOrigin: true },
      "/fused": { target: "http://localhost:8765", changeOrigin: true },
      "/images": { target: "http://localhost:8765", changeOrigin: true },
      "/sumo_map.net.xml": {
        target: "http://localhost:8765",
        changeOrigin: true,
      },
    },
  },
});
