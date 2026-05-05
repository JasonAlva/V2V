import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  publicDir: "../data",
  // MPA mode: disables SPA HTML fallback so missing /frames/*.json, /fused/*.png, etc.
  // return a real 404 instead of index.html, preventing "Unexpected token '<'" JSON parse errors.
  appType: "mpa",
  server: {
    fs: {
      allow: [".."],
    },
    proxy: {
      // Forward WebSocket upgrades to the Python WS server
      "/ws": {
        target: "ws://localhost:8765",
        ws: true,
        rewriteWsOrigin: true,
      },
    },
  },
});
