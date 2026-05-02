import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  publicDir: "../data",
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
