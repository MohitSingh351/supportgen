import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the React app proxies /api to the Django server, so no CORS setup is needed.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
