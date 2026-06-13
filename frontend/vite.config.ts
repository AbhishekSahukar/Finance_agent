import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/oem-agent": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/health": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    // Allow dynamic imports from CDN (jsPDF) — these are not bundled,
    // they are fetched at runtime when the user clicks Download PDF.
    rollupOptions: {
      external: [],
    },
  },
  optimizeDeps: {
    // Do not pre-bundle CDN URLs — they are loaded on demand
    exclude: [],
  },
});