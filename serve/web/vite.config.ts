import path from "path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// `npm run dev` proxies the APIs to a running bnk server (BNK_URL, default http://localhost:8080)
const target = process.env.BNK_URL ?? "http://localhost:8080"

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "./src") } },
  server: { proxy: { "/api": target, "/v1": target, "/health": target } },
  build: { outDir: "dist", chunkSizeWarningLimit: 1500 },
})
