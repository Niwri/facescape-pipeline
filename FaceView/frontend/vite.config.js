import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
export default defineConfig(({ mode }) => {
  // Machine-specific settings live in frontend/.env (see .env.example), not here.
  const env = loadEnv(mode, __dirname, "");
  const backend = env.BACKEND_URL;
  if (!backend) {
    throw new Error("BACKEND_URL is not set -- add it to frontend/.env (see .env.example)");
  }
  const backendWs = backend.replace(/^http/, "ws");
  return {
    plugins: [react()],
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src")
      }
    },
    server: {
      port: 5173,
      open: true,
      allowedHosts: [".ngrok-free.dev", ".ngrok-free.app", ".ngrok.io", ".ngrok.app", ".trycloudflare.com"],
      proxy: {
        "/api": backend,
        "/ws": {
          target: backendWs,
          ws: true
        },
        "/processed": {
          target: backend,
          changeOrigin: true
        }
      }
    }
  };
});
