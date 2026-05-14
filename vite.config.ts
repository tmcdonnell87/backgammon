import { defineConfig, loadEnv } from "vite";
import { resolve } from "node:path";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const allowedHosts = (env.VITE_ALLOWED_HOSTS ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

  return {
    resolve: {
      alias: {
        "@": resolve(__dirname, "src"),
      },
    },
    build: {
      target: "es2022",
      sourcemap: true,
    },
    worker: {
      format: "es",
    },
    server: {
      host: true,
      allowedHosts,
    },
  };
});
