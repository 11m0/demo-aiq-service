import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

function isLoopbackHostname(hostname: string): boolean {
  if (hostname === "localhost" || hostname === "[::1]") {
    return true;
  }

  const octets = hostname.split(".");
  return (
    octets.length === 4 &&
    octets[0] === "127" &&
    octets.every((octet) => /^\d{1,3}$/.test(octet) && Number(octet) <= 255)
  );
}

function validateBackendTarget(value: string): string {
  if (value !== value.trim() || value.includes("\\")) {
    throw new Error(
      "VITE_BACKEND_TARGET должен быть абсолютным HTTP(S) URL без пробелов и обратных слешей.",
    );
  }

  let target: URL;
  try {
    target = new URL(value);
  } catch {
    throw new Error("VITE_BACKEND_TARGET должен быть абсолютным HTTP(S) URL.");
  }

  if (target.protocol !== "http:" && target.protocol !== "https:") {
    throw new Error("VITE_BACKEND_TARGET поддерживает только протоколы http и https.");
  }
  if (target.username !== "" || target.password !== "") {
    throw new Error("VITE_BACKEND_TARGET не должен содержать учётные данные.");
  }
  if (
    target.search !== "" ||
    target.hash !== "" ||
    value.includes("?") ||
    value.includes("#")
  ) {
    throw new Error("VITE_BACKEND_TARGET не должен содержать query-параметры или fragment.");
  }
  if (target.protocol === "http:" && !isLoopbackHostname(target.hostname)) {
    throw new Error(
      "Незашифрованный VITE_BACKEND_TARGET разрешён только для localhost или loopback-адреса.",
    );
  }

  return value;
}

export default defineConfig(({ mode }) => {
  const environment = loadEnv(mode, ".", "VITE_");
  const backendTarget = validateBackendTarget(
    environment.VITE_BACKEND_TARGET || "http://127.0.0.1:8001",
  );

  return {
    plugins: [react()],
    server: {
      host: "127.0.0.1",
      port: 5173,
      strictPort: true,
      proxy: {
        "/api": {
          target: backendTarget,
          changeOrigin: true,
          // Backend допускает до 390 секунд на обработку одного ML-хода.
          proxyTimeout: 400_000,
          timeout: 410_000,
        },
      },
    },
  };
});
