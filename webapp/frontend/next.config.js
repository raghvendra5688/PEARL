/** @type {import('next').NextConfig} */
// BUILD_STATIC=true (set in the Dockerfile's build stage) produces a static export
// that FastAPI serves directly, same-origin -- see ARCHITECTURE.md #2. In dev, API
// calls go straight to the backend via NEXT_PUBLIC_API_BASE (lib/api.ts), not
// through a rewrite proxy, which can't reliably carry the /api/predict SSE stream.
// See ARCHITECTURE.md #14.
const isStaticBuild = process.env.BUILD_STATIC === "true";

const nextConfig = {
  ...(isStaticBuild ? { output: "export" } : {}),
  images: { unoptimized: true },
};

module.exports = nextConfig;
