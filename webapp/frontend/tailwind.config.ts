import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        accent: {
          DEFAULT: "#2dd4bf",
          bright: "#5eead4",
          dim: "#0f766e",
        },
        // Fixed status palette (good/warning) -- not themed. Used only for status
        // meaning (predicted class, risk flags), never as decoration -- see the
        // dataviz skill's status-color rule (icon + label, never color alone).
        status: {
          good: "#0ca30c",
          warning: "#fab219",
        },
      },
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      backgroundImage: {
        // slate-300 at low opacity, tuned for a light surface.
        "grid-pattern":
          "linear-gradient(to right, rgb(203 213 225 / 0.6) 1px, transparent 1px), linear-gradient(to bottom, rgb(203 213 225 / 0.6) 1px, transparent 1px)",
      },
      backgroundSize: {
        grid: "32px 32px",
      },
      keyframes: {
        // A slow light-sweep across the gradient-filled title text -- pearl
        // luster, not a loading-skeleton shimmer (much slower, no easing snap).
        shimmer: {
          "0%": { backgroundPosition: "200% center" },
          "100%": { backgroundPosition: "-200% center" },
        },
      },
      animation: {
        shimmer: "shimmer 5s linear infinite",
      },
      boxShadow: {
        // Two-layer soft elevation: a tight shadow near the edge plus a wider,
        // softer one for ambient depth -- reads as "lifted," not "boxed."
        card: "0 1px 2px -1px rgb(15 23 42 / 0.06), 0 8px 24px -8px rgb(15 23 42 / 0.10)",
        "card-hover": "0 2px 4px -1px rgb(15 23 42 / 0.08), 0 12px 32px -8px rgb(15 23 42 / 0.14)",
      },
    },
  },
  plugins: [],
};

export default config;
