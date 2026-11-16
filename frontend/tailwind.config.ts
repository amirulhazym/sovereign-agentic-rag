import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: "var(--bg-canvas)",
        surface: "var(--bg-surface)",
        elevated: "var(--bg-elevated)",
        borderSubtle: "var(--border-subtle)",
        textPrimary: "var(--text-primary)",
        textSecondary: "var(--text-secondary)",
        sovereign: "var(--accent-sovereign)",
        cloud: "var(--accent-cloud)",
        warning: "var(--accent-warning)",
        danger: "var(--accent-danger)",
        citationBg: "var(--citation-pill-bg)",
      },
      fontFamily: {
        sans: ["var(--font-sans)", "Inter", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "JetBrains Mono", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
