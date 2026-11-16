import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Sovereign Agentic RAG Console",
  description:
    "Bilingual EN/BM/BM-EN Agentic RAG workbench supporting LOCAL_ONPREM Qwen3.8 + Qdrant Hybrid RRF and DeepSeek Prefix Context Caching.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable} dark`}>
      <body className="min-h-screen bg-canvas font-sans text-textPrimary antialiased selection:bg-sovereign/30 selection:text-textPrimary">
        {children}
      </body>
    </html>
  );
}
