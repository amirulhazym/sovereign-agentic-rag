"use client";

import { CloudLightning, ShieldCheck } from "lucide-react";
import type { InferenceMode } from "@/types/api";

interface InferenceModeSwitchProps {
  mode: InferenceMode;
  onChange: (mode: InferenceMode) => void;
  disabled?: boolean;
}

export function InferenceModeSwitch({
  mode,
  onChange,
  disabled = false,
}: InferenceModeSwitchProps) {
  const isLocal = mode === "LOCAL_ONPREM";

  return (
    <div className="flex flex-col items-end gap-0.5">
      <div
        role="group"
        aria-label="Inference sovereignty mode"
        className="inline-flex items-center rounded-lg border border-borderSubtle bg-canvas p-0.5"
      >
        <button
          type="button"
          disabled={disabled}
          onClick={() => onChange("LOCAL_ONPREM")}
          className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sovereign ${
            isLocal
              ? "bg-sovereign/20 text-sovereign shadow-sm ring-1 ring-sovereign/40"
              : "text-textSecondary hover:text-textPrimary"
          }`}
        >
          <ShieldCheck className="h-3.5 w-3.5" aria-hidden="true" />
          <span>LOCAL_ONPREM</span>
        </button>
        <button
          type="button"
          disabled={disabled}
          onClick={() => onChange("CLOUD_API")}
          className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cloud ${
            !isLocal
              ? "bg-cloud/20 text-indigo-300 shadow-sm ring-1 ring-cloud/40"
              : "text-textSecondary hover:text-textPrimary"
          }`}
        >
          <CloudLightning className="h-3.5 w-3.5" aria-hidden="true" />
          <span>CLOUD_API</span>
        </button>
      </div>
      <span className="hidden font-mono text-[11px] text-textSecondary md:inline">
        {isLocal
          ? "Ollama Qwen3.8-4B-Distill + FastEmbed bge-m3"
          : "DeepSeek Chat · Prefix Caching On"}
      </span>
    </div>
  );
}
