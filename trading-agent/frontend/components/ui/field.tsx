"use client";

import * as React from "react";
import { Eye, EyeOff, Info } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Shared form primitives for the Settings / Setup surfaces.
 *
 * Everything sits on one palette so the app reads as a single design system
 * rather than a per-page patchwork:
 *   surface (card)   #0A0A0A / border #171717   — matches the dashboard cards
 *   inset  (input)   #0D0D0F / border #232327
 *   accent           amber (--accent), applied via the global focus ring
 */

const inputBase =
  "w-full rounded-lg bg-[#0D0D0F] border border-[#242428] px-4 py-2.5 text-[14px] " +
  "text-zinc-100 placeholder-zinc-600 transition-colors " +
  "focus:outline-none focus:border-zinc-500 disabled:opacity-60 disabled:cursor-not-allowed";

/** Label + optional hint wrapper around any control. */
export function Field({
  label,
  hint,
  htmlFor,
  className,
  children,
}: {
  label?: React.ReactNode;
  hint?: React.ReactNode;
  htmlFor?: string;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <div className={className}>
      {label && (
        <label htmlFor={htmlFor} className="mb-2 block text-[13px] font-medium text-zinc-400">
          {label}
        </label>
      )}
      {children}
      {hint && <p className="mt-2 text-[11px] leading-relaxed text-zinc-600">{hint}</p>}
    </div>
  );
}

export const TextInput = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement> & { mono?: boolean }>(
  ({ className, mono, ...props }, ref) => (
    <input ref={ref} className={cn(inputBase, mono && "font-mono placeholder:font-sans", className)} {...props} />
  )
);
TextInput.displayName = "TextInput";

/**
 * Password-style secret input: shows dots by default, with an eye toggle to
 * peek. Prefilled from the .env by the caller, so a configured key renders as
 * a filled masked field (never an empty box).
 */
export const SecretInput = React.forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(({ className, ...props }, ref) => {
  const [show, setShow] = React.useState(false);
  return (
    <div className="relative w-full">
      <input
        ref={ref}
        type={show ? "text" : "password"}
        className={cn(inputBase, "pr-11 font-mono placeholder:font-sans", className)}
        {...props}
      />
      <button
        type="button"
        onClick={() => setShow((s) => !s)}
        tabIndex={-1}
        aria-label={show ? "Hide value" : "Reveal value"}
        className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500 transition-colors hover:text-zinc-200"
      >
        {show ? <EyeOff size={16} /> : <Eye size={16} />}
      </button>
    </div>
  );
});
SecretInput.displayName = "SecretInput";

export const SelectInput = React.forwardRef<HTMLSelectElement, React.SelectHTMLAttributes<HTMLSelectElement>>(
  ({ className, children, ...props }, ref) => (
    <div className="relative">
      <select ref={ref} className={cn(inputBase, "appearance-none pr-10", className)} {...props}>
        {children}
      </select>
      <svg
        className="pointer-events-none absolute right-3.5 top-1/2 -translate-y-1/2 text-zinc-500"
        width="12" height="12" viewBox="0 0 12 12" fill="none"
      >
        <path d="M2.5 4.5L6 8l3.5-3.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  )
);
SelectInput.displayName = "SelectInput";

/**
 * Monochrome switch.
 *   tone="neutral" (default) — white track + dark knob when on (charcoal system).
 *   tone="safety"            — emerald when on / red when off (for arm/disarm
 *                              switches where colour carries real meaning).
 */
export function Toggle({
  checked,
  onChange,
  tone = "neutral",
}: {
  checked: boolean;
  onChange: () => void;
  tone?: "neutral" | "safety";
}) {
  const track =
    tone === "safety"
      ? checked ? "bg-emerald-500" : "bg-red-600"
      : checked ? "bg-zinc-100" : "bg-zinc-700";
  const knob = tone === "neutral" && checked ? "bg-zinc-900" : "bg-white";
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={onChange}
      className={cn("relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors", track)}
    >
      <span
        className={cn(
          "inline-block h-4 w-4 transform rounded-full shadow-sm transition-transform",
          knob,
          checked ? "translate-x-6" : "translate-x-1"
        )}
      />
    </button>
  );
}

/** A framed row of "title + description" with a control (usually a Toggle) on the right. */
export function ControlRow({
  title,
  description,
  children,
  className,
}: {
  title: React.ReactNode;
  description?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex items-center justify-between gap-4 rounded-xl border border-[#1c1c1f] bg-[#0D0D0F] p-4",
        className
      )}
    >
      <div className="min-w-0">
        <h4 className="text-[14px] font-medium text-zinc-200">{title}</h4>
        {description && <p className="mt-1 text-[12px] leading-relaxed text-zinc-500">{description}</p>}
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  );
}

/** Hover info tooltip used in section headers. */
export function InfoTip({ children }: { children: React.ReactNode }) {
  return (
    <div className="group relative">
      <button
        type="button"
        aria-label="More info"
        className="cursor-help rounded-full p-1 text-zinc-500 transition-colors hover:bg-white/5 hover:text-zinc-300"
      >
        <Info size={16} />
      </button>
      <div className="invisible absolute right-[-6px] top-8 z-50 w-64 rounded-xl border border-[#242428] bg-[#111113] p-4 text-[12px] leading-relaxed text-zinc-300 opacity-0 shadow-xl transition-all group-hover:visible group-hover:opacity-100">
        {children}
      </div>
    </div>
  );
}

/** A settings section rendered as a card that matches the dashboard surface. */
export function Section({
  title,
  dotColor = "bg-zinc-600",
  info,
  children,
  className,
}: {
  title: React.ReactNode;
  dotColor?: string;
  info?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("rounded-2xl border border-[#171717] bg-[#0A0A0A] p-6 md:p-7 shadow-sm", className)}>
      <div className="mb-6 flex items-center justify-between">
        <h3 className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.15em] text-zinc-500">
          <span className={cn("h-1.5 w-1.5 rounded-full", dotColor)} />
          {title}
        </h3>
        {info && <InfoTip>{info}</InfoTip>}
      </div>
      {children}
    </section>
  );
}

/** Thin gradient divider used between sub-groups inside a section. */
export function Divider({ className }: { className?: string }) {
  return <div className={cn("h-px w-full bg-gradient-to-r from-transparent via-[#242428] to-transparent", className)} />;
}
