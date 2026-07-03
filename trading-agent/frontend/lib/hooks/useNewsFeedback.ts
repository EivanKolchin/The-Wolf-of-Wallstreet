import { useCallback, useEffect, useState } from "react";
import { API_BASE, fetchFromAPI } from "../api";

const STORAGE_KEY = "news-feedback-ratings";
export const LLM_FEEDBACK_LEVELS = ["SEVERE", "SIGNIFICANT", "MILD", "INSIGNIFICANT"] as const;
export type FeedbackValue = number | (typeof LLM_FEEDBACK_LEVELS)[number];

let cachedRatings: Record<string, FeedbackValue> | null = null;

function normalizeSource(source?: string) {
  return (source || "").trim().toLowerCase();
}

function normalizeHeadline(headline?: string) {
  return (headline || "").trim();
}

function mergeRatings(next: Record<string, FeedbackValue>) {
  cachedRatings = { ...(cachedRatings || {}), ...next };
  try {
    if (typeof window !== "undefined") {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(cachedRatings));
    }
  } catch {
    // Ignore storage failures; API remains the source of truth.
  }
  return cachedRatings;
}

function loadLocalRatings() {
  if (cachedRatings) return cachedRatings;
  try {
    if (typeof window !== "undefined") {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (raw) {
        cachedRatings = JSON.parse(raw) as Record<string, FeedbackValue>;
        return cachedRatings;
      }
    }
  } catch {
    // Ignore malformed local cache and fall back to API.
  }
  cachedRatings = {};
  return cachedRatings;
}

export function newsFeedbackKey(item: {
  id?: string;
  article_hash?: string;
  headline?: string;
  source?: string;
  source_domain?: string;
}, fallbackIndex?: number) {
  if (item.article_hash) return item.article_hash;
  if (item.id) return String(item.id);
  const src = normalizeSource(item.source || item.source_domain || "");
  const head = normalizeHeadline(item.headline || "");
  if (src || head) return `${src}:${head}`;
  return `row:${fallbackIndex ?? 0}`;
}

export function useNewsFeedback() {
  const [ratings, setRatings] = useState<Record<string, FeedbackValue>>(() => loadLocalRatings());
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let mounted = true;
    fetchFromAPI("/news/feedback/ratings")
      .then((data) => {
        if (!mounted) return;
        if (data?.ratings && typeof data.ratings === "object") {
          setRatings({ ...mergeRatings(data.ratings) });
        }
      })
      .catch(console.error)
      .finally(() => { if (mounted) setLoaded(true); });
    return () => { mounted = false; };
  }, []);

  const submitFeedback = useCallback(async (
    key: string,
    value: FeedbackValue,
    payload: Record<string, unknown>,
  ) => {
    setRatings({ ...mergeRatings({ [key]: value }) });
    try {
      const res = await fetch(`${API_BASE}/news/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          rating: typeof value === "number" ? value : 0,
          selected_value: value,
          ...payload,
        }),
      });
      const data = await res.json();
      if (Array.isArray(data?.keys)) {
        const persistedValue = (data?.selected_value ?? value) as FeedbackValue;
        const next: Record<string, FeedbackValue> = {};
        for (const k of data.keys) next[String(k)] = persistedValue;
        setRatings({ ...mergeRatings(next) });
      }
    } catch (e) {
      console.error("Failed to submit news feedback", e);
    }
  }, []);

  const getRating = useCallback((key: string) => ratings[key], [ratings]);

  return { ratings, loaded, submitFeedback, getRating };
}

export function severityClass(severity: string) {
  switch ((severity || "").toUpperCase()) {
    case "SEVERE": return "bg-rose-500/10 text-rose-400";
    case "SIGNIFICANT": return "bg-amber-500/10 text-amber-400";
    case "MILD": return "bg-sky-500/10 text-sky-400";
    default: return "bg-zinc-800 text-zinc-400";
  }
}
