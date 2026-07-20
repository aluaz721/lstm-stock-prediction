import { useCallback, useEffect, useState } from "react";
import { ApiRequestError } from "./api";

interface UseApiResult<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

/**
 * Wraps an async fetch function with loading/error state and a refetch
 * trigger. deps controls when it re-fetches automatically (e.g. when the
 * selected ticker changes); call refetch() for manual re-fetches (e.g. a
 * "Refresh" button).
 */
export function useApi<T>(fetchFn: () => Promise<T>, deps: unknown[]): UseApiResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refetchCount, setRefetchCount] = useState(0);

  const load = useCallback(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchFn()
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .catch((err) => {
        if (cancelled) return;
        const message = err instanceof ApiRequestError ? err.message : "Request failed";
        setError(message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [...deps, refetchCount]);

  useEffect(() => load(), [load]);

  const refetch = useCallback(() => setRefetchCount((c) => c + 1), []);

  return { data, loading, error, refetch };
}
