// fetch wrapper logic
export async function fetchFromAPI(endpoint: string, options?: RequestInit) {
  const res = await fetch(`/api${endpoint}`, options);
  return res.json();
}
