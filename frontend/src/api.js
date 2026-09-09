// Thin fetch wrapper. Every call goes through here so error handling and the
// JSON contract live in one place instead of being re-implemented per screen.

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })

  if (!response.ok) {
    // FastAPI puts the useful message in `detail`. Surfacing it verbatim means
    // a validation failure reads as "Dates must be YYYY-MM-DD" rather than 422.
    let message = `Request failed (${response.status})`
    try {
      const body = await response.json()
      if (body.detail) {
        message = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
      }
    } catch {
      /* response had no JSON body; the status message will do */
    }
    throw new Error(message)
  }
  return response.json()
}

export const api = {
  stats: () => request('/api/stats'),
  queue: () => request('/api/queue'),
  extraction: (id) => request(`/api/extractions/${id}`),

  correct: (id, field, value, note = '') =>
    request(`/api/extractions/${id}`, {
      method: 'PATCH',
      body: JSON.stringify({ field, value, actor: 'reviewer', note }),
    }),

  approve: (id, note = '') =>
    request(`/api/extractions/${id}/approve`, {
      method: 'POST',
      body: JSON.stringify({ actor: 'reviewer', note }),
    }),

  reject: (id, note) =>
    request(`/api/extractions/${id}/reject`, {
      method: 'POST',
      body: JSON.stringify({ actor: 'reviewer', note }),
    }),

  documentUrl: (documentId) => `/api/documents/${documentId}/file`,
}
