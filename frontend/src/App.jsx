import { useCallback, useEffect, useState } from 'react'
import { api } from './api'
import Fields from './components/Fields'
import Preview from './components/Preview'
import Queue from './components/Queue'

export default function App() {
  const [queue, setQueue] = useState([])
  const [stats, setStats] = useState(null)
  const [selectedId, setSelectedId] = useState(null)
  const [detail, setDetail] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const [items, s] = await Promise.all([api.queue(), api.stats()])
      setQueue(items)
      setStats(s)
      setError(null)
      // Keep a selection alive across refreshes; fall back to the top of the
      // queue, which is the lowest-confidence item and so the most urgent.
      setSelectedId((current) =>
        current && items.some((i) => i.id === current) ? current : items[0]?.id ?? null
      )
    } catch (e) {
      setError(
        `Cannot reach the review server. Start it with ` +
          `"uvicorn src.api:app --reload" and reload this page. (${e.message})`
      )
    }
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  useEffect(() => {
    if (selectedId === null) {
      setDetail(null)
      return
    }
    let cancelled = false
    api
      .extraction(selectedId)
      .then((d) => !cancelled && setDetail(d))
      .catch((e) => !cancelled && setError(e.message))
    return () => {
      cancelled = true
    }
  }, [selectedId])

  const save = async (field, value) => {
    try {
      setDetail(await api.correct(selectedId, field, value))
      setError(null)
      refresh()
    } catch (e) {
      setError(e.message)
      throw e
    }
  }

  const decide = async (kind) => {
    if (!detail) return
    let note = ''
    if (kind === 'reject') {
      note = window.prompt('Why is this being rejected?') || ''
      if (!note.trim()) return
    }
    setBusy(true)
    try {
      await (kind === 'approve' ? api.approve(detail.id, note) : api.reject(detail.id, note))
      setSelectedId(null)
      await refresh()
      setError(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  // Reviewers work through a queue at speed. Reaching for the mouse on every
  // document is the difference between clearing forty and clearing ten.
  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === 'INPUT' || e.metaKey || e.ctrlKey || e.altKey) return
      const index = queue.findIndex((i) => i.id === selectedId)
      if (e.key === 'j' && index < queue.length - 1) setSelectedId(queue[index + 1].id)
      if (e.key === 'k' && index > 0) setSelectedId(queue[index - 1].id)
      if (e.key === 'a' && detail) decide('approve')
      if (e.key === 'x' && detail) decide('reject')
      if (e.key === 'e') {
        const weak = document.querySelector('.field-row .tick.low, .field-row .tick.mid')
        weak?.parentElement?.querySelector('input')?.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [queue, selectedId, detail])

  return (
    <div className="app">
      <header className="masthead">
        <h1>Invoice review</h1>
        {stats && (
          <div className="figures">
            <span>
              <b>{stats.documents}</b> documents
            </span>
            <span>
              <b>{stats.vendors}</b> vendors
            </span>
            <span>
              <b>{Math.round(stats.straight_through_rate * 100)}%</b> straight through
            </span>
          </div>
        )}
      </header>

      {error && <div className="banner">{error}</div>}

      <div className="workspace">
        <Queue items={queue} selectedId={selectedId} onSelect={setSelectedId} />

        {detail ? (
          <Preview extraction={detail} />
        ) : (
          <section className="preview">
            <p className="empty">
              <strong>Queue clear</strong>
              Nothing is waiting on a human. Extractions that pass the confidence
              threshold and the arithmetic checks post straight through.
            </p>
          </section>
        )}

        <section className="panel">
          {detail ? (
            <>
              <h2>
                Extracted values · {Math.round(detail.overall_confidence * 100)}% ·{' '}
                {detail.model}
              </h2>

              {detail.issues.length > 0 && (
                <ul className="issues">
                  {detail.issues.map((issue, i) => (
                    <li key={i}>{issue}</li>
                  ))}
                </ul>
              )}

              <Fields extraction={detail} onSave={save} disabled={busy} />

              {detail.line_items.length > 0 && (
                <table className="line-items">
                  <thead>
                    <tr>
                      <th>Item</th>
                      <th>Qty</th>
                      <th>Unit</th>
                      <th>Amount</th>
                    </tr>
                  </thead>
                  <tbody>
                    {detail.line_items.map((li, i) => (
                      <tr key={i}>
                        <td>{li.description}</td>
                        <td>{li.quantity ?? '—'}</td>
                        <td>{li.unit_price ?? '—'}</td>
                        <td>{li.line_total ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}

              <div className="history">
                {detail.history.map((h, i) => (
                  <div key={i}>
                    <b>{h.actor}</b> {h.action}
                    {h.field && (
                      <>
                        {' '}
                        {h.field}: {h.old_value ?? '—'} → {h.new_value ?? '—'}
                      </>
                    )}
                    {h.note && <> · {h.note}</>}
                  </div>
                ))}
              </div>

              <div className="actions">
                <button className="btn primary" disabled={busy} onClick={() => decide('approve')}>
                  Approve
                </button>
                <button className="btn danger" disabled={busy} onClick={() => decide('reject')}>
                  Reject
                </button>
              </div>

              <p className="shortcuts">
                <kbd>j</kbd> <kbd>k</kbd> move · <kbd>e</kbd> first query ·{' '}
                <kbd>a</kbd> approve · <kbd>x</kbd> reject
              </p>
            </>
          ) : (
            <p className="empty">
              <strong>No invoice selected</strong>
              Pick one from the queue to check its figures against the document.
            </p>
          )}
        </section>
      </div>
    </div>
  )
}
