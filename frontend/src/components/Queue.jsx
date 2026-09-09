const money = (amount, currency) => {
  if (amount === null || amount === undefined) return '—'
  const n = Number(amount).toLocaleString(undefined, { maximumFractionDigits: 2 })
  return currency ? `${currency} ${n}` : n
}

export default function Queue({ items, selectedId, onSelect }) {
  if (items.length === 0) {
    return (
      <nav className="queue" aria-label="Review queue">
        <h2>Awaiting review</h2>
        <p className="empty">
          <strong>Nothing to review</strong>
          Every extraction cleared the confidence threshold and its arithmetic
          checked out. Process more documents with{' '}
          <code>python cli.py samples/ --db</code>.
        </p>
      </nav>
    )
  }

  return (
    <nav className="queue" aria-label="Review queue">
      <h2>Awaiting review · {items.length}</h2>
      {items.map((item) => (
        <button
          key={item.id}
          className="queue-item"
          aria-current={item.id === selectedId}
          onClick={() => onSelect(item.id)}
        >
          <span className="vendor">{item.vendor_name || item.filename}</span>
          <span className="meta">
            <span>{money(item.total_amount, item.currency)}</span>
            <span>
              {item.issue_count > 0 && (
                <span className="flags">
                  {item.issue_count} {item.issue_count === 1 ? 'query' : 'queries'}
                </span>
              )}
            </span>
          </span>
          <span className="meta">
            <span>{item.invoice_number || 'no reference'}</span>
            <span>{Math.round(item.overall_confidence * 100)}%</span>
          </span>
        </button>
      ))}
    </nav>
  )
}
