import { useEffect, useRef, useState } from 'react'

const FIELDS = [
  ['invoice_number', 'Invoice number'],
  ['invoice_date', 'Invoice date'],
  ['due_date', 'Due date'],
  ['vendor_name', 'Vendor'],
  ['vendor_tax_id', 'Vendor tax ID'],
  ['buyer_name', 'Buyer'],
  ['currency', 'Currency'],
  ['subtotal', 'Subtotal'],
  ['tax_amount', 'Tax'],
  ['total_amount', 'Total'],
]

// The margin mark. A tick means the value was found verbatim in the document;
// a query mark means it was inferred; a flag means it could not be located and
// deserves a look. Meaning lives in the position and colour, so the glyphs stay
// quiet rather than competing with the figures.
function Tick({ score }) {
  if (score === null) return <span className="tick" aria-hidden="true" />
  if (score >= 0.85) {
    return <span className="tick high" title={`Verified against the document (${score.toFixed(2)})`}>✓</span>
  }
  if (score >= 0.6) {
    return <span className="tick mid" title={`Partly matched (${score.toFixed(2)})`}>?</span>
  }
  return <span className="tick low" title={`Not found in the document (${score.toFixed(2)})`}>!</span>
}

export default function Fields({ extraction, onSave, disabled }) {
  const [drafts, setDrafts] = useState({})
  const [saving, setSaving] = useState(null)
  const firstWeak = useRef(null)

  // Discard in-flight edits when a different invoice is opened, otherwise a
  // half-typed value would silently follow the reviewer to the next document.
  useEffect(() => setDrafts({}), [extraction.id])

  const scoreFor = (field) => {
    const row = extraction.confidences.find((c) => c.field === field)
    return row ? row.final_confidence : null
  }

  const commit = async (field) => {
    const draft = drafts[field]
    if (draft === undefined) return
    const original = extraction[field] === null ? '' : String(extraction[field])
    if (draft === original) {
      setDrafts((d) => ({ ...d, [field]: undefined }))
      return
    }
    setSaving(field)
    try {
      await onSave(field, draft)
      setDrafts((d) => ({ ...d, [field]: undefined }))
    } finally {
      setSaving(null)
    }
  }

  let weakAssigned = false

  return (
    <div>
      {FIELDS.map(([field, label]) => {
        const score = scoreFor(field)
        const stored = extraction[field] === null ? '' : String(extraction[field])
        const draft = drafts[field]
        const value = draft === undefined ? stored : draft
        const edited = draft !== undefined && draft !== stored

        // Focus target for the "jump to the first problem" shortcut.
        let ref = null
        if (!weakAssigned && score !== null && score < 0.85) {
          ref = firstWeak
          weakAssigned = true
        }

        return (
          <div key={field} className={`field-row${edited ? ' edited' : ''}`}>
            <Tick score={score} />
            <div>
              <label htmlFor={`f-${field}`}>{label}</label>
              <input
                id={`f-${field}`}
                ref={ref}
                value={value}
                disabled={disabled || saving === field}
                placeholder={score === null ? 'not found' : ''}
                onChange={(e) => setDrafts((d) => ({ ...d, [field]: e.target.value }))}
                onBlur={() => commit(field)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') e.currentTarget.blur()
                  if (e.key === 'Escape') {
                    setDrafts((d) => ({ ...d, [field]: undefined }))
                    e.currentTarget.blur()
                  }
                }}
              />
              {score !== null && score < 0.85 && (
                <div className="score">
                  {score < 0.6
                    ? 'Not found in the document text — check this one.'
                    : 'Partly matched against the document.'}
                </div>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}

export { FIELDS }
