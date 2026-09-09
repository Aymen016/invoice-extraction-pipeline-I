import { useEffect, useState } from 'react'
import { api } from '../api'

/** Shows the original document so figures can be checked against the page.
 *  Reviewing extracted numbers without the source is guesswork. */
export default function Preview({ extraction }) {
  const [missing, setMissing] = useState(false)
  const url = api.documentUrl(extraction.document_id)
  const isImage = /\.(png|jpe?g|tiff?|webp)$/i.test(extraction.filename)

  useEffect(() => {
    setMissing(false)
    let cancelled = false
    fetch(url, { method: 'HEAD' }).then((r) => {
      if (!cancelled && !r.ok) setMissing(true)
    })
    return () => {
      cancelled = true
    }
  }, [url])

  if (missing) {
    return (
      <section className="preview">
        <h2>{extraction.filename}</h2>
        <p className="empty">
          <strong>Original not stored</strong>
          Documents processed through the command line before the review server
          existed were never copied into storage. Upload this file through the
          API to see it here — the extracted values on the right are unaffected.
        </p>
      </section>
    )
  }

  return (
    <section className="preview">
      <h2>{extraction.filename}</h2>
      {isImage ? (
        <img className="preview-image" src={url} alt={`Scan of ${extraction.filename}`} />
      ) : (
        <iframe className="preview-frame" src={url} title={extraction.filename} />
      )}
    </section>
  )
}
