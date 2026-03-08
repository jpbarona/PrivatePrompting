import { useState, useRef } from 'react'

const API_ENDPOINT = 'http://localhost:8080/generate'

export default function Hero() {
  const [prompt, setPrompt] = useState('')
  const [response, setResponse] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const textareaRef = useRef(null)

  const handleSend = async () => {
    if (!prompt.trim() || loading) return
    setLoading(true)
    setError('')
    setResponse('')
    try {
      const res = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt.trim(), max_new_tokens: 256 }),
      })
      if (!res.ok) throw new Error(`Server error: ${res.status}`)
      const data = await res.json()
      setResponse(data.response ?? data.output ?? JSON.stringify(data))
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleSend()
  }

  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center px-6 pt-24 pb-16 overflow-hidden">
      {/* Background glow blobs */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -top-32 left-1/2 -translate-x-1/2 w-[700px] h-[700px] rounded-full bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 opacity-10 blur-3xl" />
        <div className="absolute bottom-0 left-0 w-96 h-96 rounded-full bg-purple-600 opacity-5 blur-3xl" />
        <div className="absolute bottom-0 right-0 w-96 h-96 rounded-full bg-blue-600 opacity-5 blur-3xl" />
      </div>

      <div className="relative z-10 max-w-3xl w-full text-center flex flex-col items-center gap-6">
        {/* Badge */}
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-surface-2 border border-white/10 text-xs text-gray-400 font-mono">
          <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
          P2P · Split-Inference · Privacy-First
        </div>

        {/* Headline */}
        <h1 className="text-5xl md:text-6xl font-bold tracking-tight leading-tight">
          LLM inference{' '}
          <span className="gradient-text">without compromising</span>{' '}
          your prompts
        </h1>

        <p className="text-lg text-gray-400 max-w-xl leading-relaxed">
          PrivatePrompting splits transformer layers across peers in a local network.
          Your prompt never leaves as a whole — it travels as hidden states,
          unintelligible without the full model.
        </p>

        {/* Input card */}
        <div
          id="hero-input"
          className="w-full max-w-2xl mt-4 rounded-2xl bg-surface-2 border border-white/10 shadow-2xl overflow-hidden group"
        >
          {/* Glow on hover */}
          <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 opacity-0 group-hover:opacity-[0.06] transition-opacity duration-500 pointer-events-none" />

          <div className="p-4">
            <textarea
              ref={textareaRef}
              rows={4}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask anything… your prompt stays private."
              className="w-full bg-transparent text-gray-100 placeholder-gray-600 font-mono text-sm resize-none outline-none leading-relaxed"
            />
          </div>

          <div className="flex items-center justify-between px-4 py-3 border-t border-white/5">
            <span className="text-xs text-gray-600 font-mono">⌘ + Enter to send</span>
            <button
              onClick={handleSend}
              disabled={loading || !prompt.trim()}
              className={`flex items-center gap-2 px-5 py-2 rounded-xl text-sm font-medium transition-all duration-200 ${
                loading || !prompt.trim()
                  ? 'bg-surface-3 text-gray-600 cursor-not-allowed'
                  : 'bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 text-white hover:opacity-90 shadow-lg hover:shadow-purple-500/25'
              }`}
            >
              {loading ? (
                <>
                  <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                  </svg>
                  Generating…
                </>
              ) : (
                <>
                  Send
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Response area */}
        {(response || error) && (
          <div className={`w-full max-w-2xl rounded-2xl border p-5 text-left font-mono text-sm leading-relaxed ${
            error
              ? 'bg-red-950/40 border-red-800/40 text-red-300'
              : 'bg-surface-2 border-white/10 text-gray-300'
          }`}>
            {error ? `Error: ${error}` : response}
          </div>
        )}
      </div>
    </section>
  )
}
