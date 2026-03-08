const features = [
  {
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 10.5V6.75a4.5 4.5 0 10-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H6.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z" />
      </svg>
    ),
    title: 'End-to-End Privacy',
    description:
      'Prompts are converted to hidden states before leaving your machine. Intermediate peers only see encrypted tensor blobs — never raw text.',
    gradient: 'from-blue-500 to-blue-700',
  },
  {
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5" />
      </svg>
    ),
    title: 'P2P Distributed',
    description:
      'Built on libp2p and hivemind DHT. No central server. Peers discover each other automatically on the local network without any cloud dependency.',
    gradient: 'from-purple-500 to-purple-700',
  },
  {
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6A2.25 2.25 0 016 3.75h2.25A2.25 2.25 0 0110.5 6v2.25a2.25 2.25 0 01-2.25 2.25H6a2.25 2.25 0 01-2.25-2.25V6zM3.75 15.75A2.25 2.25 0 016 13.5h2.25a2.25 2.25 0 012.25 2.25V18a2.25 2.25 0 01-2.25 2.25H6A2.25 2.25 0 013.75 18v-2.25zM13.5 6a2.25 2.25 0 012.25-2.25H18A2.25 2.25 0 0120.25 6v2.25A2.25 2.25 0 0118 10.5h-2.25a2.25 2.25 0 01-2.25-2.25V6zM13.5 15.75a2.25 2.25 0 012.25-2.25H18a2.25 2.25 0 012.25 2.25V18A2.25 2.25 0 0118 20.25h-2.25A2.25 2.25 0 0113.5 18v-2.25z" />
      </svg>
    ),
    title: 'Split Inference',
    description:
      'Transformer layers are partitioned across peers. Each machine only holds a shard — no single node can reconstruct the full model or conversation.',
    gradient: 'from-pink-500 to-pink-700',
  },
  {
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" />
      </svg>
    ),
    title: 'Low Latency',
    description:
      'LAN-speed tensor transfers with asynchronous pipeline overlap. Inference latency stays within 2× of a single-machine baseline at 2 peers.',
    gradient: 'from-blue-500 to-purple-600',
  },
  {
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M6.75 7.5l3 2.25-3 2.25m4.5 0h3m-9 8.25h13.5A2.25 2.25 0 0021 18V6a2.25 2.25 0 00-2.25-2.25H5.25A2.25 2.25 0 003 6v12a2.25 2.25 0 002.25 2.25z" />
      </svg>
    ),
    title: 'Open Protocol',
    description:
      'The wire format is just msgpack over raw TCP streams. Any language can implement the protocol. Add a new peer in under 100 lines.',
    gradient: 'from-purple-500 to-pink-600',
  },
  {
    icon: (
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.955 11.955 0 003 12c0 6.627 5.373 12 12 12s12-5.373 12-12c0-2.041-.507-3.964-1.399-5.649l-1.63 1.63a9.97 9.97 0 01.779 4.019 10 10 0 11-10 10 9.97 9.97 0 014.019.779l1.63-1.63A11.959 11.959 0 0112 1.5z" />
      </svg>
    ),
    title: 'Self-Hosted',
    description:
      'No API keys. No rate limits. Runs entirely on hardware you own. Works offline. Quantisation support keeps memory below 4 GB per node.',
    gradient: 'from-pink-500 to-blue-600',
  },
]

export default function Features() {
  return (
    <section id="features" className="py-24 px-6">
      <div className="max-w-6xl mx-auto">
        {/* Section header */}
        <div className="text-center mb-16">
          <span className="text-xs font-mono text-purple-400 uppercase tracking-widest">Why PrivatePrompting</span>
          <h2 className="mt-3 text-4xl font-bold tracking-tight">
            Designed for <span className="gradient-text">trust-less inference</span>
          </h2>
          <p className="mt-4 text-gray-400 max-w-xl mx-auto">
            Every design decision prioritises data sovereignty — no logs, no telemetry, no third-party servers.
          </p>
        </div>

        {/* Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map(({ icon, title, description, gradient }) => (
            <div
              key={title}
              className="group relative rounded-2xl bg-surface-2 border border-white/5 p-6 hover:border-white/10 transition-all duration-300 hover:-translate-y-0.5 hover:shadow-xl hover:shadow-black/30"
            >
              {/* Hover glow */}
              <div className={`absolute inset-0 rounded-2xl bg-gradient-to-br ${gradient} opacity-0 group-hover:opacity-5 transition-opacity duration-300 pointer-events-none`} />

              <div className={`mb-4 w-10 h-10 rounded-xl bg-gradient-to-br ${gradient} flex items-center justify-center text-white shadow-lg`}>
                {icon}
              </div>
              <h3 className="text-base font-semibold text-white mb-2">{title}</h3>
              <p className="text-sm text-gray-400 leading-relaxed">{description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
