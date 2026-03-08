const steps = [
  {
    num: '01',
    title: 'Client embeds your prompt',
    description:
      'Your machine tokenises the input and runs the first few transformer layers locally, converting raw text into a hidden-state tensor.',
    code: 'hidden = run_layers(embed(prompt), layers[:k])',
    color: 'from-blue-500 to-blue-600',
  },
  {
    num: '02',
    title: 'Tensor travels to Peer 0',
    description:
      'The hidden state tensor is serialised and sent over a direct libp2p stream to the first worker peer — never the original prompt.',
    code: 'stream.write(msgpack.pack(hidden))',
    color: 'from-blue-500 to-purple-500',
  },
  {
    num: '03',
    title: 'Peer 0 processes middle layers',
    description:
      'Worker 0 runs its assigned layer shard and forwards the updated tensor to the next peer, chaining the computation.',
    code: 'hidden = run_layers(hidden, layers[k:k+m])',
    color: 'from-purple-500 to-purple-600',
  },
  {
    num: '04',
    title: 'Peer 1 continues the pipeline',
    description:
      'Worker 1 handles its own shard and returns the final intermediate tensor back to the client across the P2P network.',
    code: 'hidden = run_layers(hidden, layers[k+m:n-k])',
    color: 'from-purple-500 to-pink-500',
  },
  {
    num: '05',
    title: 'Client decodes the output',
    description:
      'The client runs the last few layers locally, applying the LM head to project hidden states into token probabilities and sample the response.',
    code: 'tokens = sample(lm_head(run_layers(hidden, layers[-k:])))',
    color: 'from-pink-500 to-pink-600',
  },
]

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="py-24 px-6 bg-surface-2/40">
      <div className="max-w-4xl mx-auto">
        {/* Section header */}
        <div className="text-center mb-16">
          <span className="text-xs font-mono text-purple-400 uppercase tracking-widest">The Protocol</span>
          <h2 className="mt-3 text-4xl font-bold tracking-tight">
            How split inference <span className="gradient-text">works</span>
          </h2>
          <p className="mt-4 text-gray-400 max-w-xl mx-auto">
            A token never crosses a machine boundary. Only opaque floating-point tensors do.
          </p>
        </div>

        {/* Steps */}
        <div className="relative">
          {/* Vertical connector line */}
          <div className="absolute left-[2.25rem] top-0 bottom-0 w-px bg-gradient-to-b from-blue-500 via-purple-500 to-pink-500 opacity-20 hidden md:block" />

          <div className="flex flex-col gap-8">
            {steps.map(({ num, title, description, code, color }, i) => (
              <div key={num} className="relative flex gap-6 group">
                {/* Step circle */}
                <div
                  className={`relative z-10 flex-shrink-0 w-[4.5rem] h-[4.5rem] rounded-2xl bg-gradient-to-br ${color} flex items-center justify-center shadow-lg group-hover:shadow-purple-500/20 transition-shadow`}
                >
                  <span className="font-mono font-bold text-white text-sm">{num}</span>
                </div>

                {/* Content */}
                <div className="pt-1 pb-2 flex-1">
                  <h3 className="text-base font-semibold text-white mb-1">{title}</h3>
                  <p className="text-sm text-gray-400 leading-relaxed mb-3">{description}</p>
                  <code className="block bg-surface border border-white/5 rounded-lg px-4 py-2.5 text-xs font-mono text-purple-300 overflow-x-auto">
                    {code}
                  </code>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
