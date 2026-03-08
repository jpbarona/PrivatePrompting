const links = [
  {
    heading: 'Project',
    items: [
      { label: 'GitHub', href: 'https://github.com' },
      { label: 'Documentation', href: '#' },
      { label: 'Changelog', href: '#' },
    ],
  },
  {
    heading: 'Inference',
    items: [
      { label: 'How It Works', href: '#how-it-works' },
      { label: 'Features', href: '#features' },
      { label: 'Protocol Spec', href: '#' },
    ],
  },
  {
    heading: 'Community',
    items: [
      { label: 'Discord', href: '#' },
      { label: 'Twitter / X', href: '#' },
      { label: 'Contributing', href: '#' },
    ],
  },
]

export default function Footer() {
  return (
    <footer className="border-t border-white/5 bg-surface py-16 px-6">
      <div className="max-w-6xl mx-auto">
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-10 mb-12">
          {/* Brand column */}
          <div>
            <div className="flex items-center gap-2 mb-4">
              <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 flex items-center justify-center">
                <svg className="w-3.5 h-3.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18" />
                </svg>
              </div>
              <span className="font-semibold text-sm gradient-text">PrivatePrompting</span>
            </div>
            <p className="text-xs text-gray-500 leading-relaxed max-w-[200px]">
              Privacy-preserving LLM inference over peer-to-peer networks.
            </p>
          </div>

          {/* Link columns */}
          {links.map(({ heading, items }) => (
            <div key={heading}>
              <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-widest mb-4">{heading}</h4>
              <ul className="flex flex-col gap-2.5">
                {items.map(({ label, href }) => (
                  <li key={label}>
                    <a
                      href={href}
                      className="text-sm text-gray-500 hover:text-white transition-colors"
                    >
                      {label}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Bottom bar */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-8 border-t border-white/5">
          <p className="text-xs text-gray-600 font-mono">
            © {new Date().getFullYear()} PrivatePrompting. MIT License.
          </p>
          <p className="text-xs text-gray-600">
            Built with{' '}
            <span className="text-purple-500">hivemind</span> ·{' '}
            <span className="text-blue-500">libp2p</span> ·{' '}
            <span className="text-pink-500">Qwen2.5</span>
          </p>
        </div>
      </div>
    </footer>
  )
}
