import { AnimatePresence, motion } from 'framer-motion'
import { ExternalLink } from 'lucide-react'
import { lazy, Suspense } from 'react'
import { HashRouter, NavLink, Route, Routes, useLocation } from 'react-router-dom'
import { Logo } from './components/Logo'
import { Skeleton, cx } from './components/ui'
import { DataProvider } from './lib/context'

const Home = lazy(() => import('./pages/Home'))
const Explorer = lazy(() => import('./pages/Explorer'))
const Benchmark = lazy(() => import('./pages/Benchmark'))
const Compare = lazy(() => import('./pages/Compare'))
const Robustness = lazy(() => import('./pages/Robustness'))
const Replay = lazy(() => import('./pages/Replay'))
const Playground = lazy(() => import('./pages/Playground'))
const About = lazy(() => import('./pages/About'))

const NAV = [
  { to: '/explorer', label: 'Explorer' },
  { to: '/benchmark', label: 'Benchmark' },
  { to: '/compare', label: 'Compare' },
  { to: '/robustness', label: 'Robustness' },
  { to: '/replay', label: 'Replay' },
  { to: '/playground', label: 'Playground' },
  { to: '/about', label: 'Paper' },
]

export const REPO = 'https://github.com/fragompul/unsupervised-anomaly-network-intrusion'

function Nav() {
  return (
    <nav className="fixed inset-x-0 top-0 z-50">
      <div className="mx-auto flex h-14 max-w-[1500px] items-center gap-6 px-4 sm:px-6">
        <NavLink to="/" className="focus-ring flex items-center gap-2.5 rounded-lg" aria-label="Noctua home">
          <Logo />
          <span className="text-[15px] font-semibold tracking-tight">Noctua</span>
        </NavLink>
        <div className="glass-flat hidden items-center gap-0.5 rounded-full p-1 md:flex" style={{ background: 'rgba(8,11,20,0.7)', backdropFilter: 'blur(14px)' }}>
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              className={({ isActive }) =>
                cx('focus-ring rounded-full px-3.5 py-1.5 text-[13px] font-medium transition-colors', isActive ? 'bg-white/[0.12] text-white' : 'text-muted hover:text-text')
              }
            >
              {n.label}
            </NavLink>
          ))}
        </div>
        <div className="ml-auto flex items-center gap-2">
          <a
            href={REPO}
            target="_blank"
            rel="noreferrer"
            className="focus-ring glass-flat flex items-center gap-1.5 px-3 py-1.5 text-[13px] font-medium text-muted transition-colors hover:text-text"
          >
            GitHub <ExternalLink size={13} />
          </a>
        </div>
      </div>
      {/* mobile nav */}
      <div className="no-scrollbar flex gap-1 overflow-x-auto px-4 pb-2 md:hidden">
        {NAV.map((n) => (
          <NavLink key={n.to} to={n.to} className={({ isActive }) => cx('shrink-0 rounded-full px-3 py-1 text-xs font-medium', isActive ? 'bg-white/[0.14] text-white' : 'text-muted')}>
            {n.label}
          </NavLink>
        ))}
      </div>
    </nav>
  )
}

function Fallback() {
  return (
    <div className="mx-auto max-w-[1500px] px-6 pt-32">
      <Skeleton className="mb-4 h-10 w-80" />
      <Skeleton className="h-[420px] w-full" />
    </div>
  )
}

function Routed() {
  const loc = useLocation()
  return (
    <AnimatePresence mode="wait">
      <motion.main key={loc.pathname} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.18 }}>
        <Suspense fallback={<Fallback />}>
          <Routes location={loc}>
            <Route path="/" element={<Home />} />
            <Route path="/explorer" element={<Explorer />} />
            <Route path="/benchmark" element={<Benchmark />} />
            <Route path="/compare" element={<Compare />} />
            <Route path="/robustness" element={<Robustness />} />
            <Route path="/replay" element={<Replay />} />
            <Route path="/playground" element={<Playground />} />
            <Route path="/about" element={<About />} />
            <Route path="*" element={<Home />} />
          </Routes>
        </Suspense>
      </motion.main>
    </AnimatePresence>
  )
}

export default function App() {
  return (
    <HashRouter>
      <DataProvider>
        <Nav />
        <Routed />
      </DataProvider>
    </HashRouter>
  )
}
