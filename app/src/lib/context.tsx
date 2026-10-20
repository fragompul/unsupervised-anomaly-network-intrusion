import { createContext, useContext, type ReactNode } from 'react'
import { useManifest, useSummary } from './hooks'
import type { Manifest, Summary } from './types'

interface Ctx {
  manifest: Manifest | null
  summary: Summary | null
  error: Error | null
}
const C = createContext<Ctx>({ manifest: null, summary: null, error: null })

export function DataProvider({ children }: { children: ReactNode }) {
  const m = useManifest()
  const s = useSummary()
  return <C.Provider value={{ manifest: m.data, summary: s.data, error: m.error ?? s.error }}>{children}</C.Provider>
}
export const useData = () => useContext(C)
