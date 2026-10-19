import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vitest/config'

// Relative base: the app is served from a GitHub Pages sub-path and uses hash routing.
export default defineConfig({
  base: './',
  plugins: [react(), tailwindcss()],
  build: {
    target: 'es2022',
    chunkSizeWarningLimit: 1400,
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (id.includes('node_modules/three') || id.includes('@react-three') || id.includes('postprocessing')) return 'three'
          if (id.includes('framer-motion') || id.includes('motion-dom')) return 'motion'
          return undefined
        },
      },
    },
  },
  test: { environment: 'node', include: ['src/**/*.test.ts'] },
})
