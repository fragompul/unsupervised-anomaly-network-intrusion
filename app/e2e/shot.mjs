// Usage: node e2e/shot.mjs <route> <out.png> [width] [height] [waitMs] [clickSelector]
import { chromium } from '@playwright/test'

const [route = '/', out = 'shot.png', w = '1440', h = '900', wait = '2500', click] = process.argv.slice(2)
const base = process.env.BASE_URL ?? 'http://localhost:5199/'
const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] })
const page = await browser.newPage({ viewport: { width: +w, height: +h }, deviceScaleFactor: 1 })
const logs = []
page.on('console', (m) => ['error', 'warning'].includes(m.type()) && logs.push(`[${m.type()}] ${m.text()}`))
page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`))
await page.goto(`${base}#${route}`, { waitUntil: 'networkidle' })
await page.waitForTimeout(+wait)
if (click) {
  await page.mouse.click(...click.split(',').map(Number))
  await page.waitForTimeout(1200)
}
await page.screenshot({ path: out })
console.log(logs.length ? logs.join('\n') : 'no console errors')
await browser.close()
