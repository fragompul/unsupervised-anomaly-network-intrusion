import { expect, test } from '@playwright/test'

const IGNORED = [/GL Driver Message/, /GPU stall/, /THREE\.Clock/, /Failed to load resource.*paper\.pdf/]

test.beforeEach(async ({ page }) => {
  page.on('console', (m) => {
    if (m.type() === 'error' && !IGNORED.some((r) => r.test(m.text()))) throw new Error(`console error: ${m.text()}`)
  })
  page.on('pageerror', (e) => {
    throw e
  })
})

const ROUTES: [string, RegExp][] = [
  ['/', /under pressure/i],
  ['/explorer', /Top anomalies/i],
  ['/benchmark', /No single-number leaderboard/i],
  ['/compare', /one disagreement map/i],
  ['/robustness', /textbook setting/i],
  ['/replay', /guarantee/i],
  ['/playground', /Watch the verdict move/i],
  ['/about', /Contributions/i],
]

for (const [route, heading] of ROUTES) {
  test(`route ${route} renders without console errors`, async ({ page }) => {
    await page.goto(`/#${route}`)
    await expect(page.getByText(heading).first()).toBeVisible({ timeout: 20_000 })
  })
}

test('explorer: selecting a point opens the flow inspector', async ({ page }) => {
  await page.goto('/#/explorer')
  await expect(page.getByText(/Top anomalies/i)).toBeVisible({ timeout: 20_000 })
  const row = page.getByRole('button', { name: /#\d+/ }).first()
  await expect(row).toBeVisible({ timeout: 30_000 })
  await row.click()
  await expect(page.getByText(/Conformal p-value/i)).toBeVisible({ timeout: 20_000 })
  await expect(page.getByText(/autoencoder error attribution/i)).toBeVisible({ timeout: 20_000 })
})

test('playground: dragging a feature changes the verdict evidence', async ({ page }) => {
  await page.goto('/#/playground')
  const sliders = page.locator('input[type=range]')
  await expect(sliders.first()).toBeVisible({ timeout: 20_000 })
  const before = await page.locator('svg text.num').first().textContent()
  await sliders.first().fill('5')
  await sliders.nth(1).fill('-5')
  await sliders.nth(2).fill('5')
  await page.waitForTimeout(600)
  const after = await page.locator('svg text.num').first().textContent()
  expect(after).not.toBe(before)
})

test('navigation links move between pages', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('link', { name: 'Benchmark', exact: true }).first().click()
  await expect(page).toHaveURL(/#\/benchmark/)
  await page.getByRole('link', { name: 'Compare', exact: true }).first().click()
  await expect(page).toHaveURL(/#\/compare/)
})
