import { chromium } from 'playwright';
import { readdirSync, mkdirSync } from 'fs';
import path from 'path';

const HTM = path.resolve('fig/html');
const PNG = path.resolve('fig/png');
mkdirSync(PNG, { recursive: true });
const args = process.argv.slice(2);
const files = (args.length ? args.map(a => a.endsWith('.html') ? a : a + '.html')
                           : readdirSync(HTM).filter(f => f.endsWith('.html'))).sort();

const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' });
const page = await browser.newPage({ deviceScaleFactor: 2 });
for (const f of files) {
  const name = f.replace('.html', '');
  await page.goto('file://' + path.join(HTM, f), { waitUntil: 'networkidle' });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(120);
  const svg = page.locator('svg').first();
  await svg.screenshot({ path: path.join(PNG, name + '.png') });
  console.log('rendered', name + '.png');
}
await browser.close();
