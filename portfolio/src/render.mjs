import { chromium } from 'playwright';
import { readdirSync } from 'fs';
import path from 'path';

const OUT = path.resolve('out');
const args = process.argv.slice(2);
const only = args.length ? args : readdirSync(OUT).filter(f => f.endsWith('.html')).sort();

const browser = await chromium.launch({
  executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
});
const page = await browser.newPage({
  viewport: { width: 1920, height: 1080 },
  deviceScaleFactor: 2,
});
for (const f of only) {
  const name = f.replace('.html', '');
  const url = 'file://' + path.join(OUT, f);
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(200);
  const png = path.join(OUT, name + '.png');
  await page.screenshot({ path: png, clip: { x:0, y:0, width:1920, height:1080 } });
  console.log('rendered', png);
}
await browser.close();
