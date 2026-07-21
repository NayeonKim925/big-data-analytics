import { chromium } from 'playwright';
import path from 'path';

const DIR = path.resolve('charts');
const browser = await chromium.launch({
  executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
});
const page = await browser.newPage({ deviceScaleFactor: 3 });
await page.goto('file://' + path.join(DIR, 'all.html'), { waitUntil: 'networkidle' });
await page.evaluate(() => document.fonts.ready);
await page.waitForTimeout(150);
const ids = await page.$$eval('.chart', els => els.map(e => e.id));
for (const id of ids) {
  const el = await page.$('#' + id);
  await el.screenshot({ path: path.join(DIR, id + '.png'), omitBackground: true });
  console.log('chart', id);
}
await browser.close();
