// End-to-end: run the viewer in Chromium with a fake camera fed from a y4m file; screenshot + state dump.
// usage: node tests/e2e.cjs VIDEO.y4m OUTPREFIX [url-query] [seconds]
const { chromium } = require('/opt/node22/lib/node_modules/playwright');
(async () => {
  const [video, out, query = '', secs = '6'] = process.argv.slice(2);
  const browser = await chromium.launch({
    args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-video-capture=${video}`,
      '--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--ignore-gpu-blocklist'],
  });
  const vp = (process.env.VP || '390x844').split('x').map(Number);
  const ctx = await browser.newContext({ viewport: { width: vp[0], height: vp[1] }, deviceScaleFactor: 2, permissions: ['camera'] });
  const page = await ctx.newPage();
  const logs = [];
  page.on('console', m => logs.push(`[${m.type()}] ${m.text()}`));
  page.on('pageerror', e => logs.push(`[pageerror] ${e.message}`));
  await page.goto(`http://localhost:8123/${query}`);
  await page.waitForTimeout(800);
  await page.screenshot({ path: `${out}_start.png` });
  await page.click('#go');
  const t0 = Date.now();
  const samples = [];
  for (let i = 0; i < +secs * 2; i++) {
    await page.waitForTimeout(500);
    const s = await page.evaluate(() => {
      const a = window.__arcube; if (!a) return null;
      const st = a.active ? a.T[a.active] : null;
      return { active: a.active, fps: +a.fps.toFixed(1), procMs: +a.procMs.toFixed(1), markers: a.lastRes ? a.lastRes.markers.length : 0,
        n: st && st.raw ? st.raw.n : 0, timing: Object.fromEntries(Object.entries(a.timing).map(([k, v]) => [k, +v.toFixed(1)])), rms: st && st.raw ? +st.raw.rms.toFixed(2) : null, video: [a.view.vw, a.view.vh] };
    });
    samples.push({ t: ((Date.now() - t0) / 1000).toFixed(1), ...s });
    if (i === 5 || i === +secs * 2 - 1) await page.screenshot({ path: `${out}_${i}.png` });
  }
  console.log(samples.map(s => JSON.stringify(s)).join('\n'));
  console.log('--- console ---\n' + logs.slice(0, 30).join('\n'));
  await browser.close();
})();
