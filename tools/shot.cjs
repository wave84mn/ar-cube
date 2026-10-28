const { chromium } = require('/opt/node22/lib/node_modules/playwright');
(async () => {
  const [url, out, w = '1600', h = '800'] = process.argv.slice(2);
  const b = await chromium.launch({ args: ['--enable-unsafe-swiftshader', '--use-angle=swiftshader'] });
  const p = await b.newPage({ viewport: { width: +w, height: +h } });
  const logs = []; p.on('console', m => logs.push(m.text())); p.on('pageerror', e => logs.push('ERR ' + e.message));
  await p.goto(url); await p.waitForFunction(() => document.title === 'done', null, { timeout: 120000 });
  console.log(JSON.stringify(await p.evaluate(() => window.__info)));
  await p.screenshot({ path: out }); console.log(logs.join('\n')); await b.close();
})();
