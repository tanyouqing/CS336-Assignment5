/* Document-only browser QA; no models, training, or assignment tests run. */
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const implementation = process.argv.includes('--implementation');
(async () => {
  const browser = await chromium.launch({ headless: true,
    ...(process.env.BROWSER_EXECUTABLE ? { executablePath: process.env.BROWSER_EXECUTABLE } : {}) });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const requests = [];
    page.on('request', request => requests.push(request.url()));
    await page.goto(pathToFileURL(path.resolve(__dirname, '../docs', implementation ? 'code_walkthrough.html' : 'post_training_tutorial.html')).href);
    assert.equal(await page.locator('h2[id]').count(), 15);
    assert.equal(await page.locator('.lab').count(), implementation ? 5 : 4);
    assert.match(await page.locator('#adv-result').innerText(), /0\.5774/);
    assert.match(await page.locator('#dpo-result').innerText(), /0\.6444/);
    assert.match(await page.locator('#clip-result').innerText(), /被裁剪/);
    await page.locator('#clip-a').selectOption('-1');
    assert.match(await page.locator('#clip-result').innerText(), /仍活跃/);
    await page.locator('.reward[data-index="1"]').click();
    await page.locator('.reward[data-index="2"]').click();
    assert.match(await page.locator('#adv-result').innerText(), /0\/4/);
    await page.locator('#answers').click();
    assert.equal(await page.locator('details[open]').count(), implementation ? 8 : 12);
    if (implementation) {
      for (const [route, count] of [['rl', 11], ['sft', 8], ['dpo', 10]]) {
        await page.locator('#pipeline-route').selectOption(route);
        assert.match(await page.locator('#pipeline-position').innerText(), new RegExp(`1 / ${count}`));
        for (let index = 1; index < count; index++) await page.locator('#pipeline-next').click();
        assert.equal(await page.locator('#pipeline-next').isDisabled(), true);
        assert.match(await page.locator('#pipeline-position').innerText(), new RegExp(`${count} / ${count}`));
        await page.locator('#pipeline-reset').click();
        assert.equal(await page.locator('#pipeline-prev').isDisabled(), true);
      }
    }
    await page.locator('#theme').click();
    assert.equal(await page.locator('body.dark').count(), 1);
    await page.locator('#theme').click();
    const output = path.resolve(__dirname, '../.cache', implementation ? 'code-walkthrough-qa' : 'tutorial-qa');
    fs.mkdirSync(output, { recursive: true });
    await page.evaluate(() => scrollTo(0, 0));
    await page.screenshot({ path: path.join(output, 'desktop.png') });
    if (implementation) {
      await page.locator('#pipeline').scrollIntoViewIfNeeded();
      await page.screenshot({ path: path.join(output, 'pipeline.png') });
    }
    await page.locator('#clip').scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(output, 'labs.png') });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.evaluate(() => scrollTo(0, 0));
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.screenshot({ path: path.join(output, 'mobile.png') });
    assert.deepEqual(errors, []);
    assert.equal(requests.some(url => /^https?:/.test(url)), false);
    console.log('Tutorial QA passed: sections, toy math, controls, mobile layout, offline requests, browser errors.');
    console.log(output);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
