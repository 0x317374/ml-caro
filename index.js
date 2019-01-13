const puppeteer = require('puppeteer')
const fs        = require('fs-extra')

fs.ensureDirSync('./cookies')

const CONF = {
  email: 'onggiachongchong@gmail.com',
  password: '',
  gpu: true,
  google_drive_code: [`
!apt-get install -y -qq software-properties-common module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse`.trim(), `
from google.colab import auth
auth.authenticate_user()
`.trim(), `
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}
`.trim(), `
!mkdir -p drive
!google-drive-ocamlfuse drive
`.trim()],
  exec: `
!cp -a ./drive/mini-game/. ./
!cd gobang6x4 && python train.py
`.trim(),
}

let new_browser = () => {
  return puppeteer.launch({
    headless: false,
    ignoreHTTPSErrors: true,
    args: ['--disable-infobars', '--disable-infobars', '--window-size=1000,600', '--disable-features=site-per-process'],
  })
}

let save_cookie = async (page) => {
  let cookies = (await page.cookies()) || []
  if(cookies.length===0) return
  fs.writeFileSync(['./cookies', CONF.email].join('/'), JSON.stringify(cookies))
}

let load_cookie = async (page) => {
  let cookie_file = ['./cookies', CONF.email].join('/')
  if(!fs.existsSync(cookie_file)) return false
  await page.setCookie(...JSON.parse(fs.readFileSync(cookie_file, 'utf-8')))
  return true
}

let press = async (keyboard, str) => {
  str = [str].join('')
  for(let c of str.split('')) {
    await keyboard.sendCharacter(c)
  }
}

let add_code = async (page, code) => {
  code = [code].join('').trim()

  let add_code_btn = await page.waitForSelector('#top-toolbar colab-toolbar-button[command="add-code"]')
  await add_code_btn.click()
  await page.waitFor(100)
  await press(page.keyboard, code)
}

let run_all_code = async (keyboard) => {
  await keyboard.down('Control')
  await keyboard.press('F9')
  await keyboard.up('Control')
}

let delete_cell = async (page) => {
  await page.keyboard.down('Control')
  await page.keyboard.down('Shift')
  await page.keyboard.press('p')
  await page.keyboard.up('Control')
  await page.keyboard.up('Shift')
  await page.waitFor(500)
  await press(page.keyboard, 'delete cell')
  await page.keyboard.press('\n')
  await page.waitFor(500)
}

let command = async (page, cmd) => {
  await page.keyboard.down('Control')
  await page.keyboard.down('Shift')
  await page.keyboard.press('p')
  await page.keyboard.up('Control')
  await page.keyboard.up('Shift')
  await page.waitFor(500)
  await press(page.keyboard, cmd)
  await page.keyboard.press('\n')
  await page.waitFor(500)
}

const TASK = async (browser) => {
  let page = (await browser.pages())[0]
  await page._client.send('Emulation.clearDeviceMetricsOverride')
  // login
  if(!await load_cookie(page)) {
    await page.goto('https://accounts.google.com/signin/v2/identifier?hl=en&passive=true&continue=https%3A%2F%2Fcolab.research.google.com%2F&flowName=GlifWebSignIn&flowEntry=ServiceLogin', { waitUntil: 'networkidle0' })
    await page.waitForSelector('input[type="email"]')
    await page.evaluate((email) => {
      document.querySelector('input[type="email"]').value = email
    }, CONF.email)
    await page.keyboard.press('\n')
    await page.waitFor(1000)
    await page.waitForSelector('input[type="password"]')
    await page.waitFor(1000)
    await page.evaluate(password => document.querySelector('input[type="password"]').value = password, CONF.password)
    await page.keyboard.press('\n')
    await page.waitForNavigation()
    await page.waitFor(1000)
    await save_cookie(page)
  }
  await page.goto('https://colab.research.google.com/notebooks/welcome.ipynb#recent=true', { waitUntil: 'domcontentloaded' })
  await page.waitForSelector('paper-tab')
  await page.evaluate(() => document.querySelectorAll('paper-tab')[2].click())
  await page.evaluate(() => document.querySelector('paper-button.new-notebook').click())
  await page.waitForNavigation()
  await page.waitForSelector('#file-menu-button')
  // Choose GPU
  if(CONF.gpu) {
    await command(page, 'Change runtime type')
    await (await page.waitForSelector('#accelerators-menu')).click()
    await (await page.waitForSelector('#accelerators-menu paper-item[value="GPU"]')).click()
    await (await page.waitForSelector('#ok')).click()
  }
  // Reset all runtimes
  await command(page, 'Reset all runtimes')
  await page.keyboard.press('\n')
  for(let code of CONF.google_drive_code) await add_code(page, code)
  await add_code(page, CONF.exec)
  await run_all_code(page.keyboard)
  for(let i = 0; i<2; i++) {
    let frame
    let token_link
    let finding_google_link = true
    while(finding_google_link) {
      try {
        await page.waitFor(500)
      } catch(e) {
      }
      let list_frames = await page.frames()
      for(let v of list_frames) {
        let find_result = await v.evaluate(() => {
          let element = document.querySelector('pre a[target="_blank"]')
          if(element && element.getAttribute('href')) return element.getAttribute('href').length>0
          return false
        })
        if(!find_result) continue
        frame      = v
        token_link = await frame.waitForSelector('pre a[target="_blank"]')

        finding_google_link = false
        break
      }
    }
    await token_link.click()
    await browser.waitForTarget(target => target.url().indexOf('accounts.google.com')!== -1)
    await page.waitFor(500)
    let token_page = (await browser.pages())[1]
    await token_page.waitForSelector('div[tabindex="0"]')
    await token_page.waitFor(1000)
    await token_page.evaluate(() => document.querySelector('div[tabindex="0"]').click())
    let access_btn = await token_page.waitForSelector('.ZFr60d.CeoRYc')
    await token_page.waitFor(1000)
    await access_btn.click()
    await token_page.waitForNavigation()
    let token = await token_page.evaluate(() => document.querySelector('textarea').value)
    await token_page.close()
    await frame.evaluate(() => document.querySelector('input.raw_input').focus())
    await frame.evaluate(token => document.querySelector('input.raw_input').value = token, token)
    await page.waitFor(1000)
    await page.keyboard.press('\n')
    await page.waitFor(1000)
  }
  await command(page, 'Clear all outputs')
}

(async () => {
  let browser = await new_browser()
  TASK(browser).catch((e) => {console.log(e)})
})()
