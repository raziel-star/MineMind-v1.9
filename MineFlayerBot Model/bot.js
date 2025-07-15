
const mineflayer = require('mineflayer')
const { pathfinder } = require('mineflayer-pathfinder')
const pvp = require('mineflayer-pvp')
const armorManager = require('mineflayer-armor-manager')

const Vec3 = require('vec3')

const bot = mineflayer.createBot({
  host: 'cuberazi.aternos.me',
  port: 25565,
  username: 'Claude',
  version: '1.20.1'
})

bot.loadPlugin(pathfinder)
bot.loadPlugin(pvp.plugin || pvp)
armorManager(bot)

bot.once('spawn', () => {
  bot.chat('💥 הבוט מוכן להילחם בזומבים!')
  bot.armorManager.equipAll()
  startAutoPvP()
})

function startAutoPvP() {
  setInterval(() => {
    if (bot.pvp.target) return // כבר תוקף מישהו

    // מחפש את הזומבי הקרוב ביותר בטווח 6 בלוקים
    const target = bot.nearestEntity(entity =>
      entity.type === 'hostile' &&
      (entity.name.toLowerCase() === 'zombie' || entity.name.toLowerCase().includes('zombie')) &&
      bot.entity.position.distanceTo(entity.position) < 6
    )

    if (target) {
      bot.chat('🧟‍♂️ תוקף זומבי!')
      bot.pvp.attack(target)
    }
  }, 1000)
}

bot.on('entitySpawn', entity => {
  if (entity) {
    console.log('📡 ישות נוצרה:', {
      type: entity.type,
      name: entity.name,
      displayName: entity.displayName,
      username: entity.username,
      pos: entity.position
    })
  }
})

bot.on('chat', (username, message) => {
  if (username === bot.username) return
  console.log(`[CHAT] ${username}: ${message}`)
})

bot.on('playerCollect', async (collector, collected) => {
  if (collector.username === bot.username) {
    setTimeout(() => bot.armorManager.equipAll(), 500)
  }
})

// מאזין ל־stdin להרצת פקודות JS שמגיעות מפייתון
process.stdin.setEncoding('utf8')
process.stdin.on('data', data => {
  const code = data.toString().trim()
  console.log('[JS stdin received]:', code)
  try {
    eval(code)
    console.log('[JS EXECUTED]')
  } catch (err) {
    console.error('[JS ERROR]', err)
  }
})
