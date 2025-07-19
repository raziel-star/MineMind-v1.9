import os
import json
import subprocess
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from nltk.tokenize import word_tokenize
import re
import time

nltk.download('punkt', quiet=True)

def bag_of_words(tokenized_sentence, all_words):
    word_to_index = {w: i for i, w in enumerate(all_words)}
    vector = torch.zeros(len(all_words), dtype=torch.float32)
    for w in tokenized_sentence:
        idx = word_to_index.get(w)
        if idx is not None:
            vector[idx] = 1.0
    return vector

class TorchMind(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

class CommandAI:
    def __init__(self, data_file='learned_ai.json'):
        self.data_file = data_file
        self.commands = {}
        self.examples = []
        self.labels = []
        self.all_words = []
        self.label_map = {}
        self.reverse_map = {}
        self.model = None
        self.trained = False
        self.hidden_size = 128
        self.load_data()
        if len(self.labels) > 0:
            self.train()

    def add_example(self, text, label, js_code):
        tokens = word_tokenize(text.lower())
        # הוספת דוגמה רק אם לא קיימת כבר (למנוע כפילויות)
        if tokens in self.examples and label in self.labels:
            return
        self.examples.append(tokens)
        self.labels.append(label)
        self.commands[label] = js_code
        for t in tokens:
            if t not in self.all_words:
                self.all_words.append(t)
        if label not in self.label_map:
            idx = len(self.label_map)
            self.label_map[label] = idx
            self.reverse_map[idx] = label

    def train(self):
        if not self.examples:
            return
        X = torch.stack([bag_of_words(e, self.all_words) for e in self.examples])
        y = torch.tensor([self.label_map[l] for l in self.labels])
        input_size = len(self.all_words)
        output_size = len(self.label_map)
        self.model = TorchMind(input_size, self.hidden_size, output_size)
        optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for epoch in range(300):
            optimizer.zero_grad()
            output = self.model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        self.trained = True
        print(f"🧠 אימון הושלם - אוצר מילים: {input_size}, פקודות: {output_size}")

    def predict_js(self, text, username="PlayerName", threshold=0.9):
        # תחילה נבדוק פקודות עם מספר ספציפי
        m_mine = re.search(r'חצוב\s*(\d+)\s*עצים?', text)
        if m_mine:
            amount = int(m_mine.group(1))
            js_code = f"addTask(cb => mineWoodAmount({amount}, cb));"
            return js_code, "mine_wood_amount"

        m_give = re.search(r'תן\s*(\d+)\s*עצים?', text)
        if m_give:
            amount = int(m_give.group(1))
            js_code = f"addTask(cb => giveWoodAmount({amount}, cb));"
            return js_code, "give_wood_amount"

        # אם לא, ננסה לחזות פקודה רגילה בעזרת המודל
        if not self.trained:
            return None, None
        tokens = word_tokenize(text.lower())
        vec = bag_of_words(tokens, self.all_words)
        self.model.eval()
        with torch.no_grad():
            output = self.model(vec.unsqueeze(0))[0]
            probs = torch.softmax(output, dim=0)
            conf, pred = torch.max(probs, 0)
            label = self.reverse_map[pred.item()]
            print(f"[DEBUG] input: \"{text}\" | predicted: {label} | confidence: {conf.item():.2f}")
            if conf.item() < threshold:
                return None, None
            js_code = self.commands.get(label, None)
            if js_code:
                js_code = js_code.replace("{username}", username)
            return js_code, label

    def save_data(self):
        data = {
            "commands": self.commands,
            "examples": [' '.join(e) for e in self.examples],
            "labels": self.labels,
            "all_words": self.all_words
        }
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("💾 שמירת נתוני למידה הסתיימה")

    def load_data(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.commands = data.get("commands", {})
                self.labels = data.get("labels", [])
                self.examples = [e.split() for e in data.get("examples", [])]
                self.all_words = data.get("all_words", [])
                self.label_map = {}
                self.reverse_map = {}
                for label in set(self.labels):
                    idx = len(self.label_map)
                    self.label_map[label] = idx
                    self.reverse_map[idx] = label

def generate_bot_js():
    js = r"""
const mineflayer = require('mineflayer')
const { pathfinder, Movements, goals: { GoalFollow } } = require('mineflayer-pathfinder')
const pvp = require('mineflayer-pvp')
const armorManager = require('mineflayer-armor-manager')
const collectBlock = require('mineflayer-collectblock').plugin
const mcData = require('minecraft-data')

const bot = mineflayer.createBot({
  host: 'cuberazi.aternos.me',
  port: 46828,
  username: 'Claude',
  version: '1.20.1'
})

bot.loadPlugin(pathfinder)
bot.loadPlugin(pvp.plugin || pvp)
bot.loadPlugin(collectBlock)
armorManager(bot)

let taskQueue = []
let isBusy = false

function runNextTask() {
  if (isBusy || taskQueue.length === 0) return
  isBusy = true
  const task = taskQueue.shift()
  task(() => {
    isBusy = false
    runNextTask()
  })
}

function addTask(task) {
  taskQueue.push(task)
  runNextTask()
}

// חציבת עצים לפי כמות מדויקת
async function mineWoodAmount(amount, cb) {
  const data = mcData(bot.version)
  const woodBlocks = ['oak_log', 'birch_log', 'spruce_log', 'jungle_log', 'acacia_log', 'dark_oak_log']
  let mined = 0

  while (mined < amount) {
    const block = bot.findBlock({
      matching: b => woodBlocks.includes(data.blocks[b.type].name),
      maxDistance: 6
    })

    if (!block) {
      bot.chat('❌ לא מצאתי עץ לאסוף כרגע, אחכה קצת...')
      await new Promise(r => setTimeout(r, 3000))
      continue
    }

    try {
      await bot.collectBlock.collect(block)
      mined++
      bot.chat(`✅ אספתי בלוק עץ ${mined} מתוך ${amount}.`)
      await new Promise(r => setTimeout(r, 1000))
    } catch (err) {
      bot.chat('❌ שגיאה בחציבת עץ: ' + err.message)
      await new Promise(r => setTimeout(r, 2000))
    }
  }
  bot.chat(`🛑 סיימתי לחצוב ${amount} עצים.`)
  cb()
}

// זריקת עצים לפי כמות ספציפית מהאינבנטורי
async function giveWoodAmount(amount, cb) {
  const woodNames = ['oak_log', 'birch_log', 'spruce_log', 'jungle_log', 'acacia_log', 'dark_oak_log']
  let countToThrow = amount
  const itemsToThrow = []

  for (const item of bot.inventory.items()) {
    if (woodNames.includes(item.name)) {
      itemsToThrow.push(item)
    }
  }

  if (itemsToThrow.length === 0) {
    bot.chat('❌ אין לי עצים לזרוק.')
    cb()
    return
  }

  for (const item of itemsToThrow) {
    if (countToThrow <= 0) break
    const toThrowCount = Math.min(item.count, countToThrow)
    try {
      await bot.toss(item.type, null, toThrowCount)
      countToThrow -= toThrowCount
      bot.chat(`✅ זרקתי ${toThrowCount} עץ מסוג ${item.name}.`)
    } catch (err) {
      bot.chat('❌ נכשלתי לזרוק עץ: ' + err.message)
    }
  }

  if (countToThrow > 0) {
    bot.chat(`⚠ לא היה לי מספיק עצים לזרוק את כל הכמות שביקשת.`)
  }
  cb()
}

function stopMiningWood() {
  // אין כרגע תמיכה בחציבה רציפה, רק לפי כמות מדויקת
  bot.chat('🚫 הפקודה לא פעילה - חציבה לפי כמות בלבד.')
}

bot.once('spawn', () => {
  bot.chat('💥 היי, אני Claude איך אני יכול לעזור לך היום?')
  bot.armorManager.equipAll()
  startAutoPvP()
})

function startAutoPvP() {
  setInterval(() => {
    if (bot.pvp.target) return
    const target = bot.nearestEntity(entity =>
      entity.type === 'hostile' &&
      entity.name.toLowerCase().includes('zombie') &&
      bot.entity.position.distanceTo(entity.position) < 10
    )
    if (target) {
      bot.chat('🧟‍♂ תוקף זומבי!')
      bot.pvp.attack(target)
    }
  }, 1000)
}

bot.on('chat', (username, message) => {
  if (username === bot.username) return
  console.log(`[CHAT] ${username}: ${message}`)
})

bot.on('playerCollect', (collector, collected) => {
  if (collector.username === bot.username) {
    setTimeout(() => bot.armorManager.equipAll(), 500)
  }
})

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
"""
    with open("bot.js", "w", encoding="utf-8") as f:
        f.write(js)

def main():
    ai = CommandAI()

    base_cmds = [
        ("לך קדימה", "forward", "bot.setControlState('forward', true);"),
        ("לך אחורה", "back", "bot.setControlState('back', true);"),
        ("סגור", "stop", "bot.clearControlStates(); stopMiningWood();"),
        ("עצור", "stop", "bot.clearControlStates(); stopMiningWood();"),
        ("קפוץ", "jump", "bot.setControlState('jump', true); setTimeout(() => bot.setControlState('jump', false), 300);"),
        ("הסתכל למעלה", "look_up", "bot.look(bot.entity.yaw, bot.entity.pitch - 0.5, true);"),
        ("הסתכל למטה", "look_down", "bot.look(bot.entity.yaw, bot.entity.pitch + 0.5, true);"),
        ("הפעל יצירתיות", "creative_mode", "bot.chat('/gamemode creative');"),
        ("הפעל הישרדות", "survival_mode", "bot.chat('/gamemode survival');"),
        ("פתח דלת", "open_door", "addTask(openDoor);"),
        ("חפור קדימה", "dig_forward", "addTask(cb => digMultiple(1, cb));"),
        ("חפור 5", "dig_5", "addTask(cb => digMultiple(5, cb));"),
        ("עקוב אחרי", "follow_player", """
const target = bot.players['{username}']?.entity;
if (target) {
  const { GoalFollow } = require('mineflayer-pathfinder').goals;
  bot.pathfinder.setGoal(new GoalFollow(target, 1), true);
  bot.chat(`👣 עוקב אחרי {username}`);
}
"""),
        ("הפסיק לעקוב אחרי", "stop_following", "stopFollowing(() => {});"),
        ("אסוף עץ", "collect_wood", "addTask(cb => mineWoodAmount(20, cb));"),  # חציבת 20 עצים
        ("הפסק חציבה", "stop_mining", "stopMiningWood();"),
        ("כמה עצים יש לך", "count_wood", "countWood();"),
        ("תן עץ", "give_wood", "addTask(cb => giveWoodAmount(20, cb));"),  # זריקת 20 עצים
        ("אסוף זהב", "collect_gold", "addTask(collectGoldOre);"),
        ("אסוף אוכל", "collect_food", "addTask(collectFoodBlock);"),
    ]

    for text, label, js in base_cmds:
        ai.add_example(text, label, js)

    ai.train()
    ai.save_data()
    generate_bot_js()

    print("🚀 מפעיל את הבוט ומאזין לצ'אט...")

    proc = subprocess.Popen(
        ["node", "bot.js"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1
    )

    connected_flag = False

    def listen():
        nonlocal connected_flag
        for line in proc.stdout:
            line = line.strip()
            print(f"[BOT LOG] {line}")
            if line == "💥 היי, אני Claude איך אני יכול לעזור לך היום?":
                connected_flag = True
                print("✅ הבוט התחבר לעולם!")
            elif line.startswith("[CHAT]"):
                try:
                    user, msg = line[7:].split(":", 1)
                    user = user.strip()
                    msg = msg.strip()
                    print(f"[CHAT RECEIVED] {user}: {msg}")
                except:
                    continue
                js_code, label = ai.predict_js(msg, user)
                if js_code:
                    try:
                        proc.stdin.write(js_code + "\n")
                        proc.stdin.write(f"bot.chat('בטח! *used {label}*');\n")
                        proc.stdin.flush()
                    except Exception as e:
                        print(f"[ERROR writing to bot stdin]: {e}")

    threading.Thread(target=listen, daemon=True).start()

    timeout = 40
    start_time = time.time()
    while time.time() - start_time < timeout:
        if connected_flag:
            break
        time.sleep(0.1)

    if not connected_flag:
        print("❌ הבוט לא התחבר תוך 40 שניות.")

    try:
        while True:
            inp = input(">>> ")
            if inp.strip() == "exit":
                print("⚠ יציאה...")
                proc.terminate()
                break
            js_code, label = ai.predict_js(inp, "PlayerName")
            if js_code:
                proc.stdin.write(js_code + "\n")
                proc.stdin.write(f"bot.chat('*used {label}*');\n")
                proc.stdin.flush()
            else:
                print("[AI] לא זיהה פקודה.")
    except KeyboardInterrupt:
        proc.terminate()

if __name__ == "__main__":
    main()
