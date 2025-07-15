import os
import json
import subprocess
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from nltk.tokenize import word_tokenize
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

    def predict_js(self, text, threshold=0.9):
        if not self.trained:
            return None
        tokens = word_tokenize(text.lower())
        vec = bag_of_words(tokens, self.all_words)
        self.model.eval()
        with torch.no_grad():
            output = self.model(vec.unsqueeze(0))[0]
            probs = torch.softmax(output, dim=0)
            conf, pred = torch.max(probs, 0)
            print(f"[DEBUG] input: \"{text}\" | predicted: {self.reverse_map[pred.item()]} | confidence: {conf.item():.2f}")
            if conf.item() < threshold:
                return None
            label = self.reverse_map[pred.item()]
            return self.commands.get(label, None)

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

    def retrain(self):
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
        for epoch in range(100):
            optimizer.zero_grad()
            output = self.model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        self.trained = True
        print(f"🧠 אימון מחודש הושלם - אוצר מילים: {input_size}, פקודות: {output_size}")

def generate_bot_js():
    js = r"""
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
"""
    with open("bot.js", "w", encoding="utf-8") as f:
        f.write(js)

def main():
    ai = CommandAI()

    # פקודות התחלתיות לדוגמא
    base_cmds = [
        ("לך קדימה", "forward", "bot.setControlState('forward', true);"),
        ("לך אחורה", "back", "bot.setControlState('back', true);"),
        ("סגור", "stop", "bot.clearControlStates();"),
        ("עצור", "stop", "bot.clearControlStates();"),
        ("קפוץ", "jump", "bot.setControlState('jump', true); setTimeout(() => bot.setControlState('jump', false), 300);"),
        ("הסתכל למעלה", "look_up", "bot.look(bot.entity.yaw, bot.entity.pitch - 0.5, true);"),
        ("הסתכל למטה", "look_down", "bot.look(bot.entity.yaw, bot.entity.pitch + 0.5, true);"),
        ("הפעל יצירתיות", "creative_mode", "bot.chat('/gamemode creative');"),
        ("הפעל הישרדות", "survival_mode", "bot.chat('/gamemode survival');"),
        ("פתח דלת", "open_door", """
const door = bot.blockAt(bot.entity.position.offset(0, 0, 1));
if(door && door.name.includes('door')) bot.activateBlock(door);
"""),
        ("חפור קדימה", "dig_forward", """
const vec = bot.entity.position.offset(0, 0, 1);
const block = bot.blockAt(vec);
if (block) bot.dig(block);
"""),
    ]

    # הוספת פקודות לדאטה
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
            if line == "💥 הבוט מוכן להילחם בזומבים!":
                connected_flag = True
                print("✅ הבוט התחבר לעולם!")
            elif line.startswith("[CHAT]"):
                try:
                    # פורמט ההדפסה ב־bot.js הוא: [CHAT] username: message
                    user, msg = line[6:].split(":", 1)
                    user = user.strip()
                    msg = msg.strip()
                    print(f"[CHAT RECEIVED] {user}: {msg}")
                except Exception:
                    continue

                # בוט ינתח את ההודעה עם המודל וישלח פקודה אם יש
                js = ai.predict_js(msg)
                if js:
                    print(f"[AI] מבצע פקודה מהצ'אט: {msg}")
                    proc.stdin.write(js + "\n")
                    proc.stdin.flush()
                    # אפשר לשלוח גם הודעה בבוט שמאשר ביצוע
                    proc.stdin.write(f"bot.chat('🧠 מבצע פקודה: {msg}');\n")
                    proc.stdin.flush()

    thread = threading.Thread(target=listen, daemon=True)
    thread.start()

    # מחכים לבוט להתחבר עד 15 שניות
    timeout = 15
    start_time = time.time()
    while time.time() - start_time < timeout:
        if connected_flag:
            break
        time.sleep(0.1)

    if not connected_flag:
        print("❌ הבוט לא התחבר תוך 15 שניות, בדוק פרטי חיבור.")

    try:
        while True:
            inp = input(">>> ")
            if inp.strip() == "exit":
                print("⚠️ יציאה...")
                proc.terminate()
                break
            # כל קלט ידני עובר ניתוח ב-AI
            js = ai.predict_js(inp)
            if js:
                proc.stdin.write(js + "\n")
                proc.stdin.flush()
                print(f"[AI] ביצע פקודה ידנית: {inp}")
            else:
                print("[AI] לא זיהה פקודה.")
    except KeyboardInterrupt:
        proc.terminate()

if __name__ == "__main__":
    main()
