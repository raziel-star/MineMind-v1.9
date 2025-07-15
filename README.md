ChatGPT said:
markdown
Copy
Edit
# iSparkton-v1.9 the Minecraft AI Bot 🤖🧠

iSparkton is an intelligent Minecraft bot that listens to chat messages in real time, understands Hebrew commands using a lightweight neural network, and performs actions automatically in-game using Mineflayer.

---

## 🧩 Features

- 💬 Understands Hebrew commands (e.g., "תן עץ", "קפוץ 3 פעמים")
- 🧠 Powered by a custom AI model (PyTorch + NLTK)
- 🛠️ Learns new commands dynamically from chat
- ⚔️ Auto PvP: Attacks nearby zombies
- 🧥 Automatically equips armor after collection
- 🎮 Compatible with Minecraft 1.20.1 servers

---

## 📦 Requirements

### Python (for the AI model)
- Python 3.9+
- `torch`
- `nltk`

Install dependencies:
```bash
pip install torch nltk
Node.js (for the bot)
Node.js 18+

Mineflayer and plugins

Install Node.js packages:

bash
Copy
Edit
npm install mineflayer mineflayer-pvp mineflayer-pathfinder mineflayer-armor-manager vec3
🚀 Getting Started
1. Clone the project
bash
Copy
Edit
git clone https://github.com/raziel-star/MineMind-v1.9.git
cd Claude-Minecraft-Bot
2. Run the Python bot
The Python script handles the AI logic and starts the JavaScript bot:

bash
Copy
Edit
python main.py
This will:

Load or train the AI model

Generate bot.js

Launch the Mineflayer bot

Start listening to chat and controlling the bot

🧠 Teaching New Commands In-Game
You can add new commands directly from Minecraft chat!

Syntax:
diff
Copy
Edit
!למד פקודה | משפט בעברית | קודJS
Example:
bash
Copy
Edit
!למד פקודה | תן חרב | bot.chat('/give @p diamond_sword');
✅ The bot will respond: למדתי את הפקודה: תן חרב
✅ The AI is retrained instantly and saves the data.

💬 Command Usage
Example Message	What the Bot Does
!תן עץ	Runs /give @p minecraft:oak_log 64
קפוץ 5 פעמים	Jumps 5 times
תעוף	Sends /fly
חפור קדימה	Digs the block in front
!רקוד	Performs a little dance

Use ! to force a command
Use plain Hebrew text to let the AI decide silently

🧟‍♂️ Auto PvP (Zombie Mode)
Upon spawning, the bot automatically:

Equips armor

Scans for nearby zombies every second

Attacks them with mineflayer-pvp

Useful commands in chat:
arduino
Copy
Edit
health     ➜ shows HP
equip      ➜ wears any collected armor
stop       ➜ stops attacking
📁 File Structure
bash
Copy
Edit
main.py                # AI + Bot launcher
bot.js                 # Generated bot code (do not edit manually)
learned_ai.json        # Saved AI examples and commands
🛠 Tips
To restart learning, delete learned_ai.json

You can send commands directly in the Python terminal

The AI uses bag-of-words, so keep commands consistent in phrasing

🔒 Notes
Bot connects to public Minecraft servers (cracked or local)

Make sure the username is not already used in the server

Customize host/port in bot.js or main.py

💡 Example Additions
Build a Tower Command:
javascript
Copy
Edit
!למד פקודה | בנה מגדל | 
let pos = bot.entity.position.offset(0, 0, 1);
for (let i = 0; i < 5; i++) {
  setTimeout(() => {
    bot.placeBlock(bot.blockAt(pos.offset(0, i - 1, 0)), vec3(0, 1, 0));
  }, i * 500);
}
🤖 Author
Created by Raziel24
Hebrew AI + Minecraft Bot Automation
Contributions welcome!

📜 License
MIT License – use it, modify it, share it.
