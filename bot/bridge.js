/**
 * Мост между реальным Minecraft и Python-мозгом.
 *
 * Схема:
 *   Minecraft server  <--protocol-->  mineflayer (этот файл)  <--TCP JSON-->  Python
 *
 * Node-сторона отвечает только за "глаза и руки": собрать наблюдение,
 * выполнить действие. Всю логику наград и обучения считает Python.
 *
 * Запуск:
 *   node bot/bridge.js --host localhost --port 25565 --username RLAgent
 */
'use strict';

const net = require('net');
const mineflayer = require('mineflayer');

// ---------------------------------------------------------------- аргументы
function arg(name, def) {
  const i = process.argv.indexOf(`--${name}`);
  return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : def;
}
const MC_HOST = arg('host', 'localhost');
const MC_PORT = parseInt(arg('port', '25565'), 10);
const USERNAME = arg('username', 'RLAgent');
const BRAIN_HOST = arg('brain-host', '127.0.0.1');
const BRAIN_PORT = parseInt(arg('brain-port', '5599'), 10);
// 26.1 — последняя версия, которую поддерживает mineflayer (testedVersions),
// и первая деобфусцированная. 26.2/26.3 упадут с Unsupported protocol version.
const VERSION = arg('mc-version', '26.1');

// Проверяем поддержку ДО подключения — иначе ошибка вылезет невнятной
// простынёй из глубины протокола.
(function checkVersion() {
  let supported = false;
  try {
    supported = !!require('minecraft-data')(VERSION);
  } catch (e) {
    supported = false;
  }
  if (!supported) {
    console.error(`\n[bridge] ОШИБКА: версия ${VERSION} не поддерживается mineflayer.`);
    console.error('[bridge] Поддерживается до 26.1 включительно.');
    console.error('[bridge] Варианты:');
    console.error('  1) создайте мир на 26.1 и откройте его для сети;');
    console.error('  2) поставьте ViaVersion на сервер и подключайтесь как 26.1.\n');
    process.exit(1);
  }
  console.log(`[bridge] версия ${VERSION} — поддерживается`);
})();

// Словарь предметов ДОЛЖЕН совпадать с brain/spaces.py — порядок критичен,
// потому что модель работает с числовыми id.
const ITEMS = [
  'empty', 'oak_log', 'oak_planks', 'stick',
  'cobblestone', 'raw_iron', 'iron_ingot', 'diamond',
  'coal', 'crafting_table', 'furnace', 'wooden_sword',
  'wooden_pickaxe', 'wooden_axe', 'wooden_shovel', 'stone_sword',
  'stone_pickaxe', 'stone_axe', 'stone_shovel', 'iron_sword',
  'iron_pickaxe', 'iron_axe', 'iron_shovel', 'diamond_sword',
  'diamond_pickaxe', 'dirt', 'sand', 'rotten_flesh',
  'raw_gold', 'gold_ingot', 'redstone', 'flint',
  'leather', 'string', 'feather', 'gunpowder',
  'charcoal', 'bread', 'wheat', 'apple',
  'cooked_beef', 'beef', 'porkchop', 'cooked_porkchop',
  'golden_sword', 'golden_pickaxe', 'golden_axe', 'golden_shovel',
  'diamond_axe', 'diamond_shovel', 'leather_helmet', 'leather_chestplate',
  'leather_leggings', 'leather_boots', 'iron_helmet', 'iron_chestplate',
  'iron_leggings', 'iron_boots', 'diamond_helmet', 'diamond_chestplate',
  'diamond_leggings', 'diamond_boots', 'bow', 'arrow',
  'shield', 'torch', 'chest', 'ladder',
  'stone_slab', 'oak_slab', 'oak_stairs', 'cobblestone_stairs',
  'oak_door', 'bucket', 'shears', 'flint_and_steel',
  'fishing_rod', 'iron_nugget', 'stick_bundle'
];
const ITEM_ID = Object.fromEntries(ITEMS.map((n, i) => [n, i]));

const BLOCKS = [
  'air', 'stone', 'dirt', 'grass_block',
  'oak_log', 'oak_planks', 'crafting_table', 'furnace',
  'iron_ore', 'diamond_ore', 'bedrock', 'water',
  'gold_ore', 'redstone_ore', 'coal_ore', 'gravel',
  'sand_block', 'oak_leaves', 'torch_block', 'chest_block',
  'lava'
];
const BLOCK_ID = Object.fromEntries(BLOCKS.map((n, i) => [n, i]));

// Сущности — порядок обязан совпадать с brain/spaces.py.
const ENTITIES = [
  'none', 'zombie', 'skeleton', 'spider',
  'creeper', 'cow', 'pig', 'sheep',
  'chicken'
];
const ENTITY_ID = Object.fromEntries(ENTITIES.map((n, i) => [n, i]));

const FACINGS = ['south', 'west', 'north', 'east'];
// Точка спавна запоминается один раз: относительно неё агент понимает,
// насколько далеко ушёл. В симуляторе это угол мира 12x12, в реальной игре —
// место первого появления.
let spawnPoint = null;

// Масштаб, которым нормируются координаты. В бесконечном мире Minecraft
// абсолютные X/Z не нормировать нельзя — они уходят в тысячи. Берём окно
// 64 блока: внутри него позиция читается точно, дальше циклится.
const WORLD_SPAN = 64;

const FACING_YAW = { south: 0, west: Math.PI / 2, north: Math.PI, east: -Math.PI / 2 };

// ---------------------------------------------------------------- крафт-сетка
// ДОЛЖНО совпадать с brain/spaces.py: без верстака у игрока только 2x2.
const GRID_2X2_SLOTS = [0, 1, 3, 4];
const GRID_3X3_SLOTS = [0, 1, 2, 3, 4, 5, 6, 7, 8];

// Уровни инструментов — зеркало TOOL_LEVEL из brain/spaces.py.
const TOOL_LEVEL = {
  empty: 0,
  wooden_pickaxe: 1, wooden_sword: 1, wooden_axe: 1, wooden_shovel: 1,
  stone_pickaxe: 2, stone_sword: 2, stone_axe: 2, stone_shovel: 2,
  iron_pickaxe: 3, iron_sword: 3, iron_axe: 3, iron_shovel: 3,
  diamond_pickaxe: 4, diamond_sword: 4,
};
const BLOCK_REQUIRED_LEVEL = { stone: 1, cobblestone: 1, iron_ore: 2, diamond_ore: 3 };
const NEEDS_PICKAXE = new Set(['stone', 'cobblestone', 'iron_ore', 'diamond_ore', 'furnace']);
const PICKAXES = new Set([
  'wooden_pickaxe', 'stone_pickaxe', 'iron_pickaxe', 'diamond_pickaxe',
]);

function canHarvest(blockName, heldName) {
  const need = BLOCK_REQUIRED_LEVEL[blockName] || 0;
  if (need === 0) return true;
  if (NEEDS_PICKAXE.has(blockName) && !PICKAXES.has(heldName)) return false;
  return (TOOL_LEVEL[heldName] || 0) >= need;
}

// Что бот считает едой (зеркало FOOD_VALUE в brain/spaces.py).
const FOODS = new Set([
  'bread', 'apple', 'wheat', 'beef', 'cooked_beef',
  'porkchop', 'cooked_porkchop', 'rotten_flesh',
]);

// Открыт ли верстак прямо сейчас (зеркало agent.table_open в симуляторе).
let tableOpen = false;

function tableReachable() {
  const t = bot.findBlock({
    matching: (b) => b && b.name === 'crafting_table', maxDistance: 3,
  });
  return !!t;
}

function activeSlots() {
  return (tableOpen && tableReachable()) ? GRID_3X3_SLOTS : GRID_2X2_SLOTS;
}

// ---------------------------------------------------------------- бот
const bot = mineflayer.createBot({
  host: MC_HOST, port: MC_PORT, username: USERNAME, version: VERSION,
});

let ready = false;
let craftGrid = new Array(9).fill(0);  // виртуальная сетка 3x3
let heldItem = 0;

bot.once('spawn', () => {
  spawnPoint = bot.entity.position.clone();
  ready = true;
  console.log(`[bot] зашёл на ${MC_HOST}:${MC_PORT} как ${USERNAME}`);
});
bot.on('error', (e) => console.error('[bot] ошибка:', e.message));
bot.on('kicked', (r) => console.error('[bot] кикнут:', r));
bot.on('end', () => { ready = false; console.log('[bot] отключён'); });

// ---------------------------------------------------------------- наблюдение
function facingIndex() {
  const yaw = ((bot.entity.yaw % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
  const sector = Math.round(yaw / (Math.PI / 2)) % 4;
  return sector; // 0 south, 1 west, 2 north, 3 east
}

function pitchIndex() {
  const p = bot.entity.pitch;
  if (p < -0.4) return 1;   // смотрит вверх
  if (p > 0.4) return -1;   // вниз
  return 0;
}

function rotate(dx, dz, facing) {
  switch (facing) {
    case 0: return [dx, dz];
    case 1: return [-dz, dx];
    case 2: return [-dx, -dz];
    default: return [dz, -dx];
  }
}

function blockIdAt(x, y, z) {
  const b = bot.blockAt(new (require('vec3').Vec3)(x, y, z));
  if (!b) return BLOCK_ID.bedrock;
  const name = b.name;
  if (BLOCK_ID[name] !== undefined) return BLOCK_ID[name];
  if (name.includes('log')) return BLOCK_ID.oak_log;
  if (name.includes('planks')) return BLOCK_ID.oak_planks;
  if (name.includes('iron_ore')) return BLOCK_ID.iron_ore;
  if (name.includes('diamond_ore')) return BLOCK_ID.diamond_ore;
  if (name === 'cave_air' || name === 'void_air') return BLOCK_ID.air;
  return BLOCK_ID.stone;
}

// Радиусы зрения — ДОЛЖНЫ совпадать с brain/spaces.py (VIEW=5, MAP_R=6).
const VIEW_R = 2;    // куб 5x5x5
const MAP_R = 6;     // карта 13x13

function buildObservation() {
  const inv = new Array(ITEMS.length).fill(0);
  for (const it of bot.inventory.items()) {
    const id = ITEM_ID[it.name];
    if (id !== undefined) inv[id] = Math.min(inv[id] + it.count, 64) / 64;
  }

  const f = facingIndex();
  const p = bot.entity.position.floored();

  // --- ближнее 3D-зрение: куб 5x5x5 ---
  const voxels = [];
  for (let dx = -VIEW_R; dx <= VIEW_R; dx++) {
    for (let dy = -VIEW_R; dy <= VIEW_R; dy++) {
      for (let dz = -VIEW_R; dz <= VIEW_R; dz++) {
        const [rx, rz] = rotate(dx, dz, f);
        voxels.push(blockIdAt(p.x + rx, p.y + dy, p.z + rz));
      }
    }
  }

  // --- дальнее 2D-зрение: карта сверху 13x13 ---
  const blockmap = [];
  for (let dx = -MAP_R; dx <= MAP_R; dx++) {
    for (let dz = -MAP_R; dz <= MAP_R; dz++) {
      const [rx, rz] = rotate(dx, dz, f);
      let top = 0;
      // ищем верхний непустой блок в колонке
      for (let dy = 3; dy >= -3; dy--) {
        const b = blockIdAt(p.x + rx, p.y + dy, p.z + rz);
        if (b !== 0) { top = b; break; }
      }
      blockmap.push(top);
    }
  }

  // --- карта сущностей: где мобы ---
  const W = MAP_R * 2 + 1;
  const entmap = new Array(W * W).fill(0);
  for (const id in bot.entities) {
    const e = bot.entities[id];
    if (!e || e === bot.entity || !e.position) continue;
    const eid = ENTITY_ID[e.name];
    if (eid === undefined) continue;
    const dx = Math.round(e.position.x - p.x);
    const dz = Math.round(e.position.z - p.z);
    // обратный поворот: мир -> система взгляда
    let rx, rz;
    if (f === 0) { rx = dx; rz = dz; }
    else if (f === 1) { rx = dz; rz = -dx; }
    else if (f === 2) { rx = -dx; rz = -dz; }
    else { rx = -dz; rz = dx; }
    const i = rx + MAP_R, k = rz + MAP_R;
    if (i >= 0 && i < W && k >= 0 && k < W) entmap[i * W + k] = eid;
  }

  const table = bot.findBlock({
    matching: (b) => b && b.name === 'crafting_table', maxDistance: 8,
  });
  const dist = table ? p.distanceTo(table.position) : 99;

  const reach = tableReachable();
  const slots = activeSlots();

  return {
    grid: craftGrid,
    voxels,
    blockmap,
    entmap,
    inventory: inv,
    held: heldItem,
    facing: f,
    pitch: pitchIndex(),
    near_table: dist <= 3 ? 1 : 0,
    table_dist: Math.min(dist, 99),
    position: [p.x, p.y, p.z],
    // --- КООРДИНАТЫ. Python строит из них те же 14 чисел, что и симулятор.
    // Без них агент не знает, где находится: только что видит перед собой.
    x: p.x,
    y: p.y,
    z: p.z,
    spawn_x: spawnPoint ? spawnPoint.x : p.x,
    spawn_z: spawnPoint ? spawnPoint.z : p.z,
    table_x: table ? table.position.x : p.x,
    table_z: table ? table.position.z : p.z,
    world_span: WORLD_SPAN,
    step_frac: 0,
    health: bot.health ?? 20,
    food: bot.food ?? 20,
    // --- состояние крафт-сетки: Python строит по этому craft_state ---
    table_open: tableOpen ? 1 : 0,
    table_reachable: reach ? 1 : 0,
    has_3x3: slots.length === 9 ? 1 : 0,
    active_slots: slots,
  };
}

// ---------------------------------------------------------------- действия
async function lookAtFacing(f, pitchLevel) {
  const yaw = FACING_YAW[FACINGS[f]];
  const pitch = pitchLevel === 1 ? -0.9 : pitchLevel === -1 ? 0.9 : 0;
  await bot.look(yaw, pitch, true);
}

function frontPos() {
  const f = facingIndex();
  const deltas = [[0, 1], [-1, 0], [0, -1], [1, 0]];
  const [dx, dz] = deltas[f];
  const p = bot.entity.position.floored();
  const Vec3 = require('vec3').Vec3;
  return new Vec3(p.x + dx, p.y + pitchIndex(), p.z + dz);
}

async function applyAction(action) {
  const { type, arg: a } = action;
  const result = { ok: true, note: '' };
  try {
    switch (type) {
      case 'select_item': {
        const name = ITEMS[a];
        const item = bot.inventory.items().find((i) => i.name === name);
        if (!item) { result.ok = false; result.note = 'нет такого предмета'; break; }
        await bot.equip(item, 'hand');
        heldItem = a;
        break;
      }
      case 'place_in_slot': {
        if (heldItem === 0) { result.ok = false; result.note = 'рука пуста'; break; }
        if (craftGrid[a] !== 0) { result.ok = false; result.note = 'слот занят'; break; }
        // Без открытого верстака доступны только слоты 2x2 (0,1,3,4).
        if (!activeSlots().includes(a)) {
          result.ok = false; result.note = 'слот недоступен — нужна сетка 3x3'; break;
        }
        craftGrid[a] = heldItem;
        result.note = `slot ${a} <- ${ITEMS[heldItem]}`;
        break;
      }
      case 'take_from_slot': {
        if (craftGrid[a] === 0) { result.ok = false; result.note = 'слот пуст'; break; }
        craftGrid[a] = 0;
        break;
      }
      case 'clear_grid':
        craftGrid = new Array(9).fill(0);
        break;
      case 'craft': {
        // Сетку переводим в реальный рецепт mineflayer.
        const table = bot.findBlock({
          matching: (b) => b && b.name === 'crafting_table', maxDistance: 4,
        });
        const target = action.result_name;
        const mcData = require('minecraft-data')(bot.version);
        const itemDef = mcData.itemsByName[target];
        if (!itemDef) { result.ok = false; result.note = 'неизвестный рецепт'; break; }
        const recipes = bot.recipesFor(itemDef.id, null, 1, table);
        if (!recipes || recipes.length === 0) {
          result.ok = false; result.note = 'рецепт недоступен'; break;
        }
        await bot.craft(recipes[0], 1, table || undefined);
        craftGrid = new Array(9).fill(0);
        result.note = `скрафтил ${target}`;
        break;
      }
      case 'turn': {
        const f = facingIndex();
        let pitchLvl = pitchIndex();
        let nf = f;
        if (a === 0) nf = (f + 1) % 4;
        else if (a === 1) nf = (f + 3) % 4;
        else if (a === 2) pitchLvl = Math.min(1, pitchLvl + 1);
        else pitchLvl = Math.max(-1, pitchLvl - 1);
        await lookAtFacing(nf, pitchLvl);
        break;
      }
      case 'move': {
        const dirs = ['forward', 'back', 'left', 'right'];
        const d = dirs[a];
        bot.setControlState(d, true);
        await new Promise((r) => setTimeout(r, 250));
        bot.setControlState(d, false);
        break;
      }
      case 'place_block': {
        const name = ITEMS[heldItem];
        const item = bot.inventory.items().find((i) => i.name === name);
        if (!item) { result.ok = false; result.note = 'нечего ставить'; break; }
        await bot.equip(item, 'hand');
        const ref = bot.blockAt(frontPos().offset(0, -1, 0));
        if (!ref) { result.ok = false; result.note = 'нет опоры'; break; }
        const Vec3 = require('vec3').Vec3;
        await bot.placeBlock(ref, new Vec3(0, 1, 0));
        break;
      }
      case 'break_block': {
        const b = bot.blockAt(frontPos());
        if (!b || b.name === 'air' || b.name === 'bedrock') {
          result.ok = false; result.note = 'ломать нечего'; break;
        }
        // Правила ванили: неподходящий инструмент ломает блок, но дроп не
        // выпадает. Сервер сам это обеспечит — мы лишь сообщаем Python,
        // чтобы он начислил штраф world.wrong_tool, как в симуляторе.
        const heldName = ITEMS[heldItem] || 'empty';
        result.wrong_tool = canHarvest(b.name, heldName) ? 0 : 1;
        result.block = b.name;
        await bot.dig(b);
        result.note = `сломал ${b.name}`
          + (result.wrong_tool ? ' (не тот инструмент — дропа нет)' : '');
        break;
      }
      case 'use_table': {
        // Открыть/закрыть верстак. Именно это переключает сетку 2x2 <-> 3x3.
        if (tableOpen) {
          tableOpen = false;
          // Уходя, возвращаем содержимое сетки в инвентарь (как в игре).
          craftGrid = new Array(9).fill(0);
          result.note = 'верстак закрыт';
          break;
        }
        if (!tableReachable()) {
          result.ok = false; result.note = 'верстака рядом нет'; break;
        }
        tableOpen = true;
        result.note = 'верстак открыт — доступна сетка 3x3';
        break;
      }
      case 'attack': {
        // Бьём ближайшую сущность в пределах досягаемости.
        const target = bot.nearestEntity((e) =>
          e && e !== bot.entity && e.position &&
          e.position.distanceTo(bot.entity.position) < 4);
        if (!target) { result.ok = false; result.note = 'бить некого'; break; }
        await bot.lookAt(target.position.offset(0, 1.6, 0), true);
        bot.attack(target);
        result.note = `ударил ${target.name || 'моба'}`;
        result.target = target.name || '';
        break;
      }
      case 'eat': {
        const food = bot.inventory.items().find((i) => FOODS.has(i.name));
        if (!food) { result.ok = false; result.note = 'еды нет'; break; }
        if ((bot.food ?? 20) >= 20) {
          result.ok = false; result.note = 'сыт'; break;
        }
        await bot.equip(food, 'hand');
        await bot.consume();
        result.note = `съел ${food.name}`;
        break;
      }
      case 'noop':
      default:
        break;
    }
  } catch (e) {
    result.ok = false;
    result.note = e.message;
  }
  // Отошёл от верстака — он закрывается сам, сетка падает до 2x2.
  if (tableOpen && !tableReachable()) {
    tableOpen = false;
    craftGrid = new Array(9).fill(0);
    result.note += ' | отошёл от верстака, сетка 2x2';
  }
  return result;
}

// ---------------------------------------------------------------- TCP сервер
const server = net.createServer((sock) => {
  console.log('[bridge] Python-мозг подключился');
  let buf = '';

  sock.on('data', async (chunk) => {
    buf += chunk.toString();
    let nl;
    while ((nl = buf.indexOf('\n')) !== -1) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      let msg;
      try { msg = JSON.parse(line); } catch { continue; }

      let reply;
      if (!ready) {
        reply = { error: 'бот ещё не заспавнился' };
      } else if (msg.cmd === 'observe') {
        reply = { obs: buildObservation() };
      } else if (msg.cmd === 'act') {
        const res = await applyAction(msg.action || {});
        reply = { result: res, obs: buildObservation() };
      } else if (msg.cmd === 'reset') {
        craftGrid = new Array(9).fill(0);
        heldItem = 0;
        tableOpen = false;
        reply = { obs: buildObservation() };
      } else if (msg.cmd === 'chat') {
        bot.chat(String(msg.text || ''));
        reply = { ok: true };
      } else {
        reply = { error: `неизвестная команда ${msg.cmd}` };
      }
      sock.write(JSON.stringify(reply) + '\n');
    }
  });

  sock.on('error', (e) => console.error('[bridge] сокет:', e.message));
});

server.listen(BRAIN_PORT, BRAIN_HOST, () => {
  console.log(`[bridge] жду Python-мозг на ${BRAIN_HOST}:${BRAIN_PORT}`);
});
