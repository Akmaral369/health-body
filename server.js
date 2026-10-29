const http = require('http');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const db = require('./db.js');
const rag = require('./rag.js');

const PORT = process.env.PORT || 3000;
const PUBLIC_DIR = __dirname;
const ADMIN_PASSWORD = process.env.ADMIN_PASSWORD || 'admin2026';

// Хранилище сессий администратора в памяти
const activeSessions = new Set();

// In-memory Rate Limiter
const rateLimits = new Map();

function checkRateLimit(ip, maxRequests = 30, windowMs = 60000) {
  const now = Date.now();
  let record = rateLimits.get(ip);
  if (!record || now > record.resetTime) {
    record = { count: 1, resetTime: now + windowMs };
    rateLimits.set(ip, record);
    return true;
  }
  if (record.count >= maxRequests) {
    return false;
  }
  record.count++;
  return true;
}

// Очистка устаревших записей rate limit каждые 5 минут
setInterval(() => {
  const now = Date.now();
  for (const [ip, rec] of rateLimits.entries()) {
    if (now > rec.resetTime) rateLimits.delete(ip);
  }
}, 300000);

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.jfif': 'image/jpeg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.webp': 'image/webp',
  '.ico': 'image/x-icon',
  '.txt': 'text/plain; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8'
};

/**
 * Чтение тела запроса с контролем максимального размера (возврат 413 при превышении)
 */
function readBody(req, maxBytes) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    let isTooLarge = false;

    req.on('data', chunk => {
      if (isTooLarge) return;
      size += chunk.length;
      if (size > maxBytes) {
        isTooLarge = true;
        const err = new Error('Payload Too Large');
        err.statusCode = 413;
        reject(err);
      } else {
        chunks.push(chunk);
      }
    });

    req.on('end', () => {
      if (!isTooLarge) {
        resolve(Buffer.concat(chunks));
      }
    });

    req.on('error', err => {
      if (!isTooLarge) reject(err);
    });
  });
}

/**
 * Парсинг cookies из заголовка
 */
function parseCookies(req) {
  const list = {};
  const rc = req.headers.cookie;
  if (rc) {
    rc.split(';').forEach(cookie => {
      const parts = cookie.split('=');
      list[parts.shift().trim()] = decodeURI(parts.join('='));
    });
  }
  return list;
}

/**
 * Проверка авторизации администратора (cookie auth_token, заголовок Authorization или x-admin-password)
 */
function isAuthenticated(req) {
  const cookies = parseCookies(req);
  const token = cookies.auth_token;
  if (token && activeSessions.has(token)) return true;

  const authHeader = req.headers['authorization'];
  if (authHeader && authHeader.startsWith('Bearer ')) {
    const bearer = authHeader.slice(7).trim();
    if (activeSessions.has(bearer)) return true;
  }

  const passHeader = req.headers['x-admin-password'];
  if (passHeader && passHeader === ADMIN_PASSWORD) return true;

  return false;
}

/**
 * Безопасная очистка имени файла с поддержкой кириллицы и буквы 'ё'
 */
function sanitizeFilename(filename) {
  if (!filename) return 'document.txt';
  const basename = path.basename(filename);
  const cleaned = basename.replace(/[^\p{L}\p{N}\s._-]/gu, '').trim();
  return cleaned || 'document.txt';
}

/**
 * Парсинг multipart/form-data без тяжелых зависимостей
 */
function parseMultipart(buffer, boundary) {
  const boundaryBuffer = Buffer.from('--' + boundary);
  const parts = [];
  let start = buffer.indexOf(boundaryBuffer);

  while (start !== -1) {
    const end = buffer.indexOf(boundaryBuffer, start + boundaryBuffer.length);
    if (end === -1) break;

    const partBuffer = buffer.slice(start + boundaryBuffer.length, end);
    const headerEnd = partBuffer.indexOf('\r\n\r\n');
    if (headerEnd !== -1) {
      const headerText = partBuffer.slice(0, headerEnd).toString('utf-8');
      const body = partBuffer.slice(headerEnd + 4, partBuffer.length - 2); // убираем trailing \r\n
      parts.push({ headers: headerText, body });
    }
    start = end;
  }

  for (const part of parts) {
    const filenameMatch = part.headers.match(/filename="([^"]+)"/i);
    if (filenameMatch) {
      return {
        filename: filenameMatch[1],
        content: part.body.toString('utf-8')
      };
    }
  }
  return null;
}

function sendJson(res, statusCode, data) {
  const json = JSON.stringify(data);
  const headers = {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(json)
  };
  if (statusCode === 413) {
    headers['Connection'] = 'close';
  }
  res.writeHead(statusCode, headers);
  res.end(json);
}

const server = http.createServer(async (req, res) => {
  const clientIp = req.headers['x-forwarded-for'] || req.socket.remoteAddress || 'unknown';
  let decodedUrl = '/';
  try {
    decodedUrl = decodeURIComponent(req.url.split('?')[0]);
  } catch (err) {
    res.writeHead(400, { 'Content-Type': 'text/plain; charset=utf-8' });
    return res.end('Bad Request');
  }

  // --- API: ЧАТ-БОТ ---
  if (req.method === 'POST' && decodedUrl === '/api/chat') {
    if (!checkRateLimit(clientIp, 30, 60000)) {
      return sendJson(res, 429, { error: 'Слишком много запросов. Пожалуйста, подождите минуту.' });
    }

    try {
      // Лимит тела для вопроса в чате: 64 КБ
      const bodyBuffer = await readBody(req, 64 * 1024);
      const data = JSON.parse(bodyBuffer.toString('utf-8'));
      const message = (data.message || '').trim();

      if (!message) {
        return sendJson(res, 400, { error: 'Сообщение не может быть пустым.' });
      }

      const answerData = await rag.askQuestion(message);
      return sendJson(res, 200, answerData);
    } catch (err) {
      if (err.statusCode === 413) {
        return sendJson(res, 413, { error: 'Сообщение превышает лимит размера (64 КБ).' });
      }
      return sendJson(res, 500, { error: 'Ошибка обработки вопроса: ' + err.message });
    }
  }

  // --- API: ВХОД В АДМИНКУ ---
  if (req.method === 'POST' && decodedUrl === '/api/admin/login') {
    if (!checkRateLimit(clientIp, 10, 60000)) {
      return sendJson(res, 429, { error: 'Слишком много попыток входа. Попробуйте через минуту.' });
    }

    try {
      const bodyBuffer = await readBody(req, 16 * 1024);
      const data = JSON.parse(bodyBuffer.toString('utf-8'));
      const password = data.password || '';

      if (password === ADMIN_PASSWORD) {
        const token = crypto.randomBytes(32).toString('hex');
        activeSessions.add(token);
        res.setHeader('Set-Cookie', `auth_token=${token}; HttpOnly; Path=/; SameSite=Lax; Max-Age=86400`);
        return sendJson(res, 200, { ok: true, token });
      } else {
        return sendJson(res, 401, { error: 'Неверный пароль администратора.' });
      }
    } catch (err) {
      return sendJson(res, 400, { error: 'Некорректный запрос' });
    }
  }

  // --- API: ВЫХОД ИЗ АДМИНКИ ---
  if (req.method === 'POST' && decodedUrl === '/api/admin/logout') {
    const cookies = parseCookies(req);
    if (cookies.auth_token) activeSessions.delete(cookies.auth_token);
    res.setHeader('Set-Cookie', 'auth_token=; HttpOnly; Path=/; Expires=Thu, 01 Jan 1970 00:00:00 GMT');
    return sendJson(res, 200, { ok: true });
  }

  // --- API: ПРОВЕРКА СТАТУСА АДМИНА ---
  if (req.method === 'GET' && decodedUrl === '/api/admin/status') {
    return sendJson(res, 200, { authenticated: isAuthenticated(req) });
  }

  // --- API: СПИСОК ДОКУМЕНТОВ ---
  if (req.method === 'GET' && decodedUrl === '/api/admin/documents') {
    if (!isAuthenticated(req)) {
      return sendJson(res, 401, { error: 'Требуется авторизация администратора.' });
    }
    const docs = db.listDocuments();
    return sendJson(res, 200, { documents: docs });
  }

  // --- API: ЗАГРУЗКА ДОКУМЕНТА (Лимит 1 МБ) ---
  if (req.method === 'POST' && decodedUrl === '/api/admin/documents') {
    if (!isAuthenticated(req)) {
      return sendJson(res, 401, { error: 'Требуется авторизация администратора.' });
    }

    try {
      // Строгий лимит 1 МБ на тело файла (1 048 576 байт)
      const MAX_UPLOAD_SIZE = 1024 * 1024;
      const bodyBuffer = await readBody(req, MAX_UPLOAD_SIZE);
      const contentType = req.headers['content-type'] || '';

      let filename = '';
      let textContent = '';

      if (contentType.includes('application/json')) {
        const payload = JSON.parse(bodyBuffer.toString('utf-8'));
        filename = payload.title || payload.filename || 'документ.txt';
        textContent = payload.content || '';
      } else if (contentType.includes('multipart/form-data')) {
        const match = contentType.match(/boundary=(?:"([^"]+)"|([^;]+))/i);
        if (!match) {
          return sendJson(res, 400, { error: 'Не найден boundary в multipart запросе.' });
        }
        const boundary = match[1] || match[2];
        const parsed = parseMultipart(bodyBuffer, boundary);
        if (!parsed) {
          return sendJson(res, 400, { error: 'Не удалось разобрать загружаемый файл.' });
        }
        filename = parsed.filename;
        textContent = parsed.content;
      } else {
        // Обычный текстовый поток
        filename = req.headers['x-filename'] || 'документ.txt';
        textContent = bodyBuffer.toString('utf-8');
      }

      const cleanTitle = sanitizeFilename(filename);
      const ext = path.extname(cleanTitle).toLowerCase();

      // Проверка разрешённого расширения (.txt, .md)
      if (ext !== '.txt' && ext !== '.md') {
        return sendJson(res, 400, {
          error: `Недопустимый формат файла "${ext}". Разрешены только файлы .txt и .md.`
        });
      }

      if (!textContent || textContent.trim().length === 0) {
        return sendJson(res, 400, { error: 'Файл не содержит текста.' });
      }

      const result = await rag.indexDocument(cleanTitle, textContent);
      return sendJson(res, 201, {
        ok: true,
        document: result
      });
    } catch (err) {
      if (err.statusCode === 413) {
        return sendJson(res, 413, { error: 'Файл превышает допустимый размер (лимит: 1 МБ).' });
      }
      return sendJson(res, 500, { error: 'Ошибка индексации документа: ' + err.message });
    }
  }

  // --- API: УДАЛЕНИЕ ДОКУМЕНТА ---
  if (req.method === 'DELETE' && decodedUrl.startsWith('/api/admin/documents/')) {
    if (!isAuthenticated(req)) {
      return sendJson(res, 401, { error: 'Требуется авторизация администратора.' });
    }
    const id = parseInt(decodedUrl.replace('/api/admin/documents/', ''), 10);
    if (isNaN(id)) {
      return sendJson(res, 400, { error: 'Некорректный ID документа.' });
    }
    const success = db.deleteDocument(id);
    if (!success) {
      return sendJson(res, 404, { error: 'Документ не найден.' });
    }
    return sendJson(res, 200, { ok: true, deletedId: id });
  }

  // --- API: ПЕРЕИНДЕКСАЦИЯ ДОКУМЕНТА ---
  if (req.method === 'POST' && decodedUrl.startsWith('/api/admin/reindex/')) {
    if (!isAuthenticated(req)) {
      return sendJson(res, 401, { error: 'Требуется авторизация администратора.' });
    }
    const id = parseInt(decodedUrl.replace('/api/admin/reindex/', ''), 10);
    if (isNaN(id)) {
      return sendJson(res, 400, { error: 'Некорректный ID документа.' });
    }
    try {
      const result = await rag.reindexDocument(id);
      return sendJson(res, 200, { ok: true, document: result });
    } catch (err) {
      return sendJson(res, 500, { error: 'Ошибка переиндексации: ' + err.message });
    }
  }

  // --- СТРАНИЦА АДМИНКИ: /admin -> admin.html ---
  if (decodedUrl === '/admin' || decodedUrl === '/admin/') {
    decodedUrl = '/admin.html';
  }

  // --- РАЗДАЧА СТАТИЧЕСКИХ ФАЙЛОВ ---
  let filePath = path.join(PUBLIC_DIR, decodedUrl === '/' ? 'index.html' : decodedUrl);

  if (!filePath.startsWith(PUBLIC_DIR)) {
    res.writeHead(403, { 'Content-Type': 'text/plain; charset=utf-8' });
    return res.end('Forbidden');
  }

  fs.stat(filePath, (err, stats) => {
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      return res.end('404 Not Found');
    }

    if (stats.isDirectory()) {
      filePath = path.join(filePath, 'index.html');
    }

    const ext = path.extname(filePath).toLowerCase();
    const contentType = MIME_TYPES[ext] || 'application/octet-stream';

    fs.readFile(filePath, (readErr, content) => {
      if (readErr) {
        res.writeHead(500, { 'Content-Type': 'text/plain; charset=utf-8' });
        return res.end('500 Internal Server Error');
      }
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(content);
    });
  });
});

if (require.main === module) {
  server.listen(PORT, '0.0.0.0', () => {
    console.log(`Server is running at http://localhost:${PORT}/`);
    console.log(`Admin panel: http://localhost:${PORT}/admin`);
  });
}

module.exports = server;
